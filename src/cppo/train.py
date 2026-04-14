from __future__ import annotations

"""Main training entrypoint for CPPO/GRPO runs.

This file wires together:
- config loading/validation
- dataset shaping
- reward function and observability metrics
- trainer selection (GRPO vs CPPO)
- checkpoint/boundary evaluation hooks
"""

import argparse
import json
import logging
import os
import platform
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from .callbacks import CheckpointArtifactsCallback
from .config_loader import dump_resolved_config, load_config
from .eval import (
    build_eval_profiles,
    cleanup_merged_eval_model,
    combine_eval_profile_summaries,
    evaluate_checkpoint,
    save_eval_outputs,
)
from .evaluator_registry import EvaluatorRegistry
from .trainer_cppo import CPPOTrainer
from .reward import check_answer, unwrap_completion, check_format_compliance

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cppo.train")

REWARD_STATS_BUFFER: deque[dict[str, float]] = deque(maxlen=32)
RUN_GROUP_SIZE = 8
RUN_REF_LINES: dict[str, float] = {
    "kl_ref_floor": 0.001,
    "kl_ref_ceiling": 0.10,
    "clip_ratio/ref_min": 0.005,
    "grpo/reward_std_ref_floor": 0.05,
    "grpo/all_zero_ref_max": 0.50,
    "grpo/all_perfect_ref_max": 0.50,
}
SEEN_PROBLEM_IDS: set[str] = set()
SEEN_BY_SOURCE: dict[str, set[str]] = {}
SEEN_BY_DIFFICULTY: dict[str, set[str]] = {}
PROCESSED_TOTAL = 0
PROCESSED_BY_SOURCE: dict[str, int] = {}
PROCESSED_BY_DIFFICULTY: dict[str, int] = {}
RUN_REWARD_ACCURACY_WEIGHT = 1.0
RUN_REWARD_FORMAT_WEIGHT = 1.0
RUN_REWARD_EVALUATOR = EvaluatorRegistry({"default_backend": "fallback_sympy"})


def _label_key(raw: Any) -> str:
    """Normalize metric labels to safe lowercase keys."""
    s = str(raw or "unknown").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    key = "".join(out).strip("_")
    return key or "unknown"


def _get_list(values: Any, n: int, default: str) -> list[str]:
    """Return a length-`n` string list from scalar/list-like input."""
    if values is None:
        return [default] * n
    if isinstance(values, (list, tuple)):
        out = [str(v) for v in values[:n]]
        if len(out) < n:
            out.extend([default] * (n - len(out)))
        return out
    return [str(values)] * n


def _build_ref_lines(cfg: dict[str, Any]) -> dict[str, float]:
    """Build static W&B reference lines from config overrides."""
    refs = cfg.get("observability", {}).get("refs", {})
    out = dict(RUN_REF_LINES)
    for key in out:
        if key in refs:
            out[key] = float(refs[key])
    return out


def set_seed(seed: int) -> None:
    """Set Python/NumPy/Torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dict rows."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _validate_train_rows(rows: list[dict[str, Any]]) -> None:
    """Sanity-check a sample of training rows for required fields."""
    required = {"question", "answer", "source", "difficulty"}
    for i, row in enumerate(rows[:20]):
        missing = [k for k in required if k not in row or row[k] in (None, "")]
        if missing:
            raise ValueError(f"Train schema invalid at row {i}: missing {missing}")


def _build_train_dataset(rows: list[dict[str, Any]], system_prompt: str) -> Dataset:
    """Convert normalized JSON rows into TRL-compatible chat prompts."""
    dataset_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        dataset_rows.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "answer": answer,
                "id": row.get("id", row.get("problem_id", f"row_{i}")),
                "source": row.get("source", "unknown"),
                "difficulty": row.get("difficulty", "unknown"),
            }
        )
    return Dataset.from_list(dataset_rows)


def _reward_fn(completions: list[Any], answer: list[str], **kwargs: Any) -> list[float]:
    """Compute blended reward and update rolling observability metrics.

    Reward = weighted accuracy + weighted format score, clamped to [0, 1].
    """
    global PROCESSED_TOTAL
    if isinstance(answer, str):
        answers = [answer] * len(completions)
    else:
        answers = list(answer)

    n = len(completions)
    ids = _get_list(kwargs.get("id"), n, "unknown")
    sources = [_label_key(x) for x in _get_list(kwargs.get("source"), n, "unknown")]
    diffs = [_label_key(x) for x in _get_list(kwargs.get("difficulty"), n, "unknown")]
    accuracy_scores: list[float] = []
    evaluator_backends: list[str] = []
    for i in range(n):
        text = unwrap_completion(completions[i])
        eval_res = RUN_REWARD_EVALUATOR.score(
            split_name=sources[i],
            predicted_text=text,
            ground_truth=str(answers[i]) if i < len(answers) else "",
            row={
                "id": ids[i],
                "source": sources[i],
                "difficulty": diffs[i],
            },
        )
        accuracy_scores.append(float(eval_res.score))
        evaluator_backends.append(_label_key(eval_res.backend))

    format_scores = [check_format_compliance(unwrap_completion(c)) for c in completions]
    scores: list[float] = []
    for a, f in zip(accuracy_scores, format_scores):
        raw_score = (RUN_REWARD_ACCURACY_WEIGHT * a) + (RUN_REWARD_FORMAT_WEIGHT * f)
        # Keep reward bounded to [0, 1] even if custom evaluator/weights drift.
        scores.append(float(max(0.0, min(1.0, raw_score))))
    if scores:
        reward_arr = np.array(scores, dtype=np.float32)
        acc_arr = np.array(accuracy_scores, dtype=np.float32)
        fmt_arr = np.array(format_scores, dtype=np.float32)
        n = len(scores)
        non_zero_fraction = float((reward_arr > 0.0).mean())
        zero_fraction = float(sum(1 for s in scores if s == 0.0)) / float(len(scores))
        groups = [scores[i : i + RUN_GROUP_SIZE] for i in range(0, len(scores), RUN_GROUP_SIZE)]
        all_zero_groups = sum(1 for g in groups if g and all(s == 0.0 for s in g))
        all_perfect_groups = sum(1 for g in groups if g and all(s >= 0.99 for s in g))
        all_zero_fraction = float(all_zero_groups) / float(len(groups)) if groups else 0.0
        all_perfect_fraction = float(all_perfect_groups) / float(len(groups)) if groups else 0.0
        group_stds = [float(np.std(g)) for g in groups if g]
        std = float(np.mean(group_stds)) if group_stds else 0.0

        completion_texts = [unwrap_completion(c) for c in completions]
        empty_completion_fraction = (
            float(sum(1 for t in completion_texts if not str(t).strip())) / float(n) if n else 0.0
        )
        has_reasoning_fraction = (
            float(sum(1 for t in completion_texts if "<think>" in str(t) and "</think>" in str(t))) / float(n)
            if n
            else 0.0
        )
        mean_completion_chars = float(np.mean([len(str(t)) for t in completion_texts])) if completion_texts else 0.0

        metrics: dict[str, float] = {
            "reward/mean": float(reward_arr.mean()),
            "reward/std": float(reward_arr.std()),
            "reward/non_zero_fraction": non_zero_fraction,
            "reward/accuracy_mean": float(acc_arr.mean()),
            "reward/format_mean": float(fmt_arr.mean()),
            "reward/weight_accuracy": float(RUN_REWARD_ACCURACY_WEIGHT),
            "reward/weight_format": float(RUN_REWARD_FORMAT_WEIGHT),
            "exec/zero_fraction": zero_fraction,
            "grpo/all_zero_fraction": all_zero_fraction,
            "grpo/all_perfect_fraction": all_perfect_fraction,
            "grpo/reward_std_mean": std,
            "gen/empty_completion_fraction": empty_completion_fraction,
            "gen/has_reasoning_fraction": has_reasoning_fraction,
            "gen/mean_completion_chars": mean_completion_chars,
            "grpo/reward_std_ref_floor": float(RUN_REF_LINES["grpo/reward_std_ref_floor"]),
            "grpo/all_zero_ref_max": float(RUN_REF_LINES["grpo/all_zero_ref_max"]),
            "grpo/all_perfect_ref_max": float(RUN_REF_LINES["grpo/all_perfect_ref_max"]),
        }
        metrics["grpo/warn_reward_std_collapse"] = 1.0 if std < RUN_REF_LINES["grpo/reward_std_ref_floor"] else 0.0
        metrics["grpo/warn_all_zero_collapse"] = 1.0 if all_zero_fraction > RUN_REF_LINES["grpo/all_zero_ref_max"] else 0.0
        metrics["warn/problems_too_easy"] = (
            1.0 if all_perfect_fraction > RUN_REF_LINES["grpo/all_perfect_ref_max"] else 0.0
        )

        for diff in sorted(set(diffs)):
            idx = [i for i, d in enumerate(diffs) if d == diff]
            if idx:
                metrics[f"reward/mean_{diff}"] = float(np.mean([scores[i] for i in idx]))

        for src in sorted(set(sources)):
            idx = [i for i, s in enumerate(sources) if s == src]
            if idx:
                metrics[f"reward/{src}_mean"] = float(np.mean([scores[i] for i in idx]))
        for backend in sorted(set(evaluator_backends)):
            cnt = float(sum(1 for b in evaluator_backends if b == backend))
            metrics[f"reward/evaluator_{backend}_fraction"] = cnt / float(n) if n else 0.0

        PROCESSED_TOTAL += n
        for pid, src, diff in zip(ids, sources, diffs, strict=False):
            SEEN_PROBLEM_IDS.add(str(pid))
            SEEN_BY_SOURCE.setdefault(src, set()).add(str(pid))
            SEEN_BY_DIFFICULTY.setdefault(diff, set()).add(str(pid))
            PROCESSED_BY_SOURCE[src] = PROCESSED_BY_SOURCE.get(src, 0) + 1
            PROCESSED_BY_DIFFICULTY[diff] = PROCESSED_BY_DIFFICULTY.get(diff, 0) + 1

        metrics["data/unique_problems_seen"] = float(len(SEEN_PROBLEM_IDS))
        metrics["data/processed_total"] = float(PROCESSED_TOTAL)
        for src in sorted(SEEN_BY_SOURCE):
            metrics[f"data/{src}_seen"] = float(len(SEEN_BY_SOURCE[src]))
            metrics[f"data/{src}_processed"] = float(PROCESSED_BY_SOURCE.get(src, 0))
        for diff in sorted(SEEN_BY_DIFFICULTY):
            metrics[f"data/{diff}_seen"] = float(len(SEEN_BY_DIFFICULTY[diff]))
            metrics[f"data/{diff}_processed"] = float(PROCESSED_BY_DIFFICULTY.get(diff, 0))

        REWARD_STATS_BUFFER.append(metrics)
    return scores


class RewardStatsCallback(TrainerCallback):
    """Aggregate reward diagnostics and forward them to trainer/W&B logs."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return control
        if logs is None or not REWARD_STATS_BUFFER:
            return control

        keys = sorted({k for row in REWARD_STATS_BUFFER for k in row.keys()})
        agg: dict[str, float] = {}
        latest = REWARD_STATS_BUFFER[-1]
        for key in keys:
            if key.startswith("data/"):
                agg[key] = float(latest.get(key, 0.0))
                continue
            vals = [float(row[key]) for row in REWARD_STATS_BUFFER if key in row]
            if vals:
                agg[key] = float(np.mean(vals))
        # Inject static reference lines with train key roots.
        # These become train/kl_ref_floor etc. in W&B through Trainer log rewriting.
        agg["kl_ref_floor"] = float(RUN_REF_LINES["kl_ref_floor"])
        agg["kl_ref_ceiling"] = float(RUN_REF_LINES["kl_ref_ceiling"])
        agg["clip_ratio/ref_min"] = float(RUN_REF_LINES["clip_ratio/ref_min"])
        logs.update(agg)
        if wandb is not None and wandb.run is not None:
            try:
                wandb.log(agg, step=int(state.global_step))
            except Exception:
                logger.warning("Failed to log reward stats payload to W&B", exc_info=True)
        REWARD_STATS_BUFFER.clear()
        return control


class MidEvalCallback(TrainerCallback):
    """Optional quick local mid-step eval callback (debug-oriented)."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        eval_rows: list[dict[str, Any]],
        system_prompt: str,
        max_prompt_length: int,
        mid_eval_max_new_tokens: int,
        eval_cfg: dict[str, Any],
        skip_steps: set[int] | None = None,
    ):
        self.tokenizer = tokenizer
        self.eval_rows = eval_rows
        self.system_prompt = system_prompt
        self.max_prompt_length = int(max_prompt_length)
        self.mid_eval_max_new_tokens = int(mid_eval_max_new_tokens)
        self.done_steps: set[int] = set()
        self.eval_cfg = eval_cfg
        self.skip_steps = set(skip_steps or set())

    def _render_prompt(self, question: str) -> str:
        """Render a single eval prompt using chat template if available."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"{self.system_prompt}\n\n{question}"

    def _run_quick_eval(self, model: torch.nn.Module, n_problems: int) -> float:
        """Run small pass@1 snapshot without vLLM (local debug path)."""
        if not self.eval_rows:
            return 0.0

        was_training = model.training
        model.eval()

        subset = self.eval_rows[: min(n_problems, len(self.eval_rows))]
        device = next(model.parameters()).device
        correct = 0

        for row in subset:
            prompt = self._render_prompt(str(row["question"]))
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.mid_eval_max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen_tokens = output[0][inputs["input_ids"].shape[1] :]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            if check_answer(text, str(row["answer"])) == 1.0:
                correct += 1

        if was_training:
            model.train()

        return float(correct) / float(len(subset))

    def on_step_end(self, args, state, control, **kwargs):
        """Trigger quick local eval on configured steps."""
        if not state.is_world_process_zero:
            return control

        if not bool(self.eval_cfg.get("enabled", False)):
            return control

        step = int(state.global_step)
        mid_steps = set(int(x) for x in self.eval_cfg.get("steps", []))
        if step not in mid_steps or step in self.done_steps:
            return control
        if step in self.skip_steps:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        pass1 = self._run_quick_eval(model, int(self.eval_cfg.get("n_problems", 30)))
        logger.info("mid-eval step=%d pass@1=%.4f", step, pass1)

        if wandb is not None and wandb.run is not None:
            wandb.log({"mid_eval/pass@1": pass1}, step=step)

        self.done_steps.add(step)
        return control


def _build_grpo_config(cfg: dict[str, Any], run_dir: Path) -> GRPOConfig:
    """Translate project YAML into GRPOConfig for this TRL version."""
    fields = GRPOConfig.__dataclass_fields__.keys()

    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    rollout_cfg = cfg["rollout"]
    wandb_cfg = cfg.get("integrations", {}).get("wandb", {})
    hub_cfg = cfg.get("integrations", {}).get("hf_hub", {})
    lora_cfg = cfg.get("lora", {})
    artifacts_cfg = cfg.get("artifacts", {})

    report_to = ["wandb"] if bool(wandb_cfg.get("enabled", False)) else []
    bf16 = bool(train_cfg.get("bf16", torch.cuda.is_available()))
    model_dtype = torch.bfloat16 if bf16 else torch.float32

    kwargs: dict[str, Any] = {
        "output_dir": str(run_dir / "checkpoints"),
        "run_name": str(wandb_cfg.get("run_name", "cppo")),
        "max_steps": int(train_cfg["max_steps"]),
        "per_device_train_batch_size": int(train_cfg["batch_size"]),
        "gradient_accumulation_steps": int(train_cfg["gradient_accumulation_steps"]),
        "learning_rate": float(train_cfg["learning_rate"]),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        "warmup_steps": int(train_cfg.get("warmup_steps", 0)),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "cosine")),
        "logging_steps": int(train_cfg.get("logging_steps", 1)),
        "save_steps": int(train_cfg.get("save_steps", 100)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 50)),
        "save_safetensors": True,
        "seed": int(cfg["run"]["seed"]),
        "bf16": bf16,
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "num_generations": int(rollout_cfg["num_generations"]),
        "max_completion_length": int(model_cfg["max_completion_length"]),
        "temperature": float(rollout_cfg["temperature"]),
        "beta": float(rollout_cfg["beta"]),
        "loss_type": "grpo",
        "num_iterations": int(train_cfg.get("num_iterations", 1)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "report_to": report_to,
        "dataloader_drop_last": bool(train_cfg.get("dataloader_drop_last", True)),
        "remove_unused_columns": bool(train_cfg.get("remove_unused_columns", False)),
        "generation_batch_size": int(train_cfg["generation_batch_size"]),
        "model_init_kwargs": {
            "torch_dtype": model_dtype,
            "attn_implementation": str(model_cfg.get("attn_implementation", "sdpa")),
            "trust_remote_code": True,
            "token": os.environ.get(str(hub_cfg.get("token_env", "HF_TOKEN")), "") or None,
        },
        "push_to_hub": bool(hub_cfg.get("push_to_hub", False)),
        "hub_model_id": str(hub_cfg.get("hub_model_id", "")) or None,
        "num_generations_eval": int(cfg.get("eval", {}).get("n_generations", 1)),
    }

    if "max_prompt_length" in fields:
        kwargs["max_prompt_length"] = int(model_cfg["max_prompt_length"])
    if "use_vllm" in fields:
        kwargs["use_vllm"] = bool(rollout_cfg.get("use_vllm", True))
    if "vllm_mode" in fields:
        kwargs["vllm_mode"] = str(rollout_cfg.get("vllm_mode", "server"))
    if "vllm_server_base_url" in fields:
        base_url = str(rollout_cfg.get("vllm_server_base_url", "")).strip()
        kwargs["vllm_server_base_url"] = base_url or None
    if "vllm_server_host" in fields:
        kwargs["vllm_server_host"] = str(rollout_cfg.get("vllm_server_host", "127.0.0.1"))
    if "vllm_server_port" in fields:
        kwargs["vllm_server_port"] = int(rollout_cfg.get("vllm_server_port", 8000))
    if "vllm_server_timeout" in fields:
        kwargs["vllm_server_timeout"] = float(rollout_cfg.get("vllm_server_timeout", 240.0))
    if "vllm_group_port" in fields:
        kwargs["vllm_group_port"] = int(rollout_cfg.get("vllm_group_port", 51216))
    if "vllm_gpu_memory_utilization" in fields:
        kwargs["vllm_gpu_memory_utilization"] = float(rollout_cfg.get("vllm_gpu_memory_utilization", 0.5))
    if "vllm_max_model_length" in fields:
        kwargs["vllm_max_model_length"] = int(rollout_cfg.get("vllm_max_model_length"))
    if "vllm_max_model_len" in fields:
        kwargs["vllm_max_model_len"] = int(rollout_cfg.get("vllm_max_model_length"))
    if "vllm_enable_sleep_mode" in fields:
        kwargs["vllm_enable_sleep_mode"] = bool(rollout_cfg.get("vllm_enable_sleep_mode", False))

    if bool(artifacts_cfg.get("save_completions", True)) and "log_completions" in fields:
        kwargs["log_completions"] = True

    filtered = {k: v for k, v in kwargs.items() if k in fields}
    dropped = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if dropped:
        logger.info("GRPOConfig dropped unsupported keys for this TRL version: %s", dropped)

    return GRPOConfig(**filtered)


def _write_run_manifest(cfg: dict[str, Any], run_dir: Path, config_path: str, overrides: list[str]) -> None:
    """Write run metadata for reproducibility/debugging."""
    manifest = {
        "run_id": cfg["run"]["id"],
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config_path": str(Path(config_path).resolve()),
        "overrides": overrides,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mode": cfg["rollout"]["mode"],
        "num_generations": int(cfg["rollout"]["num_generations"]),
    }
    path = run_dir / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _run_boundary_eval_stage(
    *,
    cfg: dict[str, Any],
    ckpt: str,
    stage: str,
    run_dir: Path,
    step: int | None = None,
) -> None:
    """Run start/end boundary evaluation suite and persist per-split artifacts."""
    eval_cfg = cfg.get("eval", {})
    boundary_cfg = eval_cfg.get("boundary_eval", {})
    if not bool(boundary_cfg.get("enabled", True)):
        return

    boundary_splits = [str(x) for x in eval_cfg.get("boundary_splits", [])]
    if not boundary_splits:
        return

    split_map = cfg.get("data", {}).get("eval_splits", {})
    prompt_cfg = cfg.get("prompt", {})
    default_system_prompt = str(prompt_cfg.get("system_prompt", ""))
    per_split_prompt = prompt_cfg.get("eval_system_prompt_by_split", {})

    hub_cfg = cfg.get("integrations", {}).get("hf_hub", {})
    token_env = str(hub_cfg.get("token_env", "HF_TOKEN"))
    hf_token = os.environ.get(token_env, "")
    truncation_retry_cfg = eval_cfg.get("truncation_retry", {})
    profiles = build_eval_profiles(eval_cfg)
    multi_profile = len(profiles) > 1

    out_root = run_dir / "eval_boundary" / stage
    out_root.mkdir(parents=True, exist_ok=True)

    aggregate: dict[str, Any] = {
        "stage": stage,
        "checkpoint": ckpt,
        "splits": {},
    }

    for split in boundary_splits:
        split_path = Path(str(split_map.get(split, "")))
        if not split_path.exists():
            logger.warning("Boundary eval skip split=%s (missing file: %s)", split, split_path)
            continue

        profile_runs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for profile in profiles:
            summary, details = evaluate_checkpoint(
                ckpt=ckpt,
                split_name=split,
                split_path=str(split_path),
                system_prompt=str(per_split_prompt.get(split, default_system_prompt)),
                hf_token=hf_token,
                n_generations=int(profile.get("n_generations", eval_cfg.get("n_generations", 1))),
                batch_size=int(eval_cfg.get("batch_size", 8)),
                temperature=float(profile.get("temperature", eval_cfg.get("temperature", 0.6))),
                top_p=float(profile.get("top_p", eval_cfg.get("top_p", 1.0))),
                max_new_tokens=int(eval_cfg.get("max_new_tokens", 1024)),
                limit=int(boundary_cfg.get("limit", 0)),
                evaluator_cfg=eval_cfg.get("evaluator", {}),
                truncation_retry_enabled=bool(truncation_retry_cfg.get("enabled", False)),
                truncation_retry_max_retries=int(truncation_retry_cfg.get("max_retries", 0)),
                truncation_retry_max_new_tokens=int(
                    truncation_retry_cfg.get("retry_max_new_tokens", int(eval_cfg.get("max_new_tokens", 1024)))
                ),
            )
            split_out = out_root / split
            if multi_profile:
                split_out = split_out / str(profile.get("name", "profile"))
            save_eval_outputs(summary, details, split_out)
            profile_runs.append((profile, summary))

        merged_summary = combine_eval_profile_summaries(split, profile_runs)
        aggregate["splits"][split] = merged_summary
        with (out_root / split / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(merged_summary, f, ensure_ascii=False, indent=2)

        if wandb is not None and wandb.run is not None:
            n_probs = int(merged_summary.get("n_problems", 0))
            payload: dict[str, float] = {}
            for key, value in merged_summary.items():
                if isinstance(key, str) and key.startswith("pass@"):
                    payload[f"eval_boundary/{stage}/{split}_n{n_probs}/{key}"] = float(value)
            if payload:
                if step is None:
                    wandb.log(payload)
                else:
                    wandb.log(payload, step=int(step))

        logger.info(
            "Boundary eval stage=%s split=%s pass@1=%.4f pass@3=%s",
            stage,
            split,
            float(merged_summary.get("pass@1", 0.0)),
            f"{float(merged_summary['pass@3']):.4f}" if "pass@3" in merged_summary else "n/a",
        )

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    adapter_merge_cfg = eval_cfg.get("adapter_merge", {})
    if bool(adapter_merge_cfg.get("cleanup_after_eval", True)):
        cleanup_merged_eval_model(str(ckpt))


def main(config_path: str, overrides: list[str]) -> None:
    """Execute a full training run from a YAML config."""
    cfg = load_config(config_path, overrides=overrides)
    global RUN_REF_LINES
    global RUN_REWARD_ACCURACY_WEIGHT
    global RUN_REWARD_FORMAT_WEIGHT
    global RUN_REWARD_EVALUATOR
    RUN_REF_LINES = _build_ref_lines(cfg)
    reward_cfg = cfg.get("reward", {})
    RUN_REWARD_ACCURACY_WEIGHT = float(reward_cfg.get("accuracy_weight", 1.0))
    RUN_REWARD_FORMAT_WEIGHT = float(reward_cfg.get("format_weight", 1.0))
    RUN_REWARD_EVALUATOR = EvaluatorRegistry(cfg.get("eval", {}).get("evaluator", {}))
    run_dir = Path(cfg["paths"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_root = run_dir / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    resolved_config_path = run_dir / "config.resolved.yaml"
    dump_resolved_config(cfg, resolved_config_path)
    _write_run_manifest(cfg, run_dir, config_path, overrides)

    run = cfg["run"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    rollout_cfg = cfg["rollout"]
    prompt_cfg = cfg["prompt"]
    data_cfg = cfg["data"]
    wandb_cfg = cfg.get("integrations", {}).get("wandb", {})
    hub_cfg = cfg.get("integrations", {}).get("hf_hub", {})
    lora_cfg = cfg.get("lora", {})

    global RUN_GROUP_SIZE
    RUN_GROUP_SIZE = int(rollout_cfg["num_generations"])

    if bool(wandb_cfg.get("enabled", False)):
        os.environ.setdefault("WANDB_PROJECT", str(wandb_cfg.get("project", "cppo")))
        api_env = str(wandb_cfg.get("api_key_env", "WANDB_API_KEY"))
        os.environ.setdefault("WANDB_API_KEY", os.environ.get(api_env, ""))

    set_seed(int(run["seed"]))

    logger.info("Loading training data from %s", data_cfg["train_path"])
    train_rows = _read_jsonl(str(data_cfg["train_path"]))
    if not train_rows:
        raise RuntimeError(f"No training rows found at {data_cfg['train_path']}")
    _validate_train_rows(train_rows)
    train_dataset = _build_train_dataset(train_rows, system_prompt=str(prompt_cfg["system_prompt"]))

    primary_split = str(cfg.get("eval", {}).get("primary_split", ""))
    eval_rows: list[dict[str, Any]] = []
    if primary_split and primary_split in data_cfg.get("eval_splits", {}):
        eval_rows = _read_jsonl(str(data_cfg["eval_splits"][primary_split]))

    token_env = str(hub_cfg.get("token_env", "HF_TOKEN"))
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg["model_name_or_path"]),
        trust_remote_code=True,
        token=os.environ.get(token_env, "") or None,
    )
    custom_chat_template = str(model_cfg.get("chat_template", "")).strip()
    if custom_chat_template:
        tokenizer.chat_template = custom_chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = None
    if bool(lora_cfg.get("enabled", True)):
        peft_config = LoraConfig(
            r=int(lora_cfg.get("rank", 64)),
            lora_alpha=int(lora_cfg.get("alpha", 128)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=list(lora_cfg.get("target_modules", [])),
            task_type="CAUSAL_LM",
        )

    grpo_args = _build_grpo_config(cfg, run_dir=run_dir)

    callbacks: list[TrainerCallback] = [
        RewardStatsCallback(),
        CheckpointArtifactsCallback(cfg=cfg, resolved_config_path=resolved_config_path),
    ]

    mid_eval_cfg = cfg.get("eval", {}).get("mid_eval", {})
    quick_local_mid_eval = bool(mid_eval_cfg.get("quick_local_enabled", False))
    if bool(mid_eval_cfg.get("enabled", False)) and quick_local_mid_eval and eval_rows:
        mid_steps = {int(x) for x in mid_eval_cfg.get("steps", [])}
        skip_steps: set[int] = set()
        on_ckpt_cfg = cfg.get("eval", {}).get("on_checkpoint", {})
        if bool(on_ckpt_cfg.get("enabled", False)):
            save_steps = int(train_cfg.get("save_steps", 0))
            if save_steps > 0:
                skip_steps |= {s for s in mid_steps if s % save_steps == 0}
        boundary_cfg = cfg.get("eval", {}).get("boundary_eval", {})
        if bool(boundary_cfg.get("enabled", True)) and bool(boundary_cfg.get("run_at_end", True)):
            skip_steps.add(int(train_cfg.get("max_steps", 0)))
        effective_mid_steps = sorted(mid_steps - skip_steps)
        if skip_steps:
            logger.info("Mid-eval collision guard enabled. Skipping steps: %s", sorted(skip_steps))
        if mid_steps and not effective_mid_steps:
            logger.info(
                "Mid-eval is enabled but all configured steps collide with checkpoint/boundary eval. "
                "Checkpoint/boundary eval remains active."
            )

        callbacks.append(
            MidEvalCallback(
                tokenizer=tokenizer,
                eval_rows=eval_rows,
                system_prompt=str(prompt_cfg["system_prompt"]),
                max_prompt_length=int(model_cfg["max_prompt_length"]),
                mid_eval_max_new_tokens=int(mid_eval_cfg.get("max_new_tokens", 1024)),
                eval_cfg=mid_eval_cfg,
                skip_steps=skip_steps,
            )
        )
    elif bool(mid_eval_cfg.get("enabled", False)) and not quick_local_mid_eval:
        logger.info(
            "eval.mid_eval is enabled but quick_local_enabled=false. "
            "Periodic run-time eval is handled by eval.on_checkpoint (logged to W&B and saved locally)."
        )

    trainer_cls: type[GRPOTrainer]
    trainer_kwargs: dict[str, Any] = {}
    if str(rollout_cfg.get("mode", "grpo")).lower() == "cppo":
        trainer_cls = CPPOTrainer
        trainer_kwargs = {
            "cppo_pruning": float(rollout_cfg.get("cppo", {}).get("pruning", 0.875)),
            "cppo_metric": str(rollout_cfg.get("cppo", {}).get("metric", "smallest")),
            "cppo_allocation": bool(rollout_cfg.get("cppo", {}).get("allocation", False)),
            "cppo_strategy": str(rollout_cfg.get("cppo", {}).get("strategy", "author_exact")),
        }
    else:
        trainer_cls = GRPOTrainer

    trainer = trainer_cls(
        model=str(model_cfg["model_name_or_path"]),
        reward_funcs=_reward_fn,
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
        **trainer_kwargs,
    )
    for cb in callbacks:
        if hasattr(cb, "set_trainer"):
            cb.set_trainer(trainer)

    boundary_cfg = cfg.get("eval", {}).get("boundary_eval", {})
    if bool(boundary_cfg.get("enabled", True)) and bool(boundary_cfg.get("run_at_start", True)):
        logger.info("Running boundary eval at start (base model)")
        _run_boundary_eval_stage(
            cfg=cfg,
            ckpt=str(model_cfg["model_name_or_path"]),
            stage="start",
            run_dir=run_dir,
            step=0,
        )

    logger.info(
        "Starting training: mode=%s run_id=%s output=%s",
        rollout_cfg.get("mode", "grpo"),
        run["id"],
        checkpoints_root,
    )
    trainer.train()

    final_dir = checkpoints_root / "final"
    trainer.save_model(str(final_dir))
    dump_resolved_config(cfg, final_dir / "config.yaml")

    if bool(boundary_cfg.get("enabled", True)) and bool(boundary_cfg.get("run_at_end", True)):
        logger.info("Running boundary eval at end (final checkpoint)")
        _run_boundary_eval_stage(
            cfg=cfg,
            ckpt=str(final_dir),
            stage="end",
            run_dir=run_dir,
            step=int(train_cfg.get("max_steps", 0)),
        )

    logger.info("Training complete. Final checkpoint: %s", final_dir)


def cli() -> None:
    """CLI wrapper for `main`."""
    parser = argparse.ArgumentParser(description="CPPO/GRPO training entrypoint (YAML-configured)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config key with dotted.key=value (repeatable)",
    )
    args = parser.parse_args()
    main(config_path=args.config, overrides=args.overrides)


if __name__ == "__main__":
    cli()
