from __future__ import annotations

"""Checkpoint evaluation helpers (vLLM generation + pass@k reporting)."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from .evaluator_registry import EvaluatorRegistry

logger = logging.getLogger("cppo.eval")
MERGED_EVAL_MODEL_CACHE: dict[str, str] = {}


def build_eval_profiles(eval_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Build eval profiles from config.

    Default behavior is a single profile from top-level eval args.
    If `eval.passk_profiles.enabled=true`, use configured profile list.
    """
    default_n = int(eval_cfg.get("n_generations", 1))
    default_temp = float(eval_cfg.get("temperature", 0.6))
    default_top_p = float(eval_cfg.get("top_p", 1.0))
    default_profile = {
        "name": "default",
        "n_generations": default_n,
        "temperature": default_temp,
        "top_p": default_top_p,
        "report_k": int(min(3, max(1, default_n))),
    }

    passk_cfg = eval_cfg.get("passk_profiles", {})
    if not isinstance(passk_cfg, dict) or not bool(passk_cfg.get("enabled", False)):
        return [default_profile]

    raw_profiles = passk_cfg.get("profiles", {})
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        return [default_profile]

    profiles: list[dict[str, Any]] = []
    for name, cfg in raw_profiles.items():
        if not isinstance(cfg, dict):
            continue
        if not bool(cfg.get("enabled", True)):
            continue
        n = int(cfg.get("n_generations", default_n))
        n = max(1, n)
        report_k = int(cfg.get("report_k", min(3, n)))
        report_k = max(1, report_k)
        profiles.append(
            {
                "name": str(name),
                "n_generations": n,
                "temperature": float(cfg.get("temperature", default_temp)),
                "top_p": float(cfg.get("top_p", default_top_p)),
                "report_k": report_k,
            }
        )

    return profiles or [default_profile]


def combine_eval_profile_summaries(
    split_name: str,
    profile_runs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    """Merge per-profile summaries into a single split summary.

    This allows strict pass@1 and best-of-k pass@k to be reported together while
    still preserving profile-level summaries.
    """
    if not profile_runs:
        return {
            "split": split_name,
            "n_problems": 0,
            "profiles": {},
        }

    first_summary = profile_runs[0][1]
    merged: dict[str, Any] = {
        "model": first_summary.get("model"),
        "model_resolved": first_summary.get("model_resolved"),
        "split": split_name,
        "n_problems": int(first_summary.get("n_problems", 0)),
        "profiles": {},
    }

    for profile, summary in profile_runs:
        name = str(profile.get("name", "profile"))
        merged["profiles"][name] = summary
        report_k = int(profile.get("report_k", min(3, int(summary.get("n_generations", 1)))))
        metric_key = f"pass@{report_k}"
        metric_val = summary.get(metric_key)
        if metric_val is None and report_k == 1:
            metric_val = summary.get("pass@1")
        if isinstance(metric_val, (int, float)):
            merged[metric_key] = float(metric_val)

    if "pass@1" not in merged and isinstance(first_summary.get("pass@1"), (int, float)):
        merged["pass@1"] = float(first_summary["pass@1"])

    return merged


def _has_model_weights(path: Path) -> bool:
    """Return True when a directory contains full model weights."""
    if not path.exists() or not path.is_dir():
        return False
    fixed = {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
    for name in fixed:
        if (path / name).exists():
            return True
    if any(path.glob("model-*.safetensors")):
        return True
    if any(path.glob("pytorch_model-*.bin")):
        return True
    return False


def _is_adapter_only_checkpoint(path: Path) -> bool:
    """Detect LoRA adapter-only checkpoint directories."""
    has_adapter = (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
    return (
        path.exists()
        and path.is_dir()
        and (path / "adapter_config.json").exists()
        and has_adapter
        and not _has_model_weights(path)
    )


def _copy_if_exists(src: Path, dst: Path, filename: str) -> None:
    """Copy `filename` from src to dst if present and not already copied."""
    s = src / filename
    d = dst / filename
    if s.exists() and not d.exists():
        shutil.copy2(s, d)


def _resolve_eval_model_path(ckpt: str, hf_token: str) -> str:
    """Resolve model path for eval.

    If `ckpt` is adapter-only, merge adapter + base once and cache the merged path.
    """
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists() or not ckpt_path.is_dir():
        return ckpt

    ckpt_abs = str(ckpt_path.resolve())
    cached = MERGED_EVAL_MODEL_CACHE.get(ckpt_abs, "")
    if cached and Path(cached).exists():
        return cached

    if not _is_adapter_only_checkpoint(ckpt_path):
        return ckpt_abs

    merged_dir = ckpt_path / "_merged_eval_model"
    if _has_model_weights(merged_dir):
        resolved = str(merged_dir.resolve())
        MERGED_EVAL_MODEL_CACHE[ckpt_abs] = resolved
        return resolved

    with (ckpt_path / "adapter_config.json").open("r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model = str(adapter_cfg.get("base_model_name_or_path", "")).strip()
    if not base_model:
        raise RuntimeError(
            f"Adapter checkpoint is missing base_model_name_or_path: {ckpt_path / 'adapter_config.json'}"
        )

    logger.info("Merging adapter checkpoint for eval: ckpt=%s base=%s", ckpt_path, base_model)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    token = hf_token or None
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "token": token,
    }
    if torch.cuda.is_available():
        bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16 if bf16_supported else torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    peft_model = PeftModel.from_pretrained(base, str(ckpt_path), token=token)
    merged = peft_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)

    try:
        tok = AutoTokenizer.from_pretrained(str(ckpt_path), trust_remote_code=True, token=token)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, token=token)
    tok.save_pretrained(str(merged_dir))

    for name in (
        "chat_template.jinja",
        "generation_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        _copy_if_exists(ckpt_path, merged_dir, name)

    with (merged_dir / "merge_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_checkpoint": ckpt_abs,
                "base_model_name_or_path": base_model,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    try:
        del peft_model
        del base
        del merged
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    resolved = str(merged_dir.resolve())
    MERGED_EVAL_MODEL_CACHE[ckpt_abs] = resolved
    return resolved


def cleanup_merged_eval_model(ckpt: str) -> None:
    """Remove persisted merged eval model for a checkpoint path (if present)."""
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists() or not ckpt_path.is_dir():
        return

    ckpt_abs = str(ckpt_path.resolve())
    merged_dir = ckpt_path / "_merged_eval_model"
    if merged_dir.exists() and merged_dir.is_dir():
        shutil.rmtree(merged_dir, ignore_errors=True)

    cached = MERGED_EVAL_MODEL_CACHE.get(ckpt_abs, "")
    if cached:
        MERGED_EVAL_MODEL_CACHE.pop(ckpt_abs, None)


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _render_prompt(tokenizer: AutoTokenizer, system_prompt: str, question: str) -> str:
    """Render prompt text with tokenizer chat template fallback."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{system_prompt}\n\n{question}"


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator used by common code/math benchmarks."""
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    vals = [1.0 - (k / x) for x in np.arange(n - c + 1, n + 1)]
    return 1.0 - float(np.prod(vals))


def _is_truncated_generation(item: Any, max_new_tokens: int) -> bool:
    """Heuristic to detect length-truncated generations."""
    finish_reason = str(getattr(item, "finish_reason", "") or "").strip().lower()
    stop_reason = str(getattr(item, "stop_reason", "") or "").strip().lower()
    token_ids = getattr(item, "token_ids", None)

    reason_flags = {
        "length",
        "max_tokens",
        "max_token",
        "max_new_tokens",
        "max_tokens_reached",
    }
    if finish_reason in reason_flags or stop_reason in reason_flags:
        return True

    if isinstance(token_ids, (list, tuple)) and len(token_ids) >= int(max_new_tokens):
        return True
    return False


def evaluate_checkpoint(
    *,
    ckpt: str,
    split_name: str,
    split_path: str,
    system_prompt: str,
    hf_token: str,
    n_generations: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    limit: int = 0,
    evaluator_cfg: dict[str, Any] | None = None,
    truncation_retry_enabled: bool = False,
    truncation_retry_max_retries: int = 0,
    truncation_retry_max_new_tokens: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one checkpoint on one split and return summary + per-problem details."""
    from vllm import LLM, SamplingParams

    eval_model = _resolve_eval_model_path(ckpt, hf_token)

    problems = _read_jsonl(split_path)
    if not problems:
        raise RuntimeError(f"No eval rows found at {split_path}")
    if limit > 0:
        problems = problems[:limit]

    tokenizer = AutoTokenizer.from_pretrained(eval_model, trust_remote_code=True, token=(hf_token or None))
    prompts = [_render_prompt(tokenizer, system_prompt, str(p["question"])) for p in problems]

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        n=n_generations,
        max_tokens=max_new_tokens,
    )

    logger.info(
        "Evaluating checkpoint=%s resolved_model=%s split=%s n=%d n_gen=%d batch=%d",
        ckpt,
        eval_model,
        split_name,
        len(problems),
        n_generations,
        batch_size,
    )
    llm = LLM(model=eval_model, trust_remote_code=True)

    details: list[dict[str, Any]] = []
    evaluator = EvaluatorRegistry(evaluator_cfg)
    backend_counts: dict[str, int] = {}
    retry_candidates = 0
    retry_count = 0
    total_pass1 = 0
    passk_sum = 0.0
    k_for_report = min(3, n_generations)
    retry_max_tokens = int(truncation_retry_max_new_tokens or max_new_tokens)
    do_retry = bool(truncation_retry_enabled) and int(truncation_retry_max_retries) > 0 and retry_max_tokens > max_new_tokens

    for i in range(0, len(problems), batch_size):
        batch_problems = problems[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]
        outputs = llm.generate(batch_prompts, sampling)

        batch_completions: list[list[Any]] = [list(out.outputs) for out in outputs]

        if do_retry:
            max_retries = int(truncation_retry_max_retries)
            retry_sampling = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                n=1,
                max_tokens=retry_max_tokens,
            )
            retry_jobs: list[tuple[int, int]] = []
            for req_idx, req_outs in enumerate(batch_completions):
                for gen_idx, item in enumerate(req_outs):
                    if _is_truncated_generation(item, max_new_tokens):
                        retry_jobs.append((req_idx, gen_idx))
            retry_candidates += len(retry_jobs)

            for _attempt in range(max_retries):
                if not retry_jobs:
                    break
                retry_prompts = [batch_prompts[req_idx] for req_idx, _ in retry_jobs]
                retry_outputs = llm.generate(retry_prompts, retry_sampling)

                next_retry_jobs: list[tuple[int, int]] = []
                for (req_idx, gen_idx), retry_out in zip(retry_jobs, retry_outputs, strict=True):
                    if retry_out.outputs:
                        new_item = retry_out.outputs[0]
                        batch_completions[req_idx][gen_idx] = new_item
                        retry_count += 1
                        if _is_truncated_generation(new_item, retry_max_tokens):
                            next_retry_jobs.append((req_idx, gen_idx))
                retry_jobs = next_retry_jobs

        for prob, req_outs in zip(batch_problems, batch_completions, strict=True):
            gt = str(prob["answer"])
            scores: list[float] = []
            texts: list[str] = []
            used_backends: list[str] = []
            for item in req_outs:
                text = item.text
                eval_res = evaluator.score(
                    split_name=split_name,
                    predicted_text=text,
                    ground_truth=gt,
                    row=prob,
                )
                score = eval_res.score
                texts.append(text)
                scores.append(score)
                used_backends.append(eval_res.backend)
                backend_counts[eval_res.backend] = backend_counts.get(eval_res.backend, 0) + 1

            c = int(sum(1 for s in scores if s == 1.0))
            p1 = 1 if (scores and scores[0] == 1.0) else 0
            total_pass1 += p1
            passk_sum += _pass_at_k(n_generations, c, k_for_report)

            details.append(
                {
                    "id": prob.get("id", prob.get("problem_id", "unknown")),
                    "source": prob.get("source", "unknown"),
                    "difficulty": prob.get("difficulty", "unknown"),
                    "answer": gt,
                    "scores": scores,
                    "score_backends": used_backends,
                    "num_correct": c,
                    "pass@1": float(p1),
                    "completions": texts,
                }
            )

    n_probs = len(problems)
    summary = {
        "model": ckpt,
        "model_resolved": eval_model,
        "split": split_name,
        "n_problems": n_probs,
        "n_generations": n_generations,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "pass@1": (total_pass1 / n_probs) if n_probs else 0.0,
        f"pass@{k_for_report}": (passk_sum / n_probs) if n_probs else 0.0,
        "evaluator_backend_counts": backend_counts,
        "truncation_retry": {
            "enabled": bool(do_retry),
            "max_retries": int(truncation_retry_max_retries),
            "retry_max_new_tokens": int(retry_max_tokens),
            "retry_candidates": int(retry_candidates),
            "retry_count": int(retry_count),
        },
    }
    return summary, details


def save_eval_outputs(summary: dict[str, Any], details: list[dict[str, Any]], out_dir: Path) -> None:
    """Persist eval summary and detailed rows to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (out_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
