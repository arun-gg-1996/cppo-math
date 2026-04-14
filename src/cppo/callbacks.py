from __future__ import annotations

"""Trainer callbacks for checkpoint artifacts, eval, retention, and hub sync."""

import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from .eval import (
    build_eval_profiles,
    cleanup_merged_eval_model,
    combine_eval_profile_summaries,
    evaluate_checkpoint,
    save_eval_outputs,
)
from .io_artifacts import (
    append_checkpoint_row,
    copy_resolved_config_to_checkpoint,
    prune_checkpoints,
    save_checkpoint_index,
    select_best_checkpoint,
    update_best_symlink,
    update_latest_symlink,
    write_checkpoint_meta,
    write_json,
)

logger = logging.getLogger("cppo.callbacks")
try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


class CheckpointArtifactsCallback(TrainerCallback):
    """On-save hooks for reproducible checkpoint artifacts."""

    def __init__(self, cfg: dict[str, Any], resolved_config_path: Path):
        self.cfg = cfg
        self.resolved_config_path = resolved_config_path
        self.run_id = str(cfg["run"]["id"])
        self.run_dir = Path(cfg["paths"]["run_dir"])
        self.checkpoints_root = self.run_dir / "checkpoints"
        self.index_path = self.run_dir / "checkpoint_index.json"
        self._trainer_ref = None
        self.eval_last_seconds = 0.0
        self.eval_total_seconds = 0.0

    def set_trainer(self, trainer: Any) -> None:
        """Store trainer reference for post-save helper calls."""
        self._trainer_ref = trainer

    def _sync_checkpoint_to_hub(self, checkpoint_dir: Path) -> None:
        """Push a checkpoint after callback-side files are written."""
        hub_cfg = self.cfg.get("integrations", {}).get("hf_hub", {})
        if not bool(hub_cfg.get("push_to_hub", False)):
            return

        trainer = self._trainer_ref
        if trainer is None or not hasattr(trainer, "_push_from_checkpoint"):
            logger.warning("HF sync skipped: trainer does not expose _push_from_checkpoint")
            return

        try:
            trainer._push_from_checkpoint(str(checkpoint_dir))
        except Exception:
            logger.warning("HF sync failed for checkpoint=%s", checkpoint_dir, exc_info=True)

    def _resolve_checkpoint_dir(self, step: int) -> Path | None:
        """Resolve checkpoint folder for a save step with safe fallback."""
        candidate = self.checkpoints_root / f"checkpoint-{step}"
        if candidate.exists():
            return candidate

        # Fallback to latest checkpoint directory if naming differs.
        all_ckpts = sorted(
            [p for p in self.checkpoints_root.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )
        return all_ckpts[-1] if all_ckpts else None

    def _evaluate_checkpoint(self, checkpoint_dir: Path) -> tuple[float | None, dict[str, Any]]:
        """Run configured on-checkpoint eval and return primary metric + summaries."""
        eval_cfg = self.cfg.get("eval", {})
        on_ckpt = eval_cfg.get("on_checkpoint", {})
        if not bool(on_ckpt.get("enabled", False)):
            return None, {}

        split_map = self.cfg.get("data", {}).get("eval_splits", {})
        prompt_cfg = self.cfg.get("prompt", {})
        default_system_prompt = str(prompt_cfg.get("system_prompt", ""))
        per_split_prompt = prompt_cfg.get("eval_system_prompt_by_split", {})
        primary_split = str(eval_cfg.get("primary_split", ""))
        split_order = on_ckpt.get("splits") or ([primary_split] if primary_split else list(split_map.keys()))

        hf_token = ""
        hub_cfg = self.cfg.get("integrations", {}).get("hf_hub", {})
        token_env = str(hub_cfg.get("token_env", "HF_TOKEN"))
        import os

        hf_token = os.environ.get(token_env, "")

        primary_metric: float | None = None
        per_split: dict[str, Any] = {}
        truncation_retry_cfg = eval_cfg.get("truncation_retry", {})
        profiles = build_eval_profiles(eval_cfg)
        multi_profile = len(profiles) > 1
        rollout_cfg = self.cfg.get("rollout", {})
        use_server_eval = bool(rollout_cfg.get("use_vllm", True)) and str(rollout_cfg.get("vllm_mode", "")).strip().lower() == "server"
        server_base_url = str(rollout_cfg.get("vllm_server_base_url", "")).strip()
        if not server_base_url:
            use_server_eval = False
        server_timeout = float(rollout_cfg.get("vllm_server_timeout", 240.0))

        for split in split_order:
            split = str(split)
            if split not in split_map:
                logger.warning("Skipping unknown eval split '%s'", split)
                continue
            split_path = Path(str(split_map[split]))
            if not split_path.exists():
                logger.warning("Skipping eval split '%s' because file is missing: %s", split, split_path)
                continue

            profile_runs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for profile in profiles:
                summary, details = evaluate_checkpoint(
                    ckpt=str(checkpoint_dir),
                    split_name=split,
                    split_path=str(split_path),
                    system_prompt=str(per_split_prompt.get(split, default_system_prompt)),
                    hf_token=hf_token,
                    n_generations=int(profile.get("n_generations", eval_cfg.get("n_generations", 1))),
                    batch_size=int(eval_cfg.get("batch_size", 8)),
                    temperature=float(profile.get("temperature", eval_cfg.get("temperature", 0.6))),
                    top_p=float(profile.get("top_p", eval_cfg.get("top_p", 1.0))),
                    max_new_tokens=int(eval_cfg.get("max_new_tokens", 1024)),
                    report_k=int(profile.get("report_k")) if profile.get("report_k") is not None else None,
                    limit=int(on_ckpt.get("limit", 0)),
                    evaluator_cfg=eval_cfg.get("evaluator", {}),
                    truncation_retry_enabled=bool(truncation_retry_cfg.get("enabled", False)),
                    truncation_retry_max_retries=int(truncation_retry_cfg.get("max_retries", 0)),
                    truncation_retry_max_new_tokens=int(
                        truncation_retry_cfg.get("retry_max_new_tokens", int(eval_cfg.get("max_new_tokens", 1024)))
                    ),
                    use_vllm_server=bool(use_server_eval),
                    vllm_server_base_url=server_base_url,
                    vllm_server_timeout=server_timeout,
                )
                out_dir = checkpoint_dir / "passk" / split
                if multi_profile:
                    out_dir = out_dir / str(profile.get("name", "profile"))
                save_eval_outputs(summary, details, out_dir)
                profile_runs.append((profile, summary))

            merged_summary = combine_eval_profile_summaries(split, profile_runs)
            per_split[split] = merged_summary
            write_json(checkpoint_dir / "passk" / split / "summary.json", merged_summary)
            if split == primary_split:
                primary_metric = float(merged_summary.get("pass@1", 0.0))

        summary_path = checkpoint_dir / "passk" / "summary.json"
        write_json(
            summary_path,
            {
                "primary_split": primary_split,
                "primary_metric": primary_metric,
                "splits": per_split,
            },
        )
        return primary_metric, per_split

    def _save_completion_snapshot(self, checkpoint_dir: Path, step: int) -> None:
        """Persist a capped sample of prompts/completions for auditability."""
        artifacts_cfg = self.cfg.get("artifacts", {})
        if not bool(artifacts_cfg.get("save_completions", True)):
            return
        trainer = self._trainer_ref
        if trainer is None:
            return

        logs = getattr(trainer, "_logs", None)
        if not isinstance(logs, dict):
            return

        prompts = logs.get("prompt", [])
        completions = logs.get("completion", [])
        if not isinstance(prompts, list) or not isinstance(completions, list):
            return
        if not prompts or not completions:
            return

        reward_logs = logs.get("rewards", {})
        advantages = logs.get("advantages", [])
        max_items = int(artifacts_cfg.get("max_completions_per_checkpoint", 256))
        n = min(len(prompts), len(completions), max_items)
        start = max(0, len(prompts) - n)

        out_dir = checkpoint_dir / "completions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"step_{step:08d}.jsonl"

        with out_path.open("w", encoding="utf-8") as f:
            for i in range(start, len(prompts)):
                row: dict[str, Any] = {
                    "index": i,
                    "prompt": prompts[i],
                    "completion": completions[i],
                }
                if isinstance(advantages, list) and i < len(advantages):
                    row["advantage"] = advantages[i]
                if isinstance(reward_logs, dict):
                    rewards: dict[str, Any] = {}
                    for name, values in reward_logs.items():
                        if isinstance(values, list) and i < len(values):
                            rewards[str(name)] = values[i]
                    if rewards:
                        row["rewards"] = rewards
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        write_json(
            out_dir / "latest.json",
            {
                "step": step,
                "count": n,
                "path": str(out_path.resolve()),
            },
        )

    def on_save(self, args, state, control, **kwargs):
        """Main save hook: write artifacts, eval, hub sync, and retention prune."""
        if not state.is_world_process_zero:
            return control

        step = int(state.global_step)
        checkpoint_dir = self._resolve_checkpoint_dir(step)
        if checkpoint_dir is None:
            logger.warning("on_save: could not resolve checkpoint directory for step=%d", step)
            return control

        copy_resolved_config_to_checkpoint(self.resolved_config_path, checkpoint_dir)
        self._save_completion_snapshot(checkpoint_dir, step)

        primary_metric: float | None = None
        split_metrics: dict[str, Any] = {}
        eval_started = time.perf_counter()
        eval_elapsed = 0.0
        try:
            primary_metric, split_metrics = self._evaluate_checkpoint(checkpoint_dir)
        except Exception as e:
            logger.warning("Checkpoint eval failed at step=%d (%s)", step, e, exc_info=True)
        finally:
            eval_elapsed = float(time.perf_counter() - eval_started)
            self.eval_last_seconds = eval_elapsed
            self.eval_total_seconds += eval_elapsed
            logger.info(
                "Checkpoint eval timing step=%d seconds=%.2f cumulative=%.2f",
                step,
                eval_elapsed,
                self.eval_total_seconds,
            )
            adapter_merge_cfg = self.cfg.get("eval", {}).get("adapter_merge", {})
            if bool(adapter_merge_cfg.get("cleanup_after_eval", True)):
                cleanup_merged_eval_model(str(checkpoint_dir))

        if wandb is not None and wandb.run is not None:
            wb_payload: dict[str, float] = {}
            if split_metrics:
                for split, summary in split_metrics.items():
                    n_probs = int(summary.get("n_problems", 0))
                    for key, value in summary.items():
                        if isinstance(key, str) and key.startswith("pass@"):
                            wb_payload[f"eval/{split}_n{n_probs}/{key}"] = float(value)
            wb_payload["eval_timing/checkpoint_seconds"] = float(eval_elapsed)
            wb_payload["eval_timing/checkpoint_cumulative_seconds"] = float(self.eval_total_seconds)
            if wb_payload:
                wb_payload["train/global_step"] = float(step)
                wandb.log(wb_payload)

        write_checkpoint_meta(
            checkpoint_dir,
            run_id=self.run_id,
            global_step=step,
            save_reason="checkpoint",
            primary_metric=primary_metric,
        )

        row = {
            "global_step": step,
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "primary_metric": primary_metric,
            "eval_seconds": float(eval_elapsed),
            "eval_cumulative_seconds": float(self.eval_total_seconds),
            "split_metrics": split_metrics,
        }
        rows = append_checkpoint_row(self.index_path, row)

        self._sync_checkpoint_to_hub(checkpoint_dir)

        update_latest_symlink(self.checkpoints_root, checkpoint_dir)
        best_row = select_best_checkpoint(rows)
        best_dir = Path(str(best_row["checkpoint_dir"])) if best_row else None
        update_best_symlink(self.checkpoints_root, best_dir)

        retention = self.cfg.get("artifacts", {}).get("retention", {})
        if bool(retention.get("enabled", False)):
            keep_best_k = int(retention.get("keep_best_k", 3))
            keep_last_n = int(retention.get("keep_last_n", 2))
            prune_result = prune_checkpoints(
                checkpoints_root=self.checkpoints_root,
                rows=rows,
                keep_best_k=keep_best_k,
                keep_last_n=keep_last_n,
                apply=True,
            )
            write_json(
                self.run_dir / "last_prune.json",
                {
                    "keep_best_k": keep_best_k,
                    "keep_last_n": keep_last_n,
                    "deleted": prune_result["deleted"],
                    "kept_count": len(prune_result["kept"]),
                },
            )
            kept_resolved = set(prune_result["kept"])
            remaining_rows = [
                r for r in rows if str(Path(str(r.get("checkpoint_dir", ""))).resolve()) in kept_resolved
            ]
            save_checkpoint_index(self.index_path, remaining_rows)

        return control
