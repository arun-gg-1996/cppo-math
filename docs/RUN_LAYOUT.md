# Run Layout

Each run is created under:

`runs/<run_id>/`

Contents:

- `config.resolved.yaml`: fully expanded config used for training.
- `run_manifest.json`: runtime metadata (config source, mode, platform).
- `checkpoint_index.json`: checkpoint registry with metrics and timestamps.
- `checkpoints/`: all checkpoint dirs + `best` and `latest` links.

Each checkpoint directory contains:

- `config.yaml`: config snapshot copied at save time.
- `checkpoint_meta.json`: step, save time, and primary metric.
- `completions/step_<step>.jsonl`: sampled completions from training logs.
- `passk/<split>/summary.json`: split-level pass@k summary.
- `passk/<split>/details.jsonl`: per-problem evaluation outputs (single-profile mode).
- `passk/<split>/<profile_name>/summary.json` + `details.jsonl`: profile outputs (multi-profile mode, e.g. strict pass@1 + pass@3).
- `passk/summary.json`: aggregate split summaries for this checkpoint.
- `_merged_eval_model/` (only when needed):
  - auto-created if checkpoint is adapter-only and eval needs merged weights.
  - cleaned automatically after eval when `eval.adapter_merge.cleanup_after_eval=true` (default).

## Validation and cleanup scripts

- `scripts/verify_run.py`
  - quick run-health check (required files + key CPPO metrics in `trainer_state`).
- `scripts/prune_checkpoints.py`
  - manual checkpoint pruning by `--keep-best-k` and `--keep-last-n`.
  - use `--apply` to actually delete; without it, command is preview-only.

## Notes

- `checkpoints/best` and `checkpoints/latest` are symlinks where possible.
- If symlinks are not supported on the environment, code falls back to copying.
- Periodic in-training eval artifacts are produced by `eval.on_checkpoint` (saved under each checkpoint’s `passk/`).
- Boundary start/end eval artifacts are saved under `runs/<run_id>/eval_boundary/start` and `runs/<run_id>/eval_boundary/end`.
