# Code Overview (Simple)

This file explains the important code paths in plain language.

## Training flow

1. `scripts/train.py`
- Small CLI wrapper.
- Loads `src/` into `PYTHONPATH`.
- Calls `cppo.train.main(...)`.

2. `src/cppo/config_loader.py`
- Loads YAML.
- Applies `extends` chain and CLI `--set` overrides.
- Resolves paths.
- Validates CPPO math and env requirements.

3. `src/cppo/train.py`
- Reads train JSONL.
- Builds chat-format training dataset.
- Defines reward function (`accuracy + format`, clamped to `[0, 1]`).
- Chooses trainer:
  - `GRPOTrainer` for GRPO mode.
  - `CPPOTrainer` for CPPO mode.
- Registers callbacks for artifacts/eval/retention.
- Runs optional boundary eval at start and end.

4. `src/cppo/trainer_cppo.py`
- Extends TRL `GRPOTrainer`.
- Computes rewards/advantages on full groups.
- Prunes by `|advantage|`.
- Supports two strategies:
  - `author_exact` (default, paper-aligned behavior).
  - `experimental_refill` (custom variant kept for experiments).

5. `src/cppo/callbacks.py`
- On every checkpoint save:
  - copies resolved config into checkpoint
  - saves completion snapshots
  - runs checkpoint eval (`pass@k`)
  - updates checkpoint index and best/latest links
  - optionally prunes old checkpoints
  - syncs checkpoint to HF Hub when enabled

## Evaluation flow

1. `scripts/eval.py`
- Manual eval CLI.
- Supports one or many splits.
- Supports dual-profile pass@k (strict pass@1 + pass@3).

2. `src/cppo/eval.py`
- Runs vLLM generation for eval.
- Handles adapter-only checkpoints by temporary merge.
- Scores with evaluator registry.
- Saves `summary.json` + `details.jsonl`.

3. `src/cppo/evaluator_registry.py`
- Routes per-split scoring backend:
  - custom evaluator (if configured)
  - `math_verify` (if available)
  - fallback SymPy checker

4. `src/cppo/reward.py`
- Answer extraction from model output.
- Format compliance check (`<think>...</think><answer>...</answer>`).
- Symbolic/numeric equivalence via SymPy fallback path.

## Artifact helpers

- `src/cppo/io_artifacts.py`
  - JSON helpers
  - checkpoint metadata/index helpers
  - best/latest link updates
  - retention pruning (`keep-best-k`, `keep-last-n`)

- `scripts/verify_run.py`
  - quick health check for run outputs and required metrics.

## Where to start reading code

If you want to understand the project quickly:
1. `src/cppo/train.py`
2. `src/cppo/trainer_cppo.py`
3. `src/cppo/callbacks.py`
4. `src/cppo/eval.py`
