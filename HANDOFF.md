# CPPO Math Project Handoff

## Project
- Name: `cppo-math`
- Repo path: `<repo_root>`
- Goal: budget-aware CPPO/GRPO training for math reasoning with reproducible artifacts and robust evaluation.

## Current Status
- Repo moved from old path and re-audited.
- Local validation passed (compile, evaluator validation, deterministic smoke checks, local 2-step smoke train, run verification).
- Server run is ready pending your split-GPU smoke/full launch.

## Locked Decisions
- CPPO default strategy: `author_exact`
- Experimental mode kept: `experimental_refill`
- Reward:
  - `accuracy_weight: 0.8`
  - `format_weight: 0.2`
  - combined reward clamped to `[0, 1]`
- Eval profiles:
  - strict pass@1: `n=1`, `temperature=0.0`, `top_p=1.0`
  - pass@3: `n=3`, `temperature=0.6`, `top_p=0.95`
- On-checkpoint eval splits:
  - `gsm8k_test`, `svamp`, `math_500`, `amc_2023`
- Boundary eval (start + end) splits:
  - `gsm_plus`, `asdiv`, `aime_2024`, `aime_2025`, `minerva_math`, `olympiadbench`

## Data Snapshot
- Train:
  - `gsm8k_train`: 7473
- Mid eval:
  - `gsm8k_test`: 1319
  - `svamp`: 1000
  - `math_500`: 500
  - `amc_2023`: 83
- Boundary eval:
  - `gsm_plus`: 10552
  - `asdiv`: 2305
  - `aime_2024`: 30
  - `aime_2025`: 30
  - `minerva_math`: 272
  - `olympiadbench`: 675
- Train/eval overlap check by normalized question: `0`

## Artifact + Sync Behavior
- Every checkpoint stores config + metadata + local artifacts.
- Adapter-only checkpoint eval auto-merges base+adapter when needed.
- Temporary merged eval model is cleaned after eval (default).
- HF sync is re-run after callback artifacts are written, so Hub reflects checkpoint config/passk/meta.

## Key Run Commands
- Local smoke:
  - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/train.py --config configs/local/local_cpu_smoke.yaml`
- Verify run:
  - `.venv/bin/python scripts/verify_run.py`
- Server split mode:
  - GPU1: `.venv/bin/python scripts/vllm_server.py --config config.yaml`
  - GPU0: `.venv/bin/python scripts/train.py --config config.yaml`

## Next Actions
1. Run server split-GPU smoke using `configs/smoke/server_gpu_split_smoke.yaml`.
2. If smoke passes, launch full run with `config.yaml`.
3. Track progress in `docs/progress_journal/` on each major checkpoint/decision.
