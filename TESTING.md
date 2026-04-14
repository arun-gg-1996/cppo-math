# Testing Runbook

This is the strict checklist for:
- local Mac validation
- server split-GPU smoke validation
- full run preflight + post-run verification

## 0) Setup (local + server)

What this does:
- creates and activates your virtual environment
- installs dependencies
- prompts for required tokens and writes `.env`

```bash
cd <repo_root>
bash setup_env.sh
source .venv/bin/activate
```

`setup_env.sh` prompts for `WANDB_API_KEY` and `HF_TOKEN` (hidden input), updates `.env`, and runs `wandb login` + `hf auth login`.

## 1) Local Mac (no GPU) - rigorous pass

### 1.1 Static compile + import sanity

What this does:
- checks Python syntax/import errors early (fast fail before longer tests)

```bash
python -m py_compile src/cppo/*.py scripts/*.py train.py eval.py smoke_test.py
```

Pass criteria:
- no traceback

### 1.2 Evaluator robustness

What this does:
- validates answer extraction + checking logic on local datasets

```bash
python scripts/validate_evaluator.py --fail-below 0.98
```

Pass criteria:
- final line shows `[PASS] Evaluator looks robust...`

### 1.3 Deterministic smoke tests (no model training)

What this does:
- runs small unit tests for reward, pruning, allocation math, and metric tracking

```bash
python scripts/smoke_test.py --skip-trainer
```

Pass criteria:
- `deterministic unit checks: PASS`
- includes reward/extractor/pruning/allocation math checks
- includes unique-problem + per-source metrics tracking check

### 1.4 Local offline end-to-end smoke train (author-exact default)

What this does:
- runs a 2-step tiny offline training job to prove trainer wiring works end-to-end

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/train.py --config configs/local/local_cpu_smoke.yaml
```

Pass criteria:
- training finishes 2 steps
- final checkpoint saved
- no vLLM dependency required

### 1.5 Local offline smoke train (allocation on, both CPPO strategies)

What this does:
- confirms both CPPO strategy modes execute without crashes:
  - `author_exact`
  - `experimental_refill`

Author-exact:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/train.py --config configs/local/local_cpu_smoke.yaml --set rollout.cppo.allocation=true --set rollout.cppo.strategy=author_exact
```

Experimental refill:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/train.py --config configs/local/local_cpu_smoke.yaml --set rollout.cppo.allocation=true --set rollout.cppo.strategy=experimental_refill
```

Pass criteria:
- both runs finish
- logs include `cppo/author_exact_enabled=1` for author mode and `0` for experimental mode

### 1.6 Verify latest run artifacts + trainer metrics

What this does:
- checks latest run folder has required files
- checks checkpoint files exist
- checks key CPPO metrics are present in trainer logs

```bash
python scripts/verify_run.py
```

Pass criteria:
- `[PASS] Run verification passed.`

## 2) Server smoke (2xA100 split mode)

### 2.1 Start vLLM server in screen (GPU1)

What this does:
- starts rollout service on GPU1

```bash
screen -S cppo-vllm-smoke
cd <repo_root>
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python scripts/vllm_server.py --config configs/smoke/server_gpu_split_smoke.yaml
```

Detach: `Ctrl-a d`

### 2.2 Start training smoke in screen (GPU0)

What this does:
- starts policy training on GPU0 and talks to vLLM server from step 2.1

```bash
screen -S cppo-train-smoke
cd <repo_root>
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/smoke/server_gpu_split_smoke.yaml
```

Detach: `Ctrl-a d`

### 2.3 Check screens and logs

What this does:
- lets you verify both services are alive and inspect logs

```bash
screen -ls
screen -r cppo-vllm-smoke
screen -r cppo-train-smoke
```

Pass criteria:
- vLLM screen stays alive
- training screen reaches step 10 and exits cleanly

### 2.4 Verify smoke artifacts

What this does:
- validates checkpoint + eval files after smoke run

```bash
python scripts/verify_run.py --expect-eval
```

Pass criteria:
- `[PASS] Run verification passed.`
- eval summaries exist

## 3) Full run (main config, split mode)

### 3.1 Start rollout server (GPU1)

What this does:
- launches rollout service for full run

```bash
screen -S cppo-vllm
cd <repo_root>
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python scripts/vllm_server.py --config config.yaml
```

Detach: `Ctrl-a d`

### 3.2 Start training (GPU0)

What this does:
- launches full training job that uses the rollout service

```bash
screen -S cppo-train
cd <repo_root>
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config.yaml
```

Detach: `Ctrl-a d`

### 3.3 Post-run verification

What this does:
- validates final run artifacts and expected eval outputs

```bash
python scripts/verify_run.py --expect-eval
```

Optional manual eval spot-check:

```bash
python scripts/eval.py --config config.yaml --checkpoint runs/<run_id>/checkpoints/checkpoint-100 --splits gsm8k_test,svamp,math_500,amc_2023
```

What this also does now:
- if checkpoint-100 is adapter-only, eval auto-merges adapter + base into `_merged_eval_model` under that checkpoint and evaluates the merged path.
- by default, that temporary merged folder is cleaned automatically after eval (`eval.adapter_merge.cleanup_after_eval=true`).
- by default, eval runs strict `pass@1` and best-of-3 `pass@3` as separate decode profiles and merges both metrics into split summary.

## 4) W&B metric checklist

Expected CPPO core metrics:
- `cppo/pruning_ratio`
- `cppo/kept_per_group`
- `cppo/kept_fraction`
- `cppo/allocation_enabled`
- `cppo/filled_fraction`
- `cppo/author_exact_enabled`

Expected data coverage metrics (from reward callback):
- `data/unique_problems_seen`
- `data/<source>_seen`
- `data/<source>_processed`

Expected reward diagnostics:
- `reward/mean`
- `reward/std`
- `reward/<source>_mean`
- `grpo/all_zero_fraction`
- `grpo/reward_std_mean`

Notes:
- default mode is `rollout.cppo.strategy=author_exact`
- use `--set rollout.cppo.strategy=experimental_refill` for your custom variant
- periodic mid-run eval is from `eval.on_checkpoint` (W&B + local artifacts).
- `eval.mid_eval` quick-local path is debug-only and disabled by default.
- dual eval profiles are configured in `eval.passk_profiles`:
  - `pass1_strict`: `n=1`, `temperature=0.0`, `top_p=1.0`
  - `pass3`: `n=3`, `temperature=0.6`, `top_p=0.95`
