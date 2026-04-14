# Testing Runbook

## Setup

```bash
cd <repo_root>
bash setup_env.sh
source .venv/bin/activate
python data/download.py
```

## Pre-Train Evaluator Sanity Check (Full GSM8K)

Run this before training to validate extractor/evaluator behavior on the full GSM8K test set (1319 rows):

Terminal A (GPU1, vLLM server):

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/vllm_server.py --config config.yaml
```

Terminal B (GPU0, full pre-train eval):

```bash
python scripts/eval.py --config config.yaml \
  --checkpoint Qwen/Qwen2.5-1.5B-Instruct \
  --splits gsm8k_test \
  --n-generations 1 \
  --temperature 0.0 \
  --top-p 1.0 \
  --batch-size 256 \
  --out-dir runs/pretrain_eval/gsm8k_test_full
```

## Default Smoke Test (Split GPU, Silent)

Terminal A (GPU1, vLLM server):

```bash
screen -S cppo-vllm-smoke
cd <repo_root>
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python scripts/vllm_server.py --config configs/smoke/server_gpu_split_smoke.yaml --set rollout.vllm_gpu_memory_utilization=0.85
```

Detach with `Ctrl-a d`.

Terminal B (GPU0, quiet smoke train):

```bash
screen -S cppo-train-smoke
cd <repo_root>
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/smoke/server_gpu_split_smoke.yaml --set artifacts.save_completions=false --set training.logging_steps=10
```

Detach with `Ctrl-a d`.

Verify smoke outputs:

```bash
python scripts/verify_run.py --expect-eval
```

## Default Full Run (Split GPU, Silent)

Config defaults for `config.yaml`:
- base model is `Qwen/Qwen2.5-1.5B-Instruct` (CPPO GSM baseline)
- no TRL completion/reward debug table (`artifacts.save_completions=false`)
- checkpoint eval uses easy/mid splits only (boundary eval disabled)
- step-0 on-checkpoint eval runs at startup
- checkpoint `pass@1` is strict (`eval.temperature=0.0`, `eval.top_p=1.0`)
- periodic eval metrics appear as `eval/<split>_n<rows>/pass@1` (not `mid_eval/*`)
- eval batch size is `256`

Terminal A (GPU1, vLLM server, config default uses `0.85`):

```bash
screen -S cppo-vllm
cd <repo_root>
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python scripts/vllm_server.py --config config.yaml
```

Detach with `Ctrl-a d`.

Terminal B (GPU0, quiet full training):

```bash
screen -S cppo-train
cd <repo_root>
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config.yaml
```

Detach with `Ctrl-a d`.

Verify full run outputs:

```bash
python scripts/verify_run.py --expect-eval
cat runs/<run_id>/timing_summary.json
```

## Options (Only If Needed)

Check screens/logs:

```bash
screen -ls
screen -r cppo-vllm-smoke
screen -r cppo-train-smoke
screen -r cppo-vllm
screen -r cppo-train
```

Quick vLLM server test:

```bash
curl -sS http://127.0.0.1:8000/health/
curl -sS -X POST http://127.0.0.1:8000/generate/ -H "Content-Type: application/json" -d '{"prompts":["2+2 ="],"temperature":0.0,"max_tokens":8}'
```

Use `0.90` vLLM memory cap:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/vllm_server.py --config config.yaml --set rollout.vllm_gpu_memory_utilization=0.90
```

Manual eval spot check:

```bash
python scripts/eval.py --config config.yaml --checkpoint runs/<run_id>/checkpoints/checkpoint-100 --splits gsm8k_test,svamp,math_500,amc_2023
```

Local no-GPU checks:

```bash
python -m py_compile src/cppo/*.py scripts/*.py train.py eval.py smoke_test.py
python scripts/validate_evaluator.py --fail-below 0.98
python scripts/smoke_test.py --skip-trainer
```
