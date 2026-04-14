# CPPO Training Repo

YAML-first CPPO/GRPO training with reproducible checkpoint artifacts.

## Start Here

- High-level code walkthrough: `docs/CODE_OVERVIEW.md`
- Run folder artifacts: `docs/RUN_LAYOUT.md`
- Training metrics dictionary: `docs/TRAIN_METRICS.md`
- Pruning notes/risks: `docs/PRUNING_NOTES.md`

## Setup

```bash
cd <repo_root>
bash setup_env.sh
source .venv/bin/activate
python data/download.py
```

`setup_env.sh` prompts for `WANDB_API_KEY` and `HF_TOKEN` (hidden input), writes/updates `.env`, and runs `wandb login` + `hf auth login`.

## Default Run (Split GPU, Silent)

This is the default path:
- `GPU0`: policy training
- `GPU1`: vLLM server
- base model: `Qwen/Qwen2.5-1.5B-Instruct` (CPPO GSM baseline)
- no TRL completion/reward debug table (`artifacts.save_completions=false` in `config.yaml`)
- checkpoint eval uses easy/mid splits only (boundary eval disabled)
- step-0 on-checkpoint eval runs at startup
- checkpoint `pass@1` is strict (`eval.temperature=0.0`, `eval.top_p=1.0`)
- periodic eval metrics appear as `eval/<split>_n<rows>/pass@1` (not `mid_eval/*`)
- eval batch size is `256`
- main config vLLM memory cap: `rollout.vllm_gpu_memory_utilization=0.85`

Terminal A (GPU1, vLLM server):

```bash
screen -S cppo-vllm
cd <repo_root>
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python scripts/vllm_server.py --config config.yaml
```

Detach with `Ctrl-a d`.

Terminal B (GPU0, quiet training):

```bash
screen -S cppo-train
cd <repo_root>
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config.yaml
```

Detach with `Ctrl-a d`.

## Verify Run

```bash
python scripts/verify_run.py --expect-eval
```

Re-attach logs if needed:

```bash
screen -ls
screen -r cppo-vllm
screen -r cppo-train
```

## Options (Only If Needed)

Quick vLLM server test:

```bash
curl -sS http://127.0.0.1:8000/health/
curl -sS -X POST http://127.0.0.1:8000/generate/ -H "Content-Type: application/json" -d '{"prompts":["2+2 ="],"temperature":0.0,"max_tokens":8}'
```

Use `0.90` vLLM memory cap instead of `0.85`:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/vllm_server.py --config config.yaml --set rollout.vllm_gpu_memory_utilization=0.90
```

Split-GPU smoke run (quiet):

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/vllm_server.py --config configs/smoke/server_gpu_split_smoke.yaml --set rollout.vllm_gpu_memory_utilization=0.85
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/smoke/server_gpu_split_smoke.yaml --set artifacts.save_completions=false
```

Manual eval on a checkpoint:

```bash
python scripts/eval.py --config config.yaml --checkpoint runs/<run_id>/checkpoints/checkpoint-400 --splits aime_2024
```

Alternative training configs:

```bash
python scripts/train.py --config configs/cppo/cppo_g16_p0875.yaml
python scripts/train.py --config configs/cppo/grpo_g16_baseline.yaml
python scripts/train.py --config configs/local/local_cpu_smoke.yaml
```
