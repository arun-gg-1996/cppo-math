# CPPO Training Repo

YAML-first CPPO/GRPO training with reproducible checkpoint artifacts.

## Start Here

- High-level code walkthrough: `docs/CODE_OVERVIEW.md`
- Run folder artifacts: `docs/RUN_LAYOUT.md`
- Training metrics dictionary: `docs/TRAIN_METRICS.md`
- Pruning notes/risks: `docs/PRUNING_NOTES.md`

## Script Guide (Simple)

- `scripts/train.py`: starts training from a YAML config.
- `scripts/vllm_server.py`: starts the rollout server (vLLM) from the same YAML config.
- `scripts/eval.py`: evaluates one checkpoint on one or more eval datasets.
- `scripts/verify_run.py`: checks that a run folder is healthy (required files exist, checkpoints are saved, and key CPPO metrics are present in trainer logs).
- `scripts/prune_checkpoints.py`: removes old checkpoints based on your keep rules (`keep-best-k`, `keep-last-n`).
- `scripts/smoke_test.py`: quick deterministic checks for pruning logic, reward/extractor behavior, and basic wiring.

## Folder Layout

```text
.
├── config.yaml
├── configs/
│   ├── base.yaml
│   └── cppo/
│       ├── cppo_g16_p0875.yaml
│       ├── cppo_g16_p09375.yaml
│       └── grpo_g16_baseline.yaml
├── scripts/
│   ├── train.py
│   ├── vllm_server.py
│   ├── eval.py
│   ├── verify_run.py
│   ├── prune_checkpoints.py
│   └── smoke_test.py
├── src/cppo/
│   ├── train.py
│   ├── trainer_cppo.py
│   ├── callbacks.py
│   ├── io_artifacts.py
│   ├── config_loader.py
│   ├── eval.py
│   ├── evaluator_registry.py
│   └── reward.py
├── runs/
└── docs/
```

## Environment Setup

```bash
cd <repo_root>
bash setup_env.sh
source .venv/bin/activate
```

`setup_env.sh` does the full setup flow:
- creates/updates `.venv`
- installs `requirements.txt`
- prompts for required `WANDB_API_KEY` and `HF_TOKEN` (hidden input)
- writes/updates those keys in `.env` while preserving unrelated keys
- runs `wandb login` and `hf auth login`

Build datasets before smoke/full training:

```bash
python data/download.py
```

`.env` is required by default. Training fails fast if:
- `.env` is missing, or
- `integrations.wandb.enabled=true` but W&B token env var is missing, or
- `integrations.hf_hub.enabled=true` (or `push_to_hub=true`) but HF token env var is missing.
- `push_to_hub=true` but `integrations.hf_hub.hub_model_id` is empty.

`.env.example` is kept as a reference template; setup does not require copying it.

Default integration profile:
- W&B enabled
- HF Hub enabled
- `push_to_hub=true`

## Train

Default config:

```bash
python scripts/train.py --config config.yaml
```

For strict 2xA100 split mode (author-style): run vLLM on one GPU and training on the other.

`rollout.vllm_gpu_memory_utilization` is a GPU memory fraction cap for vLLM (not a compute utilization target).
Defaults:
- smoke config: `0.40`
- main config: `0.50`
On a dedicated GPU1 (no training process there), `0.80` to `0.90` is typically fine.

Terminal A (rollout server on GPU 1):

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/vllm_server.py --config config.yaml
```

Quick server test from another terminal:

```bash
curl -sS http://127.0.0.1:8000/health/
curl -sS -X POST http://127.0.0.1:8000/generate/ -H "Content-Type: application/json" -d '{"prompts":["2+2 ="],"temperature":0.0,"max_tokens":8}'
```

Higher-memory server launch example (dedicated GPU1):

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/vllm_server.py --config config.yaml --set rollout.vllm_gpu_memory_utilization=0.85
```

Terminal B (policy training on GPU 0):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config.yaml
```

Quiet full split-GPU training (suppresses TRL completion/reward debug table):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config.yaml --set artifacts.save_completions=false --set training.logging_steps=10
```

Using `screen` (recommended on remote servers):

Start vLLM service in one screen session:

```bash
screen -S cppo-vllm
cd <repo_root>
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python scripts/vllm_server.py --config config.yaml
```

Detach from session: `Ctrl-a d`

Start training in another screen session:

```bash
screen -S cppo-train
cd <repo_root>
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config.yaml
```

Detach from session: `Ctrl-a d`

Re-attach later:

```bash
screen -ls
screen -r cppo-vllm
screen -r cppo-train
```

Paper-aligned CPPO (`G=16`, `pruning=0.875`):

```bash
python scripts/train.py --config configs/cppo/cppo_g16_p0875.yaml
```

CPPO strategy modes:
- `rollout.cppo.strategy=author_exact` (default): global `|A|` pruning with author-style allocation behavior.
- `rollout.cppo.strategy=experimental_refill`: your previous per-group keep + refill allocation variant.

Run experimental refill variant explicitly:

```bash
python scripts/train.py --config config.yaml --set rollout.cppo.strategy=experimental_refill
```

GRPO baseline (`G=16`, CPPO off):

```bash
python scripts/train.py --config configs/cppo/grpo_g16_baseline.yaml
```

Local CPU smoke (tiny model, no vLLM/eval):

```bash
python scripts/train.py --config configs/local/local_cpu_smoke.yaml
```

Server GPU smoke (tiny model, CPPO + vLLM + lightweight eval):

```bash
python scripts/train.py --config configs/smoke/server_gpu_smoke.yaml
```

Server split-GPU smoke (requires vLLM server terminal):

Terminal A:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/vllm_server.py --config configs/smoke/server_gpu_split_smoke.yaml
```

Higher-memory smoke server launch example:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/vllm_server.py --config configs/smoke/server_gpu_split_smoke.yaml --set rollout.vllm_gpu_memory_utilization=0.85
```

Terminal B:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/smoke/server_gpu_split_smoke.yaml
```

Quiet split-GPU smoke (suppresses TRL completion/reward debug table):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/smoke/server_gpu_split_smoke.yaml --set artifacts.save_completions=false --set training.logging_steps=10
```

Override any field with dotted keys:

```bash
python scripts/train.py --config config.yaml --set rollout.cppo.allocation=false --set model.attn_implementation=sdpa
```

## What Gets Saved Per Checkpoint

Inside each `runs/<run_id>/checkpoints/checkpoint-<step>/`:

- `config.yaml` (resolved run config snapshot)
- `checkpoint_meta.json` (step, timestamp, primary metric)
- `completions/step_<step>.jsonl` (local completion samples)
- `passk/<split>/summary.json` and `passk/<split>/details.jsonl`
- `passk/summary.json` aggregate

At checkpoint root:

- `best` symlink
- `latest` symlink

At run root:

- `checkpoint_index.json` (all checkpoint metrics/paths)
- `last_prune.json` (if automatic retention is enabled)

HF sync note:
- when `integrations.hf_hub.push_to_hub=true`, checkpoints are pushed once by Trainer and then re-synced after callback artifacts are written.
- this ensures Hub checkpoints include `config.yaml`, pass@k outputs, and checkpoint metadata written in callback hooks.

## Evaluate a Checkpoint Manually

```bash
python scripts/eval.py --config config.yaml --checkpoint runs/<run_id>/checkpoints/checkpoint-400 --splits aime_2024
```

If the checkpoint is adapter-only (LoRA files without full model weights), eval now auto-merges it once into:

- `runs/<run_id>/checkpoints/checkpoint-<step>/_merged_eval_model`

and evaluates that merged model path automatically.

By default this temporary merged folder is cleaned up after eval (`eval.adapter_merge.cleanup_after_eval=true`).
Override for a one-off manual eval:

```bash
python scripts/eval.py --config config.yaml --checkpoint runs/<run_id>/checkpoints/checkpoint-400 --splits aime_2024 --cleanup-merged-model false
```

Default pass@k reporting now uses two separate decode profiles:

- strict `pass@1`: `n=1`, `temperature=0.0`, `top_p=1.0`
- best-of-3 `pass@3`: `n=3`, `temperature=0.6`, `top_p=0.95`

These run separately and then merge into one split summary.

If you pass manual decode flags (`--n-generations`, `--temperature`, or `--top-p`), script eval switches to single-profile mode for that call.

Default on-checkpoint suite is configured as:
- `gsm8k_test`
- `svamp`
- `math_500`
- `amc_2023`

Boundary suite (`gsm_plus`, `asdiv`, `aime_2024`, `aime_2025`, `minerva_math`, `olympiadbench`) runs:
- once at start on the base model
- once at end on the final checkpoint

Mid-eval collision guard:
- periodic mid-run eval should use `eval.on_checkpoint` (this logs to W&B and writes local pass@k artifacts).
- `eval.mid_eval` is an optional quick-local debug path and is disabled by default.
- if quick-local mid-eval is enabled and collides with checkpoint/boundary steps, those steps are skipped to avoid double evaluation.
- in split server mode, automatic checkpoint/boundary eval uses the configured vLLM server endpoint (`/generate/`) instead of starting a second local vLLM engine on GPU0.

If these JSONL files are missing under `data/clean/`, build them with:

```bash
python data/download.py
```

Validate evaluator robustness against local split answers:

```bash
python scripts/validate_evaluator.py
```

Verify latest run artifacts + key CPPO metrics:

```bash
python scripts/verify_run.py
```

What this command checks in simple terms:
- run folder exists and has core files (`config.resolved.yaml`, `run_manifest.json`, `checkpoint_index.json`)
- at least one `checkpoint-*` exists
- checkpoint has core files (`config.yaml`, `checkpoint_meta.json`, `trainer_state.json`)
- trainer log contains key CPPO metrics (like `cppo/pruning_ratio`, `cppo/kept_fraction`, `cppo/author_exact_enabled`)
- if you pass `--expect-eval`, it also checks eval summary files exist

Evaluator backend routing is configured in `eval.evaluator`:
- `default_backend: auto` tries official backend first (currently `math_verify` when installed), then falls back to local SymPy checker.
- `backend_by_split` can force a backend per split.
- `custom_by_split` supports `module:function` hooks for source-specific official checkers.
- Per-checkpoint eval summaries now include `evaluator_backend_counts`.
- Training reward accuracy uses the same router (per-source) for consistency.
- W&B now logs per-split eval metrics with dataset-size-aware keys, e.g.:
  - `eval/gsm8k_test_n1319/pass@1`
  - `eval/gsm8k_test_n1319/pass@3`
  - `eval_boundary/start/aime_2024_n30/pass@1`
  - `eval_boundary/start/aime_2024_n30/pass@3`

## Manual Checkpoint Pruning

Preview only (no deletion):

```bash
python scripts/prune_checkpoints.py --run-dir runs/<run_id> --keep-best-k 3 --keep-last-n 2
```

Actually delete:

```bash
python scripts/prune_checkpoints.py --run-dir runs/<run_id> --keep-best-k 3 --keep-last-n 2 --apply
```

`keep-best-k` keeps top-k by primary metric, `keep-last-n` keeps newest n checkpoints.

## Notes

- CPPO pruning caveats and future variant ideas: `docs/PRUNING_NOTES.md`
- Training metric reference: `docs/TRAIN_METRICS.md`
- W&B reference-line behavior and overlays: `docs/WANDB_NOTES.md`
- Evaluator backend routing: `docs/EVALUATOR_BACKENDS.md`
