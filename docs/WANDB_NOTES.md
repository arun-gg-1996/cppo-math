# W&B Notes (Reference Lines)

## Why reference lines showed up in separate charts before

Two reasons:

1. Key prefixing mismatch:
   - Trainer/W&B integration rewrites training keys as `train/<key>`.
   - If code logs keys that already start with `train/`, they become `train/train/...`.

2. W&B auto workspace behavior:
   - Auto panels are created per metric key.
   - Overlay requires explicitly adding multiple keys to the same custom panel.

## Current fix in this repo

Reference keys are now logged as base keys (no leading `train/`), e.g.:

- `kl_ref_floor`
- `kl_ref_ceiling`
- `clip_ratio/ref_min`

So W&B rewrites them to:

- `train/kl_ref_floor`
- `train/kl_ref_ceiling`
- `train/clip_ratio/ref_min`

which aligns with:

- `train/kl`
- `train/clip_ratio/*`

`RewardStatsCallback` also logs the same aggregate payload directly with:

- `wandb.log(payload, step=global_step)`

so custom reward/data keys are stable in W&B even when trainer log timing differs.

## Recommended panel overlays

- KL panel: `train/kl`, `train/kl_ref_floor`, `train/kl_ref_ceiling`
- Clip panel: `train/clip_ratio/low_mean` (or your preferred clip metric), `train/clip_ratio/ref_min`
- GRPO signal panel: `train/grpo/reward_std_mean`, `train/grpo/reward_std_ref_floor`, `train/grpo/all_zero_fraction`, `train/grpo/all_zero_ref_max`

## Eval metric naming

Checkpoint eval metrics are logged as:

- `eval/<split>_n<rows>/pass@1`
- `eval/<split>_n<rows>/pass@k` (if enabled)

Example: `eval/gsm8k_test_n1319/pass@1`

With default dual-profile eval, this is typically:

- `pass@1` from strict profile (`n=1`, `temperature=0.0`, `top_p=1.0`)
- `pass@3` from best-of-3 profile (`n=3`, `temperature=0.6`, `top_p=0.95`)

This keeps splits separated and makes the row count visible in graph names.
