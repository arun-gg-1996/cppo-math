# Training Metrics (CPPO Math)

These are the custom metrics now logged from the training reward path, aligned with the code-reasoning style where relevant for this dataset.

## CPPO-specific metrics

- `cppo/pruning_ratio`
- `cppo/kept_per_group`
- `cppo/kept_fraction`
- `cppo/allocation_enabled`
- `cppo/presample_factor`
- `cppo/filled_fraction`
- `cppo/abs_adv_mean_kept`
- `cppo/abs_adv_mean_dropped`
- `cppo/author_exact_enabled`

## Reward / GRPO signal

- `reward/mean`
- `reward/std`
- `reward/non_zero_fraction`
- `exec/zero_fraction`
- `grpo/reward_std_mean`
- `grpo/all_zero_fraction`
- `grpo/all_perfect_fraction`

Reference overlays:

- `kl_ref_floor`
- `kl_ref_ceiling`
- `clip_ratio/ref_min`
- `grpo/reward_std_ref_floor`
- `grpo/all_zero_ref_max`
- `grpo/all_perfect_ref_max`

Warning flags:

- `grpo/warn_reward_std_collapse`
- `grpo/warn_all_zero_collapse`
- `warn/problems_too_easy`

## Generation quality

- `gen/empty_completion_fraction`
- `gen/has_reasoning_fraction`
- `gen/mean_completion_chars`

## Difficulty / source slices (when present)

- `reward/mean_<difficulty>`
- `reward/<source>_mean`

Where `<difficulty>` and `<source>` come from dataset fields after key normalization.

## Data coverage (cumulative)

- `data/unique_problems_seen`
- `data/processed_total`
- `data/<source>_seen`
- `data/<source>_processed`
- `data/<difficulty>_seen`
- `data/<difficulty>_processed`

Notes:

- `data/*` counters are cumulative and logged as latest values.
- non-`data/*` metrics are aggregated from the short reward buffer in callback logging.
- the callback flushes this buffer after each log step, then starts a fresh window.
