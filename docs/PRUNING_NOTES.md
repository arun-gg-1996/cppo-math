# CPPO Pruning Notes

This note captures practical concerns to track for current CPPO-style pruning runs.

## Current modes in code

- Default mode: `rollout.cppo.strategy=author_exact`
  - matches author-style allocation behavior used for strict reproducibility.
- Optional mode: `rollout.cppo.strategy=experimental_refill`
  - keeps the per-group + refill strategy as an experiment path.

If `rollout.cppo.allocation=false`, both modes run pruning-only CPPO.

## High-Priority Risks (Current)

1. `all-wrong / all-zero` groups:
   when a group has no reward signal, pruning does not create useful learning signal and can become effectively random/tie-driven.

2. dropping medium-signal completions:
   extreme `|advantage|` selection can remove boundary/mid-signal samples that may help calibration and stable learning.

3. hard-negative overweighting on easy prompts:
   if most completions are correct, the few wrong outliers can dominate `|A|` selection and bias training updates.

4. objective/eval mismatch:
   better optimization on pruned extremes does not always translate to better `pass@1` / `pass@k`.

## Data/Run Considerations

- Mitigation ideas to keep in mind:
  - run multiple seeds on the same data to reduce selection noise effects;
  - improve dataset cleanliness/coverage to reduce outlier-driven updates.

## Future Variant To Discuss

Potential pruning variant (not implemented yet):

- dynamic slot-based hybrid pruning:
  - split completions into `low / moderate / high` buckets (for non-binary rewards);
  - allocate integer slots dynamically by observed distribution, with minimum floors;
  - select within each bucket by `|advantage|`;
  - if slots remain, fill from global `|advantage|` ranking with deduplication.

- online bucket-stat updates:
  - use rewards from normal training rollouts (no extra inference-only passes required);
  - maintain running stats (for example EMA/quantiles) and adjust bucket boundaries over time.

- binary-reward version:
  - use a 2-bucket variant (`negative / positive`) and skip `moderate`;
  - keep the same slot + fallback-to-global-`|A|` design.

This is intentionally parked for later experimentation/discussion.
