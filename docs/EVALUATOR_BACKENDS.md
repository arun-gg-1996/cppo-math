# Evaluator Backends

Evaluation is source-aware and routed by `eval.evaluator` in YAML (`configs/base.yaml`).
The same router is used in:

- checkpoint/manual eval (`scripts/eval.py`)
- training reward accuracy scoring (`src/cppo/train.py`)

## Routing

- `default_backend: auto`
  - tries `custom` for the split (if configured)
  - then tries `math_verify` (if enabled + installed + split is selected)
  - otherwise falls back to local `fallback_sympy`
- `backend_by_split` can force one backend per split.
- `custom_by_split` accepts `module:function` entries.

## Built-in backends

- `math_verify`: optional official-style symbolic verifier (`math_verify` package).
- `fallback_sympy`: local robust checker in `src/cppo/reward.py`.

Default `math_verify` split set:
`gsm8k`, `gsm8k_test`, `svamp`, `gsm_plus`, `asdiv`, `aime_2024`, `aime_2025`, `amc_2023`.
Other splits default to `fallback_sympy` unless you override config.

## Answer extraction order

`fallback_sympy` extraction in `src/cppo/reward.py` uses this order:

1. `<answer>...</answer>`
2. `\boxed{...}`
3. final-answer line patterns (for example `final answer is ...`)
4. fenced block (for example ```` ```python ... ``` ````)
5. last non-empty line fallback

This is why Qwen-style fenced outputs are handled.

## Notes

- If `math_verify` is not installed, the system logs one warning and safely uses `fallback_sympy`.
- `scripts/validate_evaluator.py` is the quick robustness check for local extracted answers.
- Per-checkpoint eval summary includes `evaluator_backend_counts` for auditability.
- If eval target is an adapter-only checkpoint, eval auto-merges adapter + base once into `_merged_eval_model` and evaluates that merged path.
