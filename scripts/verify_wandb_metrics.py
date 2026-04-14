from __future__ import annotations

"""Verify required W&B metrics exist for a run."""

import argparse
import re
from collections import defaultdict

import wandb


def _parse_splits(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Check required metric keys in a W&B run history.")
    ap.add_argument("--run", required=True, help="W&B run path: entity/project/run_id")
    ap.add_argument(
        "--expect-splits",
        default="gsm8k_test,svamp,math_500,amc_2023",
        help="Comma-separated eval splits expected in eval/<split>_n*/pass@1 keys",
    )
    ap.add_argument(
        "--require-step0-eval",
        action="store_true",
        help="Require each expected split to have pass@1 logged at step 0",
    )
    args = ap.parse_args()

    api = wandb.Api()
    run = api.run(args.run)

    keys_present: set[str] = set()
    first_step_by_key: dict[str, int] = {}
    eval_keys_by_split: dict[str, list[str]] = defaultdict(list)

    for row in run.scan_history():
        step = int(row.get("_step", -1))
        for k, v in row.items():
            if k == "_step":
                continue
            if v is None:
                continue
            keys_present.add(k)
            if k not in first_step_by_key:
                first_step_by_key[k] = step
            m = re.match(r"^eval/([^/]+)_n\d+/pass@1$", k)
            if m:
                eval_keys_by_split[m.group(1)].append(k)

    required_exact = [
        "train/reward",
        "train/reward_std",
        "train/kl",
        "train/entropy",
        "train/cppo/kept_fraction",
        "train/cppo/pruning_ratio",
        "train/kl_ref_floor",
        "train/kl_ref_ceiling",
        "train/clip_ratio/ref_min",
        "eval_timing/checkpoint_seconds",
        "eval_timing/checkpoint_cumulative_seconds",
    ]
    required_prefix = [
        "timing/train_loop_wall_seconds",
        "timing/checkpoint_eval_seconds",
        "timing/train_loop_estimated_train_only_seconds",
        "timing/total_eval_seconds",
    ]

    errors: list[str] = []

    for k in required_exact:
        if k not in keys_present:
            errors.append(f"Missing key: {k}")
    for k in required_prefix:
        if not any(x == k for x in keys_present):
            errors.append(f"Missing key: {k}")

    for split in _parse_splits(args.expect_splits):
        matched = eval_keys_by_split.get(split, [])
        if not matched:
            errors.append(f"Missing eval pass@1 key for split={split} (expected eval/{split}_n*/pass@1)")
            continue
        if args.require_step0_eval:
            if all(first_step_by_key.get(k, -1) != 0 for k in matched):
                errors.append(f"Missing step-0 eval for split={split}")

    if errors:
        print("[FAIL] W&B metric verification failed:")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)

    print("[PASS] W&B metric verification passed.")
    print(f"run={run.path}")
    print(f"metric_keys={len(keys_present)}")


if __name__ == "__main__":
    main()

