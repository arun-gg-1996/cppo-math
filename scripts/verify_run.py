from __future__ import annotations

"""Sanity checker for run-folder completeness and key training metrics."""

import argparse
import json
import re
from pathlib import Path


def _find_latest_run(root: Path) -> Path:
    """Return most recently modified run under `runs/`."""
    runs = [p for p in root.glob("runs/*") if p.is_dir()]
    if not runs:
        raise RuntimeError("No runs found under runs/")
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _read_json(path: Path) -> dict:
    """Load JSON from disk."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _assert_file(path: Path, errors: list[str]) -> None:
    """Append error if required file is missing."""
    if not path.exists():
        errors.append(f"Missing file: {path}")


def _assert_any(paths: list[Path], label: str, errors: list[str]) -> None:
    """Append error when none of the expected artifacts exist."""
    if not any(p.exists() for p in paths):
        errors.append(f"Missing required artifact group: {label}")


def _checkpoint_sort_key(path: Path) -> tuple[int, float]:
    """Sort checkpoints by numeric step, then mtime as tie-breaker."""
    m = re.search(r"checkpoint-(\d+)$", path.name)
    step = int(m.group(1)) if m else -1
    try:
        mtime = float(path.stat().st_mtime)
    except Exception:
        mtime = 0.0
    return (step, mtime)


def main() -> None:
    """Validate one run directory and print pass/fail summary."""
    ap = argparse.ArgumentParser(description="Verify a training run folder has expected artifacts and key metrics.")
    ap.add_argument("--run-dir", default="", help="Run directory path. Defaults to latest under runs/.")
    ap.add_argument("--expect-eval", action="store_true", help="Require eval pass@k artifacts.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).resolve() if args.run_dir else _find_latest_run(project_root)
    checkpoints_root = run_dir / "checkpoints"

    errors: list[str] = []
    _assert_file(run_dir / "config.resolved.yaml", errors)
    _assert_file(run_dir / "run_manifest.json", errors)
    _assert_file(run_dir / "checkpoint_index.json", errors)
    if not checkpoints_root.exists():
        errors.append(f"Missing checkpoints directory: {checkpoints_root}")

    ckpts = sorted([p for p in checkpoints_root.glob("checkpoint-*") if p.is_dir()], key=_checkpoint_sort_key)
    if not ckpts:
        errors.append(f"No checkpoint-* directories found under {checkpoints_root}")
    else:
        latest = ckpts[-1]
        _assert_file(latest / "config.yaml", errors)
        _assert_file(latest / "checkpoint_meta.json", errors)
        _assert_file(latest / "trainer_state.json", errors)

        if (latest / "trainer_state.json").exists():
            state = _read_json(latest / "trainer_state.json")
            log_history = state.get("log_history", [])
            keys: set[str] = set()
            for row in log_history:
                if isinstance(row, dict):
                    keys.update(row.keys())

            required_metric_aliases = [
                ("cppo/pruning_ratio", "train/cppo/pruning_ratio"),
                ("cppo/kept_fraction", "train/cppo/kept_fraction"),
                ("cppo/allocation_enabled", "train/cppo/allocation_enabled"),
                ("cppo/author_exact_enabled", "train/cppo/author_exact_enabled"),
                ("reward", "train/reward"),
                ("reward_std", "train/reward_std"),
                ("kl", "train/kl"),
            ]
            for aliases in required_metric_aliases:
                if not any(k in keys for k in aliases):
                    errors.append(f"Missing metric in trainer_state log_history: one of {aliases}")

        if args.expect_eval:
            _assert_any(
                [
                    latest / "passk" / "summary.json",
                    latest / "passk" / "gsm8k_test" / "summary.json",
                ],
                "eval pass@k summary",
                errors,
            )

    _assert_file(checkpoints_root / "final" / "config.yaml", errors)

    if errors:
        print("[FAIL] Run verification failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("[PASS] Run verification passed.")
    print(f"run_dir={run_dir}")
    print(f"checkpoints={len(ckpts)}")


if __name__ == "__main__":
    main()
