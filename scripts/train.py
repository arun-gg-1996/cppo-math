from __future__ import annotations

"""Thin CLI wrapper that launches `cppo.train.main` from YAML config."""

import argparse
import sys
from pathlib import Path


def main() -> None:
    """Parse CLI args, wire `src/` import path, and start training."""
    parser = argparse.ArgumentParser(description="CPPO/GRPO training entrypoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override dotted.key=value (repeatable)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from cppo.train import main as train_main

    train_main(config_path=args.config, overrides=args.overrides)


if __name__ == "__main__":
    main()
