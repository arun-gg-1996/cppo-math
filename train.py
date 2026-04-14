from __future__ import annotations

"""Compatibility launcher that delegates to `scripts/train.py`."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "train.py"), run_name="__main__")
