from __future__ import annotations

"""Compatibility launcher that delegates to `scripts/eval.py`."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "eval.py"), run_name="__main__")
