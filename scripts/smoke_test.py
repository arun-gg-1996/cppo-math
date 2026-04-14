from __future__ import annotations

"""Compatibility launcher that delegates to project-root `smoke_test.py`."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "smoke_test.py"), run_name="__main__")
