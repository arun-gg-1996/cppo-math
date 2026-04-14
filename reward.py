from __future__ import annotations

"""Compatibility re-export for reward helpers from `src/cppo/reward.py`."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cppo.reward import *  # noqa: F401,F403
