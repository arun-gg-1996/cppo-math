"""Deprecated compatibility module.

Use YAML config files under `configs/` and run via `python scripts/train.py --config ...`.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
