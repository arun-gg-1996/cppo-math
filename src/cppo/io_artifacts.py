from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def copy_resolved_config_to_checkpoint(resolved_config_path: Path, checkpoint_dir: Path) -> None:
    ensure_dir(checkpoint_dir)
    shutil.copy2(resolved_config_path, checkpoint_dir / "config.yaml")


def write_checkpoint_meta(
    checkpoint_dir: Path,
    *,
    run_id: str,
    global_step: int,
    save_reason: str,
    primary_metric: float | None,
) -> None:
    payload = {
        "run_id": run_id,
        "global_step": int(global_step),
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "save_reason": save_reason,
        "primary_metric": primary_metric,
    }
    write_json(checkpoint_dir / "checkpoint_meta.json", payload)


def _safe_symlink(target: Path, link_path: Path) -> None:
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        os.symlink(target.name, link_path)
    except OSError:
        # Fallback for environments where symlink is restricted.
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        if target.is_dir():
            shutil.copytree(target, link_path)
        else:
            shutil.copy2(target, link_path)


def update_latest_symlink(checkpoints_root: Path, checkpoint_dir: Path) -> None:
    _safe_symlink(checkpoint_dir, checkpoints_root / "latest")


def update_best_symlink(checkpoints_root: Path, best_checkpoint_dir: Path | None) -> None:
    if best_checkpoint_dir is None:
        return
    _safe_symlink(best_checkpoint_dir, checkpoints_root / "best")


def load_checkpoint_index(index_path: Path) -> list[dict[str, Any]]:
    return read_json(index_path, default=[])


def save_checkpoint_index(index_path: Path, rows: list[dict[str, Any]]) -> None:
    write_json(index_path, {"checkpoints": rows})


def load_checkpoint_rows(index_path: Path) -> list[dict[str, Any]]:
    blob = read_json(index_path, default={"checkpoints": []})
    if isinstance(blob, dict):
        rows = blob.get("checkpoints", [])
        if isinstance(rows, list):
            return rows
    return []


def append_checkpoint_row(index_path: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = load_checkpoint_rows(index_path)
    rows = [r for r in rows if int(r.get("global_step", -1)) != int(row.get("global_step", -2))]
    rows.append(row)
    rows.sort(key=lambda x: int(x.get("global_step", 0)))
    save_checkpoint_index(index_path, rows)
    return rows


def select_best_checkpoint(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None

    scored = [r for r in rows if r.get("primary_metric") is not None]
    if scored:
        scored.sort(key=lambda x: float(x["primary_metric"]), reverse=True)
        return scored[0]
    rows_sorted = sorted(rows, key=lambda x: int(x.get("global_step", 0)), reverse=True)
    return rows_sorted[0] if rows_sorted else None


def _remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if path.exists() and path.is_dir():
        shutil.rmtree(path)


def prune_checkpoints(
    *,
    checkpoints_root: Path,
    rows: list[dict[str, Any]],
    keep_best_k: int,
    keep_last_n: int,
    apply: bool,
) -> dict[str, list[str]]:
    rows_sorted = sorted(rows, key=lambda x: int(x.get("global_step", 0)))

    best_sorted = sorted(
        rows_sorted,
        key=lambda x: float(x.get("primary_metric", float("-inf"))),
        reverse=True,
    )
    keep_best = {str(r.get("checkpoint_dir")) for r in best_sorted[: max(0, keep_best_k)]}
    keep_last = {str(r.get("checkpoint_dir")) for r in rows_sorted[-max(0, keep_last_n) :]}

    keep = keep_best | keep_last

    protected = set()
    for link_name in ("best", "latest"):
        lp = checkpoints_root / link_name
        if lp.exists() or lp.is_symlink():
            try:
                protected.add(str(lp.resolve()))
            except Exception:
                pass
    keep |= protected

    deleted: list[str] = []
    kept: list[str] = []

    for row in rows_sorted:
        ckpt_dir = Path(str(row.get("checkpoint_dir", ""))).resolve()
        if str(ckpt_dir) in keep:
            kept.append(str(ckpt_dir))
            continue
        if apply:
            _remove_tree(ckpt_dir)
            deleted.append(str(ckpt_dir))
        else:
            deleted.append(str(ckpt_dir))

    return {"kept": kept, "deleted": deleted}
