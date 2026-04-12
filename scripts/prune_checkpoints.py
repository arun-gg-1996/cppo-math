from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cppo.io_artifacts import (
    load_checkpoint_rows,
    prune_checkpoints,
    save_checkpoint_index,
    select_best_checkpoint,
    update_best_symlink,
    update_latest_symlink,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune checkpoint directories by keep-best-k and keep-last-n."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory under runs/<run_id>")
    parser.add_argument("--keep-best-k", type=int, default=3)
    parser.add_argument("--keep-last-n", type=int, default=2)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete checkpoints. Without --apply this command only previews what would be deleted.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    index_path = run_dir / "checkpoint_index.json"
    checkpoints_root = run_dir / "checkpoints"

    rows = load_checkpoint_rows(index_path)
    if not rows:
        raise RuntimeError(f"No checkpoint rows found in {index_path}")

    result = prune_checkpoints(
        checkpoints_root=checkpoints_root,
        rows=rows,
        keep_best_k=int(args.keep_best_k),
        keep_last_n=int(args.keep_last_n),
        apply=bool(args.apply),
    )

    mode = "APPLY" if args.apply else "PREVIEW"
    print(f"[{mode}] keep_best_k={args.keep_best_k} keep_last_n={args.keep_last_n}")
    print(f"[{mode}] kept={len(result['kept'])} delete={len(result['deleted'])}")

    for p in result["deleted"]:
        print(f"  delete: {p}")

    if args.apply:
        kept_set = set(result["kept"])
        remaining_rows = [
            r for r in rows if str(Path(str(r.get("checkpoint_dir", ""))).resolve()) in kept_set
        ]
        save_checkpoint_index(index_path, remaining_rows)

        if remaining_rows:
            latest_row = sorted(remaining_rows, key=lambda x: int(x.get("global_step", 0)))[-1]
            latest_dir = Path(str(latest_row["checkpoint_dir"]))
            update_latest_symlink(checkpoints_root, latest_dir)

            best_row = select_best_checkpoint(remaining_rows)
            best_dir = Path(str(best_row["checkpoint_dir"])) if best_row else None
            update_best_symlink(checkpoints_root, best_dir)


if __name__ == "__main__":
    main()
