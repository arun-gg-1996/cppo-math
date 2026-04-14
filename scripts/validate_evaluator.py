from __future__ import annotations

"""Validate extraction/checking robustness on local eval datasets."""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cppo.reward import check_answer, check_format_compliance  # noqa: E402


EVAL_FILES = {
    "gsm8k_test": "data/clean/gsm8k_test.jsonl",
    "svamp": "data/clean/svamp.jsonl",
    "math_500": "data/clean/math_500.jsonl",
    "amc_2023": "data/clean/amc_2023.jsonl",
    "gsm_plus": "data/clean/gsm_plus.jsonl",
    "asdiv": "data/clean/asdiv.jsonl",
    "aime_2024": "data/clean/aime_2024.jsonl",
    "aime_2025": "data/clean/aime_2025.jsonl",
    "minerva_math": "data/clean/minerva_math.jsonl",
    "olympiadbench": "data/clean/olympiadbench.jsonl",
}


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL rows from disk."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_none_answer(gt: str) -> bool:
    """Detect standardized 'no answer' labels."""
    s = (gt or "").strip().lower()
    return s in {"none", "no answer", "cannot be determined"}


def validate_split(name: str, rows: list[dict], limit: int = 0) -> dict:
    """Run deterministic extractor/checker probes for one split."""
    if limit > 0:
        rows = rows[:limit]
    n = len(rows)
    if n == 0:
        return {"split": name, "n": 0}

    tag_ok = 0
    boxed_ok = 0
    final_line_ok = 0
    none_rows = 0
    none_ok = 0
    format_ok = 0

    for row in rows:
        gt = str(row.get("answer", "")).strip()
        pred_tag = f"<think>reasoning</think>\n<answer>{gt}</answer>"
        pred_box = f"\\boxed{{{gt}}}"
        pred_line = f"The final answer is: {gt}."

        tag_ok += int(check_answer(pred_tag, gt) == 1.0)
        boxed_ok += int(check_answer(pred_box, gt) == 1.0)
        final_line_ok += int(check_answer(pred_line, gt) == 1.0)
        format_ok += int(check_format_compliance(pred_tag) == 1.0)

        if _is_none_answer(gt):
            none_rows += 1
            none_ok += int(check_answer("<think>x</think>\n<answer>None</answer>", gt) == 1.0)

    return {
        "split": name,
        "n": n,
        "tag_match_rate": tag_ok / n,
        "boxed_match_rate": boxed_ok / n,
        "final_line_match_rate": final_line_ok / n,
        "format_match_rate": format_ok / n,
        "none_rows": none_rows,
        "none_match_rate": (none_ok / none_rows) if none_rows else None,
    }


def main() -> None:
    """CLI entrypoint for evaluator robustness validation."""
    ap = argparse.ArgumentParser(description="Validate extractor/evaluator robustness on local eval files")
    ap.add_argument("--limit", type=int, default=0, help="Optional row cap per split")
    ap.add_argument(
        "--fail-below",
        type=float,
        default=0.98,
        help="Fail if tag or boxed match rate drops below this threshold",
    )
    args = ap.parse_args()

    failures: list[str] = []
    reports: list[dict] = []
    for split, rel in EVAL_FILES.items():
        p = (PROJECT_ROOT / rel).resolve()
        if not p.exists():
            print(f"[warn] missing split file: {split} -> {p}")
            continue
        rows = _read_jsonl(p)
        rep = validate_split(split, rows, limit=args.limit)
        reports.append(rep)
        print(
            f"[{split}] n={rep['n']} tag={rep['tag_match_rate']:.3f} "
            f"boxed={rep['boxed_match_rate']:.3f} final_line={rep['final_line_match_rate']:.3f} "
            f"format={rep['format_match_rate']:.3f} none_rows={rep['none_rows']}"
        )
        if rep["tag_match_rate"] < args.fail_below:
            failures.append(f"{split}: tag_match_rate={rep['tag_match_rate']:.4f}")
        if rep["boxed_match_rate"] < args.fail_below:
            failures.append(f"{split}: boxed_match_rate={rep['boxed_match_rate']:.4f}")
        if rep["none_rows"] and rep["none_match_rate"] is not None and rep["none_match_rate"] < args.fail_below:
            failures.append(f"{split}: none_match_rate={rep['none_match_rate']:.4f}")

    # Additional extraction sanity for code-fence style output.
    code_pred = "```python\nprint(72)\n```"
    code_ok = check_answer(code_pred, "72") == 1.0
    print(f"[code_fence_case] match={code_ok}")
    if not code_ok:
        failures.append("code_fence_case: expected True")

    out = (PROJECT_ROOT / "data" / "clean" / "evaluator_validation_report.json").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"reports": reports, "failures": failures}, f, ensure_ascii=False, indent=2)
    print(f"[done] report: {out}")

    if failures:
        print("\n[FAIL]")
        for x in failures:
            print(f" - {x}")
        raise SystemExit(1)

    print("\n[PASS] Evaluator looks robust for the configured thresholds.")


if __name__ == "__main__":
    main()
