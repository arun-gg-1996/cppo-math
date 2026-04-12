"""Build locked CPPO datasets from fixed Hugging Face sources/splits.

This script intentionally avoids the local `load_dataset(...)` path because some
environments hit an fsspec glob incompatibility. Instead, it pulls parquet file
URLs from the Hugging Face datasets-server API and reads them directly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

TRAIN_DATA_PATH = str(DATA_CLEAN_DIR / "train.jsonl")

GSM8K_TEST_PATH = str(DATA_CLEAN_DIR / "gsm8k_test.jsonl")
SVAMP_PATH = str(DATA_CLEAN_DIR / "svamp.jsonl")
MATH500_PATH = str(DATA_CLEAN_DIR / "math_500.jsonl")
AMC2023_PATH = str(DATA_CLEAN_DIR / "amc_2023.jsonl")

GSM_PLUS_PATH = str(DATA_CLEAN_DIR / "gsm_plus.jsonl")
ASDIV_PATH = str(DATA_CLEAN_DIR / "asdiv.jsonl")
AIME_2024_PATH = str(DATA_CLEAN_DIR / "aime_2024.jsonl")
AIME_2025_PATH = str(DATA_CLEAN_DIR / "aime_2025.jsonl")
MINERVA_PATH = str(DATA_CLEAN_DIR / "minerva_math.jsonl")
OLYMPIADBENCH_PATH = str(DATA_CLEAN_DIR / "olympiadbench.jsonl")

MANIFEST_PATH = str(DATA_CLEAN_DIR / "dataset_manifest.json")

HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("hf_token", ""))
PARQUET_API = "https://datasets-server.huggingface.co/parquet"


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    config: str
    splits: tuple[str, ...]
    source_tag: str
    out_path: str
    stage: str  # train | mid | boundary
    note: str = ""


TRAIN_SOURCE = SourceSpec(
    dataset="openai/gsm8k",
    config="main",
    splits=("train",),
    source_tag="gsm8k_train",
    out_path=TRAIN_DATA_PATH,
    stage="train",
    note="CPPO-aligned train source",
)

MID_EVAL_SOURCES = [
    SourceSpec("openai/gsm8k", "main", ("test",), "gsm8k_test", GSM8K_TEST_PATH, "mid"),
    # Standard SVAMP benchmark usage is the full challenge set (1000).
    SourceSpec("ChilleD/SVAMP", "default", ("train", "test"), "svamp", SVAMP_PATH, "mid"),
    SourceSpec("HuggingFaceH4/MATH-500", "default", ("test",), "math_500", MATH500_PATH, "mid"),
    SourceSpec("AI-MO/aimo-validation-amc", "default", ("train",), "amc_2023", AMC2023_PATH, "mid"),
]

BOUNDARY_EVAL_SOURCES = [
    SourceSpec("qintongli/GSM-Plus", "default", ("test",), "gsm_plus", GSM_PLUS_PATH, "boundary"),
    SourceSpec("EleutherAI/asdiv", "asdiv", ("validation",), "asdiv", ASDIV_PATH, "boundary"),
    SourceSpec("HuggingFaceH4/aime_2024", "default", ("train",), "aime_2024", AIME_2024_PATH, "boundary"),
    SourceSpec("math-ai/aime25", "default", ("test",), "aime_2025", AIME_2025_PATH, "boundary"),
    SourceSpec("knoveleng/Minerva-Math", "default", ("train",), "minerva_math", MINERVA_PATH, "boundary"),
    SourceSpec("realtreetune/olympiadbench", "default", ("test",), "olympiadbench", OLYMPIADBENCH_PATH, "boundary"),
]


def _first_nonempty_ci(row: dict[str, Any], keys: list[str]) -> Any:
    row_ci = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        v = row_ci.get(key.lower())
        if v not in (None, "", [], {}):
            return v
    return None


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, list):
        return _to_text(x[0]) if x else ""
    if isinstance(x, dict):
        for k in ("text", "content", "answer", "solution", "final_answer"):
            if k in x:
                return _to_text(x[k])
    return str(x).strip()


def _extract_boxed(text: str) -> str | None:
    m = re.search(r"\\boxed\s*\{", text)
    if not m:
        return None
    i = m.end()
    depth = 1
    out: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    if depth != 0:
        return None
    ans = "".join(out).strip()
    return ans or None


def _normalize_answer(raw: Any) -> str:
    s = _to_text(raw)
    if not s:
        return ""

    # GSM8K convention.
    if "####" in s:
        s = s.split("####")[-1].strip()

    boxed = _extract_boxed(s)
    if boxed:
        return boxed

    m = re.search(r"final answer is[:\s]*([^\n]+)", s, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().strip(".").strip("$")
        boxed = _extract_boxed(candidate)
        return boxed if boxed else candidate

    return s.strip()


def _build_question(row: dict[str, Any], source_tag: str) -> str:
    body = _to_text(_first_nonempty_ci(row, ["body"]))
    q = _to_text(
        _first_nonempty_ci(
            row,
            [
                "question",
                "problem",
                "prompt",
                "input",
                "query",
                "instruction",
                "problem_text",
                "statement",
                "content",
            ],
        )
    )

    if source_tag in {"svamp", "asdiv"} and body and q:
        if q.lower() not in body.lower():
            return f"{body}\n\n{q}"
        return body
    return q


def _infer_year(row: dict[str, Any], source_tag: str, question: str, pid: str) -> int | None:
    year = _first_nonempty_ci(row, ["year", "contest_year", "aime_year", "competition_year", "date_year"])
    if year is not None:
        try:
            yi = int(str(year))
            if 1900 <= yi <= 2100:
                return yi
        except Exception:
            pass
    hay = f"{source_tag} {question} {pid}"
    m = re.search(r"\b(19\d{2}|20\d{2})\b", hay)
    return int(m.group(1)) if m else None


def _normalize_row(row: dict[str, Any], source_tag: str, idx: int, split: str) -> dict[str, Any] | None:
    question = _build_question(row, source_tag)
    answer = _normalize_answer(
        _first_nonempty_ci(
            row,
            [
                "answer",
                "final_answer",
                "target",
                "label",
                "solution",
                "gold",
                "ground_truth",
                "expected_answer",
                "final",
            ],
        )
    )
    if not question or not answer:
        return None

    pid = _to_text(_first_nonempty_ci(row, ["id", "problem_id", "uuid", "qid", "index"]))
    if not pid:
        pid = f"{source_tag}_{split}_{idx:08d}"

    difficulty = (_to_text(_first_nonempty_ci(row, ["difficulty", "level", "type", "category"])) or "unknown").lower()
    raw_source = _to_text(
        _first_nonempty_ci(
            row,
            ["source", "dataset", "dataset_name", "origin", "competition", "contest", "subject"],
        )
    ).lower()
    year = _infer_year(row, source_tag, question, pid)

    return {
        "id": pid,
        "problem_id": pid,
        "question": question,
        "answer": answer,
        "source": source_tag if not raw_source else f"{source_tag}:{raw_source}",
        "source_tag": source_tag,
        "raw_source": raw_source,
        "difficulty": difficulty,
        "year": year,
    }


def _question_key(row: dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", row["question"]).strip().lower()


def _dedupe_by_question(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = _question_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _validate_schema(rows: list[dict[str, Any]]) -> None:
    required = {"id", "problem_id", "question", "answer", "source", "difficulty"}
    for i, row in enumerate(rows):
        missing = [k for k in required if k not in row or row[k] in (None, "")]
        if missing:
            raise ValueError(f"Schema validation failed at row {i}: missing {missing}")


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_repo_relative(path: str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def _hf_headers(token: str) -> dict[str, str]:
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _list_split_parquet_urls(spec: SourceSpec, token: str) -> dict[str, list[str]]:
    params = {"dataset": spec.dataset, "config": spec.config}
    resp = requests.get(PARQUET_API, params=params, headers=_hf_headers(token), timeout=90)
    resp.raise_for_status()
    payload = resp.json()

    split_to_urls: dict[str, list[str]] = {}
    for row in payload.get("parquet_files", []):
        split_to_urls.setdefault(str(row["split"]), []).append(str(row["url"]))

    for split in spec.splits:
        if split not in split_to_urls:
            available = ", ".join(sorted(split_to_urls.keys()))
            raise RuntimeError(
                f"{spec.dataset}[{spec.config}] missing split '{split}'. Available: {available}"
            )
    return split_to_urls


def _load_source_rows(spec: SourceSpec, token: str, max_rows_per_source: int) -> list[dict[str, Any]]:
    split_to_urls = _list_split_parquet_urls(spec, token)
    normalized: list[dict[str, Any]] = []
    running_idx = 0

    print(f"Loading: {spec.dataset} ({spec.config}) splits={list(spec.splits)} as source='{spec.source_tag}'")
    for split in spec.splits:
        urls = split_to_urls[split]
        for url in tqdm(urls, desc=f"parquet:{spec.source_tag}:{split}", unit="file"):
            frame = pd.read_parquet(url)
            rows = frame.to_dict(orient="records")
            for row in rows:
                nrow = _normalize_row(row, spec.source_tag, running_idx, split)
                running_idx += 1
                if nrow is None:
                    continue
                normalized.append(nrow)
                if max_rows_per_source > 0 and len(normalized) >= max_rows_per_source:
                    print(f"  -> capped at max_rows_per_source={max_rows_per_source}")
                    return normalized

    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Build locked CPPO train/eval JSONL files.")
    parser.add_argument("--hf-token", default=HF_TOKEN, help="Optional Hugging Face token for higher rate limits.")
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        default=0,
        help="Debug cap per source (0 means no cap).",
    )
    parser.add_argument(
        "--dedupe-by-question",
        action="store_true",
        help="Optional dedupe pass (off by default to preserve exact benchmark row counts).",
    )
    args = parser.parse_args()

    all_specs = [TRAIN_SOURCE] + MID_EVAL_SOURCES + BOUNDARY_EVAL_SOURCES
    counts_by_stage: dict[str, int] = {"train": 0, "mid": 0, "boundary": 0}
    output_counts: dict[str, int] = {}
    manifest_sources: list[dict[str, Any]] = []
    train_question_keys: set[str] = set()

    for spec in all_specs:
        rows = _load_source_rows(spec, token=(args.hf_token or ""), max_rows_per_source=args.max_rows_per_source)
        if args.dedupe_by_question:
            rows = _dedupe_by_question(rows)
        _validate_schema(rows)
        _write_jsonl(spec.out_path, rows)

        output_counts[spec.source_tag] = len(rows)
        counts_by_stage[spec.stage] += len(rows)
        manifest_sources.append(
            {
                "source_tag": spec.source_tag,
                "dataset": spec.dataset,
                "config": spec.config,
                "splits": list(spec.splits),
                "stage": spec.stage,
                "rows": len(rows),
                "out_path": _to_repo_relative(spec.out_path),
                "note": spec.note,
            }
        )
        if spec.stage == "train":
            train_question_keys = {_question_key(r) for r in rows}

    # Sanity check: train/eval overlap by normalized question.
    overlap_total = 0
    for spec in MID_EVAL_SOURCES + BOUNDARY_EVAL_SOURCES:
        p = Path(spec.out_path)
        overlap = 0
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if _question_key(row) in train_question_keys:
                    overlap += 1
        overlap_total += overlap
        if overlap > 0:
            print(f"WARNING: overlap with train for {spec.source_tag}: {overlap}")

    manifest = {
        "train_source": TRAIN_SOURCE.source_tag,
        "mid_eval_sources": [s.source_tag for s in MID_EVAL_SOURCES],
        "boundary_eval_sources": [s.source_tag for s in BOUNDARY_EVAL_SOURCES],
        "counts_by_stage": counts_by_stage,
        "outputs": manifest_sources,
        "train_eval_overlap_rows": overlap_total,
    }
    Path(MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(MANIFEST_PATH).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Train rows: {output_counts.get('gsm8k_train', 0)} -> {TRAIN_DATA_PATH}")
    print("\nMid eval rows:")
    for spec in MID_EVAL_SOURCES:
        print(f"  - {spec.source_tag}: {output_counts.get(spec.source_tag, 0)} -> {spec.out_path}")
    print("\nBoundary eval rows:")
    for spec in BOUNDARY_EVAL_SOURCES:
        print(f"  - {spec.source_tag}: {output_counts.get(spec.source_tag, 0)} -> {spec.out_path}")
    print(f"\nManifest: {MANIFEST_PATH}")
    print(f"Train/Eval overlap rows by question key: {overlap_total}")


if __name__ == "__main__":
    main()
