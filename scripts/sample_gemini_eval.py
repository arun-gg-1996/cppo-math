from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_ROOT))

from cppo.config_loader import load_config  # noqa: E402
from cppo.evaluator_registry import EvaluatorRegistry  # noqa: E402
from cppo.reward import (  # noqa: E402
    check_answer,
    check_format_compliance,
    extract_answer_tag,
    extract_boxed,
    extract_code_fence,
    extract_final_answer_line,
    extract_prediction_answer,
)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    path: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _select_indices(n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    if k == 1:
        return [0]
    idxs = sorted(set(int(round(i * (n - 1) / (k - 1))) for i in range(k)))
    return idxs


def _load_api_key() -> str | None:
    for key in ("GEMINI_API_KEY", "gemini_api_key", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        val = os.environ.get(key, "").strip()
        if val:
            return val

    # Fallback to parent repo .env for convenience.
    for env_path in [PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"]:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            if k.strip().lower() in {"gemini_api_key", "google_api_key", "google_genai_api_key"}:
                return v.strip().strip('"').strip("'")
    return None


def _extract_text_from_gemini_response(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates", [])
    if not candidates:
        return ""
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    texts = [str(p.get("text", "")) for p in parts if isinstance(p, dict)]
    return "\n".join(t for t in texts if t).strip()


def _extract_finish_reason(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates", [])
    if not candidates:
        return ""
    reason = candidates[0].get("finishReason", "")
    return str(reason) if reason is not None else ""


def _extractor_branch(text: str) -> str:
    if extract_answer_tag(text):
        return "answer_tag"
    if extract_boxed(text):
        return "boxed"
    if extract_final_answer_line(text):
        return "final_answer_line"
    if extract_code_fence(text):
        return "code_fence"
    return "last_line_fallback"


def _classify_failure(
    *,
    format_ok: bool,
    routed_correct: bool,
    local_correct: bool,
    extracted: str | None,
    finish_reason: str,
) -> tuple[str, str]:
    if format_ok and routed_correct:
        return "ok", "none"

    if finish_reason.upper() == "MAX_TOKENS":
        return "truncated_output", "gemini_output"
    if extracted is None:
        return "no_extractable_answer", "gemini_output"
    if not format_ok:
        return "format_noncompliant", "gemini_output"

    # Format is good but routed evaluator disagrees with local checker.
    if (not routed_correct) and local_correct:
        return "evaluator_backend_mismatch", "pipeline_evaluator"

    return "wrong_answer_after_extraction", "gemini_output"


def _call_gemini(
    model: str,
    api_key: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    request_timeout_s: float,
) -> tuple[int, dict[str, Any]]:
    # Match parent project judge wiring: Vertex AI endpoint with API key auth.
    url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent?key={api_key}"
    body = {
        "system_instruction": {
            "parts": [
                {
                    "text": "You are a precise math reasoning assistant. Follow required output format exactly."
                }
            ]
        },
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": float(temperature), "maxOutputTokens": int(max_tokens)},
    }
    resp = requests.post(url, json=body, timeout=float(request_timeout_s))
    try:
        data = resp.json()
    except Exception:
        data = {"raw_text": resp.text}
    return resp.status_code, data


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample eval rows from each source, query Gemini, and score extraction/verification")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default="gemini-2.0-flash")
    ap.add_argument("--samples-per-split", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-output-tokens", type=int, default=None)
    ap.add_argument("--sleep-s", type=float, default=0.25)
    ap.add_argument("--request-timeout-s", type=float, default=45.0)
    ap.add_argument("--out", default="data/clean/gemini_sample_eval_report.json")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(
        args.config,
        overrides=args.overrides
        + [
            "env.require_dotenv_file=false",
            "integrations.wandb.enabled=false",
            "integrations.hf_hub.enabled=false",
            "integrations.hf_hub.push_to_hub=false",
        ],
    )
    split_map: dict[str, str] = cfg.get("data", {}).get("eval_splits", {})
    prompt_cfg = cfg.get("prompt", {})
    default_prompt = str(prompt_cfg.get("system_prompt", "")).strip()
    per_split_prompt: dict[str, str] = prompt_cfg.get("eval_system_prompt_by_split", {})
    eval_cfg = cfg.get("eval", {})
    evaluator = EvaluatorRegistry(eval_cfg.get("evaluator", {}))
    max_output_tokens = int(
        args.max_output_tokens if args.max_output_tokens is not None else eval_cfg.get("max_new_tokens", 1024)
    )

    key = _load_api_key()
    if not key:
        raise RuntimeError("No Gemini API key found in env or .env (expected GEMINI_API_KEY/gemini_api_key).")

    split_specs: list[SplitSpec] = [SplitSpec(name=k, path=v) for k, v in split_map.items()]
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    out_path = (PROJECT_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_partial() -> None:
        by_split: dict[str, dict[str, float]] = {}
        failure_by_owner: dict[str, int] = {}
        failure_by_reason: dict[str, int] = {}
        for rec in records:
            s = rec["split"]
            d = by_split.setdefault(s, {"n": 0.0, "format_ok": 0.0, "correct": 0.0, "local_correct": 0.0})
            d["n"] += 1.0
            d["format_ok"] += 1.0 if rec["format_compliant"] else 0.0
            d["correct"] += 1.0 if rec["correct"] else 0.0
            d["local_correct"] += 1.0 if rec.get("local_correct") else 0.0
            if not rec["correct"]:
                owner = str(rec.get("failure_owner", "unknown"))
                reason = str(rec.get("failure_reason", "unknown"))
                failure_by_owner[owner] = failure_by_owner.get(owner, 0) + 1
                failure_by_reason[reason] = failure_by_reason.get(reason, 0) + 1

        summary = {
            k: {
                "n": int(v["n"]),
                "format_rate": (v["format_ok"] / v["n"]) if v["n"] else 0.0,
                "correct_rate": (v["correct"] / v["n"]) if v["n"] else 0.0,
                "local_correct_rate": (v["local_correct"] / v["n"]) if v["n"] else 0.0,
            }
            for k, v in by_split.items()
        }
        out_path.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "samples_per_split": args.samples_per_split,
                    "summary": summary,
                    "failure_by_owner": failure_by_owner,
                    "failure_by_reason": failure_by_reason,
                    "records": records,
                    "errors": errors,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    for spec in split_specs:
        p = Path(str(spec.path))
        if not p.exists():
            errors.append({"split": spec.name, "error": f"missing file: {p}"})
            continue
        rows = _read_jsonl(p)
        indices = _select_indices(len(rows), int(args.samples_per_split))
        split_prompt = str(per_split_prompt.get(spec.name, default_prompt)).strip()
        for idx in indices:
            row = rows[idx]
            q = str(row.get("question", "")).strip()
            gt = str(row.get("answer", "")).strip()
            user_prompt = (
                f"{split_prompt}\n\n"
                "Solve the following problem. Keep the required output format exactly.\n\n"
                f"Problem:\n{q}"
            )
            status, data = _call_gemini(
                model=args.model,
                api_key=key,
                prompt=user_prompt,
                temperature=float(args.temperature),
                max_tokens=max_output_tokens,
                request_timeout_s=float(args.request_timeout_s),
            )
            if status != 200:
                errors.append(
                    {
                        "split": spec.name,
                        "index": idx,
                        "status": status,
                        "error": data,
                    }
                )
                # Stop early if auth/project config is broken.
                if status in {401, 403}:
                    _write_partial()
                    raise RuntimeError(
                        f"Gemini call failed with {status}. See {out_path} for details. "
                        "If 403, enable Vertex AI API and verify key/project permissions."
                    )
                _write_partial()
                continue

            raw = _extract_text_from_gemini_response(data)
            finish_reason = _extract_finish_reason(data)
            extracted = extract_prediction_answer(raw)
            extractor_branch = _extractor_branch(raw)
            fmt = check_format_compliance(raw)
            eval_res = evaluator.score(
                split_name=spec.name,
                predicted_text=raw,
                ground_truth=gt,
                row=row,
            )
            routed_score = float(eval_res.score)
            local_score = float(check_answer(raw, gt))
            routed_correct = bool(routed_score == 1.0)
            local_correct = bool(local_score == 1.0)
            failure_reason, failure_owner = _classify_failure(
                format_ok=bool(fmt == 1.0),
                routed_correct=routed_correct,
                local_correct=local_correct,
                extracted=extracted,
                finish_reason=finish_reason,
            )

            records.append(
                {
                    "split": spec.name,
                    "index": idx,
                    "id": row.get("id", row.get("problem_id", "")),
                    "ground_truth": gt,
                    "model_output": raw,
                    "finish_reason": finish_reason,
                    "output_chars": len(raw),
                    "extracted_answer": extracted,
                    "extractor_branch": extractor_branch,
                    "evaluator_backend": eval_res.backend,
                    "format_compliant": bool(fmt == 1.0),
                    "correct": routed_correct,
                    "local_correct": local_correct,
                    "routed_score": routed_score,
                    "local_score": local_score,
                    "failure_reason": failure_reason,
                    "failure_owner": failure_owner,
                }
            )
            print(
                f"[{spec.name} idx={idx}] format={int(fmt)} routed={int(routed_correct)} local={int(local_correct)} "
                f"reason={failure_reason} extracted={repr(extracted)[:72]}"
            , flush=True)
            _write_partial()
            time.sleep(float(args.sleep_s))

    _write_partial()
    print(f"[done] wrote report: {out_path}")


if __name__ == "__main__":
    main()
