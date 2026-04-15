"""Binary SymPy reward for CPPO/GRPO math training."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

try:
    from sympy.parsing.latex import parse_latex
except Exception:  # pragma: no cover
    parse_latex = None

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_FORMAT_PATTERN = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL)
_CPPO_FORMAT_PATTERN = re.compile(r"^<think>.*?</think>\n<answer>.*?</answer>$", re.DOTALL | re.MULTILINE)


def unwrap_completion(completion: Any) -> str:
    """Normalize TRL/vLLM completion payloads to a plain assistant string."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                c = msg.get("content", "")
                return c if isinstance(c, str) else str(c)
    if isinstance(completion, dict) and completion.get("role") == "assistant":
        c = completion.get("content", "")
        return c if isinstance(c, str) else str(c)
    return str(completion)


def extract_boxed(text: str) -> str | None:
    """Extract content from first \\boxed{...}, supporting nested braces."""
    if not text:
        return None
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
    s = "".join(out).strip()
    return s or None


def extract_answer_tag(text: str) -> str | None:
    """Extract text from the final `<answer>...</answer>` block."""
    if not text:
        return None
    parts = text.split("<answer>")
    if len(parts) < 2:
        return None
    tail = parts[-1]
    if "</answer>" not in tail:
        return None
    s = tail.split("</answer>")[0].strip()
    return s or None


def check_format_compliance(predicted_text: str) -> float:
    """Strict format check: one think block and one answer block."""
    if not predicted_text:
        return 0.0
    txt = str(predicted_text)
    if _FORMAT_PATTERN.match(txt) is None:
        return 0.0
    if txt.count("<think>") != 1 or txt.count("</think>") != 1:
        return 0.0
    if txt.count("<answer>") != 1 or txt.count("</answer>") != 1:
        return 0.0
    answer = extract_answer_tag(txt)
    return 1.0 if answer is not None and answer.strip() else 0.0


def cppo_format_compliance(predicted_text: str) -> float:
    """CPPO GSM-style format check used in official rewards_gsm.py."""
    if not predicted_text:
        return 0.0
    txt = str(predicted_text)
    return 1.0 if _CPPO_FORMAT_PATTERN.match(txt) is not None else 0.0


def extract_code_fence(text: str) -> str | None:
    """Extract final fenced code block body (language tag optional)."""
    if not text:
        return None
    matches = re.findall(r"```(?:[A-Za-z0-9_+-]*)?\n?(.*?)```", text, flags=re.DOTALL)
    if not matches:
        return None
    s = matches[-1].strip()
    return s or None


def extract_final_answer_line(text: str) -> str | None:
    """Extract loose `final answer is ...` style fallback."""
    if not text:
        return None
    patterns = [
        r"final answer\s*(?:is|:)\s*(.+)",
        r"answer\s*(?:is|:)\s*(.+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            s = matches[-1].strip()
            s = s.rstrip(".")
            if s:
                return s
    return None


def _clean_candidate_text(s: str) -> str:
    """Trim wrappers/markdown noise from extracted answer text."""
    s = s.strip()
    s = s.strip("`")
    s = re.sub(r"^(\*\*|__)+|(\*\*|__)+$", "", s)
    s = s.replace("\u00a0", " ")
    return s.strip()


def _strip_tex_wrappers(s: str) -> str:
    """Normalize common LaTeX wrappers into parser-friendly text."""
    s = s.strip()
    s = re.sub(r"^\$+|\$+$", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("^", "**")
    s = s.replace("−", "-")
    s = s.replace("\\,", "")
    return s.strip()


def _replace_frac(s: str) -> str:
    """Convert simple `\\frac{a}{b}` patterns to `(a)/(b)`."""
    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    prev = None
    cur = s
    while prev != cur:
        prev = cur
        cur = pattern.sub(r"(\1)/(\2)", cur)
    return cur


def _normalize_expr_string(s: str) -> str:
    """Apply text-level normalization before symbolic parsing."""
    s = _strip_tex_wrappers(s)
    s = _replace_frac(s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return s.strip()


def _try_decimal_or_fraction(s: str) -> sp.Expr | None:
    """Fast path parser for plain numeric forms."""
    s = s.strip()
    if not s:
        return None
    try:
        if "/" in s and re.fullmatch(r"\s*[-+]?\d+\s*/\s*[-+]?\d+\s*", s):
            return sp.Rational(Fraction(s))
        d = Decimal(s)
        if d.is_finite():
            return sp.nsimplify(str(d), rational=True)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return None


def _parse_to_expr(s: str) -> sp.Expr | None:
    """Safely parse candidate math text into a SymPy expression."""
    direct = _try_decimal_or_fraction(s)
    if direct is not None:
        return direct

    normalized = _normalize_expr_string(s)
    if not normalized:
        return None

    direct_norm = _try_decimal_or_fraction(normalized)
    if direct_norm is not None:
        return direct_norm

    # Never attempt parser eval on suspicious content.
    if re.search(r"[;`]", normalized):
        return None
    if re.search(r"\b(import|print|exec|eval|open|os|sys|subprocess|lambda|while|for)\b", normalized):
        return None
    for fn in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", normalized):
        if fn.lower() not in {"sin", "cos", "tan", "sqrt", "log", "ln", "exp", "abs"}:
            return None

    # Prefer latex parser when available and input still contains latex markers.
    if parse_latex is not None and "\\" in s:
        try:
            return sp.simplify(parse_latex(s))
        except Exception:
            pass

    try:
        return sp.simplify(parse_expr(normalized, transformations=_TRANSFORMS, evaluate=True))
    except Exception:
        return None


def _extract_single_number_value(s: str) -> float | None:
    """Extract one numeric literal from a string, else return None."""
    if not s:
        return None
    txt = s.replace(",", "")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
    if len(nums) != 1:
        return None
    try:
        return float(nums[0])
    except Exception:
        return None


def _extract_last_number_value(s: str) -> float | None:
    """Extract the final numeric literal from a string, else return None."""
    if not s:
        return None
    txt = s.replace(",", "")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except Exception:
        return None


def _split_top_level_commas(s: str) -> list[str]:
    """Split comma-separated tuple parts while respecting bracket nesting."""
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in s:
        if ch == "," and depth == 0:
            token = "".join(cur).strip()
            if token:
                parts.append(token)
            cur = []
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth > 0:
            depth -= 1
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _tuple_parts(s: str) -> list[str] | None:
    """Return tuple/list components for `(a,b,...)` or `[a,b,...]` answers."""
    if not s:
        return None
    raw = _strip_tex_wrappers(s)
    if len(raw) < 5:
        return None
    if not ((raw.startswith("(") and raw.endswith(")")) or (raw.startswith("[") and raw.endswith("]"))):
        return None
    inner = raw[1:-1].strip()
    if "," not in inner:
        return None
    parts = _split_top_level_commas(inner)
    if len(parts) < 2:
        return None
    return parts


def equivalent_math(pred: str, gt: str) -> bool:
    """Check symbolic/numeric equivalence between predicted and target answers."""
    if not pred or not gt:
        return False

    p0 = pred.strip()
    g0 = gt.strip()
    if p0 == g0:
        return True

    pn = _normalize_expr_string(p0)
    gn = _normalize_expr_string(g0)
    if pn and gn and pn == gn:
        return True

    p_parts = _tuple_parts(p0)
    g_parts = _tuple_parts(g0)
    if p_parts is not None or g_parts is not None:
        if p_parts is None or g_parts is None or len(p_parts) != len(g_parts):
            return False
        return all(equivalent_math(pp, gg) for pp, gg in zip(p_parts, g_parts, strict=True))

    pexpr = _parse_to_expr(p0)
    gexpr = _parse_to_expr(g0)
    if pexpr is None or gexpr is None:
        pv = _extract_single_number_value(p0)
        gv = _extract_single_number_value(g0)
        if pv is not None and gv is not None:
            return abs(pv - gv) <= 1e-8
        return False

    try:
        if sp.simplify(pexpr - gexpr) == 0:
            return True

        pval = sp.N(pexpr)
        gval = sp.N(gexpr)
        if pval.is_real and gval.is_real:
            pv = float(pval)
            gv = float(gval)
            if math.isfinite(pv) and math.isfinite(gv):
                return abs(pv - gv) <= 1e-8
    except Exception:
        return False

    return False


def extract_prediction_answer(predicted_text: str) -> str | None:
    """Extract best candidate answer from model output using ordered fallbacks."""
    if not predicted_text:
        return None

    # Prefer explicit answer channels, then robust fallbacks.
    for extractor in (extract_answer_tag, extract_boxed, extract_final_answer_line, extract_code_fence):
        out = extractor(predicted_text)
        if out:
            return _clean_candidate_text(out)

    # Last line fallback for unconstrained outputs.
    lines = [ln.strip() for ln in predicted_text.splitlines() if ln.strip()]
    if lines:
        return _clean_candidate_text(lines[-1])
    return None


def _ground_truth_candidates(ground_truth: str) -> list[str]:
    """Expand ground truth into normalized candidate set for matching."""
    gt = (ground_truth or "").strip()
    if not gt:
        return []

    cands = [gt]
    if " or " in gt.lower():
        cands.extend([x.strip() for x in re.split(r"\bor\b", gt, flags=re.IGNORECASE) if x.strip()])

    # Unit-bearing answers like "9 (apples)" should match numeric-only predictions.
    n = _extract_single_number_value(gt)
    if n is not None:
        cands.append(str(n))
        cands.append(str(int(n)) if float(n).is_integer() else str(n))

    out: list[str] = []
    seen: set[str] = set()
    for c in cands:
        cc = _clean_candidate_text(c)
        if cc and cc not in seen:
            seen.add(cc)
            out.append(cc)
    return out


def _cppo_dataset_answer(text: str) -> str | None:
    """CPPO GSM dataset answer extraction (`####`) with graceful fallback."""
    raw = (text or "").strip()
    if not raw:
        return None
    if "####" in raw:
        raw = raw.split("####", 1)[1].strip()
    raw = raw.replace(",", "")
    return raw or None


def _cppo_answer_tag(text: str) -> str | None:
    """CPPO GSM model output extraction: last `<answer>...</answer>` only."""
    out = extract_answer_tag(text)
    if not out:
        return None
    out = out.strip().replace(",", "")
    return None if out == "..." else out


def cppo_gsm_accuracy_reward(predicted_text: str, ground_truth: str) -> float:
    """CPPO GSM-style shaped accuracy reward (2.0 exact / 1.5 numeric / 0.0)."""
    pred = _cppo_answer_tag(predicted_text)
    gt = _cppo_dataset_answer(ground_truth)
    if not pred or not gt:
        return 0.0
    if pred == gt:
        return 2.0
    pred_num = _extract_single_number_value(str(pred))
    gt_num = _extract_single_number_value(str(gt))
    if pred_num is not None and gt_num is not None and abs(pred_num - gt_num) <= 1e-8:
        return 1.5
    return 0.0


def cppo_gsm_eval_match(predicted_text: str, ground_truth: str) -> bool:
    """CPPO eval_gsm-style boolean correctness for checkpoint comparison."""
    gt = _cppo_dataset_answer(ground_truth)
    if not gt:
        return False

    def _match(candidate: str | None, target: str) -> bool:
        if candidate is None:
            return False
        c = str(candidate).strip().replace(",", "")
        if c == target:
            return True
        c_single = _extract_single_number_value(c)
        try:
            t_float = float(target)
        except Exception:
            t_float = None
        if c_single is not None and t_float is not None and abs(c_single - t_float) <= 1e-8:
            return True
        c_last = _extract_last_number_value(c)
        t_last = _extract_last_number_value(target)
        if c_last is not None and t_last is not None and abs(c_last - t_last) <= 1e-8:
            return True
        return False

    if _match(predicted_text, gt):
        return True
    return _match(_cppo_answer_tag(predicted_text), gt)


def check_answer(predicted_text: str, ground_truth: str) -> float:
    """Binary reward: 1.0 if equivalent, else 0.0."""
    pred = extract_prediction_answer(predicted_text)
    gt_cands = _ground_truth_candidates(ground_truth)
    def _raw_numeric_fallback() -> float:
        # Official GSM-style numeric fallback on raw completion:
        # if tag extraction is missing/noisy but output contains a clear numeric answer,
        # accept single-number or last-number matches.
        gt_nums = [x for x in (_extract_single_number_value(gt) for gt in gt_cands) if x is not None]
        if not gt_nums:
            return 0.0
        raw = (predicted_text or "").replace("$", "").replace("%", "")
        p_single = _extract_single_number_value(raw)
        p_last = _extract_last_number_value(raw)
        for g in gt_nums:
            if p_single is not None and abs(p_single - g) <= 1e-8:
                return 1.0
            if p_last is not None and abs(p_last - g) <= 1e-8:
                return 1.0
        return 0.0

    if pred is None:
        return _raw_numeric_fallback()

    pred_low = pred.strip().lower()
    for gt in gt_cands:
        gt_low = gt.strip().lower()
        if pred == gt or pred_low == gt_low:
            return 1.0
        if pred_low in {"none", "no answer", "cannot be determined"} and gt_low in {"none", "no answer", "cannot be determined"}:
            return 1.0
        if equivalent_math(pred, gt):
            return 1.0
    return _raw_numeric_fallback()


def score_batch(completions: list[Any], answers: list[str]) -> list[float]:
    """Batch helper for reward evaluation."""
    if len(completions) != len(answers):
        n = min(len(completions), len(answers))
        completions = completions[:n]
        answers = answers[:n]
    return [check_answer(unwrap_completion(c), gt) for c, gt in zip(completions, answers)]
