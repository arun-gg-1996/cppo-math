from __future__ import annotations

"""Source-aware evaluator router for training and evaluation phases."""

import importlib
import logging
import re
from dataclasses import dataclass
from typing import Any

from .reward import check_answer, extract_prediction_answer

logger = logging.getLogger("cppo.evaluator")


DEFAULT_MATH_VERIFY_SPLITS = {
    "gsm8k",
    "gsm8k_train",
    "gsm8k_test",
    "svamp",
    "gsm_plus",
    "asdiv",
    "aime_2024",
    "aime_2025",
    "amc_2023",
}


@dataclass(frozen=True)
class EvalScore:
    score: float
    backend: str
    note: str = ""


def _to_backend_map(raw: Any) -> dict[str, str]:
    """Normalize split->backend config mapping."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        sk = str(k).strip()
        sv = str(v).strip()
        if sk and sv:
            out[sk] = sv
    return out


def _normalize_split_name(split_name: str) -> str:
    """Normalize split/source names so train/eval routing stays consistent.

    Examples:
      - gsm8k_train -> gsm8k
      - gsm8k_train:openai/gsm8k -> gsm8k
      - svamp:foo -> svamp
    """
    s = str(split_name or "").strip().lower()
    if not s:
        return "unknown"

    # Preserve dataset tag before any provider-specific suffix.
    if ":" in s:
        s = s.split(":", 1)[0].strip()

    # Normalize separators and drop noisy characters.
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    s = re.sub(r"_+", "_", s)
    if not s:
        return "unknown"

    # Canonical aliases for common train/eval variants.
    if s.startswith("gsm8k_train"):
        return "gsm8k"
    if s.startswith("gsm8k_test"):
        return "gsm8k_test"
    if s.startswith("gsm8k"):
        return "gsm8k"

    if s.startswith("svamp"):
        return "svamp"
    if s.startswith("gsm_plus"):
        return "gsm_plus"
    if s.startswith("asdiv"):
        return "asdiv"
    if s.startswith("aime_2024"):
        return "aime_2024"
    if s.startswith("aime_2025"):
        return "aime_2025"
    if s.startswith("amc_2023"):
        return "amc_2023"
    if s.startswith("math_500"):
        return "math_500"
    if s.startswith("minerva_math"):
        return "minerva_math"
    if s.startswith("olympiadbench"):
        return "olympiadbench"

    return s


def _is_none_like(text: str) -> bool:
    """Return True for common 'no answer' textual variants."""
    t = (text or "").strip().lower()
    return t in {"none", "no answer", "cannot be determined", "not enough information"}


def _ground_truth_candidates(gt: str) -> list[str]:
    """Expand ground truth variants (for example `a or b`)."""
    g = (gt or "").strip()
    if not g:
        return []
    out = [g]
    if " or " in g.lower():
        out.extend([x.strip() for x in re.split(r"\bor\b", g, flags=re.IGNORECASE) if x.strip()])
    dedup: list[str] = []
    seen: set[str] = set()
    for x in out:
        xx = x.strip()
        if xx and xx not in seen:
            seen.add(xx)
            dedup.append(xx)
    return dedup


class EvaluatorRegistry:
    """Source-aware evaluation router.

    Priority in auto mode:
      1) custom callable backend (if configured)
      2) math-verify backend (if enabled and available)
      3) local fallback checker
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = dict(cfg or {})
        self.default_backend = str(self.cfg.get("default_backend", "auto")).strip().lower() or "auto"
        self.backend_by_split = _to_backend_map(self.cfg.get("backend_by_split", {}))
        self.custom_by_split = _to_backend_map(self.cfg.get("custom_by_split", {}))
        self.backend_by_split_norm = {
            _normalize_split_name(k): str(v).strip()
            for k, v in self.backend_by_split.items()
            if str(k).strip() and str(v).strip()
        }
        self.custom_by_split_norm = {
            _normalize_split_name(k): str(v).strip()
            for k, v in self.custom_by_split.items()
            if str(k).strip() and str(v).strip()
        }

        mv_cfg = self.cfg.get("math_verify", {})
        if not isinstance(mv_cfg, dict):
            mv_cfg = {}
        self.mv_enabled = bool(mv_cfg.get("enabled", True))
        raw_mv_splits = mv_cfg.get("splits", [])
        if isinstance(raw_mv_splits, list) and raw_mv_splits:
            self.mv_splits = {str(x).strip().lower() for x in raw_mv_splits if str(x).strip()}
        else:
            self.mv_splits = set(DEFAULT_MATH_VERIFY_SPLITS)
        self.mv_splits_norm = {_normalize_split_name(x) for x in self.mv_splits}

        self._custom_cache: dict[str, Any] = {}
        self._mv_checked = False
        self._mv_available = False
        self._mv_mod: Any = None
        self._warned_missing_mv = False

    def score(
        self,
        *,
        split_name: str,
        predicted_text: str,
        ground_truth: str,
        row: dict[str, Any] | None = None,
    ) -> EvalScore:
        """Route one prediction to the configured backend and return score metadata."""
        split_raw = str(split_name or "").strip()
        split_norm = _normalize_split_name(split_raw)
        forced = str(
            self.backend_by_split.get(split_raw)
            or self.backend_by_split_norm.get(split_norm)
            or self.default_backend
        ).strip().lower()
        if forced == "auto":
            return self._score_auto(
                split_name=split_raw,
                predicted_text=predicted_text,
                ground_truth=ground_truth,
                row=row,
            )
        return self._score_with_backend(
            backend=forced,
            split_name=split_norm,
            predicted_text=predicted_text,
            ground_truth=ground_truth,
            row=row,
        )

    def _score_auto(
        self,
        *,
        split_name: str,
        predicted_text: str,
        ground_truth: str,
        row: dict[str, Any] | None = None,
    ) -> EvalScore:
        """Auto-routing policy: custom -> math-verify -> local fallback."""
        split_raw = str(split_name or "").strip()
        split_norm = _normalize_split_name(split_raw)

        custom_key = None
        if split_raw in self.custom_by_split:
            custom_key = split_raw
        elif split_norm in self.custom_by_split_norm:
            custom_key = split_norm

        if custom_key is not None:
            return self._score_with_backend(
                backend="custom",
                split_name=custom_key,
                predicted_text=predicted_text,
                ground_truth=ground_truth,
                row=row,
            )

        use_math_verify = (
            split_raw.lower() in self.mv_splits
            or split_norm in self.mv_splits_norm
        )
        if self.mv_enabled and use_math_verify:
            mv = self._score_with_backend(
                backend="math_verify",
                split_name=split_norm,
                predicted_text=predicted_text,
                ground_truth=ground_truth,
                row=row,
            )
            if mv.note != "math_verify_unavailable":
                return mv
        return self._score_with_backend(
            backend="fallback_sympy",
            split_name=split_norm,
            predicted_text=predicted_text,
            ground_truth=ground_truth,
            row=row,
        )

    def _score_with_backend(
        self,
        *,
        backend: str,
        split_name: str,
        predicted_text: str,
        ground_truth: str,
        row: dict[str, Any] | None = None,
    ) -> EvalScore:
        """Score one prediction using a specific backend with robust fallback."""
        b = backend.strip().lower()
        if b in {"fallback", "fallback_sympy", "sympy", "local"}:
            s = check_answer(predicted_text, ground_truth)
            return EvalScore(score=float(s), backend="fallback_sympy")

        if b in {"math_verify", "math-verify"}:
            mv = self._score_math_verify(predicted_text=predicted_text, ground_truth=ground_truth)
            if mv is not None:
                return mv
            # Hard fallback even if forced, to avoid dropping scores.
            s = check_answer(predicted_text, ground_truth)
            return EvalScore(score=float(s), backend="fallback_sympy", note="math_verify_unavailable")

        if b == "custom":
            fn = self._load_custom_callable(split_name)
            if fn is not None:
                try:
                    raw = fn(predicted_text, ground_truth, row or {})
                    val = 1.0 if bool(raw) else 0.0
                    if isinstance(raw, (int, float)):
                        val = float(raw)
                    return EvalScore(score=max(0.0, min(1.0, val)), backend=f"custom:{split_name}")
                except Exception as e:
                    logger.warning("Custom evaluator failed for split=%s (%s)", split_name, e)
            s = check_answer(predicted_text, ground_truth)
            return EvalScore(score=float(s), backend="fallback_sympy", note="custom_failed")

        # Unknown backend -> robust fallback.
        s = check_answer(predicted_text, ground_truth)
        return EvalScore(score=float(s), backend="fallback_sympy", note=f"unknown_backend:{b}")

    def _load_custom_callable(self, split_name: str):
        """Load and cache custom evaluator callable for a split."""
        split_raw = str(split_name or "").strip()
        split_norm = _normalize_split_name(split_raw)
        spec = (
            self.custom_by_split.get(split_raw, "")
            or self.custom_by_split_norm.get(split_norm, "")
        )
        if not spec:
            return None
        cache_key = split_norm
        if cache_key in self._custom_cache:
            return self._custom_cache[cache_key]
        mod_name, sep, fn_name = spec.partition(":")
        if not sep:
            logger.warning(
                "Invalid custom evaluator spec for split=%s: '%s' (expected 'module:function')",
                split_raw or split_norm,
                spec,
            )
            self._custom_cache[cache_key] = None
            return None
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name)
            if not callable(fn):
                raise TypeError(f"{spec} is not callable")
            self._custom_cache[cache_key] = fn
            return fn
        except Exception as e:
            logger.warning("Failed to import custom evaluator '%s' for split=%s (%s)", spec, split_raw or split_norm, e)
            self._custom_cache[cache_key] = None
            return None

    def _ensure_math_verify(self) -> None:
        """Lazy-import math_verify and cache availability flags."""
        if self._mv_checked:
            return
        self._mv_checked = True
        try:
            self._mv_mod = importlib.import_module("math_verify")
            self._mv_available = True
        except Exception:
            self._mv_mod = None
            self._mv_available = False

    def _mv_parse(self, text: str):
        """Version-tolerant wrapper around `math_verify.parse`."""
        assert self._mv_mod is not None
        parse_fn = getattr(self._mv_mod, "parse", None)
        if not callable(parse_fn):
            raise RuntimeError("math_verify.parse not found")
        try:
            return parse_fn(text)
        except TypeError:
            # Version-compatible attempt with extraction configs.
            cfgs = []
            for cls_name in ("LatexExtractionConfig", "ExprExtractionConfig", "StringExtractionConfig"):
                cls = getattr(self._mv_mod, cls_name, None)
                if cls is not None:
                    try:
                        cfgs.append(cls())
                    except Exception:
                        pass
            if cfgs:
                for kw in ("extraction_config", "extraction_configs"):
                    try:
                        return parse_fn(text, **{kw: cfgs})
                    except TypeError:
                        continue
            raise

    def _mv_verify_pair(self, a, b) -> bool:
        """Version-tolerant wrapper around `math_verify.verify`."""
        assert self._mv_mod is not None
        verify_fn = getattr(self._mv_mod, "verify", None)
        if not callable(verify_fn):
            raise RuntimeError("math_verify.verify not found")
        out = verify_fn(a, b)
        if isinstance(out, bool):
            return out
        if isinstance(out, (int, float)):
            return bool(out)
        if isinstance(out, (list, tuple, set)):
            return any(bool(x) for x in out)
        return bool(out)

    def _score_math_verify(self, *, predicted_text: str, ground_truth: str) -> EvalScore | None:
        """Score using math-verify backend; return None when backend unavailable."""
        self._ensure_math_verify()
        if not self._mv_available:
            if not self._warned_missing_mv:
                logger.warning("math-verify not installed; falling back to local evaluator where needed.")
                self._warned_missing_mv = True
            return None

        pred = extract_prediction_answer(predicted_text)
        if pred is None:
            return EvalScore(score=0.0, backend="math_verify")

        if _is_none_like(pred):
            return EvalScore(score=1.0 if _is_none_like(ground_truth) else 0.0, backend="math_verify")

        gt_cands = _ground_truth_candidates(ground_truth)
        if not gt_cands:
            return EvalScore(score=0.0, backend="math_verify")

        try:
            pred_parsed = self._mv_parse(pred)
        except Exception:
            return EvalScore(score=float(check_answer(predicted_text, ground_truth)), backend="fallback_sympy", note="math_verify_parse_pred_failed")

        for gt in gt_cands:
            try:
                gt_parsed = self._mv_parse(gt)
            except Exception:
                continue
            try:
                if self._mv_verify_pair(pred_parsed, gt_parsed):
                    return EvalScore(score=1.0, backend="math_verify")
            except Exception:
                pass
            try:
                # Some versions use reversed arg order.
                if self._mv_verify_pair(gt_parsed, pred_parsed):
                    return EvalScore(score=1.0, backend="math_verify")
            except Exception:
                pass
        return EvalScore(score=0.0, backend="math_verify")
