from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass
from typing import Any

from .reward import check_answer, extract_prediction_answer

logger = logging.getLogger("cppo.evaluator")


DEFAULT_MATH_VERIFY_SPLITS = {
    "gsm8k",
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
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        sk = str(k).strip()
        sv = str(v).strip()
        if sk and sv:
            out[sk] = sv
    return out


def _is_none_like(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"none", "no answer", "cannot be determined", "not enough information"}


def _ground_truth_candidates(gt: str) -> list[str]:
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

        mv_cfg = self.cfg.get("math_verify", {})
        if not isinstance(mv_cfg, dict):
            mv_cfg = {}
        self.mv_enabled = bool(mv_cfg.get("enabled", True))
        raw_mv_splits = mv_cfg.get("splits", [])
        if isinstance(raw_mv_splits, list) and raw_mv_splits:
            self.mv_splits = {str(x).strip() for x in raw_mv_splits if str(x).strip()}
        else:
            self.mv_splits = set(DEFAULT_MATH_VERIFY_SPLITS)

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
        split = str(split_name or "").strip()
        forced = self.backend_by_split.get(split, self.default_backend).strip().lower()
        if forced == "auto":
            return self._score_auto(split_name=split, predicted_text=predicted_text, ground_truth=ground_truth, row=row)
        return self._score_with_backend(
            backend=forced,
            split_name=split,
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
        if split_name in self.custom_by_split:
            return self._score_with_backend(
                backend="custom",
                split_name=split_name,
                predicted_text=predicted_text,
                ground_truth=ground_truth,
                row=row,
            )
        if self.mv_enabled and split_name in self.mv_splits:
            mv = self._score_with_backend(
                backend="math_verify",
                split_name=split_name,
                predicted_text=predicted_text,
                ground_truth=ground_truth,
                row=row,
            )
            if mv.note != "math_verify_unavailable":
                return mv
        return self._score_with_backend(
            backend="fallback_sympy",
            split_name=split_name,
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
        spec = self.custom_by_split.get(split_name, "")
        if not spec:
            return None
        if split_name in self._custom_cache:
            return self._custom_cache[split_name]
        mod_name, sep, fn_name = spec.partition(":")
        if not sep:
            logger.warning(
                "Invalid custom evaluator spec for split=%s: '%s' (expected 'module:function')",
                split_name,
                spec,
            )
            self._custom_cache[split_name] = None
            return None
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name)
            if not callable(fn):
                raise TypeError(f"{spec} is not callable")
            self._custom_cache[split_name] = fn
            return fn
        except Exception as e:
            logger.warning("Failed to import custom evaluator '%s' for split=%s (%s)", spec, split_name, e)
            self._custom_cache[split_name] = None
            return None

    def _ensure_math_verify(self) -> None:
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
