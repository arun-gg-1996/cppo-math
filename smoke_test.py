"""CPU-safe smoke tests for CPPO integration."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cppo.reward import check_answer, score_batch
from cppo.evaluator_registry import EvaluatorRegistry
from cppo import train as train_mod
from cppo.trainer_cppo import (
    CPPO_STRATEGY_AUTHOR_EXACT,
    CPPO_STRATEGY_EXPERIMENTAL_REFILL,
    CPPOTrainer,
    select_cppo_keep_indices,
)

LOCAL_SMOKE_MODEL = "sshleifer/tiny-gpt2"


def cppo_keep_count(group_size: int, pruning: float) -> int:
    """Compute integer kept completions for group-size/pruning pair."""
    keep_float = float(group_size) * (1.0 - float(pruning))
    keep = int(round(keep_float))
    return keep


def test_reward_unit_cases() -> None:
    """Sanity-check core reward extraction/equivalence cases."""
    gt = "1/2"
    exact = "<think>...</think>\\n\\boxed{1/2}"
    equiv = "<think>...</think>\\n\\boxed{0.5}"
    tag = "<think>...</think>\\n<answer>1/2</answer>"
    wrong = "<think>...</think>\\n\\boxed{2}"
    final_line = "<think>...</think>\\nfinal answer is 1/2"
    unit_gt = "9 (apples)"
    unit_pred = "<answer>9</answer>"

    assert check_answer(exact, gt) == 1.0
    assert check_answer(equiv, gt) == 1.0
    assert check_answer(tag, gt) == 1.0
    assert check_answer(final_line, gt) == 1.0
    assert check_answer(unit_pred, unit_gt) == 1.0
    assert check_answer(wrong, gt) == 0.0


def test_evaluator_registry_fallback_backend() -> None:
    """Ensure evaluator registry fallback backend returns expected score/backend."""
    ev = EvaluatorRegistry({"default_backend": "fallback_sympy"})
    r = ev.score(
        split_name="math_500",
        predicted_text="<think>x</think><answer>1/2</answer>",
        ground_truth="0.5",
        row={},
    )
    assert r.backend == "fallback_sympy"
    assert r.score == 1.0


def test_reward_metrics_track_unique_and_per_source() -> None:
    """Ensure reward path emits cumulative source/problem counters."""
    train_mod.REWARD_STATS_BUFFER.clear()
    train_mod.SEEN_PROBLEM_IDS.clear()
    train_mod.SEEN_BY_SOURCE.clear()
    train_mod.SEEN_BY_DIFFICULTY.clear()
    train_mod.PROCESSED_BY_SOURCE.clear()
    train_mod.PROCESSED_BY_DIFFICULTY.clear()
    train_mod.PROCESSED_TOTAL = 0

    completions = ["<answer>2</answer>", "<answer>3</answer>", "<answer>10</answer>"]
    answers = ["2", "4", "10"]
    _ = train_mod._reward_fn(
        completions=completions,
        answer=answers,
        id=["p1", "p2", "p3"],
        source=["gsm8k_train", "svamp", "gsm8k_train"],
        difficulty=["easy", "medium", "easy"],
    )

    assert train_mod.REWARD_STATS_BUFFER, "Expected reward stats buffer to be populated."
    row = train_mod.REWARD_STATS_BUFFER[-1]
    assert "data/unique_problems_seen" in row
    assert row["data/unique_problems_seen"] == 3.0
    assert row["data/gsm8k_train_seen"] == 2.0
    assert row["data/svamp_seen"] == 1.0
    assert row["data/gsm8k_train_processed"] == 2.0
    assert row["data/svamp_processed"] == 1.0


def test_cppo_pruning_keeps_top_abs_adv_not_top_reward() -> None:
    """Validate CPPO selection is by |advantage|, not raw reward."""
    rewards = [1.0, 1.0, 0.0, 1.0]
    advantages = torch.tensor([0.15, 0.05, -0.90, 0.60], dtype=torch.float32)
    kept = select_cppo_keep_indices(advantages.abs(), keep_count=2, metric="smallest")
    assert kept == [2, 3], f"Expected [2, 3], got {kept}"
    assert rewards[2] == 0.0


def test_cppo_metric_smallest_vs_largest() -> None:
    """Validate metric switch flips kept-index ordering behavior."""
    advantages = torch.tensor([0.10, -0.80, 0.20, 0.70], dtype=torch.float32)
    keep_smallest = select_cppo_keep_indices(advantages.abs(), keep_count=2, metric="smallest")
    keep_largest = select_cppo_keep_indices(advantages.abs(), keep_count=2, metric="largest")
    assert keep_smallest == [1, 3], f"Expected [1, 3], got {keep_smallest}"
    assert keep_largest == [0, 2], f"Expected [0, 2], got {keep_largest}"


def test_cppo_keep_count_g16() -> None:
    """Validate paper-aligned keep counts for common G=16 pruning ratios."""
    assert cppo_keep_count(16, 0.875) == 2
    assert cppo_keep_count(16, 0.9375) == 1


def test_cppo_allocation_budget_math_author_exact() -> None:
    """Validate author-exact allocation math on a toy example."""
    group_size = 16
    pruning = 0.875
    keep_per_group = cppo_keep_count(group_size, pruning)

    # In author-exact allocation mode, the sampler-facing batch already uses keep_per_group (k),
    # then each unique prompt is expanded back to full G before global pruning.
    sampler_facing_batch = 32
    unique_prompts = sampler_facing_batch // keep_per_group
    generated = unique_prompts * group_size
    dropped = int(generated * pruning)
    kept = generated - dropped

    assert keep_per_group == 2
    assert kept == sampler_facing_batch


def test_cppo_allocation_budget_math_experimental_refill() -> None:
    """Validate refill strategy kept-budget arithmetic."""
    group_size = 16
    pruning = 0.875
    keep_per_group = cppo_keep_count(group_size, pruning)

    target_kept = 128
    first_pass_groups = 8
    first_pass_kept = first_pass_groups * keep_per_group
    needed_after_first = max(0, target_kept - first_pass_kept)
    extra_groups = int(math.ceil(float(needed_after_first) / float(keep_per_group)))

    assert first_pass_kept == 16
    assert extra_groups == 56
    assert first_pass_kept + extra_groups * keep_per_group >= target_kept


def _tiny_dataset() -> Dataset:
    """Small deterministic dataset for fast trainer smoke checks."""
    rows = [
        {
            "prompt": "Solve and return \\boxed{1/2} only.",
            "answer": "1/2",
            "id": "smoke_1",
            "source": "smoke",
            "difficulty": "easy",
        },
        {
            "prompt": "Solve and return \\boxed{2} only.",
            "answer": "2",
            "id": "smoke_2",
            "source": "smoke",
            "difficulty": "easy",
        },
        {
            "prompt": "Solve and return \\boxed{3} only.",
            "answer": "3",
            "id": "smoke_3",
            "source": "smoke",
            "difficulty": "easy",
        },
        {
            "prompt": "Solve and return \\boxed{4} only.",
            "answer": "4",
            "id": "smoke_4",
            "source": "smoke",
            "difficulty": "easy",
        },
    ]
    return Dataset.from_list(rows)


def _reward_fn(completions, answer, **kwargs):
    """Minimal reward function for smoke training."""
    return score_batch(completions, answer)


def _build_args(**overrides: Any) -> GRPOConfig:
    """Build tiny GRPOConfig with override support."""
    fields = GRPOConfig.__dataclass_fields__.keys()
    kwargs: dict[str, Any] = {
        "output_dir": str(Path(tempfile.mkdtemp(prefix="cppo_smoke_"))),
        "max_steps": 2,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-6,
        "logging_steps": 1,
        "save_steps": 2,
        "save_safetensors": True,
        "report_to": [],
        "bf16": False,
        "num_generations": 2,
        "temperature": 0.8,
        "beta": 0.001,
        "loss_type": "grpo",
        "num_iterations": 1,
        "max_completion_length": 32,
        "use_vllm": False,
        "remove_unused_columns": False,
        "generation_batch_size": 4,
    }
    kwargs.update(overrides)
    if "max_prompt_length" in fields:
        kwargs["max_prompt_length"] = 128
    return GRPOConfig(**{k: v for k, v in kwargs.items() if k in fields})


def run_two_step_smoke_train(
    model_id: str, *, cppo: bool, allocation: bool, cppo_strategy: str = CPPO_STRATEGY_AUTHOR_EXACT
) -> list[dict[str, Any]]:
    """Run 2-step smoke train and return trainer log history rows."""
    dataset = _tiny_dataset()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cppo:
        args = _build_args(
            num_generations=16,
            generation_batch_size=32,
            per_device_train_batch_size=2,
        )
        trainer: GRPOTrainer = CPPOTrainer(
            model=model_id,
            reward_funcs=_reward_fn,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            cppo_pruning=0.875,
            cppo_metric="smallest",
            cppo_allocation=allocation,
            cppo_strategy=cppo_strategy,
        )
    else:
        args = _build_args(num_generations=2, generation_batch_size=4, per_device_train_batch_size=2)
        trainer = GRPOTrainer(
            model=model_id,
            reward_funcs=_reward_fn,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

    trainer.train()
    trainer.save_model(str(Path(args.output_dir) / "final"))
    return [dict(x) for x in trainer.state.log_history if isinstance(x, dict)]


def main() -> None:
    """Execute deterministic smoke checks and optional tiny train smoke."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-model", default=LOCAL_SMOKE_MODEL)
    parser.add_argument("--skip-trainer", action="store_true")
    args = parser.parse_args()

    print("=== CPPO Smoke Test ===")
    test_reward_unit_cases()
    test_evaluator_registry_fallback_backend()
    test_reward_metrics_track_unique_and_per_source()
    test_cppo_pruning_keeps_top_abs_adv_not_top_reward()
    test_cppo_metric_smallest_vs_largest()
    test_cppo_keep_count_g16()
    test_cppo_allocation_budget_math_author_exact()
    test_cppo_allocation_budget_math_experimental_refill()
    print("deterministic unit checks: PASS")

    if args.skip_trainer:
        print("2-step trainer smoke: SKIPPED (--skip-trainer)")
    else:
        try:
            baseline_history = run_two_step_smoke_train(args.smoke_model, cppo=False, allocation=False)
            has_cppo_in_baseline = any(any(k.startswith("cppo/") for k in row.keys()) for row in baseline_history)
            assert not has_cppo_in_baseline, "Baseline GRPO run should not emit cppo/* metrics"
            print("2-step baseline trainer smoke: PASS")

            cppo_history = run_two_step_smoke_train(
                args.smoke_model,
                cppo=True,
                allocation=True,
                cppo_strategy=CPPO_STRATEGY_AUTHOR_EXACT,
            )
            has_cppo_metrics = any(any(k.startswith("cppo/") for k in row.keys()) for row in cppo_history)
            if not has_cppo_metrics:
                raise AssertionError("CPPO run did not emit cppo/* metrics")
            filled_vals = [float(row.get("cppo/filled_fraction", 0.0)) for row in cppo_history]
            if max(filled_vals) <= 0.0:
                raise AssertionError("CPPO run emitted invalid cppo/filled_fraction")
            print("2-step CPPO trainer smoke: PASS")
        except OSError as e:
            print(f"2-step trainer smoke: SKIPPED (model download unavailable: {e})")
        except Exception as e:
            print(f"2-step trainer smoke: FAIL ({e})")
            raise

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
