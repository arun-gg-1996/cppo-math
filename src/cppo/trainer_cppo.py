from __future__ import annotations

"""CPPO trainer implementation on top of TRL's GRPOTrainer.

This module keeps two CPPO modes:
- `author_exact`: behavior aligned with the public CPPO implementation.
- `experimental_refill`: a custom mode that preserves fixed kept slots per step.
"""

import copy
import math
import random
from typing import Any

import numpy as np
import torch
from trl import GRPOTrainer
from trl.trainer.grpo_trainer import disable_gradient_checkpointing, nanstd, pad, use_adapter

CPPO_STRATEGY_AUTHOR_EXACT = "author_exact"
CPPO_STRATEGY_EXPERIMENTAL_REFILL = "experimental_refill"


def select_cppo_keep_indices(abs_advantages: torch.Tensor, keep_count: int, metric: str) -> list[int]:
    """Select completion indices to keep after ranking by absolute advantage."""
    if keep_count <= 0:
        return []
    if keep_count >= abs_advantages.numel():
        return list(range(abs_advantages.numel()))

    if metric == "smallest":
        # Prune smallest |A| -> keep largest |A|
        order = torch.argsort(abs_advantages, descending=True, stable=True)
    elif metric == "largest":
        # Prune largest |A| -> keep smallest |A|
        order = torch.argsort(abs_advantages, descending=False, stable=True)
    else:
        raise ValueError(f"Unknown CPPO metric: {metric}")

    chosen = order[:keep_count]
    chosen, _ = torch.sort(chosen)
    return [int(x) for x in chosen.tolist()]


class CPPOTrainer(GRPOTrainer):
    """GRPO trainer with CPPO completion pruning and optional dynamic allocation."""

    def __init__(
        self,
        *args,
        cppo_pruning: float,
        cppo_metric: str = "smallest",
        cppo_allocation: bool = False,
        cppo_strategy: str = CPPO_STRATEGY_AUTHOR_EXACT,
        **kwargs,
    ):
        """Initialize CPPO knobs and derive keep/presample values.

        Args:
            cppo_pruning: Fraction to prune from generated completions.
            cppo_metric: `smallest` (canonical) or `largest`.
            cppo_allocation: Enable dynamic allocation behavior.
            cppo_strategy: `author_exact` or `experimental_refill`.
        """
        super().__init__(*args, **kwargs)
        if not (0.0 <= float(cppo_pruning) < 1.0):
            raise ValueError(f"cppo_pruning must be in [0,1), got {cppo_pruning}")

        strategy = str(cppo_strategy).strip().lower()
        if strategy not in {CPPO_STRATEGY_AUTHOR_EXACT, CPPO_STRATEGY_EXPERIMENTAL_REFILL}:
            raise ValueError(
                f"Unknown cppo_strategy={cppo_strategy}. "
                f"Use {CPPO_STRATEGY_AUTHOR_EXACT}|{CPPO_STRATEGY_EXPERIMENTAL_REFILL}."
            )

        self.cppo_pruning = float(cppo_pruning)
        self.cppo_metric = str(cppo_metric)
        self.cppo_allocation = bool(cppo_allocation)
        self.cppo_strategy = strategy

        self.cppo_original_num_generations = int(self.num_generations)
        keep_float = float(self.cppo_original_num_generations) * (1.0 - self.cppo_pruning)
        self.cppo_keep_per_group = int(round(keep_float))
        if self.cppo_keep_per_group < 1 or abs(self.cppo_keep_per_group - keep_float) > 1e-9:
            raise ValueError(
                "Invalid CPPO keep count from "
                f"num_generations={self.cppo_original_num_generations} pruning={self.cppo_pruning} -> {keep_float}"
            )

        presample_float = 1.0 / max(1e-9, (1.0 - self.cppo_pruning))
        # Use ceil to avoid under-filling in refill mode when ratio is non-integer.
        self.cppo_presample_factor = max(1, int(math.ceil(presample_float - 1e-12)))

        # Author-exact allocation reshapes sampling by switching the trainer-facing group size to kept-per-group.
        # Full-group advantages still use `cppo_original_num_generations` before pruning.
        if self.cppo_strategy == CPPO_STRATEGY_AUTHOR_EXACT and self.cppo_allocation:
            self.num_generations = self.cppo_keep_per_group
            if hasattr(self.args, "num_generations"):
                self.args.num_generations = self.num_generations

    def _compute_advantages(self, rewards_per_func: torch.Tensor, num_generations: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute weighted rewards/advantages using the same math as GRPOTrainer."""
        device = self.accelerator.device
        if self.multi_objective_aggregation == "sum_then_normalize":
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            if self.scale_rewards in ["group", "none"]:
                if num_generations > 1:
                    std_rewards = rewards.view(-1, num_generations).std(dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            else:
                raise ValueError(f"Invalid value for scale_rewards: {self.scale_rewards}")
            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
            return rewards, advantages

        if self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = nanstd(grouped, dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean_k)
            reward_k = (grouped - mean_k) / (std_k + 1e-4)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            advantages = (rewards - rewards.mean()) / (std_rewards + 1e-4)
            return rewards, advantages

        raise ValueError(f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}")

    def _run_cppo_round(
        self,
        expanded_inputs: list[dict[str, Any]],
        advantage_group_size: int,
        prune_style: str,
    ) -> dict[str, Any]:
        """Run one CPPO generation-scoring round and return only kept completions.

        Important: advantages are always computed on the full group first, then pruning
        decides which completions continue to loss computation.
        """
        if prune_style not in {"global", "per_group"}:
            raise ValueError(f"Unknown prune_style={prune_style}")

        prompts = [x["prompt"] for x in expanded_inputs]
        (
            prompt_ids_list,
            completion_ids_list,
            _tool_mask_list,
            completions,
            _num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts)

        if extra_fields:
            for i, inp in enumerate(expanded_inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        rewards_per_func_all = self._calculate_rewards(expanded_inputs, prompts, completions, completion_ids_list)
        _weighted_rewards_all, advantages_all = self._compute_advantages(rewards_per_func_all, advantage_group_size)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        rewards_per_func = rewards_per_func_all[process_slice]
        advantages = advantages_all[process_slice]

        if len(completion_ids_list) % advantage_group_size != 0:
            raise RuntimeError(
                "CPPO expected local completions divisible by group size. "
                f"got={len(completion_ids_list)} advantage_group_size={advantage_group_size}"
            )

        kept_prompt_ids: list[list[int]] = []
        kept_completion_ids: list[list[int]] = []
        kept_completions: list[Any] = []
        kept_prompt_texts: list[str] = []
        kept_completion_texts: list[str] = []
        kept_advantages: list[float] = []
        kept_abs_adv: list[float] = []
        kept_rewards_rows: list[torch.Tensor] = []
        kept_sampling_logps: list[list[float] | None] = []
        dropped_abs_adv: list[float] = []

        decoded_prompt_texts = [self.processing_class.decode(ids, skip_special_tokens=True) for ids in prompt_ids_list]
        decoded_completion_texts = [
            self.processing_class.decode(ids, skip_special_tokens=True) for ids in completion_ids_list
        ]

        def _append_idx(idx: int, adv_val: float, abs_val: float) -> None:
            kept_prompt_ids.append(prompt_ids_list[idx])
            kept_completion_ids.append(completion_ids_list[idx])
            kept_completions.append(completions[idx])
            kept_prompt_texts.append(decoded_prompt_texts[idx])
            kept_completion_texts.append(decoded_completion_texts[idx])
            kept_advantages.append(adv_val)
            kept_abs_adv.append(abs_val)
            kept_rewards_rows.append(rewards_per_func[idx].detach())
            if sampling_per_token_logps_list is not None:
                kept_sampling_logps.append(sampling_per_token_logps_list[idx])
            else:
                kept_sampling_logps.append(None)

        if prune_style == "per_group":
            n_groups = len(completion_ids_list) // advantage_group_size
            for g in range(n_groups):
                start = g * advantage_group_size
                end = start + advantage_group_size
                group_adv = advantages[start:end]
                abs_group_adv = group_adv.abs()
                keep_rel = select_cppo_keep_indices(abs_group_adv, self.cppo_keep_per_group, self.cppo_metric)
                keep_set = set(keep_rel)

                for rel in range(advantage_group_size):
                    idx = start + rel
                    abs_val = float(abs_group_adv[rel].item())
                    if rel in keep_set:
                        _append_idx(idx, float(group_adv[rel].item()), abs_val)
                    else:
                        dropped_abs_adv.append(abs_val)
        else:
            total = advantages.shape[0]
            drop_count = int(total * self.cppo_pruning)
            keep_count = max(total - drop_count, 1)
            keep_indices = select_cppo_keep_indices(advantages.abs(), keep_count, self.cppo_metric)
            keep_set = set(keep_indices)

            for idx in range(total):
                abs_val = float(advantages[idx].abs().item())
                if idx in keep_set:
                    _append_idx(idx, float(advantages[idx].item()), abs_val)
                else:
                    dropped_abs_adv.append(abs_val)

        return {
            "kept_prompt_ids": kept_prompt_ids,
            "kept_completion_ids": kept_completion_ids,
            "kept_completions": kept_completions,
            "kept_prompt_texts": kept_prompt_texts,
            "kept_completion_texts": kept_completion_texts,
            "kept_advantages": kept_advantages,
            "kept_abs_adv": kept_abs_adv,
            "kept_rewards_rows": kept_rewards_rows,
            "kept_sampling_logps": kept_sampling_logps,
            "dropped_abs_adv": dropped_abs_adv,
            "generated_count": len(completion_ids_list),
        }

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """Override GRPO generation path to inject CPPO pruning/allocation.

        The eval path is intentionally untouched to keep evaluation behavior consistent
        with base GRPOTrainer.
        """
        # Keep eval path unchanged.
        if not self.model.training:
            return super()._generate_and_score_completions(inputs)

        device = self.accelerator.device
        num_generations = self.num_generations

        if len(inputs) % num_generations != 0:
            raise RuntimeError(
                f"CPPO expected generation batch divisible by num_generations. "
                f"got={len(inputs)} num_generations={num_generations}"
            )

        all_kept_prompt_ids: list[list[int]] = []
        all_kept_completion_ids: list[list[int]] = []
        all_kept_completions: list[Any] = []
        all_kept_prompt_texts: list[str] = []
        all_kept_completion_texts: list[str] = []
        all_kept_advantages: list[float] = []
        all_kept_abs_adv: list[float] = []
        all_kept_rewards_rows: list[torch.Tensor] = []
        all_kept_sampling_logps: list[list[float] | None] = []
        all_dropped_abs_adv: list[float] = []
        generated_total = 0

        def _extend_round(round_out: dict[str, Any]) -> None:
            nonlocal generated_total
            generated_total += int(round_out["generated_count"])
            all_kept_prompt_ids.extend(round_out["kept_prompt_ids"])
            all_kept_completion_ids.extend(round_out["kept_completion_ids"])
            all_kept_completions.extend(round_out["kept_completions"])
            all_kept_prompt_texts.extend(round_out["kept_prompt_texts"])
            all_kept_completion_texts.extend(round_out["kept_completion_texts"])
            all_kept_advantages.extend(round_out["kept_advantages"])
            all_kept_abs_adv.extend(round_out["kept_abs_adv"])
            all_kept_rewards_rows.extend(round_out["kept_rewards_rows"])
            all_kept_sampling_logps.extend(round_out["kept_sampling_logps"])
            all_dropped_abs_adv.extend(round_out["dropped_abs_adv"])

        if self.cppo_strategy == CPPO_STRATEGY_AUTHOR_EXACT:
            if self.cppo_allocation:
                # Author-style allocation: keep trainer-facing group size at k, but compute advantages on full G.
                # Reconstruct unique groups of size `num_generations` (k), then presample each to full G.
                unique_inputs = [copy.deepcopy(inputs[i]) for i in range(0, len(inputs), num_generations)]
                expanded_inputs: list[dict[str, Any]] = []
                for picked in unique_inputs:
                    for _ in range(self.cppo_original_num_generations):
                        expanded_inputs.append(copy.deepcopy(picked))

                round_out = self._run_cppo_round(
                    expanded_inputs,
                    advantage_group_size=self.cppo_original_num_generations,
                    prune_style="global",
                )
                _extend_round(round_out)
                target_kept = len(inputs)
                if len(all_kept_advantages) > target_kept:
                    all_kept_prompt_ids = all_kept_prompt_ids[:target_kept]
                    all_kept_completion_ids = all_kept_completion_ids[:target_kept]
                    all_kept_completions = all_kept_completions[:target_kept]
                    all_kept_prompt_texts = all_kept_prompt_texts[:target_kept]
                    all_kept_completion_texts = all_kept_completion_texts[:target_kept]
                    all_kept_advantages = all_kept_advantages[:target_kept]
                    all_kept_abs_adv = all_kept_abs_adv[:target_kept]
                    all_kept_rewards_rows = all_kept_rewards_rows[:target_kept]
                    all_kept_sampling_logps = all_kept_sampling_logps[:target_kept]
            else:
                round_out = self._run_cppo_round(
                    inputs,
                    advantage_group_size=self.cppo_original_num_generations,
                    prune_style="global",
                )
                _extend_round(round_out)
                target_kept = len(all_kept_advantages)
        else:
            # Experimental refill strategy:
            # 1) prune per original group
            # 2) if allocation is enabled, keep sampling additional groups until the
            #    target kept budget is filled (or safety cap is hit).
            first_round = self._run_cppo_round(
                inputs,
                advantage_group_size=self.cppo_original_num_generations,
                prune_style="per_group",
            )
            _extend_round(first_round)

            target_kept = len(inputs) if self.cppo_allocation else len(all_kept_advantages)
            if self.cppo_allocation and target_kept > 0:
                unique_inputs = [
                    copy.deepcopy(inputs[i])
                    for i in range(0, len(inputs), self.cppo_original_num_generations)
                ]
                rounds = 0
                while len(all_kept_advantages) < target_kept:
                    rounds += 1
                    needed = target_kept - len(all_kept_advantages)
                    groups_needed = int(math.ceil(float(needed) / float(self.cppo_keep_per_group)))
                    expanded_inputs: list[dict[str, Any]] = []
                    for _ in range(groups_needed):
                        picked = copy.deepcopy(random.choice(unique_inputs))
                        for _ in range(self.cppo_original_num_generations):
                            expanded_inputs.append(copy.deepcopy(picked))

                    round_out = self._run_cppo_round(
                        expanded_inputs,
                        advantage_group_size=self.cppo_original_num_generations,
                        prune_style="per_group",
                    )
                    _extend_round(round_out)

                    if rounds > (self.cppo_presample_factor + 4):
                        break

                if len(all_kept_advantages) > target_kept:
                    all_kept_prompt_ids = all_kept_prompt_ids[:target_kept]
                    all_kept_completion_ids = all_kept_completion_ids[:target_kept]
                    all_kept_completions = all_kept_completions[:target_kept]
                    all_kept_prompt_texts = all_kept_prompt_texts[:target_kept]
                    all_kept_completion_texts = all_kept_completion_texts[:target_kept]
                    all_kept_advantages = all_kept_advantages[:target_kept]
                    all_kept_abs_adv = all_kept_abs_adv[:target_kept]
                    all_kept_rewards_rows = all_kept_rewards_rows[:target_kept]
                    all_kept_sampling_logps = all_kept_sampling_logps[:target_kept]

        if not all_kept_completion_ids:
            raise RuntimeError("CPPO pruning removed all samples in the batch.")

        prompt_tensors = [torch.tensor(ids, device=device) for ids in all_kept_prompt_ids]
        completion_tensors = [torch.tensor(ids, device=device) for ids in all_kept_completion_ids]
        prompt_mask_tensors = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_tensors]
        completion_mask_tensors = [torch.ones_like(ids, dtype=torch.long) for ids in completion_tensors]

        prompt_ids = pad(prompt_tensors, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask_tensors, padding_value=0, padding_side="left")
        completion_ids = pad(completion_tensors, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask_tensors, padding_value=0, padding_side="right")

        any_sampling_logps = any(x is not None for x in all_kept_sampling_logps)
        if any_sampling_logps:
            filled_sampling: list[torch.Tensor] = []
            for ids, logps in zip(all_kept_completion_ids, all_kept_sampling_logps, strict=True):
                if logps is None:
                    filled_sampling.append(torch.zeros(len(ids), dtype=torch.float32, device=device))
                else:
                    filled_sampling.append(torch.tensor(logps, dtype=torch.float32, device=device))
            sampling_per_token_logps = pad(filled_sampling, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        batch_size = self.args.per_device_train_batch_size
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                old_per_token_logps = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                    )
                else:
                    model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                        )
            else:
                ref_per_token_logps = None

        vllm_importance_sampling_ratio = None
        if (
            self.use_vllm
            and self.vllm_importance_sampling_correction
            and sampling_per_token_logps is not None
            and old_per_token_logps is not None
        ):
            sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
            mask = completion_mask
            per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

            if sequence_level_is:
                per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                logps_diff = per_sequence_logps_diff
            else:
                logps_diff = per_token_logps_diff

            vllm_importance_sampling_ratio = torch.exp(logps_diff)
            if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                vllm_importance_sampling_ratio = torch.clamp(
                    vllm_importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )
            elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                    vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap, value=0.0
                )

        mode = "train"
        rewards_per_func = torch.stack(all_kept_rewards_rows, dim=0)
        advantages_tensor = torch.tensor(all_kept_advantages, dtype=torch.float32, device=device)

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        rewards_for_log = rewards_per_func.nansum(dim=1)
        self._metrics[mode]["reward"].append(rewards_for_log.mean().item())
        self._metrics[mode]["reward_std"].append(rewards_for_log.std().item())
        self._metrics[mode]["frac_reward_zero_std"].append(0.0)

        kept_count = len(all_kept_advantages)
        generated_total = max(generated_total, 1)
        dropped_abs_adv = all_dropped_abs_adv if all_dropped_abs_adv else [0.0]
        self._metrics[mode]["cppo/pruning_ratio"].append(float(self.cppo_pruning))
        self._metrics[mode]["cppo/kept_per_group"].append(float(self.cppo_keep_per_group))
        self._metrics[mode]["cppo/kept_fraction"].append(float(kept_count) / float(generated_total))
        self._metrics[mode]["cppo/allocation_enabled"].append(1.0 if self.cppo_allocation else 0.0)
        self._metrics[mode]["cppo/presample_factor"].append(float(self.cppo_presample_factor))
        self._metrics[mode]["cppo/filled_fraction"].append(float(kept_count) / float(max(target_kept, 1)))
        self._metrics[mode]["cppo/abs_adv_mean_kept"].append(float(np.mean(all_kept_abs_adv)))
        self._metrics[mode]["cppo/abs_adv_mean_dropped"].append(float(np.mean(dropped_abs_adv)))
        self._metrics[mode]["cppo/author_exact_enabled"].append(
            1.0 if self.cppo_strategy == CPPO_STRATEGY_AUTHOR_EXACT else 0.0
        )

        self._logs["prompt"].extend(all_kept_prompt_texts)
        self._logs["completion"].extend(all_kept_completion_texts)
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(advantages_tensor.tolist())

        num_items_in_batch = torch.tensor(
            int(sum(len(x) for x in all_kept_completion_ids)),
            dtype=torch.long,
            device=device,
        )
        output: dict[str, Any] = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages_tensor,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if vllm_importance_sampling_ratio is not None:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if sampling_per_token_logps is not None:
            output["sampling_per_token_logps"] = sampling_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output
