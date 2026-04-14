from __future__ import annotations

"""Manual checkpoint evaluation entrypoint with pass@k profile support."""

import argparse
import json
import os
import sys
from pathlib import Path


def _parse_splits(raw: str | None) -> list[str]:
    """Parse comma-separated split names from CLI."""
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def main() -> None:
    """Run eval for selected splits and write summary/details artifacts."""
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and save pass@k artifacts")
    parser.add_argument("--config", default="config.yaml", help="YAML config path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or model id")
    parser.add_argument("--splits", default="", help="Comma-separated eval split names (default: primary split)")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split problem cap")
    parser.add_argument("--n-generations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--out-dir", default="", help="Optional output root for eval outputs")
    parser.add_argument(
        "--cleanup-merged-model",
        choices=["true", "false"],
        default=None,
        help="If set, override eval.adapter_merge.cleanup_after_eval for this command.",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override dotted.key=value")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from cppo.config_loader import load_config
    from cppo.eval import (
        build_eval_profiles,
        cleanup_merged_eval_model,
        combine_eval_profile_summaries,
        evaluate_checkpoint,
        save_eval_outputs,
    )

    cfg = load_config(args.config, overrides=args.overrides)
    eval_cfg = cfg.get("eval", {})
    prompt_cfg = cfg.get("prompt", {})
    default_system_prompt = str(prompt_cfg.get("system_prompt", ""))
    per_split_prompt = prompt_cfg.get("eval_system_prompt_by_split", {})
    split_map = cfg.get("data", {}).get("eval_splits", {})
    if not isinstance(split_map, dict) or not split_map:
        raise RuntimeError("No data.eval_splits configured.")

    requested_splits = _parse_splits(args.splits)
    if not requested_splits:
        on_checkpoint_splits = eval_cfg.get("on_checkpoint", {}).get("splits", [])
        if isinstance(on_checkpoint_splits, list) and on_checkpoint_splits:
            requested_splits = [str(s).strip() for s in on_checkpoint_splits if str(s).strip()]
        else:
            primary = str(eval_cfg.get("primary_split", "")).strip()
            requested_splits = [primary] if primary else sorted(split_map.keys())

    hub_cfg = cfg.get("integrations", {}).get("hf_hub", {})
    token_env = str(hub_cfg.get("token_env", "HF_TOKEN"))
    hf_token = os.environ.get(token_env, "")

    n_generations = int(args.n_generations if args.n_generations is not None else eval_cfg.get("n_generations", 1))
    batch_size = int(args.batch_size if args.batch_size is not None else eval_cfg.get("batch_size", 8))
    temperature = float(args.temperature if args.temperature is not None else eval_cfg.get("temperature", 0.6))
    top_p = float(args.top_p if args.top_p is not None else eval_cfg.get("top_p", 1.0))
    max_new_tokens = int(args.max_new_tokens if args.max_new_tokens is not None else eval_cfg.get("max_new_tokens", 1024))
    truncation_retry_cfg = eval_cfg.get("truncation_retry", {})
    rollout_cfg = cfg.get("rollout", {})
    use_server_eval = bool(rollout_cfg.get("use_vllm", True)) and str(rollout_cfg.get("vllm_mode", "")).strip().lower() == "server"
    server_base_url = str(rollout_cfg.get("vllm_server_base_url", "")).strip()
    if not server_base_url:
        use_server_eval = False
    server_timeout = float(rollout_cfg.get("vllm_server_timeout", 240.0))

    ckpt = str(Path(args.checkpoint).resolve()) if Path(args.checkpoint).exists() else args.checkpoint
    root_out = Path(args.out_dir).resolve() if args.out_dir else Path(ckpt) / "passk"
    root_out.mkdir(parents=True, exist_ok=True)

    aggregate: dict[str, object] = {
        "checkpoint": ckpt,
        "splits": {},
    }
    user_forced_single_profile = (
        args.n_generations is not None
        or args.temperature is not None
        or args.top_p is not None
    )
    profile_list = build_eval_profiles(eval_cfg)
    multi_profile = len(profile_list) > 1 and not user_forced_single_profile
    adapter_merge_cfg = eval_cfg.get("adapter_merge", {})
    cleanup_after_eval = bool(adapter_merge_cfg.get("cleanup_after_eval", True))
    if args.cleanup_merged_model is not None:
        cleanup_after_eval = str(args.cleanup_merged_model).strip().lower() == "true"

    for split in requested_splits:
        if split not in split_map:
            print(f"[warn] skip unknown split: {split}")
            continue
        split_path = Path(str(split_map[split]))
        if not split_path.exists():
            print(f"[warn] skip missing file for split={split}: {split_path}")
            continue

        if user_forced_single_profile:
            profiles: list[dict[str, object]] = [
                {
                    "name": "manual",
                    "n_generations": int(n_generations),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "report_k": int(n_generations),
                }
            ]
        else:
            profiles = profile_list

        profile_runs: list[tuple[dict[str, object], dict[str, object]]] = []
        for profile in profiles:
            summary, details = evaluate_checkpoint(
                ckpt=ckpt,
                split_name=split,
                split_path=str(split_path),
                system_prompt=str(per_split_prompt.get(split, default_system_prompt)),
                hf_token=hf_token,
                n_generations=int(profile["n_generations"]),
                batch_size=batch_size,
                temperature=float(profile["temperature"]),
                top_p=float(profile["top_p"]),
                max_new_tokens=max_new_tokens,
                report_k=int(profile.get("report_k")) if profile.get("report_k") is not None else None,
                limit=int(args.limit if args.limit is not None else 0),
                evaluator_cfg=eval_cfg.get("evaluator", {}),
                truncation_retry_enabled=bool(truncation_retry_cfg.get("enabled", False)),
                truncation_retry_max_retries=int(truncation_retry_cfg.get("max_retries", 0)),
                truncation_retry_max_new_tokens=int(truncation_retry_cfg.get("retry_max_new_tokens", max_new_tokens)),
                use_vllm_server=bool(use_server_eval),
                vllm_server_base_url=server_base_url,
                vllm_server_timeout=server_timeout,
            )
            out_dir = root_out / split
            if multi_profile:
                out_dir = out_dir / str(profile["name"])
            save_eval_outputs(summary, details, out_dir)
            profile_runs.append((profile, summary))

        merged_summary = combine_eval_profile_summaries(split, profile_runs)
        aggregate["splits"][split] = merged_summary
        split_root = root_out / split
        split_root.mkdir(parents=True, exist_ok=True)
        with (split_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(merged_summary, f, ensure_ascii=False, indent=2)
        pass_keys = sorted([k for k in merged_summary.keys() if isinstance(k, str) and k.startswith("pass@")])
        pass_bits = " ".join(f"{k}={float(merged_summary.get(k, 0.0)):.4f}" for k in pass_keys)
        print(f"[ok] split={split} {pass_bits} out={split_root}")

    with (root_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    if cleanup_after_eval:
        cleanup_merged_eval_model(ckpt)
    print(f"[done] wrote aggregate summary: {root_out / 'summary.json'}")


if __name__ == "__main__":
    main()
