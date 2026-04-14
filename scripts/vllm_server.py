from __future__ import annotations

"""Launch TRL-managed vLLM server using project YAML configuration."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cppo.config_loader import load_config  # noqa: E402


def _build_command(cfg: dict) -> list[str]:
    """Build `trl vllm-serve` command from rollout/model config."""
    rollout_cfg = cfg.get("rollout", {})
    model_cfg = cfg.get("model", {})

    if not bool(rollout_cfg.get("use_vllm", False)):
        raise ValueError("rollout.use_vllm=false. Set it true to launch vLLM server.")
    if str(rollout_cfg.get("vllm_mode", "server")) != "server":
        raise ValueError(
            "rollout.vllm_mode must be 'server' for split GPU launch. "
            f"Got {rollout_cfg.get('vllm_mode')!r}."
        )

    model_name = str(model_cfg.get("model_name_or_path", "")).strip()
    if not model_name:
        raise ValueError("model.model_name_or_path is required.")

    host = str(rollout_cfg.get("vllm_server_host", "127.0.0.1"))
    port = int(rollout_cfg.get("vllm_server_port", 8000))
    gpu_util = float(rollout_cfg.get("vllm_gpu_memory_utilization", 0.7))
    max_model_len = int(
        rollout_cfg.get(
            "vllm_max_model_length",
            int(model_cfg.get("max_prompt_length", 1024)) + int(model_cfg.get("max_completion_length", 1024)),
        )
    )
    dtype = str(rollout_cfg.get("vllm_dtype", "auto"))

    cmd = [
        "trl",
        "vllm-serve",
        "--model",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_util),
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        dtype,
    ]
    return cmd


def main() -> None:
    """CLI wrapper to print/launch vLLM server command."""
    parser = argparse.ArgumentParser(description="Launch TRL vLLM server from YAML config (split GPU mode).")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config key with dotted.key=value (repeatable)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command and exit")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config, overrides=args.overrides)
    cmd = _build_command(cfg)

    pretty = " ".join(shlex.quote(x) for x in cmd)
    print(pretty)
    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
