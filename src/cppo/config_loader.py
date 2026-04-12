from __future__ import annotations

import copy
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping at top-level: {path}")
    return data


def _resolve_extends(path: Path) -> dict[str, Any]:
    cfg = _read_yaml(path)
    extends = cfg.pop("extends", None)
    if not extends:
        return cfg

    base_path = (path.parent / extends).resolve()
    base_cfg = _resolve_extends(base_path)
    return _deep_merge(base_cfg, cfg)


def _set_by_dotted_key(cfg: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    parts = dotted_key.split(".")
    cur: dict[str, Any] = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]

    value: Any = raw_value
    low = raw_value.lower()
    if low in {"true", "false"}:
        value = low == "true"
    else:
        try:
            if "." in raw_value:
                value = float(raw_value)
            else:
                value = int(raw_value)
        except ValueError:
            value = raw_value

    cur[parts[-1]] = value


def _infer_project_root(config_path: Path) -> Path:
    """Find repo root by walking upward until src/cppo exists."""
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / "src" / "cppo").exists():
            return candidate
    return Path.cwd().resolve()


def _resolve_path(base_dir: Path, value: str | Path) -> str:
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _resolve_config_paths(cfg: dict[str, Any], config_path: Path, project_root: Path) -> None:
    cfg.setdefault("paths", {})
    cfg["paths"]["project_root"] = str(project_root)

    base_dir = project_root
    run_cfg = cfg.setdefault("run", {})
    output_root = Path(str(run_cfg.get("output_root", "runs")))
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    run_cfg["output_root"] = str(output_root)

    data_cfg = cfg.setdefault("data", {})
    if "train_path" in data_cfg:
        data_cfg["train_path"] = _resolve_path(base_dir, str(data_cfg["train_path"]))
    split_map = data_cfg.get("eval_splits", {})
    if isinstance(split_map, dict):
        data_cfg["eval_splits"] = {
            str(split): _resolve_path(base_dir, str(path))
            for split, path in split_map.items()
        }


def _require(cfg: dict[str, Any], key: str) -> Any:
    parts = key.split(".")
    cur: Any = cfg
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            raise ValueError(f"Missing required config key: {key}")
        cur = cur[part]
    return cur


def _validate_cppo(cfg: dict[str, Any]) -> None:
    mode = _require(cfg, "rollout.mode")
    group_size = int(_require(cfg, "rollout.num_generations"))
    if group_size < 2:
        raise ValueError("rollout.num_generations must be >= 2")

    if mode != "cppo":
        return

    pruning = float(_require(cfg, "rollout.cppo.pruning"))
    metric = str(_require(cfg, "rollout.cppo.metric"))
    strategy = str(cfg.get("rollout", {}).get("cppo", {}).get("strategy", "author_exact"))
    if not (0.0 <= pruning < 1.0):
        raise ValueError(f"rollout.cppo.pruning must be in [0,1), got {pruning}")
    if metric not in {"smallest", "largest"}:
        raise ValueError(f"rollout.cppo.metric must be smallest|largest, got {metric}")
    if strategy not in {"author_exact", "experimental_refill"}:
        raise ValueError(
            "rollout.cppo.strategy must be author_exact|experimental_refill, "
            f"got {strategy}"
        )

    keep_float = group_size * (1.0 - pruning)
    keep = int(round(keep_float))
    if keep < 1:
        raise ValueError(
            f"CPPO pruning leaves no kept completions. group_size={group_size} pruning={pruning} keep={keep_float}"
        )
    if abs(keep_float - keep) > 1e-9:
        raise ValueError(
            "CPPO requires integer keep count: num_generations * (1-pruning) must be integer. "
            f"Got {keep_float}"
        )


def _validate_generation_math(cfg: dict[str, Any]) -> None:
    batch_size = int(_require(cfg, "training.batch_size"))
    world_size = int(_require(cfg, "training.world_size"))
    grad_acc = int(_require(cfg, "training.gradient_accumulation_steps"))
    num_g = int(_require(cfg, "rollout.num_generations"))

    global_batch = batch_size * world_size
    generation_batch = int(_require(cfg, "training.generation_batch_size"))

    if generation_batch % global_batch != 0:
        raise ValueError(
            f"training.generation_batch_size ({generation_batch}) must be divisible by global batch ({global_batch})"
        )
    if generation_batch % num_g != 0:
        raise ValueError(
            f"training.generation_batch_size ({generation_batch}) must be divisible by rollout.num_generations ({num_g})"
        )

    steps_per_generation = generation_batch // global_batch
    num_iterations = int(_require(cfg, "training.num_iterations"))
    generate_every = steps_per_generation * num_iterations
    if grad_acc % generate_every != 0:
        raise ValueError(
            "Unsafe GRPO/CPPO setup: gradient_accumulation_steps must be divisible by generate_every. "
            f"Got grad_acc={grad_acc}, generate_every={generate_every}"
        )


def _validate_env(cfg: dict[str, Any], project_root: Path) -> dict[str, str]:
    env_cfg = cfg.get("env", {})
    dotenv_required = bool(env_cfg.get("require_dotenv_file", False))
    dotenv_path = Path(str(env_cfg.get("dotenv_path", project_root / ".env")))
    if not dotenv_path.is_absolute():
        dotenv_path = (project_root / dotenv_path).resolve()
    if dotenv_required and not dotenv_path.exists():
        raise ValueError(
            f"Missing required .env at {dotenv_path}. Create it with required keys before training."
        )
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)

    env_map: dict[str, str] = {}

    wandb_cfg = cfg.get("integrations", {}).get("wandb", {})
    if bool(wandb_cfg.get("enabled", False)):
        key_name = str(wandb_cfg.get("api_key_env", "WANDB_API_KEY"))
        value = os.environ.get(key_name, "").strip()
        if not value:
            raise ValueError(
                f"W&B is enabled but env var {key_name} is missing/empty. Set it in .env or environment."
            )
        env_map[key_name] = value

    hub_cfg = cfg.get("integrations", {}).get("hf_hub", {})
    if bool(hub_cfg.get("enabled", False)) or bool(hub_cfg.get("push_to_hub", False)):
        token_name = str(hub_cfg.get("token_env", "HF_TOKEN"))
        token = os.environ.get(token_name, "").strip()
        if not token:
            raise ValueError(
                f"HF integration is enabled but env var {token_name} is missing/empty."
            )
        env_map[token_name] = token
    if bool(hub_cfg.get("push_to_hub", False)):
        hub_model_id = str(hub_cfg.get("hub_model_id", "")).strip()
        if not hub_model_id:
            raise ValueError("HF push_to_hub is enabled but integrations.hf_hub.hub_model_id is empty.")

    return env_map


def load_config(config_path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(config_path).resolve()
    project_root = _infer_project_root(path)

    cfg = _resolve_extends(path)
    overrides = overrides or []
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Use dotted.key=value")
        k, v = override.split("=", 1)
        _set_by_dotted_key(cfg, k.strip(), v.strip())

    _resolve_config_paths(cfg, path, project_root)

    run = cfg.setdefault("run", {})
    run.setdefault("seed", 42)
    run_id = run.get("id")
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run["id"] = run_id

    output_root = Path(str(run.get("output_root", "runs")))
    run_dir = (output_root / run_id).resolve()
    cfg["paths"]["run_dir"] = str(run_dir)

    model_cfg = cfg.setdefault("model", {})
    rollout_cfg = cfg.setdefault("rollout", {})
    train_cfg = cfg.setdefault("training", {})
    env_cfg = cfg.setdefault("env", {})
    env_cfg.setdefault("require_dotenv_file", True)
    env_cfg.setdefault("dotenv_path", str((project_root / ".env").resolve()))
    train_cfg.setdefault("world_size", int(os.environ.get("WORLD_SIZE", "1")))
    train_cfg.setdefault("num_iterations", 1)

    if "generation_batch_size" not in train_cfg:
        train_cfg["generation_batch_size"] = int(train_cfg.get("batch_size", 1)) * int(rollout_cfg.get("num_generations", 1))
    rollout_cfg.setdefault("cppo", {})
    rollout_cfg["cppo"].setdefault("strategy", "author_exact")
    if "vllm_max_model_length" not in rollout_cfg:
        rollout_cfg["vllm_max_model_length"] = int(model_cfg.get("max_prompt_length", 1024)) + int(model_cfg.get("max_completion_length", 1024))

    _validate_cppo(cfg)
    _validate_generation_math(cfg)
    _validate_env(cfg, project_root)

    return cfg


def dump_resolved_config(cfg: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
