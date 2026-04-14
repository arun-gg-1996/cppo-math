#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
REQ_FILE="${ROOT_DIR}/requirements.txt"
ENV_FILE="${ROOT_DIR}/.env"

PY_BIN="${PYTHON_BIN:-python3.10}"

if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PY_BIN} not found."
  echo "Install Python 3.10 and rerun, or set PYTHON_BIN explicitly."
  exit 1
fi

PY_VER="$(${PY_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PY_VER}" != "3.10" ]]; then
  echo "ERROR: Python 3.10 required, found ${PY_VER} from ${PY_BIN}"
  exit 1
fi

echo "Using ${PY_BIN} (${PY_VER})"
echo "Creating/updating venv at ${VENV_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PY_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQ_FILE}"

prompt_required_secret() {
  local key_name="$1"
  local value=""
  while [[ -z "${value}" ]]; do
    read -r -s -p "Enter ${key_name}: " value
    printf "\n" >&2
    if [[ -z "${value}" ]]; then
      echo "ERROR: ${key_name} is required." >&2
    fi
  done
  printf "%s" "${value}"
}

upsert_env_key() {
  local key="$1"
  local value="$2"
  local tmp_file
  tmp_file="$(mktemp)"

  awk -v key="${key}" -v value="${value}" '
    BEGIN { updated = 0 }
    $0 ~ "^[[:space:]]*" key "=" {
      if (!updated) {
        print key "=" value
        updated = 1
      }
      next
    }
    { print }
    END {
      if (!updated) {
        print key "=" value
      }
    }
  ' "${ENV_FILE}" > "${tmp_file}"

  mv "${tmp_file}" "${ENV_FILE}"
}

touch "${ENV_FILE}"

echo
echo "Token setup (required):"
WANDB_API_KEY="$(prompt_required_secret "WANDB_API_KEY")"
HF_TOKEN="$(prompt_required_secret "HF_TOKEN")"

upsert_env_key "WANDB_API_KEY" "${WANDB_API_KEY}"
upsert_env_key "HF_TOKEN" "${HF_TOKEN}"

if ! command -v wandb >/dev/null 2>&1; then
  echo "ERROR: wandb CLI not found after dependency install."
  exit 1
fi
if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: hf CLI not found after dependency install."
  exit 1
fi

echo
echo "Running CLI logins..."
wandb login "${WANDB_API_KEY}"
hf auth login --token "${HF_TOKEN}"

echo
echo "Environment ready."
echo "Updated env file: ${ENV_FILE}"
python -V
python - <<'PY'
import importlib.metadata as md
for name in ["torch","transformers","trl","peft","accelerate","vllm","datasets","sympy","wandb"]:
    print(f"{name}=={md.version(name)}")
PY

echo
echo "Ready to run:"
cat <<'EOF'
source .venv/bin/activate

# Split-GPU smoke
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/vllm_server.py --config configs/smoke/server_gpu_split_smoke.yaml
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py --config configs/smoke/server_gpu_split_smoke.yaml

# Full split-GPU run
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/vllm_server.py --config config.yaml
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py --config config.yaml
EOF
