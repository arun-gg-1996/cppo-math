#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
REQ_FILE="${ROOT_DIR}/requirements.txt"

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

echo
echo "Environment ready."
python -V
python - <<'PY'
import importlib.metadata as md
for name in ["torch","transformers","trl","peft","accelerate","vllm","datasets","sympy","wandb"]:
    print(f"{name}=={md.version(name)}")
PY
