#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

ok()   { printf " ${GREEN}[OK]${NC}\n"; }
fail() { printf " ${RED}[FAILED]${NC}\n"; exit 1; }
warn() { printf " ${YELLOW}[WARN]${NC} %s\n" "$1"; }

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-.venv}"
MIRROR="${PIP_INDEX_URL:-https://pypi.org/simple}"

VLLM_VERSION="${VLLM_VERSION:-0.20.2}"
TORCH_BACKEND="${TORCH_BACKEND:-auto}"

FLAGOS_PYPI="${FLAGOS_PYPI:-https://resource.flagos.net/repository/flagos-pypi-hosted/simple}"
FLAGTREE_VERSION="${FLAGTREE_VERSION:-0.6.0rc1}"

# Whether to replace Triton with FlagTree.
# For temporary container validation, you can run:
#   USE_FLAGTREE=0 bash setup.sh
USE_FLAGTREE="${USE_FLAGTREE:-1}"

# Whether to require `import triton` immediately after installing FlagTree.
# Current FlagTree wheels may fail here in NVIDIA CUDA containers because they try
# to load non-NVIDIA backend plugins such as metaxTritonPlugin.so.
STRICT_TRITON_IMPORT="${STRICT_TRITON_IMPORT:-0}"

# Whether `uv pip check` failure should fail the whole setup.
# FlagTree may provide the `triton` Python module but not satisfy package metadata
# requirements such as "vllm requires triton".
STRICT_PIP_CHECK="${STRICT_PIP_CHECK:-0}"

CUDA_HOME="${CUDA_HOME:-}"

setup_cuda_env() {
  local candidate

  if [ -z "${CUDA_HOME}" ]; then
    for candidate in /usr/local/cuda /usr/local/cuda-*; do
      if [ -x "${candidate}/bin/nvcc" ]; then
        CUDA_HOME="${candidate}"
        break
      fi
    done
  fi

  if [ -n "${CUDA_HOME}" ]; then
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    if [ -d "${CUDA_HOME}/lib64" ]; then
      if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
      else
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64"
      fi
    fi
  fi
}

# ── Check we're in the repo root ─────────────────────────────
if [ ! -f "pyproject.toml" ] || [ ! -d "src/flaggems_vllm" ]; then
  echo "Error: run this script from the FlagGems-vllm repo root."
  echo "Current directory: $(pwd)"
  echo "Expected:"
  echo "  pyproject.toml"
  echo "  src/flaggems_vllm/"
  echo
  echo "Current files:"
  ls -la
  exit 1
fi

setup_cuda_env

# ── Basic system preflight ───────────────────────────────────
printf "Checking basic commands ..."
for cmd in curl git gcc g++ make; do
  if ! command -v "$cmd" &>/dev/null; then
    printf "\n"
    warn "$cmd not found; build may fail"
  fi
done
ok

# ── Check NVIDIA GPU ─────────────────────────────────────────
printf "Checking NVIDIA GPU ..."
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"
  printf " %s" "${GPU_NAME:-unknown}"
  ok
else
  warn "nvidia-smi not found; CUDA tests will not work"
fi

# ── Check nvcc ───────────────────────────────────────────────
printf "Checking nvcc ..."
if command -v nvcc &>/dev/null; then
  NVCC_VERSION="$(nvcc --version | tail -1 || true)"
  printf " %s" "${NVCC_VERSION:-unknown}"
  if [ -n "${CUDA_HOME}" ]; then
    printf " (CUDA_HOME=%s)" "${CUDA_HOME}"
  fi
  ok
else
  warn "nvcc not found; set CUDA_HOME to your CUDA toolkit path if the compiler is installed. Source builds requiring CUDA compiler may fail"
fi

# ── Detect or install uv ─────────────────────────────────────
printf "Checking uv ..."
if command -v uv &>/dev/null; then
  printf " %s" "$(uv --version)"
  ok
else
  printf " not found, installing ...\n"
  command -v curl &>/dev/null || {
    echo "Error: curl is required to install uv."
    exit 1
  }

  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"

  command -v uv &>/dev/null || { printf "uv installation"; fail; }
  printf "  Installed %s" "$(uv --version)"
  ok
fi

# ── Install Python via uv ────────────────────────────────────
printf "Installing Python ${PYTHON_VERSION} ..."
uv python install "${PYTHON_VERSION}" \
  --python-preference only-managed \
  -q \
  || fail
ok

# ── Create clean virtual environment ─────────────────────────
printf "Creating virtual environment (${VENV_DIR}) ..."
uv venv "${VENV_DIR}" \
  --python "${PYTHON_VERSION}" \
  --python-preference only-managed \
  --seed \
  -q \
  || fail
ok

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

printf "Python: %s" "$(python --version)"
ok

# ── Constraints ──────────────────────────────────────────────
CONSTRAINTS_FILE="${VENV_DIR}/constraints.txt"

cat > "${CONSTRAINTS_FILE}" <<'EOF'
setuptools>=64,<82
cmake>=3.20,<4
EOF

# ── Install build tools ──────────────────────────────────────
printf "Installing build tools ..."
uv pip install -q \
  "setuptools>=64,<82" \
  "wheel" \
  "packaging" \
  "scikit-build-core==0.12.2" \
  "pybind11==3.0.3" \
  "cmake>=3.20,<4" \
  "ninja==1.13.0" \
  "PyYAML>=6.0" \
  --constraint "${CONSTRAINTS_FILE}" \
  --default-index "${MIRROR}" \
  || fail
ok

# ── Install vLLM and matching PyTorch backend ────────────────
printf "Installing vLLM==${VLLM_VERSION} with torch backend=${TORCH_BACKEND} ..."
uv pip install -q \
  "vllm==${VLLM_VERSION}" \
  --torch-backend="${TORCH_BACKEND}" \
  --constraint "${CONSTRAINTS_FILE}" \
  --default-index "${MIRROR}" \
  || fail
ok

# ── Verify torch/vLLM before Triton replacement ──────────────
printf "Checking torch/vLLM ..."
python - <<'PY' || fail
import torch
import vllm

print(" torch:", torch.__version__)
print(" torch cuda:", torch.version.cuda)
print(" cuda available:", torch.cuda.is_available())
print(" vllm:", vllm.__version__)
PY
ok

# ── Remove Triton and install FlagTree ───────────────────────
if [ "${USE_FLAGTREE}" = "1" ]; then
  printf "Removing Triton packages ..."
  TRITON_PKGS="$(uv pip list 2>/dev/null | awk 'tolower($1) ~ /^triton/ {print $1}' || true)"

  if [ -n "${TRITON_PKGS}" ]; then
    echo "${TRITON_PKGS}" | xargs -r uv pip uninstall -q || true
  fi
  ok

  printf "Installing FlagTree==${FLAGTREE_VERSION} ..."
  uv pip install -q \
    "flagtree===${FLAGTREE_VERSION}" \
    --index "${FLAGOS_PYPI}" \
    --default-index "${MIRROR}" \
    || fail
  ok

  # Do not force `import triton` by default here.
  # Some FlagTree wheels may try to load vendor plugins during import.
  printf "Checking FlagTree package ..."
  uv pip show flagtree >/tmp/flagtree_show.log || fail
  cat /tmp/flagtree_show.log
  ok

  if [ "${STRICT_TRITON_IMPORT}" = "1" ]; then
    printf "Strictly checking Triton/FlagTree import ..."
    python - <<'PY' || fail
import triton
print(" triton module:", triton.__file__)
print(" triton version:", getattr(triton, "__version__", "unknown"))
PY
    ok
  else
    warn "Skipping strict 'import triton' check; set STRICT_TRITON_IMPORT=1 to enable it"
  fi
else
  warn "Skipping FlagTree installation; keeping Triton installed by vLLM/PyTorch"
fi

# ── Install FlagGems-vllm ────────────────────────────────────
printf "Installing FlagGems-vllm editable ..."
uv pip install -q \
  --no-build-isolation \
  -e ".[test]" \
  --constraint "${CONSTRAINTS_FILE}" \
  --default-index "${MIRROR}" \
  || fail
ok

# ── Verify installation ──────────────────────────────────────
printf "Verifying imports ..."
python - <<'PY' || fail
import torch
import vllm
import flaggems_vllm

print(" torch:", torch.__version__)
print(" torch cuda:", torch.version.cuda)
print(" cuda available:", torch.cuda.is_available())
print(" vllm:", vllm.__version__)
print(" flaggems_vllm device:", getattr(flaggems_vllm, "device", "unknown"))
PY
ok

# ── Optional Triton import check ─────────────────────────────
if [ "${STRICT_TRITON_IMPORT}" = "1" ]; then
  printf "Strictly verifying Triton import ..."
  python - <<'PY' || fail
import triton
print(" triton module:", triton.__file__)
print(" triton version:", getattr(triton, "__version__", "unknown"))
PY
  ok
else
  printf "Checking Triton import non-strictly ..."
  if python - <<'PY'
import triton
print(" triton module:", triton.__file__)
print(" triton version:", getattr(triton, "__version__", "unknown"))
PY
  then
    ok
  else
    printf "\n"
    warn "Triton import failed, but STRICT_TRITON_IMPORT=0 so setup continues"
  fi
fi

# ── Dependency check ─────────────────────────────────────────
printf "Running dependency check ..."
if uv pip check; then
  ok
else
  printf "\n"
  if [ "${STRICT_PIP_CHECK}" = "1" ]; then
    fail
  else
    warn "uv pip check failed, but STRICT_PIP_CHECK=0 so setup continues"
  fi
fi

# ── Run smoke test ───────────────────────────────────────────
printf "Collecting tests ..."
COLLECT_LOG="/tmp/flaggems_vllm_collect.log"

python -m pytest tests --collect-only -q > "${COLLECT_LOG}" 2>&1 || {
  printf "\n"
  cat "${COLLECT_LOG}"
  fail
}

tail -1 "${COLLECT_LOG}"
ok

printf "\n${GREEN}FlagGems-vllm setup complete.${NC}\n"
printf "Activate with: source %s/bin/activate\n" "${VENV_DIR}"
printf "\nQuick validation:\n"
printf "  pytest tests --collect-only\n"
printf "  pytest tests -q --quick\n"
printf "  pytest benchmark --collect-only\n"
