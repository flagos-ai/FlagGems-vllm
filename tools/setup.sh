#!/usr/bin/env bash

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# `uv`'s --torch-backend=auto probes the local CUDA driver version and maps
# it to one of uv's own known backend tags (see `uv pip install --help`).
# On H20 runners the driver reports CUDA 13.3, which is newer than any tag
# uv 0.11.22/0.13.1 know about (highest is cu132); `auto` then silently
# drops torch/torchvision/torchaudio from the install instead of erroring
# or falling back, so `import torch` fails right after a "successful" vLLM
# install. Pin the exact vendor-qualified torch build FlagGems itself uses
# for this backend (see FlagGems' src/flag_gems/backends.yaml
# `nvidia-cuda133` entry) instead of relying on that probe.
TORCH_BACKEND="${TORCH_BACKEND:-auto}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0+cu130}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0+cu130}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0+cu130}"
FLAG_GEMS_VERSION="${FLAG_GEMS_VERSION:-}"

FLAGOS_PYPI="${FLAGOS_PYPI:-https://resource.flagos.net/repository/flagos-pypi-hosted/simple}"
# Vendor-qualified wheels (the `+cu130`-suffixed torch builds below) are only
# published under the per-vendor FlagOS index, not the shared "hosted" one.
FLAGOS_PYPI_NVIDIA="${FLAGOS_PYPI_NVIDIA:-https://resource.flagos.net/repository/flagos-pypi-nvidia/simple}"
# Matches FlagGems' nvidia-cuda133 backend (src/flag_gems/backends.yaml):
# older FlagTree builds are missing/broken Hopper warp-specialize (TLE)
# support that some ops here require (e.g. AttributeError on
# `buffered_tensor.reshape`, WGMMA type/shape assertion failures).
FLAGTREE_VERSION="${FLAGTREE_VERSION:-0.7.0}"

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

# Whether to pass `--refresh` to the vLLM/FlagGems/FlagGems-vllm `uv pip
# install` calls below. On a persistent (non-ephemeral) runner, `uv`'s global
# cache survives across jobs; if a prior job's cache entries went stale or
# were partially evicted, `uv` can report a cached install as successful
# without actually materializing the package files into the freshly
# recreated venv (e.g. `import torch` failing right after a "successful"
# vLLM install). `--refresh` forces revalidation of cache entries instead of
# trusting their metadata. Off by default since it slows down every install;
# CI for persistent runners should set UV_REFRESH=1.
UV_REFRESH="${UV_REFRESH:-0}"
UV_REFRESH_FLAG=()
if [ "${UV_REFRESH}" = "1" ]; then
  UV_REFRESH_FLAG=(--refresh)
fi

CUDA_HOME="${CUDA_HOME:-}"

print_help() {
  cat <<EOF
Usage: $(basename "$0") [-h|--help]

Bootstraps a GPU-ready dev environment for FlagGems-vllm: creates a uv-managed
virtual environment, installs vLLM + a matching PyTorch build, installs
FlagGems and FlagGems-vllm (editable), optionally swaps Triton for FlagTree,
and runs a test-collection smoke check.

This script takes no positional arguments; all configuration is done via
environment variables (each shown below with its default):

  PYTHON_VERSION=${PYTHON_VERSION}            Python version to install via uv
  VENV_DIR=${VENV_DIR}                  Target virtual environment directory
  PIP_INDEX_URL=${MIRROR}
                               Default package index (used for most installs)

  VLLM_VERSION=${VLLM_VERSION}             vLLM version to install
  TORCH_VERSION=${TORCH_VERSION}      Pinned vendor-qualified torch version,
                               installed from FLAGOS_PYPI_NVIDIA before vLLM.
                               Set to "" to fall back to --torch-backend
                               probing instead (see TORCH_BACKEND below).
  TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}  Matching torchaudio version
  TORCHVISION_VERSION=${TORCHVISION_VERSION} Matching torchvision version
  TORCH_BACKEND=${TORCH_BACKEND}               Torch backend passed to --torch-backend
                               (e.g. auto, cpu, cu121, cu124, ...); only used
                               when TORCH_VERSION=""
  FLAG_GEMS_VERSION=          FlagGems version to pin; empty installs latest

  FLAGOS_PYPI_NVIDIA=${FLAGOS_PYPI_NVIDIA}
                               Vendor-qualified index that publishes the
                               +cu130-suffixed torch/torchaudio/torchvision
                               wheels pinned above
  FLAGOS_PYPI=${FLAGOS_PYPI}
                               Index used for FlagTree
  FLAGTREE_VERSION=${FLAGTREE_VERSION}            FlagTree version to install

  USE_FLAGTREE=${USE_FLAGTREE}                   Replace Triton with FlagTree (1) or keep
                               the Triton installed by vLLM/PyTorch (0)
  STRICT_TRITON_IMPORT=${STRICT_TRITON_IMPORT}       Fail setup if \`import triton\` fails (1) or
                               only warn (0)
  STRICT_PIP_CHECK=${STRICT_PIP_CHECK}          Fail setup if \`uv pip check\` reports
                               conflicts (1) or only warn (0)

  UV_REFRESH=${UV_REFRESH}                     Pass --refresh to the vLLM/FlagGems/
                               FlagGems-vllm \`uv pip install\` calls, forcing
                               revalidation of uv's cache instead of trusting
                               it. Recommended on persistent (non-ephemeral)
                               runners where a stale cache can make \`uv\`
                               report a cached install as successful without
                               actually installing into the fresh venv.

  CUDA_HOME=                  CUDA toolkit path; auto-detected from
                               /usr/local/cuda* if unset
  USE_FLAGCX=OFF               Disable flagcx's communication path at
                               runtime, matching FlagGems' nvidia-cuda133
                               backend env

Example:
  USE_FLAGTREE=0 VLLM_VERSION=0.20.2 bash tools/setup.sh
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $arg" >&2
      echo "Run with -h or --help for usage." >&2
      exit 1
      ;;
  esac
done

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

  # Matches FlagGems' nvidia-cuda133 backend env (src/flag_gems/backends.yaml):
  # flagcx is installed as a dependency (some packages expect it importable)
  # but its communication path is disabled at runtime for this backend.
  export USE_FLAGCX="${USE_FLAGCX:-OFF}"
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
UV_VERSION="0.11.22"
UV_MIRROR="https://resource.flagos.net/repository/flagos-filestore/utils"

printf "Checking uv ..."
export PATH="${HOME}/.local/bin:$PATH"
if command -v uv &>/dev/null; then
  printf " $(uv --version)"
  ok
else
  printf " not found, installing ...\n"
  ARCH=$(uname -m)
  mkdir -p "$HOME/.local/bin"
  curl -sSf "${UV_MIRROR}/uv-${ARCH}-${UV_VERSION}-linux-gnu.tar.gz" \
    | tar xz -C "$HOME/.local/bin" 2>/dev/null \
    || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
  command -v uv &>/dev/null || { printf "uv installation"; fail; }
  printf "Installed $(uv --version)"
  ok
fi
# Persist PATH for subsequent GitHub Actions steps
[ -n "${GITHUB_PATH:-}" ] && echo "$HOME/.local/bin" >> "$GITHUB_PATH"

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

cat > "${CONSTRAINTS_FILE}" <<EOF
setuptools>=64,<82
cmake>=3.20,<4
torch==${TORCH_VERSION}
torchaudio==${TORCHAUDIO_VERSION}
torchvision==${TORCHVISION_VERSION}
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
# Install the pinned, vendor-qualified torch stack first, from the index that
# actually publishes it, so vLLM's resolver finds it already satisfied and
# never falls through to --torch-backend's driver-version probing (see the
# TORCH_VERSION comment above for why that probe is unsafe on this runner).
# Set TORCH_VERSION="" to opt back into --torch-backend="${TORCH_BACKEND}"
# probing instead (e.g. on a host where the pinned +cu130 build is wrong).
if [ -n "${TORCH_VERSION}" ]; then
  printf "Installing torch==${TORCH_VERSION} (torchaudio==${TORCHAUDIO_VERSION}, torchvision==${TORCHVISION_VERSION}) ...\n"
  # Root cause (found via -v on the persistent h20 runner): `uv` discovers
  # this repo's pyproject.toml and treats it as a workspace root, so this
  # install gets resolved against the workspace's *declared* dependency set
  # (and reconciled like a sync: it even logs the venv's existing build
  # tools as "Unnecessary package") rather than treated as a plain,
  # standalone `pip install`. Since this repo's pyproject.toml does not list
  # torch, uv's resolution never actually schedules it for installation —
  # no error, exit 0, hence the "successful" install that leaves torch
  # missing. --no-sources (tried first) only covers the `tool.uv.sources`
  # table and did not stop this; --no-config stops uv from discovering
  # pyproject.toml/uv.toml at all, so there is no workspace to resolve
  # against in the first place.
  uv pip install -v --no-config \
    "torch==${TORCH_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --default-index "${FLAGOS_PYPI_NVIDIA}" \
    "${UV_REFRESH_FLAG[@]}" \
    || fail
  ok

  # No --index FLAGOS_PYPI_NVIDIA here: torch is already installed and
  # satisfies the constraint above, so vLLM's resolver never needs to look
  # for it in any index. Adding that index anyway pulls in unrelated
  # packages it also hosts (e.g. a "vllm<=0.20.2+flagos" listing) that can
  # shadow the plain PyPI vllm==0.20.2 release under uv's default
  # first-index-wins resolution, making the install unsatisfiable.
  printf "Installing vLLM==${VLLM_VERSION} ...\n"
  uv pip install \
    "vllm==${VLLM_VERSION}" \
    --constraint "${CONSTRAINTS_FILE}" \
    --default-index "${MIRROR}" \
    "${UV_REFRESH_FLAG[@]}" \
    || fail
  ok
else
  printf "Installing vLLM==${VLLM_VERSION} with torch backend=${TORCH_BACKEND} ...\n"
  uv pip install \
    "vllm==${VLLM_VERSION}" \
    --torch-backend="${TORCH_BACKEND}" \
    --constraint "${CONSTRAINTS_FILE}" \
    --default-index "${MIRROR}" \
    "${UV_REFRESH_FLAG[@]}" \
    || fail
  ok
fi

# ── Verify torch/vLLM before Triton replacement ──────────────
dump_installed_and_fail() {
  echo "---- uv pip list (debug dump on failure) ----"
  uv pip list
  fail
}

printf "Checking torch/vLLM ..."
python - <<'PY' || dump_installed_and_fail
import torch
import vllm

print(" torch:", torch.__version__)
print(" torch cuda:", torch.version.cuda)
print(" cuda available:", torch.cuda.is_available())
print(" vllm:", vllm.__version__)
PY
ok

# ── Install FlagGems (required dependency) ───────────────────
# Must run after vLLM/torch are installed: FlagGems's packaging does not
# declare torch as a dependency even though `import flag_gems` requires it,
# so installing it first could leave a torch-less, unimportable flag_gems.
printf "Installing FlagGems${FLAG_GEMS_VERSION:+==${FLAG_GEMS_VERSION}} ..."
uv pip install -q \
  "flag-gems${FLAG_GEMS_VERSION:+==${FLAG_GEMS_VERSION}}" \
  --constraint "${CONSTRAINTS_FILE}" \
  --default-index "${MIRROR}" \
  "${UV_REFRESH_FLAG[@]}" \
  || fail
ok

printf "Verifying FlagGems import ..."
python - <<'PY' || fail
import flag_gems

print(" flag_gems:", getattr(flag_gems, "__version__", "unknown"))
print(" flag_gems path:", flag_gems.__file__)
assert hasattr(flag_gems, "enable"), "flag_gems.enable is missing"
assert hasattr(flag_gems, "use_gems"), "flag_gems.use_gems is missing"
PY
ok

# ── Install FlagGems-vllm before Triton replacement ──────────
printf "Installing FlagGems-vllm editable ..."
uv pip install -q \
  --no-build-isolation \
  -e ".[test]" \
  --constraint "${CONSTRAINTS_FILE}" \
  --default-index "${MIRROR}" \
  "${UV_REFRESH_FLAG[@]}" \
  || fail
ok

# ── Remove Triton and install FlagTree ───────────────────────
if [ "${USE_FLAGTREE}" = "1" ]; then
  printf "Removing Triton packages ..."
  TRITON_UNINSTALL_ATTEMPTS=0
  while python -m pip show triton >/dev/null 2>&1; do
    python -m pip uninstall -y triton >/dev/null || fail
    TRITON_UNINSTALL_ATTEMPTS=$((TRITON_UNINSTALL_ATTEMPTS + 1))
    if [ "${TRITON_UNINSTALL_ATTEMPTS}" -ge 10 ]; then
      fail
    fi
  done
  ok

  printf "Installing FlagTree==${FLAGTREE_VERSION} ..."
  python -m pip install -q \
    "flagtree===${FLAGTREE_VERSION}" \
    --index-url "${FLAGOS_PYPI}" \
    || fail
  ok

  # Do not force `import triton` by default here.
  # Some FlagTree wheels may try to load vendor plugins during import.
  printf "Checking FlagTree package ..."
  python -m pip show flagtree >/tmp/flagtree_show.log || fail
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

# ── Verify installation ──────────────────────────────────────
printf "Verifying imports ..."
python - <<'PY' || fail
import flag_gems
import torch
import vllm

import flaggems_vllm

print(" torch:", torch.__version__)
print(" torch cuda:", torch.version.cuda)
print(" cuda available:", torch.cuda.is_available())
print(" vllm:", vllm.__version__)
print(" flag_gems:", getattr(flag_gems, "__version__", "unknown"))
print(" flaggems_vllm device:", getattr(flaggems_vllm, "device", "unknown"))
PY
ok

if [ "${USE_FLAGTREE}" = "1" ]; then
  printf "Checking FlagTree TLE support ..."
  python - <<'PY' || fail
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

print(" has_triton_tle(3, 6, 0):", has_triton_tle(3, 6, 0))
if not has_triton_tle(3, 6, 0):
    raise SystemExit("FlagTree TLE support is unavailable")
PY
  ok
fi

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
