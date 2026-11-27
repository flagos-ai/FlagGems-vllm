#!/usr/bin/env bash
# Build FlagCX from source against the active Python/torch interpreter.
#
# Platform hooks are centralized here so that adding or removing a platform is
# a small, reviewable change:
#
#   adaptor   platform       gflags/glog   platform-specific environment
#   du        Hygon/DTK      yes           CUDA_PATH/CUDA_HOME
#   nvidia    NVIDIA/CUDA    yes           CUDA_PATH/CUDA_HOME
#   metax     MetaX/MACA     no            DEVICE_HOME/CCL_HOME/USE_METAX
#   musa      MThreads/MUSA  yes           none
#
# To add a platform:
#   1. add the adaptor name to SUPPORTED_ADAPTORS;
#   2. add its probe command to ADAPTOR_PROBES (or require FLAGCX_ADAPTOR);
#   3. add a prepare_<adaptor> function and its case in prepare_platform.
# To remove a platform, remove those three entries and its prepare_* function.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="${FLAGCX_REPOSITORY:-https://github.com/flagos-ai/FlagCX.git}"
REVISION="${FLAGCX_REVISION:-v0.14.1}"
CACHE_DIR="${FLAGCX_SRC_CACHE:-${HOME}/.cache/flaggems/src/FlagCX}"
OUTPUT_DIR="${1:-${ROOT_DIR}/dist}"

SUPPORTED_ADAPTORS=(du nvidia metax musa)
ADAPTOR_PROBES=(
  "hy-smi:du"
  "nvidia-smi:nvidia"
  "mx-smi:metax"
  "mthreads-gmi:musa"
)

log() { echo "[flagcx-wrapper] $*"; }

die() {
  log "ERROR: $*"
  exit 1
}

check_headers() {
  local header missing=0
  for header in gflags/gflags.h glog/stl_logging.h; do
    if ! find /usr /usr/local /opt -path "*/include/${header}" -print -quit 2>/dev/null | grep -q .; then
      log "missing ${header}; install libgflags-dev and libgoogle-glog-dev"
      missing=1
    fi
  done
  return "${missing}"
}

detect_adaptor() {
  if [ -n "${FLAGCX_ADAPTOR:-}" ]; then
    echo "${FLAGCX_ADAPTOR}"
    return
  fi

  local probe command_name adaptor
  for probe in "${ADAPTOR_PROBES[@]}"; do
    command_name="${probe%%:*}"
    adaptor="${probe##*:}"
    if command -v "${command_name}" >/dev/null 2>&1; then
      echo "${adaptor}"
      return
    fi
  done

  die "cannot detect an adaptor; set FLAGCX_ADAPTOR to one of: ${SUPPORTED_ADAPTORS[*]}"
}

detect_cuda_path() {
  if [ -n "${CUDA_PATH:-}" ]; then
    echo "${CUDA_PATH}"
    return
  fi
  for candidate in /opt/dtk-*/cuda/cuda-12 /opt/dtk/cuda/cuda-12; do
    if [ -d "${candidate}" ]; then
      echo "${candidate}"
      return
    fi
  done
  if [ -n "${CUDA_HOME:-}" ]; then
    echo "${CUDA_HOME}"
  fi
}

prepare_cuda_platform() {
  local label="$1"
  local cuda_path
  log "${label}: checking gflags/glog headers"
  check_headers
  cuda_path="$(detect_cuda_path)"
  if [ -n "${cuda_path}" ]; then
    export CUDA_PATH="${cuda_path}"
    export CUDA_HOME="${cuda_path}"
    log "${label}: CUDA compatibility path=${cuda_path}"
  else
    log "${label}: no CUDA compatibility path found; leaving it to the toolchain"
  fi
}

prepare_du() {
  prepare_cuda_platform "Hygon/DTK"
}

prepare_nvidia() {
  prepare_cuda_platform "NVIDIA/CUDA"
}

prepare_metax() {
  log "MetaX/MACA: using MACA headers; gflags/glog preflight skipped"
  export DEVICE_HOME="${DEVICE_HOME:-/opt/maca}"
  export CCL_HOME="${CCL_HOME:-/opt/maca}"
  export USE_METAX=1
}

prepare_musa() {
  log "MThreads/MUSA: checking gflags/glog headers"
  check_headers
}

prepare_platform() {
  case "$1" in
    du) prepare_du ;;
    nvidia) prepare_nvidia ;;
    metax) prepare_metax ;;
    musa) prepare_musa ;;
    *) die "unsupported adaptor '$1'; supported: ${SUPPORTED_ADAPTORS[*]}" ;;
  esac
}

sync_source() {
  if [ -d "${CACHE_DIR}/.git" ]; then
    log "reusing ${CACHE_DIR}"
    git -C "${CACHE_DIR}" fetch --tags --quiet origin
    git -C "${CACHE_DIR}" checkout --quiet "${REVISION}"
  else
    rm -rf "${CACHE_DIR}"
    mkdir -p "$(dirname "${CACHE_DIR}")"
    log "cloning ${REPOSITORY} at ${REVISION}"
    git clone --quiet "${REPOSITORY}" "${CACHE_DIR}"
    git -C "${CACHE_DIR}" checkout --quiet "${REVISION}"
  fi

  local path missing=0
  while IFS= read -r path; do
    if [ ! -d "${CACHE_DIR}/${path}" ] || [ -z "$(ls -A "${CACHE_DIR}/${path}" 2>/dev/null)" ]; then
      missing=1
    fi
  done < <(git -C "${CACHE_DIR}" config -f .gitmodules --get-regexp path | awk '{print $2}')

  if [ "${missing}" -eq 1 ]; then
    git -C "${CACHE_DIR}" submodule update --init --recursive --depth 1 --quiet
  else
    log "submodules already populated; skipping network update"
  fi
}

main() {
  local adaptor
  adaptor="$(detect_adaptor)"
  prepare_platform "${adaptor}"
  export FLAGCX_ADAPTOR="${adaptor}"
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0

  sync_source
  mkdir -p "${OUTPUT_DIR}"
  log "building FlagCX adaptor=${adaptor} torch=$(python3 -c 'import torch; print(torch.__version__)')"
  python3 -m pip wheel "${CACHE_DIR}" --no-build-isolation --no-deps -w "${OUTPUT_DIR}"
  ls -lh "${OUTPUT_DIR}"/flagcx-*.whl
}

main "$@"
