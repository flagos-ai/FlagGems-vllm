#!/usr/bin/env bash
# Build FlagCX from source against the interpreter running this script.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="${FLAGCX_REPOSITORY:-https://github.com/flagos-ai/FlagCX.git}"
REVISION="${FLAGCX_REVISION:-v0.14.1}"
CACHE_DIR="${FLAGCX_SRC_CACHE:-${HOME}/.cache/flaggems/src/FlagCX}"
OUTPUT_DIR="${1:-${ROOT_DIR}/dist}"
log() { echo "[flagcx-wrapper] $*"; }
check_headers() {
  local missing=0
  for header in gflags/gflags.h glog/stl_logging.h; do
    if ! find /usr /usr/local /opt -path "*/include/${header}" -print -quit 2>/dev/null | grep -q .; then
      log "missing ${header}; install libgflags-dev and libgoogle-glog-dev"
      missing=1
    fi
  done
  return "${missing}"
}
detect_adaptor() {
  if [ -n "${FLAGCX_ADAPTOR:-}" ]; then echo "${FLAGCX_ADAPTOR}"; return; fi
  if command -v hy-smi >/dev/null 2>&1; then echo du
  elif command -v nvidia-smi >/dev/null 2>&1; then echo nvidia
  elif command -v mx-smi >/dev/null 2>&1; then echo metax
  elif command -v mthreads-gmi >/dev/null 2>&1; then echo musa
  else log "cannot detect adaptor; set FLAGCX_ADAPTOR"; return 1; fi
}
detect_cuda_path() {
  if [ -n "${CUDA_PATH:-}" ]; then echo "${CUDA_PATH}"; return; fi
  for candidate in /opt/dtk-*/cuda/cuda-12 /opt/dtk/cuda/cuda-12; do
    if [ -d "${candidate}" ]; then echo "${candidate}"; return; fi
  done
  if [ -n "${CUDA_HOME:-}" ]; then echo "${CUDA_HOME}"; fi
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
  git -C "${CACHE_DIR}" submodule update --init --recursive --quiet
}
main() {
  check_headers
  local adaptor cuda_path
  adaptor="$(detect_adaptor)"
  cuda_path="$(detect_cuda_path)"
  sync_source
  mkdir -p "${OUTPUT_DIR}"
  export FLAGCX_ADAPTOR="${adaptor}"
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  if [ -n "${cuda_path}" ]; then
    export CUDA_PATH="${cuda_path}"
    export CUDA_HOME="${cuda_path}"
  fi
  log "building adaptor=${adaptor} torch=$(python3 -c 'import torch; print(torch.__version__)')"
  python3 -m pip wheel "${CACHE_DIR}" --no-build-isolation --no-deps -w "${OUTPUT_DIR}"
  ls -lh "${OUTPUT_DIR}"/flagcx-*.whl
}
main "$@"
