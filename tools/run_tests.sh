#!/usr/bin/env bash
set -euo pipefail

VENDOR="${1:-nvidia}"
CHANGED_FILES_FILE="${CHANGED_FILES_FILE:-}"

export DNN_VENDOR="${VENDOR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

echo "===================================================="
echo "Running FlagGems-vllm smoke tests"
echo "Vendor: ${DNN_VENDOR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "===================================================="

python -c "import torch, flaggems_vllm; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('vendor:', flaggems_vllm.vendor_name); print('device:', flaggems_vllm.device); assert torch.cuda.is_available(), 'CUDA is required for these smoke tests'"

selection="$(python tools/select_tests.py \
  --changed-files "${CHANGED_FILES_FILE}" \
  --format shell)"
eval "${selection}"

echo "Test selection mode: ${TEST_SELECTION_MODE}"
if [[ -n "${SELECTED_TESTS}" ]]; then
  echo "Selected tests: ${SELECTED_TESTS}"
fi
if [[ -n "${SELECTED_BENCHMARKS}" ]]; then
  echo "Selected benchmarks: ${SELECTED_BENCHMARKS}"
fi

if [[ "${TEST_SELECTION_MODE}" == "skip" || -z "${SELECTED_TESTS}${SELECTED_BENCHMARKS}" ]]; then
  echo "No tests or benchmarks selected."
  exit 0
fi

if [[ -n "${SELECTED_TESTS}" ]]; then
  # shellcheck disable=SC2086
  pytest -q ${SELECTED_TESTS} --quick
fi

if [[ -n "${SELECTED_BENCHMARKS}" ]]; then
  # shellcheck disable=SC2086
  pytest -q ${SELECTED_BENCHMARKS} --level core
fi
