#!/usr/bin/env bash
set -euo pipefail

VENDOR="${1:-nvidia}"

export DNN_VENDOR="${VENDOR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

echo "===================================================="
echo "Running FlagGems-vllm smoke tests"
echo "Vendor: ${DNN_VENDOR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "===================================================="

python -c "import torch, flaggems_vllm; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('vendor:', flaggems_vllm.vendor_name); print('device:', flaggems_vllm.device); assert torch.cuda.is_available(), 'CUDA is required for these smoke tests'"

pytest -q \
  tests/test_outer.py \
  tests/test_bincount.py \
  tests/test_silu_and_mul.py \
  tests/test_moe_align_block_size.py \
  --quick
