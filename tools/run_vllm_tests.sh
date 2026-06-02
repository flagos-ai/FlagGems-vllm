#!/usr/bin/env bash
set -euo pipefail

VENDOR="${1:-nvidia}"

export DNN_VENDOR="${VENDOR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export VLLM_CONFIGURE_LOGGING="${VLLM_CONFIGURE_LOGGING:-0}"

echo "===================================================="
echo "Running FlagGems-vllm vLLM comparison tests"
echo "Vendor: ${DNN_VENDOR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "===================================================="

python -c "import torch, vllm, flaggems_vllm; print('torch:', torch.__version__); print('vllm:', vllm.__version__); print('cuda available:', torch.cuda.is_available()); print('vendor:', flaggems_vllm.vendor_name); print('device:', flaggems_vllm.device); assert torch.cuda.is_available(), 'CUDA is required for vLLM comparison tests'"

pytest -q \
  tests/test_grouped_topk.py \
  tests/test_topk_softmax.py \
  --quick

pytest -q \
  tests/test_topk_softplus_sqrt.py -k "test_topk_softplus_sqrt_vs_vllm" \
  --quick

pytest -q \
  tests/test_top_k_per_row_decode.py -k "test_top_k_per_row_decode_vs_vllm" \
  --quick
