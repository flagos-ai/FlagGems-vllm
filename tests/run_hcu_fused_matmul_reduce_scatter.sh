#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# BW/SHCA launcher. Run on every node with the same master and distinct NODE_RANK.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

unset CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1}"
export FLAGTREE_BACKEND=hcu USE_FLAGCX=ON
export LD_LIBRARY_PATH="/opt/shca-userspace${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export IBV_DRIVERS="${IBV_DRIVERS:-/opt/shca-userspace/libshca}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=bond1}"
export FLAGCX_IB_HCA="${FLAGCX_IB_HCA:-shca_0}"
export FLAGCX_USE_HETERO_COMM=1 FLAGCX_MEM_ENABLE=1
export FLAGCX_VMM_ENABLE=0 FLAGCX_P2P_DISABLE=0
export AMDGCN_USE_BUFFER_OPS=0
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p /var/log/shca

exec python -m torch.distributed.run \
  --nnodes="${NNODES:-1}" \
  --node-rank="${NODE_RANK:-0}" \
  --nproc-per-node="${NPROC_PER_NODE:-2}" \
  --rdzv-backend=static \
  --master-addr="${MASTER_ADDR:-127.0.0.1}" \
  --master-port="${MASTER_PORT:-29501}" \
  tests/run_fused_matmul_reduce_scatter.py "$@"
