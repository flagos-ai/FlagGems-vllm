#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if (( $# == 0 )); then
  cat >&2 <<'USAGE'
Usage: bash tools/run_metax_einsum.sh PYTHON_ARGS...

Run inside an existing MetaX environment. Optional environment variables:
  METAX_EINSUM_PYTHON          Python interpreter (default: python3)
  METAX_DEEP_GEMM_ROOT         Directory containing the vendor deep_gemm package
  METAX_EINSUM_GPU             Physical GPU index or comma-separated indices
  METAX_EINSUM_EXPECTED_SHA256 Expected SHA256 of the vendor einsum.py

Example:
  METAX_EINSUM_PYTHON=/opt/conda/bin/python \
  METAX_DEEP_GEMM_ROOT=/path/to/vendor-packages \
  METAX_EINSUM_GPU=0 \
  bash tools/run_metax_einsum.sh -m pytest -q tests/test_int8_einsum.py --quick
USAGE
  exit 2
fi

EINSUM_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EINSUM_PYTHON=${METAX_EINSUM_PYTHON:-python3}
EINSUM_PATH="$EINSUM_REPO_ROOT/tools/metax_einsum_compat:$EINSUM_REPO_ROOT/src:$EINSUM_REPO_ROOT"

if [[ -n ${METAX_DEEP_GEMM_ROOT:-} ]]; then
  if [[ ! -d $METAX_DEEP_GEMM_ROOT/deep_gemm ]]; then
    echo "METAX_DEEP_GEMM_ROOT must contain the vendor deep_gemm package." >&2
    exit 2
  fi
  EINSUM_PATH="$METAX_DEEP_GEMM_ROOT:$EINSUM_PATH"
fi

export PYTHONPATH="$EINSUM_PATH${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n ${METAX_EINSUM_GPU:-} ]]; then
  if [[ ! $METAX_EINSUM_GPU =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "METAX_EINSUM_GPU must contain physical GPU indices, such as 0 or 0,2." >&2
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES="$METAX_EINSUM_GPU"
  export MACA_VISIBLE_DEVICES="$METAX_EINSUM_GPU"
  export METAX_VISIBLE_DEVICES="$METAX_EINSUM_GPU"
fi

"$EINSUM_PYTHON" - <<'PY'
import hashlib
import importlib
import os
from pathlib import Path

import torch
import triton

module = importlib.import_module("deep_gemm.einsum")
source = Path(module.__file__).resolve()
digest = hashlib.sha256(source.read_bytes()).hexdigest()
expected = os.environ.get("METAX_EINSUM_EXPECTED_SHA256")
if expected and digest != expected:
    raise SystemExit(f"Vendor einsum SHA256 mismatch: expected {expected}, got {digest}")
print(f"Vendor BF16 baseline: {source}")
print(f"Vendor einsum SHA256: {digest}")
print(f"Torch: {torch.__version__}; Triton: {triton.__version__} ({triton.__file__})")
PY

cd -- "$EINSUM_REPO_ROOT"
exec "$EINSUM_PYTHON" "$@"
