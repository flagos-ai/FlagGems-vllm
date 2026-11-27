#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
minimax_python="${M3_PYTHON:-python3}"
minimax_prefix="src:."
if [[ -n "${M3_SITE:-}" ]]; then
    test -d "$M3_SITE"
    minimax_prefix="$minimax_prefix:$M3_SITE"
fi
export PYTHONPATH="$minimax_prefix${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    exec "$minimax_python" -m tools.minimax_qknorm_rope_cache.run_all --help
fi

export MINIMAX_DEVICE=0
minimax_expect_device=false
for minimax_arg in "$@"; do
    if $minimax_expect_device; then
        MINIMAX_DEVICE="$minimax_arg"
        minimax_expect_device=false
    elif [[ "$minimax_arg" == --device ]]; then
        minimax_expect_device=true
    elif [[ "$minimax_arg" == --device=* ]]; then
        MINIMAX_DEVICE="${minimax_arg#--device=}"
    elif [[ "$minimax_arg" == --out || "$minimax_arg" == --out=* ]]; then
        printf '%s\n' 'run_all.sh creates its own output directory' >&2
        exit 2
    fi
done
if ! [[ "$MINIMAX_DEVICE" =~ ^[0-9]+$ ]] || $minimax_expect_device; then
    printf '%s\n' '--device requires an NPU index' >&2
    exit 2
fi

minimax_dir="$(mktemp -d "${M3_RESULT_PARENT:-$PWD}/minimax-results.XXXXXX")"
minimax_status=0
"$minimax_python" -c '
import torch
import torch_npu
import triton
import flaggems_vllm

if flaggems_vllm.vendor_name != "ascend" or not torch.npu.is_available():
    raise RuntimeError("An available Ascend NPU and FlagTree runtime are required")
print(torch.__version__, torch_npu.__version__, triton.__version__)
print(triton.__file__, flaggems_vllm.__file__)
' 2>&1 | tee "$minimax_dir/preflight.log" || minimax_status=$?
if [[ "$minimax_status" == 0 ]]; then
    "$minimax_python" -m pytest -q tests/test_minimax_qknorm_rope_cache.py \
        2>&1 | tee "$minimax_dir/correctness.log" || minimax_status=$?
fi
if [[ "$minimax_status" == 0 ]]; then
    "$minimax_python" -m tools.minimax_qknorm_rope_cache.run_all \
        "$@" --out "$minimax_dir" 2>&1 | tee "$minimax_dir/benchmark.log" || minimax_status=$?
fi
minimax_parent="$(dirname "$minimax_dir")"
minimax_name="$(basename "$minimax_dir")"
tar -czf "$minimax_dir.tar.gz" -C "$minimax_parent" "$minimax_name"
printf 'UPLOAD=%s.tar.gz\n' "$minimax_dir"
exit "$minimax_status"
