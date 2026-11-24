#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${HC_PYTHON:-python}"
if [[ -n "${HC_SITE:-}" ]]; then export PYTHONPATH="$HC_SITE:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"; else export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"; fi
export TASK_QUEUE_ENABLE=0
OUT="$(mktemp -d "$PWD/hc-merged-results.XXXXXX")"
status=0
"$PYTHON" -c 'import torch,torch_npu,triton,flag_gems,flaggems_vllm; from triton.experimental.tle.language.dsa import tile_load; assert torch.npu.is_available(); print("Triton:",triton.__file__); print("FlagGems-vllm:",flaggems_vllm.__file__); print("HC:",flaggems_vllm.hc_collapse.__module__)' 2>&1 | tee "$OUT/preflight.log"
"$PYTHON" -m pytest -q tests/test_hc_collapse.py::test_hc_collapse 2>&1 | tee "$OUT/correctness.log" || status=1
"$PYTHON" tools/hc_collapse/run_all.py --out "$OUT" "$@" 2>&1 | tee "$OUT/benchmark.log" || status=1
git diff -- src/flaggems_vllm/ops/hc_collapse.py src/flaggems_vllm/runtime/backend/_ascend/ops/hc_collapse.py > "$OUT/implementation.diff"
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "Results: $OUT.tar.gz"
exit "$status"
