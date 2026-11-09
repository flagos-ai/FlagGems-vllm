#!/usr/bin/env bash
# Run every triton-ascend compiler-gap probe and print a one-line summary.
# Expected outcomes are documented in README.md.
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="${PYTHONPATH:-src:FlagGems/src}"
probes="tools/triton_ascend_probes"

run() {
  local name="$1"; shift
  echo "=== $name"
  timeout 600 python "$@" 2>&1 | grep -E "RESULT|SIGSEGV|SIGABRT|overflow|LLVM ERROR|different memory|pointer_cast|read before first write|Unsupported ptr|maxerr" | head -3
}

run "probe_01 copy (L1 residency)"   "$probes/probe_01_l1_resident_dot.py" copy
run "probe_01 dot  (L1 residency)"   "$probes/probe_01_l1_resident_dot.py" dot
run "probe_02 col (slice -> dot)"    "$probes/probe_02_slice_as_dot_operand.py" col
run "probe_02 row (slice -> dot)"    "$probes/probe_02_slice_as_dot_operand.py" row
run "probe_03 (insert_slice -> dot)" "$probes/probe_03_insert_slice_dot.py"
run "probe_04 noscope_sync"          "$probes/probe_04_scope_sync.py" noscope_sync
run "probe_04 scope (default)"       "$probes/probe_04_scope_sync.py" scope
run "probe_04 scope (no auto sync)"  "$probes/probe_04_scope_sync.py" scope_noauto
run "probe_04 scope + manual sync"   "$probes/probe_04_scope_sync.py" scope_sync
run "probe_05 (vec chain fusion)"    "$probes/probe_05_vec_chain_fusion.py"
run "probe_06 (2-chunk L1 budget)"   "$probes/probe_06_cross_chunk_l1_budget.py"
