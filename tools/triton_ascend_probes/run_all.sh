#!/usr/bin/env bash
# Run every triton-ascend probe and print the one-line expectation.
#
# Layout:
#   supported/  capabilities that already work on stock triton-ascend
#               (reference only -- NOT requirements)
#   gaps/       capabilities that are missing/broken and that AscendC does
#               implement (these are the requirements)
#   defects/    crashes/regressions with a supported workaround
#               (bug reports, not requirements)
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="${PYTHONPATH:-src:FlagGems/src}"
P="tools/triton_ascend_probes"

run() {
  local name="$1"; shift
  echo "=== $name"
  timeout 600 python "$@" 2>&1 \
    | grep -E "RESULT|SIGSEGV|SIGABRT|overflow|LLVM ERROR|different memory|pointer_cast|read before first write|Unsupported ptr|maxerr" \
    | head -3
}

echo "################ supported (expect PASS) ################"
run "s1 slice -> dot (col)"        "$P/supported/s1_slice_dot.py" col
run "s1 slice -> dot (row)"        "$P/supported/s1_slice_dot.py" row
run "s1 slice -> dot (l1_col)"     "$P/supported/s1_slice_dot.py" l1_col
run "s1 slice -> dot (l1_row)"     "$P/supported/s1_slice_dot.py" l1_row
run "s2 L1-resident dot"           "$P/supported/s2_l1_resident_dot.py" dot
run "s2 L1 copy alias"             "$P/supported/s2_l1_resident_dot.py" copy
run "s3 L1 subview assemble"       "$P/supported/s3_l1_subview_assemble.py"
run "s4 manual cross-core sync"    "$P/supported/s4_manual_cross_core_sync.py"
run "s6 scoped CV hand-off"        "$P/supported/s6_scoped_cv_handoff.py"

echo "################ gaps (expect FAIL: these are requirements) ################"
run "g1 nd->zN baseline (plain)"   "$P/gaps/g1_nd2nz_staging.py" plain
run "g1 nd->zN l1a (A panels)"     "$P/gaps/g1_nd2nz_staging.py" l1a
run "g1 nd->zN l1b (B panels)"     "$P/gaps/g1_nd2nz_staging.py" l1b
run "g1 nd->zN l1  (all panels)"   "$P/gaps/g1_nd2nz_staging.py" l1
run "g2 vec chain fusion"          "$P/gaps/g2_vec_chain_fusion.py"
run "g3 pipelined UB row shift"   "$P/gaps/g3_pipelined_ub_view_shift.py"

echo "################ defects (expect FAIL; workaround exists) ################"
run "d1 scope (default)"           "$P/defects/d1_scope_auto_sync.py" scope
run "d1 scope (no auto sync)"      "$P/defects/d1_scope_auto_sync.py" scope_noauto
run "d2 insert_slice -> dot"       "$P/defects/d2_insert_slice_dot.py" insert
run "d2 insert_slice -> L1 dot"    "$P/defects/d2_insert_slice_dot.py" l1_insert
run "d3 2D ND L1 view"             "$P/defects/d3_l1_view_2d.py" nd
run "d3 L1 target_shape view"      "$P/defects/d3_l1_view_2d.py" target
run "d4 single-read UB shift"      "$P/defects/d4_extract_slice_pipeline.py"
run "d5 noauto multidot"            "$P/defects/d5_noauto_multidot.py"
