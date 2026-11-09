#!/usr/bin/env bash
# 一键运行 triton-ascend 编译器能力缺口探针测试并输出摘要。
# 详细用例分析与预期结果参见 README.md。
set -u
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="${PYTHONPATH:-src:FlagGems/src}"
probes="tools/triton_ascend_probes"

run() {
  local name="$1"; shift
  echo "=== $name"
  timeout 600 python "$@" 2>&1 | grep -E "RESULT|SIGSEGV|SIGABRT|overflow|LLVM ERROR|different memory|pointer_cast|read before first write|Unsupported ptr|out operands|Unknown PIPE|Failed to bind|maxerr" | head -3
}

run "探针 01: dot   (L1 驻留 - 推荐写法)"    "$probes/probe_01_l1_resident_dot.py" dot
run "探针 01: copy  (deprecated copy API)"  "$probes/probe_01_l1_resident_dot.py" copy
run "探针 01: old   (旧写法，预期 FAIL)"     "$probes/probe_01_l1_resident_dot.py" old
run "探针 02: col     (UB 基线)"            "$probes/probe_02_slice_as_dot_operand.py" col
run "探针 02: row     (UB 基线)"            "$probes/probe_02_slice_as_dot_operand.py" row
run "探针 02: l1_col  (显式 L1 + 列切片)"    "$probes/probe_02_slice_as_dot_operand.py" l1_col
run "探针 02: l1_row  (显式 L1 + 行切片)"    "$probes/probe_02_slice_as_dot_operand.py" l1_row
run "探针 03: l1_copy (L1 subview 拼装)"     "$probes/probe_03_insert_slice_dot.py" l1_copy
run "探针 03: insert  (insert_slice，预期 FAIL)" "$probes/probe_03_insert_slice_dot.py" insert
run "探针 03: l1_insert (显式 L1，预期 FAIL)" "$probes/probe_03_insert_slice_dot.py" l1_insert
run "探针 04: noscope_sync (无 scope 同步)" "$probes/probe_04_scope_sync.py" noscope_sync
run "探针 04: scope (默认自动同步)"         "$probes/probe_04_scope_sync.py" scope
run "探针 04: scope (关闭自动同步)"         "$probes/probe_04_scope_sync.py" scope_noauto
run "探针 04: scope (显式手动同步)"         "$probes/probe_04_scope_sync.py" scope_sync
run "探针 05: (向量长链融合)"              "$probes/probe_05_vec_chain_fusion.py"
run "探针 06: (跨 chunk L1 预算)"          "$probes/probe_06_cross_chunk_l1_budget.py"

run "probe_07 plain (zN baseline)"   "$probes/probe_07_zn_staging_budget.py" plain
run "probe_07 l1a (zN UB budget)"    "$probes/probe_07_zn_staging_budget.py" l1a
run "probe_07 l1b (zN UB budget)"    "$probes/probe_07_zn_staging_budget.py" l1b
run "probe_07 l1  (zN UB budget)"    "$probes/probe_07_zn_staging_budget.py" l1
