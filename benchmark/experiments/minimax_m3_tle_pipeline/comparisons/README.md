# What changed relative to draft #979

The parent is
[`fead47f`](https://github.com/flagos-ai/FlagGems-vllm/commit/fead47fdc6d4b81cc73b0d6488834dce75d95bf1).
The new draft adds only `benchmark/experiments/minimax_m3_tle_pipeline/`.
It leaves `src/flaggems_vllm/ops/minimax_m3_sparse_attn.py` byte-identical to the
parent. Use the pinned parent-to-experiment commit comparison in the PR body
to review this incremental change: the upstream `main` diff also includes the
unmerged parent PR.

For comparing actual implementations, read the following in order:

1. [#979 operator -> frozen measured serving baseline](pr979-to-measured-baseline.diff).
   This exposes API/packaging differences and the later sparse-wide arm, keeping
   those separate from the TLE rewrite. The
   [AST comparison](pr979-baseline.json) records the five common unchanged JITs
   and the three additional serving JITs. These facts do not imply equivalence
   of the two host APIs.
2. [Frozen measured baseline -> experimental adapter](measured-baseline-to-tle-adapter.diff).
   This shows the conditional launch replacement, explicit ring/WS/conversion
   choices and preserved dispatch/fallback coverage in the tested harness.
3. [WS/producer-conversion kernels](../kernels/tle_pipeline.py) and
   [non-WS full-row-TMA kernel](../kernels/tle_nows.py).
   These supply the TMA/mbarrier/WGMMA implementation called by the adapter.

The `.diff` files are complete textual source comparisons with descriptive
paths, not patches to apply to the installed operator. The exact measured
workspace is available through `../reproduce.py --prepare PATH`; final kernel
copies are reformatted for review with function AST equivalence checked.
The first comparison uses zero context to avoid whitespace-only context lines.

Unlike the original FlagAttention #85 implementation, this experiment keeps
BF16 QK/PV and the frozen compiler's scaled-C PV accumulation order. Neither
the attention selection window nor TopK nor compute precision was relaxed.
The results compare full `attn_auto` calls against the frozen serving baseline;
they do not claim a direct apples-to-apples benchmark of #85 or #979's API.
