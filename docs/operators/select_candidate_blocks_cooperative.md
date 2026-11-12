<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# select_candidate_blocks_cooperative

## Operator delivery summary

- **Op:** vLLM `vllm.models.deepseek_v41.nvidia.specialized_decode.attention.candidate_topk.select_candidate_blocks_cooperative`
  / FlagGems `flaggems_vllm.select_candidate_blocks_cooperative`
- **Status:** complete (implemented, tested and measured on NVIDIA H20)
- **Backend:** NVIDIA (Triton / FlagTree); library-only API, not registered into
  PyTorch dispatch.

## Description

DeepSeek-V4.1 decode candidate-block selection. For each decode row it scores
every block of the padded logits row by its maximum, pins the block holding the
newest token, selects the top-`k` blocks over the row's valid prefix, and writes
their ids (`-1` padded, unsorted) into `out`. Consumers treat each row as a set.

vLLM's own operator is a thin wrapper over a separate top-k op: two Triton
kernels (block scores, finalize) plus the CUDA `cooperative_topk`. This port
keeps that layering:

1. `_block_scores_kernel` (Triton) — per-block NaN-propagating max; pins the
   newest block to `+inf`; writes each row's valid block count and pre-fills
   `out` with `-1`. Autotuned (`select_candidate_blocks_cooperative_scores`).
2. Row-prefix top-k — this repo's Triton `persistent_topk` (preferred; currently
   the faster of the two Triton backends on NVIDIA), falling back to
   `cooperative_topk`. vLLM's indexer treats the two as interchangeable backends.
3. `_finalize_candidates_kernel` (Triton) — drops out-of-range / `-inf` ids to
   `-1`. Only needed for `cooperative_topk`; skipped for `persistent_topk`, which
   already emits only valid ids and `-1` padding (proven across the test matrix).

## Specification

- **dtype:** `logits` fp32, `row_ends` int32, `out` int32.
- **Shapes (`supports_cooperative_candidates`):** `0 < rows <= 64`,
  `k in (512, 1024, 2048)`, `nblocks = cdiv(width, block_size)` divisible by 4.
- **Output rules:** `out[rows, k]` candidate block ids, `-1` padded, unsorted.
  The newest block is always kept. Selection matches vLLM up to ties at the
  selection boundary (the vLLM docstring already declares these approximate).
- **Layout:** any non-negative `logits` strides; `row_ends` and `out` contiguous.
- **Unsupported paths** raise `NotImplementedError`; there is no torch compute
  fallback.

## Paths

- **general:** score kernel → `persistent_topk` → (finalize only for the
  `cooperative_topk` fallback).
- **early return:** empty width writes `out = -1` and returns.
- **external-library:** none.
- **unsupported:** shapes/dtypes outside the matrix above raise.

## No torch compute fallback

Pass. Production code uses only Triton kernels, `flag_gems`/`flaggems_vllm` ops
(`persistent_topk` / `cooperative_topk`), and memory/metadata ops
(`torch.empty`, shape/stride/dtype reads). No `torch` compute/cast op is used to
produce results.

## Autotune

- `select_candidate_blocks_cooperative_scores` — the score kernel's `TILE`
  (32..1024) and `num_warps` (4/8), keyed on `nblocks`, `BLOCK_SIZE`, `K`.
- The finalize kernel has a fixed launch and an explicit **no-autotune
  exemption**: it is a single launch of at most `k` elements per row and is
  skipped entirely on the preferred `persistent_topk` path.

## Tests

```bash
pytest -v -s tests/test_select_candidate_blocks_cooperative.py
```

On H20: **35 passed, 12 skipped**. Covered: parity with a torch reference across
rows/lengths/k and two data distributions; a port of vLLM's
`test_cooperative_candidates_match_stock` (incl. CUDA-graph replay); NaN,
non-contiguous strides, non-power-of-two block size, empty/short-row `-1`
padding, unsupported-path errors, and a no-torch-compute static audit. The 12
skips are the direct vs-vLLM comparisons, skipped when the pinned vLLM build
lacks the native op.

## Benchmark

```bash
pytest -q benchmark/test_select_candidate_blocks_cooperative.py --level core
```

Baseline is vLLM's native `select_candidate_blocks_cooperative` when available,
otherwise a vectorized torch reference.

Measured on H20, non-graph (eager) CUDA-event timing, median -- matching the
graph-disabled methodology of the `baseline/` native collectors.
`SpeedUp = latency_native / latency_gems`. "this op" is this PR on current
`main`; "+dispatch" additionally applies the companion `persistent_topk`
sampled-dispatch change.

| rows | seq_len | k    | native (ms) | this op (ms) | SpeedUp | +dispatch (ms) | SpeedUp |
| ---: | ------: | ---: | ----------: | -----------: | ------: | -------------: | ------: |
| 1    | 4096    | 2048 | 0.065       | 0.119        | 0.55x   | 0.085          | 0.77x   |
| 16   | 65536   | 2048 | 0.064       | 0.123        | 0.52x   | 0.084          | 0.76x   |
| 16   | 131072  | 512  | 0.064       | 0.128        | 0.50x   | 0.086          | 0.74x   |
| 32   | 131072  | 2048 | 0.079       | 0.150        | 0.52x   | 0.092          | 0.85x   |
| 64   | 16384   | 2048 | 0.099       | 0.108        | 0.91x   | 0.108          | 0.91x   |
| 64   | 524288  | 2048 | 0.156       | 0.178        | 0.87x   | 0.179          | 0.87x   |

Under eager timing the companion dispatch change lifts `num_rows<=32` from
~0.5x to 0.74x-0.85x; 64 rows is ~0.9x. The residual gap is per-call host /
launch overhead (input validation, the top-k's host dispatch, allocations) that
is not yet at the 0.9 bar for small/mid batch in eager mode and is the next
optimization target.

Notes:

- The `native` column is a lean reconstruction of vLLM's op (its two Triton
  kernels + the CUDA `cooperative_topk` + a directly-allocated workspace),
  because the pinned vLLM build lacks the Python wrapper. It omits that
  wrapper's host overhead, so it is a conservative eager baseline; the
  authoritative figure comes from the native baseline collector on a vLLM build
  that ships the op.
- For reference, the same shapes under CUDA graph (GPU time, which the
  production decode path amortizes to) are 0.92x-1.04x with +dispatch.

## Residual risks

- Boundary-tie selection is approximate (as in vLLM); tests compare the selected
  **score** multiset, which is tie-invariant.
- The preferred `persistent_topk` path skips finalize on the assumption that it
  emits only in-range ids and `-1` padding — validated across the test matrix;
  the `cooperative_topk` fallback keeps finalize.
- Tuned and measured on H20 (SM90); other NVIDIA arches use the same paths but
  are not separately tuned.
