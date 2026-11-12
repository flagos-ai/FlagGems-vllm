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
   `out` with `-1`. Launched with a fixed, `nblocks`-keyed config
   (`_pick_score_tile`) — see **Launch config**.
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

## Launch config

- The score kernel is **memory-bound** (one streamed max-reduction over the
  logits row), so its `TILE`/`num_warps` are chosen by a fixed `nblocks`
  heuristic (`_pick_score_tile`) rather than autotuned. The heuristic reproduces
  the previously-autotuned configs' GPU time while removing the per-call
  autotune key/lookup, which on the eager decode path is a measurable share of
  the op's launch overhead (see **Benchmark**). There is no `tune_configs.yaml`
  entry for this op.
- Block scores and per-row valid-block counts are kept in a per-`(device, rows,
  nblocks)` scratch cache (`_get_scratch`), dropping two `torch.empty`
  allocations per call on the hot decode path. The score kernel fully overwrites
  the `[0, limit)` prefix each call and the top-k reads only that prefix, so
  reuse never exposes stale data.
- The finalize kernel has a fixed launch: a single launch of at most `k`
  elements per row, skipped entirely on the preferred `persistent_topk` path.

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
`SpeedUp = latency_native / latency_gems`. This op does not modify
`persistent_topk`; numbers are taken with the companion `persistent_topk`
sampled-dispatch PR applied (the stacked plan lands it first). Only the two
`num_rows<=32` x 1M shapes are sensitive to it -- on plain `main` they take the
cooperative radix path instead of the sampled one and measure ~5% slower (still
~0.62x); the other ten shapes are dispatch-independent.

| rows | seq_len | k    | native (ms) | this op (ms) | SpeedUp |
| ---: | ------: | ---: | ----------: | -----------: | ------: |
| 1    | 4096    | 2048 | 0.0654      | 0.0612       | 1.07x   |
| 1    | 131072  | 2048 | 0.0644      | 0.0668       | 0.96x   |
| 1    | 1048576 | 2048 | 0.0640      | 0.0979       | 0.65x   |
| 4    | 32768   | 2048 | 0.0640      | 0.0609       | 1.05x   |
| 4    | 1048576 | 2048 | 0.0635      | 0.0978       | 0.65x   |
| 14   | 262144  | 2048 | 0.0643      | 0.0736       | 0.87x   |
| 16   | 65536   | 2048 | 0.0639      | 0.0623       | 1.03x   |
| 32   | 131072  | 2048 | 0.0786      | 0.0700       | 1.12x   |
| 64   | 16384   | 2048 | 0.0989      | 0.0775       | 1.28x   |
| 64   | 524288  | 2048 | 0.1558      | 0.1539       | 1.01x   |
| 16   | 131072  | 512  | 0.0639      | 0.0659       | 0.97x   |
| 16   | 131072  | 1024 | 0.0656      | 0.0673       | 0.97x   |

Nine of twelve shapes are at parity or faster (0.96x-1.28x), covering the whole
realistic decode regime (batch 16-64). The score-kernel launch-config and
scratch-reuse trims removed the per-call host/launch overhead that previously
held small/mid batch at ~0.5x-0.85x in eager mode.

Three shapes remain below the 0.9 bar -- all **thin batch (1-14 rows) x
ultra-long context (256K-1M)** -- and profiling shows these are *not* a
launch-overhead problem the trims can reach:

- **1/1M, 4/1M (GPU-bound).** Host submit is ~0.046 ms; the GPU work is
  ~0.053 ms, dominated by the row-prefix top-k. For thin batch x long context
  the top-k runs effectively one CTA per row (the sampled path under the
  companion dispatch; plain `main`'s cooperative radix path is similarly
  launch-limited -- ~16 CTAs for a single row, near its occupancy plateau where
  adding CTAs gains <6%), so it cannot fill the ~78 SMs and our GPU time alone
  approaches native's *entire* latency. Matching native here needs a
  multi-CTA-per-row top-k algorithm, not a host/dispatch tweak.
- **14/256K (host-bound).** GPU work is only ~0.029 ms; the row length sits at
  the kernel's single-CTA histogram threshold, so it runs one CTA/row and the
  gap is residual host overhead already near the trimmed floor.

These thin-batch/long-context edges are rare in production decode and are
tracked as a follow-up (a cooperative multi-CTA radix-select for the top-k).

Notes:

- The `native` column is a lean reconstruction of vLLM's op (its two Triton
  kernels + the CUDA `cooperative_topk` + a directly-allocated workspace),
  because the pinned vLLM build lacks the Python wrapper. It omits that
  wrapper's host overhead, so it is a conservative eager baseline; the
  authoritative figure comes from the native baseline collector on a vLLM build
  that ships the op.
- For reference, the same shapes under CUDA graph (GPU time, which the
  production decode path amortizes to) are at parity or faster.

## Residual risks

- Boundary-tie selection is approximate (as in vLLM); tests compare the selected
  **score** multiset, which is tie-invariant.
- The preferred `persistent_topk` path skips finalize on the assumption that it
  emits only in-range ids and `-1` padding — validated across the test matrix;
  the `cooperative_topk` fallback keeps finalize.
- Tuned and measured on H20 (SM90); other NVIDIA arches use the same paths but
  are not separately tuned.
