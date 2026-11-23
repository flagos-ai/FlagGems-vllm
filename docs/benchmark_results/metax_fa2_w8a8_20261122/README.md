# MetaX C550 FA2 W8A8 retained optimizations

## Operator fixes and optimizations

The original accuracy fix keeps probability-residual compensation for cancellation-sensitive attention and adjusts the C550 LSE absolute tolerance from `2e-5` to `1.5e-3`. Attention-output unit tolerance remains `0.025`; the residual-selection bound remains `0.01`. No additional tolerance relaxation is included.

Retained optimizations in this update:

- Stream packed D256 PV through two D128 FP32 accumulators on the selected non-residual path; retain probability high/low residual terms on compensated paths. Skip accumulator scaling only when its multiplier is exactly one and the workload guard permits it.
- Skip empty decode partitions using current device lengths; use an exact V-only path for a single KV token without LSE. Use the tested small-batch/short-KV decode MMA path while retaining the long-KV producer and merge behavior.
- Compute residual-head flags on GPU and reuse immutable tensor metadata within a call. Tensor values and sequence lengths are not cached across calls.
- Isolate short-Q/long-KV requests in mixed batches, reorder tiles for small head groups, and extend existing BM64 packed processing to eligible D256 short-prefill workloads. Native MetaX compiler-specific dispatch stays guarded; NVIDIA tuning is unchanged.

The final short-prefill dispatch selects `1024 <= maxK < 2048`, `maxQ >= 1024`, sufficient query/head work, and bounded padding. Existing physical-cache checks and the 512 MiB packed-storage cap remain. Newly selected shapes add about 1.5–146.5 MiB of packed storage; its cost is included in public-entry timing. No Torch compute fallback, public API change, or new external compute dependency is introduced.

## Performance

The latest complete measurement contains **562 shapes / 14,420 captured calls**, with two counterbalanced before/after rounds. **Before** is the preceding retained implementation; **after** adds the last short-KV packed-dispatch optimization. This is not a measurement of original PR head versus the complete accumulated change.

| Metric | Before last dispatch change | Final retained implementation |
|---|---:|---:|
| Invocation-weighted mean BF16/INT8, all 562 shapes | 0.891951x | 0.895368x |
| Invocation-weighted mean BF16/INT8, 62 changed shapes / 650 calls | 0.364339x | 0.562330x |
| Invocation-weighted INT8 latency sum | 141.019753 s | 140.210734 s |

All 62 changed shapes improve; their invocation-weighted INT8 self-speedup is **1.563106x**. Final BF16/INT8 range: **0.417330x–3.681171x**, with 59/562 shapes at least as fast as BF16. The overall gain is small, and the target of 1.3x is not reached.

Primary metric: `sum(count_i * (BF16_i / INT8_i)) / sum(count_i)`. This is neither the ratio of weighted time sums nor model throughput. Each INT8 latency averages its two round medians; the common BF16 reference averages four medians, one per implementation and round. Both independent BF16 columns and all timing samples are retained.

The complete rounds give `0.892681x -> 0.897700x` and `0.891329x -> 0.892793x`. Unchanged paths fluctuate: decode self-speedup is `0.989278x`, and identical-source A/A controls range from `0.858415x` to `1.019321x`. The roughly 0.38% aggregate improvement is smaller than observed control variation, so it is not evidence of a statistically established overall gain. No control-based normalization was applied. Historical 1.3x results used an older software environment and a different measurement policy and are not directly comparable.

**Baseline:** Native MetaX BF16 FA2, `flash_attn.flash_attn_interface.flash_attn_varlen_func`, compiled extension SHA256 `9818898a573fc375b78148f5086d899ea4646883786c5661748e71cb64922578`; not PyTorch SDPA or Triton. Both routes receive equivalent inputs; baseline BF16 dequantization happens before timing.

**Environment:** C550 GPU6; PyTorch `2.10.0+metax3.9.0.8`; native MetaX Triton `3.6.0+metax3.9.0.8`; FA2 distribution `2.6.3+metax3.9.0.8torch2.10`; vLLM `0.32.0+empty`. Image digest: `cr.metax-tech.com/public-library/maca-pytorch@sha256:85f1e96f88aece7202e6320c72a7b214079e6538af951e55e4f203569ccd29b8`.

**Timing:** Normal public entry, GPU Events, six alternating INT8/BF16 batches, target 20 ms per batch, minimum two calls, four warmups, median, CPU affinity 48; no CUDA Graph. The second full round reverses worker creation and per-case execution order. Input generation, compilation and numerical checks are outside timing; normal packing, attention, flags, merge and host-launch waiting inside the Event interval are included.

Shapes, complete CUQ/USED sequence metadata and counts are captured data. Tensor values, scales and physical page mappings are seeded synthetic inputs. These are operator measurements, not original-activation replay or end-to-end model results.

[All 562 index rows](REAL562.md) · [Summary](summary.json) · [Source manifest](manifest.json) · [Forward samples](measurements-forward.jsonl) · [Reverse samples](measurements-reverse.jsonl)

## Validation scripts

The submitted kernel/launcher hashes match the tested final snapshot in `manifest.json`. The latest strict suite passed **75 tests**, and two complete before/after rounds passed **2,248 full-output checks**, maximum absolute difference **0.015625**. Replay tolerance remains `atol=0.03, rtol=0.03`. Strict FP32-reference checks retain output `atol=0.025`, LSE `atol=0.0015`, `rtol=1.3e-6`; full replay checks output, with LSE covered separately by unit tests.

Run the latest 75-case suite in the same MetaX environment:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_d256_streaming_pv_precision.py \
  tests/test_mixed_short_split_precision.py \
  tests/test_mixed_short_split_edges.py \
  tests/test_prefill_head_order_precision.py \
  tests/test_d256_short_prefill_pack_precision.py
```

Earlier retained changes also have focused tests for D256 cancellation, split PV, decode scheduling, mutable sequence lengths and residual flags. Those files are included; their earlier runs are not presented as a new combined-suite run of this update.

The published steady-state timer preserves the measured timer's function AST. It requires Linux CPU 48 in the process affinity set and an isolated visible MetaX device. Use a fresh output directory:

```bash
PYTHONHASHSEED=0 PYTHONPATH=src python tools/benchmark_metax_int8_real_shapes_steady.py \
  --shapes /path/to/real_shapes/shapes.json \
  --output /path/to/new_results \
  --timing event --batches 6 --repeat 2 --warmup 4
```

For a quick hotspot check, add `--indices 0,18,216,321,343`. The captured manifest SHA256 is `543386162d804b4f142a2b385510c51303f0e5c62dced45cc60ae2b54093ee7f`. A standalone invocation reproduces one implementation's timing policy; reproducing the paired comparison additionally requires separate source processes, serialized case-wise A/B requests, and reversing both creation and execution order in the second round. The archived samples preserve the actual paired measurements.
