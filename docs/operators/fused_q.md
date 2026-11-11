# fused_q NVIDIA H20 implementation

`flaggems_vllm.fused_q` preserves the public arguments, defaults and three
outputs of the explicitly ported vLLM Triton baseline. It fuses interleaved
MQA query RoPE/FP8 packing with indexer RoPE, UE8M0 quantization and weight scaling.
This is a library API, with no ATen schema; it is excluded from `enable()`'s
ATen registration list and remains available as a public callable.

## Interface and paths

Return value: `(index_q_fp8, index_weights_out, mqa_q)`.
With `quantize_mqa=True`, `mqa_q` packs `[ql_nope; RoPE(q_pe)]` in OCP E4M3 FP8.
With False, it contains only rotated `q_pe` in its original dtype. With
`has_indexer=False`, the first two outputs retain the baseline's unwritten
placeholder semantics; their contents must not be consumed or compared.

The optimized scope is the previously measured full NVIDIA H20 (78 SM, CC 9.0):

- Tokens: 1, 6, 24, 96, 384, 1536, 4096; Q heads: 16, 32, 64.
- Contiguous BF16 `q_pe[N,H,64]`, `ql_nope[N,H,512]`, `index_q[N,32,128]` and
  two `[16384,64]` caches on the same device.
- INT64 positions; FP32 singleton q_scale; FP32 index weights `[N,32]`.
- Indexer enabled, index interleave disabled, MQA quantization enabled.

H=16/32 uses compact task grids `[N,48]`/`[N,64]`; H=64 uses `[N,3,32]`.
PDL is disabled in this optimized scope. Other baseline-supported shapes,
layouts and flags use the original Triton grid and architecture PDL policy.
Other vendors, CPU computation, autograd and arbitrary invalid input layouts
are not supported claims. This contribution targets NVIDIA inference.

## Fixed-launch autotune exemption

**Explicit no-autotune exemption:** this change intentionally retains the
validated one-warp kernel and the deterministic metadata-only grid policy.
It does not introduce `libtuner`, `libentry` or unused tune-config entries.
The upstream baseline notes a 1/2/4/8-warp sweep, and the completed H20
experiments established the chosen grid/PDL policy with one warp. Adding
runtime tuning or another launch wrapper would change the measured entry.

The exemption is documented under the repository workflow's explicit-exemption
rule and is subject to maintainer review. If maintainers require tuner
integration, make it a separate change with new package-level correctness and
performance evidence; do not claim that a newly tuned entry inherits the old
measurements. No target `tune_configs.yaml` modification is needed for this
fixed-launch submission.

## Baseline and benchmark contract

`baseline/fused_q.py` contains the unchanged port explicitly supplied by the
contributor, plus the eight collector contract fields/functions. `native()`
returns that port's `fused_q`; it never returns the optimized candidate or a
Torch mathematical implementation. It does not dynamically require installed
vLLM or silently change between vLLM versions/CuTeDSL dispatch paths.

The baseline is labelled `ported_vllm_triton_snapshot`, not an installed-vLLM
measurement. The original pure port's SHA256 is
`af98585e5cb8302047aa3b7e8185946b95e2ad9bdb4a1a9310983807d673cf36`.
Reference upstream API: `vllm.models.deepseek_v32.common.kernels.fused_q`.

The collector and benchmark share 21 BF16 bindings with six-field keys
`[N,H,512,64,32,128]`; q_scale=.5, softmax scale=128**-.5 and head scale=32**-.5.
The benchmark reuses exactly the same input constructor and records the
explicit semantic key, actual inputs, full config and candidate path.
External baseline records are accepted only when input/output config matches.
`speedup` is the on-device baseline/candidate ratio; `compared_speedup`, if
`--base-data` is supplied, is a separate historical/platform comparison.

Only non-Graph kernel/operator modes are accepted. Output allocation remains
inside both timed callables. Numerical references and input-preservation
checks run outside timing. The project benchmark protocol differs from the
earlier eight-trial experimental runner: remeasure after package integration.

## Verification and evidence boundary

Functional tests call the actual `flaggems_vllm.fused_q` export. They preserve
the 123-case reference matrix and add 21 full optimized workloads, actual
grid/PDL checks, bitwise comparison to the port, repeatability and input
preservation. Separate collector tests add six bounded contract checks.
`--quick` skips large functional cases and is not full acceptance.

The standalone implementation previously passed 123 cases and CUDA memcheck
(zero errors), and achieved 1.47091x equal-weight geometric mean speedup over
21 H20 shapes. N>=96 achieved 1.95125x; N=4096/H=16 achieved 3.24010x. Those
are existing standalone measurements, not results of this new package entry.
The small N=6/H=64 case was about 1.7% slower (0.112 us); disclose this tradeoff.

Production uses Torch only for allocation, dtype/metadata and device-property
queries. All arithmetic and casts in the kernel/helpers are Triton operations.
Neither test references nor benchmark code is imported by production.

Run package tests, benchmark and memcheck after copying. No whole-model,
CUDA Graph performance or untested-device speedup is claimed.
