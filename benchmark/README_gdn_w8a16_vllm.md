# Installed vLLM-MetaX GDN baseline

`bench_gdn_w8a16_vllm.py` compares the migrated MetaX `gdn_w8a16`, the
**installed vLLM-MetaX default GDN path**, and a PyTorch reference in the same
runtime. This is distinct from `test_gdn_w8a16_compare.py`, whose `metax` entry
is the existing **FlagGems-vllm** implementation.

Run inside an existing vendor vLLM-MetaX container, without replacing its
compiler or installed libraries:

```bash
export CUDA_VISIBLE_DEVICES=6 MACA_VISIBLE_DEVICES=6  # choose an idle GPU
export GDN_SHAPE_FILE=/path/to/external/shapes.json
export GDN_RESULT_DIR=/tmp/gdn_vllm_results
export GDN_IMPL_COMMIT=$(git rev-parse HEAD)
python benchmark/bench_gdn_w8a16_vllm.py
```

Real-shape data is external. The original source is
`/data/fzy/real_shapes_20261012/gdn_w8a16_fp8__int8/shapes.json`.
`GDN_REPO` optionally selects a frozen source checkout. `GDN_CASES` optionally
selects comma-separated 1-based record indices. No explicit real-shape list is
bundled in the repository.

## Comparison boundaries

- **Prefill:** instantiate `ChunkGatedDeltaRule` under the installed platform's
  default configuration and load its general plugins. Require the selected
  backend to be Triton/FLA and external FLA to be disabled. Use precomputed gates
  and `use_qk_l2norm_in_kernel=False`, matching the layer's prefill core call.
  Chunk metadata is prepared before capture, as it is in vLLM's metadata builder.
- **Decode:** call the installed packed-decode or fused-sigmoid-update wrapper
  identified by the input record. Both include Q/K normalization, gates and
  in-place FP32 cache updates. The MetaX implementation calls the public
  `gdn_w8a16(..., decode=True)` interface with raw gates and packed or separate
  QKV. Gate generation, normalization, recurrence and cache writeback are fused.
- All paths share raw random BF16 Q/K/V of the requested shape. Gates are
  synthetic, not production activations. Decode derives BF16 raw gate inputs
  from the historical stable gate distribution; FP32 A_log/dt_bias are zero.
- Reset cache state before each replay **outside** GPU event timing. Measure
  31 alternating samples, each averaging three independently timed CUDA Graph
  replays. Validate eager and captured output/final state at atol=0.002,
  rtol=0.02 before reporting latency. Failures have no performance ratio.
- Exclude convolution, projection, scheduling, input generation, metadata
  creation, compilation and capture. This is not full-model inference throughput.

The candidate is loaded from the selected checkout with only its runtime-vendor
and chunk-module imports isolated. The kernel source, math and launch parameters
remain unchanged, allowing comparison without importing unrelated FlagGems
operators into vendor containers. The output records source SHA256 hashes,
installed torch/Triton/vLLM/vLLM-MetaX versions and the selected backend. Do not
combine these timings with earlier measurements from another runtime or scope.

## Precision and validation

Core mode leaves the FP32 input state immutable and defaults to FP32 intermediates.
The performance benchmark explicitly enables `mixed_precision=True` for prefill.
Mixed-precision chunks use
BF16/FP16 intermediate buffers and dot operands with FP32 accumulation/state.
This is not numerically equivalent to fully FP32 intermediates. Standard
output/state tolerances are atol=0.002, rtol=0.02, matching this operator benchmark.
The benchmark also records a separate strict state check (atol=1e-4,
rtol=0.002) and relative RMS error; failures of that stricter check remain visible.

Run independent functional coverage, including multi-step decode and dynamic
Graph replay metadata, with `python benchmark/check_gdn_w8a16_metax.py`.
This checks the public backend entry without requiring unrelated package imports.

Temporary chunk workspace is bounded at 256 MiB; returned output/final-state
allocations are additional. No INT8 state quantization or Torch compute fallback
is used. MetaX launch configurations are selected offline; NVIDIA code and its
autotuning are unchanged. Decode positive state slots must be unique and valid;
nonpositive slots are padding. End-to-end model quality is not evaluated.

The mixed-precision option is input-distribution dependent: some BF16 unit cases
with normalized Q/K and stronger beta gates do not meet the FP32-reference
standard tolerance. It is opt-in, not the default core precision contract.
The default FP32 unit tests retain their original strict state tolerances.
