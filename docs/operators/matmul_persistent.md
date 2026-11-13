# vLLM batch-invariant matmul

Public entry: `flaggems_vllm.matmul_persistent(a, b, *, out=None)`.
Reference: `vllm.model_executor.determinism.batch_invariant.matmul_batch_invariant`.
This is a port of vLLM's deterministic inference workload, not an ATen dispatch replacement.
The internal vLLM two-dimensional helper named `matmul_persistent(a, b, bias=None)` is
not the public contract of this contribution. No bias parameter or global mode is exported.

## Specification and paths

- NVIDIA CUDA SM80+, FP16/BF16/FP32, both input ranks at least two, identical dtype/device,
  compatible K and broadcastable leading dimensions. Output shape is broadcast batch + `[M,N]`,
  with input dtype/device. Inference only; requires-grad inputs are explicitly unsupported.
- CPU, HIP, XPU, sparse layouts and other dtypes raise `NotImplementedError`. Invalid
  dimensions/dtypes and incompatible out tensors raise errors before launch.
- Empty outputs return an allocation; K=0 produces zeros. NaN/Inf follow the ported Triton
  accumulation. No finite-value promise or training/autograd support is added.
- Matrix transposes, nonzero storage offsets, strided inputs and proven non-overlapping
  strided outputs are supported. `out` must match exactly; shared storage with either input
  is conservatively rejected, even for disjoint views. Ambiguous overlapping layouts are rejected.
- All results are computed into a fresh tensor before the optional `out.copy_`.
- On H20 only, FP16/BF16 with 0<M<=128, N in {16,32}, K in {16,32,64} use
  32x32x64, 4 warps, 3 stages. For ND x 2D, M is the flattened row count.
- Within that domain, singleton/leading-one batch shapes use views directly. A non-affine
  flattened broadcast batch uses the no-copy broadcast kernel with int64 batch offsets.
- Outside that domain the original expand/reshape preparation and original BMM kernel/config
  are used. MM retains the original BF16 tuned table and architecture defaults.
  There is no general-domain performance claim. Physical GPUs other than H20 disable the new path.
- Configuration is cached by input device index, resolved lazily. Module import performs no
  CUDA query or global ATen override. An input on a non-current CUDA device is launched under
  its device context. Warm the exact input device/path before CUDA Graph capture.

## Fixed-launch no-autotune exemption

This contribution requests the repository workflow's explicit no-autotune exemption.
The scope is vLLM batch invariance: fixed K partition/reduction order must remain independent
of batch composition. It freezes the measured H20 small tile and existing vLLM general/tuned
configs. Runtime shape-timing selection, libtuner wrappers and new tune_configs entries are
not added. The three Triton kernel ASTs are unchanged from the audited experiment/snapshot.
If autotuning is required by review, introduce it separately and revalidate strict equality,
row/batch invariance and performance; these experiments do not validate such a change.

## Production no-torch-compute audit

The candidate has no baseline/test imports, torch matmul/reductions/casts, torch.compile,
or torch.library registration. Allowed torch calls are CUDA metadata/context queries,
`torch.empty` and `torch.zeros`. Tensor metadata, expand/as_strided/reshape and optional
copy_ implement layout preparation and output placement only. A fallback reshape may copy,
matching the original general path. The baseline snapshot contains unrelated vLLM helpers
solely as reference provenance; none is imported by production.

## Evidence and validation boundary

The fifth experiment, not this final integrated entry, measured 24 FP16/BF16 cases:
2.32284x same-run geometric mean vs the frozen original, 1.04707x vs the fourth-round
best control. For broadcast cases 14/15, independent 16-round medians were 5.728/5.680 us,
pooled p99 6.272/6.304 us. These are CUDA-event microbenchmark statistics, not request p99.
All 120 experimental combinations and 15 focused combinations passed their checks.

This bundle changes validation, device caching and fast-path bounds, and removes experimental
configuration. Therefore those measured speedups motivate this implementation; they are not
new measurements of it. Run the installed-package acceptance before making final PR claims.

The new project functional matrix is 176 items, including original 24 hot cases,
FP32, signed values, domain boundaries, original large tuned bucket row invariance,
broadcast repeat-rank regression, offsets/strides/out, errors, empty/K=0, special values,
no-copy launch storage and four CUDA Graph correctness cases. It deliberately does not
install the snapshot's unrelated global ATen overrides, softmax/mean, or compiled mm/addmm tests.
CUDA Graph correctness does not establish Graph performance. Multi-GPU device caching is
CPU-policy checked; real multi-GPU execution remains a separate hardware validation gap.

Project benchmark runs 12 semantic shape keys x 2 dtypes in non-Graph kernel mode, with
same-run baseline/candidate timing. Inputs are seed42 rand, including output allocation.
The final package is somewhat stricter and includes new Host checks; remeasure its full calls.
Default compute-sanitizer memcheck must exit zero and explicitly report zero errors.
Do not suppress CUDA API errors to turn a failed default run into a passing result.
