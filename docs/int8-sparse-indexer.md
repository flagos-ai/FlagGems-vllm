# INT8 sparse indexer

HY4 can use these four public operators when its NVIDIA GPU lacks E4M3 FP8
support. They quantize the indexer's floating-point projections; they do not
reinterpret checkpoint INT8 weights or change Linear/MoE quantization. The
framework selects a matching query quantizer, cache writer and logits pair
before constructing the model. There is no execution-error retry in FP8 or
PyTorch.

## API and numerical contract

All inputs must share a CUDA device. Floating-point inputs must be finite.
This is an inference-only NVIDIA path targeting SM80 and newer; autograd and
other vendors are outside its scope.

| Operator | Inputs | Result |
| --- | --- | --- |
| `per_token_group_quant_int8(x, group_size, *, use_ue8m0=True)` | FP16/BF16/FP32 `[M,128*G]`, feature stride 1, `group_size=128` | Contiguous INT8 values and FP32 `[M,G]` scales |
| `indexer_k_quant_and_cache_int8(k, cache, slots, quant_block_size, scale_fmt)` | FP16/BF16/FP32 `[T,128]`; contiguous UINT8 `[P,B,132]` cache; contiguous INT32/INT64 slots; block size 128; `scale_fmt=None` or `"ue8m0"` | In-place cache update; negative and out-of-capacity slots are ignored |
| `int8_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True)` | `(INT8[M,H,128], None)`; `(INT8[N,128], FP32[N])`; FP32 `[M,H]` weights; INT32 `[M]` start/end offsets | FP32 `[M,N]` logits |
| `int8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables, schedule_metadata, max_model_len, clean_logits=False)` | Contiguous `(INT8[B,N,H,128], None)` (or `[B,H,128]`); UINT8 `[P,S,1,132]`; FP32 `[B*N,H]` weights; INT32 `[B]` or `[B,N]` lengths; INT32 `[B,L]` or `[B,N,L]` page tables | FP32 `[B*N,max_model_len]` logits |

Logits support `H=16/32/64`, dimension 128 and unit feature/head-weight
strides. Prefill accepts leading query/key/weight strides. Paged queries and
cache storage must be contiguous. Unsupported formats raise before launch;
there is no Torch compute fallback. Empty outputs allocate without a launch.
`schedule_metadata` is accepted for interface compatibility; the paged kernel
uses a static grid instead of a device schedule.

For each 128-element group, `scale=max(amax(abs(x)),eps)/127`, optionally
rounded up to a power of two. The query epsilon is `1e-10` and the key epsilon
is `1e-4`. Quantization uses round-to-nearest, ties-to-even, clamped to
`[-127,127]`. Both divisions use `tl.div_rn`; approximate division can change
the result at rounding boundaries.

Each physical cache page stores all `S*128` INT8 value bytes followed by `S`
FP32 scales. It is **not** an interleaved 132-byte record per token. The
existing byte-copy `cp_gather_indexer_k_quant_cache` can gather this format into
an INT8 workspace. Query scales are folded into the FP32 head weights by the
framework, so each logit is:

```text
sum_h(relu(int32_dot(q_int8[h], k_int8) * k_scale) * weight_with_q_scale[h])
```

Invalid columns are `-inf` with `clean_logits=True`, otherwise zero. The
framework's TopK operator still receives valid sequence ranges. Paged kernels
mask invalid logical/physical pages and reload lengths and page tables on every
CUDA graph replay. Output allocations have fixed shapes for capture, and every
output element is written on each execution.

## Integration and tuning

The four functions are exported from `ops` and registered by the top-level
package's existing callable registrar. Operator markers, test aliases,
benchmark entries and NVIDIA configurations accompany them. Existing FP8
kernels are unchanged. The shared BF16 sparse MLA dispatcher checks the actual
GPU capability before selecting TLE descriptor copies: pre-Hopper devices keep
the generic Triton path even when the TLE Python APIs are installed.

| Path | Dispatch | Tuning |
| --- | --- | --- |
| Empty/unsupported | Metadata checks; empty allocation or explicit exception | None |
| Small query/cache quantization | One independent 128-element reduction per program | Fixed four warps; fixed-width exemption |
| At least 1024 query groups | Several independent rows per program, same RNE arithmetic | `per_token_group_quant_int8_batched`, key `(GROUP_COUNT,GROUPS)`; 8/16/32-row blocks, 2/4 warps |
| Small prefill (`M<=16`) | Row/head dot without padded query rows or serial head-pair loops | `int8_mqa_logits_rows`, key `(M,N,H)`; key tile 32/64/128, four warps, one stage |
| Dense prefill | INT8 tensor-core dot with INT32 accumulation, FP32 epilogue | `int8_mqa_logits`, key `(M,N,H)`; row/key/head tiles and 1/2/3 stages, retaining original configs |
| Paged decode | Static row/key grid, live device lengths/page tables | `int8_paged_mqa_logits`, key `(ROWS,CAPACITY,H)`; `BLOCK_N=64/128`, four warps, one stage |
| Gather/TopK/MLA | Existing public byte-copy, TopK and BF16 MLA operators | Existing configurations |

Production-path audit of `ops/int8_sparse_indexer.py`:

| Call family | Purpose | Compute/copy? |
| --- | --- | --- |
| `torch.empty` and dtype constants | Uninitialized outputs and metadata validation | No |
| `shape`, `stride`, `numel`, `is_contiguous`, `device`, `dtype` | Dispatch and pointer geometry | No |
| `view(torch.float32)`, `unsqueeze` | No-copy cache reinterpretation and rank normalization | No |
| Triton `_quantize_query`, `_quantize_query_batched`, `_quantize_cache`, `_prefill_rows`, `_prefill`, `_decode` | All numerical work | Triton kernels |

Tests and benchmark baselines alone use Torch rounding, INT32/double dot,
scaling, ReLU and reduction. Production code imports neither reference.

## Validation

Development base: `f775b37f3066e2935742d26c7592a5dd55499159`. Validation used
Python 3.12, Torch 2.11.0+cu129, vLLM 0.24.0 and Triton/FlagTree 3.6 on H100
80GB. Run from the repository root:

```bash
PYTHONPATH=src VLLM_PLUGINS=fl CUDA_VISIBLE_DEVICES=0 \
  python -m pytest tests/test_int8_sparse_indexer.py -q

PYTHONPATH=src VLLM_PLUGINS=fl CUDA_VISIBLE_DEVICES=0 \
  python -m pytest benchmark/test_int8_mqa_logits.py \
    benchmark/test_int8_paged_mqa_logits.py \
    benchmark/test_per_token_group_quant_int8.py \
    benchmark/test_indexer_k_quant_and_cache_int8.py \
    --level core --iter 30 --warmup 10 --record json --output int8-benchmark.json -q
```

The 93 functional cases pass. They cover float input dtypes, RNE ties and zero,
empty and partial tiles, leading strides, all supported head counts, page sizes
16/64/128, ignored slots, valid-range cleanup, input preservation and graph
replay after changing Q/K, slots, weights, lengths and page mappings. References
run on CPU; logits use double precision to represent the INT32 dot exactly.
Quantized values match exactly; logits use `rtol=1e-5, atol=2e-3`.
The 41 TLE capability regressions also pass, including SM80/86/89 rejection
and SM90/100 acceptance with complete TLE APIs. Two generic BF16 sparse MLA
cases (64/128 heads, dimension 576, sink and variable TopK length) pass against
the existing reference with `FLAGGEMS_FLASHMLA_SPARSE_TLE=0`.

All six INT8 kernels also compile for `GPUTarget("cuda",80,32)`. The three dot kernels
emit signed INT8 MMA and no E4M3/E5M2 instructions. This is a compile check,
**not A100 execution evidence**. Compute Sanitizer and Nsight Compute were not
available in the development environment.

The four standard Torch-reference benchmark entrypoints remain available.
The matched [H100 INT8/FP8 report](hy4-int8-h100.md) records 44 component and
complete-indexer-chain cases, including native DeepGEMM FP8, raw round samples,
compiled resources, accepted/rejected tuning candidates and comparison limits.
No Torch numerical fallback is used in any production path. The Torch
acceptance gate is met; the report additionally compares against FP8 and
documents the remaining gap to native DeepGEMM on large prefill shapes.

Plugin integration is covered in
[vllm-plugin-FL #622](https://github.com/flagos-ai/vllm-plugin-FL/pull/622): real
INT8 query/cache/gather/logits/TopK composition, refreshed CUDA graph replay and
opaque full-graph Inductor query quantization. Remaining validation is A100
execution plus end-to-end checkpoint generation and quality evaluation: changing
the indexer's quantization format can change selected sparse tokens. BF16 MLA
KV is required on the portable path; FP8 MLA KV and checkpoint quantization
coverage are separate contracts.
