# Qwen4 self-developed kernels

This directory contains exactly eight Qwen4 device kernels extracted from the
Qwen3.8-Flash-Next production implementation: three HyperConnection kernels,
three QSA kernels, and two PLE state-I/O kernels. Model wiring, checkpoint
loading, vLLM custom-op registration, and machine-specific paths are outside
this repository.

The QSA family deliberately excludes the five model-vendor kernels already
present in the production package (sparse, expand, single-cache-row store, and
related paged helpers). Only the three names in the table below are included.

The read-only production source snapshot used for extraction has these SHA256
digests (the target wrappers add only standalone guards and exports):

```text
hyperconnection.py  398f4882df5cc99f8378f191cd163f8ee585bd9f16056ce1fce08653f6e7e5a9
qsa.py              40a3190b04af28a1bc5944e29a334b041384ee6cc12bdd8c8a7382e23b207342
ple_state.py        479ce6065068ec0fc1729bd9a26ecf9a6a3c5124bce931f5d2174b8ebb894dee
```

| Family | Kernel | FlagGems-vllm entry point | Source |
|---|---|---|---|
| HC | `_grouped_gemma_rmsnorm_kernel` | `qwen4_grouped_gemma_rmsnorm` | `src/flaggems_vllm/ops/qwen4/hyperconnection.py` |
| HC | `_hc_gate_reduce_kernel` | `qwen4_hc_gate_reduce` | same |
| HC | `_hc_inject_combine_kernel` | `qwen4_hc_inject_combine` | same |
| QSA | `_qsa_mqa_paged_dot_kernel` | `qwen4_qsa_mqa_paged_dot` | `src/flaggems_vllm/ops/qwen4/qsa.py` |
| QSA | `_store_qsa_kv_rows_kernel` | `qwen4_store_qsa_kv_rows` | same |
| QSA | `_compress_norm_mrope_store_qsa_groups_kernel` | `qwen4_compress_norm_mrope_store_groups` | same |
| PLE | `_ple_state_gather_kernel_3d` | `ple_state_gather` | `src/flaggems_vllm/ops/qwen4/ple_state.py` |
| PLE | `_ple_state_scatter_kernel_3d` | `ple_state_scatter_` | same |

## Contracts and exact references

The reference implementations below are test/benchmark-only Torch formulas.
Production wrappers allocate with `torch.empty*`, read metadata, launch
Triton, and never use a Torch compute fallback. Unsupported device, dtype,
shape, layout, or alias contracts raise instead of silently changing the
operator implementation.

### HyperConnection

For `x` shaped `[..., hc_count * hidden]` and `w` shaped
`[hc_count * hidden]`, grouped Gemma RMSNorm is:

```python
def torch_grouped_gemma_rmsnorm(x, w, hc_count, eps):
    h = w.numel() // hc_count
    x3 = x.reshape(-1, hc_count, h).float()
    w2 = w.reshape(hc_count, h).float()
    inv_rms = torch.rsqrt(x3.square().mean(-1, keepdim=True) + eps)
    return (x3 * inv_rms * (1.0 + w2)).to(x.dtype).reshape_as(x)
```

For gate logits and normalized branches shaped `[..., hc_count * hidden]`:

```python
def torch_hc_gate_reduce(logits, normed, hc_count):
    h = logits.shape[-1] // hc_count
    shape = (*logits.shape[:-1], hc_count, h)
    return (
        (torch.sigmoid(logits.float().reshape(shape)) * normed.float().reshape(shape))
        .mean(-2)
        .to(normed.dtype)
    )
```

For injection logits `[..., hc_count]`, block output `[..., hidden]`, and
residual `[..., hc_count * hidden]`:

```python
def torch_hc_inject_combine(injection_logits, block_output, residual, hc_count):
    h = block_output.shape[-1]
    residual3 = residual.reshape(*block_output.shape[:-1], hc_count, h)
    alpha = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
    result = residual3.float() + block_output.float().unsqueeze(-2) * alpha.unsqueeze(
        -1
    )
    return result.to(residual.dtype).reshape_as(residual)
```

The three wrappers accept BF16/FP16 accelerator tensors. RMSNorm and gate
reduce require same-device contiguous tensors; injection additionally accepts
the packed projection's row-strided logits view, while block output and
residual remain contiguous. The HC count, hidden dimensions, dtype, and
positive epsilon are validated before launch.

### QSA paged MQA dot

`qwen4_qsa_mqa_paged_dot` requires BF16 `q` shaped `[rows, 4, 128]` and cache
shaped `[pages, page_size, 1, 128]`. For request `r` and compressed token `t`:

```python
visible = min(
    (query_position[r] + 1) // compress_ratio,
    sequence_lengths[token_to_req[r]] // compress_ratio,
)
score[r, t] = sum(relu(q[r, head] @ key[r, t]) for head in range(4)) / sqrt(128)
```

The page-table lookup and visibility mask are part of the reference contract;
non-visible, invalid-request, and invalid-page entries are `-inf`, and
`visible_blocks` is written as int32. The wrapper does not dispatch the
vendor sparse/expand/TopK kernels.

### QSA paired K/V store

For a valid unique flattened slot `s`, `block = s // page_size` and
`token = s % page_size`; the reference performs two independent indexed writes:

```python
def torch_store_qsa_kv_rows(k_cache, v_cache, slots, key, value):
    block = slots.long() // k_cache.shape[1]
    token = slots.long() % k_cache.shape[1]
    k_cache.index_put_((block, token), key, accumulate=False)
    v_cache.index_put_((block, token), value, accumulate=False)
```

The Triton kernel preserves arbitrary cache and row strides and ignores an
out-of-range slot. Its parallel write contract requires unique valid slots;
duplicate-slot ordering is intentionally not claimed.

### QSA fused compression, Gemma RMSNorm, MRoPE, and store

The fused entry point performs, in order:

1. FP32 accumulation of `compress_ratio` paged raw-cache rows;
2. BF16 materialization of the mean;
3. FP32 Gemma RMSNorm and a second BF16 materialization;
4. Neox MRoPE on the first 64 channels;
5. paged compressed-cache stores, with the remaining 64 channels passed through.

The exact reference starts with `pooled = (raw.float().mean(1)).to(bfloat16)`,
then computes `normalized = (pooled.float() * rsqrt(mean(pooled.float()**2) +
eps) * (norm_weight.float() + 1)).to(bfloat16)`. The rotary first/second
halves are rotated in FP32 and written as BF16.

For each row, the executable Torch reference is equivalent to:

```python
pooled = (
    torch.stack(raw_rows).float().sum(0) / compress_ratio
).to(torch.bfloat16)
normalized = (
    pooled.float()
    * torch.rsqrt(pooled.float().square().mean() + eps)
    * (norm_weight.float() + 1)
).to(torch.bfloat16)
freq = torch.arange(32, device=pooled.device)
axis = torch.where(
    (freq % 3 == 1) & (freq < 3 * section_h),
    height_position,
    torch.where(
        (freq % 3 == 2) & (freq < 3 * section_w),
        width_position,
        time_position,
    ),
)
cos = cos_sin_cache[axis, freq].float()
sin = cos_sin_cache[axis, 32 + freq].float()
first, second = normalized[:32].float(), normalized[32:64].float()
stored = torch.cat(
    ((first * cos - second * sin).to(torch.bfloat16),
     (second * cos + first * sin).to(torch.bfloat16),
     normalized[64:])
)
```

`raw_rows` are looked up through `raw_block_table` for the `compress_ratio`
logical positions ending at the current position; `stored` is written to the
flattened `compressed_slots` location. The contiguous-section mode replaces
the `freq % 3` selector with the documented T/H/W ranges.

Qwen4's actual MRoPE cache is interleaved. With `mrope_section=(11, 11, 10)`
and `mrope_interleaved=True`, frequency lane `f` selects T/H/W using
`f % 3 == 0/1/2`, subject to the section counts. It is **not** three
contiguous `[11, 11, 10]` slices. The wrapper also supports the explicit
contiguous-section mode when `mrope_interleaved=False`; that mode is not the
Qwen4 checkpoint layout. The benchmark and correctness test exercise the
interleaved mode.

The fused wrapper currently guards the production shape `head_dim=128`,
`rotary_dim=64`, valid section sum `32`, same-device same-dtype floating cache
and norm tensors, and positive ratio/epsilon. The public path is generic
Triton, but only the NVIDIA H100 result in the source evidence has been run;
other accelerators remain unverified until their own compile/correctness/E2E
checks are run.

### PLE state gather/scatter

PLE state is a possibly strided logical `[cache_rows, hidden, width]` view.
Gather preserves the inner strides and uses the original index for validity:

```python
def torch_ple_state_gather(state, indices):
    valid = (indices >= 0) & (indices < state.shape[0])
    bounded = indices.clamp(0, state.shape[0] - 1)
    rows = torch.ops.aten.index_select.default(state, 0, bounded)
    return torch.where(valid.view(-1, 1, 1), rows, torch.zeros_like(rows))
```

In particular, `-1` is checked **before** any safe-address clamp. It is a
NULL/padding row and must not become valid row zero. Scatter receives an
explicit boolean `write_mask`; the reference writes only when the original
index is in range and the mask is true. Callers mask NULL/padding rows and
earlier duplicate writers so the final enabled writer is deterministic. The
kernel itself also rejects negative and out-of-range indices.

The exact scatter reference used by the tests/benchmark is:

```python
def torch_ple_state_scatter(state, indices, rows, write_mask):
    result = state.clone()
    for row, index in enumerate(indices.tolist()):
        if bool(write_mask[row]) and 0 <= index < state.shape[0]:
            result[index].copy_(rows[row])
    return result
```

Both PLE wrappers require a same-device int32/int64 index tensor and a floating
state on a Triton accelerator. Scatter without an explicit mask raises
`NotImplementedError`; there is no Torch loop fallback.

## Tests and benchmark

Correctness tests are in `tests/test_qwen4_self_kernels.py`. They cover BF16
and FP16 HC paths, non-power-of-two dimensions, paged/invalid QSA metadata,
invalid cache slots, interleaved MRoPE `[11, 11, 10]`, non-contiguous PLE
state, NULL `-1`, out-of-range indices, duplicate destinations, explicit
write masks, and repeated deterministic execution.

The reproducible benchmark entry is:

```shell
PYTHONPATH=src python benchmark/qwen4_self_kernels.py \
  --device cuda --rows 1,8,64 --warmup 50 --iters 500 \
  --output benchmark/results/qwen4_self_kernels.json
```

It runs correctness before timing, uses CUDA events with warmup and
synchronization, reports Torch-reference versus Triton latency and speedup,
and records device/dtype/shape metadata in JSON. The pytest smoke wrapper is
`benchmark/test_qwen4_self_kernels.py`.

The checked-in source evidence on an NVIDIA H100 reports CUDA-graph replay
speedups of 5.795–9.357x (HC RMSNorm), 3.880–5.253x (HC gate reduce),
5.137–5.197x (HC injection), 19.632–25.618x (QSA MQA dot), 2.648–3.548x
(QSA K/V store), 23.189–31.009x (QSA fused compression), 1.168–2.887x (PLE
gather), and 4.408–147.969x (PLE scatter) for rows 1/8/64. Those numbers are
source-worktree evidence, not a claim that an unverified backend has been
validated by them.

## Autotune and portability policy

These kernels retain the production launch constants used by the source
implementation (`BLOCK_N=32`, `BLOCK_D=128` for QSA MQA, `BLOCK_D=128` for
fused compression, and `_BLOCK_SIZE=1024` for PLE). They are fixed-shape
Qwen4 paths, so no autotune table is added in this extraction commit; this is
an explicit no-autotune exemption. A future tuning change must add a
`runtime.get_tuned_config()` entry and benchmark each target backend before it
changes these constants.

The device guard is backend-neutral (`device.type` not CPU/meta, same-device
inputs) and the kernel bodies use public Triton operations. H100/NVIDIA is the
only platform with current performance evidence. AMD, Hygon, and other
accelerators are not described as verified by this branch.
