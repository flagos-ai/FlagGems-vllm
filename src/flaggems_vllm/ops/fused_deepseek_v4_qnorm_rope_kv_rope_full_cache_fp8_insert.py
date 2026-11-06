# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

import torch
import triton
import triton.language as tl


def _detect_launch_pdl_support():
    """Whether this Triton runtime accepts ``launch_pdl`` as a launch kwarg."""
    try:
        fn = triton.runtime.jit.JITFunction.run
        return "launch_pdl" in inspect.signature(fn).parameters
    except Exception:
        return False


def _detect_gdc_primitives_support():
    """Whether this Triton exposes the device-side PDL primitives.

    A PDL launch is only safe when the launch attribute is set *and* the kernel
    emits ``gdc_wait`` / ``gdc_launch_dependents``. Setting the attribute alone
    lets the kernel start while its producer is still writing (read-after-write
    race), so both halves are gated together.
    """
    try:
        return hasattr(tl.extra.cuda, "gdc_wait") and hasattr(
            tl.extra.cuda, "gdc_launch_dependents"
        )
    except Exception:
        return False


def _platform_supports_pdl(device=None) -> bool:
    """Platform gate for PDL, mirroring vLLM's ``is_arch_support_pdl()``.

    PDL needs Hopper or newer (sm90+) on CUDA. Every other backend stays off:
    the ROCm Triton backend rejects the ``launch_pdl`` kwarg outright (even
    when it is False), and XPU / CPU builds have no programmatic launch at all.
    """
    if torch.version.hip is not None:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return major >= 9


_LAUNCH_PDL_KWARG_SUPPORTED = _detect_launch_pdl_support()
_GDC_PRIMITIVES_SUPPORTED = _detect_gdc_primitives_support()


@triton.jit
def _fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert_kernel(
    q_ptr,
    kv_ptr,
    slot_mapping_ptr,
    position_ids_ptr,
    cos_sin_cache_ptr,
    fp8_scale_ptr,
    q_fp8_scale_inv_ptr,
    q_fp8_ptr,
    k_cache_ptr,
    N,
    H,
    num_tokens_insert,
    eps,
    k_cache_stride0,
    k_cache_stride1,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    NO_ROPE_OFFSET: tl.constexpr,
    FP8_MAX: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    APPLY_Q_NORM: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    pid = tl.program_id(0)

    # int32 row indices: total = N * (H + 1) and N * H * HEAD_DIM both fit.
    rows = pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    h1 = H + 1
    total = N * h1

    valid = rows < total
    token = rows // h1
    slot = rows - token * h1

    is_q = slot < H
    q_row_mask = valid & is_q
    kv_row_mask = valid & (slot == H)

    offs_d = tl.arange(0, HEAD_DIM)
    nope_cols = offs_d < NO_ROPE_OFFSET

    q_base = (token * H + slot).to(tl.int64) * HEAD_DIM
    kv_base = token.to(tl.int64) * HEAD_DIM

    # ------------------------------------------------------------------
    # 0) PDL: the index math above overlaps the producer's tail, then wait
    #    before the first touch of an upstream tensor. Mirrors the CUDA
    #    kernel, where cudaGridDependencySynchronize() sits after index
    #    computation and before the q/kv loads.
    # ------------------------------------------------------------------
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()

    # ------------------------------------------------------------------
    # 1) one merged full-row load: each row takes its base from q or kv
    # ------------------------------------------------------------------
    row_base_ptr = tl.where(
        is_q[:, None], q_ptr + q_base[:, None], kv_ptr + kv_base[:, None]
    )
    x = tl.load(row_base_ptr + offs_d[None, :], mask=valid[:, None], other=0.0).to(
        tl.float32
    )

    # ------------------------------------------------------------------
    # 2) weight-free RMSNorm on the Q side (fp32 accumulation). The rope
    #    tile below reuses `scale` (rotation commutes with it).
    # ------------------------------------------------------------------
    scale = tl.full((ROWS_PER_BLOCK,), 1.0, tl.float32)
    if APPLY_Q_NORM:
        mean_sq = tl.sum(x * x, axis=1) * (1.0 / HEAD_DIM)
        rstd = tl.rsqrt(mean_sq + eps)
        scale = tl.where(is_q, rstd, 1.0)
        x = x * scale[:, None]

    # ------------------------------------------------------------------
    # 3) per-tensor FP8 (e4m3) quantization, NoPE region only. The rope
    #    columns are stored from the dedicated tile in step 5.
    # ------------------------------------------------------------------
    kv_scale = 1.0 / tl.load(fp8_scale_ptr).to(tl.float32)
    q_scale = tl.load(q_fp8_scale_inv_ptr).to(tl.float32)
    row_scale = tl.where(is_q, q_scale, kv_scale)

    xq_f32 = x * row_scale[:, None]
    xq_f32 = tl.minimum(tl.maximum(xq_f32, -FP8_MAX), FP8_MAX)
    out_nope = xq_f32.to(tl.float8e4nv)

    tl.store(
        q_fp8_ptr + q_base[:, None] + offs_d[None, :],
        out_nope,
        mask=q_row_mask[:, None] & nope_cols[None, :],
    )

    # ------------------------------------------------------------------
    # 4) paged KV cache insertion, NoPE columns (int64 scatter into the
    #    cache; the rope columns follow in step 5).
    # ------------------------------------------------------------------
    tok_in_range = valid & (token < num_tokens_insert)
    sm = tl.load(slot_mapping_ptr + token, mask=tok_in_range, other=-1)
    sm_ok = sm >= 0
    sm_safe = tl.where(sm_ok, sm, 0)

    page = sm_safe // CACHE_BLOCK_SIZE
    page_off = sm_safe - page * CACHE_BLOCK_SIZE
    k_base = page * k_cache_stride0 + page_off * k_cache_stride1

    kv_store_mask = (kv_row_mask & sm_ok)[:, None]
    tl.store(
        k_cache_ptr + k_base[:, None] + offs_d[None, :],
        out_nope,
        mask=kv_store_mask & nope_cols[None, :],
    )

    # ------------------------------------------------------------------
    # 5) RoPE side tile: one (even, odd) pair per thread, cos/sin as a
    #    plain contiguous per-row load. No full-width reshape, no gather
    #    division, no identity-select for the NoPE region.
    # ------------------------------------------------------------------
    rope_offs = tl.arange(0, ROPE_DIM)
    rope_ptrs = row_base_ptr + (NO_ROPE_OFFSET + rope_offs)[None, :]
    x_rope = tl.load(rope_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)

    xr = tl.reshape(x_rope, (ROWS_PER_BLOCK, HALF_ROPE_DIM, 2))
    even, odd = tl.split(xr)

    pos = tl.load(position_ids_ptr + token, mask=valid, other=0).to(tl.int32)
    half_idx = tl.arange(0, HALF_ROPE_DIM)
    cos_base = cos_sin_cache_ptr + pos[:, None] * ROPE_DIM
    cos_v = tl.load(cos_base + half_idx[None, :], mask=valid[:, None], other=0.0)
    sin_v = tl.load(
        cos_base + half_idx[None, :] + HALF_ROPE_DIM, mask=valid[:, None], other=0.0
    )

    rot_e = even * cos_v - odd * sin_v
    rot_o = even * sin_v + odd * cos_v
    x_rope_rot = tl.reshape(tl.join(rot_e, rot_o), (ROWS_PER_BLOCK, ROPE_DIM))

    # Rotate-then-scale == scale-then-rotate (both linear); fold the norm
    # `scale` and the fp8 `row_scale` into one multiply.
    rope_scale = scale * row_scale
    xr_f32 = x_rope_rot * rope_scale[:, None]
    xr_f32 = tl.minimum(tl.maximum(xr_f32, -FP8_MAX), FP8_MAX)
    out_rope = xr_f32.to(tl.float8e4nv)

    rope_dst = (NO_ROPE_OFFSET + rope_offs)[None, :]
    tl.store(
        q_fp8_ptr + q_base[:, None] + rope_dst,
        out_rope,
        mask=q_row_mask[:, None],
    )
    tl.store(
        k_cache_ptr + k_base[:, None] + rope_dst,
        out_rope,
        mask=kv_store_mask,
    )

    # ------------------------------------------------------------------
    # 6) PDL: every store is issued, dependents may start. Mirrors the
    #    CUDA kernel's cudaTriggerProgrammaticLaunchCompletion() at exit.
    # ------------------------------------------------------------------
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _check_full_cache_fp8_insert_args(
    q,
    kv,
    q_fp8,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    fp8_scale,
    q_fp8_scale_inv,
    eps,
    cache_block_size,
    head_dim,
    rope_dim,
):
    """Validate the argument contract of the CUDA op this kernel mirrors."""
    named_tensors = (
        ("q", q),
        ("kv", kv),
        ("q_fp8", q_fp8),
        ("k_cache", k_cache),
        ("slot_mapping", slot_mapping),
        ("position_ids", position_ids),
        ("cos_sin_cache", cos_sin_cache),
        ("fp8_scale", fp8_scale),
        ("q_fp8_scale_inv", q_fp8_scale_inv),
    )
    for name, tensor in named_tensors:
        assert isinstance(tensor, torch.Tensor), f"{name} must be a torch.Tensor"
        assert tensor.is_cuda, f"{name} must be a CUDA tensor"
        assert tensor.device == q.device, f"{name} must be on the same device as q"

    # q / kv: read-only inputs, decoded as fp32 inside the kernel.
    assert q.dim() == 3, "q must be 3D [N, H, 512]"
    assert kv.dim() == 2, "kv must be 2D [N, 512]"
    assert q.is_contiguous(), "q must be contiguous"
    assert kv.is_contiguous(), "kv must be contiguous"
    assert q.dtype == kv.dtype, "q and kv dtype must match"
    assert q.dtype in (
        torch.float16,
        torch.bfloat16,
    ), "q/kv must be float16 or bfloat16"
    assert q.shape[2] == head_dim, f"q last dim must be {head_dim}"
    assert kv.shape[0] == q.shape[0], "kv must have the same number of rows as q"
    assert kv.shape[1] == head_dim, f"kv last dim must be {head_dim}"

    # q_fp8: pre-allocated FP8 output, matches q dim for dim.
    assert q_fp8.dtype == torch.float8_e4m3fn, "q_fp8 must be float8_e4m3fn"
    assert q_fp8.shape == q.shape, "q_fp8 must have the same shape as q"
    assert q_fp8.is_contiguous(), "q_fp8 must be contiguous"

    # k_cache: paged FP8 cache, updated in place.
    assert k_cache.dim() == 3, "k_cache must be 3D [num_blocks, block_size, 512]"
    assert k_cache.dtype == torch.float8_e4m3fn, "k_cache must be float8_e4m3fn"
    assert k_cache.stride(2) == 1, "k_cache last dim must be contiguous"
    assert (
        k_cache.shape[1] == cache_block_size
    ), "k_cache.size(1) must equal cache_block_size"
    assert k_cache.shape[2] == head_dim, f"k_cache last dim must be {head_dim}"
    assert (
        isinstance(cache_block_size, int) and cache_block_size > 0
    ), "cache_block_size must be a positive int"

    # Index tensors (contiguity: the kernel indexes them flat, exactly like
    # the CUDA op -- assert instead of silently reading wrong addresses).
    assert slot_mapping.dim() == 1, "slot_mapping must be 1D"
    assert slot_mapping.dtype == torch.int64, "slot_mapping must be int64"
    assert slot_mapping.numel() <= q.shape[0], "slot_mapping.numel() must be <= N"
    assert slot_mapping.is_contiguous(), "slot_mapping must be contiguous"
    assert position_ids.dim() == 1, "position_ids must be 1D"
    assert position_ids.dtype == torch.int64, "position_ids must be int64"
    assert position_ids.numel() == q.shape[0], "position_ids.numel() must equal N"
    assert position_ids.is_contiguous(), "position_ids must be contiguous"

    # int32 in-kernel index bounds: `rows`/`token`/`slot` are 32-bit, and the
    # rope path multiplies positions by ROPE_DIM in 32-bit. The row bases are
    # widened to int64 afterwards, so only these products need the guard.
    assert (
        q.shape[0] * (q.shape[1] + 1) < 2**31
    ), "N * (H + 1) must fit int32 in-kernel row indexing"

    # RoPE table: cos[0:32] || sin[32:64] per row, GPT-J interleaved pairing.
    assert cos_sin_cache.dim() == 2, "cos_sin_cache must be 2D [max_pos, 64]"
    assert cos_sin_cache.dtype == torch.float32, "cos_sin_cache must be float32"
    assert (
        cos_sin_cache.shape[1] == rope_dim
    ), f"cos_sin_cache.size(1) must equal {rope_dim}"
    assert (
        cos_sin_cache.is_contiguous()
    ), "cos_sin_cache must be contiguous (kernel assumes row stride = rope_dim)"
    assert (
        cos_sin_cache.shape[0] * rope_dim < 2**31
    ), "max_pos * rope_dim must fit int32 in the rope index path"

    # Per-tensor FP8 scales: both are length-1 float32 tensors.
    for name, tensor in (
        ("fp8_scale", fp8_scale),
        ("q_fp8_scale_inv", q_fp8_scale_inv),
    ):
        assert tensor.dtype == torch.float32, f"{name} must be float32"
        assert tensor.numel() == 1, f"{name} must contain exactly one element"

    assert eps is not None, "eps must be provided, the CUDA op has no default"


def fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
    q,
    kv,
    q_fp8,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    fp8_scale,
    q_fp8_scale_inv,
    eps,
    cache_block_size,
    apply_q_norm=True,
):
    head_dim = 512
    rope_dim = 64

    _check_full_cache_fp8_insert_args(
        q,
        kv,
        q_fp8,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        fp8_scale,
        q_fp8_scale_inv,
        eps,
        cache_block_size,
        head_dim,
        rope_dim,
    )

    N, H, _ = q.shape

    # Empty batch means nothing to do. H == 0 is NOT an early return: the CUDA
    # op still processes the KV-only rows (slotsPerToken == 1), so the kernel
    # must run for the cache insert.
    if N == 0:
        return

    half_rope_dim = rope_dim // 2
    no_rope_offset = head_dim - rope_dim
    fp8_max = 448.0

    rows_per_block = 8
    total_rows = N * (H + 1)
    grid = (triton.cdiv(total_rows, rows_per_block),)

    # PDL is a launch attribute plus device-side primitives; both must be on.
    enable_pdl = (
        _LAUNCH_PDL_KWARG_SUPPORTED
        and _GDC_PRIMITIVES_SUPPORTED
        and _platform_supports_pdl(q.device)
    )
    launch_kwargs = {"num_warps": 8, "num_stages": 1}
    if enable_pdl:
        launch_kwargs["launch_pdl"] = True

    _fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert_kernel[grid](
        q,
        kv,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        fp8_scale,
        q_fp8_scale_inv,
        q_fp8,
        k_cache,
        N,
        H,
        int(slot_mapping.numel()),
        float(eps),
        int(k_cache.stride(0)),
        int(k_cache.stride(1)),
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_dim,
        HALF_ROPE_DIM=half_rope_dim,
        NO_ROPE_OFFSET=no_rope_offset,
        FP8_MAX=fp8_max,
        ROWS_PER_BLOCK=rows_per_block,
        CACHE_BLOCK_SIZE=int(cache_block_size),
        APPLY_Q_NORM=bool(apply_q_norm),
        LAUNCH_PDL=enable_pdl,
        **launch_kwargs,
    )
