"""Pure-Triton dense grouped fused-MoE for MUSA.

The layout follows the grouped-contiguous idea: assign every routed token to
its expert, pad each expert group to a multiple of BLOCK_M, then run one dense
grouped GEMM over the padded matrix with a per-tile expert id.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _histogram_kernel(ids_ptr, counts_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    e = tl.load(ids_ptr + offs, mask=mask, other=0)
    tl.atomic_add(counts_ptr + e, 1, mask=mask)


@triton.jit
def _padded_offsets_kernel(
    counts_ptr,
    offsets_ptr,
    E: tl.constexpr,
    BM: tl.constexpr,
):
    acc = 0
    for e in range(E):
        c = tl.load(counts_ptr + e)
        tl.store(offsets_ptr + e, acc)
        nt = tl.cdiv(c, BM)
        acc += nt * BM
    tl.store(offsets_ptr + E, acc)


@triton.jit
def _fill_m_indices_kernel(
    offsets_ptr,
    counts_ptr,
    m_indices_ptr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    e = tl.program_id(0)
    base = tl.load(offsets_ptr + e)
    c = tl.load(counts_ptr + e)
    n = tl.cdiv(c, BM) * BM
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(m_indices_ptr + base + offs, e, mask=mask)


@triton.jit
def _fill_neg_kernel(ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offs, -1, mask=offs < n)


@triton.jit
def _assign_padded_kernel(
    ids_ptr,
    offsets_ptr,
    counters_ptr,
    inv_perm_ptr,
    row_token_ptr,
    numel,
    topk: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    e = tl.load(ids_ptr + offs, mask=mask, other=0)
    base = tl.load(offsets_ptr + e, mask=mask, other=0)
    local = tl.atomic_add(counters_ptr + e, 1, mask=mask)
    pos = base + local
    token = offs // topk
    j = offs % topk
    tl.store(inv_perm_ptr + token * topk + j, pos, mask=mask)
    tl.store(row_token_ptr + pos, token, mask=mask)


@triton.jit
def _compact_padded_kernel(
    src_ptr,
    dst_ptr,
    row_token_ptr,
    total,
    hidden,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    row_mask = rows < total
    col_mask = cols < hidden
    token = tl.load(row_token_ptr + rows, mask=row_mask, other=-1)
    valid = (token >= 0) & row_mask
    x = tl.load(
        src_ptr + token[:, None] * hidden + cols[None, :],
        mask=valid[:, None] & col_mask[None, :],
        other=0.0,
    )
    tl.store(
        dst_ptr + rows[:, None] * hidden + cols[None, :],
        x,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m_indices_ptr,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BN)
    pm = pid // num_n
    pn = pid % num_n
    expert = tl.load(m_indices_ptr + pm * BM)

    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = (
        b_ptr
        + expert * K * N
        + offs_k[:, None] * N
        + offs_n[None, :]
    )
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


@triton.jit
def _silu_mul_kernel(src_ptr, dst_ptr, total, n_inter: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    row = offs // n_inter
    col = offs % n_inter
    gate = tl.load(src_ptr + row * (2 * n_inter) + col, mask=mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(
        src_ptr + row * (2 * n_inter) + n_inter + col, mask=mask, other=0.0
    ).to(tl.float32)
    y = gate * tl.sigmoid(gate) * up
    tl.store(dst_ptr + offs, y.to(dst_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _unpermute_kernel(
    src_ptr,
    out_ptr,
    inv_perm_ptr,
    weights_ptr,
    num_tokens,
    hidden,
    topk: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK + tl.arange(0, BLOCK)
    mask_h = offs_h < hidden
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.static_range(topk):
        row = tl.load(inv_perm_ptr + pid_t * topk + j)
        w = tl.load(weights_ptr + pid_t * topk + j)
        vals = tl.load(src_ptr + row * hidden + offs_h, mask=mask_h, other=0)
        acc += vals.to(tl.float32) * w.to(tl.float32)
    tl.store(
        out_ptr + pid_t * hidden + offs_h,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_h,
    )


_TW_CACHE = {}


def _transpose_weights(w):
    key = (w.data_ptr(), w.shape, w.dtype)
    entry = _TW_CACHE.get(key)
    if entry is not None and entry[0]() is w:
        return entry[1]
    t = w.permute(0, 2, 1).contiguous()
    _TW_CACHE[key] = (__import__("weakref").ref(w), t)
    return t


_CACHE = {}


def _get_buffers(M, E, H, I, topk, device, dtype, max_tokens):
    key = (M, E, H, I, topk, str(device), dtype)
    entry = _CACHE.get(key)
    if entry is not None:
        return entry
    if _CACHE:
        _CACHE.clear()
        if hasattr(torch, "musa"):
            torch.musa.empty_cache()
    meta = torch.zeros(3 * E + 1, dtype=torch.int32, device=device)
    row_token = torch.empty(max_tokens, dtype=torch.int32, device=device)
    inv_perm = torch.empty(M * topk, dtype=torch.int32, device=device)
    m_indices = torch.empty(max_tokens, dtype=torch.int32, device=device)
    hidden_perm = torch.empty(max_tokens, H, dtype=dtype, device=device)
    gate_up = torch.empty(max_tokens, 2 * I, dtype=dtype, device=device)
    act = torch.empty(max_tokens, I, dtype=dtype, device=device)
    entry = (
        meta,
        row_token,
        inv_perm,
        m_indices,
        hidden_perm,
        gate_up,
        act,
    )
    _CACHE[key] = entry
    return entry


def dense_fused_moe(hidden_states, w1, w2, topk_weights, topk_ids):
    """Run the dense grouped (padded-contiguous) BF16/fp16 path."""
    M, H = hidden_states.shape
    E, N, K = w1.shape
    I = N // 2
    if I < 256 or N % 128 or H % 64 or I % 64:
        raise NotImplementedError("dense grouped MoE requires I>=256 and 64/128 alignment")
    topk = topk_ids.size(1)
    numel = M * topk
    device = hidden_states.device
    dtype = hidden_states.dtype
    BM = 128
    max_tokens = ((numel + E * (BM - 1) + BM - 1) // BM) * BM
    (
        meta,
        row_token,
        inv_perm,
        m_indices,
        hidden_perm,
        gate_up,
        act,
    ) = _get_buffers(M, E, H, I, topk, device, dtype, max_tokens)
    counts = meta[:E]
    offsets = meta[E : 2 * E + 1]
    counters = meta[2 * E + 1 :]
    meta.zero_()

    _histogram_kernel[(triton.cdiv(numel, 4096),)](
        topk_ids.reshape(-1), counts, numel, BLOCK=4096
    )
    _padded_offsets_kernel[(1,)](counts, offsets, E=E, BM=BM)
    total = int(offsets[E].item())

    _fill_m_indices_kernel[(E,)](
        offsets, counts, m_indices, E=E, BM=BM, BLOCK=256
    )
    _fill_neg_kernel[(triton.cdiv(total, 1024),)](row_token, total, BLOCK=1024)
    _assign_padded_kernel[(triton.cdiv(numel, 1024),)](
        topk_ids.reshape(-1),
        offsets,
        counters,
        inv_perm,
        row_token,
        numel,
        topk=topk,
        BLOCK=1024,
    )
    _compact_padded_kernel[
        (triton.cdiv(total, 8), triton.cdiv(H, 256))
    ](
        hidden_states,
        hidden_perm,
        row_token,
        total,
        H,
        BLOCK_M=8,
        BLOCK_H=256,
        num_warps=4,
    )

    w1t = _transpose_weights(w1)
    w2t = _transpose_weights(w2)

    g1 = (total // BM) * triton.cdiv(2 * I, 128)
    _grouped_gemm_kernel[(g1,)](
        hidden_perm,
        w1t,
        gate_up,
        m_indices,
        total,
        2 * I,
        H,
        BM=128,
        BN=128,
        BK=64,
        num_warps=4,
        num_stages=2,
    )
    _silu_mul_kernel[(triton.cdiv(total * I, 1024),)](
        gate_up,
        act,
        total * I,
        n_inter=I,
        BLOCK=1024,
        num_warps=4,
    )
    g2 = (total // BM) * triton.cdiv(H, 128)
    _grouped_gemm_kernel[(g2,)](
        act,
        w2t,
        hidden_perm,
        m_indices,
        total,
        H,
        I,
        BM=128,
        BN=128,
        BK=64,
        num_warps=4,
        num_stages=2,
    )
    out = torch.empty((M, H), dtype=dtype, device=device)
    _unpermute_kernel[(M, triton.cdiv(H, 512))](
        hidden_perm,
        out,
        inv_perm,
        topk_weights,
        M,
        H,
        topk=topk,
        BLOCK=512,
    )
    return out
