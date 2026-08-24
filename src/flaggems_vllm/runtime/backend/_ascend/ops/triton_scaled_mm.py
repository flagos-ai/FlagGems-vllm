import torch
import triton
import triton.language as tl


def is_weak_contiguous(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def scaled_mm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACCUMULATOR_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_A: tl.constexpr,
    BLOCK_SIZE_SCALE_B: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # NOTE: Some tensor inputs are so large, they will cause int32 overflow
    # so it is necessary to use tl.int64 for all the offsets, else SEGV will
    # eventually occur.
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    last_k_offset = (num_k_blocks - 1) * BLOCK_SIZE_K
    a_ptrs = (
        a_ptr
        + stride_am * offs_m[:, None]
        + stride_ak * (offs_k[None, :] + last_k_offset)
    )
    b_ptrs = (
        b_ptr
        + stride_bk * (offs_k[:, None] + last_k_offset)
        + stride_bn * offs_n[None, :]
    )

    # First iteration
    if EVEN_K:
        if EVEN_M:
            a = tl.load(a_ptrs)
        else:
            masks_am = offs_m < M
            a = tl.load(a_ptrs, mask=masks_am[:, None])
        if EVEN_N:
            b = tl.load(b_ptrs)
        else:
            masks_bn = offs_n < N
            b = tl.load(b_ptrs, mask=masks_bn[None, :])
    else:
        masks_k = (offs_k + last_k_offset) < K
        if EVEN_M:
            a = tl.load(a_ptrs, mask=masks_k[None, :])
        else:
            masks_am = offs_m < M
            a = tl.load(a_ptrs, mask=masks_am[:, None] & masks_k[None, :])
        if EVEN_N:
            b = tl.load(b_ptrs, mask=masks_k[:, None])
        else:
            masks_bn = offs_n < N
            b = tl.load(b_ptrs, mask=masks_k[:, None] & masks_bn[None, :])

    accumulator = tl.dot(a, b, out_dtype=ACCUMULATOR_DTYPE)

    # Remaining iterations
    for k in tl.range(1, num_k_blocks):
        a_ptrs -= BLOCK_SIZE_K * stride_ak
        b_ptrs -= BLOCK_SIZE_K * stride_bk

        if EVEN_M:
            a = tl.load(a_ptrs)
        else:
            masks_am = offs_m < M
            a = tl.load(a_ptrs, mask=masks_am[:, None])
        if EVEN_N:
            b = tl.load(b_ptrs)
        else:
            masks_bn = offs_n < N
            b = tl.load(b_ptrs, mask=masks_bn[None, :])

        accumulator = tl.dot(a, b, accumulator, out_dtype=ACCUMULATOR_DTYPE)

    # Apply scale at end
    offs_sm = tl.arange(0, BLOCK_SIZE_SCALE_A)
    offs_sn = tl.arange(0, BLOCK_SIZE_SCALE_B)

    scale_a_ptrs = (
        scale_a_ptr + offs_sm + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M
    )
    scale_b_ptrs = (
        scale_b_ptr + offs_sn + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    )

    if EVEN_M:
        scale_a = tl.load(scale_a_ptrs)
    else:
        masks_scale_am = offs_sm + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M < M
        scale_a = tl.load(scale_a_ptrs, mask=masks_scale_am)
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M,))
    if EVEN_N:
        scale_b = tl.load(scale_b_ptrs)
    else:
        masks_scale_bn = offs_sn + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N < N
        scale_b = tl.load(scale_b_ptrs, mask=masks_scale_bn)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N,))

    # Fused scale
    accumulator *= scale_a[:, None] * scale_b[None, :]

    # Add bias, it's already in output format, so add it after conversion.
    if HAS_BIAS:
        if EVEN_N:
            bias = tl.load(bias_ptr + offs_n)
        else:
            bias_mask = offs_n < N
            bias = tl.load(bias_ptr + offs_n, bias_mask)
        accumulator += bias

    # Save output
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]

    # Auto Convert to output format
    if EVEN_M:
        if EVEN_N:
            tl.store(c_ptrs, accumulator)
        else:
            tl.store(c_ptrs, accumulator, mask=offs_n[None, :] < N)
    else:
        if EVEN_N:
            tl.store(c_ptrs, accumulator, mask=offs_m[:, None] < M)
        else:
            tl.store(
                c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
            )


# input   - [M, K]
# weight - [K, N]
def triton_scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: torch.Tensor | None = None,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
    use_heuristic=True,
) -> torch.Tensor:
    M, K = input.shape
    N = weight.shape[1]

    assert N > 0 and K > 0 and M > 0
    assert weight.shape[0] == K
    assert input.dtype == weight.dtype

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b

    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)
    assert out_dtype.is_floating_point
    assert bias is None or bias.is_floating_point()
    assert is_weak_contiguous(input)
    assert is_weak_contiguous(weight)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    result = torch.empty((M, N), dtype=out_dtype, device=input.device)

    has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1

    if use_heuristic:
        is_small_N = N < 8192
        next_power_of_2_M = max(32, triton.next_power_of_2(M))
        if next_power_of_2_M <= 32:
            tile_shape = (64, 64, 256) if is_small_N else (64, 128, 256)
        elif next_power_of_2_M <= 64:
            tile_shape = (64, 64, 256)
        elif next_power_of_2_M <= 128:
            tile_shape = (64, 128, 128)
        else:
            tile_shape = (128, 128, 128)

    block_size_m, block_size_n, block_size_k = tile_shape

    block_size_sa = 1 if has_scalar(scale_a) else block_size_m
    block_size_sb = 1 if has_scalar(scale_b) else block_size_n

    accumulator_dtype = tl.float32 if input.is_floating_point() else tl.int32

    # A = input, B = weight, C = result
    # A = M x K, B = K x N, C = M x N
    scaled_mm_kernel[grid](
        input,
        weight,
        scale_a,
        scale_b,
        result,
        bias,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        result.stride(0),
        result.stride(1),
        accumulator_dtype,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_SCALE_A=block_size_sa,
        BLOCK_SIZE_SCALE_B=block_size_sb,
        HAS_BIAS=bias is not None,
    )

    return result
