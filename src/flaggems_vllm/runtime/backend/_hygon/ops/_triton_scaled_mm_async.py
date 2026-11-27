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

from functools import lru_cache

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = None


# (M, N, K): ((BLOCK_M, BLOCK_N, BLOCK_K), num_stages, async B).
# These configurations were validated on gfx936 with the Llama FP8 trace.
_ASYNC_CONFIGS = {
    (1, 6144, 4096): ((16, 128, 64), 1, True),
    (1, 4096, 4096): ((16, 64, 64), 1, True),
    (1, 28672, 4096): ((16, 128, 64), 1, True),
    (1, 4096, 14336): ((16, 64, 64), 1, True),
    (32, 6144, 4096): ((32, 128, 64), 1, True),
    (32, 4096, 4096): ((16, 128, 64), 1, True),
    (32, 28672, 4096): ((32, 128, 128), 1, False),
    (32, 4096, 14336): ((16, 128, 64), 1, True),
    (128, 6144, 4096): ((64, 64, 64), 1, True),
    (128, 4096, 4096): ((64, 128, 64), 2, True),
    (128, 28672, 4096): ((64, 128, 64), 2, True),
    (128, 4096, 14336): ((64, 128, 64), 2, True),
}


@lru_cache(maxsize=1)
def _supports_async_fp8():
    if tle is None or not hasattr(tle, "load"):
        return False
    try:
        from triton.backends.compiler import GPUTarget
        from triton.compiler.compiler import make_backend

        backend = make_backend(GPUTarget("hip", "gfx936", 64))
        options = backend.parse_options({})
        copy_dtypes = getattr(options, "async_copy_dtypes", ())
        return "fp8e4nv" in copy_dtypes and "resolve_dot" in (
            backend.get_codegen_implementation(options)
        )
    except (AttributeError, ImportError, RuntimeError, ValueError):
        return False


def get_async_config(a, b, sa, sb, m, n, k):
    config = _ASYNC_CONFIGS.get((m, n, k))
    # Pointer alignment is available only when the launcher sees real tensors.
    if config is None or torch.compiler.is_compiling():
        return None
    if (
        a.is_cuda
        and a.device == b.device == sa.device == sb.device
        and a.stride() == (k, 1)
        and b.stride() == (1, k)
        and a.data_ptr() % 16 == 0
        and b.data_ptr() % 16 == 0
        and sa.dtype == torch.float32
        and sa.shape == (m, 1)
        and sb.shape == (n, 1)
        and sa.is_contiguous()
        and sb.is_contiguous()
        and _supports_async_fp8()
    ):
        props = torch.cuda.get_device_properties(a.device)
        if getattr(props, "gcnArchName", "").split(":", 1)[0] == "gfx936":
            return config
    return None


@triton.jit
def scaled_mm_async_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_A: tl.constexpr,
    BLOCK_SIZE_SCALE_B: tl.constexpr,
    ASYNC_B: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = tl.float32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # NOTE: Some tensor inputs are so large, they will cause int32 overflow
    # so it is necessary to use tl.int64 for all the offsets, else SEGV will
    # eventually occur.

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    # NOTE: BLOCK_SIZE_SCALE_A could be 1 or BLOCK_SIZE_M, so need to create
    # appropriate offsets and masks for each case. Same goes for
    # BLOCK_SIZE_SCALE_B.
    offsets_scale_am = (
        tl.arange(0, BLOCK_SIZE_SCALE_A)
        + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M
    )
    masks_scale_am = offsets_scale_am < M

    offsets_scale_bn = (
        tl.arange(0, BLOCK_SIZE_SCALE_B)
        + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    )
    masks_scale_bn = offsets_scale_bn < N

    scale_a_ptrs = scale_a_ptr + offsets_scale_am
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn

    tile_a = a_ptr + pid_m.to(tl.int64) * BLOCK_SIZE_M * stride_am
    tile_b = b_ptr + pid_n.to(tl.int64) * BLOCK_SIZE_N * stride_bn
    local_m = tl.arange(0, BLOCK_SIZE_M)
    local_n = tl.arange(0, BLOCK_SIZE_N)
    local_k = tl.arange(0, BLOCK_SIZE_K)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        kk = k * BLOCK_SIZE_K + local_k
        aa = tile_a + local_m[:, None] * stride_am + kk[None, :] * stride_ak
        bb = tile_b + kk[:, None] * stride_bk + local_n[None, :] * stride_bn
        if M % BLOCK_SIZE_M == 0 and K % BLOCK_SIZE_K == 0:
            a = tle.load(aa, is_async=True)
        else:
            a = tle.load(aa, masks_am[:, None] & (kk[None, :] < K), 0, is_async=True)
        if ASYNC_B:
            if N % BLOCK_SIZE_N == 0 and K % BLOCK_SIZE_K == 0:
                b = tle.load(bb, is_async=True)
            else:
                b = tle.load(
                    bb, (kk[:, None] < K) & masks_bn[None, :], 0, is_async=True
                )
        else:
            b = tl.load(bb, (kk[:, None] < K) & masks_bn[None, :], 0)
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

    # Apply scale at end.
    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_a = tl.load(scale_a_ptrs[:, None], masks_scale_a)
    # Need to broadcast to the appropriate size, if scale_a is already
    # (BLOCK_SIZE_M, 1) then it will broadcast to its own shape. Same goes
    # for scale_b below.
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))
    accumulator = scale_a * accumulator.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    accumulator = scale_b.T * accumulator.to(tl.float32)

    # Convert to output format.
    c = accumulator.to(c_ptr.type.element_ty)

    # Save output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)
