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

"""Marlin GEMM for vLLM. W4A16 (uint4b8) only; other paths raise.

Consumes vLLM Marlin-repacked weights (gptq_marlin_repack output):
int32 (K//16, N*2) tensor organized as (K//16, N//64) units of 128 words,
each unit holding one 16x64 (k, n) weight tile. Scales follow
marlin_permute_scales layout (K//gs, N) with the 64-column scale_perm.
Unlike fused_marlin_moe, the repack step is required, not rejected.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry, libtuner

# vllm.scalar_types.uint4b8.id: exponent=0, mantissa=4, bias=8, nan_repr=IEEE_754
UINT4B8_ID = (4 << 8) | (8 << 17) | (1 << 50)

_M_TILE = 16
_N_TILE = 64
_K_TILE = 16

_W4A16_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=3),
]


@libentry()
@libtuner(
    configs=_W4A16_CONFIGS,
    key=["size_m", "size_n", "size_k"],
    strategy=["log", "log", "log"],
)
@triton.jit(do_not_specialize=["size_m"])
def _marlin_gemm_w4a16_kernel(
    a_ptr,
    b_q_ptr,
    b_scales_ptr,
    bias_ptr,
    c_ptr,
    size_m,
    size_n,
    size_k,
    size_g,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < size_m
    kk = tl.arange(0, BLOCK_K)
    n_loc = tl.arange(0, BLOCK_N)

    # Marlin word address within a 128-word unit.
    waddr = (
        4 * (((n_loc % 8) * 4)[None, :] + ((kk % 8) // 2)[:, None])
        + (n_loc // 16)[None, :]
    )
    J = (4 * (kk % 2) + 2 * (kk // 8))[:, None] + ((n_loc // 8) % 2)[None, :]

    scale_row_stride = size_n
    # Inverse of vLLM get_scale_perm within each 64-column chunk:
    # original column n sits at physical (n%8)*8 + n//8.
    rn_phys = pid_n * BLOCK_N + (n_loc % 8) * 8 + n_loc // 8
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, size_k // BLOCK_K):
        W = tl.load(b_q_ptr + (kb * (size_n // BLOCK_N) + pid_n) * 128 + waddr)
        b_dtype = a_ptr.dtype.element_ty
        nib = (((W >> (4 * J)) & 15).to(tl.float32) - 8.0).to(b_dtype)
        a = tl.load(
            a_ptr + rm[:, None] * size_k + (kb * BLOCK_K + kk)[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        blk = tl.dot(a, nib)
        g = ((kb * BLOCK_K) // size_g).to(tl.int32)
        sn = tl.load(b_scales_ptr + g * scale_row_stride + rn_phys)
        acc += blk * sn[None, :].to(tl.float32)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + rn)
        acc += bias[None, :].to(tl.float32)

    out = acc.to(c_ptr.dtype.element_ty)
    tl.store(
        c_ptr + rm[:, None] * size_n + rn[None, :],
        out,
        mask=mask_m[:, None],
    )


def _b_q_type_id(b_q_type) -> int:
    return b_q_type.id if hasattr(b_q_type, "id") else int(b_q_type)


def marlin_gemm(
    a: torch.Tensor,
    c: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_bias: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    a_scales: Optional[torch.Tensor],
    global_scale: Optional[torch.Tensor],
    b_zeros: Optional[torch.Tensor],
    g_idx: Optional[torch.Tensor],
    perm: Optional[torch.Tensor],
    workspace: torch.Tensor,
    b_q_type,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    if _b_q_type_id(b_q_type) != UINT4B8_ID:
        raise NotImplementedError(
            f"quant type id {_b_q_type_id(b_q_type)} not supported, "
            "only W4A16 uint4b8 is implemented"
        )
    if g_idx is not None or perm is not None:
        raise NotImplementedError("act_order (g_idx/perm) is not supported")
    if a_scales is not None or global_scale is not None:
        raise NotImplementedError("quantized activations are not supported")
    if b_zeros is not None or is_zp_float:
        raise NotImplementedError("asymmetric zero-point is not supported")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError(f"activation dtype {a.dtype} not supported")
    if a.dtype != b_scales.dtype:
        raise NotImplementedError("b_scales dtype must match activation dtype")
    if size_k % _K_TILE or size_n % _N_TILE:
        raise NotImplementedError(
            f"size_k must be a multiple of {_K_TILE}, size_n of {_N_TILE}"
        )
    if b_q_weight.dtype != torch.int32 or tuple(b_q_weight.shape) != (
        size_k // _K_TILE,
        size_n * 2,
    ):
        raise NotImplementedError(
            "b_q_weight must be int32 marlin-repacked (K//16, N*2)"
        )
    num_groups = b_scales.shape[0]
    if num_groups == 0 or size_k % num_groups or (size_k // num_groups) % _K_TILE:
        raise NotImplementedError("group size must be a multiple of 16 dividing size_k")
    if tuple(b_scales.shape) != (num_groups, size_n):
        raise NotImplementedError("b_scales must be marlin-permuted (K//gs, N)")

    assert a.is_cuda and a.is_contiguous()
    assert b_q_weight.is_contiguous() and b_scales.is_contiguous()

    if c is None:
        c = torch.empty((size_m, size_n), dtype=a.dtype, device=a.device)
    else:
        assert c.shape == (size_m, size_n) and c.dtype == a.dtype and c.is_contiguous()

    size_g = size_k // num_groups
    if b_bias is not None:
        assert b_bias.dtype == a.dtype and tuple(b_bias.shape) == (size_n,)
        bias_ptr = b_bias
        has_bias = True
    else:
        bias_ptr = workspace if workspace.numel() > 0 else c
        has_bias = False

    grid = (triton.cdiv(size_m, _M_TILE), size_n // _N_TILE)
    _marlin_gemm_w4a16_kernel[grid](
        a,
        b_q_weight,
        b_scales,
        bias_ptr,
        c,
        size_m,
        size_n,
        size_k,
        size_g,
        HAS_BIAS=has_bias,
        BLOCK_M=_M_TILE,
        BLOCK_N=_N_TILE,
        BLOCK_K=_K_TILE,
    )
    return c
