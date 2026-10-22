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

"""Constant-specialized NVIDIA block schedule; no TLE/reorder dependency."""

import torch
import triton
import triton.language as tl

from ._per_token_group_quant_fp8_common import _clamp_like_upstream, _is_h20

# Fixed schedules qualified with the native H20 BF16 CUDA Graph benchmark.
# Explicit autotune exemption: preserve the qualified configuration and bounded
# register footprint; no online tuning or compiler-reorder capability is needed.
_H20_BLOCK_CONFIGS = {
    (7, 512, 512, False): (1, 1),
    (7, 512, 512, True): (1, 1),
    (7, 4096, 256, False): (1, 1),
    (7, 4096, 256, True): (1, 1),
    (83, 512, 64, False): (8, 1),
    (83, 512, 64, True): (8, 1),
    (2048, 4096, 256, False): (8, 1),
    (2048, 4096, 256, True): (16, 4),
    (2048, 13824, 512, True): (8, 4),
    (4096, 13824, 512, True): (16, 4),
    (8192, 13824, 512, True): (16, 4),
}


def get_h20_block_config(
    x, group_size, fp8_dtype, eps, column_major_scales, scale_ue8m0
):
    if (
        x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or fp8_dtype != torch.float8_e4m3fn
        or x.ndim != 2
        or eps != 1e-10
        or column_major_scales
        or not isinstance(scale_ue8m0, bool)
        or not x.is_contiguous()
    ):
        return None
    config = _H20_BLOCK_CONFIGS.get((x.shape[0], x.shape[1], group_size, scale_ue8m0))
    if config is None or not _is_h20(x.device.index):
        return None
    return config


@triton.jit
def _quantize_block(
    X,
    Q,
    S,
    TOTAL: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    UE: tl.constexpr,
):
    groups = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    if TOTAL % ROWS == 0:
        valid = tl.full((ROWS,), True, tl.int1)
    else:
        valid = groups < TOTAL
    offsets = groups[:, None] * GROUP + cols[None, :]
    values = tl.load(X + offsets, valid[:, None] & (cols[None, :] < GROUP), other=0).to(
        tl.float32
    )
    maximum = tl.maximum(tl.max(tl.abs(values), axis=1), 1e-10)
    scale = maximum * (1.0 / 448.0)
    if UE:
        scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scale), 1e-10))))
    q = _clamp_like_upstream(values / scale[:, None]).to(Q.dtype.element_ty)
    tl.store(Q + offsets, q, valid[:, None] & (cols[None, :] < GROUP))
    tl.store(S + groups, scale, valid)


def quantize_group_block(x, group_size, *, scale_ue8m0, rows, num_warps):
    """Qualified tile path; its gains are separate from load reorder."""
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty(
        (x.shape[0], x.shape[1] // group_size), device=x.device, dtype=torch.float32
    )
    _quantize_block[(triton.cdiv(x.numel() // group_size, rows),)](
        x,
        q,
        s,
        TOTAL=x.numel() // group_size,
        GROUP=group_size,
        BLOCK=triton.next_power_of_2(group_size),
        ROWS=rows,
        UE=scale_ue8m0,
        num_warps=num_warps,
        num_stages=1,
    )
    return q, s
