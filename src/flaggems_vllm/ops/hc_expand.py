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

"""Triton implementation of the mHC stream expansion operation."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_expand"),
    key=["numel", "H", "N"],
    strategy=["log", "default", "default"],
)
@triton.jit
def _hc_expand_kernel(
    x,
    out,
    numel,
    H: tl.constexpr,
    N: tl.constexpr,
    STRIDE_M: tl.constexpr,
    STRIDE_H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = index < numel
    row = index // H
    col = index - row * H
    value = tl.load(x + row * STRIDE_M + col * STRIDE_H, mask=mask, other=0)
    for stream in tl.static_range(N):
        tl.store(out + (row * N + stream) * H + col, value, mask=mask)


def hc_expand(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """``x.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()``.

    Each input element is loaded once and written to all ``hc_mult`` output
    streams. ``x`` may be a non-contiguous rank-2 tensor.
    """
    m, h = x.shape
    out = x.new_empty(m, hc_mult, h)
    if out.numel():
        numel = m * h
        grid = lambda META: (triton.cdiv(numel, META["BLOCK"]),)
        _hc_expand_kernel[grid](
            x,
            out,
            numel,
            h,
            hc_mult,
            *x.stride(),
        )
    return out
