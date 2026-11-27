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

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn


@triton.jit
def _l2norm_exact(X, Y, eps, ROWS: tl.constexpr, BD: tl.constexpr):
    row = tl.arange(0, ROWS)[:, None]
    col = tl.arange(0, BD)[None, :]
    offs = row * BD + col
    x16 = tl.load(X + offs)
    x = x16.to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 1) + eps)
    tl.store(Y + offs, x16 * r[:, None].to(Y.dtype.element_ty))


@triton.jit
def _l2norm_mask(X, Y, eps, M, N, BD: tl.constexpr, MBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_idx = tl.arange(0, MBLOCK)[:, None]
    col_idx = tl.arange(0, BD)[None, :]
    mask = ((row_idx + pid * MBLOCK) < M) & (col_idx < N)
    offs = pid * MBLOCK * N + row_idx * N + col_idx
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 1) + eps)
    tl.store(Y + offs, x * r[:, None], mask=mask)


def _launch(X, Y, eps, M, N, BD, MBLOCK, *, grid=None, warmup=False):
    m = int(M)
    n = int(N)
    bd = int(BD)
    e = float(eps)
    if n == bd and m >= 1 and (m & (m - 1)) == 0 and m <= 64 and X.dtype == Y.dtype:
        _l2norm_exact[(1,)](X, Y, e, ROWS=m, BD=bd)
    else:
        mb = int(MBLOCK)
        _l2norm_mask[(triton.cdiv(m, mb),)](X, Y, e, m, n, bd, mb)


def l2norm_fwd(x, eps=1e-6, output_dtype=None):
    """Forward-only dense row normalization with an FP32 sum of squares."""
    if x.device.type != "npu" or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("expected contiguous BF16 NPU input")
    if x.ndim < 1 or not 1 <= x.shape[-1] <= 1024:
        raise ValueError("normalization width must be in [1, 1024]")
    if output_dtype is None:
        output_dtype = x.dtype
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("output dtype must be BF16 or FP32")
    n = x.shape[-1]
    m = x.numel() // n
    y = torch.empty_like(x, dtype=output_dtype)
    if m:
        with torch_device_fn.device(x.device):
            _launch(x, y, eps, m, n, triton.next_power_of_2(n), 32)
    return y
