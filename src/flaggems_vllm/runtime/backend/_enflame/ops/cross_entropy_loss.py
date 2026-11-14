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

import logging
import os

import torch
import triton
import triton.language as tl
from torch.nn import _reduction as _Reduction

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry
from flaggems_vllm.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _tgt_words(tgt):
    # Zero-copy int32 word view for index targets. torch_gcu keeps ATen int64
    # values either as compact int32 words (element j at word j) or, with
    # TORCH_GCU_ENABLE_INT64_AND_UINT64=1, in the low word of each 8-byte slot
    # (word 2*j); either way a Triton int64 load at the 8-byte dtype stride
    # reads neighbor-joined garbage, so targets are read through an int32 view
    # with the matching word stride (see indexer_k_quant_and_cache). Class
    # indices and ignore_index fit the low word, so this matches the
    # truncation semantics of an explicit cast without the extra kernel.
    if tgt.dtype == torch.int32:
        return tgt, 1
    if tgt.dtype == torch.int64 and tgt.device.type == "gcu" and tgt.ndim >= 1:
        word_stride = (
            2 if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") == "1" else 1
        )
        return tgt.contiguous().view(torch.int32), word_stride
    return tgt.to(torch.int32), 1


def _ce_configs():
    return runtime.get_tuned_config("cross_entropy_loss") + [
        triton.Config({"BLOCK_C": 1024, "BLOCK_D": 1}, num_warps=4),
        triton.Config({"BLOCK_C": 2048, "BLOCK_D": 1}, num_warps=4),
        triton.Config({"BLOCK_C": 4096, "BLOCK_D": 1}, num_warps=4),
        triton.Config({"BLOCK_C": 4096, "BLOCK_D": 1}, num_warps=8),
    ]


# Row-loop arm: GCU400 CTA dispatch costs ~77ns/CTA, so per-row grids with
# N >> 48 programs pay a dispatch floor that dominates the kernel itself for
# small per-row work. Large-N calls instead cap the grid and let each program
# loop over a contiguous chunk of rows (one launch, no wave dispatch gaps).
_ROW_PROG_CAP = 48
_ROWS_MIN_N = 49


@libentry()
@triton.autotune(
    configs=_ce_configs(),
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["ignore_index"])
def celoss_indices_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    out_ptr,
    w_tgt_ptr,
    ignore_index,
    C,
    D,
    TGT_WORD_STRIDE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = ext.program_id(0)
    pid_d = ext.program_id(1)
    d_start = pid_d * BLOCK_D
    d_stop = tl.minimum(d_start + BLOCK_D, D)
    for d in range(d_start, d_stop):
        offset_c = tl.arange(0, BLOCK_C)
        tgt = tl.load(tgt_ptr + (pid_n * D + d) * TGT_WORD_STRIDE)
        valid = tgt != ignore_index
        tgt_c = tl.minimum(tl.maximum(tgt, 0), C - 1).to(tl.int32)
        inp_base = inp_ptr + pid_n * C * D + d

        tmp_max = tl.zeros([BLOCK_C], dtype=tl.float32)
        tmp_sum = tl.zeros([BLOCK_C], dtype=tl.float32)
        for off in range(0, C, BLOCK_C):
            offset = off + offset_c
            mask = offset < C
            inp = tl.load(inp_base + offset * D, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            cur_max = tl.maximum(tmp_max, inp)
            cur_exp = tl.exp(inp - cur_max)
            tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
            tmp_max = cur_max
        final_max = tl.max(tmp_max, axis=0)
        final_sum = tl.log(tl.sum(tmp_sum * tl.exp(tmp_max - final_max), axis=0))

        inp_tgt = tl.load(inp_base + tgt_c * D).to(tl.float32)
        out = final_sum + final_max - inp_tgt
        if w_ptr is None:
            w_tgt = tl.where(valid, 1.0, 0.0)
        else:
            w_tgt = tl.where(valid, tl.load(w_ptr + tgt_c).to(tl.float32), 0.0)
        tl.store(w_tgt_ptr + pid_n * D + d, w_tgt)
        tl.store(out_ptr + pid_n * D + d, out * w_tgt)


@libentry()
@triton.autotune(
    configs=_ce_configs(),
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["ignore_index"])
def celoss_indices_kernel_rows(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    out_ptr,
    w_tgt_ptr,
    ignore_index,
    C,
    D,
    N,
    rows_per_cta,
    TGT_WORD_STRIDE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_p = ext.program_id(0)
    pid_d = ext.program_id(1)
    d_start = pid_d * BLOCK_D
    d_stop = tl.minimum(d_start + BLOCK_D, D)
    n_start = pid_p * rows_per_cta
    n_stop = tl.minimum(n_start + rows_per_cta, N)
    offset_c = tl.arange(0, BLOCK_C)
    for n in range(n_start, n_stop):
        for d in range(d_start, d_stop):
            tgt = tl.load(tgt_ptr + (n * D + d) * TGT_WORD_STRIDE)
            valid = tgt != ignore_index
            tgt_c = tl.minimum(tl.maximum(tgt, 0), C - 1).to(tl.int32)
            inp_base = inp_ptr + n * C * D + d

            tmp_max = tl.zeros([BLOCK_C], dtype=tl.float32)
            tmp_sum = tl.zeros([BLOCK_C], dtype=tl.float32)
            for off in range(0, C, BLOCK_C):
                offset = off + offset_c
                mask = offset < C
                inp = tl.load(inp_base + offset * D, mask=mask, other=-float("inf")).to(
                    tl.float32
                )
                cur_max = tl.maximum(tmp_max, inp)
                cur_exp = tl.exp(inp - cur_max)
                tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
                tmp_max = cur_max
            final_max = tl.max(tmp_max, axis=0)
            final_sum = tl.log(tl.sum(tmp_sum * tl.exp(tmp_max - final_max), axis=0))

            inp_tgt = tl.load(inp_base + tgt_c * D).to(tl.float32)
            out = final_sum + final_max - inp_tgt
            if w_ptr is None:
                w_tgt = tl.where(valid, 1.0, 0.0)
            else:
                w_tgt = tl.where(valid, tl.load(w_ptr + tgt_c).to(tl.float32), 0.0)
            tl.store(w_tgt_ptr + n * D + d, w_tgt)
            tl.store(out_ptr + n * D + d, out * w_tgt)


@libentry()
@triton.autotune(
    configs=_ce_configs(),
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["label_smoothing"])
def celoss_probability_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    out_ptr,
    label_smoothing,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = ext.program_id(0)
    pid_d = ext.program_id(1)
    d_start = pid_d * BLOCK_D
    d_stop = tl.minimum(d_start + BLOCK_D, D)
    for d in range(d_start, d_stop):
        offset_c = tl.arange(0, BLOCK_C)
        inp_base = inp_ptr + pid_n * C * D + d
        tgt_base = tgt_ptr + pid_n * C * D + d

        tmp_max = tl.zeros([BLOCK_C], dtype=tl.float32)
        tmp_sum = tl.zeros([BLOCK_C], dtype=tl.float32)
        for off in range(0, C, BLOCK_C):
            offset = off + offset_c
            mask = offset < C
            inp = tl.load(inp_base + offset * D, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            cur_max = tl.maximum(tmp_max, inp)
            cur_exp = tl.exp(inp - cur_max)
            tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
            tmp_max = cur_max
        final_max = tl.max(tmp_max, axis=0)
        final_sum = tl.log(tl.sum(tmp_sum * tl.exp(tmp_max - final_max), axis=0))
        lse = final_sum + final_max

        _sum = tl.zeros([BLOCK_C], dtype=tl.float32)
        for off in range(0, C, BLOCK_C):
            offset = off + offset_c
            mask = offset < C
            inp = tl.load(inp_base + offset * D, mask=mask, other=0.0).to(tl.float32)
            tgt = tl.load(tgt_base + offset * D, mask=mask, other=0.0).to(tl.float32)
            tgt = tgt * (1.0 - label_smoothing) + label_smoothing / C
            if w_ptr is None:
                w = tl.where(mask, 1.0, 0.0)
            else:
                w = tl.load(w_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            _sum += (lse - inp) * tgt * w
        out = tl.sum(_sum, axis=0)
        tl.store(out_ptr + pid_n * D + d, out)


@libentry()
@triton.autotune(
    configs=_ce_configs(),
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["ignore_index", "label_smoothing"])
def celoss_indices_smooth_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    out_ptr,
    w_tgt_ptr,
    ignore_index,
    label_smoothing,
    C,
    D,
    TGT_WORD_STRIDE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = ext.program_id(0)
    pid_d = ext.program_id(1)
    d_start = pid_d * BLOCK_D
    d_stop = tl.minimum(d_start + BLOCK_D, D)
    for d in range(d_start, d_stop):
        offset_c = tl.arange(0, BLOCK_C)
        tgt = tl.load(tgt_ptr + (pid_n * D + d) * TGT_WORD_STRIDE)
        valid = tgt != ignore_index
        tgt_c = tl.minimum(tl.maximum(tgt, 0), C - 1).to(tl.int32)

        if w_ptr is None:
            w_tgt = tl.where(valid, 1.0, 0.0)
        else:
            w_tgt = tl.where(valid, tl.load(w_ptr + tgt_c).to(tl.float32), 0.0)
        tl.store(w_tgt_ptr + pid_n * D + d, w_tgt)

        inp_base = inp_ptr + pid_n * C * D + d

        tmp_max = tl.zeros([BLOCK_C], dtype=tl.float32)
        tmp_sum = tl.zeros([BLOCK_C], dtype=tl.float32)
        for off in range(0, C, BLOCK_C):
            offset = off + offset_c
            mask = offset < C
            inp = tl.load(inp_base + offset * D, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            cur_max = tl.maximum(tmp_max, inp)
            cur_exp = tl.exp(inp - cur_max)
            tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
            tmp_max = cur_max
        final_max = tl.max(tmp_max, axis=0)
        final_sum = tl.log(tl.sum(tmp_sum * tl.exp(tmp_max - final_max), axis=0))
        lse = final_sum + final_max

        _sum = tl.zeros([BLOCK_C], dtype=tl.float32)
        for off in range(0, C, BLOCK_C):
            offset = off + offset_c
            mask = offset < C
            inp = tl.load(inp_base + offset * D, mask=mask, other=0.0).to(tl.float32)
            if w_ptr is None:
                w = tl.where(mask, 1.0, 0.0)
            else:
                w = tl.load(w_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            smooth = tl.where(
                offset == tgt,
                1.0 - label_smoothing + label_smoothing / C,
                label_smoothing / C,
            )
            _sum += (lse - inp) * smooth * w
        out = tl.sum(_sum, axis=0)
        out = tl.where(valid, out, 0.0)
        tl.store(out_ptr + pid_n * D + d, out)


@libentry()
@triton.autotune(
    configs=_ce_configs(),
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["ignore_index", "mean_num"])
def celoss_indices_bwd(
    out_grad_ptr,
    inp_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    ignore_index,
    mean_num,
    C,
    D,
    TGT_WORD_STRIDE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = ext.program_id(0)
    pid_d = ext.program_id(1)
    d_start = pid_d * BLOCK_D
    d_stop = tl.minimum(d_start + BLOCK_D, D)
    for d in range(d_start, d_stop):
        offset_c = tl.arange(0, BLOCK_C)
        tgt = tl.load(tgt_ptr + (pid_n * D + d) * TGT_WORD_STRIDE)
        valid = tgt != ignore_index
        tgt_c = tl.minimum(tl.maximum(tgt, 0), C - 1).to(tl.int32)
        out_grad = tl.load(out_grad_ptr + pid_n * D + d).to(tl.float32)

        if w_ptr is None:
            w_tgt = 1.0
        else:
            w_tgt = tl.load(w_ptr + tgt_c).to(tl.float32)

        inp_base = inp_ptr + pid_n * C * D + d
        grad_base = inp_grad_ptr + pid_n * C * D + d

        tmp_max = tl.zeros([BLOCK_C], dtype=tl.float32)
        tmp_sum = tl.zeros([BLOCK_C], dtype=tl.float32)
        for off in range(0, C, BLOCK_C):
            offset = off + offset_c
            mask = offset < C
            inp = tl.load(inp_base + offset * D, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            cur_max = tl.maximum(tmp_max, inp)
            cur_exp = tl.exp(inp - cur_max)
            tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
            tmp_max = cur_max
        final_max = tl.max(tmp_max, axis=0)
        final_sum = tl.sum(tmp_sum * tl.exp(tmp_max - final_max), axis=0)

        for off in range(0, C, BLOCK_C):
            offset = off + offset_c
            mask = offset < C
            inp = tl.load(inp_base + offset * D, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            minus_one = (offset == tgt).to(tl.float32)
            inp_grad = (
                (tl.exp(inp - final_max) / final_sum - minus_one)
                * w_tgt
                * out_grad
                * mean_num
            )
            tl.store(grad_base + offset * D, inp_grad, mask=mask & valid)


@libentry()
@triton.autotune(
    configs=_ce_configs(),
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["label_smoothing", "mean_num"])
def celoss_probability_bwd(
    out_grad_ptr,
    inp_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    label_smoothing,
    mean_num,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = ext.program_id(0)
    pid_d = ext.program_id(1)
    d_start = pid_d * BLOCK_D
    d_stop = tl.minimum(d_start + BLOCK_D, D)
    for d in range(d_start, d_stop):
        offset_c = tl.arange(0, BLOCK_C)
        out_grad = tl.load(out_grad_ptr + pid_n * D + d).to(tl.float32)
        inp_base = inp_ptr + pid_n * C * D + d
        tgt_base = tgt_ptr + pid_n * C * D + d
        grad_base = inp_grad_ptr + pid_n * C * D + d

        tmp_max = tl.zeros([BLOCK_C], dtype=tl.float32)
        tmp_sum = tl.zeros([BLOCK_C], dtype=tl.float32)
        w_tgt_sum = tl.zeros([BLOCK_C], dtype=tl.float32)
        for off in range(0, C, BLOCK_C):
            offset = off + offset_c
            mask = offset < C
            inp = tl.load(inp_base + offset * D, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            tgt = tl.load(tgt_base + offset * D, mask=mask, other=0.0).to(tl.float32)
            tgt = tgt * (1.0 - label_smoothing) + label_smoothing / C
            if w_ptr is None:
                w = tl.where(mask, 1.0, 0.0)
            else:
                w = tl.load(w_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            w_tgt_sum += tgt * w

            cur_max = tl.maximum(tmp_max, inp)
            cur_exp = tl.exp(inp - cur_max)
            tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
            tmp_max = cur_max
        final_max = tl.max(tmp_max, axis=0)
        final_sum = tl.sum(tmp_sum * tl.exp(tmp_max - final_max), axis=0)
        w_tgt_sum = tl.sum(w_tgt_sum, axis=0)

        for off in range(0, C, BLOCK_C):
            offset = off + offset_c
            mask = offset < C
            inp = tl.load(inp_base + offset * D, mask=mask, other=0.0).to(tl.float32)
            tgt = tl.load(tgt_base + offset * D, mask=mask, other=0.0).to(tl.float32)
            tgt = tgt * (1.0 - label_smoothing) + label_smoothing / C
            if w_ptr is None:
                w = tl.where(mask, 1.0, 0.0)
            else:
                w = tl.load(w_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            grad = w_tgt_sum / final_sum * tl.exp(inp - final_max) - tgt * w
            inp_grad = grad * out_grad * mean_num
            tl.store(grad_base + offset * D, inp_grad, mask=mask)


@libentry()
@triton.autotune(
    configs=_ce_configs(),
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["ignore_index", "label_smoothing", "mean_num"])
def celoss_indices_smooth_bwd(
    out_grad_ptr,
    inp_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    ignore_index,
    label_smoothing,
    mean_num,
    C,
    D,
    TGT_WORD_STRIDE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = ext.program_id(0)
    pid_d = ext.program_id(1)
    d_start = pid_d * BLOCK_D
    d_stop = tl.minimum(d_start + BLOCK_D, D)
    for d in range(d_start, d_stop):
        offset_c = tl.arange(0, BLOCK_C)
        tgt = tl.load(tgt_ptr + (pid_n * D + d) * TGT_WORD_STRIDE)
        valid = tgt != ignore_index
        out_grad = tl.load(out_grad_ptr + pid_n * D + d).to(tl.float32)
        inp_base = inp_ptr + pid_n * C * D + d
        grad_base = inp_grad_ptr + pid_n * C * D + d

        tmp_max = tl.zeros([BLOCK_C], dtype=tl.float32)
        tmp_sum = tl.zeros([BLOCK_C], dtype=tl.float32)
        w_sum = tl.zeros([BLOCK_C], dtype=tl.float32)
        for off in range(0, C, BLOCK_C):
            offset = off + offset_c
            mask = offset < C
            inp = tl.load(inp_base + offset * D, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            if w_ptr is None:
                w = tl.where(mask, 1.0, 0.0)
            else:
                w = tl.load(w_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            smooth = tl.where(
                offset == tgt,
                1.0 - label_smoothing + label_smoothing / C,
                label_smoothing / C,
            )
            w_sum += smooth * w

            cur_max = tl.maximum(tmp_max, inp)
            cur_exp = tl.exp(inp - cur_max)
            tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
            tmp_max = cur_max
        final_max = tl.max(tmp_max, axis=0)
        final_sum = tl.sum(tmp_sum * tl.exp(tmp_max - final_max), axis=0)
        w_sum = tl.sum(w_sum, axis=0)

        for off in range(0, C, BLOCK_C):
            offset = off + offset_c
            mask = offset < C
            inp = tl.load(inp_base + offset * D, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            if w_ptr is None:
                w = tl.where(mask, 1.0, 0.0)
            else:
                w = tl.load(w_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            smooth = tl.where(
                offset == tgt,
                1.0 - label_smoothing + label_smoothing / C,
                label_smoothing / C,
            )
            grad = w_sum / final_sum * tl.exp(inp - final_max) - smooth * w
            inp_grad = grad * out_grad * mean_num
            tl.store(grad_base + offset * D, inp_grad, mask=mask & valid)


@libentry()
@triton.jit
def sum_and_scale(
    inp_ptr,
    out_ptr,
    N,
    scalebyw: tl.constexpr,
    BLOCK_N: tl.constexpr = 1024,
    scale=1.0,
    mean_num=None,
):
    mid_sum = tl.zeros(
        [
            BLOCK_N,
        ],
        dtype=tl.float32,
    )
    if scalebyw:
        mid_wgt = tl.zeros(
            [
                BLOCK_N,
            ],
            dtype=tl.float32,
        )
        for off in range(0, N, BLOCK_N):
            offset = off + tl.arange(0, BLOCK_N)
            inp_ptrs = inp_ptr + offset
            mask = offset < N
            inp_vals = tl.load(inp_ptrs, mask=mask, other=0.0)
            mid_sum += inp_vals
            wgt_ptrs = scale + offset
            wgt_vals = tl.load(wgt_ptrs, mask=mask, other=0.0)
            mid_wgt += wgt_vals
        out_val = tl.sum(mid_sum)
        scale_val = tl.sum(mid_wgt)
        tl.store(mean_num, scale_val)
    else:
        for off in range(0, N, BLOCK_N):
            offset = off + tl.arange(0, BLOCK_N)
            inp_ptrs = inp_ptr + offset
            mask = offset < N
            inp_vals = tl.load(inp_ptrs, mask=mask, other=0.0)
            mid_sum += inp_vals
        out_val = tl.sum(mid_sum)
        scale_val = scale
    out_val /= scale_val
    tl.store(out_ptr, out_val)


class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, weight, reduction, ignore_index, label_smoothing):
        logger.debug("GEMS CrossEntropyLoss")

        shape = list(inp.shape)
        dim = inp.ndim
        N = 1 if dim == 1 else shape[0]
        C = shape[0] if dim == 1 else shape[1]
        D = inp.numel() // N // C
        axis = 0 if dim == 1 else 1
        del shape[axis]

        inp = inp.contiguous()
        tgt = target.contiguous()
        tgt_word_stride = 1
        if tgt.ndim != dim:
            tgt, tgt_word_stride = _tgt_words(tgt)
        weight = weight.contiguous() if weight is not None else None
        out = torch.empty(shape, dtype=torch.float32, device=inp.device)
        grid = lambda meta: (N, triton.cdiv(D, meta["BLOCK_D"]))

        if tgt.ndim == dim:
            # target probabilities
            with torch_device_fn.device(inp.device):
                celoss_probability_kernel[grid](
                    inp,
                    tgt,
                    weight,
                    out,
                    label_smoothing,
                    C,
                    D,
                )
        elif label_smoothing == 0:
            # target indices
            w_tgt = torch.empty(shape, dtype=torch.float32, device=inp.device)
            with torch_device_fn.device(inp.device):
                if N >= _ROWS_MIN_N:
                    num_progs = min(N, _ROW_PROG_CAP)
                    rows_per_cta = -(-N // num_progs)
                    grid_rows = lambda meta: (
                        num_progs,
                        triton.cdiv(D, meta["BLOCK_D"]),
                    )
                    celoss_indices_kernel_rows[grid_rows](
                        inp,
                        tgt,
                        weight,
                        out,
                        w_tgt,
                        ignore_index,
                        C,
                        D,
                        N,
                        rows_per_cta,
                        TGT_WORD_STRIDE=tgt_word_stride,
                    )
                else:
                    celoss_indices_kernel[grid](
                        inp,
                        tgt,
                        weight,
                        out,
                        w_tgt,
                        ignore_index,
                        C,
                        D,
                        TGT_WORD_STRIDE=tgt_word_stride,
                    )
        else:
            w_tgt = torch.empty(shape, dtype=torch.float32, device=inp.device)
            with torch_device_fn.device(inp.device):
                celoss_indices_smooth_kernel[grid](
                    inp,
                    tgt,
                    weight,
                    out,
                    w_tgt,
                    ignore_index,
                    label_smoothing,
                    C,
                    D,
                    TGT_WORD_STRIDE=tgt_word_stride,
                )

        if reduction == 1:  # MEAN
            out_reduce = torch.empty([], dtype=inp.dtype, device=inp.device)
            if tgt.ndim == dim:
                sum_and_scale[(1,)](out, out_reduce, N * D, False, scale=N * D)
            else:
                wgt_sum = torch.empty([], dtype=torch.float32, device=inp.device)
                sum_and_scale[(1,)](
                    out, out_reduce, N * D, True, scale=w_tgt, mean_num=wgt_sum
                )
            out = out_reduce
        elif reduction == 2:  # SUM
            out_reduce = torch.empty([], dtype=inp.dtype, device=inp.device)
            sum_and_scale[(1,)](out, out_reduce, N * D, False)
            out = out_reduce

        if inp.requires_grad:
            ctx.save_for_backward(inp, tgt, weight)
            ctx.N = N
            ctx.C = C
            ctx.D = D
            ctx.ignore_index = ignore_index
            ctx.label_smoothing = label_smoothing
            ctx.shape = shape
            ctx.mean_num = 1
            ctx.tgt_word_stride = tgt_word_stride
            if reduction == 1:
                ctx.mean_num = N * D if tgt.ndim == dim else wgt_sum

        return out.to(inp.dtype)

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS CrossEntropyLoss VJP")

        inp, tgt, weight = ctx.saved_tensors
        N = ctx.N
        C = ctx.C
        D = ctx.D
        ignore_index = ctx.ignore_index
        label_smoothing = ctx.label_smoothing
        mean_num = (
            1 / ctx.mean_num.item()
            if isinstance(ctx.mean_num, torch.Tensor)
            else 1 / ctx.mean_num
        )
        shape = ctx.shape

        out_grad = out_grad.broadcast_to(shape).contiguous()

        inp_grad = torch.zeros(inp.shape, dtype=inp.dtype, device=inp.device)
        grid = lambda meta: (N, triton.cdiv(D, meta["BLOCK_D"]))
        if tgt.ndim == inp.ndim:
            celoss_probability_bwd[grid](
                out_grad,
                inp,
                tgt,
                weight,
                inp_grad,
                label_smoothing,
                mean_num,
                C,
                D,
            )
        elif label_smoothing == 0:
            celoss_indices_bwd[grid](
                out_grad,
                inp,
                tgt,
                weight,
                inp_grad,
                ignore_index,
                mean_num,
                C,
                D,
                TGT_WORD_STRIDE=ctx.tgt_word_stride,
            )
        else:
            celoss_indices_smooth_bwd[grid](
                out_grad,
                inp,
                tgt,
                weight,
                inp_grad,
                ignore_index,
                label_smoothing,
                mean_num,
                C,
                D,
                TGT_WORD_STRIDE=ctx.tgt_word_stride,
            )
        return inp_grad, None, None, None, None, None


def cross_entropy_loss(
    inp,
    target,
    weight=None,
    reduction="mean",
    ignore_index=-100,
    label_smoothing=0.0,
):
    return CrossEntropyLoss.apply(
        inp,
        target,
        weight,
        _Reduction.get_enum(reduction),
        ignore_index,
        label_smoothing,
    )
