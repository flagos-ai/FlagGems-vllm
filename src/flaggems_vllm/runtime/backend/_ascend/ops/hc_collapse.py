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

try:
    import triton.experimental.tle as tle
    from triton.experimental.tle.language.dsa import tile_load
except ImportError:
    tle = None
    tile_load = None


def widths(h, channels, cap=1024):
    block = min(cap, 1 << (h.bit_length() - 1))
    remainder = h % block
    tail = 1 << (remainder - 1).bit_length() if remainder else block
    channel_block = min(4, channels & -channels)
    return block, tail, channel_block


@triton.jit
def _hc_generic_tv(
    X,
    P,
    Y,
    T: tl.constexpr,
    HC: tl.constexpr,
    H: tl.constexpr,
    B: tl.constexpr,
    TW: tl.constexpr,
    BC: tl.constexpr,
    N: tl.constexpr,
    PITCH: tl.constexpr,
    ST: tl.constexpr,
    SC: tl.constexpr,
    SH: tl.constexpr,
    PT: tl.constexpr,
    PC: tl.constexpr,
):
    R: tl.constexpr = H % B
    for t in range(tl.program_id(0).to(tl.int64), T, N):
        for ht in range(H // B):
            start = ht * B
            col = tl.arange(0, B)
            # Traversal 1 lets the last full input window start at any logical column.
            if SH == 1:
                view = tle.make_strided_view(
                    X,
                    shape=[T, HC, H],
                    strides=[ST, SC, SH],
                    tile=[1, BC, B],
                    traversal_strides=[1, BC, 1],
                )
            acc = tl.full((B,), 0, tl.float32)
            for cb in range(HC // BC):
                c = cb * BC + tl.arange(0, BC)
                weight = tl.load(P + t * PT + c.to(tl.int64) * PC).to(tl.float32)
                if SH == 1:
                    tile = tile_load(
                        view, index=(t, cb, start), dst_space=tle.language.dsa.ascend.UB
                    )
                    x = tl.reshape(tile, (BC, B))
                else:
                    addr = (
                        t * ST
                        + c.to(tl.int64)[:, None] * SC
                        + (start + col[None, :]) * SH
                    )
                    x = tl.load(X + addr)
                acc += tl.sum(x.to(tl.float32) * weight[:, None], axis=0)
            tl.store(Y + t * PITCH + start + col, acc.to(Y.dtype.element_ty))
        if R:
            start = H - TW
            col = tl.arange(0, TW)
            # Traversal 1 lets the last full input window start at any logical column.
            if SH == 1:
                view = tle.make_strided_view(
                    X,
                    shape=[T, HC, H],
                    strides=[ST, SC, SH],
                    tile=[1, BC, TW],
                    traversal_strides=[1, BC, 1],
                )
            acc = tl.full((TW,), 0, tl.float32)
            for cb in range(HC // BC):
                c = cb * BC + tl.arange(0, BC)
                weight = tl.load(P + t * PT + c.to(tl.int64) * PC).to(tl.float32)
                if SH == 1:
                    tile = tile_load(
                        view, index=(t, cb, start), dst_space=tle.language.dsa.ascend.UB
                    )
                    x = tl.reshape(tile, (BC, TW))
                else:
                    addr = (
                        t * ST
                        + c.to(tl.int64)[:, None] * SC
                        + (start + col[None, :]) * SH
                    )
                    x = tl.load(X + addr)
                acc += tl.sum(x.to(tl.float32) * weight[:, None], axis=0)
            if TW != R:
                acc = tl.gather(acc, (col + TW - R) % TW, axis=0)
            tl.store(Y + t * PITCH + H - R + col, acc.to(Y.dtype.element_ty))


def _general(x, p):
    t, hc, h = x.shape
    if p.shape != (t, hc):
        raise ValueError("expected weights with shape [T, HC]")
    if not hc:
        return x.new_zeros((t, h))
    if not t or not h:
        return x.new_empty((t, h))
    b, tw, bc = widths(h, hc)
    alignment = max(1, 32 // x.element_size())
    used = h if h % b == 0 else h - h % b + tw
    pitch = triton.cdiv(used, alignment) * alignment
    y = x.new_empty((t, pitch))
    n = min(40, t)
    _hc_generic_tv[(n,)](
        x,
        p,
        y,
        t,
        hc,
        h,
        b,
        tw,
        bc,
        n,
        pitch,
        *x.stride(),
        *p.stride(),
        use_commonir_tensor_view=True,
        enable_fp_fusion=False,
    )
    return y[:, :h].contiguous()


@triton.jit
def _hc4_tv(
    X,
    P,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    B: tl.constexpr,
    TW: tl.constexpr,
    N: tl.constexpr,
    PITCH: tl.constexpr,
    ST: tl.constexpr,
    SC: tl.constexpr,
    SH: tl.constexpr,
    PT: tl.constexpr,
    PC: tl.constexpr,
):
    R: tl.constexpr = H % B
    JOBS: tl.constexpr = T * (H // B) if R == 0 else T
    for job in range(tl.program_id(0).to(tl.int64), JOBS, N):
        t = job // (H // B) if R == 0 else job
        c = tl.arange(0, 4)
        weight = tl.load(P + t * PT + c.to(tl.int64) * PC).to(tl.float32)
        for block in range(1 if R == 0 else H // B):
            ht = job % (H // B) if R == 0 else block
            start = ht * B
            col = tl.arange(0, B)
            # Traversal 1 lets the last full input window start at any logical column.
            if SH == 1:
                view = tle.make_strided_view(
                    X,
                    shape=[T, 4, H],
                    strides=[ST, SC, SH],
                    tile=[1, 4, B],
                    traversal_strides=[1, 4, 1],
                )
            if SH == 1:
                tile = tile_load(
                    view, index=(t, 0, start), dst_space=tle.language.dsa.ascend.UB
                )
                x = tl.reshape(tile, (4, B))
            else:
                addr = (
                    t * ST + c.to(tl.int64)[:, None] * SC + (start + col[None, :]) * SH
                )
                x = tl.load(X + addr)
            acc = tl.sum(x.to(tl.float32) * weight[:, None], axis=0)
            tl.store(Y + t * PITCH + start + col, acc.to(Y.dtype.element_ty))
        if R:
            start = H - TW
            col = tl.arange(0, TW)
            # Traversal 1 lets the last full input window start at any logical column.
            if SH == 1:
                view = tle.make_strided_view(
                    X,
                    shape=[T, 4, H],
                    strides=[ST, SC, SH],
                    tile=[1, 4, TW],
                    traversal_strides=[1, 4, 1],
                )
            if SH == 1:
                tile = tile_load(
                    view, index=(t, 0, start), dst_space=tle.language.dsa.ascend.UB
                )
                x = tl.reshape(tile, (4, TW))
            else:
                addr = (
                    t * ST + c.to(tl.int64)[:, None] * SC + (start + col[None, :]) * SH
                )
                x = tl.load(X + addr)
            acc = tl.sum(x.to(tl.float32) * weight[:, None], axis=0)
            if TW != R:
                acc = tl.gather(acc, (col + TW - R) % TW, axis=0)
            tl.store(Y + t * PITCH + H - R + col, acc.to(Y.dtype.element_ty))


def _hc4(x, p):
    t, hc, h = x.shape
    if p.shape != (t, hc):
        raise ValueError("expected weights with shape [T, HC]")
    if hc != 4:
        raise ValueError("HC=4 required")
    if not t or not h:
        return x.new_empty((t, h))
    if t >= 40 and h <= 128 and x.stride(2) != 1:
        return _general(x, p)
    b, tw, _ = widths(h, hc)
    alignment = max(1, 32 // x.element_size())
    used = h if h % b == 0 else h - h % b + tw
    pitch = triton.cdiv(used, alignment) * alignment
    y = x.new_empty((t, pitch))
    n = min(40, t * (h // b) if h % b == 0 else t)
    _hc4_tv[(n,)](
        x,
        p,
        y,
        t,
        h,
        b,
        tw,
        n,
        pitch,
        *x.stride(),
        *p.stride(),
        use_commonir_tensor_view=True,
        enable_fp_fusion=False,
    )
    return y[:, :h].contiguous()


def dispatch_path(x):
    t, hc, h = x.shape
    if not t or not h or not hc:
        return "empty"
    if hc != 4 or (t >= 40 and h <= 128):
        return "general"
    # Historical continuous mid-batch H5120 measurements favor general.
    # This policy is provisional until same-run integrated A/B validation.
    if 40 <= t <= 256 and h == 5120 and x.stride(2) == 1:
        return "general"
    return "hc4"


def hc_collapse(x, pre_mix):
    from flaggems_vllm.ops.hc_collapse import validate_inputs

    validate_inputs(x, pre_mix)
    if tle is None or tile_load is None:
        raise ImportError("hc_collapse requires FlagTree TLE tensor-view tile_load")
    if x.device.type != "npu":
        raise NotImplementedError("HC tensor-view implementation requires Ascend NPU")
    with torch.npu.device(x.device):
        if dispatch_path(x) == "hc4":
            return _hc4(x, pre_mix)
        return _general(x, pre_mix)
