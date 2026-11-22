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


import math
from threading import Lock

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
from triton.experimental.tle.language.dsa.ascend import sub_vec_id
from triton.runtime import driver

from flaggems_vllm.runtime.backend._ascend.ops.flash_attention.direct import (
    _attention_task_direct,
)
from flaggems_vllm.runtime.backend._ascend.ops.flash_attention.scheduling import (
    effective_kv_splits,
)
from flaggems_vllm.runtime.backend._ascend.ops.flash_attention.submission import (
    _raw_stream_nowait,
    bind_runner,
    cube_core_limit,
)


@triton.jit
def _merge_one_q1(
    O,
    PARTIAL,
    LSE,
    b,
    head,
    n,
    H: tl.constexpr,
    B: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    row = (b * H + head).to(tl.int64)
    d = tl.arange(0, 256)
    active, _ = effective_kv_splits(n, SPLITS, 512)
    if active == 1:
        out = tl.load(PARTIAL + row * 256 + d, volatile=True)
    else:
        split = tl.arange(0, BLOCK_SPLITS)
        ls = tl.load(
            LSE + (split * H + head) * B + b,
            split < active,
            -float("inf"),
            volatile=True,
        )
        max_lse = tl.max(ls, 0)
        valid = max_lse > -float("inf")
        weight = tl.exp(ls - tl.where(valid, max_lse, 0.0))
        partial = tl.load(
            PARTIAL + (split[:, None] * B * H + row) * 256 + d[None, :],
            split[:, None] < active,
            0.0,
            volatile=True,
        )
        denom = tl.where(valid, tl.sum(weight, 0), 1.0)
        out = tl.sum(partial * (weight / denom)[:, None], 0)
    tl.store(O + row * 256 + d, out)


@triton.jit(
    do_not_specialize_on_alignment=[
        "Q",
        "K",
        "V",
        "O",
        "USED",
        "TABLE",
        "WORK",
        "PROBS",
    ]
)
def _raw_q1_one(
    Q,
    K,
    V,
    O,
    USED,
    TABLE,
    WORK,
    PROBS,
    H: tl.constexpr,
    HK: tl.constexpr,
    B: tl.constexpr,
    PAGE: tl.constexpr,
    TS: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
):
    W = tl.num_programs(0)
    worker = tl.program_id(0)
    G: tl.constexpr = H // HK
    qs: tl.constexpr = H * 256
    kp: tl.constexpr = PAGE * HK * 256
    kr: tl.constexpr = HK * 256
    scores = WORK
    pv = WORK + W * 16 * 512
    partial = pv + W * 16 * 256
    lse = partial + SPLITS * B * H * 256
    for item in range(worker, B * HK * SPLITS, W):
        task, split = item // SPLITS, item % SPLITS
        b, kh = task // HK, task % HK
        n = tl.load(USED + b)
        if n <= 1:
            with tle.scope(core_mode="vector"):
                if split == 0 and sub_vec_id() == 0:
                    d = tl.arange(0, 256)
                    g = tl.arange(0, G)
                    physical = tl.load(TABLE + b.to(tl.int64) * TS, n == 1, 0).to(
                        tl.int64
                    )
                    val = tl.load(V + physical * kp + kh * 256 + d, n == 1, 0)
                    dst = O + b.to(tl.int64) * qs + (kh * G + g[:, None]) * 256
                    tl.store(dst + d[None, :], val[None, :])
        else:
            active, _ = effective_kv_splits(n, SPLITS, 512)
            if split < active:
                _attention_task_direct(
                    Q,
                    K,
                    V,
                    partial,
                    lse,
                    None,
                    USED,
                    TABLE,
                    scores,
                    PROBS,
                    pv,
                    b,
                    kh,
                    split,
                    worker,
                    H,
                    HK,
                    B,
                    1,
                    SCALE,
                    qs,
                    256,
                    qs,
                    qs,
                    256,
                    qs,
                    kp,
                    kr,
                    256,
                    kp,
                    kr,
                    256,
                    TS,
                    PAGE,
                    False,
                    False,
                    16,
                    32,
                    512,
                    SPLITS,
                )
    # A persistent grid occupies at most one mixed program per Cube.
    # All workers reach the same global barrier, including when every K is 0/1.
    tle.dsa.ascend.sync_block_all("all", 0)
    with tle.scope(core_mode="vector"):
        lane = worker * 2 + sub_vec_id()
        for output_row in range(lane, B * H, W * 2):
            batch, head = output_row // H, output_row % H
            n = tl.load(USED + batch)
            if n > 1:
                _merge_one_q1(
                    O,
                    partial,
                    lse,
                    batch,
                    head,
                    n,
                    H,
                    B,
                    SPLITS,
                    triton.next_power_of_2(SPLITS),
                )


_launchers = {}
_stream_scratch = {}
_scratch_lock = Lock()
_MAX_SCRATCH_BYTES = 48 * 1024 * 1024


def _q1_scratch(q, work_size, prob_size, stream):
    dtype, device = q.dtype, q.device
    key = None
    if stream is not None and not torch.npu.is_current_stream_capturing():
        # One buffer per stream is sufficient: the single persistent launch
        # consumes both regions before the next launch on this stream.
        key = (device.index, stream)
        workspace = _stream_scratch.get(key)
        if workspace is not None and (
            workspace[0].numel() >= work_size and workspace[1].numel() >= prob_size
        ):
            return workspace
    workspace = (
        torch.empty((work_size,), dtype=torch.float32, device=device),
        torch.empty((prob_size,), dtype=dtype, device=device),
    )
    if key is not None:
        with _scratch_lock:
            retained = sum(
                4 * w.numel() + 2 * p.numel()
                for k, (w, p) in _stream_scratch.items()
                if k != key
            )
            if retained + 4 * work_size + 2 * prob_size <= _MAX_SCRATCH_BYTES:
                _stream_scratch[key] = workspace
    return workspace


def raw_q1_output(q, k, v, cu, used, table, max_k, scale, device):
    """Fuse singleton output with sparse work on the raw-paged Decode path."""
    if (
        q.ndim != 3
        or k.ndim != 4
        or k.shape != v.shape
        or q.dtype != torch.bfloat16
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or q.shape[2] != 256
        or k.shape[3] != 256
        or not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous())
        or cu.dtype != torch.int32
        or cu.ndim != 1
        or not cu.is_contiguous()
        or used.ndim != 1
        or not used.is_contiguous()
        or table.ndim != 2
        or not 1 <= q.size(0) <= 1024
        or not 32768 <= max_k <= table.size(1) * k.size(1)
    ):
        return None
    b, h, _ = q.shape
    pages, page, hk, _ = k.shape
    if (
        not hk
        or h % hk
        or h // hk not in (4, 8)
        or page not in (16, 32)
        or b != table.size(0)
        or b != used.numel()
        or cu.numel() != b + 1
        # Keep the pre-existing small-cache staged and Q-direct paths.
        or (h == 8 and hk == 1 and 32 <= b <= 128 and pages * page <= b * 4096)
        or (
            page == 32
            and h == 16
            and hk == 2
            and b >= 512
            and b % 4 == 0
            and pages * page <= b * 2048
        )
    ):
        return None
    if scale is None:
        scale = 0.0625
    elif type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        return None
    index = device.index
    w = cube_core_limit(index)
    # Bound the partial workspace while allowing large batches to remain on
    # the direct path. Keep at least two splits for sparse long requests.
    if b * h > 8192:
        return None
    tasks = (b * h + 15) // 16
    splits = max(2, min(16, w // tasks)) if tasks <= w else 4
    out = torch.empty_like(q)
    partials = splits * h * b * 256
    active = torch.npu.current_device() == index
    stream = None
    if active:
        current_stream = _raw_stream_nowait or driver.active.get_current_stream
        stream = current_stream(index)
    work, probs = _q1_scratch(
        q, w * 16 * (512 + 256) + splits * h * b + partials, w * 16 * 512, stream
    )
    constexprs = (h, hk, b, page, table.stride(0), scale, splits)
    args = (q, k, v, out, used, table, work, probs, *constexprs)
    key = (index, q.dtype, w, constexprs)
    compiled = _launchers.get(key)
    if compiled is not None and active:
        compiled(args)
    else:
        grid = (w, 1, 1)
        with torch.npu.device(device):
            entry = _raw_q1_one[grid](*args, enable_fp_fusion=False, multibuffer=False)
        if len(_launchers) < 256 and hasattr(entry, "__getitem__"):
            _launchers[key] = bind_runner(entry, grid, index)
    return out
