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

from flaggems_vllm.runtime.backend._ascend.ops.flash_attention.submission import (
    bind_runner,
)


@triton.jit(do_not_specialize_on_alignment=["V", "O", "CUQ", "USED", "TABLE"])
def _singleton_paged_output(
    V,
    O,
    CUQ,
    USED,
    TABLE,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    PAGE: tl.constexpr,
    TS: tl.constexpr,
    GROUP: tl.constexpr,
    BG: tl.constexpr,
    BATCH_TILE: tl.constexpr,
    DENSE_Q: tl.constexpr,
):
    kh, first = tl.program_id(0), tl.program_id(1) * BATCH_TILE
    d = tl.arange(0, 256)
    g = tl.arange(0, BG)
    for offset in range(BATCH_TILE):
        batch = first + offset
        if B % BATCH_TILE == 0 or batch < B:
            if DENSE_Q:
                # With Q <= 1 and as many query rows as requests, every
                # request owns exactly one row. No device metadata read.
                row, has_query = batch, True
            else:
                row = tl.load(CUQ + batch)
                has_query = tl.load(CUQ + batch + 1) > row
            if has_query:
                has_kv = tl.load(USED + batch) > 0
                page = tl.load(TABLE + batch.to(tl.int64) * TS, has_kv, 0).to(tl.int64)
                val = tl.load(V + (page * PAGE * HK + kh) * 256 + d, has_kv, 0)
                dst = O + (row.to(tl.int64) * H + kh * GROUP + g[:, None]) * 256
                tl.store(dst + d[None, :], val[None, :], g[:, None] < GROUP)


_singleton_launchers = {}


def singleton_paged_output(q, k, v, cuq, used, table, device):
    """Fast output-only path; caller guarantees the maximum KV length is one."""
    q_shape, kv_shape, table_shape = q.shape, k.shape, table.shape
    dtype = q.dtype
    if (
        len(q_shape) != 3
        or len(kv_shape) != 4
        or q_shape[-1] != 256
        or kv_shape[-1] != 256
        or kv_shape != v.shape
        or kv_shape[1] not in (16, 32)
        or dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != dtype
        or v.dtype != dtype
        or not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous())
        or cuq.ndim != 1
        or cuq.dtype != torch.int32
        or not cuq.is_contiguous()
        or not used.is_contiguous()
        or len(table_shape) != 2
        or table_shape[0] != cuq.numel() - 1
        or table_shape[1] < 1
        or used.shape != (table_shape[0],)
    ):
        return None
    total, h, _ = q_shape
    hk, b = kv_shape[2], table_shape[0]
    if not hk or not h or h % hk or h // hk > 32 or not b or total > b:
        return None
    out = torch.empty_like(q)
    if not total:
        return out
    # Bound the number of program waves for large batches; keep each
    # physical-page read contiguous within the Vector program.
    tasks = b * hk
    batch_tile = 8 if tasks > 512 else 4 if tasks > 192 else 1
    group = h // hk
    configs = (
        b,
        h,
        hk,
        kv_shape[1],
        table.stride(0),
        group,
        1 << (group - 1).bit_length(),
        batch_tile,
        total == b,
    )
    index = device.index
    key = (index, dtype, configs)
    compiled = _singleton_launchers.get(key)
    inputs = (v, out, None if total == b else cuq, used, table, *configs)
    # Only the compiled code/grid is cached. Metadata and output pointers are
    # supplied each call, and the runner resolves the current stream.
    if compiled is not None and torch.npu.current_device() == index:
        compiled(inputs)
    else:
        grid = (hk, triton.cdiv(b, batch_tile), 1)
        with torch.npu.device(device):
            entry = _singleton_paged_output[grid](*inputs)
        if len(_singleton_launchers) < 256 and hasattr(entry, "__getitem__"):
            _singleton_launchers[key] = bind_runner(entry, grid, index)
    return out
