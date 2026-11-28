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

"""Stage compact paged KV once for repeated dense TMA reads during prefill."""

import triton
import triton.language as tl


@triton.jit
def _pack_prefill_pages(
    K,
    V,
    Table,
    Lengths,
    PackedK,
    PackedV,
    page_stride,
    table_stride,
    CAPACITY: tl.constexpr,
    PAGE: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(1)
    length = tl.load(Lengths + batch)
    element = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = element // ROW
    valid = row < length
    page = tl.load(
        Table + batch.to(tl.int64) * table_stride + row // PAGE, valid, other=0
    )
    src = page.to(tl.int64) * page_stride + (row % PAGE) * ROW + element % ROW
    dst = batch * CAPACITY * ROW + element
    tl.store(PackedK + dst, tl.load(K + src, valid, other=0), row < CAPACITY)
    tl.store(PackedV + dst, tl.load(V + src, valid, other=0), row < CAPACITY)


def pack_prefill_pages(k, v, table, lengths, max_k):
    batch = lengths.numel()
    _, page, kv_heads, dim = k.shape
    shape = (batch * max_k, kv_heads, dim)
    packed_k, packed_v = k.new_empty(shape), v.new_empty(shape)
    block = 2048
    _pack_prefill_pages[(triton.cdiv(max_k * kv_heads * dim, block), batch)](
        k,
        v,
        table,
        lengths,
        packed_k,
        packed_v,
        k.stride(0),
        table.stride(0),
        max_k,
        page,
        kv_heads * dim,
        block,
    )
    return packed_k, packed_v
