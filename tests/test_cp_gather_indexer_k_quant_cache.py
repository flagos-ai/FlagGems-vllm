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
import os

import pytest
import torch
from packaging.version import InvalidVersion, Version

from flaggems_vllm.ops import cp_gather_indexer_k_quant_cache

_TARGET_VLLM_VERSION = Version("0.20.2")
_NEXT_VLLM_VERSION = Version("0.21.0")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)


def _default_fp8_dtype():
    if getattr(torch.version, "hip", None) is not None and hasattr(
        torch, "float8_e4m3fnuz"
    ):
        return torch.float8_e4m3fnuz
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    pytest.skip("float8_e4m3fn is required for cp_gather_indexer_k_quant_cache")


def _check_target_vllm_version(vllm):
    version = getattr(vllm, "__version__", "0.0.0")
    try:
        parsed = Version(version.split("+", 1)[0])
        if parsed < _TARGET_VLLM_VERSION or parsed >= _NEXT_VLLM_VERSION:
            return False
    except InvalidVersion:
        pass
    return True


def _load_vllm_cuda_op_and_fp8_dtype():
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
    if getattr(torch.version, "cuda", None) is None:
        return None, _default_fp8_dtype(), False
    try:
        import vllm
        import vllm._custom_ops as ops
        from vllm.platforms import current_platform
    except Exception:
        return None, _default_fp8_dtype(), False

    if not _check_target_vllm_version(vllm):
        return None, _default_fp8_dtype(), False

    if not hasattr(ops, "cp_gather_indexer_k_quant_cache"):
        return None, _default_fp8_dtype(), False

    def vllm_gather(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens):
        ops.cp_gather_indexer_k_quant_cache(
            kv_cache,
            dst_k,
            dst_scale,
            block_table,
            cu_seq_lens,
        )

    return vllm_gather, current_platform.fp8_dtype(), True


def torch_gather(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens):
    block_size = kv_cache.size(1)
    head_dim = dst_k.size(1)
    quant_block_size = head_dim * 4 // dst_scale.size(1)
    num_quant_blocks = head_dim // quant_block_size

    flat_cache = kv_cache.view(kv_cache.shape[0], -1)
    cache_values = flat_cache[:, : block_size * head_dim]
    cache_scales = flat_cache[:, block_size * head_dim :].view(torch.float32)
    dst_k_bytes = dst_k.view(torch.uint8)
    dst_scale_float = dst_scale.view(torch.float32)

    for batch_idx in range(block_table.size(0)):
        seq_start = int(cu_seq_lens[batch_idx].item())
        seq_end = int(cu_seq_lens[batch_idx + 1].item())
        for token_idx in range(seq_start, seq_end):
            batch_offset = token_idx - seq_start
            block_table_id = batch_offset // block_size
            block_offset = batch_offset % block_size
            block_id = int(block_table[batch_idx, block_table_id].item())

            src_value_start = block_offset * head_dim
            src_value_end = src_value_start + head_dim
            dst_k_bytes[token_idx].copy_(
                cache_values[block_id, src_value_start:src_value_end]
            )

            src_scale_start = block_offset * num_quant_blocks
            src_scale_end = src_scale_start + num_quant_blocks
            dst_scale_float[token_idx].copy_(
                cache_scales[block_id, src_scale_start:src_scale_end]
            )


def _make_cache(num_blocks, block_size, head_dim, quant_block_size, fp8_dtype, device):
    cache_stride = head_dim + head_dim * 4 // quant_block_size
    k_cache = torch.empty(
        (num_blocks, block_size, cache_stride),
        dtype=torch.uint8,
        device=device,
    )
    num_quant_blocks = head_dim // quant_block_size
    flat_cache = k_cache.view(num_blocks, -1)
    values = flat_cache[:, : block_size * head_dim].view(fp8_dtype)
    values.copy_(torch.randn(values.shape, device=device).to(fp8_dtype))
    scales = flat_cache[:, block_size * head_dim :].view(torch.float32)
    scales.copy_(
        torch.rand(
            num_blocks,
            block_size * num_quant_blocks,
            device=device,
            dtype=torch.float32,
        )
        + 0.01
    )
    return k_cache


def _make_gather_metadata(seq_lens, block_size, device):
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    cu_seqlen = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    cu_seqlen[1:] = torch.cumsum(seq_lens_tensor, dim=0)

    max_blocks = max(math.ceil(seq_len / block_size) for seq_len in seq_lens)
    block_table = torch.full(
        (len(seq_lens), max_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    next_block = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        num_seq_blocks = math.ceil(seq_len / block_size)
        blocks = torch.arange(
            next_block,
            next_block + num_seq_blocks,
            dtype=torch.int32,
            device=device,
        )
        block_table[batch_idx, :num_seq_blocks] = blocks
        next_block += num_seq_blocks

    return block_table, cu_seqlen, next_block


@pytest.mark.cp_gather_indexer_k_quant_cache
@pytest.mark.parametrize(
    "seq_lens,block_size,head_dim,quant_block_size,extra_tokens",
    [
        ([13, 7, 16], 8, 128, 128, 0),
        ([17, 1, 33, 9], 16, 512, 128, 5),
    ],
)
@torch.inference_mode()
def test_cp_gather_indexer_k_quant_cache_matches_reference(
    seq_lens,
    block_size,
    head_dim,
    quant_block_size,
    extra_tokens,
):
    vllm_op, fp8_dtype, has_vllm = _load_vllm_cuda_op_and_fp8_dtype()

    torch.manual_seed(1)
    device = torch.device("cuda")
    block_table, cu_seqlen, num_blocks = _make_gather_metadata(
        seq_lens,
        block_size,
        device,
    )
    valid_tokens = int(cu_seqlen[-1].item())
    allocated_tokens = valid_tokens + extra_tokens

    k_cache = _make_cache(
        num_blocks,
        block_size,
        head_dim,
        quant_block_size,
        fp8_dtype,
        device,
    )

    num_scale_bytes = head_dim * 4 // quant_block_size
    sentinel = 0x7B
    gems_k = torch.empty(
        (allocated_tokens, head_dim),
        dtype=fp8_dtype,
        device=device,
    )
    gems_k.view(torch.uint8).fill_(sentinel)
    reference_k = torch.empty_like(gems_k)
    reference_k.view(torch.uint8).fill_(sentinel)
    gems_scale = torch.full(
        (allocated_tokens, num_scale_bytes),
        sentinel,
        dtype=torch.uint8,
        device=device,
    )
    reference_scale = torch.full_like(gems_scale, sentinel)

    if has_vllm:
        vllm_op(
            k_cache,
            reference_k,
            reference_scale,
            block_table,
            cu_seqlen,
        )
    else:
        torch_gather(
            k_cache,
            reference_k,
            reference_scale,
            block_table,
            cu_seqlen,
        )
    cp_gather_indexer_k_quant_cache(
        k_cache,
        gems_k,
        gems_scale,
        block_table,
        cu_seqlen,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        gems_k.view(torch.uint8),
        reference_k.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(gems_scale, reference_scale, rtol=0, atol=0)
    if extra_tokens:
        assert torch.all(gems_k[valid_tokens:].view(torch.uint8) == sentinel)
        assert torch.all(gems_scale[valid_tokens:] == sentinel)


@pytest.mark.cp_gather_indexer_k_quant_cache
@pytest.mark.parametrize("batch_size", [1, 16, 17, 31, 32, 63, 64])
@pytest.mark.parametrize("head_dim", [128, 512])
@torch.inference_mode()
def test_packed_shared_prefix_graph(batch_size, head_dim):
    """Shared/reordered physical pages, empty rows, offsets and graph reuse."""
    device = "cuda"
    block_size = 64
    num_blocks = 11
    scale_bytes = head_dim // 128 * 4
    page_bytes = block_size * (head_dim + scale_bytes)
    stride = page_bytes + 576
    backing = torch.empty(num_blocks * stride + 64, dtype=torch.uint8, device=device)
    cache = backing.as_strided(
        (num_blocks, block_size, head_dim + scale_bytes),
        (stride, head_dim + scale_bytes, 1),
        storage_offset=64,
    )
    flat = cache.view(num_blocks, -1)
    for b in range(num_blocks):
        flat[b, : block_size * head_dim].fill_(b + 1)
        flat[b, block_size * head_dim :].view(torch.float32).fill_(b + 0.25)
    # Rows have padding, a nonzero storage offset, and shared physical prefixes.
    table_backing = torch.full(
        (batch_size + 2, 5), -1, dtype=torch.int32, device=device
    )
    table = table_backing[1:-1, 1:4]
    cu = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
    capacity = batch_size * 129 + 7
    out_backing = torch.full(
        (capacity + 2, head_dim), 123, dtype=torch.uint8, device=device
    )
    scales_backing = torch.full(
        (capacity + 2, scale_bytes), 123, dtype=torch.uint8, device=device
    )
    out = out_backing[1:-1].view(_default_fp8_dtype())
    scales = scales_backing[1:-1]

    def update(version):
        lengths = [([65, 0, 129, 1][(i + version) % 4]) for i in range(batch_size)]
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)
        cu.copy_(torch.tensor(offsets, dtype=torch.int32, device=device))
        table.copy_(
            torch.tensor(
                [[2, (i + version) % 5 + 4, 1] for i in range(batch_size)],
                dtype=torch.int32,
                device=device,
            )
        )
        flat[2, : block_size * head_dim].fill_(20 + version)
        flat[2, block_size * head_dim :].view(torch.float32).fill_(20.25 + version)
        out_backing.fill_(123)
        scales_backing.fill_(123)

    update(0)
    cp_gather_indexer_k_quant_cache(cache, out, scales, table, cu)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cp_gather_indexer_k_quant_cache(cache, out, scales, table, cu)
    # Both replays use new cache bytes and request/physical block metadata.
    for version in [1, 2]:
        update(version)
        expected = torch.full_like(out.view(torch.uint8), 123).view(out.dtype)
        expected_scales = torch.full_like(scales, 123)
        torch_gather(cache, expected, expected_scales, table, cu)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
        assert bool(torch.all(out_backing[[0, -1]] == 123))
        assert bool(torch.all(scales_backing[[0, -1]] == 123))


@pytest.mark.cp_gather_indexer_k_quant_cache
@pytest.mark.parametrize(
    "stride,high_block,storage_offset",
    [(2 * 1024 * 1024, 8192, 0), (1039680, 8729, 2**33)],
)
@torch.inference_mode()
def test_packed_scale_offset_exceeds_uint32_graph(stride, high_block, storage_offset):
    """A valid packed cache at 16 GiB detects int32 wrap without an OOB read.

    Only two tiny pages are touched. An old uint32 scale offset wraps the high
    block to allocated block zero, producing a safe, deterministic wrong value.
    """
    block_size, head_dim = 64, 128
    page_bytes = block_size * (head_dim + 4)
    required = storage_offset + high_block * stride + page_bytes
    free, _ = torch.cuda.mem_get_info()
    if free < required + 512 * 1024 * 1024:
        pytest.skip("packed 64-bit address regression requires 17 GiB free GPU memory")
    backing = torch.empty(required, dtype=torch.uint8, device="cuda")
    cache = backing.as_strided(
        (high_block + 1, block_size, head_dim + 4),
        (stride, head_dim + 4, 1),
        storage_offset=storage_offset,
    )
    flat = cache.view(high_block + 1, -1)
    for b, value, scale in [(0, 1, 1.25), (high_block, 17, 2.5)]:
        flat[b, : block_size * head_dim].fill_(value)
        flat[b, block_size * head_dim :].view(torch.float32).fill_(scale)
    # The second geometry is the installed native planner's packed stride and
    # a physical block ID from the failed scheduler dump. Guard storage turns
    # the old negative int32 address into an in-bounds, distinguishable value.
    old_offset = (high_block * (stride // 4) + 2**31) % 2**32 - 2**31
    old_address = storage_offset + block_size * head_dim + old_offset * 4
    assert 0 <= old_address < required - 4
    backing[old_address : old_address + 4].view(torch.float32).fill_(1.25)
    table = torch.tensor([[high_block]], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    out = torch.empty((1, head_dim), dtype=_default_fp8_dtype(), device="cuda")
    scales = torch.empty((1, 4), dtype=torch.uint8, device="cuda")
    expected = torch.empty_like(out)
    expected_scales = torch.empty_like(scales)
    torch_gather(cache, expected, expected_scales, table, cu)
    cp_gather_indexer_k_quant_cache(cache, out, scales, table, cu)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        scales.view(torch.float32), expected_scales.view(torch.float32), rtol=0, atol=0
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cp_gather_indexer_k_quant_cache(cache, out, scales, table, cu)
    for version in [1, 2]:
        flat[high_block, : block_size * head_dim].fill_(30 + version)
        flat[high_block, block_size * head_dim :].view(torch.float32).fill_(
            3.25 + version
        )
        out.view(torch.uint8).fill_(123)
        scales.fill_(123)
        torch_gather(cache, expected, expected_scales, table, cu)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
