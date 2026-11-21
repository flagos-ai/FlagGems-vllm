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

import flaggems_vllm
from flaggems_vllm.ops.cp_gather_indexer_k_quant_cache import (
    cp_gather_indexer_k_quant_cache as generic_cp_gather_indexer_k_quant_cache,
)

# Bind the top-level entry: on vendor backends with a specialized
# implementation (runtime/backend/_<vendor>/ops), this is the vendor version
# (replaced at import time); elsewhere it is the generic one.
cp_gather_indexer_k_quant_cache = flaggems_vllm.cp_gather_indexer_k_quant_cache

# Device-agnostic entry point: the runtime resolves the vendor device name --
# "cuda" on NVIDIA / MetaX / Hygon / T-Head, "musa" on MThreads, "npu" on
# Ascend. Never hard-code "cuda" here.
device = flaggems_vllm.device
torch_device_fn = flaggems_vllm.runtime.torch_device_fn
_device_module = getattr(torch, device, None)
_HAS_DEVICE = _device_module is not None and _device_module.is_available()

_TARGET_VLLM_VERSION = Version("0.20.2")
_NEXT_VLLM_VERSION = Version("0.21.0")

pytestmark = pytest.mark.skipif(
    not _HAS_DEVICE,
    reason=f"requires an available {device} device",
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
    if device != "cuda" or getattr(torch.version, "cuda", None) is None:
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
    num_blocks, block_size, _ = kv_cache.shape
    head_dim = dst_k.size(1)
    dst_k_bytes = dst_k.view(torch.uint8)
    dst_scale_bytes = dst_scale.view(torch.uint8)
    num_scale_bytes = dst_scale_bytes.size(1)
    num_tokens = int(cu_seq_lens[-1].item())
    if num_tokens == 0:
        return

    flat_cache = kv_cache.view(num_blocks, -1)
    cache_values = flat_cache[:, : block_size * head_dim].view(
        num_blocks, block_size, head_dim
    )
    cache_scales = flat_cache[:, block_size * head_dim :].view(
        num_blocks, block_size, num_scale_bytes
    )

    cu_seq_lens = cu_seq_lens.long()
    seq_lens = cu_seq_lens[1:] - cu_seq_lens[:-1]
    batch_ids = torch.repeat_interleave(
        torch.arange(seq_lens.numel(), device=kv_cache.device),
        seq_lens,
        output_size=num_tokens,
    )
    token_offsets = (
        torch.arange(num_tokens, device=kv_cache.device) - cu_seq_lens[batch_ids]
    )
    block_ids = block_table[batch_ids, token_offsets // block_size].long()
    block_offsets = token_offsets % block_size

    dst_k_bytes[:num_tokens] = cache_values[block_ids, block_offsets]
    dst_scale_bytes[:num_tokens] = cache_scales[block_ids, block_offsets]


def _make_cache(num_blocks, block_size, head_dim, quant_block_size, device):
    # The op is a byte-exact gather, so value bytes need not be valid fp8.
    # Generating them as raw uint8 avoids an fp8 cast kernel, which some
    # vendor devices do not provide.
    cache_stride = head_dim + head_dim * 4 // quant_block_size
    k_cache = torch.randint(
        0,
        256,
        (num_blocks, block_size, cache_stride),
        dtype=torch.uint8,
        device=device,
    )
    num_quant_blocks = head_dim // quant_block_size
    flat_cache = k_cache.view(num_blocks, -1)
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
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    cu_seqlen = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    cu_seqlen[1:] = torch.cumsum(seq_lens_tensor, dim=0)

    max_blocks = max(math.ceil(seq_len / block_size) for seq_len in seq_lens)
    num_blocks = sum(math.ceil(seq_len / block_size) for seq_len in seq_lens)
    # Hand out physical blocks in a shuffled order so that a kernel which
    # ignores block_table (or mixes up batch ids) cannot pass by accident.
    physical_blocks = torch.randperm(num_blocks, dtype=torch.int32)
    block_table = torch.full((len(seq_lens), max_blocks), -1, dtype=torch.int32)
    next_block = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        num_seq_blocks = math.ceil(seq_len / block_size)
        block_table[batch_idx, :num_seq_blocks] = physical_blocks[
            next_block : next_block + num_seq_blocks
        ]
        next_block += num_seq_blocks

    return block_table.to(device), cu_seqlen.to(device), num_blocks


@pytest.mark.cp_gather_indexer_k_quant_cache
@pytest.mark.parametrize(
    "seq_lens,block_size,head_dim,quant_block_size,extra_tokens",
    [
        # batch <= 16: linear batch scan, TOKEN_BLOCK 1 / 2.
        ([13, 7, 16], 8, 128, 128, 0),
        ([17, 1, 33, 9], 16, 512, 128, 5),
        # batch > 16: binary-search batch lookup, incl. empty sequences.
        (
            [5, 0, 19, 64, 1, 7, 0, 33, 12, 3, 70, 8, 2, 41, 16, 9, 27, 4, 1, 30],
            64,
            128,
            128,
            3,
        ),
        # Many tokens: TOKEN_BLOCK 32, several quant blocks per head. vLLM's
        # CUDA kernel writes one scale per 128 bytes, so quant_block_size
        # stays >= 128 to keep it a valid reference.
        ([300, 129, 511, 64, 1, 257], 64, 256, 128, 0),
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
    block_table, cu_seqlen, num_blocks = _make_gather_metadata(
        seq_lens,
        block_size,
        device,
    )
    valid_tokens = sum(seq_lens)
    allocated_tokens = valid_tokens + extra_tokens

    k_cache = _make_cache(
        num_blocks,
        block_size,
        head_dim,
        quant_block_size,
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

    reference_op = vllm_op if has_vllm else torch_gather
    reference_op(
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
    torch_device_fn.synchronize()

    torch.testing.assert_close(
        gems_k.view(torch.uint8).cpu(),
        reference_k.view(torch.uint8).cpu(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(gems_scale.cpu(), reference_scale.cpu(), rtol=0, atol=0)
    if extra_tokens:
        assert torch.all(gems_k[valid_tokens:].view(torch.uint8) == sentinel)
        assert torch.all(gems_scale[valid_tokens:] == sentinel)


@pytest.mark.cp_gather_indexer_k_quant_cache
@pytest.mark.skipif(device != "cuda", reason="CUDA graph backend required")
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
@pytest.mark.skipif(device != "cuda", reason="CUDA graph backend required")
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


@pytest.mark.cp_gather_indexer_k_quant_cache
@torch.inference_mode()
def test_cp_gather_indexer_k_quant_cache_generic_batch_31():
    # One token per sequence makes cu_seqlen = [0, 1, ..., 31]. The
    # previous iteration bound left token 0 unwritten in the generic kernel.
    batch_size, block_size, head_dim, scale_bytes = 31, 16, 128, 4
    fp8_dtype = _default_fp8_dtype()
    cache = torch.ones(
        (batch_size, block_size, head_dim + scale_bytes),
        dtype=torch.uint8,
        device=device,
    )
    block_table = torch.arange(batch_size, dtype=torch.int32, device=device).view(-1, 1)
    cu_seqlen = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    dst_k = torch.zeros((batch_size, head_dim), dtype=torch.uint8, device=device)
    dst_scale = torch.zeros((batch_size, scale_bytes), dtype=torch.uint8, device=device)

    generic_cp_gather_indexer_k_quant_cache(
        cache, dst_k.view(fp8_dtype), dst_scale, block_table, cu_seqlen
    )
    for output in (dst_k, dst_scale):
        actual = output.cpu()
        torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)
