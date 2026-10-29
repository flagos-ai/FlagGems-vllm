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
