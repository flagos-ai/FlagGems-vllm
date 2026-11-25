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

import pytest
import torch

import flaggems_vllm
import flaggems_vllm.testing as fg_testing

pytestmark = pytest.mark.dequantize_and_gather_k_cache

try:
    from vllm.models.deepseek_v4.common.ops import (
        dequantize_and_gather_k_cache as vllm_dequantize_and_gather_k_cache,
    )

    # The vllm reference kernel does not compile on every backend (pointer bitcast on
    # Ascend), so import success alone is not enough: probe with a minimal call.
    try:
        _pr = torch.zeros((1, 1, 64), dtype=torch.bfloat16, device=flaggems_vllm.device)
        _pk = torch.zeros((1, 1024), dtype=torch.uint8, device=flaggems_vllm.device)
        vllm_dequantize_and_gather_k_cache(
            _pr,
            _pk,
            torch.full((1,), 1, dtype=torch.int32, device=flaggems_vllm.device),
            torch.full((1,), 1, dtype=torch.int32, device=flaggems_vllm.device),
            torch.zeros((1, 1), dtype=torch.int32, device=flaggems_vllm.device),
            64,
            0,
        )
        flaggems_vllm.runtime.torch_device_fn.synchronize()
        _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE = True
    except Exception:
        _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE = False
except Exception:
    vllm_dequantize_and_gather_k_cache = None
    _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE = False


def _op_runs():
    """Probe once that this backend can actually run the operator."""
    try:
        _nope, _rope = 448, 64
        _ss = (_nope + 63) // 64 + 1
        _bs = 64
        _stride = _bs * (_nope + _rope * 2) + _bs * _ss
        out = torch.zeros(
            (1, 1, _nope + _rope), dtype=torch.bfloat16, device=flaggems_vllm.device
        )
        k_cache = torch.zeros(
            (1, _stride), dtype=torch.uint8, device=flaggems_vllm.device
        )
        flaggems_vllm.dequantize_and_gather_k_cache(
            out,
            k_cache,
            torch.full((1,), 1, dtype=torch.int32, device=flaggems_vllm.device),
            torch.full((1,), 1, dtype=torch.int32, device=flaggems_vllm.device),
            torch.zeros((1, 1), dtype=torch.int32, device=flaggems_vllm.device),
            _bs,
            rope_dim=_rope,
            nope_dim=_nope,
            scale_slots=_ss,
        )
        flaggems_vllm.runtime.torch_device_fn.synchronize()
        return True
    except Exception:
        return False


_OP_RUNS = _op_runs()


def _fill_cache(k_cache, expected_rows, block_size, nope_dim, rope_dim, scale_slots):
    token_data_size = nope_dim + rope_dim * 2
    for slot, row in expected_rows.items():
        block = slot // block_size
        pos = slot % block_size
        base = pos * token_data_size
        # Ascend rejects fp8 in aclnnInplaceCopy, so build the bytes on CPU
        # and move them as uint8.
        x = (
            (torch.arange(nope_dim, dtype=torch.float32) / 32.0 + slot / 8.0)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
            .to(k_cache.device)
        )
        rope = (
            (torch.arange(rope_dim, dtype=torch.float32) / 16.0 + slot)
            .to(torch.bfloat16)
            .view(torch.uint8)
            .to(k_cache.device)
        )
        k_cache[block, base : base + nope_dim] = x
        k_cache[block, base + nope_dim : base + nope_dim + rope_dim * 2] = rope
        scale_base = block_size * token_data_size + pos * scale_slots
        k_cache[block, scale_base : scale_base + scale_slots] = 127
        row[..., :nope_dim] = (
            x.cpu()
            .view(torch.float8_e4m3fn)
            .to(torch.float32)
            .to(k_cache.device)
            .to(torch.bfloat16)
        )
        row[..., nope_dim : nope_dim + rope_dim] = rope.view(torch.bfloat16)


@pytest.mark.parametrize(
    ("batch", "seq_len", "gather_len", "block_size", "nope_dim", "rope_dim"),
    [
        (1, 6, 3, 4, 64, 16),
        (2, 12, 5, 64, 448, 64),
    ],
)
@pytest.mark.skipif(
    not _OP_RUNS,
    reason="requires a backend able to run this operator",
)
def test_dequantize_and_gather_k_cache_accuracy(
    batch, seq_len, gather_len, block_size, nope_dim, rope_dim
):
    device = flaggems_vllm.device
    scale_slots = (nope_dim + 63) // 64 + (1 if nope_dim % 64 == 0 else 0)
    output_dim = nope_dim + rope_dim
    token_data_size = nope_dim + rope_dim * 2
    block_stride = block_size * token_data_size + block_size * scale_slots
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch * blocks_per_seq
    k_cache = torch.zeros((num_blocks, block_stride), device=device, dtype=torch.uint8)
    out = torch.empty(
        (batch, gather_len, output_dim), device=device, dtype=torch.bfloat16
    )
    expected = torch.empty_like(out)
    rows = {}
    for req in range(batch):
        start_pos = seq_len - gather_len
        for local_i in range(gather_len):
            pos = start_pos + local_i
            physical_block = req * blocks_per_seq + pos // block_size
            slot = physical_block * block_size + pos % block_size
            rows[slot] = expected[req : req + 1, local_i : local_i + 1, :]
    _fill_cache(k_cache, rows, block_size, nope_dim, rope_dim, scale_slots)

    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    gather_lens = torch.full((batch,), gather_len, device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
        batch, blocks_per_seq
    )
    flaggems_vllm.dequantize_and_gather_k_cache(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
        scale_slots=scale_slots,
    )

    fg_testing.assert_close(out, expected, dtype=torch.bfloat16, equal_nan=True)


@pytest.mark.skipif(
    not _OP_RUNS or not _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE,
    reason="requires a vllm dequantize_and_gather_k_cache reference to compare against",
)
def test_dequantize_and_gather_k_cache_vllm_accuracy():
    device = flaggems_vllm.device
    batch = 2
    seq_len = 12
    gather_len = 5
    block_size = 64
    nope_dim = 448
    rope_dim = 64
    scale_slots = 8
    output_dim = nope_dim + rope_dim
    token_data_size = nope_dim + rope_dim * 2
    block_stride = block_size * token_data_size + block_size * scale_slots
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch * blocks_per_seq
    k_cache = torch.zeros((num_blocks, block_stride), device=device, dtype=torch.uint8)
    expected_rows = torch.empty(
        (batch, gather_len, output_dim), device=device, dtype=torch.bfloat16
    )
    rows = {}
    for req in range(batch):
        start_pos = seq_len - gather_len
        for local_i in range(gather_len):
            pos = start_pos + local_i
            physical_block = req * blocks_per_seq + pos // block_size
            slot = physical_block * block_size + pos % block_size
            rows[slot] = expected_rows[req : req + 1, local_i : local_i + 1, :]
    _fill_cache(k_cache, rows, block_size, nope_dim, rope_dim, scale_slots)

    actual = torch.empty_like(expected_rows)
    expected = torch.empty_like(expected_rows)
    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    gather_lens = torch.full((batch,), gather_len, device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
        batch, blocks_per_seq
    )

    flaggems_vllm.dequantize_and_gather_k_cache(
        actual,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
        scale_slots=scale_slots,
    )
    vllm_dequantize_and_gather_k_cache(
        expected,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        0,
    )

    fg_testing.assert_close(actual, expected, dtype=torch.bfloat16, equal_nan=True)
