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

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

import flaggems_vllm

# Load the host-only scheduler without importing MetaX kernels on other vendors.
_scheduler_path = (
    Path(__file__).resolve().parents[1]
    / "src/flaggems_vllm/runtime/backend/_metax/ops/flash_attention/scheduling.py"
)
_spec = importlib.util.spec_from_file_location(
    "_metax_flash_attention_scheduler_test", _scheduler_path
)
_scheduler = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _scheduler
_spec.loader.exec_module(_scheduler)
MetaXAttentionScheduler = _scheduler.MetaXAttentionScheduler
MetaXKernelFamily = _scheduler.MetaXKernelFamily
validate_metax_attention_plan = _scheduler.validate_metax_attention_plan

pytestmark = pytest.mark.flash_attn_varlen_func


def _decode_plan(ratio=6, **overrides):
    values = dict(
        is_bfloat16=True,
        is_paged=True,
        block_size=784,
        is_seqused_k=True,
        max_seqlen_q=ratio,
        max_seqlen_k=4096,
        total_q=64 * ratio,
        batch_size=64,
        num_heads=2,
        num_heads_k=2,
        head_size=256,
        seqlenq_ngroups_swapped=True,
    )
    values.update(overrides)
    return MetaXAttentionScheduler.build(**values)


@pytest.mark.parametrize("ratio", [5, 6, 8, 16])
@pytest.mark.parametrize("num_splits", [0, 2, 3, 16])
def test_metax_swapped_decode_split_plan(ratio, num_splits):
    plan = _decode_plan(ratio, num_splits=num_splits)
    assert plan.family is MetaXKernelFamily.SPLIT_KV
    assert plan.split_kv_block_m == 16
    assert plan.split_kv_q_tiles == 1
    assert 2 <= plan.max_splits <= 32
    validate_metax_attention_plan(plan)


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_splits": 1},
        {"is_bfloat16": False},
        {"is_paged": False},
        {"is_local": True},
        {"is_alibi": True},
        {"is_softcap": True},
        {"is_dropout": True},
        {"is_seqused_k": False},
        {"total_q": 64 * 6 - 1},
        {"num_heads": 4},
        {"head_size": 128},
        {"seqlenq_ngroups_swapped": False, "is_cu_seqlens_q": True},
        {"max_seqlen_k": 16},
        {"max_seqlen_q": 17, "total_q": 64 * 17},
    ],
)
def test_metax_swapped_decode_guard_miss(overrides):
    plan = _decode_plan(**overrides)
    assert plan.family is MetaXKernelFamily.LEGACY
    validate_metax_attention_plan(plan)


@pytest.mark.parametrize("max_k, block_n", [(1024, 16), (4095, 16), (4096, 16)])
def test_metax_swapped_decode_kv_tile_boundary(max_k, block_n):
    plan = _decode_plan(max_seqlen_k=max_k)
    assert plan.family is MetaXKernelFamily.SPLIT_KV
    assert plan.split_kv_block_n == block_n
    validate_metax_attention_plan(plan)


def _reference(q, k, v, table, lengths, ratio):
    outputs, lses = [], []
    for i, length in enumerate(lengths):
        if length == 0:
            outputs.append(torch.zeros_like(q[i : i + 1], dtype=torch.float32))
            lses.append(torch.full((q.shape[1], 1), -torch.inf, device=q.device))
            continue
        ids = table[i, : (length + k.shape[1] - 1) // k.shape[1]].long()
        ki = k[ids].reshape(-1, 2, 256)[:length].float()
        vi = v[ids].reshape(-1, 2, 256)[:length].float()
        ki, vi = (x.repeat_interleave(ratio, dim=1) for x in (ki, vi))
        scores = torch.einsum("qhd,khd->hqk", q[i : i + 1].float(), ki) / 16
        outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), vi))
        lses.append(torch.logsumexp(scores, dim=-1))
    return torch.cat(outputs), torch.cat(lses, dim=1)


@pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX route")
@pytest.mark.parametrize("ratio", [5, 6, 8, 16])
@pytest.mark.parametrize("page", [16, 32, 784])
@pytest.mark.parametrize("num_splits", [0, 3])
@pytest.mark.parametrize("b", [1, 4])
@torch.inference_mode()
def test_metax_swapped_decode_paged_graph(ratio, page, num_splits, b):
    torch.manual_seed(41)
    maximum = 2049
    pages = (maximum + page - 1) // page
    # Noncontiguous Q, output, page table and cache page strides.
    q = torch.randn(b, 4 * ratio, 256, device="cuda", dtype=torch.bfloat16)
    q = q[:, : 2 * ratio]
    out_storage = torch.full_like(q.repeat(1, 2, 1), 13)
    out = out_storage[:, : 2 * ratio]
    k = torch.randn(b * pages, page * 2, 2, 256, device="cuda", dtype=q.dtype)
    k = k[:, :page]
    v = torch.randn_like(k.repeat(1, 2, 1, 1))[:, :page]
    table_storage = torch.full((b, pages * 2), -1, dtype=torch.int32, device="cuda")
    table = table_storage[:, :pages]
    table.copy_(torch.randperm(b * pages, device="cuda").reshape(b, pages))
    cu = torch.arange(b + 1, device="cuda", dtype=torch.int32)
    used = torch.empty(b, device="cuda", dtype=torch.int32)

    def run():
        return flaggems_vllm.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu,
            seqused_k=used,
            max_seqlen_q=1,
            max_seqlen_k=262144,
            block_table=table,
            causal=True,
            num_splits=num_splits,
            return_softmax_lse=True,
        )

    lengths = [maximum, 17, 0, 1023][:b]
    used.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
    expected, expected_lse = _reference(q, k, v, table, lengths, ratio)
    actual, lse = run()
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual.float(), expected, atol=0.002, rtol=0.02)
    torch.testing.assert_close(lse, expected_lse, atol=0.0001, rtol=0.0001)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, lse = run()
    for lengths in ([1, 784, maximum, 0], [maximum, 0, 16, 1024]):
        lengths = lengths[:b]
        q.mul_(0.75)
        used.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        expected, expected_lse = _reference(q, k, v, table, lengths, ratio)
        torch.testing.assert_close(actual.float(), expected, atol=0.002, rtol=0.02)
        torch.testing.assert_close(lse, expected_lse, atol=0.0001, rtol=0.0001)
        assert torch.all(out_storage[:, 2 * ratio :] == 13)
