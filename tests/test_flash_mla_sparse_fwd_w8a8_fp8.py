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

import dataclasses
import random
from typing import List

import pytest
import torch

import flaggems_vllm
from benchmark import test_flash_mla_sparse_fwd_w8a8_fp8 as sparse_fixture

random.seed(42)


@dataclasses.dataclass
class Flashmla_Sparse_Test_Param:
    s_q: int
    s_kv: int
    topk: int
    h_q: int = 128
    h_kv: int = 1
    d_qk: int = 512
    d_v: int = 512
    is_all_indices_invalid: bool = False
    num_warmup: int = 5
    num_runs: int = 10
    have_attn_sink: bool = False
    have_topk_length: bool = False
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = flaggems_vllm.device


_flashmla_sparse_counter = 0


class FlashmlaSparseTestKit:
    @staticmethod
    def _init_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def get_correctness_test_params_flashmla():
        cases = [
            Flashmla_Sparse_Test_Param(
                s_q,
                s_kv,
                topk,
                h_q,
                d_qk=d_qk,
                have_attn_sink=have_attn_sink,
                have_topk_length=have_topk_length,
            )
            for s_q in [1, 62, 213]
            for h_q in [128, 64]
            for d_qk in [512, 576]
            for s_kv, topk in [
                (592, 128),
                (1840, 256),
                (1592, 384),
                (1521, 512),
                (95, 128),
                (153, 256),
                (114, 384),
            ]
            for have_attn_sink in [True, False]
            for have_topk_length in [True, False]
        ]
        return cases

    @staticmethod
    def _randperm_batch(
        batch_size: int,
        perm_range: torch.Tensor,
        perm_size: int,
        paddings: List[int],
    ) -> torch.Tensor:
        """
        Generate random permutations in batch
        The return tensor, denoted as `res`, has a shape of [batch_size, perm_size]. `0 <= res[i, :] < perm_range[i]`
        holds.
        Values within each row are unique.
        If, for some `i`, `perm_range[i] < perm_size` holds, then `res[i, :]` contains values in `[0, perm_range[i])`
        as many as possible, and the rest are filled with `padding`.
        """
        assert not torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        perm_range_max = max(int(torch.max(perm_range).item()), perm_size)
        rand = torch.rand(batch_size, perm_range_max, dtype=torch.float32)
        rand[
            torch.arange(0, perm_range_max).broadcast_to(batch_size, perm_range_max)
            >= perm_range.view(batch_size, 1)
        ] = float("-inf")
        res = rand.topk(perm_size, dim=-1, sorted=True).indices.to(torch.int32)
        if len(paddings) == 1:
            res[res >= perm_range.view(batch_size, 1)] = paddings[0]
        else:
            fillers = torch.tensor(paddings, dtype=torch.int32).index_select(
                0,
                torch.randint(0, len(paddings), (res.numel(),), dtype=torch.int32),
            )
            res.masked_scatter_(res >= perm_range.view(batch_size, 1), fillers)
        torch.use_deterministic_algorithms(False)
        return res

    @staticmethod
    def make_input_flashmla(param: Flashmla_Sparse_Test_Param):
        """Create input data for sparse MLA operator by referring to the FlashMLA examples"""
        s_q = param.s_q
        s_kv = param.s_kv
        h_q = param.h_q
        h_kv = param.h_kv
        d_qk = param.d_qk
        topk = param.topk
        have_attn_sink = param.have_attn_sink
        have_topk_length = param.have_topk_length
        is_all_indices_invalid = param.is_all_indices_invalid
        dtype = param.dtype
        device = param.device

        global _flashmla_sparse_counter
        FlashmlaSparseTestKit._init_seed(_flashmla_sparse_counter)
        _flashmla_sparse_counter = _flashmla_sparse_counter + 1

        q = (
            torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device) / 10
            + (random.random() - 0.5) / 10
        )
        kv = (
            torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device) / 10
            + (random.random() - 0.5) / 10
        )
        q = q.clamp_(-10, 10)
        kv = kv.clamp_(-10, 10)
        invalid_indices_candidate = [
            -2147483648,
            -123456,
            -1,
            s_kv,
            114514,
            1919810,
            2147480000,
            2147483647,
        ]
        indices = FlashmlaSparseTestKit._randperm_batch(
            s_q,
            torch.full((s_q,), s_kv, dtype=torch.int32),
            topk,
            invalid_indices_candidate,
        ).view(s_q, h_kv, topk)
        if is_all_indices_invalid:
            all_indices_invalid_mask = torch.randn(s_q, device="cpu") < -2
            indices[
                all_indices_invalid_mask[:, None, None].broadcast_to(indices.shape)
            ] = random.choice(invalid_indices_candidate)
        indices = indices.to(device)

        attn_sink = None
        if have_attn_sink:
            attn_sink = torch.randn((h_q,), dtype=torch.float32, device=device)
            mask = torch.randn((h_q,), dtype=torch.float32, device=device)
            attn_sink[mask < -0.5] = float("-inf")
            attn_sink[mask > +0.5] = float("+inf")

        topk_length = None
        if have_topk_length:
            topk_length = torch.randint(
                0, max(topk + 1, 64), (s_q,), dtype=torch.int32, device=device
            ).clamp_max(topk)
        return q, kv, indices, attn_sink, topk_length


@pytest.mark.flash_mla_sparse_fwd_w8a8_fp8
@pytest.mark.parametrize(
    "param", FlashmlaSparseTestKit.get_correctness_test_params_flashmla()
)
def test_flash_mla_sparse_flashmla(param: Flashmla_Sparse_Test_Param):
    if not sparse_fixture.HAS_FP8_MLA:
        pytest.skip("FP8 MLA requires NVIDIA Hopper and Triton TLE")
    if not sparse_fixture.HAS_VLLM_FLASHMLA_SPARSE:
        raise RuntimeError("Official vLLM CUDA SparseMLA baseline is required")
    q, kv, indices, attn_sink, topk_length = FlashmlaSparseTestKit.make_input_flashmla(
        param
    )
    sm_scale = 0.5
    reference, _, reference_lse = sparse_fixture.vllm_flash_mla_sparse_fwd(
        q, kv, indices, sm_scale, param.d_v, attn_sink, topk_length
    )
    packed = sparse_fixture.quantize_sparse_fp8_fixture(
        q, kv, indices, sm_scale, param.d_v, attn_sink, topk_length
    )
    output, lse = sparse_fixture.run_sparse_fp8_fixture(*packed)
    sparse_fixture.assert_sparse_fixture_close(
        output[:, 0], lse[:, :, 0], reference, reference_lse
    )


@pytest.mark.flash_mla_sparse_fwd_w8a8_fp8
@pytest.mark.skipif(
    not sparse_fixture.HAS_FP8_MLA, reason="FP8 MLA requires NVIDIA Hopper and TLE"
)
def test_flash_mla_sparse_fp8_long_reduction(monkeypatch):
    # Real record 195 exposes the error from persistent FP8 PV accumulation.
    monkeypatch.setattr(sparse_fixture, "_flashmla_sparse_counter", 195)
    param = sparse_fixture.TestParam(547, 98816, 8320, h_q=64, d_qk=512)
    inputs = next(sparse_fixture.FlashmlaSparseBenchmark.make_input_flashmla(param))
    reference, _, reference_lse = sparse_fixture.vllm_flash_mla_sparse_fwd(*inputs)
    packed = sparse_fixture.quantize_sparse_fp8_fixture(*inputs)
    output, lse = sparse_fixture.run_sparse_fp8_fixture(*packed)
    sparse_fixture.assert_sparse_fixture_close(
        output[:, 0], lse[:, :, 0], reference, reference_lse
    )
