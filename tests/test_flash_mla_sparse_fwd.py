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
import importlib
import random
from types import SimpleNamespace
from typing import List, Optional, Tuple

import pytest
import torch

import flaggems_vllm

random.seed(42)


def _get_vllm_flashmla_sparse_reference():
    """Return vLLM's sparse FlashMLA op only when its extension is usable."""
    try:
        from vllm.v1.attention.ops import flashmla
    except Exception as error:
        return None, f"vLLM FlashMLA could not be imported: {error}"

    support_check = getattr(flashmla, "is_flashmla_sparse_supported", None)
    if not callable(support_check):
        return None, "vLLM does not expose is_flashmla_sparse_supported()"

    try:
        support = support_check()
    except Exception as error:
        return None, f"vLLM FlashMLA capability check failed: {error}"

    if isinstance(support, tuple):
        supported, reason = support
    else:
        supported, reason = bool(support), None
    if not supported:
        return None, reason or "vLLM sparse FlashMLA is not supported"

    reference = getattr(flashmla, "flash_mla_sparse_fwd", None)
    if not callable(reference):
        return None, "vLLM sparse FlashMLA reference is not callable"
    return reference, None


vllm_flash_mla_sparse_fwd, VLLM_FLASHMLA_SPARSE_UNAVAILABLE_REASON = (
    _get_vllm_flashmla_sparse_reference()
)
HAS_VLLM_FLASHMLA_SPARSE = vllm_flash_mla_sparse_fwd is not None
if not HAS_VLLM_FLASHMLA_SPARSE:
    torch.set_float32_matmul_precision("high")


def _has_hq4_cuda() -> bool:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    return torch.cuda.get_device_capability()[0] >= 8


_HAS_HQ4_CUDA = _has_hq4_cuda()


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.parametrize(
    "updates,missing,expected",
    [
        ({}, None, True),
        ({"shared_memory_per_block_optin": 215039}, None, False),
        ({"shared_memory_per_block_optin": 215040}, None, True),
        ({"name": "NVIDIA H100"}, None, False),
        ({"major": 8}, None, False),
        ({"minor": 1}, None, False),
        ({}, "shared_memory_per_block_optin", False),
        ({}, "name", False),
        ({}, "major", False),
        ({}, "minor", False),
    ],
)
def test_flash_mla_sparse_quad_pipeline_resource_gate(
    monkeypatch, updates, missing, expected
):
    module = importlib.import_module("flaggems_vllm.ops.flashmla_sparse")
    properties = dict(
        name="NVIDIA H20", major=9, minor=0, shared_memory_per_block_optin=232448
    )
    properties.update(updates)
    if missing:
        del properties[missing]
    monkeypatch.setattr(module.triton, "__version__", "3.7.1")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(**properties),
    )
    gate = module._use_quad_gather_pipeline
    gate.cache_clear()
    try:
        assert gate(torch.device("cuda", 0)) is expected
    finally:
        gate.cache_clear()


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.parametrize("version", ["3.7.0", "3.8.0", "3.7.1.dev0"])
def test_flash_mla_sparse_quad_pipeline_other_compilers(monkeypatch, version):
    module = importlib.import_module("flaggems_vllm.ops.flashmla_sparse")
    monkeypatch.setattr(module.triton, "__version__", version)

    def unexpected_query(device):
        pytest.fail("An unsupported compiler must retain the original schedule")

    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected_query)
    gate = module._use_quad_gather_pipeline
    gate.cache_clear()
    try:
        assert gate(torch.device("cuda", 0)) is False
    finally:
        gate.cache_clear()


@pytest.mark.flash_mla_sparse_fwd
def test_flash_mla_sparse_quad_pipeline_device_cache(monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.flashmla_sparse")
    monkeypatch.setattr(module.triton, "__version__", "3.7.1")
    queries = []

    def properties(device):
        queries.append(device)
        return SimpleNamespace(
            name="NVIDIA H20",
            major=9,
            minor=0,
            shared_memory_per_block_optin=232448 if device.index == 0 else 163840,
        )

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    gate = module._use_quad_gather_pipeline
    gate.cache_clear()
    try:
        assert gate(torch.device("cpu")) is False
        assert gate(torch.device("cuda")) is False
        assert not queries
        for _ in range(3):
            assert gate(torch.device("cuda", 0)) is True
            assert gate(torch.device("cuda", 1)) is False
        assert queries == [torch.device("cuda", 0), torch.device("cuda", 1)]
    finally:
        gate.cache_clear()


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


# used by make_input_flashmla
_flashmla_sparse_counter = 0


class FlashmlaSparseTestKit:
    # used by torch vertion flashmla_sparse
    @staticmethod
    def _merge_two_lse(
        lse0: torch.Tensor, lse1: Optional[torch.Tensor], s_q: int, h_q: int
    ) -> torch.Tensor:
        if lse1 is None:
            return lse0

        return torch.logsumexp(
            torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
            dim=0,
        )

    # torch version flashmla_sparse
    @staticmethod
    def torch_flash_mla_sparse_fwd(
        s_q: int,
        s_kv: int,
        h_q: int,
        h_kv: int,
        d_qk: int,
        topk: int,
        q: torch.Tensor,  # [s_q, h_q, d_qk]
        kv: torch.Tensor,  # [s_q, 1, d_qk]
        indices: torch.Tensor,  # [s_q, 1, topk]
        sm_scale: float,
        d_v: int,
        attn_sink: Optional[torch.Tensor],  # [h_q]
        topk_length: Optional[torch.Tensor],  # [s_q]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
        - o: [s_q, h_q, dv]
        - o_fp32: [s_q, h_q, dv]
        - max_logits: [s_q, h_q]
        - lse: [s_q, h_q]
        """
        indices = indices.clone().squeeze(1)
        if topk_length is not None:
            mask = torch.arange(topk, device=topk_length.device).unsqueeze(
                0
            ).broadcast_to(s_q, topk) >= topk_length.unsqueeze(1)
            indices[mask] = -1
        invalid_mask = (indices < 0) | (indices >= s_kv)
        indices[invalid_mask] = 0
        q = q.float()
        gathered_kv = (
            kv.index_select(dim=0, index=indices.flatten())
            .reshape(s_q, topk, d_qk)
            .float()
        )
        P = q @ gathered_kv.transpose(1, 2)
        P *= sm_scale
        P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

        orig_lse = torch.logsumexp(P, dim=-1)
        max_logits = P.max(dim=-1).values

        lse_for_o = FlashmlaSparseTestKit._merge_two_lse(orig_lse, attn_sink, s_q, h_q)
        if not torch.is_inference_mode_enabled():
            lse_for_o = lse_for_o.clone()
        lse_for_o[lse_for_o == float("-inf")] = float(
            "+inf"
        )  # So that corresponding O will be 0
        s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
        out = s_for_o @ gathered_kv[..., :d_v]

        lonely_q_mask = orig_lse == float("-inf")
        orig_lse[lonely_q_mask] = float("+inf")
        return (out.to(torch.bfloat16), max_logits, orig_lse)

    @staticmethod
    def get_correctness_test_params():
        cases = [
            Flashmla_Sparse_Test_Param(s_q, s_kv, topk, h_q, h_kv, d_qk, d_v)
            for s_q in [64, 128, 512]
            for s_kv in [1024, 2048, 4096]
            for h_q in [64, 128, 256]
            for h_kv in [1]
            for d_qk in [576]
            for d_v in [512]
            for topk in [64, 128, 256]
        ]
        return cases

    @staticmethod
    def _init_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def make_input(param: Flashmla_Sparse_Test_Param):
        """Create input data for sparse MLA operator"""
        S = param.s_q
        H = param.h_q
        DQK = param.d_qk
        SKV = param.s_kv
        HKV = param.h_kv
        topk = param.topk
        dtype = param.dtype
        device = param.device
        requires_grad = False

        FlashmlaSparseTestKit._init_seed(42)

        q = torch.randn((S, H, DQK), dtype=dtype, device=device).requires_grad_(
            requires_grad
        )
        kv = torch.randn((SKV, HKV, DQK), dtype=dtype, device=device).requires_grad_(
            requires_grad
        )

        indices = torch.full((S, HKV, topk), SKV, dtype=torch.int32, device=device)
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[t, h, : len(i_i)] = i_i

        return q, kv, indices

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


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.parametrize("param", FlashmlaSparseTestKit.get_correctness_test_params())
def test_flashmla_sparse(param):
    """Sparse MLA forward propagation test"""
    # Skip FlashMLA unsupported cases
    if param.h_q != 64 and param.h_q != 128:
        # RuntimeError: Unsupported h_q: 256
        # FlashMLA csrc/api/sparse_fwd.h:197
        # FlashMLA requires that h_q is 64 or 128
        return

    if param.topk % 128 != 0:
        # Assertion `params.topk % (2*B_TOPK) == 0` failed
        # FlashMLA csrc/sm90/prefill/sparse/phase1.cuh:577
        # FlashMLA csrc/sm90/prefill/sparse/config.h:27 "B_TOPK = 64"
        # topk not divisible by 128, not supported by FlashMLA
        return

    # Create input
    q, kv, indices = FlashmlaSparseTestKit.make_input(param)
    sm_scale = param.d_qk**-0.5

    if HAS_VLLM_FLASHMLA_SPARSE:
        ref_output, ref_max_logbits, ref_lse = vllm_flash_mla_sparse_fwd(
            q, kv, indices, sm_scale, param.d_v
        )
    else:
        (
            ref_output,
            ref_max_logbits,
            ref_lse,
        ) = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
            param.s_q,
            param.s_kv,
            param.h_q,
            param.h_kv,
            param.d_qk,
            param.topk,
            q,
            kv,
            indices,
            sm_scale,
            param.d_v,
            None,
            None,
        )

    # Your operator implementation
    your_output, your_max_logbits, your_lse = flaggems_vllm.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        param.d_v,
    )

    # Accuracy comparison
    flaggems_vllm.testing.assert_close(your_output, ref_output, param.dtype, atol=1e-2)
    flaggems_vllm.testing.assert_close(
        your_max_logbits, ref_max_logbits, torch.float32, atol=1e-4
    )
    flaggems_vllm.testing.assert_close(your_lse, ref_lse, torch.float32, atol=1e-4)


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.parametrize(
    "param", FlashmlaSparseTestKit.get_correctness_test_params_flashmla()
)
def test_flash_mla_sparse_flashmla(param: Flashmla_Sparse_Test_Param):
    """Sparse MLA forward propagation test from FlashMLA"""
    # Create input
    q, kv, indices, attn_sink, topk_length = FlashmlaSparseTestKit.make_input_flashmla(
        param
    )
    sm_scale = 0.5

    if HAS_VLLM_FLASHMLA_SPARSE:
        ref_output, ref_max_logbits, ref_lse = vllm_flash_mla_sparse_fwd(
            q, kv, indices, sm_scale, param.d_v, attn_sink, topk_length
        )
    else:
        (
            ref_output,
            ref_max_logbits,
            ref_lse,
        ) = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
            param.s_q,
            param.s_kv,
            param.h_q,
            param.h_kv,
            param.d_qk,
            param.topk,
            q,
            kv,
            indices,
            sm_scale,
            param.d_v,
            attn_sink,
            topk_length,
        )

    # Your operator implementation
    your_output, your_max_logbits, your_lse = flaggems_vllm.flash_mla_sparse_fwd(
        q, kv, indices, sm_scale, param.d_v, attn_sink, topk_length
    )

    # Accuracy comparison
    torch.testing.assert_close(
        your_output, ref_output, atol=8e-4, rtol=3.01 / 128, equal_nan=False
    )  # cos_diff_tol=7e-6
    torch.testing.assert_close(
        your_max_logbits,
        ref_max_logbits,
        atol=1e-6,
        rtol=2.01 / 65536,
        equal_nan=False,
    )
    torch.testing.assert_close(
        your_lse, ref_lse, atol=1e-6, rtol=2.01 / 65536, equal_nan=False
    )


def test_flash_mla_sparse_hq4_flag_isolation(monkeypatch):
    """The default call must remain routed to the pre-existing implementation."""
    module = importlib.import_module("flaggems_vllm.ops.flashmla_sparse")
    sentinel = object()
    seen = []

    def fake_default(*args):
        seen.append(args)
        return sentinel

    monkeypatch.setattr(module, "_flash_mla_sparse_fwd_default", fake_default)
    q = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    kv = torch.empty((1, 1, 512), dtype=torch.bfloat16)
    indices = torch.empty((1, 1, 128), dtype=torch.int32)
    assert module.flash_mla_sparse_fwd(q, kv, indices, 1.0) is sentinel
    assert len(seen) == 1

    with pytest.raises(ValueError, match="enable_hq4_sparse_prefill=True"):
        module.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            1.0,
            out=torch.empty((1, 4, 512), dtype=torch.bfloat16),
        )
    assert len(seen) == 1


@pytest.mark.parametrize(
    "aliased_name",
    [
        "q",
        "kv",
        "indices",
        "attn_sink",
        "topk_length",
        "pair_metadata",
        "quad_metadata",
    ],
)
def test_flash_mla_sparse_hq4_rejects_out_storage_alias(monkeypatch, aliased_name):
    module = importlib.import_module("flaggems_vllm.ops.flashmla_sparse")
    values = {
        "q": torch.empty((4, 4, 512), dtype=torch.bfloat16),
        "kv": torch.empty((8, 1, 512), dtype=torch.bfloat16),
        "indices": torch.empty((4, 1, 2176), dtype=torch.int32),
        "attn_sink": torch.empty((4,), dtype=torch.float32),
        "topk_length": torch.empty((4,), dtype=torch.int32),
        "pair_metadata": torch.empty((2,), dtype=torch.int32),
        "quad_metadata": torch.empty((1,), dtype=torch.int32),
        "out": torch.empty((4, 4, 512), dtype=torch.bfloat16),
    }

    def fake_overlap(first, second):
        assert first is values["out"]
        return second is values[aliased_name]

    monkeypatch.setattr(module, "_tensors_share_storage", fake_overlap)
    with pytest.raises(ValueError, match=f"storage with {aliased_name}"):
        module.flash_mla_sparse_fwd(
            values["q"],
            values["kv"],
            values["indices"],
            1.0,
            attn_sink=values["attn_sink"],
            topk_length=values["topk_length"],
            enable_hq4_sparse_prefill=True,
            out=values["out"],
            return_stats=False,
            pair_metadata=values["pair_metadata"],
            pair_window_size=128,
            quad_metadata=values["quad_metadata"],
            max_kv_length=1280,
        )


def test_flash_mla_sparse_hq4_rejects_real_q_alias():
    q = torch.empty((1, 4, 512), dtype=torch.bfloat16)
    kv = torch.empty((1, 1, 512), dtype=torch.bfloat16)
    indices = torch.empty((1, 1, 32), dtype=torch.int32)
    with pytest.raises(ValueError, match="storage with q"):
        flaggems_vllm.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            1.0,
            enable_hq4_sparse_prefill=True,
            out=q,
        )


@pytest.mark.parametrize("grad_name", ["q", "kv", "attn_sink", "out"])
def test_flash_mla_sparse_hq4_is_inference_only(grad_name):
    values = {
        "q": torch.empty((1, 4, 512), dtype=torch.bfloat16),
        "kv": torch.empty((1, 1, 512), dtype=torch.bfloat16),
        "indices": torch.empty((1, 1, 32), dtype=torch.int32),
        "attn_sink": torch.empty((4,), dtype=torch.float32),
        "out": torch.empty((1, 4, 512), dtype=torch.bfloat16),
    }
    values[grad_name].requires_grad_(True)
    with pytest.raises(RuntimeError, match=f"{grad_name}.requires_grad"):
        flaggems_vllm.flash_mla_sparse_fwd(
            values["q"],
            values["kv"],
            values["indices"],
            1.0,
            attn_sink=values["attn_sink"],
            enable_hq4_sparse_prefill=True,
            out=values["out"],
        )


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("s_q", [64, 128])
def test_flash_mla_sparse_hq4_padded_q_and_out(s_q):
    """Cover common prefill lengths and padded physical head storage."""
    s_kv, topk = 4096, 128
    torch.manual_seed(20260923)
    q_storage = torch.randn((s_q, 64, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    q = q_storage[:, :4]
    kv = torch.randn((s_kv, 1, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    indices = torch.randint(0, s_kv, (s_q, 1, topk), device="cuda", dtype=torch.int32)
    topk_length = torch.randint(1, topk + 1, (s_q,), device="cuda", dtype=torch.int32)
    reference, ref_max, ref_lse = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
        s_q,
        s_kv,
        4,
        1,
        512,
        topk,
        q,
        kv,
        indices,
        512**-0.5,
        512,
        None,
        topk_length,
    )

    output_storage = torch.full(
        (s_q, 64, 512), 17.0, device="cuda", dtype=torch.bfloat16
    )
    out = output_storage[:, :4]
    untouched = output_storage[:, 4:].clone()
    output, max_logits, lse = flaggems_vllm.flash_mla_sparse_fwd(
        q_storage,
        kv,
        indices,
        512**-0.5,
        topk_length=topk_length,
        enable_hq4_sparse_prefill=True,
        logical_num_heads=4,
        out=out,
    )
    assert output is out
    torch.testing.assert_close(
        output, reference, atol=8e-4, rtol=3.01 / 128, equal_nan=False
    )
    torch.testing.assert_close(
        max_logits, ref_max, atol=1e-6, rtol=2.01 / 65536, equal_nan=False
    )
    torch.testing.assert_close(
        lse, ref_lse, atol=1e-6, rtol=2.01 / 65536, equal_nan=False
    )
    assert torch.equal(output_storage[:, 4:], untouched)


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_mla_sparse_hq4_metadata_and_invalid_descriptor_fallback():
    """Producer metadata is accurate and malformed quad metadata is safe."""
    s_q, s_kv = 8, 34944
    torch.manual_seed(9487)
    source_topk = torch.arange(2048, device="cuda", dtype=torch.int32).repeat(s_q, 1)
    query_start = torch.tensor([0, s_q], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([4096 + s_q], device="cuda", dtype=torch.int32)
    gather_lens = torch.tensor([s_q + 127], device="cuda", dtype=torch.int32)
    indices, lengths, pairs, quads = flaggems_vllm.combine_topk_swa_indices(
        source_topk,
        query_start,
        seq_lens,
        gather_lens,
        128,
        4,
        2048,
        34944,
        2048,
        enable_hq4_sparse_prefill=True,
        return_pair_metadata=True,
        return_quad_metadata=True,
    )
    q = torch.randn((s_q, 4, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    kv = torch.randn((s_kv, 1, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    reference, _, _ = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
        s_q,
        s_kv,
        4,
        1,
        512,
        indices.shape[-1],
        q,
        kv,
        indices[:, None],
        512**-0.5,
        512,
        None,
        lengths,
    )

    def run(pair_descriptors, quad_descriptors):
        output, max_logits, lse = flaggems_vllm.flash_mla_sparse_fwd(
            q,
            kv,
            indices[:, None],
            512**-0.5,
            topk_length=lengths,
            enable_hq4_sparse_prefill=True,
            out=torch.empty_like(q),
            return_stats=False,
            pair_metadata=pair_descriptors,
            pair_window_size=128,
            quad_metadata=quad_descriptors,
            max_kv_length=1280,
        )
        assert max_logits is None and lse is None
        torch.testing.assert_close(
            output, reference, atol=8e-4, rtol=3.01 / 128, equal_nan=False
        )

    run(pairs, quads)
    bad_pairs = pairs.clone()
    bad_quads = quads.clone()
    bad_pairs[0] = 0
    bad_quads[0] = 1
    run(bad_pairs, bad_quads)


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(
    not _HAS_HQ4_CUDA,
    reason="requires an NVIDIA CUDA GPU with native BF16 support",
)
@pytest.mark.parametrize("s_q", [512, 1024, 2048, 4096])
def test_flash_mla_sparse_hq4_benchmark_active_shapes(s_q):
    """Check the metadata path at every remaining benchmark-active SQ."""
    torch.manual_seed(16000 + s_q)
    source_topk = torch.arange(2048, device="cuda", dtype=torch.int32).repeat(s_q, 1)
    query_start = torch.tensor([0, s_q], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([4096 + s_q], device="cuda", dtype=torch.int32)
    gather_lens = torch.tensor([s_q + 127], device="cuda", dtype=torch.int32)
    indices, lengths, pairs, quads = flaggems_vllm.combine_topk_swa_indices(
        source_topk,
        query_start,
        seq_lens,
        gather_lens,
        128,
        4,
        2048,
        34944,
        2048,
        enable_hq4_sparse_prefill=True,
        return_pair_metadata=True,
        return_quad_metadata=True,
    )
    q = torch.randn((s_q, 4, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    kv = torch.randn((34944, 1, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    output, max_logits, lse = flaggems_vllm.flash_mla_sparse_fwd(
        q,
        kv,
        indices[:, None],
        512**-0.5,
        topk_length=lengths,
        enable_hq4_sparse_prefill=True,
        out=torch.empty_like(q),
        return_stats=False,
        pair_metadata=pairs,
        pair_window_size=128,
        quad_metadata=quads,
        max_kv_length=1280,
    )
    assert max_logits is None and lse is None

    # A compact independent reference keeps the largest active shape from
    # materializing the full SQ x 2176 x 512 gathered-KV tensor. Beginning,
    # middle, and tail rows cover quad groups as well as top-k growth points.
    sample_rows = sorted(
        {
            0,
            1,
            2,
            3,
            s_q // 2 - 1,
            s_q // 2,
            s_q // 2 + 1,
            s_q - 4,
            s_q - 3,
            s_q - 2,
            s_q - 1,
        }
    )
    row_ids = torch.tensor(sample_rows, device="cuda", dtype=torch.int64)
    q_sample = q.index_select(0, row_ids)
    indices_sample = indices.index_select(0, row_ids)[:, None]
    lengths_sample = lengths.index_select(0, row_ids)
    reference, _, _ = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
        len(sample_rows),
        kv.shape[0],
        4,
        1,
        512,
        indices.shape[1],
        q_sample,
        kv,
        indices_sample,
        512**-0.5,
        512,
        None,
        lengths_sample,
    )
    torch.testing.assert_close(
        output.index_select(0, row_ids),
        reference,
        atol=8e-4,
        rtol=3.01 / 128,
        equal_nan=False,
    )


def _hq4_sink(kind):
    if kind == "none":
        return None
    if kind == "finite":
        values = [0.25, -0.75, 1.5, -2.0]
    else:
        values = [float("-inf"), float("inf"), 0.5, -0.5]
    return torch.tensor(values, device="cuda", dtype=torch.float32)


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("s_q", [1, 2, 3, 4, 11])
@pytest.mark.parametrize("sink_kind", ["none", "finite", "infinite"])
def test_flash_mla_sparse_hq4_single_boundaries(s_q, sink_kind):
    """Cover clamped lengths, invalid IDs, tails, and sink edge values."""
    s_kv, topk = 19, 32
    torch.manual_seed(3100 + s_q)
    q = torch.randn((s_q, 4, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    kv = torch.randn((s_kv, 1, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    indices = torch.arange(topk, dtype=torch.int32).remainder(s_kv).repeat(s_q, 1)
    indices[:, 1] = -1
    indices[:, 3] = s_kv
    indices[:, 5] = -2147483648
    indices[:, 7] = 2147483647
    indices = indices[:, None].cuda()
    length_values = [-3, 0, topk + 7, 7, topk]
    topk_length = torch.tensor(
        [length_values[row % len(length_values)] for row in range(s_q)],
        device="cuda",
        dtype=torch.int32,
    )
    sink = _hq4_sink(sink_kind)
    reference, ref_max, ref_lse = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
        s_q,
        s_kv,
        4,
        1,
        512,
        topk,
        q,
        kv,
        indices,
        512**-0.5,
        512,
        sink,
        topk_length,
    )
    output, max_logits, lse = flaggems_vllm.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        512**-0.5,
        attn_sink=sink,
        topk_length=topk_length,
        enable_hq4_sparse_prefill=True,
    )
    torch.testing.assert_close(
        output, reference, atol=8e-4, rtol=3.01 / 128, equal_nan=False
    )
    torch.testing.assert_close(
        max_logits, ref_max, atol=1e-6, rtol=2.01 / 65536, equal_nan=False
    )
    torch.testing.assert_close(
        lse, ref_lse, atol=1e-6, rtol=2.01 / 65536, equal_nan=False
    )


def _make_hq4_pair_mode_case():
    rows = [
        [0, 1, 2] + list(range(100, 105)),
        [0, 1, 2] + list(range(100, 106)),
        [3, 4, 5] + list(range(200, 328)),
        [3, 4, 5] + list(range(201, 329)),
        [6, 7] + list(range(330, 335)),
        [6, 7, 8] + list(range(330, 336)),
        [9, 10] + list(range(350, 478)),
        [9, 10, 11] + list(range(351, 479)),
        [12, 13, -1, 512],
        [15, 16, 17, 18, 19],
        [-1, 512, -2147483648, 2147483647],
    ]
    indices = torch.full((11, 1, 2176), -1, dtype=torch.int32)
    for row_id, values in enumerate(rows):
        indices[row_id, 0, : len(values)] = torch.tensor(values, dtype=torch.int32)
    lengths = torch.tensor([len(values) for values in rows], dtype=torch.int32)
    pairs = torch.tensor(
        [
            (3 << 3) | 1,
            (3 << 3) | 2,
            (2 << 3) | 3,
            (2 << 3) | 4,
            (3000 << 3) | 7,
            0,
        ],
        dtype=torch.int32,
    )
    return indices.cuda(), lengths.cuda(), pairs.cuda()


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("sink_kind", ["none", "infinite"])
def test_flash_mla_sparse_hq4_pair_modes_and_bad_descriptor(sink_kind):
    torch.manual_seed(3911)
    indices, lengths, pairs = _make_hq4_pair_mode_case()
    q = torch.randn((11, 4, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    kv = torch.randn((512, 1, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    sink = _hq4_sink(sink_kind)
    reference, _, _ = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
        11,
        512,
        4,
        1,
        512,
        2176,
        q,
        kv,
        indices,
        512**-0.5,
        512,
        sink,
        lengths,
    )
    output, max_logits, lse = flaggems_vllm.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        512**-0.5,
        attn_sink=sink,
        topk_length=lengths,
        enable_hq4_sparse_prefill=True,
        out=torch.empty_like(q),
        return_stats=False,
        pair_metadata=pairs,
        pair_window_size=128,
    )
    assert max_logits is None and lse is None
    torch.testing.assert_close(
        output, reference, atol=8e-4, rtol=3.01 / 128, equal_nan=False
    )


def _make_exact_hq4_metadata_case(s_q):
    source = torch.randperm(2048, device="cuda", dtype=torch.int32).repeat(s_q, 1)
    query_start = torch.tensor([0, s_q], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([4096 + s_q], device="cuda", dtype=torch.int32)
    gather_lens = torch.tensor([s_q + 127], device="cuda", dtype=torch.int32)
    metadata = flaggems_vllm.combine_topk_swa_indices(
        source,
        query_start,
        seq_lens,
        gather_lens,
        128,
        4,
        2048,
        34944,
        2048,
        enable_hq4_sparse_prefill=True,
        return_pair_metadata=True,
        return_quad_metadata=True,
    )
    return source, query_start, seq_lens, gather_lens, metadata


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("s_q", [1, 2, 3, 4, 11])
def test_flash_mla_sparse_hq4_pair_quad_tails(s_q):
    torch.manual_seed(7200 + s_q)
    _, _, _, _, (indices, lengths, pairs, quads) = _make_exact_hq4_metadata_case(s_q)
    q = torch.randn((s_q, 4, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    kv = torch.randn((4096, 1, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    sink = _hq4_sink("infinite")
    reference, _, _ = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
        s_q,
        4096,
        4,
        1,
        512,
        2176,
        q,
        kv,
        indices[:, None],
        512**-0.5,
        512,
        sink,
        lengths,
    )
    output, max_logits, lse = flaggems_vllm.flash_mla_sparse_fwd(
        q,
        kv,
        indices[:, None],
        512**-0.5,
        attn_sink=sink,
        topk_length=lengths,
        enable_hq4_sparse_prefill=True,
        out=torch.empty_like(q),
        return_stats=False,
        pair_metadata=pairs,
        pair_window_size=128,
        quad_metadata=quads,
        max_kv_length=1280,
    )
    assert max_logits is None and lse is None
    torch.testing.assert_close(
        output, reference, atol=8e-4, rtol=3.01 / 128, equal_nan=False
    )


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_mla_sparse_hq4_cuda_graph_regenerates_metadata_same_stream():
    """Replay observes changed source IDs/lengths because metadata is regenerated."""
    s_q = 8
    torch.manual_seed(8128)
    source, query_start, seq_lens, gather_lens, _ = _make_exact_hq4_metadata_case(s_q)
    q = torch.randn((s_q, 4, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    kv = torch.randn((4096, 1, 512), device="cuda", dtype=torch.bfloat16) * 0.1
    out = torch.empty_like(q)

    def run():
        metadata = flaggems_vllm.combine_topk_swa_indices(
            source,
            query_start,
            seq_lens,
            gather_lens,
            128,
            4,
            2048,
            34944,
            2048,
            enable_hq4_sparse_prefill=True,
            return_pair_metadata=True,
            return_quad_metadata=True,
        )
        indices, lengths, pairs, quads = metadata
        flaggems_vllm.flash_mla_sparse_fwd(
            q,
            kv,
            indices[:, None],
            512**-0.5,
            topk_length=lengths,
            enable_hq4_sparse_prefill=True,
            out=out,
            return_stats=False,
            pair_metadata=pairs,
            pair_window_size=128,
            quad_metadata=quads,
            max_kv_length=1280,
        )
        return metadata

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indices, lengths, pairs, quads = run()

    def check_output():
        reference, _, _ = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
            s_q,
            4096,
            4,
            1,
            512,
            2176,
            q,
            kv,
            indices[:, None],
            512**-0.5,
            512,
            None,
            lengths,
        )
        torch.testing.assert_close(
            out, reference, atol=8e-4, rtol=3.01 / 128, equal_nan=False
        )

    graph.replay()
    check_output()
    assert quads[0].item() == 1

    # These writes and replay are ordered on the current stream. Changing
    # source IDs breaks the first quad, while changing sequence metadata also
    # changes the generated indices and lengths.
    source[1, 0] = (source[1, 0] + 1) % 2048
    seq_lens.add_(4)
    gather_lens.add_(4)
    graph.replay()
    check_output()
    assert quads[0].item() == 0
