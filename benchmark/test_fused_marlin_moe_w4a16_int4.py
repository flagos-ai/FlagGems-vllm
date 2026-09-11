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

# vLLM imports (baseline). Optional: when vllm is not installed (e.g. in CI),
# the entire benchmark is skipped via the skipif marker below.
try:
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
        fused_marlin_moe as vllm_fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        quantize_weights,
    )
    from vllm.scalar_type import scalar_types

    VLLM_QUANT_TYPE = scalar_types.uint4b8
    HAS_VLLM_FUSED_MARLIN_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MARLIN_MOE = False

import flaggems_vllm

# FlagGems wrapper under test
from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT4B8
from flaggems_vllm.ops.fused_marlin_moe import fused_marlin_moe as gems_fused_marlin_moe

from . import base


def is_cuda_available():
    if flaggems_vllm.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()
ASCEND_AVAILABLE = flaggems_vllm.vendor_name == "ascend"

GROUP_SIZE = 128


def _wna16_quantize_per_expert(w_fp):
    """
    Per-expert GPTQ-style INT4 quantization for FlagGems wna16 kernel layout.

    Input  w_fp: (E, out_dim, in_dim), bf16/fp16
    Output w_q:   (E, out_dim, in_dim // 2), uint8 (two nibbles per byte)
           scales: (E, out_dim, in_dim // GROUP_SIZE), same dtype as w_fp
    """
    E, out_dim, in_dim = w_fp.shape
    assert in_dim % GROUP_SIZE == 0
    w_q = torch.empty(E, out_dim, in_dim // 2, device=w_fp.device, dtype=torch.uint8)
    scales = torch.empty(
        E, out_dim, in_dim // GROUP_SIZE, device=w_fp.device, dtype=w_fp.dtype
    )
    for e in range(E):
        _, q_e, sc_e, _ = quantize_weights(
            w_fp[e].T, VLLM_QUANT_TYPE, GROUP_SIZE, False, False
        )
        q_e = q_e.T.contiguous().to(torch.uint8)
        sc_e = sc_e.T
        w_q[e] = q_e[:, 1::2] * 16 + q_e[:, ::2]
        scales[e] = sc_e
    return w_q, scales


def _marlin_quantize_per_expert(w_fp):
    """
    Per-expert Marlin-layout INT4 quantization for vLLM's fused_marlin_moe.

    Input  w_fp: (E, out_dim, in_dim), bf16/fp16
    Output qweight: stacked (E, ...), int32 (Marlin packed layout)
           scales:  stacked (E, ...), same dtype as w_fp
    """
    qweight_l, scales_l = [], []
    E = w_fp.shape[0]
    for e in range(E):
        # marlin_quantize expects (in_dim, out_dim)
        _, qw, sc, _, _, _ = marlin_quantize(
            w_fp[e].T.contiguous(), VLLM_QUANT_TYPE, GROUP_SIZE, act_order=False
        )
        qweight_l.append(qw)
        scales_l.append(sc)
    qweight = torch.stack(qweight_l, dim=0).contiguous()
    scales = torch.stack(scales_l, dim=0).contiguous()
    return qweight, scales


def _ascend_weights(e, k, n, dtype):
    import torch_npu

    torch.manual_seed(7)
    result = []
    for ni, ki in [(2 * n, k), (k, n)]:
        w = torch.randint(0, 256, (e, ni, ki // 2), device="npu", dtype=torch.uint8)
        s = torch.rand((e, ni, ki // 128), device="npu", dtype=dtype) * 0.03
        native = []
        for ei in range(e):
            q = w[ei].to(torch.int32)
            q = torch.stack((q & 15, q >> 4), dim=-1).reshape(ni, ki) - 8
            native.append(torch_npu.npu_convert_weight_to_int4pack(q.T.contiguous()))
        wp = torch.stack(native)
        sn = s.transpose(1, 2).contiguous()
        result.append((w, s, wp, sn, torch.zeros_like(sn)))
    return result


def _ascend_baseline(x, ww, p, ids, *, vllm_dispatch=False):
    import torch_npu

    e = ww[0][0].shape[0]
    a, idx, counts, _ = torch_npu.npu_moe_init_routing_v2(
        x,
        ids.to(torch.int32),
        expert_num=e,
        active_num=x.shape[0] * ids.shape[1],
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
        active_expert_range=[0, e],
        quant_mode=-1,
    )
    if vllm_dispatch:
        counts = counts.to(torch.int64)
    for j in range(2):
        _, _, w, s, z = ww[j]
        a = torch_npu.npu_grouped_matmul(
            x=[a],
            weight=[w],
            antiquant_scale=[s],
            antiquant_offset=[z],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=counts,
            output_dtype=x.dtype,
        )[0]
        if j == 0:
            a = torch_npu.npu_swiglu(a)
    if vllm_dispatch:
        idx = torch.abs(idx)
        p = p.to(a.dtype)
    return torch_npu.npu_moe_token_unpermute(a, idx, probs=p)


def _ascend_vllm_baseline(x, ww, p, ids):
    """Single-device vLLM-Ascend W4A16 AllGather/GMM path, EP=1.

    Mirrors vllm-project/vllm-ascend c5055c8086d56ea1b2714b0f555ba67edd05e945:
    quantization/methods/wna16/w4a16.py and ops/fused_moe/token_dispatcher.py.
    Weight repacking is outside timing, as in process_weights_after_loading.
    """
    return _ascend_baseline(x, ww, p, ids, vllm_dispatch=True)


def _ascend_reference(x, ww, p, ids):
    x = x.cpu()
    p = p.cpu()
    ids = ids.cpu()
    y = torch.zeros_like(x, dtype=torch.float32)
    decoded = []
    for w, s, *_ in ww:
        w = w.cpu().to(torch.int32)
        s = s.cpu()
        q = (
            torch.stack((w & 15, w >> 4), dim=-1).reshape(
                *w.shape[:-1], w.shape[-1] * 2
            )
            - 8
        )
        decoded.append(
            (q.float() * s.float().repeat_interleave(128, dim=-1)).to(x.dtype).float()
        )
    for m in range(x.shape[0]):
        for t in range(ids.shape[1]):
            e = int(ids[m, t])
            h = (x[m].float() @ decoded[0][e].T).to(x.dtype).float()
            a, b = h.chunk(2)
            a = (torch.nn.functional.silu(a) * b).to(x.dtype).float()
            z = (a @ decoded[1][e].T).to(x.dtype).float()
            y[m] += z * p[m, t]
    return y.to(x.dtype)


def _ascend_gems_call(x, ww, p, ids):
    import flaggems_vllm
    from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT4B8

    return flaggems_vllm.fused_marlin_moe_w4a16_int4(
        x,
        ww[0][0],
        ww[1][0],
        None,
        None,
        ww[0][1],
        ww[1][1],
        p,
        ids,
        QUANT_TYPE_UINT4B8,
    )


def _ascend_inputs(m, e, k, t, seed=7):
    torch.manual_seed(seed + m)
    x = torch.randn((m, k), device="npu", dtype=torch.bfloat16) * 0.1
    ids = torch.rand((m, e), device="npu").topk(t, -1).indices.to(torch.int32)
    p = torch.softmax(torch.randn((m, t), device="npu"), -1)
    return x, p, ids


class FusedMarlinMoEW4A16INT4Benchmark(base.Benchmark):
    """
    Benchmark for fused_marlin_moe W4A16 INT4 (fused-dequant MoE GEMM).

    Compares FlagGems against vLLM Marlin on CUDA or the Ascend W4A16 chain.
    Both consume per-group-128 GPTQ uint4b8 weights (different packed layouts).
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # The three production MoE architectures from profile_fused_marlin_moe.py
        # over the decode token range (1 .. 256).
        self.shapes = [
            # Mixtral-8x7B
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (8, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (32, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            # DeepSeek-V3 (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (4, 256, 7168, 2048, 8),
            (8, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (32, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (128, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
            # Qwen3-5-397B-A17B
            (1, 512, 4096, 1024, 10),
            (4, 512, 4096, 1024, 10),
            (8, 512, 4096, 1024, 10),
            (16, 512, 4096, 1024, 10),
            (32, 512, 4096, 1024, 10),
            (64, 512, 4096, 1024, 10),
            (128, 512, 4096, 1024, 10),
            (256, 512, 4096, 1024, 10),
            # DeepSeek-V4-Flash
            (1, 256, 4096, 2048, 6),
            (4, 256, 4096, 2048, 6),
            (8, 256, 4096, 2048, 6),
            (16, 256, 4096, 2048, 6),
            (32, 256, 4096, 2048, 6),
            (64, 256, 4096, 2048, 6),
            (128, 256, 4096, 2048, 6),
            (256, 256, 4096, 2048, 6),
        ]

    def _get_ascend_input_iter(self, dtype):
        geometry = None
        ww = None
        for m, e, k, n, t in self.shapes:
            if geometry != (e, k, n):
                ww = _ascend_weights(e, k, n, dtype)
                geometry = (e, k, n)
            x, p, ids = _ascend_inputs(m, e, k, t)
            torch.testing.assert_close(
                _ascend_gems_call(x, ww, p, ids),
                _ascend_baseline(x, ww, p, ids),
                rtol=0.02,
                atol=0.02,
            )
            torch.testing.assert_close(
                _ascend_gems_call(x, ww, p, ids),
                _ascend_vllm_baseline(x, ww, p, ids),
                rtol=0.02,
                atol=0.02,
            )
            yield (x, ww, p, ids)

    def get_input_iter(self, cur_dtype):
        if ASCEND_AVAILABLE:
            yield from self._get_ascend_input_iter(cur_dtype)
            return
        for config in self.shapes:
            yield from self._gen(config, cur_dtype)

    def _gen(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flaggems_vllm.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

        # Original FP weights (kept only as source for both quantizers).
        w1_fp = (
            torch.randn(
                num_experts,
                intermediate_size * 2,
                hidden_size,
                device=device,
                dtype=dtype,
            )
            / 10.0
        )
        w2_fp = (
            torch.randn(
                num_experts,
                hidden_size,
                intermediate_size,
                device=device,
                dtype=dtype,
            )
            / 10.0
        )

        # FlagGems wna16 layout
        w1_q_wna16, w1_scale_wna16 = _wna16_quantize_per_expert(w1_fp)
        w2_q_wna16, w2_scale_wna16 = _wna16_quantize_per_expert(w2_fp)

        # vLLM Marlin layout
        w1_q_marlin, w1_scale_marlin = _marlin_quantize_per_expert(w1_fp)
        w2_q_marlin, w2_scale_marlin = _marlin_quantize_per_expert(w2_fp)

        del w1_fp, w2_fp
        torch.cuda.empty_cache()

        # Routing
        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        # vLLM requires fp32 topk_weights; FlagGems wrapper is dtype-agnostic.

        # Both ops get the same tuple; each picks what it needs.
        yield (
            hidden_states,
            w1_q_wna16,
            w2_q_wna16,
            w1_scale_wna16,
            w2_scale_wna16,
            w1_q_marlin,
            w2_q_marlin,
            w1_scale_marlin,
            w2_scale_marlin,
            topk_weights,
            topk_ids,
        )


def _vllm_baseline(
    hidden_states,
    w1_q_wna16,
    w2_q_wna16,
    w1_scale_wna16,
    w2_scale_wna16,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    topk_weights,
    topk_ids,
):
    """Baseline: vLLM's CUDA Marlin fused_marlin_moe."""
    return vllm_fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1_q_marlin,
        w2=w2_q_marlin,
        bias1=None,
        bias2=None,
        w1_scale=w1_scale_marlin,
        w2_scale=w2_scale_marlin,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=VLLM_QUANT_TYPE.id,
    )


def _gems_call(
    hidden_states,
    w1_q_wna16,
    w2_q_wna16,
    w1_scale_wna16,
    w2_scale_wna16,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    topk_weights,
    topk_ids,
):
    """FlagGems' Triton wna16 fused_marlin_moe (Phase 2)."""
    return gems_fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1_q_wna16,
        w2=w2_q_wna16,
        bias1=None,
        bias2=None,
        w1_scale=w1_scale_wna16,
        w2_scale=w2_scale_wna16,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=QUANT_TYPE_UINT4B8,
    )


@pytest.mark.fused_marlin_moe
@pytest.mark.skipif(
    not ASCEND_AVAILABLE and not HAS_VLLM_FUSED_MARLIN_MOE,
    reason="vllm not installed; CUDA baseline unavailable",
)
@pytest.mark.skipif(
    not (CUDA_AVAILABLE or ASCEND_AVAILABLE),
    reason="requires NVIDIA Hopper or Ascend",
)
def test_fused_marlin_moe_w4a16_int4():
    """
    Benchmark the active backend using its same-precision W4A16 baseline.
    CUDA uses vLLM Marlin; Ascend uses the torch_npu W4A16 primitive chain.
    """
    baseline_op, gems_op = _vllm_baseline, _gems_call
    if ASCEND_AVAILABLE:
        baseline_op, gems_op = _ascend_vllm_baseline, _ascend_gems_call
    bench = FusedMarlinMoEW4A16INT4Benchmark(
        op_name="fused_marlin_moe_w4a16_int4",
        torch_op=baseline_op,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(gems_op)
    bench.run()
