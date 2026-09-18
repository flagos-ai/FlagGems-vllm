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
    import vllm._custom_ops as vllm_ops
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
        fused_marlin_moe as vllm_fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        mxfp4_marlin_process_scales,
    )
    from vllm.scalar_type import scalar_types

    VLLM_QUANT_TYPE_FP4 = scalar_types.float4_e2m1f
    HAS_VLLM_FUSED_MARLIN_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MARLIN_MOE = False

# vLLM 0.6.2 on Hygon registers no Marlin MoE ops (torch.ops._moe_C is empty),
# so the Hygon path compares against the native Triton fused_experts kernel.
try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts as vllm_fused_experts,
    )

    HAS_VLLM_FUSED_EXPERTS = True
except ImportError:
    HAS_VLLM_FUSED_EXPERTS = False

import flaggems_vllm

# FlagGems wrapper under test
from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_FP4_E2M1
from flaggems_vllm.ops.fused_marlin_moe import fused_marlin_moe as gems_fused_marlin_moe

from . import base


def is_supported_device():
    if flaggems_vllm.vendor_name == "hygon":
        return True
    if flaggems_vllm.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


SUPPORTED_DEVICE = is_supported_device()
HAS_REQUIRED_VLLM = (
    HAS_VLLM_FUSED_EXPERTS
    if flaggems_vllm.vendor_name == "hygon"
    else HAS_VLLM_FUSED_MARLIN_MOE
)

# =============================================================================
# MXFP4 (FP4 E2M1 + per-32 E8M0) benchmark: FlagGems Triton vs vLLM Marlin.
# Both consume the same FP4 weights + E8M0 scale in their respective layouts.
# =============================================================================
MXFP4_GROUP_SIZE = 32
_E2M1_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_MID = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_E2M1_MAX = 6.0

# -----------------------------------------------------------------------------
# Hygon path helpers. The Hygon backend consumes plain output-major MXFP4
# codes with uint8 E8M0 scales; the baseline is vLLM's native BF16 Triton
# fused_experts running the same decoded weights. Both implementations are
# checked against a PyTorch fp32 reference on every benchmarked shape.
# -----------------------------------------------------------------------------


def _hygon_reference(hidden_states, w1_ref, w2_ref, topk_weights, topk_ids):
    """PyTorch SwiGLU MoE ground truth, rounding each GEMM stage into the
    activation dtype like the Hygon kernel."""
    dtype = hidden_states.dtype
    accumulation_dtype = torch.float64 if dtype == torch.float16 else torch.float32
    m, k = hidden_states.shape
    topk = topk_ids.shape[1]
    flat_ids = topk_ids.flatten()
    flat_weights = topk_weights.flatten()
    result = torch.zeros((m * topk, k), dtype=dtype, device=hidden_states.device)
    for expert in range(w1_ref.shape[0]):
        routes = torch.where(flat_ids == expert)[0]
        if routes.numel() == 0:
            continue
        x = hidden_states[routes // topk].to(accumulation_dtype)
        gate_up = x @ w1_ref[expert].to(accumulation_dtype).T
        gate, up = gate_up.to(dtype).float().chunk(2, -1)
        act = (torch.nn.functional.silu(gate) * up).to(dtype)
        out = (
            act.to(accumulation_dtype) @ w2_ref[expert].to(accumulation_dtype).T
        ).float()
        result[routes] = (out * flat_weights[routes, None].float()).to(dtype)
    return result.view(m, topk, k).float().sum(1).to(dtype)


def _hygon_relative_errors(actual, expected):
    delta = actual.float() - expected.float()
    rms = (
        delta.square().mean().sqrt()
        / expected.float().square().mean().sqrt().clamp_min(1e-12)
    )
    peak = delta.abs().max() / expected.float().abs().max().clamp_min(1e-12)
    return rms.item(), peak.item()


def _hygon_decode_mxfp4(w_q, scales, dtype):
    """Decode plain-layout MXFP4 codes with uint8 E8M0 scales to activation
    dtype, one expert at a time to bound scratch memory."""
    num_experts, out_dim, packed_k = w_q.shape
    in_dim = packed_k * 2
    lut = torch.tensor(_E2M1_POS, device=w_q.device)
    ref = torch.empty((num_experts, out_dim, in_dim), device=w_q.device, dtype=dtype)
    for expert in range(num_experts):
        packed = w_q[expert].to(torch.int32)
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(out_dim, in_dim)
        values = lut[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)
        exponent = scales[expert].float().repeat_interleave(MXFP4_GROUP_SIZE, dim=-1)
        ref[expert] = (values * torch.exp2(exponent - 127)).to(dtype)
    return ref


_HYGON_ADDRESS_PATCH = None


def _hygon_ensure_vllm_expert_offset_int64(weights):
    """Process-local 64-bit addressing fix for the installed vLLM 0.6.2 kernel.

    The Triton expert offset in vllm 0.6.2's fused_moe kernel is int32; on
    gfx936 its generated buffer load also has a 2 GiB resource range, so large
    expert banks overflow. Patch the JIT source of this benchmark process's
    kernel only; the installed package is not modified.
    """
    global _HYGON_ADDRESS_PATCH
    if _HYGON_ADDRESS_PATCH is not None and _HYGON_ADDRESS_PATCH["address_patch"]:
        return _HYGON_ADDRESS_PATCH
    required = any(
        sum((size - 1) * stride for size, stride in zip(w.shape, w.stride()))
        * w.element_size()
        + w.element_size()
        >= 2**31 - 2
        for w in weights
    )
    if not required:
        print("HYGON_VLLM_ADDRESS", dict(address_patch=False), flush=True)
        return dict(address_patch=False)
    import hashlib
    import importlib
    import importlib.util
    import sys
    import tempfile
    from pathlib import Path

    module = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
    original = module.fused_moe_kernel.src
    before = "off_experts = tl.load(expert_ids_ptr + pid_m)"
    after = before + ".to(tl.int64)"
    if original.count(before) != 1 or after in original:
        raise RuntimeError(
            "Unexpected vLLM source; review the address fix before benchmarking"
        )
    patched = original.replace(before, after, 1)
    patch_dir = tempfile.TemporaryDirectory(prefix="hygon-vllm-address-")
    path = Path(patch_dir.name) / "reference.py"
    path.write_text(
        "import triton\nimport triton.language as tl\n\n@triton.jit\n" + patched
    )
    name = "_hygon_vllm_address_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    copied = importlib.util.module_from_spec(spec)
    sys.modules[name] = copied
    spec.loader.exec_module(copied)
    module.fused_moe_kernel = copied.fused_moe_kernel
    # Keep the temporary directory alive for the process lifetime.
    _hygon_ensure_vllm_expert_offset_int64._patch_dir = patch_dir
    _HYGON_ADDRESS_PATCH = dict(
        address_patch=True,
        original_kernel_sha256=hashlib.sha256(original.encode()).hexdigest(),
        patched_kernel_sha256=hashlib.sha256(patched.encode()).hexdigest(),
    )
    print("HYGON_VLLM_ADDRESS", _HYGON_ADDRESS_PATCH, flush=True)
    return _HYGON_ADDRESS_PATCH


def _make_hygon_weights(num_experts, hidden_size, intermediate_size, dtype):
    """Plain-layout MXFP4 weight bank plus the same weights decoded for the
    native BF16 fused_experts baseline."""
    torch.manual_seed(7)
    device = flaggems_vllm.device
    w1 = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w1_scale = torch.randint(
        121,
        126,
        (num_experts, 2 * intermediate_size, hidden_size // MXFP4_GROUP_SIZE),
        device=device,
        dtype=torch.uint8,
    )
    w2_scale = torch.randint(
        121,
        126,
        (num_experts, hidden_size, intermediate_size // MXFP4_GROUP_SIZE),
        device=device,
        dtype=torch.uint8,
    )
    w1_bf16 = _hygon_decode_mxfp4(w1, w1_scale, dtype)
    w2_bf16 = _hygon_decode_mxfp4(w2, w2_scale, dtype)
    return (w1, w2, w1_scale, w2_scale, w1_bf16, w2_bf16)


def _hygon_verify(op_name, config, inputs):
    """Check both implementations against the fp32 reference before timing."""
    (hidden_states, _, _, _, _, w1_bf16, w2_bf16, _, _, topk_weights, topk_ids) = inputs
    expected = _hygon_reference(hidden_states, w1_bf16, w2_bf16, topk_weights, topk_ids)
    checks = (
        ("flaggems", _gems_call_mxfp4(*inputs)),
        ("vllm", _vllm_baseline_mxfp4(*inputs)),
    )
    errors = []
    for name, output in checks:
        rms, peak = _hygon_relative_errors(output, expected)
        assert rms < 0.01 and peak < 0.02, (
            f"{op_name} {config}: {name} mismatch against fp32 reference "
            f"(relative_rms={rms}, relative_peak={peak})"
        )
        errors.append((name, round(rms, 6), round(peak, 6)))
    print(f"HYGON_VERIFY {config} {errors}", flush=True)


def _quantize_mxfp4_2d(w_2d, group_size):
    """Round-to-nearest MXFP4. Returns nibbles (uint8 [0,15]) + E8M0 scale."""
    out_dim, in_dim = w_2d.shape
    ng = in_dim // group_size
    device = w_2d.device
    wg = w_2d.reshape(out_dim, ng, group_size).to(torch.float32)
    amax = wg.abs().amax(dim=-1, keepdim=True)
    exp = torch.ceil(torch.log2((amax / _E2M1_MAX).clamp(min=1e-30))).clamp(-127, 127)
    scale = torch.exp2(exp)
    e8m0_byte = (exp + 127.0).to(torch.uint8)
    wn = wg / scale
    sign = wn < 0
    a = wn.abs().clamp(max=_E2M1_MAX)
    mag = torch.bucketize(a, torch.tensor(_E2M1_MID, device=device))
    nibbles = (sign.to(torch.uint8) * 8 + mag.to(torch.uint8)).reshape(out_dim, in_dim)
    scale_e8m0 = e8m0_byte.squeeze(-1).view(torch.float8_e8m0fnu)
    return nibbles, scale_e8m0


def _mxfp4_quantize_per_expert(w_fp):
    """FlagGems MXFP4 layout: packed uint8 (E, out, in//2) + E8M0 scale."""
    E, out_dim, in_dim = w_fp.shape
    w_q = torch.empty(E, out_dim, in_dim // 2, device=w_fp.device, dtype=torch.uint8)
    scales = torch.empty(
        E,
        out_dim,
        in_dim // MXFP4_GROUP_SIZE,
        device=w_fp.device,
        dtype=torch.float8_e8m0fnu,
    )
    for e in range(E):
        nib, sc = _quantize_mxfp4_2d(w_fp[e], MXFP4_GROUP_SIZE)
        w_q[e] = nib[:, 1::2] * 16 + nib[:, ::2]
        scales[e] = sc
    return w_q, scales


def _marlin_mxfp4_quantize_per_expert(w_fp, dtype):
    """vLLM Marlin MXFP4 layout from the same nibbles/scale (numerically aligned)."""
    E, out_dim, in_dim = w_fp.shape
    qweight_l, scales_l = [], []
    for e in range(E):
        nib, sc = _quantize_mxfp4_2d(w_fp[e], MXFP4_GROUP_SIZE)
        packed = (nib[:, 1::2] * 16 + nib[:, ::2]).to(torch.uint8)
        perm = torch.empty(0, dtype=torch.int, device=w_fp.device)
        qw = vllm_ops.gptq_marlin_repack(
            packed.view(torch.int32).T.contiguous(), perm, in_dim, out_dim, 4, False
        )
        ms = marlin_permute_scales(
            sc.T.to(dtype), in_dim, out_dim, MXFP4_GROUP_SIZE, False
        )
        ms = mxfp4_marlin_process_scales(ms, input_dtype=None).to(torch.float8_e8m0fnu)
        qweight_l.append(qw)
        scales_l.append(ms)
    return torch.stack(qweight_l, 0).contiguous(), torch.stack(scales_l, 0).contiguous()


class FusedMarlinMoEW4A16MXFP4Benchmark(base.Benchmark):
    """MXFP4 (FP4 E2M1 + E8M0) MoE: FlagGems Triton vs vLLM Marlin."""

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            # Mixtral-8x7B
            (1, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            # DeepSeek-V3 (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
            # Qwen3-5-397B-A17B
            (1, 512, 4096, 1024, 10),
            (16, 512, 4096, 1024, 10),
            (64, 512, 4096, 1024, 10),
            (256, 512, 4096, 1024, 10),
            # DeepSeek-V4-Flash
            (1, 256, 4096, 2048, 6),
            (16, 256, 4096, 2048, 6),
            (64, 256, 4096, 2048, 6),
            (256, 256, 4096, 2048, 6),
        ]

    def get_input_iter(self, cur_dtype):
        if flaggems_vllm.vendor_name == "hygon":
            yield from self._get_hygon_input_iter(cur_dtype)
            return
        for config in self.shapes:
            yield from self._gen(config, cur_dtype)

    def _get_hygon_input_iter(self, dtype):
        geometry = None
        weights = None
        for config in self.shapes:
            num_tokens, num_experts, hidden_size, intermediate_size, top_k = config
            if num_tokens * top_k > 16384:
                # The Hygon kernel allocates route workspaces of O(E * routes).
                print(
                    f"Skipping {config}: Hygon fused Marlin MoE supports at "
                    "most 16384 routes"
                )
                continue
            next_geometry = (num_experts, hidden_size, intermediate_size)
            if geometry != next_geometry:
                # Drop the previous geometry's tensors before allocating the new
                # bank; generator locals keep them alive otherwise.
                weights = inputs = None
                w1, w2, w1_scale, w2_scale, w1_bf16, w2_bf16 = [None] * 6
                torch.cuda.empty_cache()
                weights = _make_hygon_weights(
                    num_experts, hidden_size, intermediate_size, dtype
                )
                _hygon_ensure_vllm_expert_offset_int64([weights[4], weights[5]])
                geometry = next_geometry
            w1, w2, w1_scale, w2_scale, w1_bf16, w2_bf16 = weights
            torch.manual_seed(7 + num_tokens)
            hidden_states = (
                torch.randn((num_tokens, hidden_size), device=flaggems_vllm.device)
                * 0.1
            ).to(dtype)
            topk_ids = (
                torch.rand((num_tokens, num_experts), device=flaggems_vllm.device)
                .topk(top_k, dim=-1)
                .indices
            )
            topk_weights = torch.softmax(
                torch.randn((num_tokens, top_k), device=flaggems_vllm.device),
                dim=-1,
            )
            inputs = (
                hidden_states,
                w1,
                w2,
                w1_scale,
                w2_scale,
                w1_bf16,
                w2_bf16,
                None,
                None,
                topk_weights,
                topk_ids,
            )
            _hygon_verify(self.op_name, config, inputs)
            yield inputs

    def _gen(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flaggems_vllm.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
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
                num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
            )
            / 10.0
        )

        w1_q_fg, w1_scale_fg = _mxfp4_quantize_per_expert(w1_fp)
        w2_q_fg, w2_scale_fg = _mxfp4_quantize_per_expert(w2_fp)
        w1_q_marlin, w1_scale_marlin = _marlin_mxfp4_quantize_per_expert(w1_fp, dtype)
        w2_q_marlin, w2_scale_marlin = _marlin_mxfp4_quantize_per_expert(w2_fp, dtype)

        del w1_fp, w2_fp
        torch.cuda.empty_cache()

        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        yield (
            hidden_states,
            w1_q_fg,
            w2_q_fg,
            w1_scale_fg,
            w2_scale_fg,
            w1_q_marlin,
            w2_q_marlin,
            w1_scale_marlin,
            w2_scale_marlin,
            topk_weights,
            topk_ids,
        )


def _vllm_baseline_mxfp4(
    hidden_states,
    w1_q_fg,
    w2_q_fg,
    w1_scale_fg,
    w2_scale_fg,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    topk_weights,
    topk_ids,
):
    """Baseline: vLLM's CUDA Marlin fused_marlin_moe (NVIDIA) or native BF16
    fused_experts (Hygon)."""
    if flaggems_vllm.vendor_name == "hygon":
        return vllm_fused_experts(
            hidden_states,
            w1_q_marlin,
            w2_q_marlin,
            topk_weights,
            topk_ids,
        )
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
        quant_type_id=VLLM_QUANT_TYPE_FP4.id,
    )


def _gems_call_mxfp4(
    hidden_states,
    w1_q_fg,
    w2_q_fg,
    w1_scale_fg,
    w2_scale_fg,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    topk_weights,
    topk_ids,
):
    """FlagGems' Triton MXFP4 fused_marlin_moe."""
    gems_op = (
        flaggems_vllm.fused_marlin_moe
        if flaggems_vllm.vendor_name == "hygon"
        else gems_fused_marlin_moe
    )
    return gems_op(
        hidden_states=hidden_states,
        w1=w1_q_fg,
        w2=w2_q_fg,
        bias1=None,
        bias2=None,
        w1_scale=w1_scale_fg,
        w2_scale=w2_scale_fg,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=QUANT_TYPE_FP4_E2M1,
        group_size=MXFP4_GROUP_SIZE,
    )


@pytest.mark.fused_marlin_moe_w4a16_mxfp4
@pytest.mark.skipif(
    not HAS_REQUIRED_VLLM, reason="required vLLM baseline is unavailable"
)
@pytest.mark.skipif(
    not SUPPORTED_DEVICE, reason="requires NVIDIA Hopper or a Hygon device"
)
def test_fused_marlin_moe_w4a16_mxfp4():
    """
    Benchmark FlagGems MXFP4 fused_marlin_moe (Triton) vs vLLM MXFP4
    fused_marlin_moe (CUDA Marlin) on Hopper, or vs vLLM native BF16
    fused_experts on Hygon. Both run FP4 E2M1 + per-32 E8M0 W4A16 GEMM.
    """
    bench = FusedMarlinMoEW4A16MXFP4Benchmark(
        op_name="fused_marlin_moe_w4a16_mxfp4",
        torch_op=_vllm_baseline_mxfp4,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_call_mxfp4)
    bench.run()
