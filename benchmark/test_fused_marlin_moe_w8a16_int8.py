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
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        quantize_weights,
    )
    from vllm.scalar_type import scalar_types

    VLLM_QUANT_TYPE_INT8 = scalar_types.uint8b128
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
from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT8B128, fused_marlin_moe

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

GROUP_SIZE = 128

# -----------------------------------------------------------------------------
# Hygon path helpers. The Hygon backend consumes plain output-major uint8b128
# weights; the baseline is vLLM's native INT8 W8A16 Triton fused_experts on
# the shared channel-scale subset (each channel's scale is repeated across
# its quantization groups). Both implementations are checked against a
# PyTorch fp32 reference on every benchmarked shape before timing them.
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
    """Plain-layout INT8 weight bank on the shared channel-scale subset.

    Returns the Hygon codes/scales, the int8_w8a16 baseline form (signed int8
    weights + per-channel scales) and the BF16-decoded reference weights of
    the same quantized values.
    """
    torch.manual_seed(7)
    device = flaggems_vllm.device
    w1 = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size),
        device=device,
        dtype=torch.uint8,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=torch.uint8,
    )
    channel1 = (
        torch.rand((num_experts, 2 * intermediate_size, 1), device=device) * 0.001
        + 0.001
    ).to(dtype)
    channel2 = (
        torch.rand((num_experts, hidden_size, 1), device=device) * 0.001 + 0.001
    ).to(dtype)
    w1_scale = channel1.expand(-1, -1, hidden_size // GROUP_SIZE).contiguous()
    w2_scale = channel2.expand(-1, -1, intermediate_size // GROUP_SIZE).contiguous()
    w1_native = (w1.to(torch.int16) - 128).to(torch.int8)
    w2_native = (w2.to(torch.int16) - 128).to(torch.int8)
    w1_ref = torch.empty_like(w1_native, dtype=dtype)
    w2_ref = torch.empty_like(w2_native, dtype=dtype)
    for expert in range(num_experts):
        # One expert at a time to bound the fp32 scratch.
        w1_ref[expert] = (w1_native[expert].float() * channel1[expert].float()).to(
            dtype
        )
        w2_ref[expert] = (w2_native[expert].float() * channel2[expert].float()).to(
            dtype
        )
    return (
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_native,
        w2_native,
        channel1[..., 0].contiguous(),
        channel2[..., 0].contiguous(),
        w1_ref,
        w2_ref,
    )


def _hygon_verify(op_name, config, inputs, w1_ref, w2_ref):
    """Check both implementations against the fp32 reference before timing."""
    (hidden_states, _, _, _, _, _, _, _, _, topk_weights, topk_ids) = inputs
    expected = _hygon_reference(hidden_states, w1_ref, w2_ref, topk_weights, topk_ids)
    checks = (
        ("flaggems", _gems_call_int8(*inputs)),
        ("vllm", _vllm_baseline_int8(*inputs)),
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


def _wna16_quantize_per_expert_int8(w_fp):
    """
    Per-expert GPTQ-style INT8 quantization for FlagGems wna16 kernel layout.
    INT8 is one byte per element — no nibble packing — so K-dim stays in_dim.

    Input  w_fp: (E, out_dim, in_dim), bf16/fp16
    Output w_q:   (E, out_dim, in_dim), uint8
           scales: (E, out_dim, in_dim // GROUP_SIZE), same dtype as w_fp
    """
    E, out_dim, in_dim = w_fp.shape
    assert in_dim % GROUP_SIZE == 0
    w_q = torch.empty(E, out_dim, in_dim, device=w_fp.device, dtype=torch.uint8)
    scales = torch.empty(
        E, out_dim, in_dim // GROUP_SIZE, device=w_fp.device, dtype=w_fp.dtype
    )
    for e in range(E):
        _, q_e, sc_e, _ = quantize_weights(
            w_fp[e].T, VLLM_QUANT_TYPE_INT8, GROUP_SIZE, False, False
        )
        q_e = q_e.T.contiguous().to(torch.uint8)
        sc_e = sc_e.T
        w_q[e] = q_e
        scales[e] = sc_e
    return w_q, scales


def _marlin_repack_per_expert_int8(w_q, scales):
    """Repack the same UINT8 codes and scales for vLLM's CUDA Marlin kernel."""
    qweight_l, scales_l = [], []
    E, out_dim, in_dim = w_q.shape
    perm = torch.empty(0, dtype=torch.int, device=w_q.device)
    for e in range(E):
        qw = vllm_ops.gptq_marlin_repack(
            w_q[e].view(torch.int32).T.contiguous(), perm, in_dim, out_dim, 8
        )
        sc = marlin_permute_scales(
            scales[e].T.contiguous(), in_dim, out_dim, GROUP_SIZE
        )
        qweight_l.append(qw)
        scales_l.append(sc)
    qweight = torch.stack(qweight_l, dim=0).contiguous()
    scales = torch.stack(scales_l, dim=0).contiguous()
    return qweight, scales


class FusedMarlinMoEW8A16INT8Benchmark(base.Benchmark):
    """
    Benchmark for fused_marlin_moe W8A16 INT8 (fused-dequant MoE GEMM).
    Sister of FusedMarlinMoEW4A16INT4Benchmark (W4A16 INT4).
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (tokens, experts, hidden, intermediate, topk)
            for experts, hidden, intermediate, topk in (
                (8, 4096, 14336, 2),  # Mixtral-8x7B
                (256, 7168, 2048, 8),  # DeepSeek-V3 (TP=8)
                (512, 4096, 1024, 10),  # Qwen3.5-397B-A17B
                (256, 4096, 2048, 6),  # DeepSeek-V4-Flash
            )
            for tokens in (1, 16, 64, 256, 1024, 4096, 16384)
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
                (
                    w1,
                    w2,
                    w1_scale,
                    w2_scale,
                    w1_native,
                    w2_native,
                    channel1,
                    channel2,
                    w1_ref,
                    w2_ref,
                ) = [None] * 10
                torch.cuda.empty_cache()
                weights = _make_hygon_weights(
                    num_experts, hidden_size, intermediate_size, dtype
                )
                _hygon_ensure_vllm_expert_offset_int64([weights[8], weights[9]])
                geometry = next_geometry
            (
                w1,
                w2,
                w1_scale,
                w2_scale,
                w1_native,
                w2_native,
                channel1,
                channel2,
                w1_ref,
                w2_ref,
            ) = weights
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
                w1_native,
                w2_native,
                channel1,
                channel2,
                topk_weights,
                topk_ids,
            )
            _hygon_verify(self.op_name, config, inputs, w1_ref, w2_ref)
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
                num_experts,
                hidden_size,
                intermediate_size,
                device=device,
                dtype=dtype,
            )
            / 10.0
        )

        # FlagGems wna16 INT8 layout (unpacked)
        w1_q_wna16, w1_scale_wna16 = _wna16_quantize_per_expert_int8(w1_fp)
        w2_q_wna16, w2_scale_wna16 = _wna16_quantize_per_expert_int8(w2_fp)

        # Repacking preserves quantized values; it is outside benchmark timing.
        w1_q_marlin, w1_scale_marlin = _marlin_repack_per_expert_int8(
            w1_q_wna16, w1_scale_wna16
        )
        w2_q_marlin, w2_scale_marlin = _marlin_repack_per_expert_int8(
            w2_q_wna16, w2_scale_wna16
        )

        del w1_fp, w2_fp
        torch.cuda.empty_cache()

        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        inputs = (
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
        yield inputs


def _vllm_baseline_int8(
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
    """Baseline: vLLM's CUDA Marlin fused_marlin_moe (NVIDIA) or native INT8
    W8A16 fused_experts (Hygon)."""
    if flaggems_vllm.vendor_name == "hygon":
        return vllm_fused_experts(
            hidden_states,
            w1_q_marlin,
            w2_q_marlin,
            topk_weights,
            topk_ids,
            use_int8_w8a16=True,
            w1_scale=w1_scale_marlin,
            w2_scale=w2_scale_marlin,
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
        quant_type_id=VLLM_QUANT_TYPE_INT8.id,
    )


def _gems_call_int8(
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
    """FlagGems' Triton wna16 fused_marlin_moe W8A16."""
    gems_op = (
        flaggems_vllm.fused_marlin_moe
        if flaggems_vllm.vendor_name == "hygon"
        else fused_marlin_moe
    )
    return gems_op(
        bias1=None,
        bias2=None,
        quant_type_id=QUANT_TYPE_UINT8B128,
        hidden_states=hidden_states,
        w1=w1_q_wna16,
        w2=w2_q_wna16,
        w1_scale=w1_scale_wna16,
        w2_scale=w2_scale_wna16,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


@pytest.mark.fused_marlin_moe
@pytest.mark.skipif(
    not HAS_REQUIRED_VLLM, reason="required vLLM baseline is unavailable"
)
@pytest.mark.skipif(
    not SUPPORTED_DEVICE, reason="requires NVIDIA Hopper or a Hygon device"
)
def test_fused_marlin_moe_w8a16_int8():
    """
    Benchmark FlagGems fused_marlin_moe W8A16 (Triton wna16) vs vLLM
    fused_marlin_moe W8A16 (CUDA Marlin) on Hopper, or vs vLLM native INT8
    W8A16 fused_experts on Hygon. Both use the same UINT8 codes and
    per-group-128 scales, in native and Marlin-repacked layouts respectively.
    """
    bench = FusedMarlinMoEW8A16INT8Benchmark(
        op_name="fused_marlin_moe_w8a16_int8",
        torch_op=_vllm_baseline_int8,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_call_int8)
    bench.run()
