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

try:
    import vllm._custom_ops as vllm_ops
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
        fused_marlin_moe as vllm_fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        fp8_fused_exponent_bias_into_scales,
        pack_fp8_to_int32,
    )
    from vllm.scalar_type import scalar_types

    VLLM_QUANT_TYPE_FP8 = scalar_types.float8_e4m3fn
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
from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_FP8_E4M3, fused_marlin_moe

from . import base
from .test_fused_marlin_moe_w4a16_int4 import mthreads_input_iter, mthreads_shapes


def is_supported_device():
    if flaggems_vllm.vendor_name in ("hygon", "mthreads"):
        return True
    if flaggems_vllm.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return 90 <= sm_version_num < 100


SUPPORTED_DEVICE = is_supported_device()
HAS_REQUIRED_VLLM = (
    HAS_VLLM_FUSED_EXPERTS
    if flaggems_vllm.vendor_name in ("hygon", "mthreads")
    else HAS_VLLM_FUSED_MARLIN_MOE
)
GROUP_SIZE = 128

# -----------------------------------------------------------------------------
# Hygon path helpers. The Hygon backend consumes plain output-major E4M3FN
# weights; the baseline is vLLM's native BF16 Triton fused_experts running the
# same decoded weights. Both implementations are checked against a PyTorch
# fp32 reference on every benchmarked shape before timing them.
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
    """Plain-layout E4M3FN weight bank plus the same weights decoded for the
    native BF16 fused_experts baseline."""
    torch.manual_seed(7)
    device = flaggems_vllm.device

    def make_weight(out_dim, in_dim):
        raw = torch.randint(
            0, 254, (num_experts, out_dim, in_dim), device=device, dtype=torch.uint8
        )
        raw = torch.where(raw == 127, torch.zeros_like(raw), raw)
        scale = (
            torch.rand((num_experts, out_dim, in_dim // GROUP_SIZE), device=device)
            * 0.001
            + 0.001
        ).to(dtype)
        ref = torch.empty((num_experts, out_dim, in_dim), device=device, dtype=dtype)
        for expert in range(num_experts):
            values = raw[expert].view(torch.float8_e4m3fn).float()
            expanded = scale[expert].float().repeat_interleave(GROUP_SIZE, dim=-1)
            ref[expert] = (values * expanded).to(dtype)
        return raw, scale, ref

    w1, w1_scale, w1_bf16 = make_weight(2 * intermediate_size, hidden_size)
    w2, w2_scale, w2_bf16 = make_weight(hidden_size, intermediate_size)
    return (w1, w2, w1_scale, w2_scale, w1_bf16, w2_bf16)


def _hygon_verify(op_name, config, inputs):
    """Check both implementations against the fp32 reference before timing."""
    (hidden_states, w1_bf16, w2_bf16, _, _, _, _, _, _, topk_weights, topk_ids) = inputs
    expected = _hygon_reference(hidden_states, w1_bf16, w2_bf16, topk_weights, topk_ids)
    checks = (
        ("flaggems", _gems_call_fp8(*inputs)),
        ("vllm", _vllm_baseline_fp8(*inputs)),
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


def _quantize_per_expert_fp8(w_fp):
    """Quantize each expert to E4M3 with one scale per 128 weights."""
    num_experts, out_dim, in_dim = w_fp.shape
    assert in_dim % GROUP_SIZE == 0
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)
    num_groups = in_dim // GROUP_SIZE
    w_q = torch.empty(num_experts, out_dim, in_dim, device=w_fp.device, dtype=fp8_dtype)
    scales = torch.empty(
        num_experts,
        out_dim,
        num_groups,
        device=w_fp.device,
        dtype=w_fp.dtype,
    )
    for expert in range(num_experts):
        w_grouped = w_fp[expert].reshape(out_dim, num_groups, GROUP_SIZE).float()
        scales_fp = (w_grouped.abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(
            min=1e-8
        )
        q_expert = (
            (w_grouped / scales_fp).clamp(fp8_info.min, fp8_info.max).to(fp8_dtype)
        )
        w_q[expert] = q_expert.reshape(out_dim, in_dim)
        scales[expert] = scales_fp.squeeze(-1).to(w_fp.dtype)
    return w_q, scales.contiguous()


def _marlin_repack_per_expert_fp8(w_q, scales, dtype):
    """Convert E4M3 weights and per-group scales to vLLM Marlin layout."""
    num_experts, out_dim, in_dim = w_q.shape
    perm = torch.empty(0, dtype=torch.int, device=w_q.device)
    qweight_list = []
    scale_list = []
    for expert in range(num_experts):
        qweight = pack_fp8_to_int32(w_q[expert], size_k_first=False)
        qweight = vllm_ops.gptq_marlin_repack(
            b_q_weight=qweight.T.contiguous(),
            perm=perm,
            size_k=in_dim,
            size_n=out_dim,
            num_bits=8,
        )
        marlin_scales = marlin_permute_scales(
            s=scales[expert].T.to(dtype).contiguous(),
            size_k=in_dim,
            size_n=out_dim,
            group_size=GROUP_SIZE,
        )
        marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)
        qweight_list.append(qweight)
        scale_list.append(marlin_scales)
    return (
        torch.stack(qweight_list, dim=0).contiguous(),
        torch.stack(scale_list, dim=0).contiguous(),
    )


class FusedMarlinMoEW8A16FP8Benchmark(base.Benchmark):
    """Compare the same E4M3 codes/scales in native and Marlin-repacked layouts."""

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self._weight_cache = {}

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

        if flaggems_vllm.vendor_name == "mthreads":
            self.shapes = mthreads_shapes(self.shapes, "fp8")

    def get_input_iter(self, cur_dtype):
        if flaggems_vllm.vendor_name == "mthreads":
            yield from mthreads_input_iter(
                self, cur_dtype, "fp8", _gems_call_fp8, _vllm_baseline_fp8
            )
            return
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
            # This file's tuple holds the baseline weights first and the
            # Hygon (gems) weights second, matching the NVIDIA layout below.
            inputs = (
                hidden_states,
                w1_bf16,
                w2_bf16,
                None,
                None,
                w1,
                w2,
                w1_scale,
                w2_scale,
                topk_weights,
                topk_ids,
            )
            _hygon_verify(self.op_name, config, inputs)
            yield inputs

    def _get_quantized_weights(
        self, dtype, device, num_experts, hidden_size, intermediate_size
    ):
        cache_key = (dtype, str(device), num_experts, hidden_size, intermediate_size)
        cached = self._weight_cache.get(cache_key)
        if cached is not None:
            return cached
        self._weight_cache.clear()

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
        w1_q_fp8, w1_scale_fp8 = _quantize_per_expert_fp8(w1_fp)
        w2_q_fp8, w2_scale_fp8 = _quantize_per_expert_fp8(w2_fp)
        w1_q_marlin, w1_scale_marlin = _marlin_repack_per_expert_fp8(
            w1_q_fp8, w1_scale_fp8, dtype
        )
        w2_q_marlin, w2_scale_marlin = _marlin_repack_per_expert_fp8(
            w2_q_fp8, w2_scale_fp8, dtype
        )
        cached = (
            w1_q_marlin,
            w1_scale_marlin,
            w2_q_marlin,
            w2_scale_marlin,
            w1_q_fp8,
            w1_scale_fp8,
            w2_q_fp8,
            w2_scale_fp8,
        )
        self._weight_cache[cache_key] = cached
        del w1_fp, w2_fp
        torch.cuda.empty_cache()
        return cached

    def _gen(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flaggems_vllm.device
        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        (
            w1_q_marlin,
            w1_scale_marlin,
            w2_q_marlin,
            w2_scale_marlin,
            w1_q_fp8,
            w1_scale_fp8,
            w2_q_fp8,
            w2_scale_fp8,
        ) = self._get_quantized_weights(
            dtype, device, num_experts, hidden_size, intermediate_size
        )

        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        inputs = (
            hidden_states,
            w1_q_marlin,
            w2_q_marlin,
            w1_scale_marlin,
            w2_scale_marlin,
            w1_q_fp8,
            w2_q_fp8,
            w1_scale_fp8,
            w2_scale_fp8,
            topk_weights,
            topk_ids,
        )
        yield inputs


def _vllm_baseline_fp8(
    hidden_states,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    w1_q_fp8,
    w2_q_fp8,
    w1_scale_fp8,
    w2_scale_fp8,
    topk_weights,
    topk_ids,
):
    """Baseline: vLLM's CUDA Marlin fused_marlin_moe (NVIDIA) or native BF16
    fused_experts (Hygon)."""
    if flaggems_vllm.vendor_name in ("hygon", "mthreads"):
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
        quant_type_id=VLLM_QUANT_TYPE_FP8.id,
    )


def _gems_call_fp8(
    hidden_states,
    w1_q_marlin,
    w2_q_marlin,
    w1_scale_marlin,
    w2_scale_marlin,
    w1_q_fp8,
    w2_q_fp8,
    w1_scale_fp8,
    w2_scale_fp8,
    topk_weights,
    topk_ids,
):
    gems_op = (
        flaggems_vllm.fused_marlin_moe
        if flaggems_vllm.vendor_name in ("hygon", "mthreads")
        else fused_marlin_moe
    )
    return gems_op(
        bias1=None,
        bias2=None,
        quant_type_id=QUANT_TYPE_FP8_E4M3,
        hidden_states=hidden_states,
        w1=w1_q_fp8,
        w2=w2_q_fp8,
        w1_scale=w1_scale_fp8,
        w2_scale=w2_scale_fp8,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


@pytest.mark.fused_marlin_moe_w8a16_fp8
@pytest.mark.skipif(
    not HAS_REQUIRED_VLLM, reason="required vLLM baseline is unavailable"
)
@pytest.mark.skipif(
    not SUPPORTED_DEVICE, reason="requires NVIDIA Hopper, Hygon, or Moore Threads"
)
def test_fused_marlin_moe_w8a16_fp8():
    """Compare identical E4M3 weights and per-group-128 scales; on Hygon the
    baseline is vLLM's native BF16 fused_experts over the decoded weights."""
    bench = FusedMarlinMoEW8A16FP8Benchmark(
        op_name="fused_marlin_moe_w8a16_fp8",
        torch_op=_vllm_baseline_fp8,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_call_fp8)
    bench.run()
