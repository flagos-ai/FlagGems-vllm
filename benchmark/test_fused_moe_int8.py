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

from . import base

# Per-vendor native compute precision: (fp8, int8), per vendor documentation.
# Vendors left unlisted are skipped rather than assumed to work.
_OFFICIAL_PRECISION = {
    "nvidia": (True, True),
    "mthreads": (True, True),
    "hygon": (False, True),
    "metax": (False, True),
    "thead": (False, True),
    "ascend": (False, True),
}
_SUPPORTS_FP8, _SUPPORTS_INT8 = _OFFICIAL_PRECISION.get(
    flaggems_vllm.vendor_name, (False, False)
)

# Vendor has native INT8, but the kernels this benchmark reaches do not run
# there yet (a kernel-side gap, not a hardware limitation).
_INT8_UNSUPPORTED = {
    "mthreads",  # MUSA Triton has no tl.extra.cuda namespace
}

VENDOR = flaggems_vllm.vendor_name

# Vendors whose vLLM build ships the Triton fused_experts_impl.
VLLM_NATIVE_VENDORS = {"nvidia", "metax"}

if VENDOR in VLLM_NATIVE_VENDORS:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


class FusedMoEINT8Benchmark(base.Benchmark):
    """
    Benchmark for fused_experts_impl with INT8 W8A8 quantization.

    Weights are pre-quantized to INT8 with per-channel (per output-dim) scales.
    Activations are dynamically quantized per-token inside the kernel.
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def init_user_config(self):
        super().init_user_config()
        if self.torch_op is None:
            self.to_bench_metrics = [
                metric
                for metric in self.to_bench_metrics
                if metric not in ("latency_base", "speedup")
            ]

    def set_shapes(self, shape_file_path=None):
        # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
        self.shapes = [
            # Mixtral-like shapes
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            (512, 8, 4096, 14336, 2),
            # MiniMax per-channel W8A8 shapes (TP=8 shard).
            (64, 128, 6144, 384, 4),
            (1024, 128, 6144, 384, 4),
            (8192, 128, 6144, 384, 4),
            # DeepSeek-V3-like shapes (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (4, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (128, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._int8_input_fn(config, cur_dtype)

    def _int8_input_fn(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flaggems_vllm.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

        # Generate INT8 weights one expert at a time to avoid OOM on large E.
        w1_int8 = torch.empty(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=torch.int8,
        )
        w2_int8 = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.int8,
        )
        for e in range(num_experts):
            w1_int8[e] = to_int8(
                torch.randn(
                    intermediate_size * 2,
                    hidden_size,
                    device=device,
                    dtype=torch.float16,
                )
                * 50
            )
            w2_int8[e] = to_int8(
                torch.randn(
                    hidden_size,
                    intermediate_size,
                    device=device,
                    dtype=torch.float16,
                )
                * 50
            )

        # Synthetic per-channel scales [E, output_dim]
        w1_scale = (
            torch.rand(
                num_experts,
                intermediate_size * 2,
                device=device,
                dtype=torch.float32,
            )
            * 0.01
            + 0.001
        )
        w2_scale = (
            torch.rand(num_experts, hidden_size, device=device, dtype=torch.float32)
            * 0.01
            + 0.001
        )

        # Routing
        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        yield (
            hidden_states,
            w1_int8,
            w2_int8,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
        )


def _vllm_fused_moe_int8_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Wrapper to call vllm fused_experts_impl with INT8."""
    return vllm_fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        use_int8_w8a8=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


def _gems_fused_moe_int8_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Wrapper to call FlagGems fused_experts_impl with INT8."""
    return flaggems_vllm.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        use_int8_w8a8=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


def _get_baseline_op():
    if VENDOR in VLLM_NATIVE_VENDORS:
        return _vllm_fused_moe_int8_wrapper
    return None


@pytest.mark.fused_experts_impl
@pytest.mark.skipif(
    not _SUPPORTS_INT8 or flaggems_vllm.vendor_name in _INT8_UNSUPPORTED,
    reason="no native INT8 support / known unsupported on this vendor "
    "(per vendor documentation)",
)
def test_fused_experts_impl_int8():
    """
    Benchmark FlagGems vs vLLM fused_experts_impl with INT8 W8A8 quantization.
    """
    bench = FusedMoEINT8Benchmark(
        op_name="fused_experts_impl",
        torch_op=_get_baseline_op(),
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_moe_int8_wrapper)
    bench.run()
