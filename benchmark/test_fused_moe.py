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

"""Benchmark FlagGems fused_experts_impl against each platform production MoE entry."""

import pytest
import torch

import flaggems_vllm

from . import base

VENDOR = flaggems_vllm.vendor_name

# Vendors whose vLLM build ships the Triton fused_experts_impl.
VLLM_NATIVE_VENDORS = {"nvidia", "metax"}

if VENDOR in VLLM_NATIVE_VENDORS:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )

# (num_tokens, num_experts, hidden_size, intermediate_size, topk)
_SHAPES = [
    # Mixtral-like
    (1, 8, 4096, 14336, 2),
    (4, 8, 4096, 14336, 2),
    (16, 8, 4096, 14336, 2),
    (64, 8, 4096, 14336, 2),
    (128, 8, 4096, 14336, 2),
    (256, 8, 4096, 14336, 2),
    (512, 8, 4096, 14336, 2),
    # DeepSeek-V3-like (TP=8 shard)
    (1, 256, 7168, 2048, 8),
    (4, 256, 7168, 2048, 8),
    (16, 256, 7168, 2048, 8),
    (64, 256, 7168, 2048, 8),
    (128, 256, 7168, 2048, 8),
    (256, 256, 7168, 2048, 8),
    # Qwen3.6-35B-A3B, real prefill chunks from the p1024/p4096/p32768 runs.
    (1036, 256, 2048, 128, 8),
    (4435, 256, 2048, 128, 8),
    (4108, 256, 2048, 128, 8),
    (13422, 256, 2048, 128, 8),
    (16384, 256, 2048, 128, 8),
    (1036, 256, 2048, 512, 8),
    (4435, 256, 2048, 512, 8),
    (4108, 256, 2048, 512, 8),
    (13422, 256, 2048, 512, 8),
    (16384, 256, 2048, 512, 8),
]


def _get_supported_dtypes():
    if VENDOR == "thead":
        # The PPU DeepGEMM expert path only accepts bfloat16 weights.
        return [torch.bfloat16]
    return [torch.bfloat16, torch.float16]


def _input_fn(shape, dtype, device):
    num_tokens, num_experts, hidden_size, intermediate_size, topk = shape
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, dtype=dtype, device=device
    )
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, dtype=dtype, device=device
    )
    gating = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)
    yield hidden_states, w1, w2, topk_weights, topk_ids


def _vllm_fused_moe_op(hidden_states, w1, w2, topk_weights, topk_ids):
    return vllm_fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
    )


# Production baseline used by this benchmark and by the accuracy tests.
# Kept in the benchmark module so the vLLM entry point being measured is
# visible here instead of hidden in a library testing helper.
_CACHE = {"key": None, "layer": None, "workspace_ready": False}


def _build_layer(num_experts, hidden_size, intermediate_size, topk, dtype):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.v1.worker.workspace import init_workspace_manager

    if not _CACHE["workspace_ready"]:
        init_workspace_manager(torch.device("cuda"))
        _CACHE["workspace_ready"] = True
    ctx = VllmConfig()
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    try:
        with set_current_vllm_config(ctx):
            layer = FusedMoE(
                num_experts,
                topk,
                hidden_size,
                intermediate_size,
                params_dtype=dtype,
                quant_config=None,
                tp_size=1,
                dp_size=1,
                pcp_size=1,
            )
    finally:
        torch.set_default_device(previous)
    # vLLM 0.25 moved the expert weights onto a nested RoutedExperts module.
    _CACHE.update(layer=layer, experts=getattr(layer, "routed_experts", layer), ctx=ctx)


def thead_fused_moe_op(hidden_states, w1, w2, topk_weights, topk_ids):
    """Run vLLM's production PPU expert method on the given top-k inputs."""
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    _, hidden_size = hidden_states.shape
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[2]
    topk = topk_ids.shape[1]
    key = (
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        hidden_states.dtype,
        w1.data_ptr(),
        w2.data_ptr(),
    )
    if key != _CACHE["key"]:
        _CACHE["layer"] = None
        torch.cuda.empty_cache()
        _build_layer(
            num_experts, hidden_size, intermediate_size, topk, hidden_states.dtype
        )
        _CACHE["key"] = key
        experts = _CACHE["experts"]
        with set_current_vllm_config(_CACHE["ctx"]):
            experts.w13_weight.data.copy_(w1)
            experts.w2_weight.data.copy_(w2)
            experts.quant_method.process_weights_after_loading(experts)

    experts = _CACHE["experts"]
    ctx = _CACHE["ctx"]
    # Call the production expert method with the same top-k inputs as the
    # FlagGems op, so neither side pays for router reconstruction.
    with torch.no_grad(), set_current_vllm_config(ctx), set_forward_context(None, ctx):
        return experts.quant_method.apply(
            layer=experts,
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )


def metax_fused_moe_op(hidden_states, w1, w2, topk_weights, topk_ids):
    """Run vLLM-MetaX's production unquantized MoE method."""
    import vllm_metax
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    # The flagos MetaX image does not ship the mctlass extension.  Without
    # it the vendor TritonExperts path asserts in its BF16 block-size
    # override, so keep the available MetaX Triton path selected.
    try:
        import mctlassEx  # noqa: F401
    except Exception:
        import vllm_metax.envs as _mx_envs

        _mx_envs.MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE = False
        _mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API = False

    # The plugin's OOT method is registered by this entry point; importing
    # vllm_metax alone leaves FusedMoE on the upstream Triton fallback.
    if not _CACHE.get("metax_registered"):
        vllm_metax.register_customized()
        _CACHE["metax_registered"] = True

    # Keep subsequent FlagGems allocations on the same device; _build_layer
    # constructs the vLLM layer under an explicit default device.
    torch.set_default_device(hidden_states.device)

    num_experts = w1.shape[0]
    intermediate_size = w2.shape[2]
    hidden_size = w2.shape[1]
    topk = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32)
    struct_key = (
        num_experts,
        hidden_size,
        intermediate_size,
        topk,
        hidden_states.dtype,
    )
    # Reuse the layer for every token count sharing the same expert geometry;
    # rebuilding it per shape retains too much vLLM workspace on 64 GiB C550s.
    if struct_key != _CACHE.get("metax_struct"):
        _CACHE["layer"] = None
        torch.cuda.empty_cache()
        _build_layer(
            num_experts, hidden_size, intermediate_size, topk, hidden_states.dtype
        )
        _CACHE["metax_struct"] = struct_key
        _CACHE["metax_weight_key"] = None

    experts = _CACHE["experts"]
    weight_key = (w1.data_ptr(), w2.data_ptr(), hidden_states.dtype)
    if weight_key != _CACHE.get("metax_weight_key"):
        with set_current_vllm_config(_CACHE["ctx"]):
            experts.w13_weight.data = w1
            experts.w2_weight.data = w2
            experts.quant_method.process_weights_after_loading(experts)
            if hasattr(experts, "ensure_moe_quant_config_init"):
                experts.ensure_moe_quant_config_init()
        _CACHE["metax_weight_key"] = weight_key

    ctx = _CACHE["ctx"]
    with torch.no_grad(), set_current_vllm_config(ctx), set_forward_context(None, ctx):
        return experts.quant_method.apply(
            layer=experts,
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )


def mthreads_fused_moe_op(hidden_states, w1, w2, topk_weights, topk_ids):
    """Run vLLM-MUSA's production MoE method on the given top-k inputs."""
    # vLLM's model executor must be imported before vllm_musa so the vendor
    # patch and the MusaUnquantizedFusedMoEMethod OOT registration are active.
    import vllm.model_executor.layers.fused_moe.fused_moe  # noqa: F401
    import vllm_musa  # noqa: F401
    import vllm_musa.model_executor  # noqa: F401
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    # Keep subsequent FlagGems allocations on the same device; _build_layer
    # constructs the vLLM layer under an explicit default device.
    torch.set_default_device(hidden_states.device)

    try:  # MoE custom ops live in the in-tree extension on MUSA.
        import vllm._moe_C  # noqa: F401
    except ImportError:  # pragma: no cover - non-MUSA builds
        pass

    _, hidden_size = hidden_states.shape
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[2]
    topk = topk_ids.shape[1]
    # Production passes int32 routing ids; the DeepGEMM gate rejects int64.
    topk_ids = topk_ids.to(torch.int32)
    key = (
        hidden_states.shape[0],
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        hidden_states.dtype,
        w1.data_ptr(),
        w2.data_ptr(),
    )
    if key != _CACHE["key"]:
        _CACHE["layer"] = None
        _build_layer(
            num_experts, hidden_size, intermediate_size, topk, hidden_states.dtype
        )
        _CACHE["key"] = key
        experts = _CACHE["experts"]
        with set_current_vllm_config(_CACHE["ctx"]):
            experts.w13_weight.data.copy_(w1)
            experts.w2_weight.data.copy_(w2)
            experts.quant_method.process_weights_after_loading(experts)
            if hasattr(experts, "ensure_moe_quant_config_init"):
                experts.ensure_moe_quant_config_init()

    experts = _CACHE["experts"]
    ctx = _CACHE["ctx"]
    with torch.no_grad(), set_current_vllm_config(ctx), set_forward_context(None, ctx):
        return experts.quant_method.forward_oot(
            experts,
            hidden_states,
            topk_weights,
            topk_ids,
            None,
            None,
        )


# Vendors whose production MoE is not the stock vLLM fused_experts_impl.
PRODUCTION_MOE_VENDORS = {
    "thead": thead_fused_moe_op,
    "metax": metax_fused_moe_op,
    "mthreads": mthreads_fused_moe_op,
}


def production_moe_op(vendor_name):
    """Return the platform production fused MoE callable, or None."""
    return PRODUCTION_MOE_VENDORS.get(vendor_name)


_PRODUCTION_BASELINE_VENDORS = {"thead", "metax", "mthreads"}


def _get_baseline_op():
    if VENDOR in _PRODUCTION_BASELINE_VENDORS:
        return production_moe_op(VENDOR)
    if VENDOR in VLLM_NATIVE_VENDORS:
        return _vllm_fused_moe_op
    return None


class FusedMoEBenchmark(base.GenericBenchmarkExcluse1D):
    DEFAULT_SHAPE_DESC = "num_tokens, num_experts, hidden_size, intermediate_size, topk"

    def init_user_config(self):
        super().init_user_config()
        if self.torch_op is None:
            self.to_bench_metrics = [
                metric
                for metric in self.to_bench_metrics
                if metric not in ("latency_base", "speedup")
            ]

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = list(_SHAPES)


@pytest.mark.fused_experts_impl
def test_fused_moe():
    bench = FusedMoEBenchmark(
        input_fn=_input_fn,
        op_name="fused_experts_impl",
        torch_op=_get_baseline_op(),
        gems_op=flaggems_vllm.fused_experts_impl,
        dtypes=_get_supported_dtypes(),
    )
    bench.run()
