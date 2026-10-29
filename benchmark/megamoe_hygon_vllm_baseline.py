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

"""Native vLLM expert-parallel W8A8 baseline for Hygon MegaMoE.

The benchmark boundary is:

prequantized INT8 A1 + FP32 scale + TopK
    -> vLLM native all-gather dispatch
    -> local W8A8 experts
    -> vLLM native reduce-scatter finalize
    -> BF16 output

The HCU vLLM 0.18.1 TritonExperts entry point rejects INT8 input even though
its lower-level W8A8 helpers accept an already quantized A1. The narrow
adapter below removes only the first activation quantizer and preserves the
installed dynamic INT8 qA2 implementation. It is benchmark-only and does not
participate in FlagGems-vllm operator dispatch.

The caller must initialize vLLM distributed/model-parallel state, its workspace
manager, and a forward context whose DP metadata contains the local token
count. The EP device communicator must use allgather_reducescatter.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VLLM_REQUIRED_VERSION = "0.18.1"
COMMUNICATION_BACKEND = "allgather_reducescatter"
OUTPUT_DTYPE_NAME = "torch.bfloat16"


# Contract inspection must work on a workstation without the HCU runtime.
# Execution fails explicitly in _require_runtime.
_RUNTIME_IMPORT_ERROR: BaseException | None = None
try:  # pragma: no cover - availability is environment-specific
    import torch
    import vllm
    import vllm.model_executor.layers.fused_moe.fused_moe as _fused_moe
    import vllm.model_executor.layers.fused_moe.modular_kernel as _mk
    from vllm.distributed import get_ep_group as _get_ep_group
    from vllm.model_executor.layers.fused_moe.activation import (
        MoEActivation as _MoEActivation,
    )
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig as _FusedMoEConfig,
    )
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEParallelConfig as _FusedMoEParallelConfig,
    )
    from vllm.model_executor.layers.fused_moe.config import (
        RoutingMethodType as _RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.config import (
        int8_w8a8_moe_quant_config as _int8_w8a8_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size as _moe_align_block_size,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
        MoEPrepareAndFinalizeNaiveDPEPModular as _NaiveDPEPPrepare,
    )
    from vllm.model_executor.layers.fused_moe.utils import (
        _resize_cache,
    )
    from vllm.model_executor.layers.fused_moe.utils import (
        moe_kernel_quantize_input as _moe_kernel_quantize_input,
    )
    from vllm.triton_utils import tl as _tl
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _RUNTIME_IMPORT_ERROR = exc
    torch = None  # type: ignore[assignment]
    vllm = None  # type: ignore[assignment]
    _fused_moe = None
    _mk = None
    _get_ep_group = None
    _MoEActivation = None
    _FusedMoEConfig = None
    _FusedMoEParallelConfig = None
    _RoutingMethodType = None
    _int8_w8a8_moe_quant_config = None
    _moe_align_block_size = None
    _NaiveDPEPPrepare = object  # type: ignore[assignment,misc]
    _resize_cache = None
    _moe_kernel_quantize_input = None
    _tl = None


_TritonExpertsBase = _fused_moe.TritonExperts if _fused_moe is not None else object


def _require_runtime() -> None:
    if _RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "The HCU vLLM runtime is required to execute the native EP baseline"
        ) from _RUNTIME_IMPORT_ERROR

    version = str(vllm.__version__).split("+", 1)[0]
    if version != VLLM_REQUIRED_VERSION:
        raise RuntimeError(
            f"native EP baseline requires vLLM {VLLM_REQUIRED_VERSION}; "
            f"got {vllm.__version__}"
        )
    if not str(_mk.__name__).startswith("vllm_hcu."):
        raise RuntimeError(
            "Expected the HCU override of fused_moe.modular_kernel; "
            f"got {_mk.__name__}"
        )


@dataclass(frozen=True)
class NativeEpContract:
    """One fixed-shape, linear-placement EP benchmark case."""

    world_size: int
    rank: int
    tokens_per_rank: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int

    def __post_init__(self) -> None:
        integer_fields = (
            "world_size",
            "rank",
            "tokens_per_rank",
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "top_k",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {value!r}")
        if self.world_size not in (2, 4, 8):
            raise ValueError("world_size must be one of 2, 4, or 8")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size})")
        for name in (
            "tokens_per_rank",
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "top_k",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_experts % self.world_size:
            raise ValueError("num_experts must be divisible by world_size")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.world_size

    @property
    def local_expert_begin(self) -> int:
        return self.rank * self.experts_per_rank

    @property
    def local_expert_end(self) -> int:
        return self.local_expert_begin + self.experts_per_rank


def make_linear_expert_map(contract: NativeEpContract, *, device: Any) -> Any:
    """Build vLLM's global-to-local map for contiguous expert placement."""

    _require_runtime()
    result = torch.full(
        (contract.num_experts,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    result[contract.local_expert_begin : contract.local_expert_end] = torch.arange(
        contract.experts_per_rank,
        dtype=torch.int32,
        device=device,
    )
    return result


class PrequantizedHcuTritonExperts(_TritonExpertsBase):
    """HCU Triton experts accepting an already-quantized dynamic INT8 A1."""

    def __init__(self, moe_config: Any, quant_config: Any) -> None:
        _require_runtime()
        super().__init__(moe_config, quant_config)
        if quant_config.quant_dtype is not torch.int8:
            raise TypeError("PrequantizedHcuTritonExperts requires INT8 activations")
        if not quant_config.use_int8_w8a8:
            raise ValueError("The quantization descriptor is not W8A8 INT8")
        if not quant_config.per_act_token_quant:
            raise ValueError("Dynamic per-token activation quantization is required")
        if quant_config.block_shape is not None:
            raise NotImplementedError(
                "Blockwise activation quantization is unsupported"
            )
        if quant_config.a1_scale is not None or quant_config.a2_scale is not None:
            raise NotImplementedError("Only dynamic A1/A2 scales are supported")
        if quant_config.w1_bias is not None or quant_config.w2_bias is not None:
            raise NotImplementedError("Biased experts are outside this baseline")

    def apply(
        self,
        output: Any,
        hidden_states: Any,
        w1: Any,
        w2: Any,
        topk_weights: Any,
        topk_ids: Any,
        activation: Any,
        global_num_experts: int,
        expert_map: Any,
        a1q_scale: Any,
        a2_scale: Any,
        workspace13: Any,
        workspace2: Any,
        expert_tokens_meta: Any,
        apply_router_weight_on_input: bool,
    ) -> None:
        """Run both HCU W8A8 expert GEMMs while preserving dynamic qA2."""

        if hidden_states.dtype is not torch.int8 or not hidden_states.is_contiguous():
            raise TypeError("dispatched A1 must be contiguous torch.int8")
        if hidden_states.ndim != 2:
            raise ValueError("dispatched A1 must be rank 2")
        if a1q_scale is None or a1q_scale.dtype is not torch.float32:
            raise TypeError("dispatched A1 scale must be torch.float32")
        if tuple(a1q_scale.shape) != (hidden_states.shape[0], 1):
            raise ValueError("dispatched A1 scale must have shape [tokens, 1]")
        if not a1q_scale.is_contiguous():
            raise ValueError("dispatched A1 scale must be contiguous")
        if w1.dtype is not torch.int8 or w2.dtype is not torch.int8:
            raise TypeError("expert weights must be torch.int8")
        if output.dtype is not torch.bfloat16:
            raise TypeError("native EP output/workspace dtype must be torch.bfloat16")
        if (
            workspace13.dtype is not torch.bfloat16
            or workspace2.dtype is not torch.bfloat16
        ):
            raise TypeError("native EP workspaces must be torch.bfloat16")
        if hidden_states.shape[1] != w1.shape[2]:
            raise ValueError("A1 hidden dimension does not match w13")
        if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
            raise ValueError("gathered TopK IDs and weights must have equal shapes")
        if expert_tokens_meta is not None:
            raise NotImplementedError("Batched-expert metadata is unsupported")
        if a2_scale is not None:
            raise NotImplementedError("Static A2 quantization is unsupported")
        if apply_router_weight_on_input:
            raise NotImplementedError("Router-on-input weighting is unsupported")
        if activation is not _MoEActivation.SILU:
            raise NotImplementedError("This Qwen baseline is pinned to gated SiLU")
        if expert_map is None:
            raise ValueError("expert_map is required for native expert parallelism")

        local_experts, num_tokens, n13, hidden_size, top_k = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        if global_num_experts <= local_experts:
            raise ValueError("global_num_experts must exceed local experts in EP mode")

        config = _fused_moe.try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k,
            self.quant_config.config_name(torch.bfloat16),
            num_tokens,
            block_shape=self.block_shape,
        )
        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k, n13))
        activation_out_dim = self.adjust_N_for_activation(n13, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k, activation_out_dim)
        )
        intermediate_cache3 = _resize_cache(
            workspace2, (num_tokens, top_k, hidden_size)
        )

        # ignore_invalid_experts is required for EP maps whose remote experts
        # are represented by -1; omitting it caused the historical NP4 VMFault.
        sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size(
            topk_ids,
            config["BLOCK_SIZE_M"],
            global_num_experts,
            expert_map,
            ignore_invalid_experts=True,
        )

        _fused_moe.dispatch_fused_moe_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            self.w1_scale,
            self.w1_zp,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            top_k,
            config,
            compute_type=_tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=True,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=True,
            block_shape=None,
            B_bias=None,
        )
        self.activation(
            activation,
            intermediate_cache2,
            intermediate_cache1.view(-1, n13),
        )

        qintermediate_cache2, a2q_scale = _moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=None,
            quant_dtype=torch.int8,
            per_act_token_quant=True,
            block_shape=None,
        )
        if qintermediate_cache2.dtype is not torch.int8:
            raise RuntimeError("HCU qA2 did not produce torch.int8")
        if a2q_scale is None or a2q_scale.dtype is not torch.float32:
            raise RuntimeError("HCU qA2 did not produce FP32 per-token scales")

        # Remote experts are represented by -1 blocks. Clear their output so
        # reduce-scatter sums only contributions from the local owner.
        intermediate_cache3.zero_()
        _fused_moe.dispatch_fused_moe_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            self.w2_scale,
            self.w2_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            1,
            config,
            compute_type=_tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=True,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=True,
            block_shape=None,
            B_bias=None,
        )
        self.moe_sum(intermediate_cache3, output)


class PrequantizedNativeEpPrepare(_NaiveDPEPPrepare):
    """Bypass qA1 and inherit vLLM's native reduce-scatter finalize."""

    def __init__(self, *, is_sequence_parallel: bool = False) -> None:
        _require_runtime()
        super().__init__(is_sequence_parallel=is_sequence_parallel)
        self.bound_input: tuple[Any, Any] | None = None

    def prepare(
        self,
        a1: Any,
        topk_weights: Any,
        topk_ids: Any,
        num_experts: int,
        expert_map: Any,
        apply_router_weight_on_input: bool,
        quant_config: Any,
        defer_input_quant: bool = False,
    ) -> Any:
        if self.bound_input is None:
            raise RuntimeError("No prequantized A1 is bound to the prepare adapter")
        if apply_router_weight_on_input:
            raise NotImplementedError("Router-on-input weighting is unsupported")
        if defer_input_quant:
            raise RuntimeError("Prequantized HCU experts must not defer quantization")
        if expert_map is None or tuple(expert_map.shape) != (num_experts,):
            raise ValueError("native EP requires a global-to-local expert_map")
        if (
            quant_config.quant_dtype is not torch.int8
            or not quant_config.use_int8_w8a8
            or not quant_config.per_act_token_quant
            or quant_config.block_shape is not None
        ):
            raise TypeError("prepare requires dynamic per-token INT8 W8A8")

        hidden_q, hidden_scale = self.bound_input
        if a1.dtype is not torch.bfloat16 or a1.shape != hidden_q.shape:
            raise ValueError("invalid BF16 output metadata carrier")

        result = _get_ep_group().dispatch(
            hidden_q,
            topk_weights,
            topk_ids,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=[hidden_scale],
        )
        if not isinstance(result, tuple) or len(result) != 4:
            raise RuntimeError("native EP dispatch did not return its scale payload")
        hidden_q, topk_weights, topk_ids, scales = result
        if not isinstance(scales, list) or len(scales) != 1:
            raise RuntimeError("native EP dispatch returned invalid A1 scales")
        hidden_scale = scales[0]
        if hidden_q.dtype is not torch.int8 or hidden_scale.dtype is not torch.float32:
            raise TypeError("native EP dispatch changed INT8/FP32 payload dtypes")
        if tuple(hidden_scale.shape) != (hidden_q.shape[0], 1):
            raise ValueError("native EP dispatch returned invalid scale shape")
        return hidden_q, hidden_scale, None, topk_ids, topk_weights


class HcuNativeEpInt8Baseline:
    """Complete native-EP baseline with prequantized input."""

    backend_name = "hcu_vllm_0.18.1_native_ep_agr_s_w8a8_prequantized"
    input_contract = "INT8 A1 + FP32 per-token scale + FP32 TopK weights + INT64 IDs"
    communication_contract = "all-gather dispatch + reduce-scatter finalize"
    first_input_quantization_in_timed_region = False
    intermediate_quantization_in_timed_region = True

    def __init__(
        self,
        *,
        contract: NativeEpContract,
        w13: Any,
        w2: Any,
        w13_scale: Any,
        w2_scale: Any,
        expert_map: Any | None = None,
    ) -> None:
        _require_runtime()
        self.contract = contract
        self.w13 = w13
        self.w2 = w2
        self.w13_scale = w13_scale
        self.w2_scale = w2_scale
        self.expert_map = (
            make_linear_expert_map(contract, device=w13.device)
            if expert_map is None
            else expert_map
        )
        self._validate_weights_and_map()
        self._ep_runtime = self._validate_ep_runtime()

        parallel_config = _FusedMoEParallelConfig(
            tp_size=1,
            pcp_size=1,
            dp_size=contract.world_size,
            ep_size=contract.world_size,
            tp_rank=0,
            pcp_rank=0,
            dp_rank=contract.rank,
            ep_rank=contract.rank,
            sp_size=1,
            use_ep=True,
            all2all_backend=COMMUNICATION_BACKEND,
            enable_eplb=False,
        )
        moe_config = _FusedMoEConfig(
            num_experts=contract.num_experts,
            experts_per_token=contract.top_k,
            hidden_dim=contract.hidden_size,
            intermediate_size_per_partition=contract.intermediate_size,
            num_local_experts=contract.experts_per_rank,
            num_logical_experts=contract.num_experts,
            activation=_MoEActivation.SILU,
            device=w13.device,
            routing_method=_RoutingMethodType.Default,
            moe_parallel_config=parallel_config,
            in_dtype=torch.bfloat16,
            router_logits_dtype=torch.float32,
            moe_backend="triton",
            has_bias=False,
            is_act_and_mul=True,
            is_lora_enabled=False,
            disable_inplace=True,
        )
        quant_config = _int8_w8a8_moe_quant_config(
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            a1_scale=None,
            a2_scale=None,
            per_act_token_quant=True,
            block_shape=None,
        )
        self.prepare = PrequantizedNativeEpPrepare(is_sequence_parallel=False)
        self.experts = PrequantizedHcuTritonExperts(moe_config, quant_config)
        self.kernel = _mk.FusedMoEKernel(
            self.prepare,
            self.experts,
            moe_parallel_config=parallel_config,
            inplace=False,
        )
        self.metadata_carrier = torch.empty(
            (contract.tokens_per_rank, contract.hidden_size),
            dtype=torch.bfloat16,
            device=w13.device,
        )

    def _validate_weights_and_map(self) -> None:
        cfg = self.contract
        local_e = cfg.experts_per_rank
        expected_shapes = {
            "w13": (local_e, 2 * cfg.intermediate_size, cfg.hidden_size),
            "w2": (local_e, cfg.hidden_size, cfg.intermediate_size),
            "w13_scale": (local_e, 2 * cfg.intermediate_size, 1),
            "w2_scale": (local_e, cfg.hidden_size, 1),
        }
        tensors = {
            "w13": self.w13,
            "w2": self.w2,
            "w13_scale": self.w13_scale,
            "w2_scale": self.w2_scale,
        }
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != expected_shapes[name]:
                raise ValueError(
                    f"{name} must have shape {expected_shapes[name]}, "
                    f"got {tuple(tensor.shape)}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if self.w13.dtype is not torch.int8 or self.w2.dtype is not torch.int8:
            raise TypeError("w13 and w2 must be torch.int8")
        if (
            self.w13_scale.dtype is not torch.float32
            or self.w2_scale.dtype is not torch.float32
        ):
            raise TypeError("w13_scale and w2_scale must be torch.float32")
        device = self.w13.device
        if device.type != "cuda":
            raise ValueError("weights must reside on a CUDA-alike HCU device")
        if any(tensor.device != device for tensor in tensors.values()):
            raise ValueError("weights and scales must share one HCU device")

        expert_map = self.expert_map
        if tuple(expert_map.shape) != (cfg.num_experts,):
            raise ValueError("expert_map must contain one entry per global expert")
        if expert_map.dtype is not torch.int32:
            raise TypeError("expert_map must be torch.int32")
        if not expert_map.is_contiguous() or expert_map.device != device:
            raise ValueError("expert_map must be contiguous and colocated with weights")

        values = expert_map.detach().cpu().tolist()
        expected = [-1] * cfg.num_experts
        expected[cfg.local_expert_begin : cfg.local_expert_end] = list(
            range(cfg.experts_per_rank)
        )
        if values != expected:
            raise ValueError(
                "expert_map must exactly match contiguous linear expert placement "
                f"for rank {cfg.rank}"
            )

    def _validate_ep_runtime(self) -> dict[str, Any]:
        try:
            group = _get_ep_group()
        except (AssertionError, RuntimeError) as exc:
            raise RuntimeError("vLLM expert-parallel group is not initialized") from exc
        actual_world = int(group.world_size)
        actual_rank = int(group.rank_in_group)
        if (
            actual_world != self.contract.world_size
            or actual_rank != self.contract.rank
        ):
            raise RuntimeError(
                "EP group does not match the requested contract: "
                f"group=({actual_world}, {actual_rank}), "
                f"contract=({self.contract.world_size}, {self.contract.rank})"
            )
        communicator = getattr(group, "device_communicator", None)
        backend = getattr(communicator, "all2all_backend", None)
        if backend != COMMUNICATION_BACKEND:
            raise RuntimeError(
                f"EP communicator must use {COMMUNICATION_BACKEND!r}; got {backend!r}"
            )
        manager = getattr(communicator, "all2all_manager", None)
        manager_name = (
            None
            if manager is None
            else f"{manager.__class__.__module__}.{manager.__class__.__qualname__}"
        )
        if manager is None or "AgRsAll2AllManager" not in manager_name:
            raise RuntimeError(
                "allgather_reducescatter did not instantiate AgRsAll2AllManager"
            )
        return {
            "world_size": actual_world,
            "rank": actual_rank,
            "all2all_backend": backend,
            "manager": manager_name,
        }

    def _validate_inputs(
        self,
        hidden_q: Any,
        hidden_scale: Any,
        topk_weights: Any,
        topk_ids: Any,
    ) -> None:
        cfg = self.contract
        expected_hidden = (cfg.tokens_per_rank, cfg.hidden_size)
        expected_topk = (cfg.tokens_per_rank, cfg.top_k)
        if hidden_q.dtype is not torch.int8 or tuple(hidden_q.shape) != expected_hidden:
            raise TypeError(f"hidden_q must be torch.int8 {expected_hidden}")
        if hidden_scale.dtype is not torch.float32 or tuple(hidden_scale.shape) != (
            cfg.tokens_per_rank,
            1,
        ):
            raise TypeError(
                f"hidden_scale must be torch.float32 {(cfg.tokens_per_rank, 1)}"
            )
        if (
            topk_weights.dtype is not torch.float32
            or tuple(topk_weights.shape) != expected_topk
        ):
            raise TypeError(f"topk_weights must be torch.float32 {expected_topk}")
        if topk_ids.dtype is not torch.int64 or tuple(topk_ids.shape) != expected_topk:
            raise TypeError(f"topk_ids must be torch.int64 {expected_topk}")
        tensors = (hidden_q, hidden_scale, topk_weights, topk_ids)
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("all native EP inputs must be contiguous")
        if any(tensor.device != self.w13.device for tensor in tensors):
            raise ValueError("inputs, weights, scales, and routes must share one HCU")

    def audit_input_values(
        self,
        hidden_q: Any,
        hidden_scale: Any,
        topk_weights: Any,
        topk_ids: Any,
        *,
        normalized_atol: float = 4e-3,
    ) -> dict[str, Any]:
        """Synchronizing, untimed input audit for benchmark setup."""

        self._validate_inputs(hidden_q, hidden_scale, topk_weights, topk_ids)
        ids_min = int(topk_ids.min())
        ids_max = int(topk_ids.max())
        scales_finite = bool(torch.isfinite(hidden_scale).all())
        weights_finite = bool(torch.isfinite(topk_weights).all())
        normalized = bool(
            torch.allclose(
                topk_weights.sum(dim=-1),
                torch.ones(
                    (self.contract.tokens_per_rank,),
                    dtype=torch.float32,
                    device=topk_weights.device,
                ),
                atol=normalized_atol,
                rtol=0.0,
            )
        )
        if ids_min < 0 or ids_max >= self.contract.num_experts:
            raise ValueError("topk_ids contain an out-of-range global expert")
        if not scales_finite or not weights_finite or not normalized:
            raise ValueError(
                "input scales/TopK weights failed finite/normalization audit"
            )
        return {
            "topk_id_min": ids_min,
            "topk_id_max": ids_max,
            "hidden_scales_finite": scales_finite,
            "topk_weights_finite": weights_finite,
            "topk_weights_normalized": normalized,
        }

    def forward(
        self,
        hidden_q: Any,
        hidden_scale: Any,
        topk_weights: Any,
        topk_ids: Any,
    ) -> Any:
        """Run the full all-gather -> experts -> reduce-scatter baseline."""

        self._validate_inputs(hidden_q, hidden_scale, topk_weights, topk_ids)
        if self.prepare.bound_input is not None:
            raise RuntimeError(
                "Concurrent calls on one native EP adapter are unsupported"
            )
        self.prepare.bound_input = (hidden_q, hidden_scale)
        try:
            output = self.kernel.apply(
                hidden_states=self.metadata_carrier,
                w1=self.w13,
                w2=self.w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=_MoEActivation.SILU,
                global_num_experts=self.contract.num_experts,
                expert_map=self.expert_map,
                apply_router_weight_on_input=False,
            )
        finally:
            self.prepare.bound_input = None
        if output.dtype is not torch.bfloat16:
            raise RuntimeError(f"native EP returned unexpected dtype {output.dtype}")
        expected_shape = (
            self.contract.tokens_per_rank,
            self.contract.hidden_size,
        )
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(f"native EP returned shape {tuple(output.shape)}")
        return output

    __call__ = forward

    def metadata(self, *, include_source_hashes: bool = True) -> dict[str, Any]:
        """Return serializable timing-boundary and runtime evidence."""

        return {
            "backend_name": self.backend_name,
            "source": "FlagGems-vllm benchmark native EP baseline",
            "contract": asdict(self.contract),
            "placement": "linear_contiguous",
            "input_contract": self.input_contract,
            "output_dtype": OUTPUT_DTYPE_NAME,
            "communication": self.communication_contract,
            "ep_runtime": dict(self._ep_runtime),
            "timing_boundary": {
                "included": [
                    "native_all_gather_dispatch_hidden_topk_and_scale",
                    "local_w8a8_gemm1",
                    "gated_silu",
                    "dynamic_per_token_qA2",
                    "local_w8a8_gemm2_and_topk_weight",
                    "native_reduce_scatter_finalize",
                ],
                "excluded": ["first_activation_quantization", "router"],
            },
            "runtime_evidence": runtime_evidence(
                include_source_hashes=include_source_hashes
            ),
        }


def runtime_evidence(*, include_source_hashes: bool = True) -> dict[str, Any]:
    """Return auditable proof of the installed vLLM/HCU execution path."""

    _require_runtime()
    module_names = (
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "vllm.model_executor.layers.fused_moe.modular_kernel",
        "vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep",
        "vllm.model_executor.layers.fused_moe.moe_align_block_size",
    )
    sources: dict[str, dict[str, Any]] = {}
    for name in module_names:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        entry: dict[str, Any] = {"path": str(path)}
        if include_source_hashes:
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        sources[name] = entry

    generic_apply = inspect.getsource(_fused_moe.TritonExperts.apply)
    generic_accepts_int8 = (
        "torch.int8"
        in generic_apply.partition("hidden_states.dtype in")[2].partition("]")[0]
    )
    return {
        "packages": {
            "vllm": str(vllm.__version__),
            "torch": str(torch.__version__),
        },
        "sources": sources,
        "public_entrypoint": "FusedMoEKernel.apply",
        "prepare": (
            "PrequantizedNativeEpPrepare.prepare -> get_ep_group().dispatch"
            "(extra_tensors=[hidden_scale])"
        ),
        "finalize": "MoEPrepareAndFinalizeNaiveDPEPModular.finalize",
        "experts": (
            "PrequantizedHcuTritonExperts using installed HCU fused_moe helpers"
        ),
        "generic_triton_experts_accepts_int8_input": generic_accepts_int8,
        "first_input_quantization_in_timed_region": False,
        "intermediate_qA2_in_timed_region": True,
        "source_patch_required": False,
    }


__all__ = [
    "COMMUNICATION_BACKEND",
    "HcuNativeEpInt8Baseline",
    "NativeEpContract",
    "PrequantizedHcuTritonExperts",
    "PrequantizedNativeEpPrepare",
    "VLLM_REQUIRED_VERSION",
    "make_linear_expert_map",
    "runtime_evidence",
]
