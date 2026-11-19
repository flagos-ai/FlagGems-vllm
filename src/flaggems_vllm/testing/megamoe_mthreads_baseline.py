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

"""Reference vLLM-MUSA expert-parallel baseline for MUSA MegaMoE.

The operator itself never imports this module. It exists so a reported
candidate-to-baseline ratio can be reproduced on a host that carries the
vLLM-MUSA runtime: it drives vLLM's own native EP MoE path over the same
frozen Qwen3.6 tensors the operator benchmarks consume, and reports the
event-bracketed per-call latency (warmup plus averaged iterations, slowest
rank wins) that ``vllm_reference.csv`` records as ``vllm_mean_ms``.

The vLLM stack is imported lazily: importing this module on a FlagTree-only
host must not fail. The engine helpers below are adapted from the vLLM-MUSA
layer-0 MoE benchmark used to produce the reference CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20

# vLLM symbols resolved by :func:`_require_runtime`.
torch: Any = None
dist: Any = None
parallel_state: Any = None
AgRsAll2AllManager: Any = None
VllmConfig: Any = None
set_current_vllm_config: Any = None
get_forward_context: Any = None
set_forward_context: Any = None
current_platform: Any = None
safe_open: Any = None
MCCL_ABI_PATH: Optional[str] = None
_RUNTIME: Optional[Dict[str, Any]] = None


def _require_runtime() -> Dict[str, Any]:
    """Import the vLLM-MUSA runtime once, after applying the MCCL ABI shim."""
    global _RUNTIME, torch, dist, parallel_state, AgRsAll2AllManager, VllmConfig
    global set_current_vllm_config, get_forward_context, set_forward_context
    global current_platform, safe_open, MCCL_ABI_PATH
    if _RUNTIME is not None:
        return _RUNTIME
    import torch as _torch
    import torch.distributed as _dist

    torch = _torch
    dist = _dist
    MCCL_ABI_PATH = configure_musa_vllm_collective_abi()
    # The vLLM dtype registry references FP4 even on builds without it; this
    # FP8-only baseline never allocates an FP4 tensor.
    if not hasattr(torch, "float4_e2m1fn_x2"):
        torch.float4_e2m1fn_x2 = torch.float8_e4m3fn
    import vllm_musa  # noqa: F401
    from safetensors import safe_open as _safe_open
    from vllm.config import VllmConfig as _VllmConfig
    from vllm.config import set_current_vllm_config as _set_vllm_config
    from vllm.distributed import parallel_state as _parallel_state
    from vllm.distributed.device_communicators.all2all import (
        AgRsAll2AllManager as _AgRsAll2AllManager,
    )
    from vllm.forward_context import get_forward_context as _get_forward_context
    from vllm.forward_context import set_forward_context as _set_forward_context
    from vllm.platforms import current_platform as _current_platform

    safe_open = _safe_open
    VllmConfig = _VllmConfig
    set_current_vllm_config = _set_vllm_config
    parallel_state = _parallel_state
    AgRsAll2AllManager = _AgRsAll2AllManager
    get_forward_context = _get_forward_context
    set_forward_context = _set_forward_context
    current_platform = _current_platform
    _RUNTIME = {
        "mccl": MCCL_ABI_PATH,
        "vllm_musa": vllm_musa,
        "torch": torch,
    }
    return _RUNTIME


def configure_musa_vllm_collective_abi() -> str | None:
    """Configure vLLM PyNccl wrapper to load Moore Threads' MCCL library.

    vLLM's CudaCommunicator uses the PyNccl interface for all_gatherv/reduce_scatterv.
    On MUSA there is no NVIDIA NCCL backend; vLLM-MUSA deployments expose MCCL through
    the NCCL ABI by setting VLLM_NCCL_SO_PATH to libmccl.so.2. The official MUSA serving
    environment uses this mechanism, so the benchmark must reproduce it before vLLM
    imports vllm.envs. We do NOT implement a new communication layer here.
    """
    if os.environ.get("VLLM_NCCL_SO_PATH"):
        return os.environ["VLLM_NCCL_SO_PATH"]

    candidates: list[Path] = []
    musa_home = os.environ.get("MUSA_HOME")
    if musa_home:
        candidates.append(Path(musa_home) / "lib" / "libmccl.so.2")
        candidates.append(Path(musa_home) / "lib64" / "libmccl.so.2")

    for raw in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if raw:
            candidates.append(Path(raw) / "libmccl.so.2")

    candidates.extend(
        [
            Path("/usr/local/musa/lib/libmccl.so.2"),
            Path("/usr/local/musa/lib64/libmccl.so.2"),
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            path = str(candidate.resolve())
            os.environ["VLLM_NCCL_SO_PATH"] = path
            # Do not let a stale environment setting disable the communicator.
            os.environ.setdefault("VLLM_DISABLE_PYNCCL", "0")
            return path

    raise RuntimeError(
        "MUSA EP requires vLLM PyNcclCommunicator to load MCCL through the NCCL ABI, "
        "but libmccl.so.2 was not found. Set VLLM_NCCL_SO_PATH explicitly, e.g. "
        "VLLM_NCCL_SO_PATH=/usr/local/musa/lib/libmccl.so.2"
    )


@contextmanager
def parallel_context(world_size, rank, local_rank):
    cfg = VllmConfig()
    parallel = cfg.parallel_config
    parallel.data_parallel_size = world_size
    parallel.enable_expert_parallel = True
    parallel.distributed_executor_backend = "external_launcher"
    with set_current_vllm_config(cfg):
        parallel_state.init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method="env://",
            backend="mccl",
        )
        try:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                prefill_context_model_parallel_size=1,
                decode_context_model_parallel_size=1,
                backend="mccl",
            )
            ep = parallel_state.get_ep_group()
            if (
                ep.world_size != world_size
                or parallel_state.get_dp_group().world_size != world_size
            ):
                raise RuntimeError("vLLM did not initialize the requested DP/EP groups")
            manager = None
            if world_size > 1:
                communicator = ep.device_communicator
                manager = getattr(communicator, "all2all_manager", None)
                pynccl = getattr(communicator, "pynccl_comm", None)
                if not isinstance(manager, AgRsAll2AllManager) or getattr(
                    pynccl, "disabled", True
                ):
                    raise RuntimeError(
                        f"Expected the vLLM AgRs manager backed by MCCL ({MCCL_ABI_PATH})"
                    )
            yield cfg, manager
        finally:
            parallel_state.destroy_model_parallel()
            parallel_state.destroy_distributed_environment()


@contextmanager
def forward_context(tokens, world_size, cfg):
    sizes = torch.full((world_size,), tokens, dtype=torch.int64, device="cpu")
    with set_forward_context(
        attn_metadata=None,
        vllm_config=cfg,
        num_tokens=tokens,
        num_tokens_across_dp=sizes,
    ):
        if world_size == 1:
            yield
        else:
            dp = get_forward_context().dp_metadata
            with dp.sp_local_sizes(sequence_parallel_size=1):
                yield


def load_inputs(
    shared, rank, tokens, hidden_size, num_experts, top_k, world_size, device
):
    fields = (
        ("input_fp8_bits", torch.uint8, hidden_size),
        ("input_scales", torch.float32, hidden_size // 128),
        ("topk_weights", torch.float32, top_k),
        ("topk_idx", torch.int64, top_k),
    )
    tensors = []
    for field, dtype, width in fields:
        tensor = shared[field]
        if (
            tensor.dtype != dtype
            or tensor.ndim != 3
            or tensor.shape[0] != world_size
            or tensor.shape[2] != width
        ):
            raise ValueError(
                f"Unexpected frozen {field}: {tensor.shape}, {tensor.dtype}"
            )
        if tokens > tensor.shape[1]:
            raise ValueError(f"{field} has only {tensor.shape[1]} tokens per rank")
        tensors.append(tensor[rank, :tokens].clone().contiguous())
    bits, scales, weights, ids = tensors
    if int(ids.min()) < 0 or int(ids.max()) >= num_experts:
        raise ValueError("Frozen top-k contains an invalid expert ID")
    if not torch.isfinite(weights).all() or (weights.sum(dim=-1) <= 0).any():
        raise ValueError("Frozen top-k contains invalid routing weights")
    return (
        bits.view(torch.float8_e4m3fn).to(device),
        scales.to(device),
        weights.to(device),
        ids.to(device=device, dtype=torch.int32),
    )


def pad_inputs(inputs, target_tokens):
    hidden, scales, weights, ids = inputs
    pad = target_tokens - hidden.shape[0]
    if pad <= 0:
        return inputs
    padded = tuple(
        torch.cat((tensor, tensor.new_zeros((pad, tensor.shape[1]))))
        for tensor in (hidden, scales, weights)
    )
    return (*padded, torch.cat((ids, ids[-1:].expand(pad, -1))))


def load_qwen36_layer0_real_weights(
    *,
    model_dir: Path,
    layer_idx: int,
    rank: int,
    world_size: int,
    local_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load only this EP rank's real Qwen3.6 FP8 expert weights.

    The checkpoint stores one gate/up/down tensor per expert. vLLM fused-MoE
    expects packed local tensors:
      W1       [E_local, 2*I, H]  = concat(gate_proj, up_proj)
      W2       [E_local, H, I]    = down_proj
      W1_scale [E_local, 2*I/128, H/128]
      W2_scale [E_local, H/128, I/128]

    Memory policy for v20:
      * inspect checkpoint keys only (metadata),
      * read/pack ONLY experts owned by this EP rank,
      * never construct a 256-expert global tensor on every rank.
    """
    index_path = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")

    config_json = json.loads(config_path.read_text(encoding="utf-8"))
    text_cfg = config_json.get("text_config", config_json)
    expected_global_experts = int(text_cfg["num_experts"])
    if expected_global_experts % world_size != 0:
        raise ValueError(
            f"num_experts={expected_global_experts} must be divisible by EP={world_size}"
        )
    if local_experts != expected_global_experts // world_size:
        raise ValueError(
            f"local_experts={local_experts} inconsistent with E={expected_global_experts}, EP={world_size}"
        )

    local_start = rank * local_experts
    local_end = local_start + local_experts
    prefix = f"model.language_model.layers.{layer_idx}.mlp.experts."

    direct_layer_file = model_dir / f"layers-{layer_idx}.safetensors"
    candidate_files: list[Path] = []
    if direct_layer_file.exists():
        candidate_files.append(direct_layer_file)

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    layer_markers = (f"layers.{layer_idx}.", f"layer.{layer_idx}.")
    for key, fn in weight_map.items():
        if any(marker in key for marker in layer_markers):
            candidate = model_dir / fn
            if candidate.exists() and candidate not in candidate_files:
                candidate_files.append(candidate)

    if not candidate_files:
        raise FileNotFoundError(
            f"No checkpoint shard found for layer {layer_idx}; expected {direct_layer_file}"
        )

    # Key discovery only: no tensor payload is materialized here.
    key_to_shard: dict[str, Path] = {}
    expert_pat = re.compile(
        re.escape(prefix) + r"(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )
    local_keys: dict[tuple[int, str], str] = {}
    discovered_ids: set[int] = set()
    for shard in candidate_files:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                m = expert_pat.match(key)
                if m:
                    e = int(m.group(1))
                    kind = m.group(2)
                    discovered_ids.add(e)
                    key_to_shard[key] = shard
                    if local_start <= e < local_end:
                        local_keys[(e, kind)] = key

    expected_ids = set(range(expected_global_experts))
    if not expected_ids.issubset(discovered_ids):
        missing = sorted(expected_ids - discovered_ids)
        raise KeyError(
            "Qwen3.6 checkpoint expert ids are incomplete. "
            f"Expected {expected_global_experts} experts, found {len(discovered_ids)}; "
            f"missing={missing[:32]}"
        )

    incomplete = [
        (e, kind)
        for e in range(local_start, local_end)
        for kind in ("gate_proj", "up_proj", "down_proj")
        if (e, kind) not in local_keys
    ]
    if incomplete:
        raise KeyError(
            "Missing local Qwen3.6 expert tensors: "
            + ", ".join(f"expert {e} {kind}" for e, kind in incomplete[:32])
        )

    def read(key: str) -> torch.Tensor:
        shard = key_to_shard[key]
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            return f.get_tensor(key).contiguous()

    def scale_key_for(weight_key: str) -> str:
        base = weight_key[: -len(".weight")]
        candidates = (
            base + ".weight_scale_inv",
            weight_key + "_scale_inv",
            base + "_weight_scale_inv",
            base + "_scale_inv",
        )
        for candidate in candidates:
            if candidate in key_to_shard:
                return candidate
        # weight_scale_inv may not be matched by expert_pat, so search the
        # local shard key set explicitly without loading the tensor.
        shard = key_to_shard[weight_key]
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            key_set = set(f.keys())
        for candidate in candidates:
            if candidate in key_set:
                key_to_shard[candidate] = shard
                return candidate
        raise KeyError(f"Missing FP8 weight_scale_inv for {weight_key}")

    gate_up_local: list[torch.Tensor] = []
    down_local: list[torch.Tensor] = []
    w1_scale_local: list[torch.Tensor] = []
    w2_scale_local: list[torch.Tensor] = []

    expected_gate_shape = None
    expected_up_shape = None
    expected_down_shape = None

    for e in range(local_start, local_end):
        gate_key = local_keys[(e, "gate_proj")]
        up_key = local_keys[(e, "up_proj")]
        down_key = local_keys[(e, "down_proj")]

        gate = read(gate_key)
        up = read(up_key)
        down = read(down_key)

        if expected_gate_shape is None:
            expected_gate_shape = tuple(gate.shape)
            expected_up_shape = tuple(up.shape)
            expected_down_shape = tuple(down.shape)
        if tuple(gate.shape) != expected_gate_shape:
            raise ValueError(
                f"Inconsistent gate_proj shape at expert {e}: {tuple(gate.shape)} vs {expected_gate_shape}"
            )
        if tuple(up.shape) != expected_up_shape:
            raise ValueError(
                f"Inconsistent up_proj shape at expert {e}: {tuple(up.shape)} vs {expected_up_shape}"
            )
        if tuple(down.shape) != expected_down_shape:
            raise ValueError(
                f"Inconsistent down_proj shape at expert {e}: {tuple(down.shape)} vs {expected_down_shape}"
            )

        if (
            gate.dtype != torch.float8_e4m3fn
            or up.dtype != torch.float8_e4m3fn
            or down.dtype != torch.float8_e4m3fn
        ):
            raise TypeError(
                f"Expected FP8 E4M3 weights at expert {e}, got "
                f"gate={gate.dtype}, up={up.dtype}, down={down.dtype}"
            )

        gate_scale = read(scale_key_for(gate_key)).float().contiguous()
        up_scale = read(scale_key_for(up_key)).float().contiguous()
        down_scale = read(scale_key_for(down_key)).float().contiguous()

        gate_up_local.append(torch.cat([gate, up], dim=0).contiguous())
        down_local.append(down)
        w1_scale_local.append(torch.cat([gate_scale, up_scale], dim=0).contiguous())
        w2_scale_local.append(down_scale)

    # Only E_local tensors are stacked.
    gate_up = torch.stack(gate_up_local, dim=0).contiguous()
    down = torch.stack(down_local, dim=0).contiguous()
    w1_scale = torch.stack(w1_scale_local, dim=0).contiguous()
    w2_scale = torch.stack(w2_scale_local, dim=0).contiguous()

    block_shape = [128, 128]
    expected_w1_scale = (
        local_experts,
        gate_up.shape[1] // 128,
        gate_up.shape[2] // 128,
    )
    expected_w2_scale = (local_experts, down.shape[1] // 128, down.shape[2] // 128)
    if tuple(w1_scale.shape) != expected_w1_scale:
        raise ValueError(
            f"W1 scale shape mismatch: {tuple(w1_scale.shape)} vs {expected_w1_scale}"
        )
    if tuple(w2_scale.shape) != expected_w2_scale:
        raise ValueError(
            f"W2 scale shape mismatch: {tuple(w2_scale.shape)} vs {expected_w2_scale}"
        )

    gate_up = gate_up.to(device=device, non_blocking=True).contiguous()
    down = down.to(device=device, non_blocking=True).contiguous()
    w1_scale = w1_scale.to(device=device, non_blocking=True).contiguous()
    w2_scale = w2_scale.to(device=device, non_blocking=True).contiguous()
    block_shape_tensor = torch.tensor(block_shape, device=device, dtype=torch.int32)

    return gate_up, down, w1_scale, w2_scale, block_shape_tensor


def expert_forward(
    moe,
    hidden,
    scales,
    weights,
    ids,
    w1,
    w2,
    w1_scale,
    w2_scale,
    expert_map,
    num_experts,
):
    """Adapt the vLLM expert APIs to accept the already quantized A1 tensor."""
    block_shape = [128, 128]
    tokens = hidden.shape[0]
    width = w1.shape[1]
    output_width = w2.shape[1]
    top_k = ids.shape[1]
    config_dtype = moe._get_config_dtype_str(
        use_fp8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        dtype=torch.bfloat16,
    )
    config = moe.try_get_optimal_moe_config(
        w1.size(),
        w2.size(),
        top_k,
        config_dtype,
        tokens,
        block_shape=block_shape,
    )
    cache = torch.empty(
        tokens * top_k * max(width, output_width),
        device=hidden.device,
        dtype=torch.bfloat16,
    )
    gate_up = cache[: tokens * top_k * width].view(tokens, top_k, width)
    partials = cache[: tokens * top_k * output_width].view(tokens, top_k, output_width)
    activation = moe.MoEActivation.from_str("silu")
    activation_width = moe.mk.FusedMoEExpertsModular.adjust_N_for_activation(
        width, activation
    )
    activated = torch.empty(
        (tokens * top_k, activation_width), device=hidden.device, dtype=torch.bfloat16
    )
    output = torch.empty(
        (tokens, output_width), device=hidden.device, dtype=torch.bfloat16
    )
    assignment = moe._prepare_expert_assignment(
        ids,
        config,
        tokens,
        top_k,
        num_experts,
        expert_map,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        block_shape=block_shape,
        ignore_invalid_experts=True,
    )
    options = dict(
        compute_type=moe.tl.bfloat16,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=block_shape,
        B_bias=None,
    )
    moe.dispatch_fused_moe_kernel(
        hidden,
        w1,
        gate_up,
        scales,
        w1_scale,
        None,
        weights,
        *assignment,
        False,
        top_k,
        config,
        **options,
    )
    moe.apply_moe_activation(activation, activated, gate_up.view(-1, width))
    quantized, a2_scale = moe.moe_kernel_quantize_input(
        A=activated,
        A_scale=None,
        quant_dtype=moe._get_config_quant_dtype(use_fp8_w8a8=True, use_int8_w8a8=False),
        per_act_token_quant=False,
        block_shape=block_shape,
    )
    partials.zero_()
    moe.dispatch_fused_moe_kernel(
        quantized,
        w2,
        partials,
        a2_scale,
        w2_scale,
        None,
        weights,
        *assignment,
        True,
        1,
        config,
        **options,
    )
    moe.ops.moe_sum(partials, output)
    return output


def infer(manager, moe, inputs, expert_args, tokens, events=None):
    """Dispatch A1 and its scales, run vLLM experts, then return local BF16 output."""
    hidden, scales, weights, ids = inputs
    if manager is not None:
        hidden, weights, ids, (scales,) = manager.dispatch(
            hidden,
            weights,
            ids,
            is_sequence_parallel=True,
            extra_tensors=[scales],
        )[:4]
    if events is not None:
        events[1].record()
    output = expert_forward(moe, hidden, scales, weights, ids, *expert_args)
    if events is not None:
        events[2].record()
    if manager is not None:
        output = manager.combine(output, is_sequence_parallel=True)
    return output[:tokens]


def benchmark(manager, moe, inputs, expert_args, tokens, warmup, iters):
    for _ in range(warmup):
        infer(manager, moe, inputs, expert_args, tokens)
        torch.musa.synchronize()

    samples = []
    for _ in range(iters):
        events = [torch.musa.Event(enable_timing=True) for _ in range(4)]
        wall_start = time.perf_counter_ns()
        dist.barrier()
        events[0].record()
        output = infer(manager, moe, inputs, expert_args, tokens, events)
        events[-1].record()
        events[-1].synchronize()
        wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
        stages = [a.elapsed_time(b) for a, b in zip(events, events[1:])]
        samples.append([*stages, events[0].elapsed_time(events[-1]), wall_ms])
    return output, samples


@dataclass(frozen=True)
class MtsBaselineContract:
    """Shape and placement contract shared with the candidate measurement."""

    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    world_size: int
    ep_min_tokens: int = 32
    layer_index: int = 0

    def __post_init__(self) -> None:
        if self.world_size not in (1, 2, 4, 8):
            raise ValueError(f"unsupported world size {self.world_size}")
        if self.num_experts % self.world_size:
            raise ValueError("num_experts must be divisible by world size")
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(f"invalid top_k {self.top_k}")
        if self.ep_min_tokens < 1:
            raise ValueError("ep_min_tokens must be positive")

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.world_size


def _dist_env() -> Dict[str, int]:
    """Return the torchrun rank layout, failing loudly outside torchrun."""
    try:
        return {
            key: int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        }
    except KeyError as error:
        raise RuntimeError(
            f"{error.args[0]} is unset; start this baseline with torchrun"
        ) from error


class MtsVllmBaseline:
    """Open vLLM-MUSA's native EP MoE path once and time it per token count."""

    def __init__(
        self, contract: MtsBaselineContract, model_dir: Path, input_dir: Path
    ) -> None:
        self.contract = contract
        self.model_dir = Path(model_dir)
        self.input_dir = Path(input_dir)
        self._manager: Any = None
        self._moe: Any = None
        self._expert_args: Any = None
        self._shared: Any = None
        self._cfg: Any = None
        self._parallel: Any = None
        self._device: Any = None
        self._manifest: Dict[str, Any] = {}

    def __enter__(self) -> "MtsVllmBaseline":
        _require_runtime()
        dist_env = _dist_env()
        rank = dist_env["RANK"]
        local_rank = dist_env["LOCAL_RANK"]
        world_size = dist_env["WORLD_SIZE"]
        if world_size != self.contract.world_size:
            raise ValueError(
                f"torchrun world {world_size} != contract world {self.contract.world_size}"
            )
        torch.musa.set_device(local_rank)
        self._device = torch.device("musa", local_rank)
        if type(current_platform).__module__ != "vllm_musa.platform":
            raise RuntimeError(
                f"expected the vLLM-MUSA platform, got {current_platform}"
            )
        os.environ["VLLM_MUSA_FUSED_MOE_DISPATCH"] = "upstream"
        import vllm.model_executor.layers.fused_moe.fused_moe as moe
        import vllm_musa.model_executor.layers.fused_moe.fused_moe  # noqa: F401

        if moe.fused_experts_impl.__name__ != "_musa_fused_experts_impl_dispatch":
            raise RuntimeError("the vLLM-MUSA expert API patch is not active")
        self._moe = moe

        config = json.loads((self.model_dir / "config.json").read_text())
        config = config.get("text_config", config)
        expected = (
            int(config["hidden_size"]),
            int(config["moe_intermediate_size"]),
            int(config["num_experts"]),
            int(config["num_experts_per_tok"]),
            world_size,
        )
        actual = (
            self.contract.hidden_size,
            self.contract.intermediate_size,
            self.contract.num_experts,
            self.contract.top_k,
            self.contract.world_size,
        )
        if actual != expected:
            raise ValueError(
                f"contract {actual} does not match model config {expected}"
            )

        manifest_path = self.input_dir / "manifest.json"
        self._manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        )
        self._shared_path = self.input_dir / "shared.pt"
        self._shared = torch.load(
            self._shared_path, mmap=True, map_location="cpu", weights_only=True
        )

        self._parallel = parallel_context(world_size, rank, local_rank)
        self._cfg, self._manager = self._parallel.__enter__()
        w1, w2, w1_scale, w2_scale, _ = load_qwen36_layer0_real_weights(
            model_dir=self.model_dir,
            layer_idx=self.contract.layer_index,
            rank=rank,
            world_size=world_size,
            local_experts=self.contract.local_experts,
            device=self._device,
        )
        expert_map = torch.full(
            (self.contract.num_experts,), -1, device=self._device, dtype=torch.int32
        )
        begin = rank * self.contract.local_experts
        end = begin + self.contract.local_experts
        expert_map[begin:end] = torch.arange(
            self.contract.local_experts, device=self._device, dtype=torch.int32
        )
        self._expert_args = (
            w1,
            w2,
            w1_scale,
            w2_scale,
            expert_map,
            self.contract.num_experts,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._parallel is not None:
            self._parallel.__exit__(exc_type, exc, tb)
            self._parallel = None

    def _padded(self, tokens: int) -> int:
        padded = max(tokens, self.contract.ep_min_tokens)
        return padded if self._manager is not None else tokens

    def _inputs(self, tokens: int) -> Any:
        padded = self._padded(tokens)
        inputs = load_inputs(
            self._shared,
            _dist_env()["RANK"],
            tokens,
            self.contract.hidden_size,
            self.contract.num_experts,
            self.contract.top_k,
            self.contract.world_size,
            self._device,
        )
        return pad_inputs(inputs, padded), padded

    def forward(self, tokens: int) -> Any:
        """Return the vLLM-MUSA MoE output for ``tokens`` tokens per rank."""
        inputs, padded = self._inputs(tokens)
        with forward_context(padded, self.contract.world_size, self._cfg):
            output, _samples = benchmark(
                self._manager, self._moe, inputs, self._expert_args, tokens, 0, 1
            )
        return output

    def benchmark(
        self, tokens: int, warmup: int = DEFAULT_WARMUP, iters: int = DEFAULT_ITERS
    ) -> Dict[str, Any]:
        """Time one rank; latency is the event-bracketed mean over iterations."""
        inputs, padded = self._inputs(tokens)
        started = time.monotonic()
        with forward_context(padded, self.contract.world_size, self._cfg):
            output, samples = benchmark(
                self._manager,
                self._moe,
                inputs,
                self._expert_args,
                tokens,
                warmup,
                iters,
            )
        totals = [row[3] for row in samples]
        stages = [list(column) for column in zip(*samples)]
        mean = sum(totals) / len(totals)
        return {
            "rank": _dist_env()["RANK"],
            "world": self.contract.world_size,
            "tokens": tokens,
            "padded_tokens": padded,
            "samples": len(totals),
            "total_mean_ms": mean,
            "total_min_ms": min(totals),
            "total_max_ms": max(totals),
            "dispatch_mean_ms": sum(stages[0]) / len(stages[0]),
            "expert_mean_ms": sum(stages[1]) / len(stages[1]),
            "combine_mean_ms": sum(stages[2]) / len(stages[2]),
            "wall_mean_ms": sum(stages[4]) / len(stages[4]),
            "elapsed_s": time.monotonic() - started,
            "finite": bool(output.isfinite().all().item()),
        }

    def metadata(self) -> Dict[str, Any]:
        """Return environment evidence for the recorded numbers."""
        from importlib import metadata as importlib_metadata

        versions = {}
        for name in ("torch", "vllm", "vllm-musa", "flagtree", "triton"):
            try:
                versions[name] = importlib_metadata.version(name)
            except importlib_metadata.PackageNotFoundError:
                versions[name] = "unavailable"
        return {
            "baseline": "vllm-musa native EP eager",
            "python": sys.version.split()[0],
            "versions": versions,
            "mccl_library": MCCL_ABI_PATH,
            "contract": {
                "hidden_size": self.contract.hidden_size,
                "intermediate_size": self.contract.intermediate_size,
                "num_experts": self.contract.num_experts,
                "top_k": self.contract.top_k,
                "world_size": self.contract.world_size,
                "ep_min_tokens": self.contract.ep_min_tokens,
                "layer_index": self.contract.layer_index,
            },
            "model_dir": str(self.model_dir),
            "input_dir": str(self.input_dir),
        }


def compare(
    candidate: Callable[[int], float],
    baseline: Callable[[int], float],
    tokens: Sequence[int],
    repeats: int = 3,
) -> List[Dict[str, Any]]:
    """Alternate candidate and baseline per token count to cancel drift."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows = []
    for count in tokens:
        candidate_samples = []
        baseline_samples = []
        for _ in range(repeats):
            candidate_samples.append(float(candidate(count)))
            baseline_samples.append(float(baseline(count)))
        candidate_mean = sum(candidate_samples) / len(candidate_samples)
        baseline_mean = sum(baseline_samples) / len(baseline_samples)
        rows.append(
            {
                "tokens": count,
                "candidate_ms": candidate_mean,
                "baseline_ms": baseline_mean,
                "ratio_baseline_over_candidate": (
                    baseline_mean / candidate_mean if candidate_mean else float("inf")
                ),
                "candidate_samples": candidate_samples,
                "baseline_samples": baseline_samples,
            }
        )
    return rows


def _token_list(value: str) -> List[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=_token_list, default=[1])
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--ep-min-tokens", type=int, default=32)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    dist_env = _dist_env()
    rank = dist_env["RANK"]
    config = json.loads((arguments.model_dir / "config.json").read_text())
    config = config.get("text_config", config)
    contract = MtsBaselineContract(
        hidden_size=int(config["hidden_size"]),
        intermediate_size=int(config["moe_intermediate_size"]),
        num_experts=int(config["num_experts"]),
        top_k=int(config["num_experts_per_tok"]),
        world_size=dist_env["WORLD_SIZE"],
        ep_min_tokens=arguments.ep_min_tokens,
        layer_index=arguments.layer_index,
    )
    results = []
    with MtsVllmBaseline(
        contract, arguments.model_dir, arguments.input_dir
    ) as baseline:
        for tokens in arguments.tokens:
            summary = baseline.benchmark(tokens, arguments.warmup, arguments.iters)
            results.append(summary)
            if rank == 0:
                print(
                    "MEGAMOE_BASELINE " + json.dumps(summary, sort_keys=True),
                    flush=True,
                )
        metadata = baseline.metadata()
    if rank == 0 and arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
