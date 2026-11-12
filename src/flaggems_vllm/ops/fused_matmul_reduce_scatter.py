# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Inference-only TLE GEMM + reduce-scatter with the PyTorch symm_mem signature.

The implementation supports 2D FP16/BF16, scatter_dim=0, sum/avg, aligned
shapes, and NVIDIA SM90+ or FlagTree HCU descriptor lowering. The HCU path
requires validation on target hardware. Communication uses a private FlagCX communicator
bootstrapped from the existing process group. It never owns the PyTorch group.
All ranks must call this API, configure, and clear in matching order.
"""

from __future__ import annotations

import importlib
import threading
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d


@dataclass(frozen=True)
class GemmReduceScatterConfig:
    """Collective launch parameters; select the same configuration on all ranks.

    Autotune exemption: independent libtuner trials can diverge across ranks and
    deadlock this producer/consumer protocol. The benchmark provides an opt-in
    coordinated sweep. Production calls never benchmark or change configuration.
    """

    block_m: int = 128
    block_n: int = 128
    block_k: int = 64
    group_m: int = 8
    num_warps: int = 8
    num_stages: int = 2
    epilogue_subtile: bool = False
    num_scatter_sms: int = 16
    num_reduction_sms: int = 15
    rs_block_m: int = 128
    rs_block_n: int = 64
    rs_num_warps: int = 4

    def validate(self):
        for name, value in asdict(self).items():
            if name != "epilogue_subtile" and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if type(self.epilogue_subtile) is not bool:
            raise ValueError("epilogue_subtile must be a bool")
        if self.block_m not in (64, 128, 256):
            raise ValueError("block_m must be 64, 128, or 256")
        if self.block_n not in (64, 128, 256):
            raise ValueError("block_n must be 64, 128, or 256")
        if self.block_k not in (32, 64, 128):
            raise ValueError("block_k must be 32, 64, or 128")
        if self.rs_block_m not in (128, 256) or self.rs_block_n not in (64, 128):
            raise ValueError("unsupported reduce-scatter tile")
        if self.num_warps not in (4, 8) or self.rs_num_warps not in (4, 8):
            raise ValueError("num_warps and rs_num_warps must be 4 or 8")

    def rs_config(self, reduce_op, world_size):
        return {
            "BLOCK_M": self.rs_block_m,
            "BLOCK_N": self.rs_block_n,
            "scatter_num_warps": self.rs_num_warps,
            "reduce_num_warps": self.rs_num_warps,
            "final_num_warps": self.rs_num_warps,
            "scale": 1.0 / world_size if reduce_op == "avg" else 1.0,
        }

    def estimated_shared_memory(self):
        staged = self.num_stages * (self.block_m + self.block_n) * self.block_k * 2
        output_n = self.block_n // 2 if self.epilogue_subtile else self.block_n
        return staged + self.block_m * output_n * 2 + 512


_DEFAULT_CONFIG = GemmReduceScatterConfig()
_HCU_DEFAULT_CONFIG = GemmReduceScatterConfig(
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=4,
    num_stages=1,
)
_CONFIGS = {}
_CONTEXTS = {}
_LOCK = threading.RLock()


def _get_backend():
    """HIP exposes torch.cuda tensors; select the actual compiler backend."""
    if torch.version.hip is None:
        return "nvidia"
    try:
        selected = importlib.import_module("triton._flagtree_backend").FLAGTREE_BACKEND
    except ImportError:
        selected = None
    if selected != "hcu":
        raise NotImplementedError(
            "HIP execution requires a FlagTree build with FLAGTREE_BACKEND=hcu"
        )
    return "hcu"


def _default_config():
    return _HCU_DEFAULT_CONFIG if _get_backend() == "hcu" else _DEFAULT_CONFIG


def _validate_device(device):
    backend = _get_backend()
    if backend == "nvidia" and torch.cuda.get_device_capability(device)[0] < 9:
        raise NotImplementedError(
            "the NVIDIA persistent TMA path requires SM90 or newer"
        )
    return backend


def _resolve_group(group_name):
    if not dist.is_initialized():
        raise RuntimeError("initialize torch.distributed before calling this operator")
    if not isinstance(group_name, str):
        raise TypeError("group_name must be a string")
    group = c10d._resolve_process_group(group_name)
    if group.rank() < 0:
        raise ValueError("the current rank is not a member of group_name")
    return group


def configure_fused_matmul_reduce_scatter(group_name, config):
    """Set launch parameters before first use, or after collective cache clearing.

    Call on every member of the group. Configuration agreement is checked when
    a context is created. No work or tuning is performed by this function.
    """
    if not isinstance(config, GemmReduceScatterConfig):
        raise TypeError("config must be a GemmReduceScatterConfig")
    config.validate()
    group = _resolve_group(group_name)
    with _LOCK:
        if any(key[0] is group for key in _CONTEXTS):
            raise RuntimeError("clear the group's context cache before reconfiguring")
        _CONFIGS[group] = config


def clear_fused_matmul_reduce_scatter_cache(group_name=None):
    """Collectively release cached FlagCX resources before destroying the group.

    Outputs own their storage and remain valid. This never destroys a PyTorch
    process group. Call on all ranks in matching order, outside graph capture.
    """
    group = _resolve_group(group_name) if group_name is not None else None
    with _LOCK:
        keys = [key for key in _CONTEXTS if group is None or key[0] is group]
        for key in keys:
            context = _CONTEXTS[key]
            context.close()
            del _CONTEXTS[key]


def _validate_operands(A, B, reduce_op, scatter_dim):
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise TypeError("A and B must be tensors")
    if A.ndim != 2 or B.ndim != 2:
        raise NotImplementedError("the TLE path currently supports 2D A and B")
    if reduce_op not in ("sum", "avg"):
        raise ValueError("reduce_op must be sum or avg")
    if scatter_dim != 0:
        raise NotImplementedError("the TLE path currently supports scatter_dim=0")
    if A.shape[1] != B.shape[0]:
        raise ValueError("A.shape[1] must equal B.shape[0]")
    if A.device != B.device or A.dtype != B.dtype:
        raise ValueError("A and B must have the same device and dtype")
    if A.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("the TLE path supports FP16 and BF16")
    if A.device.type != "cuda":
        raise NotImplementedError("the TLE path requires CUDA/HIP device tensors")
    if A.requires_grad or B.requires_grad:
        raise NotImplementedError("this operator supports inference only")
    if min(*A.shape, *B.shape) == 0:
        raise NotImplementedError("empty matrices are not supported by this path")


def _validate_shape(m, n, k, world_size, config):
    if world_size < 2:
        raise NotImplementedError("the TLE collective requires at least two ranks")
    if m % world_size:
        raise ValueError("M must be divisible by the process group size")
    if m // world_size < 256 or m // world_size % config.block_m:
        raise NotImplementedError("M/group_size must be >=256 and aligned to block_m")
    if n % config.block_n or n % config.rs_block_n or k % config.block_k:
        raise NotImplementedError(
            "N and local K must be aligned to the configured tiles"
        )


def _load_tle():
    try:
        return importlib.import_module(
            "flaggems_vllm.ops._fused_matmul_reduce_scatter_tle"
        )
    except ImportError as exc:
        raise NotImplementedError(
            "install FlagTree with TLE and a matching FlagCX device runtime"
        ) from exc


def fused_matmul_reduce_scatter(A, B, reduce_op, scatter_dim, group_name):
    """Return the rank-local shard of reduce(A @ B).

    Signature matches torch.ops.symm_mem.fused_matmul_reduce_scatter. Supports
    2D FP16/BF16 sum/avg along dimension 0 on NVIDIA SM90+ or experimental HCU,
    without autograd or graph
    capture. A is [M, local_K], B is [local_K, N], output is [M/group_size, N].
    B transpose packing and A contiguous copying are included in the call.
    Inputs, shape/configuration sequence, and process-group membership must agree
    across ranks. Rank order must contain equal, contiguous blocks of node ranks.
    """
    _validate_operands(A, B, reduce_op, scatter_dim)
    group = _resolve_group(group_name)
    config = _CONFIGS.get(group, _default_config())
    m, k = A.shape
    n = B.shape[1]
    _validate_shape(m, n, k, group.size(), config)
    with torch.cuda.device(A.device), _LOCK:
        backend = _validate_device(A.device)
        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError("CUDA graph capture is not supported yet")
        tle = _load_tle()
        from ._fused_matmul_reduce_scatter_runtime import CollectiveContext

        key = (group, A.device, A.dtype, m, n, k, config)
        context = _CONTEXTS.get(key)
        if context is None:
            context = CollectiveContext(group, A.device, A.dtype, m, n, k, config, tle)
            _CONTEXTS[key] = context
        current = torch.cuda.current_stream(A.device)
        if context.used:
            current.wait_event(context.done)
        a = A.contiguous()
        weight = B.transpose(0, 1).contiguous()
        if a.data_ptr() % 16:
            a = a.clone()
        if weight.data_ptr() % 16:
            weight = weight.clone()
        context.counters.zero_()

        descriptor_device = A.device

        def alloc_descriptor(size, alignment, stream):
            return torch.empty(size, dtype=torch.int8, device=descriptor_device)

        # HCU lowers descriptors to pointer accesses, requiring no CUDA TMA
        # descriptor allocator. Keep the existing NVIDIA path separate.
        if backend == "nvidia":
            tle.triton.set_allocator(alloc_descriptor)
        context.rs_stream.wait_stream(current)
        output = torch.empty((m // group.size(), n), device=A.device, dtype=A.dtype)
        tle.gemm_rs_producer_persistent(
            a,
            weight,
            context.gemm_out,
            context.rs.scatter_signal_buf,
            context.counters,
            group.size(),
            context.rs.local_world_size,
            group.rank(),
            context.num_gemm_sms,
            BLOCK_SIZE_M=config.block_m,
            BLOCK_SIZE_N=config.block_n,
            BLOCK_SIZE_K=config.block_k,
            GROUP_SIZE_M=config.group_m,
            STAGES=config.num_stages,
            NUM_WARPS=config.num_warps,
            EPILOGUE_SUBTILE=config.epilogue_subtile,
        )
        with torch.cuda.stream(context.rs_stream):
            tle.reduce_scatter_2d_op(
                context.gemm_out,
                context.rs,
                output=output,
                ready_flags=context.rs.scatter_signal_buf,
                config=config.rs_config(reduce_op, group.size()),
            )
        current.wait_stream(context.rs_stream)
        context.done.record(current)
        context.used = True
        return output
