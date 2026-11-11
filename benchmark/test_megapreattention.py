# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Paired full-window benchmark in the Hygon vLLM application image.

Baseline uses vLLM fused postprocessing with the Hygon norm precision adapter
and the official FL cache update. Default AG/GEMM uses RCCL and vLLM BF16 linear;
MPA_BASELINE_AG=dushmem selects the NV-style two-stream peer-copy/BLAS
decomposition. Source hashes, selected paths, and differences are recorded;
no cross-hardware NV-equivalent speedup is inferred.
"""
import ctypes
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import flaggems_vllm
from tests.test_megapreattention import Output, run_distributed

from . import base
from .conftest import Config

if flaggems_vllm.vendor_name != "hygon":
    pytest.skip("MegaPreAttention benchmark requires Hygon", allow_module_level=True)
else:
    from triton.language.extra.hip import libdevice
    from vllm.triton_utils import tl, triton


# The fused postprocessor below is derived from vLLM 0.24.0+flagos.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Original fused_qk_norm_rope.py SHA256:
# 3703af53dfef0c7e53035da2bee03ecada7fbd7698b40b05d6fd1ef9cf5077a6
# Only Hygon BF16/D256 norm reduction and scalar rsqrt rounding are adapted.
@triton.jit
def _hygon_mean_square_256(x):
    base = tl.arange(0, 256) & ~3
    squared = x * x
    values = tl.gather(squared, base, 0) + tl.gather(squared, base + 1, 0)
    values = values + tl.gather(squared, base + 2, 0)
    values = values + tl.gather(squared, base + 3, 0)
    for bit in tl.static_range(2, 8):
        values = values + tl.gather(values, tl.arange(0, 256) ^ (1 << bit), 0)
    return tl.sum(tl.gather(values, tl.full((1,), 0, tl.int32), 0), 0) / 256.0


@triton.jit
def _fused_qk_rmsnorm_rope_gate_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_gate_stride_t,
    k_stride_t,
    q_out_stride_t,
    k_out_stride_t,
    gate_out_stride_t,
    cache_stride_p,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    eps: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)

    if is_k:
        in_base = k_ptr + token * k_stride_t + local_head * head_dim
        w_ptr = k_weight_ptr
        out_base = k_out_ptr + token * k_out_stride_t + local_head * head_dim
    else:
        in_base = q_gate_ptr + token * q_gate_stride_t + local_head * 2 * head_dim
        w_ptr = q_weight_ptr
        out_base = q_out_ptr + token * q_out_stride_t + local_head * head_dim

    # --- RMSNorm: variance over the full head_dim ---
    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < head_dim
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    if head_dim == 256 and INPUT_DTYPE == tl.bfloat16:
        var = _hygon_mean_square_256(x)
        inv_rms = libdevice.rsqrt((var + eps).to(tl.float64)).to(tl.float32)
    else:
        var = tl.sum(x * x, axis=0) / head_dim
        inv_rms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    # Round-trip through INPUT_DTYPE so the RoPE input matches the bf16-storage
    # behavior of the unfused (qk_rmsnorm -> memory -> apply_rope) reference path.
    x_norm = (x * inv_rms * w).to(INPUT_DTYPE).to(tl.float32)

    # --- Pass-through tail [rotary_dim, head_dim): RMSNorm-only, no rotation ---
    # The rotary head [0, rotary_dim) will be overwritten by the RoPE store below.
    if HAS_PASS:
        pass_mask = head_mask & (head_offs >= rotary_dim)
        tl.store(out_base + head_offs, x_norm, mask=pass_mask)

    # --- Partial RoPE on the first rotary_dim elements ---
    # Triton lacks easy sub-vector slicing of x_norm, so we recompute the
    # normalized rotary halves on a smaller block (next_pow2(half_rotary)).
    # The extra ~rotary_dim element reload hits L1, so the cost is negligible.
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < half_rotary
    x_rot1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    x_rot2 = tl.load(in_base + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    w_rot1 = tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    w_rot2 = tl.load(w_ptr + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    x_rot1 = (x_rot1 * inv_rms * w_rot1).to(INPUT_DTYPE).to(tl.float32)
    x_rot2 = (x_rot2 * inv_rms * w_rot2).to(INPUT_DTYPE).to(tl.float32)

    # Always use int64 for position to avoid overflow in address computation.
    pos = tl.load(positions_ptr + token).to(tl.int64)
    cache_offset = pos * cache_stride_p
    cos = tl.load(
        cos_sin_cache_ptr + cache_offset + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + cache_offset + half_rotary + rot_offs,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)

    o1 = x_rot1 * cos - x_rot2 * sin
    o2 = x_rot2 * cos + x_rot1 * sin
    tl.store(out_base + rot_offs, o1, mask=rot_mask)
    tl.store(out_base + half_rotary + rot_offs, o2, mask=rot_mask)

    # --- Gate copy (q heads only, verbatim) ---
    if not is_k:
        gate_in_base = in_base + head_dim
        gate_out_base = gate_out_ptr + token * gate_out_stride_t + local_head * head_dim
        g = tl.load(gate_in_base + head_offs, mask=head_mask, other=0.0)
        tl.store(gate_out_base + head_offs, g, mask=head_mask)


def fused_qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused split + QK-RMSNorm + (partial) RoPE + gate copy for Qwen3.5 attn.

    Args:
        q_gate: (n_tokens, num_q_heads * 2 * head_dim) -- per head: [q|gate]
        k: (n_tokens, num_kv_heads * head_dim)
        q_weight: (head_dim,) GemmaRMSNorm effective weight (already +1)
        k_weight: (head_dim,) GemmaRMSNorm effective weight (already +1)
        cos_sin_cache: (max_pos, rotary_dim) packed [cos|sin]
        positions: (n_tokens,) int32 or int64
        eps: RMSNorm epsilon
        num_q_heads: number of Q heads (after TP split)
        num_kv_heads: number of KV heads (after TP split)
        head_dim: per-head dimension
        rotary_dim: rotary dimension; must be even and <= head_dim

    Returns:
        (q_out, k_out, gate_out) -- all contiguous (n_tokens, heads * head_dim).
        ``gate_out`` is the raw (pre-sigmoid) gate.
    """
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            f"rotary_dim must be a positive even integer <= head_dim, "
            f"got rotary_dim={rotary_dim}, head_dim={head_dim}"
        )

    n_tokens = q_gate.shape[0]
    q_out = torch.empty(
        (n_tokens, num_q_heads * head_dim), dtype=q_gate.dtype, device=q_gate.device
    )
    k_out = torch.empty(
        (n_tokens, num_kv_heads * head_dim), dtype=k.dtype, device=k.device
    )
    gate_out = torch.empty_like(q_out)
    if n_tokens == 0:
        return q_out, k_out, gate_out

    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rot_half_block = triton.next_power_of_2(half_rotary)
    num_warps = max(1, head_block // 64)

    grid = (n_tokens, num_q_heads + num_kv_heads)
    _fused_qk_rmsnorm_rope_gate_kernel[grid](
        q_gate,
        k,
        q_out,
        k_out,
        gate_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        gate_out.stride(0),
        cos_sin_cache.stride(0),
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        half_rotary,
        eps,
        INPUT_DTYPE=tl.bfloat16 if q_gate.dtype == torch.bfloat16 else tl.float16,
        HEAD_BLOCK=head_block,
        ROT_HALF_BLOCK=rot_half_block,
        HAS_PASS=rotary_dim < head_dim,
        num_warps=num_warps,
        num_stages=2,
    )
    return q_out, k_out, gate_out


_SESSION = None


def _library():
    source = (
        Path(__file__).resolve().parents[1] / "tests" / "megapreattention_dushmem.hip"
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = Path(tempfile.gettempdir()) / f"mpa-dushmem-{digest[:16]}.so"
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            subprocess.run(
                [
                    "/opt/dtk/bin/hipcc",
                    "-shared",
                    "-fPIC",
                    "-fgpu-rdc",
                    "-mcode-object-version=4",
                    "-O2",
                    "--offload-arch=gfx936",
                    "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
                    "-I/opt/dtk/include/dushmem",
                    "-L/opt/dtk/lib/dushmem",
                    "-Wl,-rpath,/opt/dtk/lib/dushmem",
                    str(source),
                    "-ldushmem_host",
                    "-ldushmem_device",
                    "-o",
                    str(output),
                ],
                check=True,
                timeout=120,
                capture_output=False,
                text=True,
            )
    lib = ctypes.CDLL(str(output))
    signatures = {
        "mpa_uid_size": ([], ctypes.c_size_t),
        "mpa_uid": ([ctypes.c_void_p], ctypes.c_int),
        "mpa_init": ([ctypes.c_int] * 3 + [ctypes.c_void_p], ctypes.c_int),
        "mpa_rank": ([], ctypes.c_int),
        "mpa_world": ([], ctypes.c_int),
        "mpa_local_world": ([], ctypes.c_int),
        "mpa_alloc": ([ctypes.c_size_t], ctypes.c_void_p),
        "mpa_peer": ([ctypes.c_void_p, ctypes.c_int], ctypes.c_void_p),
        "mpa_copy": (
            [ctypes.c_void_p] * 2 + [ctypes.c_size_t, ctypes.c_void_p],
            ctypes.c_int,
        ),
        "mpa_barrier": ([ctypes.c_void_p], ctypes.c_int),
        "mpa_free": ([ctypes.c_void_p], None),
        "mpa_finalize": ([], None),
    }
    for name, (args, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    return lib, digest


def _check(status, operation):
    if status:
        raise RuntimeError(f"duSHMEM/HIP {operation} failed: {status}")


class _Session:
    def __init__(self, inputs):
        if inputs.group is not dist.group.WORLD:
            raise NotImplementedError("baseline duSHMEM requires the full WORLD group")
        self.rank = dist.get_rank(inputs.group)
        self.world = dist.get_world_size(inputs.group)
        self.control = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
        if os.environ.get("DUSHMEM_REMOTE_TRANSPORT", "none") != "none":
            raise ValueError(
                "same-node baseline requires DUSHMEM_REMOTE_TRANSPORT=none"
            )
        os.environ.setdefault("DUSHMEM_REMOTE_TRANSPORT", "none")
        os.environ.setdefault("DUSHMEM_IB_ENABLE_IBGDA", "false")
        os.environ.setdefault("DUSHMEM_SYMMETRIC_SIZE", "512M")
        os.environ.setdefault("DUSHMEM_DEBUG", "WARN")
        self.lib, self.source_sha256 = _library()
        size = self.lib.mpa_uid_size()
        if not 0 < size <= 4096:
            raise RuntimeError(f"invalid duSHMEM unique ID size: {size}")
        uid = ctypes.create_string_buffer(size)
        error = None
        if self.rank == 0:
            status = self.lib.mpa_uid(uid)
            error = f"unique ID error {status}" if status else None
        state = [error, bytes(uid)]
        dist.broadcast_object_list(state, src=0, group=self.control)
        if state[0]:
            raise RuntimeError(state[0])
        uid.raw = state[1]
        print(self.rank, "duSHMEM init begin", flush=True)
        _check(
            self.lib.mpa_init(self.rank, self.world, inputs.hidden.device.index, uid),
            "init",
        )
        if (self.lib.mpa_rank(), self.lib.mpa_world(), self.lib.mpa_local_world()) != (
            self.rank,
            self.world,
            self.world,
        ):
            raise RuntimeError("duSHMEM rank/world/same-node identity mismatch")
        self.active = 0
        print(self.rank, "duSHMEM init ready", flush=True)


def close_vendor_session():
    global _SESSION
    if _SESSION is not None:
        if _SESSION.active:
            raise RuntimeError("duSHMEM allocations are still live")
        torch.cuda.synchronize()
        dist.barrier(group=_SESSION.control)
        _SESSION.lib.mpa_finalize()
        dist.destroy_process_group(_SESSION.control)
        _SESSION = None


class VendorAllGatherMatmul:
    def __init__(self, inputs):
        global _SESSION
        if _SESSION is None:
            _SESSION = _Session(inputs)
        self.session = _SESSION
        self.rank, self.world = self.session.rank, self.session.world
        self.lib = self.session.lib
        self.nbytes = inputs.hidden.numel() * inputs.hidden.element_size()
        print(self.rank, "duSHMEM allocation begin", self.nbytes, flush=True)
        self.allocation = self.lib.mpa_alloc(self.nbytes)
        if not self.allocation:
            raise RuntimeError("duSHMEM allocation returned null")
        self.session.active += 1
        self.peer_ptrs = [
            self.lib.mpa_peer(self.allocation, r) for r in range(self.world)
        ]
        if not all(self.peer_ptrs):
            raise RuntimeError("duSHMEM did not establish every same-node P2P mapping")
        dist.barrier(group=self.session.control)
        self.stream = torch.cuda.Stream(device=inputs.hidden.device)
        self.gathered = torch.empty(
            (inputs.hidden.shape[0] * self.world, inputs.hidden.shape[1]),
            dtype=inputs.hidden.dtype,
            device=inputs.hidden.device,
        )
        self.products = torch.empty(
            (self.gathered.shape[0], inputs.weight.shape[0]),
            dtype=inputs.hidden.dtype,
            device=inputs.hidden.device,
        )
        self.shards = self.gathered.chunk(self.world)
        self.out_shards = self.products.chunk(self.world)
        self.evidence = {
            "baseline_kind": "vllm-fl-vendor-dushmem-pipelined",
            "communication_gemm": "duSHMEM peer mapping + HIP P2P copy + ATen HIP BLAS, two-stream rank pipeline",
            "overlaps_communication_gemm": True,
            "overlap_evidence": "GPU timeline verified: 27B TP2/32K; actual overlap amount is shape-dependent",
            "nv_schedule": (
                "torch2.11 _pipelined_multi_all_gather_and_consume; "
                "local publish, barrier, alternating copy/GEMM, join, barrier"
            ),
            "vendor_shim_sha256": self.session.source_sha256,
            "custom_gpu_compute_or_sync_kernel": False,
            "python_epoch": False,
            "duSHMEM_heap_bytes": os.environ["DUSHMEM_SYMMETRIC_SIZE"],
        }
        print(self.rank, "duSHMEM mappings ready", flush=True)

    def _barrier(self, stream):
        _check(self.lib.mpa_barrier(stream.cuda_stream), "stream barrier")

    def _copy(self, dst, src, stream):
        _check(self.lib.mpa_copy(dst, src, self.nbytes, stream.cuda_stream), "copy")

    def __call__(self, inputs):
        main = torch.cuda.current_stream(inputs.hidden.device)
        self._barrier(main)
        self._copy(self.allocation, inputs.hidden.data_ptr(), main)
        self._barrier(main)
        self.stream.wait_stream(main)
        weight_t = inputs.weight.t()
        torch.mm(inputs.hidden, weight_t, out=self.out_shards[self.rank])
        for step in range(1, self.world):
            stream = self.stream if step % 2 else main
            peer = (self.rank + step) % self.world
            with torch.cuda.stream(stream):
                self._copy(self.shards[peer].data_ptr(), self.peer_ptrs[peer], stream)
                torch.mm(self.shards[peer], weight_t, out=self.out_shards[peer])
        # Match NV's returned gathered A, even though the projection only needs C.
        local_stream = self.stream if self.world % 2 == 0 else main
        with torch.cuda.stream(local_stream):
            self._copy(
                self.shards[self.rank].data_ptr(),
                inputs.hidden.data_ptr(),
                local_stream,
            )
        main.wait_stream(self.stream)
        self._barrier(main)
        return self.gathered, self.products

    def close(self):
        if self.allocation:
            torch.cuda.synchronize()
            dist.barrier(group=self.session.control)
            self.lib.mpa_free(self.allocation)
            self.allocation = None
            self.session.active -= 1


def _compiled_method_bridge(method):
    """Keep the precision-adapted vLLM Triton kernel opaque to torch2.9 functionalization.

    This FlagTree3.6 image removed specialize_impl, which torch2.9 uses to
    identify Triton output mutations. Direct tracing silently returns wrong
    outputs here. A custom-op boundary preserves the fused kernel call;
    Graph replay adds no GPU kernel and no Python callback.
    """
    import sys
    from types import FunctionType, ModuleType

    original = fused_qk_rmsnorm_rope_gate

    @torch.library.custom_op(
        "megapreattention_benchmark::vllm_qk_rope", mutates_args=()
    )
    def wrapped(
        q_gate: torch.Tensor,
        key: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        rope: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        rotary_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return original(
            q_gate,
            key,
            q_weight,
            k_weight,
            rope,
            positions,
            eps,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
        )

    @wrapped.register_fake
    def fake(
        q_gate,
        key,
        q_weight,
        k_weight,
        rope,
        positions,
        eps,
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
    ):
        q = torch.empty(
            (q_gate.shape[0], q_heads * head_dim),
            device=q_gate.device,
            dtype=q_gate.dtype,
        )
        k = torch.empty(
            (key.shape[0], kv_heads * head_dim), device=key.device, dtype=key.dtype
        )
        return q, k, torch.empty_like(q)

    # Clone only this function's globals. Do not patch installed vLLM, torch,
    # FlagTree, or the process-wide model class.
    module_name = "megapreattention_benchmark_vllm_bridge"
    module = ModuleType(module_name)
    module.__dict__.update(method.__globals__)
    module.__dict__.update(__name__=module_name, fused_qk_rmsnorm_rope_gate=wrapped)
    sys.modules[module_name] = module
    bridge = FunctionType(
        method.__code__,
        module.__dict__,
        method.__name__,
        method.__defaults__,
        method.__closure__,
    )
    module._project_qkv_gate = bridge
    return bridge


_BRIDGED_METHOD = None


def _baseline_policy():
    requested = os.environ.get("MPA_BASELINE_AG", "vllm-rccl")
    if requested not in ("vllm-rccl", "serial-rccl", "dushmem"):
        raise ValueError(f"unknown MPA_BASELINE_AG={requested!r}")
    return (
        requested,
        os.environ.get("MPA_BASELINE_BLAS", "rocblas"),
        {
            "baseline_policy": (
                "vllm-rccl-norm-precision-v2"
                if requested == "vllm-rccl"
                else "explicit-diagnostic-override"
            ),
            "candidate_used_for_baseline_selection": False,
            "baseline_selection_rule": "one high-level platform path for all cases",
        },
    )


class NativeWindow:
    """Library AG/GEMM + vLLM projection method + precision-adapted post + FL cache.

    The text baseline enables vLLM's own fast fused norm/RoPE/gate path.
    The fixed manifest has identical T/H/W position rows; edge layouts are
    accuracy-only and are rejected by the baseline runner.
    """

    def __init__(self, inputs):
        ag_kind, blas, selection = _baseline_policy()
        if blas not in ("rocblas", "hipblaslt"):
            raise ValueError(f"unknown MPA_BASELINE_BLAS={blas!r}")
        torch.backends.cuda.preferred_blas_library(
            "cublaslt" if blas == "hipblaslt" else "cublas"
        )
        rope = SimpleNamespace(
            cos_sin_cache=inputs.rope, rotary_dim=inputs.config.rotary_dim
        )
        edge = False
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm
        from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

        qnorm = SimpleNamespace(
            weight=inputs.q_norm, variance_epsilon=inputs.config.eps
        )
        knorm = SimpleNamespace(
            weight=inputs.k_norm, variance_epsilon=inputs.config.eps
        )
        qnorm_call = lambda x: GemmaRMSNorm.forward_native(qnorm, x)
        knorm_call = lambda x: GemmaRMSNorm.forward_native(knorm, x)
        # The projection method reads .weight even for its fused fast path.
        qnorm_call.weight, qnorm_call.variance_epsilon = (
            inputs.q_norm,
            inputs.config.eps,
        )
        knorm_call.weight, knorm_call.variance_epsilon = (
            inputs.k_norm,
            inputs.config.eps,
        )
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        from vllm.v1.attention.backend import AttentionType
        from vllm_fl.dispatch.backends.flaggems.impl.attention import AttentionFLImpl

        self.linear_method = UnquantizedLinearMethod()
        self.linear_layer = SimpleNamespace(weight=inputs.weight)
        self.cache_method = AttentionFLImpl.do_kv_cache_update
        self.cache_context = SimpleNamespace(
            attn_type=AttentionType.DECODER, kv_cache_dtype="auto"
        )
        self.cache_layer = SimpleNamespace(_k_scale=inputs.scale, _v_scale=inputs.scale)
        self.context = SimpleNamespace(
            use_fused_qk_norm_rope_gate=not edge,
            attn_output_gate=True,
            q_size=inputs.config.q_size,
            kv_size=inputs.config.kv_size,
            num_heads=inputs.config.num_q_heads,
            num_kv_heads=inputs.config.num_kv_heads,
            head_dim=inputs.config.head_dim,
            q_norm=qnorm_call,
            k_norm=knorm_call,
            rotary_emb=rope,
        )
        global _BRIDGED_METHOD
        if _BRIDGED_METHOD is None:
            _BRIDGED_METHOD = _compiled_method_bridge(
                Qwen3NextAttention._project_qkv_gate
            )
        self.method = _BRIDGED_METHOD
        # Official FL attention backend uses this cache implementation because
        # the application wheel has no vllm._C_cache_ops binary.
        from flag_gems import reshape_and_cache_flash

        self.cache_op = reshape_and_cache_flash
        self.scale = inputs.scale
        self.group_name = inputs.group.group_name
        self.post = torch.compile(self._project, fullgraph=True, dynamic=False)
        blocks, heads, block_size, _ = inputs.cache.shape
        self.kv_cache = torch.empty(
            (2, blocks, block_size, heads, inputs.config.head_dim),
            device=inputs.cache.device,
            dtype=inputs.cache.dtype,
        )
        self.key_cache, self.value_cache = self.kv_cache.unbind(0)
        self.reset_cache(inputs)
        self.ag = None
        if ag_kind == "dushmem":
            self.ag = VendorAllGatherMatmul(inputs)
        self.evidence = {
            "norm_precision": {
                "version": "hygon-bf16-d256-fp32-tree-fp64-scalar-rsqrt-v1",
                "source_file": __file__,
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "vendor_original_sha256": (
                    "3703af53dfef0c7e53035da2bee03ecada7fbd7698b40b05d6fd1ef9cf5077a6"
                ),
                "installed_package_modified": False,
                "extra_kernel_launches": 0,
            },
            "triton_compile_compatibility": (
                "test-only custom_op boundary around precision-adapted vLLM fused kernel; original "
                "method bytecode/globals clone; no installed package patch"
            ),
            "bridge_source_sha256": hashlib.sha256(
                inspect.getsource(_compiled_method_bridge).encode()
            ).hexdigest(),
            "baseline_kind": "vllm-high-level-rccl-linear-fl-cache",
            "linear_entry": "vllm.UnquantizedLinearMethod.apply",
            "linear_source_file": inspect.getfile(UnquantizedLinearMethod),
            "linear_source_sha256": hashlib.sha256(
                inspect.getsource(UnquantizedLinearMethod.apply).encode()
            ).hexdigest(),
            "cache_backend_entry": "AttentionFLImpl.do_kv_cache_update",
            "vendor_coverage": {
                "communication": "RCCL through torch.distributed",
                "gemm": "vLLM BF16 linear -> HIP BLAS; verify GPU trace",
                "qk_norm_rope": "vLLM-derived fused Triton kernel, Hygon BF16/D256 norm precision fix",
                "cache": "official FL backend / FlagGems; no Hygon-specific binary in image",
            },
            "vllm_version": importlib.metadata.version("vllm"),
            "plugin_version": importlib.metadata.version("vllm-plugin-fl"),
            "torch_version": torch.__version__,
            "blas_requested": blas,
            "blas_preference": str(torch.backends.cuda.preferred_blas_library()),
            "overlaps_communication_gemm": False,
            "comparable_to_nv_native_baseline": False,
            "symm_mem_status": (
                "unpatched FD-pointer import fails; process-local FD-value fix passes mapping "
                "but fused call still times out; not accepted for baseline"
            ),
            "cache_source_sha256": hashlib.sha256(
                inspect.getsource(self.cache_op).encode()
            ).hexdigest(),
            "cache_source_file": inspect.getfile(self.cache_op),
            "projection_method_sha256": hashlib.sha256(
                inspect.getsource(self.method).encode()
            ).hexdigest(),
            "projection_file": inspect.getfile(self.method),
            "text_fused_path": not edge,
            "cache_op": "flag_gems.reshape_and_cache_flash (official vllm_fl attention backend)",
            "communication_gemm": "RCCL all_gather_into_tensor then vLLM UnquantizedLinearMethod.apply; serial",
            "compile": (
                "fullgraph=True,dynamic=False; original vLLM projection method; library cache"
                " launched separately"
            ),
            "cache_layout": (
                "separate contiguous [blocks,block_size,heads,D]; same logical values as "
                "candidate; normalization only outside timing"
            ),
            "cache_layout_reason": (
                "installed FL cache kernel supports dense in-block layout, not NV adapter "
                "combined/strided cache views"
            ),
            "cache_stride_adapter_timed": False,
        }

        if self.ag is not None:
            self.evidence.update(self.ag.evidence)
        self.evidence.update(selection)
        self.evidence["baseline_ag_selected"] = ag_kind

    def close(self):
        if self.ag is not None:
            self.ag.close()

    def _project(self, qkv, positions):
        return self.method(self.context, qkv, positions)

    def reset_cache(self, inputs):
        d = self.context.head_dim
        self.key_cache.copy_(inputs.cache[..., :d].transpose(1, 2))
        self.value_cache.copy_(inputs.cache[..., d:].transpose(1, 2))

    def cache_for_check(self, inputs):
        # Logical cache normalization belongs only to accuracy checks, never
        # the measured/captured implementation.
        return torch.cat(
            (self.key_cache.transpose(1, 2), self.value_cache.transpose(1, 2)), dim=-1
        )

    def __call__(self, inputs):
        if self.ag is None:
            gathered = torch.empty(
                (inputs.positions.shape[1], inputs.hidden.shape[1]),
                device=inputs.hidden.device,
                dtype=inputs.hidden.dtype,
            )
            dist.all_gather_into_tensor(gathered, inputs.hidden, group=inputs.group)
            product = self.linear_method.apply(self.linear_layer, gathered)
        else:
            gathered, product = self.ag(inputs)
        q, k, v, gate = self.post(product, inputs.positions)
        self.cache_method(
            self.cache_context,
            self.cache_layer,
            k.view(-1, self.context.num_kv_heads, self.context.head_dim),
            v.view(-1, self.context.num_kv_heads, self.context.head_dim),
            self.kv_cache,
            inputs.slots,
        )
        return Output(
            q,
            k,
            v,
            gate.view(q.shape[0], self.context.num_heads, self.context.head_dim),
            (gathered, product),
        )


class MegaPreAttentionBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))

    def __init__(self):
        super().__init__("megapreattention", self.native, gems_op=self.candidate)
        self.paired_results = {}

    @staticmethod
    def native():
        """RCCL/rocBLAS/compiled-post baseline label."""

    @staticmethod
    def candidate():
        """Hygon candidate label."""

    def get_input_iter(self, dtype):
        yield from self.shapes

    def get_latency(
        self, op, tokens, tp, hidden, q_heads, kv_heads, head_dim, rotary_dim
    ):
        if not torch.version.hip or torch.cuda.device_count() < tp:
            pytest.skip(f"requires {tp} Hygon GPUs")
        root = os.environ.get(
            "MEGAPREATTENTION_DATA",
            "/data/lixichen/megapreattention_data/layer3_nv_20261027_v1",
        )
        if not Path(root, "manifest.json").exists():
            pytest.skip("layer3 dataset unavailable")
        dims = (hidden, q_heads, kv_heads)
        models = {(2048, 16, 2): "Qwen3.6-35B-A3B", (5120, 24, 4): "Qwen3.6-27B"}
        if dims not in models or (head_dim, rotary_dim) != (256, 64):
            raise NotImplementedError(
                "benchmark dataset covers only the two D256/R64 models"
            )
        case_id = f"{models[dims]}_tp{tp}_t{tokens}"
        mode = Config.mode.value
        if mode not in ("kernel", "operator", "cudagraph"):
            raise ValueError("use kernel, operator or cudagraph mode")
        key = (case_id, mode, Config.warm_up, Config.repetition)
        if key not in self.paired_results:
            self.paired_results[key] = run_distributed(
                tp,
                [case_id],
                root=root,
                mode=mode,
                check_only=False,
                warmup=Config.warm_up,
                iterations=Config.repetition,
                rounds=5,
            )
        name = "native" if op == self.native else "megapreattention"
        return self.paired_results[key]["summary"][case_id][name]


@pytest.mark.megapreattention
def test_megapreattention():
    benchmark = MegaPreAttentionBenchmark()
    benchmark.run()
    if Config.record_json:
        samples = {key[0]: value for key, value in benchmark.paired_results.items()}
        Path(Config.output).with_suffix(".samples.json").write_text(
            json.dumps(samples, indent=2) + "\n"
        )
