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
"""Single-file FlagGems-vLLM MambaMixer benchmark.

Like other operators in benchmark/, one pytest test loops through predefined
shapes and base.Benchmark.run() renders the project's standard result table.

Shape = (batch, seq_len, hidden_size, intermediate_size).
For this Mamba-1 implementation, seq_len == 1 means single-token decode;
seq_len > 1 means pure prefill. These paths share ONE performance table.

A real vLLM 0.24.0 MambaMixer.forward_impl is used as the reference, with
TP=1 F.linear adapters for the projection modules. Native vLLM conv/SSM
kernels are retained. The candidate is FlagGems-vllm's Triton MambaMixer.
The test measures the entire mixer, NOT an end-to-end model runner.

Stateful kernels update caches in place: each measurement restores both Conv
and SSM states BEFORE the timed interval. This requires a custom get_latency
method; default base.Benchmark.get_latency would benchmark mutated states.

The benchmark covers torch.float16, torch.bfloat16, torch.float32. The
FlagGems MambaMixer must also support FP32. Use the accompanying FP32 operator
patch; registering a dtype in the benchmark alone is not sufficient.

Run from repo root:
    pytest -s benchmark/test_mamba_mixer_perf.py \
        --level core --warmup 100 --iter 500 --mode kernel
    # Optional: isolate FP32 after enabling the FP32 FlagGems MambaMixer:
    pytest -s benchmark/test_mamba_mixer_perf.py \
        --level core --dtypes float32 --warmup 100 --iter 500 --mode kernel

Optional: MAMBA_BENCH_GPU=0, MAMBA_BENCH_STATE_SIZE=16,
          MAMBA_BENCH_DT_RANK=48, MAMBA_BENCH_KERNEL_SIZE=4,
          MAMBA_BENCH_RMS_NORM=0, MAMBA_BENCH_BIAS=0
"""

from __future__ import annotations

import contextlib
import importlib
import math
import os
import statistics
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import base, consts

@dataclass
class CachePair:
    conv_storage: torch.Tensor
    ssm_state: torch.Tensor
    initial_conv_storage: torch.Tensor
    initial_ssm_state: torch.Tensor
    conv_state_dim_first: bool

    @property
    def conv_state(self) -> torch.Tensor:
        if self.conv_state_dim_first:
            return self.conv_storage
        return self.conv_storage.transpose(-1, -2)

    def reset(self) -> None:
        self.conv_storage.copy_(self.initial_conv_storage)
        self.ssm_state.copy_(self.initial_ssm_state)


class _LinearTuple(nn.Module):
    """TP=1 stand-in for vLLM parallel linear modules.

    vLLM Linear layers return (output, deferred_bias). For dt_proj,
    skip_bias_add=True means the bias is returned separately and applied by
    the SSM kernel. Other projections add bias in the linear operation.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        skip_bias_add: bool = False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone())
        self.skip_bias_add = skip_bias_add

    def forward(self, x: torch.Tensor):
        if self.skip_bias_add:
            return F.linear(x, self.weight, None), self.bias
        return F.linear(x, self.weight, self.bias), None


class _ConvWeights(nn.Module):
    def __init__(
        self, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone())


@dataclass
class BenchmarkLayer:
    layer: nn.Module
    mixer_module: object
    mamba_ssm_module: object
    weights: dict[str, torch.Tensor | None]
    prefix: str
    use_rms_norm: bool
    rms_norm_eps: float


def _randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    return (
        torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        .mul_(scale)
        .to(dtype)
    )


def build_vllm_mamba_layer(
    *,
    hidden_size: int,
    intermediate_size: int,
    state_size: int,
    time_step_rank: int,
    kernel_size: int,
    dtype: torch.dtype,
    device: torch.device,
    use_rms_norm: bool = False,
    rms_norm_eps: float = 1e-5,
    use_bias: bool = False,
    use_conv_bias: bool = True,
    seed: int = 2026,
) -> BenchmarkLayer:
    """Build an actual MambaMixer instance and attach TP=1 benchmark modules.

    The benchmark invokes the installed vLLM MambaMixer.forward_impl method.
    Only the four projection modules are lightweight F.linear adapters;
    vLLM's native Mamba convolution and SSM kernels are used.
    """
    mixer_module = importlib.import_module(
        "vllm.model_executor.layers.mamba.mamba_mixer"
    )
    mamba_ssm_module = importlib.import_module(
        "vllm.model_executor.layers.mamba.ops.mamba_ssm"
    )
    norm_module = importlib.import_module(
        "vllm.model_executor.layers.layernorm"
    )
    MambaMixer = mixer_module.MambaMixer
    RMSNorm = norm_module.RMSNorm

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    in_proj_weight = _randn(
        (2 * intermediate_size, hidden_size),
        device=device, dtype=dtype, scale=1.0 / math.sqrt(hidden_size),
        generator=generator,
    )
    in_proj_bias = (
        _randn(
            (2 * intermediate_size,), device=device, dtype=dtype,
            scale=0.02, generator=generator,
        )
        if use_bias else None
    )
    conv1d_weight = _randn(
        (intermediate_size, 1, kernel_size),
        device=device, dtype=dtype, scale=0.15, generator=generator,
    )
    conv1d_bias = (
        _randn(
            (intermediate_size,), device=device, dtype=dtype,
            scale=0.02, generator=generator,
        )
        if use_conv_bias else None
    )
    x_proj_weight = _randn(
        (time_step_rank + 2 * state_size, intermediate_size),
        device=device, dtype=dtype,
        scale=1.0 / math.sqrt(intermediate_size), generator=generator,
    )
    dt_proj_weight = _randn(
        (intermediate_size, time_step_rank),
        device=device, dtype=dtype,
        scale=1.0 / math.sqrt(time_step_rank), generator=generator,
    )
    dt_proj_bias = _randn(
        (intermediate_size,), device=device, dtype=dtype,
        scale=0.05, generator=generator,
    )
    A = -torch.exp(
        _randn(
            (intermediate_size, state_size), device=device,
            dtype=torch.float32, scale=0.1, generator=generator,
        )
    )
    D = torch.ones(
        (intermediate_size,), device=device, dtype=torch.float32
    )
    out_proj_weight = _randn(
        (hidden_size, intermediate_size),
        device=device, dtype=dtype,
        scale=1.0 / math.sqrt(intermediate_size), generator=generator,
    )
    out_proj_bias = (
        _randn(
            (hidden_size,), device=device, dtype=dtype,
            scale=0.02, generator=generator,
        )
        if use_bias else None
    )

    # Construct the vLLM layer object without invoking its distributed/model
    # constructor. Its actual forward_impl and _ssm_transform methods are used.
    layer = MambaMixer.__new__(MambaMixer)
    nn.Module.__init__(layer)

    prefix = "benchmark.mamba_mixer"
    layer.time_step_rank = time_step_rank
    layer.ssm_state_size = state_size
    layer.use_rms_norm = use_rms_norm
    layer.activation = "silu"
    layer.is_lora_enabled = False
    layer.conv_kernel_size = kernel_size
    layer.intermediate_size = intermediate_size
    layer.prefix = prefix
    # vLLM 0.24.0 forward_impl reads these two fields. "none" means one
    # state slot per sequence and no prefix-cache block-table handling.
    layer.cache_config = SimpleNamespace(
        mamba_block_size=256,
        mamba_cache_mode="none",
    )
    layer.model_config = None
    layer.kv_cache = (torch.empty(0, device=device), torch.empty(0, device=device))

    layer.in_proj = _LinearTuple(in_proj_weight, in_proj_bias)
    layer.conv1d = _ConvWeights(conv1d_weight, conv1d_bias)
    layer.x_proj = _LinearTuple(x_proj_weight, None)
    layer.dt_proj = _LinearTuple(
        dt_proj_weight, dt_proj_bias, skip_bias_add=True
    )
    layer.out_proj = _LinearTuple(out_proj_weight, out_proj_bias)
    layer.A = nn.Parameter(A)
    layer.D = nn.Parameter(D)

    if use_rms_norm:
        layer.dt_layernorm = RMSNorm(
            time_step_rank, eps=rms_norm_eps, has_weight=True
        ).to(device=device, dtype=dtype)
        layer.b_layernorm = RMSNorm(
            state_size, eps=rms_norm_eps, has_weight=True
        ).to(device=device, dtype=dtype)
        layer.c_layernorm = RMSNorm(
            state_size, eps=rms_norm_eps, has_weight=True
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            layer.dt_layernorm.weight.fill_(1.0)
            layer.b_layernorm.weight.fill_(1.0)
            layer.c_layernorm.weight.fill_(1.0)
    else:
        layer.dt_layernorm = None
        layer.b_layernorm = None
        layer.c_layernorm = None

    weights: dict[str, torch.Tensor | None] = {
        "in_proj_weight": layer.in_proj.weight,
        "in_proj_bias": layer.in_proj.bias,
        "conv1d_weight": layer.conv1d.weight,
        "conv1d_bias": layer.conv1d.bias,
        "x_proj_weight": layer.x_proj.weight,
        "dt_proj_weight": layer.dt_proj.weight,
        "dt_proj_bias": layer.dt_proj.bias,
        "A": layer.A,
        "D": layer.D,
        "out_proj_weight": layer.out_proj.weight,
        "out_proj_bias": layer.out_proj.bias,
        "dt_norm_weight": (
            layer.dt_layernorm.weight if use_rms_norm else None
        ),
        "b_norm_weight": (
            layer.b_layernorm.weight if use_rms_norm else None
        ),
        "c_norm_weight": (
            layer.c_layernorm.weight if use_rms_norm else None
        ),
    }

    return BenchmarkLayer(
        layer=layer,
        mixer_module=mixer_module,
        mamba_ssm_module=mamba_ssm_module,
        weights=weights,
        prefix=prefix,
        use_rms_norm=use_rms_norm,
        rms_norm_eps=rms_norm_eps,
    )


def make_cache_pair(
    *,
    batch_size: int,
    intermediate_size: int,
    state_size: int,
    kernel_size: int,
    activation_dtype: torch.dtype,
    device: torch.device,
    conv_state_dim_first: bool,
    seed: int,
) -> CachePair:
    """Allocate vLLM-layout Conv State and FP32 SSM State plus reset images."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    slots = batch_size + 1  # slot 0 is vLLM NULL_BLOCK_ID
    if conv_state_dim_first:
        conv_shape = (slots, intermediate_size, kernel_size - 1)
    else:
        # vLLM's default SD storage; forward_impl transposes the last axes.
        conv_shape = (slots, kernel_size - 1, intermediate_size)

    initial_conv = _randn(
        conv_shape, device=device, dtype=activation_dtype,
        scale=0.03, generator=generator,
    )
    initial_ssm = _randn(
        (slots, intermediate_size, state_size),
        device=device, dtype=torch.float32,
        scale=0.02, generator=generator,
    )
    return CachePair(
        conv_storage=initial_conv.clone(),
        ssm_state=initial_ssm.clone(),
        initial_conv_storage=initial_conv,
        initial_ssm_state=initial_ssm,
        conv_state_dim_first=conv_state_dim_first,
    )


def make_vllm_metadata(
    *,
    mode: str,
    batch_size: int,
    seq_len: int,
    cache_indices: torch.Tensor,
    device: torch.device,
    context_len: int = 512,
):
    """Construct vLLM 0.24.0 Mamba1AttentionMetadata for pure prefill/decode."""
    from vllm.v1.attention.backends.mamba1_attn import (
        Mamba1AttentionMetadata,
    )
    from vllm.v1.attention.backends.utils import (
        compute_causal_conv1d_metadata,
    )

    if mode == "prefill":
        query_start_loc_cpu = (
            torch.arange(batch_size + 1, dtype=torch.int32) * seq_len
        )
        query_start_loc = query_start_loc_cpu.to(device=device)
        has_initial_states = torch.zeros(
            (batch_size,), dtype=torch.bool, device=device
        )
        nums_dict, batch_ptr, token_chunk_offset_ptr = (
            compute_causal_conv1d_metadata(
                query_start_loc_cpu, device=device
            )
        )
        return Mamba1AttentionMetadata(
            num_prefills=batch_size,
            num_prefill_tokens=batch_size * seq_len,
            num_decodes=0,
            num_decode_tokens=0,
            num_reqs=batch_size,
            has_initial_states_p=has_initial_states,
            query_start_loc_p=query_start_loc,
            num_computed_tokens_p=None,
            state_indices_tensor_p=cache_indices,
            state_indices_tensor_d=None,
            query_start_loc_d=None,
            num_accepted_tokens=None,
            block_idx_last_scheduled_token=None,
            block_idx_first_scheduled_token_p=None,
            block_idx_last_computed_token=None,
            block_idx_last_scheduled_token_prev_step=None,
            seq_lens=torch.full(
                (batch_size,), seq_len, dtype=torch.int32, device=device
            ),
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )

    if mode == "decode":
        # vLLM 0.24.0 stores decode state indices as [batch, 1] when
        # speculative decoding and cache-all are disabled.
        state_indices_d = cache_indices[:, None].contiguous()
        return Mamba1AttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=batch_size,
            num_decode_tokens=batch_size,
            num_reqs=batch_size,
            has_initial_states_p=None,
            query_start_loc_p=None,
            num_computed_tokens_p=None,
            state_indices_tensor_p=None,
            state_indices_tensor_d=state_indices_d,
            query_start_loc_d=None,
            num_accepted_tokens=None,
            block_idx_last_scheduled_token=None,
            block_idx_first_scheduled_token_p=None,
            block_idx_last_computed_token=None,
            block_idx_last_scheduled_token_prev_step=None,
            seq_lens=torch.full(
                (batch_size,), context_len, dtype=torch.int32, device=device
            ),
        )

    raise ValueError(f"unknown mode: {mode}")


@contextlib.contextmanager
def vllm_forward_context_for_benchmark(
    benchmark_layer: BenchmarkLayer,
    metadata,
):
    """Inject only the metadata lookup required by MambaMixer.forward_impl.

    The actual vLLM forward_context machinery is intentionally bypassed:
    this is an operator microbenchmark, not a scheduler/runner benchmark.
    """
    module = benchmark_layer.mixer_module
    old_get_forward_context = module.get_forward_context
    old_selective_state_update = module.selective_state_update

    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={benchmark_layer.prefix: metadata}
    )
    # The native Triton SSU function is used directly, matching the low-level
    # reference used by the correctness test. This avoids requiring a global
    # vLLM SSU backend to be initialized by a full model runner.
    module.selective_state_update = (
        benchmark_layer.mamba_ssm_module.selective_state_update
    )
    try:
        yield
    finally:
        module.get_forward_context = old_get_forward_context
        module.selective_state_update = old_selective_state_update


def run_vllm_forward(
    benchmark_layer: BenchmarkLayer,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    cache: CachePair,
) -> None:
    benchmark_layer.layer.kv_cache = (
        cache.conv_storage,
        cache.ssm_state,
    )
    benchmark_layer.layer.forward_impl(hidden_states, output)


def run_flaggems_forward(
    benchmark_layer: BenchmarkLayer,
    triton_mamba_mixer: Callable,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    cache: CachePair,
    *,
    mode: str,
    query_start_loc: torch.Tensor | None,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    num_actual_tokens: int,
    null_block_id: int,
) -> None:
    w = benchmark_layer.weights
    triton_mamba_mixer(
        hidden_states,
        w["in_proj_weight"],
        w["conv1d_weight"],
        w["x_proj_weight"],
        w["dt_proj_weight"],
        w["A"],
        w["D"],
        w["out_proj_weight"],
        in_proj_bias=w["in_proj_bias"],
        conv1d_bias=w["conv1d_bias"],
        dt_proj_bias=(w["dt_proj_bias"].float() if w["dt_proj_bias"] is not None else None),
        out_proj_bias=w["out_proj_bias"],
        conv_state=cache.conv_state,
        ssm_state=cache.ssm_state,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        dt_norm_weight=w["dt_norm_weight"],
        b_norm_weight=w["b_norm_weight"],
        c_norm_weight=w["c_norm_weight"],
        use_rms_norm=benchmark_layer.use_rms_norm,
        rms_norm_eps=benchmark_layer.rms_norm_eps,
        activation="silu",
        is_decode=(mode == "decode"),
        output=output,
        num_actual_tokens=num_actual_tokens,
        null_block_id=null_block_id,
    )


def assert_correctness(
    *,
    vllm_output: torch.Tensor,
    flaggems_output: torch.Tensor,
    vllm_cache: CachePair,
    flaggems_cache: CachePair,
    dtype: torch.dtype,
) -> dict[str, float]:
    if dtype == torch.bfloat16:
        atol, rtol = 3e-2, 3e-2
    elif dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    elif dtype == torch.float32:
        # Check the full forward, including Conv/SSM state, at FP32 precision.
        # The FlagGems linear kernel uses tf32x3 for FP32 GEMM inputs.
        atol, rtol = 1e-3, 1e-3
    else:
        raise ValueError(f"Unsupported benchmark dtype: {dtype}")

    torch.testing.assert_close(
        flaggems_output, vllm_output, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        flaggems_cache.conv_state,
        vllm_cache.conv_state,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        flaggems_cache.ssm_state,
        vllm_cache.ssm_state,
        atol=atol,
        rtol=rtol,
    )

    return {
        "output_max_abs_diff": float(
            (flaggems_output.float() - vllm_output.float()).abs().max().item()
        ),
        "conv_state_max_abs_diff": float(
            (
                flaggems_cache.conv_state.float()
                - vllm_cache.conv_state.float()
            ).abs().max().item()
        ),
        "ssm_state_max_abs_diff": float(
            (
                flaggems_cache.ssm_state.float()
                - vllm_cache.ssm_state.float()
            ).abs().max().item()
        ),
    }


# One operator, one test function, one automatic result table per dtype.
# Each entry is (batch, seq_len, hidden_size, intermediate_size).
# seq_len == 1 routes to Decode; seq_len > 1 routes to Prefill.
CORE_SHAPES = [
    (1, 1, 768, 1536),
    (8, 1, 768, 1536),
    (1, 128, 768, 1536),
    (1, 512, 768, 1536),
]

MORE_SHAPES = [
    (32, 1, 768, 1536),
    (64, 1, 768, 1536),
    (2, 17, 768, 1536),
    (4, 512, 768, 1536),
    (1, 4096, 768, 1536),
    (8, 2048, 768, 1536),
]


@dataclass
class _Workload:
    shape: tuple[int, int, int, int]
    dtype: torch.dtype
    mode: str
    device: torch.device
    layer: BenchmarkLayer
    metadata: object
    hidden_states: torch.Tensor
    baseline_output: torch.Tensor
    gems_output: torch.Tensor
    baseline_cache: CachePair
    gems_cache: CachePair
    query_start_loc: torch.Tensor | None
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor | None
    null_block_id: int
    gems_op: Callable


def _baseline_forward(workload: _Workload) -> None:
    workload.layer.layer.forward_impl(
        workload.hidden_states, workload.baseline_output
    )


def _gems_forward(workload: _Workload) -> None:
    run_flaggems_forward(
        workload.layer,
        workload.gems_op,
        workload.hidden_states,
        workload.gems_output,
        workload.gems_cache,
        mode=workload.mode,
        query_start_loc=workload.query_start_loc,
        cache_indices=workload.cache_indices,
        has_initial_state=workload.has_initial_state,
        num_actual_tokens=workload.hidden_states.shape[0],
        null_block_id=workload.null_block_id,
    )


def _validate_once(workload: _Workload) -> None:
    """Check outputs AND updated states before measuring a shape."""
    workload.baseline_cache.reset()
    workload.gems_cache.reset()
    with torch.inference_mode():
        with vllm_forward_context_for_benchmark(
            workload.layer, workload.metadata
        ):
            _baseline_forward(workload)
            _gems_forward(workload)
    torch.cuda.synchronize(workload.device)
    diffs = assert_correctness(
        vllm_output=workload.baseline_output,
        flaggems_output=workload.gems_output,
        vllm_cache=workload.baseline_cache,
        flaggems_cache=workload.gems_cache,
        dtype=workload.dtype,
    )
    print(
        f"  correctness PASS: shape={workload.shape}, "
        f"output={diffs['output_max_abs_diff']:.3e}, "
        f"conv={diffs['conv_state_max_abs_diff']:.3e}, "
        f"ssm={diffs['ssm_state_max_abs_diff']:.3e}"
    )


def _make_workload(
    shape: tuple[int, int, int, int], dtype: torch.dtype, device: torch.device
) -> _Workload:
    from vllm.model_executor.layers.mamba.mamba_utils import (
        is_conv_state_dim_first,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
    from flaggems_vllm.ops.mamba_mixer import mamba_mixer

    batch, seq_len, hidden_size, intermediate_size = shape
    # Match Mamba-1 defaults unless an environment override was given.
    state_size = int(os.getenv("MAMBA_BENCH_STATE_SIZE", "16"))
    rank = int(os.getenv("MAMBA_BENCH_DT_RANK", str(math.ceil(hidden_size / 16))))
    kernel_size = int(os.getenv("MAMBA_BENCH_KERNEL_SIZE", "4"))
    use_rms_norm = os.getenv("MAMBA_BENCH_RMS_NORM", "0") == "1"
    use_bias = os.getenv("MAMBA_BENCH_BIAS", "0") == "1"
    conv_layout = is_conv_state_dim_first()
    mode = "decode" if seq_len == 1 else "prefill"
    n_tokens = batch * seq_len

    layer = build_vllm_mamba_layer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        state_size=state_size,
        time_step_rank=rank,
        kernel_size=kernel_size,
        dtype=dtype,
        device=device,
        use_rms_norm=use_rms_norm,
        use_bias=use_bias,
    )
    hidden_states = (
        torch.randn(n_tokens, hidden_size, device=device, dtype=dtype) * 0.2
    )
    cache_indices = torch.arange(
        1, batch + 1, device=device, dtype=torch.int32
    )
    metadata = make_vllm_metadata(
        mode=mode,
        batch_size=batch,
        seq_len=seq_len,
        cache_indices=cache_indices,
        device=device,
    )
    query_start_loc = (
        torch.arange(batch + 1, device=device, dtype=torch.int32) * seq_len
        if mode == "prefill"
        else None
    )
    has_initial_state = (
        torch.zeros(batch, device=device, dtype=torch.bool)
        if mode == "prefill"
        else None
    )
    # Two independent caches, initialized using the same seed.
    seed = 10000 + batch * 100 + seq_len
    def new_cache():
        return make_cache_pair(
            batch_size=batch,
            intermediate_size=intermediate_size,
            state_size=state_size,
            kernel_size=kernel_size,
            activation_dtype=dtype,
            device=device,
            conv_state_dim_first=conv_layout,
            seed=seed,
        )

    baseline_cache, gems_cache = new_cache(), new_cache()
    # The vLLM forward_impl reads its two caches through layer.kv_cache.
    layer.layer.kv_cache = (baseline_cache.conv_storage, baseline_cache.ssm_state)
    return _Workload(
        shape=shape,
        dtype=dtype,
        mode=mode,
        device=device,
        layer=layer,
        metadata=metadata,
        hidden_states=hidden_states,
        baseline_output=torch.empty_like(hidden_states),
        gems_output=torch.empty_like(hidden_states),
        baseline_cache=baseline_cache,
        gems_cache=gems_cache,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        null_block_id=NULL_BLOCK_ID,
        gems_op=mamba_mixer,
    )


def _time_stateful_forward(
    fn: Callable,
    reset: Callable,
    device: torch.device,
    mode: consts.BenchMode,
    warmup_ms: float,
    repetition_ms: float,
) -> float:
    """Median ms for *forward only*, excluding cache reset and sync setup.

    BenchMode.KERNEL uses CUDA Events for the full forward; OPERATOR uses a
    synchronized CPU wall timer, consistent with the repository's two modes.
    The per-run state restore happens outside both timing intervals.
    """
    if warmup_ms < 0 or repetition_ms <= 0:
        raise ValueError("--warmup must be >=0 and --iter must be >0 (milliseconds)")

    is_kernel = mode == consts.BenchMode.KERNEL
    # The first execution ensures compilation is complete before timed samples.
    reset()
    torch.cuda.synchronize(device)
    fn()
    torch.cuda.synchronize(device)

    def call_once() -> float:
        reset()
        torch.cuda.synchronize(device)
        if is_kernel:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            fn()
            end_evt.record()
            end_evt.synchronize()
            return start_evt.elapsed_time(end_evt)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        return (time.perf_counter() - start) * 1000.0

    warm_start = time.perf_counter()
    while (time.perf_counter() - warm_start) * 1000.0 < warmup_ms:
        call_once()

    samples = []
    measure_start = time.perf_counter()
    while not samples or (time.perf_counter() - measure_start) * 1000.0 < repetition_ms:
        samples.append(call_once())
    return float(statistics.median(samples))


class MambaMixerBenchmark(base.Benchmark):
    """Use benchmark/base.py's standard table & JSON recording."""

    def __init__(self):
        super().__init__(
            op_name="mamba_mixer",
            torch_op=_baseline_forward,
            gems_op=_gems_forward,
            dtypes=[torch.float16, torch.bfloat16, torch.float32],
        )
        self.shape_desc = "B, S, H, I"

    def set_shapes(self, shape_file_path=None):
        # Unlike pointwise benchmarks, these are Mamba workload tuples.
        self.shapes = CORE_SHAPES.copy()
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(MORE_SHAPES)
        self.shape_desc = "batch, seq_len, hidden, intermediate"

    def get_input_iter(self, dtype):
        gpu = int(os.getenv("MAMBA_BENCH_GPU", "0"))
        device = torch.device("cuda", gpu)
        for shape in self.shapes:
            workload = _make_workload(shape, dtype, device)
            _validate_once(workload)
            # The dict is unpacked by base.Benchmark into keyword arguments.
            yield ({"workload": workload},)

    def record_shapes(self, *args, **kwargs):
        # Put the actual operator shape in the standard Size Detail column.
        return tuple(kwargs["workload"].shape)

    def get_latency(self, op, *args, **kwargs):
        # Override the default timer because both implementations mutate caches.
        workload = kwargs["workload"]
        reset = (
            workload.baseline_cache.reset
            if op is _baseline_forward
            else workload.gems_cache.reset
        )
        with torch.inference_mode():
            with vllm_forward_context_for_benchmark(
                workload.layer, workload.metadata
            ):
                return _time_stateful_forward(
                    fn=lambda: op(workload),
                    reset=reset,
                    device=workload.device,
                    mode=base.Config.mode,
                    warmup_ms=float(base.Config.warm_up),
                    repetition_ms=float(base.Config.repetition),
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")
def test_mamba_mixer_perf():
    """One pytest item; benchmark.run() prints standard one-table-per-dtype results."""
    if base.Config.mode not in (consts.BenchMode.KERNEL, consts.BenchMode.OPERATOR):
        pytest.skip("Stateful MambaMixer benchmark supports kernel/operator only")
    torch.cuda.set_device(int(os.getenv("MAMBA_BENCH_GPU", "0")))
    # The vLLM/PyTorch FP32 F.linear baseline uses IEEE FP32, not TF32.
    # This avoids comparing reduced-precision Torch vs tf32x3 Triton GEMM.
    torch.backends.cuda.matmul.allow_tf32 = False
    bench = MambaMixerBenchmark()
    bench.run()

