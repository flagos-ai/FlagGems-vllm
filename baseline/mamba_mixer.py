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

from __future__ import annotations

import importlib
import math
import os
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

OP_NAME = "mamba_mixer"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
IS_INPLACE = True
# args[2] and args[3] are the actual mutable state storages. The output
# (args[1]) is overwritten completely and need not be restored between runs.
MUTATED_INPUTS = (2, 3)
NATIVE_UNAVAILABLE_REASON = None

# (batch, seq_len, hidden_size, intermediate_size): exactly the benchmark grid.
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


def _parameters(hidden_size: int) -> dict:
    """Keep the benchmark's Mamba-1 shape and optional-feature defaults."""
    return {
        "state_size": int(os.getenv("MAMBA_BENCH_STATE_SIZE", "16")),
        "dt_rank": int(
            os.getenv("MAMBA_BENCH_DT_RANK", str(math.ceil(hidden_size / 16)))
        ),
        "kernel_size": int(os.getenv("MAMBA_BENCH_KERNEL_SIZE", "4")),
        "rms_norm": os.getenv("MAMBA_BENCH_RMS_NORM", "0") == "1",
        "bias": os.getenv("MAMBA_BENCH_BIAS", "0") == "1",
    }


def grid() -> list[dict]:
    shapes = list(CORE_SHAPES)
    if os.getenv("MAMBA_BASELINE_COMPREHENSIVE", "0") == "1":
        shapes.extend(MORE_SHAPES)
    return [{"shape": shape} for shape in shapes]


def key_shape(binding) -> list[int]:
    return list(binding["shape"])


def native():
    """Resolve the native vLLM implementation, without loading FlagGems kernels."""
    global NATIVE_UNAVAILABLE_REASON
    try:
        mixer = importlib.import_module("vllm.model_executor.layers.mamba.mamba_mixer")
        importlib.import_module("vllm.model_executor.layers.mamba.ops.mamba_ssm")
        importlib.import_module("vllm.v1.attention.backends.mamba1_attn")
        importlib.import_module("vllm.v1.attention.backends.utils")
        if not callable(getattr(mixer.MambaMixer, "forward_impl", None)):
            raise AttributeError("MambaMixer.forward_impl is unavailable")
    except (ImportError, AttributeError, RuntimeError) as exc:
        NATIVE_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return _native_forward


class _TupleLinear(nn.Module):
    """Single-GPU projection with the (output, deferred_bias) vLLM API."""

    def __init__(self, weight, bias=None, *, defer_bias=False):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.defer_bias = defer_bias

    def forward(self, x):
        if self.defer_bias:
            return F.linear(x, self.weight), self.bias
        return F.linear(x, self.weight, self.bias), None


class _ConvWeights(nn.Module):
    """Container matching vLLM's conv1d.weight and conv1d.bias attributes."""

    def __init__(self, weight, bias):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if bias is not None else None


def _make_layer(shape, dtype, device, opts):
    """Build vLLM's real mixer object, avoiding distributed parallel init."""
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer

    _, _, hidden, intermediate = shape
    state, rank, kernel = (opts["state_size"], opts["dt_rank"], opts["kernel_size"])
    gen = torch.Generator(device=device)
    gen.manual_seed(2026)

    def random_weight(dims, scale, *, dt=dtype):
        return (
            torch.randn(dims, generator=gen, device=device, dtype=torch.float32)
            .mul_(scale)
            .to(dt)
        )

    def bias(dims, *, enabled=True, scale=0.02):
        return random_weight(dims, scale) if enabled else None

    layer = MambaMixer.__new__(MambaMixer)
    nn.Module.__init__(layer)
    prefix = "benchmark.mamba_mixer"
    layer.prefix = prefix
    layer.time_step_rank = rank
    layer.ssm_state_size = state
    layer.intermediate_size = intermediate
    layer.conv_kernel_size = kernel
    layer.use_rms_norm = opts["rms_norm"]
    layer.activation = "silu"
    layer.is_lora_enabled = False
    layer.model_config = None
    layer.cache_config = SimpleNamespace(mamba_block_size=256, mamba_cache_mode="none")

    use_bias = opts["bias"]
    layer.in_proj = _TupleLinear(
        random_weight((2 * intermediate, hidden), hidden**-0.5),
        bias((2 * intermediate,), enabled=use_bias),
    )
    layer.conv1d = _ConvWeights(
        random_weight((intermediate, 1, kernel), 0.15),
        bias((intermediate,)),
    )
    layer.x_proj = _TupleLinear(
        random_weight((rank + 2 * state, intermediate), intermediate**-0.5)
    )
    layer.dt_proj = _TupleLinear(
        random_weight((intermediate, rank), rank**-0.5),
        bias((intermediate,), scale=0.05),
        defer_bias=True,
    )
    layer.A = nn.Parameter(
        -torch.exp(random_weight((intermediate, state), 0.1, dt=torch.float32))
    )
    layer.D = nn.Parameter(torch.ones(intermediate, device=device, dtype=torch.float32))
    layer.out_proj = _TupleLinear(
        random_weight((hidden, intermediate), intermediate**-0.5),
        bias((hidden,), enabled=use_bias),
    )

    for attr, width in (
        ("dt_layernorm", rank),
        ("b_layernorm", state),
        ("c_layernorm", state),
    ):
        if opts["rms_norm"]:
            norm = RMSNorm(width, eps=1e-5, has_weight=True).to(
                device=device, dtype=dtype
            )
            with torch.no_grad():
                norm.weight.fill_(1.0)
            setattr(layer, attr, norm)
        else:
            setattr(layer, attr, None)
    layer.eval()
    return layer


def _make_metadata(mode, batch, seq_len, cache_indices, device):
    """Native vLLM 0.24.x Mamba1AttentionMetadata for pure prefill/decode."""
    from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadata

    common = dict(
        num_reqs=batch,
        num_accepted_tokens=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
    )
    if mode == "prefill":
        from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

        loc_cpu = torch.arange(batch + 1, dtype=torch.int32) * seq_len
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            loc_cpu, device=device
        )
        return Mamba1AttentionMetadata(
            **common,
            num_prefills=batch,
            num_prefill_tokens=batch * seq_len,
            num_decodes=0,
            num_decode_tokens=0,
            has_initial_states_p=torch.zeros(batch, dtype=torch.bool, device=device),
            query_start_loc_p=loc_cpu.to(device=device),
            num_computed_tokens_p=None,
            state_indices_tensor_p=cache_indices,
            state_indices_tensor_d=None,
            query_start_loc_d=None,
            seq_lens=torch.full((batch,), seq_len, dtype=torch.int32, device=device),
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
    return Mamba1AttentionMetadata(
        **common,
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=batch,
        num_decode_tokens=batch,
        has_initial_states_p=None,
        query_start_loc_p=None,
        num_computed_tokens_p=None,
        state_indices_tensor_p=None,
        state_indices_tensor_d=cache_indices[:, None].contiguous(),
        query_start_loc_d=None,
        seq_lens=torch.full((batch,), 512, dtype=torch.int32, device=device),
    )


def build_inputs(binding, dtype, device):
    """Independent, deterministic state and weight allocation per collection."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    shape = tuple(binding["shape"])
    batch, seq_len, hidden, intermediate = shape
    opts = _parameters(hidden)
    state_size, kernel = opts["state_size"], opts["kernel_size"]
    if state_size < 1 or kernel < 2 or opts["dt_rank"] < 1:
        raise ValueError("state_size and dt_rank must be > 0; kernel_size must be >= 2")

    layer = _make_layer(shape, dtype, device, opts)
    mode = "decode" if seq_len == 1 else "prefill"
    gen = torch.Generator(device=device)
    gen.manual_seed(2026 + batch + seq_len)
    hidden_states = torch.randn(
        (batch * seq_len, hidden), dtype=dtype, device=device, generator=gen
    ).mul_(0.2)
    output = torch.empty_like(hidden_states)

    # Slot 0 is NULL_BLOCK_ID; real requests use slots [1..batch].
    slots = batch + 1
    if is_conv_state_dim_first():
        conv_shape = (slots, intermediate, kernel - 1)
    else:
        conv_shape = (slots, kernel - 1, intermediate)
    state_gen = torch.Generator(device=device)
    state_gen.manual_seed(10000 + batch * 100 + seq_len)
    conv_state = (
        torch.randn(conv_shape, device=device, dtype=torch.float32, generator=state_gen)
        .mul_(0.03)
        .to(dtype)
    )
    ssm_state = torch.randn(
        (slots, intermediate, state_size),
        device=device,
        dtype=torch.float32,
        generator=state_gen,
    ).mul_(0.02)

    cache_indices = torch.arange(1, slots, device=device, dtype=torch.int32)
    metadata = _make_metadata(mode, batch, seq_len, cache_indices, device)
    mixer_module = importlib.import_module(
        "vllm.model_executor.layers.mamba.mamba_mixer"
    )
    ssm_module = importlib.import_module(
        "vllm.model_executor.layers.mamba.ops.mamba_ssm"
    )
    # The collector snapshots only args[2]/args[3], then restores their values
    # outside all timed regions. Do not allocate or copy caches inside native().
    args = (
        hidden_states,
        output,
        conv_state,
        ssm_state,
        layer,
        metadata,
        mixer_module,
        ssm_module,
    )
    return args, {}


def _native_forward(
    hidden_states,
    output,
    conv_state,
    ssm_state,
    layer,
    metadata,
    mixer_module,
    ssm_module,
):
    """Invoke the actual vLLM Mamba-1 forward_impl under its native metadata."""
    old_context = mixer_module.get_forward_context
    old_ssu = mixer_module.selective_state_update
    mixer_module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={layer.prefix: metadata}
    )
    # Match benchmark/test_mamba_mixer.py: select vLLM's native SSU directly.
    mixer_module.selective_state_update = ssm_module.selective_state_update
    layer.kv_cache = (conv_state, ssm_state)
    try:
        with torch.inference_mode():
            layer.forward_impl(hidden_states, output)
    finally:
        mixer_module.get_forward_context = old_context
        mixer_module.selective_state_update = old_ssu
    return output


def config(binding, dtype) -> dict:
    """Machine-readable tensor geometry for collection JSON and NCU reports."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    batch, seq_len, hidden, intermediate = binding["shape"]
    opts = _parameters(hidden)
    slots = batch + 1
    state, rank, kernel = (opts["state_size"], opts["dt_rank"], opts["kernel_size"])
    conv_first = is_conv_state_dim_first()
    conv_shape = (
        [slots, intermediate, kernel - 1]
        if conv_first
        else [slots, kernel - 1, intermediate]
    )

    def tensor(shape, dt):
        return {"shape": list(shape), "dtype": str(dt)}

    return {
        "inputs": {
            "hidden_states": tensor((batch * seq_len, hidden), dtype),
            "in_proj_weight": tensor((intermediate * 2, hidden), dtype),
            "conv1d_weight": tensor((intermediate, 1, kernel), dtype),
            "x_proj_weight": tensor((rank + state * 2, intermediate), dtype),
            "dt_proj_weight": tensor((intermediate, rank), dtype),
            "A": tensor((intermediate, state), torch.float32),
            "D": tensor((intermediate,), torch.float32),
            "out_proj_weight": tensor((hidden, intermediate), dtype),
            "conv_state": tensor(conv_shape, dtype),
            "ssm_state": tensor((slots, intermediate, state), torch.float32),
            "state_indices": tensor(
                (batch, 1) if seq_len == 1 else (batch,), torch.int32
            ),
        },
        "outputs": {"output": tensor((batch * seq_len, hidden), dtype)},
        "dims": {
            "batch": batch,
            "seq_len": seq_len,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "state_size": state,
            "time_step_rank": rank,
            "kernel_size": kernel,
            "mode": "decode" if seq_len == 1 else "prefill",
            "conv_state_dim_first": conv_first,
            "use_rms_norm": opts["rms_norm"],
            "use_bias": opts["bias"],
        },
        "mutation": ["conv_state", "ssm_state", "output"],
        "timing": "CUDA event, no CUDA graph; state restoration outside timing",
        "native_api": "vllm MambaMixer.forward_impl",
    }
