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

"""NVIDIA vLLM baseline for Qwen4 PLE short convolution.

The timed callable is ``vllm.models.qwen4_exp.common.ops.ple.ple_conv``
(vLLM 0.33). It writes SiLU(conv) into ``residual`` and updates ``conv_state``.
``fuse=1`` also passes ``outer_residual`` so the native kernel adds that
residual in the same launch. There is no PyTorch conv1d fallback: if the
vLLM symbol cannot be imported, ``native()`` returns None.
"""

import importlib

import torch

# Collector JSON key and benchmark op_name. The vLLM symbol is also ple_conv.
OP_NAME = "ple_conv"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = True
# residual and conv_state are written by the native kernel.
MUTATED_INPUTS = (1, 2)
NATIVE_UNAVAILABLE_REASON = None

_NATIVE_MODULE = "vllm.models.qwen4_exp.common.ops.ple"
_NATIVE_NAME = "ple_conv"
# Speculative window used with the production kernel (K=4, dilation=3).
_SPEC_QUERY_LEN = 4

# mode, rows, channels, fuse, kernel, dilation.
# decode/spec rows are request counts. prefill rows are the token count of
# one request. fuse selects the native outer-residual addition.
_SHAPES = [
    ("decode", 1, 512, 0, 4, 3),
    ("decode", 1, 512, 1, 4, 3),
    ("decode", 64, 2048, 0, 4, 3),
    ("decode", 64, 2048, 1, 4, 3),
    ("prefill", 33, 1024, 0, 4, 3),
    ("prefill", 1024, 10240, 0, 4, 3),
    ("prefill", 1024, 10240, 1, 4, 3),
    ("spec", 16, 2048, 0, 4, 3),
    ("spec", 64, 2048, 1, 4, 3),
]


def native():
    """Resolve vLLM ``ple_conv``. Return None when this image has no such op."""
    global NATIVE_UNAVAILABLE_REASON
    try:
        module = importlib.import_module(_NATIVE_MODULE)
    except ImportError as exc:
        NATIVE_UNAVAILABLE_REASON = f"{_NATIVE_MODULE}: {exc}"
        return None
    op = getattr(module, _NATIVE_NAME, None)
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = f"{_NATIVE_MODULE}.{_NATIVE_NAME} is not callable"
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return op


def grid():
    return [{"shape": list(shape)} for shape in _SHAPES]


def _unpack(binding):
    mode, rows, channels, fuse, kernel, dilation = binding["shape"]
    return mode, int(rows), int(channels), int(fuse), int(kernel), int(dilation)


def build_inputs(binding, dtype, device):
    """Arguments for ``ple_conv(*args, **kwargs)``.

    Layout matches the vLLM call in ``Qwen4ExpPLELayer``: activation is
    written into ``residual``, and the cache is ``[slots, channels, window]``.
    """
    mode, rows, channels, fuse, kernel, dilation = _unpack(binding)
    spec_query_len = _SPEC_QUERY_LEN if mode == "spec" else 1
    if mode == "prefill":
        nreq, tokens = 1, rows
        state_indices = torch.zeros(1, dtype=torch.int32, device=device)
        query_start_loc = torch.tensor([0, tokens], dtype=torch.int32, device=device)
        has_initial_states = torch.ones(1, dtype=torch.bool, device=device)
        num_accepted_tokens = None
    elif mode == "spec":
        nreq, tokens = rows, rows * spec_query_len
        state_indices = torch.arange(nreq, dtype=torch.int32, device=device)
        query_start_loc = (
            torch.arange(nreq + 1, device=device, dtype=torch.int32) * spec_query_len
        )
        has_initial_states = None
        num_accepted_tokens = torch.ones(nreq, dtype=torch.int32, device=device)
    elif mode == "decode":
        nreq, tokens = rows, rows
        state_indices = torch.arange(nreq, dtype=torch.int32, device=device)
        query_start_loc = None
        has_initial_states = torch.ones(nreq, dtype=torch.bool, device=device)
        num_accepted_tokens = None
    else:
        raise ValueError(f"unsupported PLE mode: {mode}")

    state_len = (kernel - 1) * dilation
    state_width = state_len + spec_query_len - 1
    # H20 collection uses rows + 2 cache slots for every mode.
    state_rows = rows + 2
    inputs = torch.randn(tokens, channels, dtype=dtype, device=device)
    residual = torch.empty(tokens, channels, dtype=dtype, device=device)
    conv_state = torch.randn(
        state_rows, channels, state_width, dtype=dtype, device=device
    )
    conv_weights = torch.randn(channels, kernel, dtype=dtype, device=device)
    outer_residual = (
        torch.randn(tokens, channels, dtype=dtype, device=device) if fuse else None
    )
    args = (
        inputs,
        residual,
        conv_state,
        conv_weights,
        state_indices,
        outer_residual,
    )
    kwargs = {
        "mode": mode,
        "dilation": dilation,
        "query_start_loc": query_start_loc,
        "num_accepted_tokens": num_accepted_tokens,
        "has_initial_states": has_initial_states,
        "spec_query_len": spec_query_len,
    }
    return args, kwargs


def key_shape(binding):
    """JSON primary key. Same list the H20 vLLM 0.33 collection used."""
    return list(binding["shape"])


def config(binding, dtype):
    """Input and output shapes written into the baseline JSON."""
    mode, rows, channels, fuse, kernel, dilation = _unpack(binding)
    # spec_query_len is recorded as the workload constant 4 for every mode.
    # Only spec actually extends the cache by that window.
    active_query = _SPEC_QUERY_LEN if mode == "spec" else 1
    if mode == "prefill":
        nreq, tokens = 1, rows
    elif mode == "spec":
        nreq, tokens = rows, rows * active_query
    else:
        nreq, tokens = rows, rows
    state_len = (kernel - 1) * dilation
    state_width = state_len + active_query - 1
    state_rows = rows + 2
    dt = str(dtype)
    inputs = {
        "inputs": {"shape": [tokens, channels], "dtype": dt},
        "residual": {
            "shape": [tokens, channels],
            "dtype": dt,
            "note": "原地写回",
        },
        "conv_state": {
            "shape": [state_rows, channels, state_width],
            "dtype": dt,
            "note": "原地写回",
        },
        "conv_weights": {"shape": [channels, kernel], "dtype": dt},
        "state_indices": {"shape": [nreq], "dtype": "torch.int32"},
        "outer_residual": (
            {"shape": [tokens, channels], "dtype": dt} if fuse else None
        ),
        "mode": {"scalar": mode},
        "dilation": {"scalar": dilation},
    }
    return {
        "inputs": inputs,
        "outputs": {
            "residual": {"shape": [tokens, channels], "dtype": dt},
            "conv_state": {"shape": [state_rows, channels, state_width], "dtype": dt},
        },
        "dims": {
            "mode": mode,
            "rows": rows,
            "channels": channels,
            "fuse_residual": bool(fuse),
            "kernel_size": kernel,
            "dilation": dilation,
            "spec_query_len": _SPEC_QUERY_LEN,
        },
        "native_api": f"{_NATIVE_MODULE}.{_NATIVE_NAME}",
    }
