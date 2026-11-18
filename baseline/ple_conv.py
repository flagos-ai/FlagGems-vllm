# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""vLLM ``ple_conv`` CUDA baseline collector workload (installed NV interface only)."""

from __future__ import annotations

import importlib

import torch

OP_NAME = "ple_conv"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = True
# residual (1) and conv_state (2) are written by both native and FlagGems paths.
MUTATED_INPUTS = (1, 2)
NATIVE_API = "vllm.models.qwen4_exp.common.ops.ple.ple_conv"
NATIVE_UNAVAILABLE_REASON = None

# mode, rows, channels, fuse_residual
SHAPES = [
    ("decode", 1, 512, False),
    ("decode", 1, 512, True),
    ("decode", 64, 2048, False),
    ("decode", 64, 2048, True),
    ("prefill", 33, 1024, False),
    ("prefill", 1024, 10240, False),
    ("prefill", 1024, 10240, True),
    ("spec", 16, 2048, False),
    ("spec", 64, 2048, True),
]

KERNEL_SIZE = 4
DILATION = 3
SPEC_QUERY_LEN = 4


def native():
    """Resolve installed vLLM ``ple_conv``; return None if unavailable."""
    global NATIVE_UNAVAILABLE_REASON
    try:
        mod = importlib.import_module("vllm.models.qwen4_exp.common.ops.ple")
    except ImportError as exc:
        NATIVE_UNAVAILABLE_REASON = f"{NATIVE_API}: ImportError: {exc}"
        return None
    op = getattr(mod, "ple_conv", None)
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = f"{NATIVE_API} is not callable"
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return op


def grid():
    return [
        dict(
            mode=mode,
            rows=rows,
            channels=channels,
            fuse_residual=fuse,
            kernel_size=KERNEL_SIZE,
            dilation=DILATION,
            spec_query_len=SPEC_QUERY_LEN,
        )
        for mode, rows, channels, fuse in SHAPES
    ]


def key_shape(binding):
    return [
        binding["mode"],
        binding["rows"],
        binding["channels"],
        int(binding["fuse_residual"]),
        binding["kernel_size"],
        binding["dilation"],
    ]


def build_inputs(binding, dtype, device):
    mode = binding["mode"]
    rows = binding["rows"]
    channels = binding["channels"]
    dilation = binding["dilation"]
    kernel_size = binding["kernel_size"]
    spec_query_len = binding["spec_query_len"]
    fuse = binding["fuse_residual"]
    state_len = (kernel_size - 1) * dilation
    query_len = spec_query_len if mode == "spec" else 1
    state_width = state_len + (query_len - 1 if mode == "spec" else 0)

    if mode == "decode":
        tokens = rows
        indices = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
        kwargs = {
            "mode": "decode",
            "dilation": dilation,
            "has_initial_states": torch.ones(rows, dtype=torch.bool, device=device),
        }
    elif mode == "prefill":
        tokens = rows
        indices = torch.tensor([1], dtype=torch.int32, device=device)
        kwargs = {
            "mode": "prefill",
            "dilation": dilation,
            "query_start_loc": torch.tensor(
                [0, tokens], dtype=torch.int32, device=device
            ),
            "has_initial_states": torch.tensor([True], dtype=torch.bool, device=device),
        }
    elif mode == "spec":
        tokens = rows * query_len
        indices = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
        kwargs = {
            "mode": "spec",
            "dilation": dilation,
            "query_start_loc": torch.arange(rows + 1, dtype=torch.int32, device=device)
            * query_len,
            "num_accepted_tokens": torch.ones(rows, dtype=torch.int32, device=device),
            "spec_query_len": query_len,
        }
    else:
        raise ValueError(f"unsupported mode: {mode}")

    inputs = torch.randn(tokens, channels, dtype=dtype, device=device)
    residual = torch.randn_like(inputs)
    conv_state = torch.randn(
        rows + 2, channels, state_width, dtype=dtype, device=device
    )
    conv_weights = torch.randn(channels, kernel_size, dtype=dtype, device=device) * 0.1
    outer = torch.randn_like(inputs) if fuse else None
    args = (inputs, residual, conv_state, conv_weights, indices, outer)
    return args, kwargs


def config(binding, dtype):
    mode = binding["mode"]
    rows = binding["rows"]
    channels = binding["channels"]
    query_len = binding["spec_query_len"] if mode == "spec" else 1
    tokens = rows if mode != "spec" else rows * query_len
    state_len = (binding["kernel_size"] - 1) * binding["dilation"]
    state_width = state_len + (query_len - 1 if mode == "spec" else 0)
    dt = str(dtype)
    return {
        "inputs": {
            "inputs": {"shape": [tokens, channels], "dtype": dt},
            "residual": {"shape": [tokens, channels], "dtype": dt, "note": "原地写回"},
            "conv_state": {
                "shape": [rows + 2, channels, state_width],
                "dtype": dt,
                "note": "原地写回",
            },
            "conv_weights": {
                "shape": [channels, binding["kernel_size"]],
                "dtype": dt,
            },
            "state_indices": {
                "shape": [1 if mode == "prefill" else rows],
                "dtype": "torch.int32",
            },
            "outer_residual": (
                {"shape": [tokens, channels], "dtype": dt}
                if binding["fuse_residual"]
                else None
            ),
            "mode": {"scalar": mode},
            "dilation": {"scalar": binding["dilation"]},
        },
        "outputs": {
            "residual": {"shape": [tokens, channels], "dtype": dt},
            "conv_state": {
                "shape": [rows + 2, channels, state_width],
                "dtype": dt,
            },
        },
        "dims": dict(binding),
        "native_api": NATIVE_API,
    }
