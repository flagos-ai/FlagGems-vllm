# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Installed / pinned vLLM ``ple_conv`` baseline and shared workload matrix."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch

OP_NAME = "ple_conv"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = True
# residual (1) and conv_state (2) are written by both native and FlagGems paths.
MUTATED_INPUTS = (1, 2)
NATIVE_API = "vllm.models.qwen4_exp.common.ops.ple.ple_conv"
NATIVE_UNAVAILABLE_REASON = None
NATIVE_VERSION = None

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


def _load_pinned_source(path: Path):
    """Load upstream ple.py without a full vLLM install (stubs for imports)."""
    if "vllm" not in sys.modules:
        sys.modules["vllm"] = types.ModuleType("vllm")
    if "vllm.platforms" not in sys.modules:
        platforms = types.ModuleType("vllm.platforms")

        class _Platform:
            @staticmethod
            def is_arch_support_pdl():
                if not torch.cuda.is_available():
                    return False
                major, _ = torch.cuda.get_device_capability()
                return major >= 9

        platforms.current_platform = _Platform()
        sys.modules["vllm.platforms"] = platforms
    if "vllm.triton_utils" not in sys.modules:
        triton_utils = types.ModuleType("vllm.triton_utils")
        import triton
        import triton.language as tl

        triton_utils.tl = tl
        triton_utils.triton = triton
        sys.modules["vllm.triton_utils"] = triton_utils
    for name in (
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
        "vllm.v1.attention.backends.utils",
    ):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    utils = sys.modules["vllm.v1.attention.backends.utils"]
    if not hasattr(utils, "NULL_BLOCK_ID"):
        utils.NULL_BLOCK_ID = 0

    name = "_fgv_pinned_ple_conv_" + path.resolve().as_posix().replace("/", "_")
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return module


def native():
    """Return upstream ``ple_conv`` or None, retaining the failure reason."""
    global NATIVE_UNAVAILABLE_REASON, NATIVE_VERSION
    NATIVE_VERSION = None
    pinned = os.environ.get("FGV_PLE_NATIVE_SOURCE")
    try:
        if pinned:
            module = _load_pinned_source(Path(pinned).resolve(strict=True))
            op = getattr(module, "ple_conv")
            NATIVE_VERSION = f"pinned:{pinned}"
        else:
            import vllm
            from vllm.models.qwen4_exp.common.ops.ple import ple_conv as op

            NATIVE_VERSION = getattr(vllm, "__version__", None)
    except Exception as exc:  # noqa: BLE001 - resolution failures stay N/A
        NATIVE_UNAVAILABLE_REASON = f"{NATIVE_API}: {type(exc).__name__}: {exc}"
        return None
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
            "query_start_loc": torch.arange(
                rows + 1, dtype=torch.int32, device=device
            )
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
    return {
        "inputs": {
            "inputs": {"shape": [tokens, channels], "dtype": str(dtype)},
            "residual": {"shape": [tokens, channels], "dtype": str(dtype)},
            "conv_state": {
                "shape": [rows + 2, channels, state_width],
                "dtype": str(dtype),
            },
            "conv_weights": {
                "shape": [channels, binding["kernel_size"]],
                "dtype": str(dtype),
            },
            "state_indices": {
                "shape": [1 if mode == "prefill" else rows],
                "dtype": "torch.int32",
            },
            "outer_residual": (
                {"shape": [tokens, channels], "dtype": str(dtype)}
                if binding["fuse_residual"]
                else None
            ),
            "mode": {"scalar": mode},
            "dilation": {"scalar": binding["dilation"]},
        },
        "outputs": {
            "residual": {"shape": [tokens, channels], "dtype": str(dtype), "inplace": True},
            "conv_state": {
                "shape": [rows + 2, channels, state_width],
                "dtype": str(dtype),
                "inplace": True,
            },
        },
        "dims": dict(binding),
        "native_version": NATIVE_VERSION,
        "native_api": NATIVE_API,
        "baseline_kind": "installed-vllm-or-pinned",
        "mutation": "residual,conv_state",
        "graph": False,
    }
