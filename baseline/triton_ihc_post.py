# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""HY V4 iHC post native resolution and the shared benchmark workload grid."""

import hashlib
import importlib
import inspect
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import torch

OP_NAME = "triton_ihc_post"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ()
NATIVE_UNAVAILABLE_REASON = None
NATIVE_ARTIFACT = None
NATIVE_MODULE = "vllm.models.hy_v4.nvidia.triton_ihc"
NATIVE_NAME = "triton_ihc_post"
SOURCE_PIN = "a17e79abde45715e79a00deb99ad34be49162231"
SOURCE_PATH = "vllm/models/hy_v4/nvidia/triton_ihc.py"
SOURCE_SHA256 = "f47036732264faaf08dd039aad89e0cf1d0eba445767c1ae5c64e8ed2d1ea215"
# HY V4 hidden size 6144 and four channels; token counts are workload selections.
TOKENS = (1, 7, 64, 256, 1024, 4096)
LAYOUTS = ("contiguous", "all_strided")


@lru_cache(maxsize=1)
def native():
    """Resolve the installed original callable, recording its actual provenance."""
    global NATIVE_UNAVAILABLE_REASON, NATIVE_ARTIFACT
    try:
        module = importlib.import_module(NATIVE_MODULE)
        op = getattr(module, NATIVE_NAME)
    except (ImportError, AttributeError, OSError) as exc:
        NATIVE_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
        return None
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = f"{NATIVE_MODULE}.{NATIVE_NAME} is not callable"
        return None
    original = inspect.unwrap(op)
    NATIVE_ARTIFACT = {
        "resolved_api": f"{NATIVE_MODULE}.{NATIVE_NAME}",
        "callable_module": getattr(original, "__module__", None),
        "reference_commit": SOURCE_PIN,
    }
    try:
        NATIVE_ARTIFACT["vllm_version"] = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        NATIVE_ARTIFACT["vllm_version"] = None
    source = inspect.getsourcefile(original) or getattr(module, "__file__", None)
    NATIVE_ARTIFACT["source_file"] = source
    try:
        NATIVE_ARTIFACT["source_sha256"] = hashlib.sha256(
            Path(source).read_bytes()
        ).hexdigest()
    except (TypeError, OSError):
        pass  # Source identity is diagnostic, not a requirement for installed vLLM.
    if getattr(original, "__module__", None) != NATIVE_MODULE:
        NATIVE_UNAVAILABLE_REASON = (
            "native callable is redirected outside its source module"
        )
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return op


def grid():
    """12 model bindings / 24 dtype rows, including full-entry copy workloads."""
    return [
        {"tokens": n, "channels": 4, "hidden": 6144, "layout": layout}
        for n in TOKENS
        for layout in LAYOUTS
    ]


def _strides(shape, strided):
    stride = 2 if strided else 1
    result = []
    for size in reversed(shape):
        result.insert(0, stride)
        stride *= size
    return result


def _tensor(shape, dtype, device, strided):
    if not strided:
        return torch.randn(shape, dtype=dtype, device=device)
    # Stepped final dimension, nonzero storage offset and prefix/suffix canaries.
    parent = torch.randn((*shape[:-1], 2 * shape[-1] + 2), dtype=dtype, device=device)
    return parent[..., 1:-1:2]


def build_inputs(binding, dtype, device):
    if dtype not in DTYPES:
        raise ValueError("iHC post workloads support float16 and bfloat16")
    n, hc, d = binding["tokens"], binding["channels"], binding["hidden"]
    layout = binding["layout"]
    if layout not in LAYOUTS:
        raise ValueError(f"unknown iHC post layout: {layout}")
    strided = layout == "all_strided"
    x = _tensor((n, d), dtype, device, strided)
    residual = _tensor((n, hc, d), dtype, device, strided)
    post = _tensor((n, hc), torch.float32, device, strided)
    return (x, residual, post), {}


def key_shape(binding):
    return (
        f"n{binding['tokens']}_hc{binding['channels']}_d{binding['hidden']}"
        f"_{binding['layout']}"
    )


def config(binding, dtype):
    n, hc, d = binding["tokens"], binding["channels"], binding["hidden"]
    strided = binding["layout"] == "all_strided"
    inputs = {}
    for name, shape, value_dtype in (
        ("x", [n, d], dtype),
        ("residual", [n, hc, d], dtype),
        ("post", [n, hc], torch.float32),
    ):
        if strided:
            parent_shape = [*shape[:-1], 2 * shape[-1] + 2]
            strides = _strides(parent_shape, False)
            strides[-1] = 2
        else:
            parent_shape = shape
            strides = _strides(shape, False)
        inputs[name] = {
            "shape": shape,
            "dtype": str(value_dtype),
            "stride": strides,
            "storage_offset": int(strided),
            "parent_shape": parent_shape,
        }
    return {
        "inputs": inputs,
        "outputs": {
            "output": {
                "shape": [n, hc, d],
                "dtype": str(dtype),
                "stride": [hc * d, d, 1],
                "aliases_inputs": False,
            }
        },
        "dims": dict(binding),
        "mutation": "none",
        "input_aliasing": "none in benchmark; legal overlap tested separately",
        "arithmetic": "FP32 multiply, then FP32 add, then output cast; no FMA",
        "tolerance": {"rtol": 0, "atol": 0, "equal_nan": True},
        "timing_scope": "complete public callable including allocation and contiguous copies",
        "source": {
            "native_api": f"{NATIVE_MODULE}.{NATIVE_NAME}",
            "commit": SOURCE_PIN,
            "path": SOURCE_PATH,
            "sha256": SOURCE_SHA256,
            "model_shape": "HY V4 hidden=6144, hc_mult=4; token counts are test selections",
        },
        "native_artifact": NATIVE_ARTIFACT,
        "native_unavailable_reason": NATIVE_UNAVAILABLE_REASON,
    }
