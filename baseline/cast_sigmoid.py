# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Pinned cast_sigmoid workloads and exact native resolution.

The native symbol is vLLM's ``_cast_sigmoid`` (grouped-topk gating scores).
It only exists on vLLM builds that ship the GLM-5-Next model package, so
``native()`` may legitimately return ``None``; that is recorded in
``NATIVE_UNAVAILABLE_REASON`` and never worked around with an eager
expression.
"""

import hashlib
import importlib
import inspect
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import torch

OP_NAME = "cast_sigmoid"
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
IS_INPLACE = False
MUTATED_INPUTS = ()
NATIVE_UNAVAILABLE_REASON = None
NATIVE_ARTIFACT = None
NATIVE_MODULE = "vllm.models.glm5next.common.kda"
NATIVE_NAME = "_cast_sigmoid"
NATIVE_OUT_DTYPE = torch.float32

# Small / medium / large, mirrored from benchmark/core_shapes.yaml.
SHAPES = (
    (1,),
    (1, 256),
    (128, 256),
    (4096, 256),
    (1048576,),
    (4096, 4096),
)


@lru_cache(maxsize=1)
def native():
    """Never substitute an eager or source-sliced expression for native."""
    global NATIVE_UNAVAILABLE_REASON, NATIVE_ARTIFACT
    try:
        module = importlib.import_module(NATIVE_MODULE)
        op = getattr(module, NATIVE_NAME)
    except Exception as exc:
        NATIVE_UNAVAILABLE_REASON = (
            f"{NATIVE_MODULE}.{NATIVE_NAME}: {type(exc).__name__}: {exc}"
        )
        return None
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = (
            f"{NATIVE_MODULE}.{NATIVE_NAME} is not callable ({type(op).__name__})"
        )
        return None
    NATIVE_UNAVAILABLE_REASON = None
    original = inspect.unwrap(op)
    NATIVE_ARTIFACT = {
        "resolved_api": f"{NATIVE_MODULE}.{NATIVE_NAME}",
        "callable_module": getattr(original, "__module__", None),
    }
    try:
        NATIVE_ARTIFACT["vllm_version"] = metadata.version("vllm")
    except Exception as exc:
        NATIVE_ARTIFACT["version_error"] = f"{type(exc).__name__}: {exc}"
    try:
        source_file = inspect.getsourcefile(original) or getattr(
            module, "__file__", None
        )
        NATIVE_ARTIFACT["source_file"] = source_file
        NATIVE_ARTIFACT["source_sha256"] = hashlib.sha256(
            Path(source_file).read_bytes()
        ).hexdigest()
    except Exception as exc:
        NATIVE_ARTIFACT["source_error"] = f"{type(exc).__name__}: {exc}"
    return op


def grid():
    """Six bindings x three dtypes: one binding per shape."""
    return [{"shape": list(shape)} for shape in SHAPES]


def key_shape(binding):
    return "x".join(str(dim) for dim in binding["shape"]) or "scalar"


def build_inputs(binding, dtype, device):
    if dtype not in DTYPES:
        raise ValueError("cast_sigmoid workloads require float16, bfloat16 or fp32")
    x = torch.randn(tuple(binding["shape"]), dtype=dtype, device=device)
    return (x,), {}


def config(binding, dtype):
    shape = list(binding["shape"])
    return {
        "inputs": {
            "x": {
                "shape": shape,
                "stride": list(_strides(shape)),
                "storage_offset": 0,
                "dtype": str(dtype),
            }
        },
        "output": {"shape": shape, "dtype": str(NATIVE_OUT_DTYPE)},
        "attributes": {"out_dtype": str(NATIVE_OUT_DTYPE)},
    }


def _strides(shape):
    strides, acc = [], 1
    for dim in reversed(shape):
        strides.append(acc)
        acc *= dim
    return reversed(strides)
