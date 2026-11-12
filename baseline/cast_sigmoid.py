# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Pinned cast_sigmoid workloads and exact vLLM native resolution.

The native symbol is vLLM's ``_cast_sigmoid``, the fused FP32-cast + sigmoid
leaf used for the KDA / grouped-topk gating scores:

    @torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
    def _cast_sigmoid(x: torch.Tensor) -> torch.Tensor:
        return x.float().sigmoid()

It therefore always returns FP32 and only exists on vLLM builds that ship
``vllm/models/glm5next`` (v0.33.0 and newer). On older builds the import fails,
so ``native()`` returns ``None`` with ``NATIVE_UNAVAILABLE_REASON`` set and the
benchmark picks ``torch.sigmoid`` instead. ``native()`` never fabricates a
proxy callable for a missing symbol, so "native available" stays verifiable.
"""

import importlib
import inspect
from functools import lru_cache
from importlib import metadata

import torch

OP_NAME = "cast_sigmoid"
# vLLM calls _cast_sigmoid on the KDA beta tensor, whose dtype follows the
# model dtype; FP32 is supported as a no-op cast.
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
IS_INPLACE = False
MUTATED_INPUTS = ()
NATIVE_MODULE = "vllm.models.glm5next.common.kda"
NATIVE_NAME = "_cast_sigmoid"
NATIVE_OUT_DTYPE = torch.float32
NATIVE_UNAVAILABLE_REASON = None
NATIVE_ARTIFACT = None

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
    """Resolve ``vllm.models.glm5next.common.kda._cast_sigmoid``."""
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
        # vLLM decorates the leaf with torch.compile, so unwrap back to the
        # Python function before inspecting its source.
        NATIVE_ARTIFACT["callable_source"] = inspect.getsource(original).strip()
    except Exception as exc:
        NATIVE_ARTIFACT["callable_source_error"] = f"{type(exc).__name__}: {exc}"
    return op


def grid():
    """Six bindings x three dtypes: one binding per shape."""
    return [{"shape": list(shape)} for shape in SHAPES]


def key_shape(binding):
    """Simple unary op: the key is the first tensor's dimensions."""
    return list(binding["shape"])


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
        # The native op always writes FP32, whatever the input dtype.
        "outputs": {"output": {"shape": shape, "dtype": str(NATIVE_OUT_DTYPE)}},
        "dims": {"numel": int(torch.tensor(shape).prod().item())},
    }


def _strides(shape):
    strides, acc = [], 1
    for dim in reversed(shape):
        strides.append(acc)
        acc *= dim
    return reversed(strides)
