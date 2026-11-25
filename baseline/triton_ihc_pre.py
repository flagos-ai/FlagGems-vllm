# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""HY V4 iHC pre native resolution and the shared benchmark workload grid."""

import hashlib
import importlib
import inspect
from contextlib import contextmanager
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import torch

OP_NAME = "triton_ihc_pre"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ()
NATIVE_UNAVAILABLE_REASON = None
NATIVE_ARTIFACT = None
NATIVE_MODULE = "vllm.models.hy_v4.nvidia.triton_ihc"
NATIVE_NAME = "triton_ihc_pre"
SOURCE_PIN = "a17e79abde45715e79a00deb99ad34be49162231"
SOURCE_PATH = "vllm/models/hy_v4/nvidia/triton_ihc.py"
SOURCE_SHA256 = "f47036732264faaf08dd039aad89e0cf1d0eba445767c1ae5c64e8ed2d1ea215"


@contextmanager
def _cpu_single_thread():
    """Bound this operator's CPU preparation/validation, restoring caller state."""
    previous = torch.get_num_threads()
    if previous == 1:
        yield
        return
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


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
    """Model schedule boundaries plus odd channel/reduction and copy workloads."""
    rows = [
        (n, 4, 6144, "contiguous") for n in (1, 7, 64, 255, 256, 257, 1024, 4096, 8192)
    ]
    rows += [
        (1, 1, 1, "contiguous"),
        (7, 3, 257, "contiguous"),
        (256, 3, 257, "contiguous"),
        (64, 4, 4096, "contiguous"),
    ]
    rows += [(n, 4, 6144, "all_strided") for n in (1, 256, 4096)]
    return [
        dict(tokens=n, channels=hc, hidden=d, layout=layout)
        for n, hc, d, layout in rows
    ]


def _tensor(shape, dtype, device, strided=False):
    if strided:
        parent = torch.randn(
            (*shape[:-1], 2 * shape[-1] + 2), dtype=dtype, device=device
        )
        return parent[..., 1:-1:2]
    return torch.randn(shape, dtype=dtype, device=device)


@_cpu_single_thread()
def build_inputs(binding, dtype, device):
    if dtype not in DTYPES:
        raise ValueError("iHC pre requires float16 or bfloat16 x")
    n, hc, d = (binding[k] for k in ("tokens", "channels", "hidden"))
    if binding["layout"] not in ("contiguous", "all_strided"):
        raise ValueError("unknown layout")
    torch.manual_seed(42)
    strided = binding["layout"] == "all_strided"
    x = _tensor((n, hc, d), dtype, device, strided)
    weight = _tensor((2 * hc, hc * d), torch.float32, device)
    scale = _tensor((2,), torch.float32, device, strided)
    scale.mul_(0.01)
    base = _tensor((2 * hc,), torch.float32, device, strided)
    return (x, weight, scale, base, 2.0, 1e-6, 1e-5), {}


def key_shape(binding):
    return f"n{binding['tokens']}_hc{binding['channels']}_d{binding['hidden']}_{binding['layout']}"


def config(binding, dtype):
    n, hc, d = (binding[k] for k in ("tokens", "channels", "hidden"))
    inputs = {}
    for name, shape, dt in (
        ("x", [n, hc, d], dtype),
        ("weight", [2 * hc, hc * d], torch.float32),
        ("scale", [2], torch.float32),
        ("base", [2 * hc], torch.float32),
    ):
        strided = binding["layout"] == "all_strided" and name != "weight"
        parent = [*shape[:-1], 2 * shape[-1] + 2] if strided else shape
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * parent[i + 1]
        if strided:
            strides[-1] = 2
        inputs[name] = dict(
            shape=shape, dtype=str(dt), stride=strides, storage_offset=int(strided)
        )
    inputs.update(
        {
            k: {"scalar": v}
            for k, v in (("magnitude", 2.0), ("hc_eps", 1e-6), ("norm_eps", 1e-5))
        }
    )
    return dict(
        inputs=inputs,
        outputs={
            "y": dict(shape=[n, d], dtype=str(dtype), stride=[d, 1]),
            "post": dict(shape=[n, hc], dtype=str(torch.float32), stride=[hc, 1]),
        },
        dims=dict(binding),
        mutation="none",
        input_aliasing="none",
        arithmetic="FP32 projection and gates, no FMA, final y cast; norm divisor HC*D",
        tolerance={"y": dict(rtol=1e-2, atol=2e-2), "post": dict(rtol=1e-5, atol=2e-5)},
        timing_scope="complete public callable including allocation and contiguous copies",
        source={
            "native_api": f"{NATIVE_MODULE}.{NATIVE_NAME}",
            "reference_commit": SOURCE_PIN,
        },
        native_artifact=NATIVE_ARTIFACT,
        native_unavailable_reason=NATIVE_UNAVAILABLE_REASON,
    )
