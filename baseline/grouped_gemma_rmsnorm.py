# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact Qwen4Exp NVIDIA source-native RMSNorm and reproducible workloads.

This module never imports a candidate implementation. The optional source path
explicitly selects an unchanged pinned-source environment, rather than silently
substituting for a missing installed native operator.
"""

import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

OP_NAME = "grouped_gemma_rmsnorm"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
IS_INPLACE = False
NATIVE_MODULE = "vllm.models.qwen4_exp.nvidia.ops.hc"
SOURCE_REFERENCE = "a17e79abde45715e79a00deb99ad34be49162231"
PINNED_SOURCE_SHA256 = (
    "b30b6e3d15a6d69cd1b6c79182721e386fd0aa282e0c8fcfd42ef7a5246eccd0"
)
NATIVE_UNAVAILABLE_REASON = None
NATIVE_PROVENANCE = {}

# Decode, odd reductions, model-sized prefill, and the 8192-element group path.
SHAPES = [
    (1, 1, 1),
    (1, 3, 1),
    (1, 4, 4),
    (3, 12, 4),
    (17, 508, 4),
    (3, 762, 2),
    (128, 5120, 2),
    (1, 8192, 1),
    (128, 10240, 4),
    (512, 10240, 4),
    (17, 32768, 4),
    (128, 32768, 4),
    (3, 16386, 2),
]


def native(backend=None):
    """Return the original callable or None, retaining failure and provenance."""
    global NATIVE_UNAVAILABLE_REASON, NATIVE_PROVENANCE
    NATIVE_UNAVAILABLE_REASON = None
    NATIVE_PROVENANCE = {}
    backend = (
        backend
        or os.environ.get("GEMS_VENDOR")
        or os.environ.get("DNN_VENDOR")
        or os.environ.get("FLAGGEMS_VENDOR")
    )
    if backend is None:
        backend = (
            "nvidia"
            if "+cu" in torch.__version__ and not torch.version.hip
            else "unverified"
        )
    if backend != "nvidia":
        NATIVE_UNAVAILABLE_REASON = (
            f"No audited {backend} vendor-native grouped Gemma RMSNorm; "
            "NVIDIA source is not a same-device vendor baseline"
        )
        return None
    selected_source = os.environ.get("FGV_GEMMA_PINNED_NATIVE_SOURCE")
    try:
        if selected_source:
            path = Path(selected_source).resolve(strict=True)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != PINNED_SOURCE_SHA256:
                raise ValueError(
                    "explicit pinned native source has an unexpected SHA256"
                )
            name = "_fgv_pinned_source_native_qwen4_hc_" + digest[:16]
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
            environment = "explicit-pinned-source-native"
        else:
            module = importlib.import_module(NATIVE_MODULE)
            path = Path(module.__file__).resolve(strict=True)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            environment = "installed-vllm-native"
        op = getattr(module, OP_NAME)
        if not callable(op) or op.__module__ != module.__name__:
            raise TypeError("native must be the original source callable")
        NATIVE_PROVENANCE = {
            "environment": environment,
            "module": op.__module__,
            "file": str(path),
            "sha256": digest,
            "api": NATIVE_MODULE + "." + OP_NAME,
            "source_reference": SOURCE_REFERENCE if selected_source else None,
        }
        return op
    except Exception as exc:
        NATIVE_UNAVAILABLE_REASON = (
            f"{NATIVE_MODULE}.{OP_NAME}: {type(exc).__name__}: {exc}"
        )
        return None


def grid():
    return [
        {
            "n": n,
            "dim": dim,
            "num_groups": groups,
            "weight_dtype": str(weight_dtype).split(".")[-1],
            "shared": shared,
            "row_gap": 7 if index % 2 else 0,
            "weight_rank": 2 if shared else 1,
            "eps": 1e-6 if index % 2 else 1e-5,
            "distribution": "normal",
            "seed": 20261102,
        }
        for index, (n, dim, groups) in enumerate(SHAPES)
        for shared in (False, True)
        for weight_dtype in DTYPES
    ]


def build_inputs(binding, dtype, device):
    n, dim = binding["n"], binding["dim"]
    groups = binding["num_groups"]
    generator = torch.Generator(device="cpu").manual_seed(binding["seed"])
    row_stride = dim + binding["row_gap"]
    # Generate on CPU for deterministic values across vendor RNG implementations.
    backing = torch.randn((n, row_stride), generator=generator, dtype=torch.float32)
    backing = backing.to(device=device, dtype=dtype)
    x = backing[:, :dim]
    weight_numel = dim // groups if binding["shared"] else dim
    weight_shape = (1, weight_numel) if binding["weight_rank"] == 2 else (weight_numel,)
    weight = torch.randn(weight_shape, generator=generator, dtype=torch.float32)
    weight = weight.to(device=device, dtype=getattr(torch, binding["weight_dtype"]))
    return (x, weight, binding["eps"], groups), {}


def key_shape(binding):
    # Include every binding field; x dtype is the collector's separate dtype key.
    return json.dumps(binding, sort_keys=True, separators=(",", ":"))


def config(binding, dtype):
    n, dim = binding["n"], binding["dim"]
    weight_numel = dim // binding["num_groups"] if binding["shared"] else dim
    weight_shape = [1, weight_numel] if binding["weight_rank"] == 2 else [weight_numel]
    weight_stride = [weight_numel, 1] if len(weight_shape) == 2 else [1]
    return {
        "inputs": {
            "x": {
                "shape": [n, dim],
                "dtype": str(dtype),
                "stride": [dim + binding["row_gap"], 1],
                "layout": "strided",
            },
            "weight": {
                "shape": weight_shape,
                "dtype": "torch." + binding["weight_dtype"],
                "stride": weight_stride,
                "layout": "contiguous",
                "shared": binding["shared"],
            },
            "eps": {"scalar": binding["eps"]},
            "num_groups": {"scalar": binding["num_groups"]},
        },
        "outputs": {
            "y": {
                "shape": [n, dim],
                "dtype": str(dtype),
                "stride": [dim, 1],
                "alias": "fresh; no input alias",
            }
        },
        "compute_dtype": "torch.float32",
        "affine": "z + z * weight; fused multiply-add allowed",
        "mutated_inputs": [],
        "binding": dict(binding),
    }
