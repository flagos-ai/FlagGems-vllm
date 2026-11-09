# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Pinned Kimi-K3 output-gate workloads and exact native resolution."""

import hashlib
import importlib
import inspect
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import torch

OP_NAME = "gate_sigmoid_mul"
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
IS_INPLACE = False
MUTATED_INPUTS = ()
NATIVE_UNAVAILABLE_REASON = None
NATIVE_ARTIFACT = None
NATIVE_MODULE = "vllm.models.kimi_k3.nvidia.mla"
NATIVE_NAME = "_gate_sigmoid_mul"
SOURCE_PIN = "a17e79abde45715e79a00deb99ad34be49162231"
SOURCE_PATH = "vllm/models/kimi_k3/nvidia/mla.py"
SOURCE_SHA256 = "d41ee77a5690b6960bf3ec3957bacfafbe1956cc1e386316aecb884d819915e2"
MODEL_PIN = "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
TOKENS = (1, 16, 64, 256, 511, 512, 1024, 4096)
WIDTHS = (1536, 3072, 6144, 12288)
LAYOUTS = ("contiguous", "split_gate")
GATE_OFFSET = 1536 + 512 + 64
_DTYPES = {str(dtype): dtype for dtype in DTYPES}
_DTYPES.update({str(dtype).split(".", 1)[1]: dtype for dtype in DTYPES})


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
        source_hash = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
        NATIVE_ARTIFACT["source_sha256"] = source_hash
        NATIVE_ARTIFACT["reference_file_matches"] = source_hash == SOURCE_SHA256
    except Exception as exc:
        NATIVE_ARTIFACT["source_error"] = f"{type(exc).__name__}: {exc}"
    try:
        NATIVE_ARTIFACT["callable_source_sha256"] = hashlib.sha256(
            inspect.getsource(original).encode("utf-8")
        ).hexdigest()
    except Exception as exc:
        NATIVE_ARTIFACT["callable_source_error"] = f"{type(exc).__name__}: {exc}"
    return op


def grid():
    """64 bindings x three dtypes; TP choices are workload selections."""
    return [
        {"tokens": tokens, "width": width, "layout": layout}
        for tokens in TOKENS
        for width in WIDTHS
        for layout in LAYOUTS
    ]


def _input_dtypes(binding, dtype):
    def resolve(value):
        return _DTYPES[value] if isinstance(value, str) else value

    attn_dtype = resolve(binding.get("attn_dtype", dtype))
    gate_dtype = resolve(binding.get("gate_dtype", dtype))
    if attn_dtype not in DTYPES or gate_dtype not in DTYPES:
        raise ValueError("gate workloads require float16, bfloat16 or float32")
    return attn_dtype, gate_dtype


def build_inputs(binding, dtype, device):
    tokens, width = binding["tokens"], binding["width"]
    attn_dtype, gate_dtype = _input_dtypes(binding, dtype)
    attn_out = torch.randn((tokens, width), dtype=attn_dtype, device=device)
    if binding["layout"] == "contiguous":
        gate = torch.randn((tokens, width), dtype=gate_dtype, device=device)
    elif binding["layout"] == "split_gate":
        parent = torch.randn(
            (tokens, width + GATE_OFFSET), dtype=gate_dtype, device=device
        )
        gate = parent[:, GATE_OFFSET:]
    else:
        raise ValueError(f"Unknown gate layout: {binding['layout']}")
    return (attn_out, gate), {}


def key_shape(binding):
    key = f"t{binding['tokens']}_w{binding['width']}_{binding['layout']}"
    if "attn_dtype" in binding or "gate_dtype" in binding:
        key += (
            f"_attn{binding.get('attn_dtype', 'default')}"
            f"_gate{binding.get('gate_dtype', 'default')}"
        )
    return key


def config(binding, dtype):
    tokens, width = binding["tokens"], binding["width"]
    attn_dtype, gate_dtype = _input_dtypes(binding, dtype)
    split = binding["layout"] == "split_gate"
    gate_stride = width + GATE_OFFSET if split else width
    gate_offset = GATE_OFFSET if split else 0
    return {
        "inputs": {
            "attn_out": {
                "shape": [tokens, width],
                "stride": [width, 1],
                "storage_offset": 0,
                "dtype": str(attn_dtype),
            },
            "gate": {
                "shape": [tokens, width],
                "stride": [gate_stride, 1],
                "storage_offset": gate_offset,
                "dtype": str(gate_dtype),
                "parent_shape": [tokens, gate_stride],
            },
        },
        "outputs": {
            "out": {
                "shape": [tokens, width],
                "dtype": str(torch.promote_types(attn_dtype, gate_dtype)),
                "aliases_inputs": False,
            }
        },
        "dims": {
            **binding,
            "tensor_parallel_selection": 12288 // width,
            "layout": binding["layout"],
        },
        "coverage": (
            "model_bfloat16_hotspot"
            if attn_dtype == gate_dtype == torch.bfloat16
            else "supplemental_dtype"
        ),
        "mutation": "none",
        "native_artifact": NATIVE_ARTIFACT,
        "source": {
            "native_api": f"{NATIVE_MODULE}.{NATIVE_NAME}",
            "commit": SOURCE_PIN,
            "path": SOURCE_PATH,
            "sha256": SOURCE_SHA256,
            "pin_scope": "target reference; installed native artifact unverified",
            "model_config_repo": "moonshotai/Kimi-K3",
            "model_config_commit": MODEL_PIN,
            "compile": {"backend": "current_platform.simple_compile_backend"},
            "rounding": "native compiled callable; intermediate rounding unverified",
            "shape_evidence": "model geometry; token/TP choices are test selections",
        },
    }
