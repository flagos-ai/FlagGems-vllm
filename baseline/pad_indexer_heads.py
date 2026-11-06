# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Pinned GLM5Next compiled leaf and shared no-Graph benchmark workloads."""

import ast
import hashlib
import importlib
import math
import os
from pathlib import Path

import torch

OP_NAME = "pad_indexer_heads"
DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn]
IS_INPLACE = False
SOURCE_COMMIT = "a17e79abde45715e79a00deb99ad34be49162231"
SOURCE_URL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/"
    f"{SOURCE_COMMIT}/vllm/models/glm5next/common/attention.py"
)
SOURCE_SHA256 = "e83364215dccc2871e47906a8a5286852bd6a922a6a322394ff8f27db80952e0"
NATIVE_UNAVAILABLE_REASON = None
NATIVE_PROVENANCE = None
_RESOLVED_NATIVE = None


def native():
    """Resolve the real compiled entry; optionally load its hash-verified leaf.

    Older vLLM wheels predate GLM5Next. PAD_INDEXER_NATIVE_SOURCE may point to
    the complete pinned attention.py. Only its original compile configuration
    and decorated leaf are executed, using vLLM's actual platform backend and
    graph-partition options. No mathematical replacement is synthesized.
    """
    global NATIVE_UNAVAILABLE_REASON, NATIVE_PROVENANCE, _RESOLVED_NATIVE
    source = os.environ.get("PAD_INDEXER_NATIVE_SOURCE")
    if source:
        raw = Path(source).read_bytes()
        if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
            raise ValueError("pad_indexer_heads native source SHA256 mismatch")
        try:
            from vllm.model_executor.utils import maybe_disable_graph_partition
            from vllm.platforms import current_platform
        except (ImportError, AttributeError) as exc:
            NATIVE_UNAVAILABLE_REASON = repr(exc)
            return None
        tree = ast.parse(raw, filename=source)
        nodes = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "_INDEXER_COMPILE"
                    for target in node.targets
                )
            )
            or (isinstance(node, ast.FunctionDef) and node.name == "_pad_indexer_heads")
        ]
        if len(nodes) != 2:
            raise ValueError("pinned native compile configuration/leaf missing")
        namespace = {
            "torch": torch,
            "current_platform": current_platform,
            "maybe_disable_graph_partition": maybe_disable_graph_partition,
            "__name__": "pinned_vllm_glm5next_attention",
        }
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), source, "exec"), namespace
        )
        op = namespace["_pad_indexer_heads"]
        NATIVE_PROVENANCE = {
            "kind": "pinned-source-compiled-leaf",
            "url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "compile": namespace["_INDEXER_COMPILE"],
        }
    else:
        try:
            module = importlib.import_module("vllm.models.glm5next.common.attention")
            op = module._pad_indexer_heads
        except (ImportError, AttributeError) as exc:
            NATIVE_UNAVAILABLE_REASON = repr(exc)
            return None
        NATIVE_PROVENANCE = {"kind": "installed-vllm", "file": module.__file__}
    if not callable(op) or not hasattr(op, "_torchdynamo_orig_callable"):
        NATIVE_UNAVAILABLE_REASON = "native entry is not a torch.compile callable"
        return None
    NATIVE_UNAVAILABLE_REASON = None
    _RESOLVED_NATIVE = op
    NATIVE_PROVENANCE["cache_reset_outside_timing"] = True
    return op


def grid():
    # (tokens, heads, tail tuple, added heads, layout). Rank2 is the weights call.
    return [
        (t, 16, tail, 16, "contiguous")
        for t in (1, 7, 128, 8192)
        for tail in ((), (128,))
    ] + [
        (7, 31, (7,), 1, "contiguous"),
        (128, 32, (128,), 0, "contiguous"),
        (7, 63, (), 1, "row_gap"),
        (7, 16, (3, 5), 3, "tail_gap"),
    ]


def build_inputs(binding, dtype, device):
    if _RESOLVED_NATIVE is not None:
        # Independent configurations exceed Dynamo's global recompile budget.
        # Clear only this native leaf before qualification/warmup, never inside
        # the measured callable; otherwise Dynamo may silently time eager cat.
        torch._dynamo.reset_code(_RESOLVED_NATIVE._torchdynamo_orig_callable.__code__)
    t, h, tail, pad, layout = binding
    tail = tuple(tail)
    shape = (t, h, *tail)
    # Construct through bytes so FP8 storage support is independent of arithmetic.
    itemsize = torch.empty((), dtype=dtype).element_size()
    source_shape = (t * 2, h, *tail) if layout == "row_gap" else shape
    if layout == "tail_gap":
        source_shape = (*shape[:-1], shape[-1] * 2)
    generator = torch.Generator().manual_seed(20261106)
    # Transfer flat bytes: some vendor copies select channels-last for rank4,
    # which would make a subsequent dtype view illegal.
    raw = torch.randint(
        0,
        256,
        (math.prod(source_shape) * itemsize,),
        dtype=torch.uint8,
        generator=generator,
    )
    x = raw.to(device).view(dtype).reshape(source_shape)
    if layout == "row_gap":
        x = x[::2]
    elif layout == "tail_gap":
        x = x[..., ::2]
    return (x, pad), {}


def key_shape(binding):
    t, h, tail, pad, layout = binding
    return [t, h, *tail, pad, layout]


def config(binding, dtype):
    t, h, tail, pad, layout = binding
    shape = (t, h, *tail)
    storage_shape = list(shape)
    if layout == "tail_gap":
        storage_shape[-1] *= 2
    strides = [math.prod(storage_shape[axis + 1 :]) for axis in range(len(shape))]
    if layout == "row_gap":
        strides[0] *= 2
    elif layout == "tail_gap":
        strides[-1] *= 2
    return {
        "inputs": {
            "x": {
                "shape": list(shape),
                "stride": strides,
                "dtype": str(dtype),
                "layout": layout,
            },
            "pad": {"scalar": pad},
        },
        "outputs": {
            "output": {
                "shape": [t, h + pad, *tail],
                "dtype": str(dtype),
                "contiguous": True,
            }
        },
        "alias": "fresh output, including pad=0; input unchanged",
        "copy": "storage bits, zero padding encoding",
        "source_commit": SOURCE_COMMIT,
    }
