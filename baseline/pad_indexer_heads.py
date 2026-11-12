# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""GLM5Next native import and shared no-Graph benchmark workloads."""

import math

import torch

OP_NAME = "pad_indexer_heads"
DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = None
_RESOLVED_NATIVE = None


def native():
    """Import the installed vLLM entry, or report why it is unavailable."""
    global NATIVE_UNAVAILABLE_REASON, _RESOLVED_NATIVE
    _RESOLVED_NATIVE = None
    try:
        from vllm.models.glm5next.common.attention import _pad_indexer_heads
    except ImportError as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    if not callable(_pad_indexer_heads):
        NATIVE_UNAVAILABLE_REASON = "vLLM _pad_indexer_heads is not callable"
        return None
    NATIVE_UNAVAILABLE_REASON = None
    _RESOLVED_NATIVE = _pad_indexer_heads
    return _pad_indexer_heads


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
    original = getattr(_RESOLVED_NATIVE, "_torchdynamo_orig_callable", None)
    if original is not None:
        # Independent configurations exceed Dynamo's global recompile budget.
        # Clear only this native leaf before qualification/warmup, never inside
        # the measured callable; otherwise Dynamo may silently time eager cat.
        torch._dynamo.reset_code(original.__code__)
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
    }
