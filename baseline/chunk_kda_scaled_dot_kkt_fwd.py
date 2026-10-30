# SPDX-License-Identifier: Apache-2.0
"""Native-only workload definition; never imported by production kernels."""
import importlib

import torch

OP_NAME = "chunk_kda_scaled_dot_kkt_fwd"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False


_SHAPES = [
    (1, 1, 1, 64, 64),
    (1, 65, 4, 128, 64),
    (1, 512, 8, 128, 64),
    (1, 2048, 16, 128, 64),
    (1, 513, 4, 128, 64, 17, 0, 64, 432),
    (1, 1024, 8, 128, 64, 128, 128, 128, 128, 128, 128, 128, 128),
]


def native():
    """Resolve the native vLLM KDA stage; return None when unavailable."""
    try:
        mod = importlib.import_module("vllm.model_executor.layers.fla.ops.kda")
    except ImportError:
        return None
    op = getattr(mod, OP_NAME, None)
    return op if callable(op) else None


def grid():
    return [
        dict(B=b, T=t, H=h, K=k, BT=bt, lengths=lengths or None)
        for b, t, h, k, bt, *lengths in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    b, t, h, k, bt = [binding[n] for n in ("B", "T", "H", "K", "BT")]
    gen = torch.Generator().manual_seed(2702)
    q, key = [
        (torch.randn((b, t, h, k), generator=gen) * 0.2).to(dtype) for _ in range(2)
    ]
    gate = torch.empty((b, t, h, k), dtype=torch.float32)
    lengths = binding["lengths"]
    spans = (
        [(0, sum(lengths[:i]), sum(lengths[: i + 1])) for i in range(len(lengths))]
        if lengths is not None
        else [(ib, 0, t) for ib in range(b)]
    )
    for ib, start, end in spans:
        for begin in range(start, end, bt):
            n = min(bt, end - begin)
            gate[ib, begin : begin + n] = (
                -torch.rand((n, h, k), generator=gen) * 0.1
            ).cumsum(0)
    beta = torch.rand((b, t, h), generator=gen)
    kw = dict(scale=k**-0.5, chunk_size=bt, output_dtype=torch.float32)
    if lengths is not None:
        cu = [0]
        for n in lengths:
            cu.append(cu[-1] + n)
        kw["cu_seqlens"] = torch.tensor(cu, device=device, dtype=torch.int64)
        kw["chunk_indices"] = torch.tensor(
            [(i, c) for i, n in enumerate(lengths) for c in range((n + bt - 1) // bt)],
            device=device,
            dtype=torch.int64,
        ).reshape(-1, 2)
    return tuple(x.to(device).contiguous() for x in (q, key, gate, beta)), kw


def key_shape(binding):
    return [binding[n] for n in ("B", "T", "H", "K", "BT")] + (binding["lengths"] or [])


def config(binding, dtype):
    """Describe the actual inputs and the two newly allocated output matrices."""
    b, t, h, k, bt = [binding[n] for n in ("B", "T", "H", "K", "BT")]
    lengths = binding["lengths"]
    inputs = {
        "q": {"shape": [b, t, h, k], "dtype": str(dtype)},
        "k": {"shape": [b, t, h, k], "dtype": str(dtype)},
        "gk": {
            "shape": [b, t, h, k],
            "dtype": str(torch.float32),
            "note": "chunk-local cumulative log2 gates",
        },
        "beta": {"shape": [b, t, h], "dtype": str(torch.float32)},
        "scale": {"scalar": k**-0.5},
        "chunk_size": {"scalar": bt},
        "output_dtype": {"scalar": str(torch.float32)},
        "cu_seqlens": {"scalar": None},
        "chunk_indices": {"scalar": None},
    }
    if lengths is not None:
        num_chunks = sum((n + bt - 1) // bt for n in lengths)
        inputs["cu_seqlens"] = {
            "shape": [len(lengths) + 1],
            "dtype": str(torch.int64),
        }
        inputs["chunk_indices"] = {
            "shape": [num_chunks, 2],
            "dtype": str(torch.int64),
        }
    return {
        "inputs": inputs,
        "outputs": {
            "A": {"shape": [b, t, h, bt], "dtype": str(torch.float32)},
            "Aqk": {"shape": [b, t, h, bt], "dtype": str(torch.float32)},
        },
        "dims": {
            "B": b,
            "T": t,
            "H": h,
            "K": k,
            "BT": bt,
            "lengths": lengths,
            "layout": "contiguous seq-first",
        },
    }
