# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native FLA suboperator and model-derived workload definitions."""

import importlib
import math

import torch
import torch.nn.functional as F

OP_NAME = "chunk_kda_fwd_intra_token_parallel"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = True
MUTATED_INPUTS = (4, 5)
NATIVE_UNAVAILABLE_REASON = None

_INPUT_SEED = 20261027

# (label, heads, evidence): all profiles use K=128, HV=H, TP=1.
_MODEL_PROFILES = (
    ("kimi-linear-48b-a3b", 32, "chunk-path-safe-gate-false"),
    ("kimi-k3", 96, "geometry-only-safe-gate-true-bypasses-target"),
    ("glm-5.3-flash", 64, "geometry-only-exact-call-unverified"),
)
# B, per-sequence lengths. Multiple lengths mean packed B=1, not padded batch.
# B/T are selected workloads, not observed production traces. No decode cases:
# the inspected Kimi model switches to fused_recurrent for q_len <= 64.
_WORKLOADS = (
    (1, (128,)),
    (1, (512,)),
    (1, (2048,)),
    (1, (8192,)),
    (1, (32768,)),
    (4, (2048,)),
    (1, (1024,) * 8),
    (1, (1300, 547, 2048, 963, 271, 3063)),
)


def native():
    """Return the exact upstream FLA suboperator, or None if unavailable."""
    global NATIVE_UNAVAILABLE_REASON
    try:
        module = importlib.import_module("fla.ops.kda.chunk_intra_token_parallel")
    except ImportError as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    op = getattr(module, OP_NAME, None)
    NATIVE_UNAVAILABLE_REASON = (
        None if callable(op) else f"{module.__name__}.{OP_NAME} absent"
    )
    return op if callable(op) else None


def grid():
    return [
        dict(
            profile=label,
            evidence=evidence,
            B=batch,
            lengths=list(lengths),
            H=heads,
            HV=heads,
            K=128,
            BT=64,
            BC=16,
        )
        for label, heads, evidence in _MODEL_PROFILES
        for batch, lengths in _WORKLOADS
    ]


def build_inputs(binding, dtype, device):
    """Build post-normalization/post-cumsum inputs outside the measured region."""
    batch, lengths, heads = binding["B"], binding["lengths"], binding["H"]
    torch.manual_seed(_INPUT_SEED)
    lengths = tuple(lengths)
    if batch < 1 or not lengths or min(lengths) <= 64:
        raise ValueError("prefill workloads require positive batch and lengths > 64")
    packed = len(lengths) > 1
    if packed and batch != 1:
        raise ValueError("packed workloads require B=1")
    total = sum(lengths)
    shape = (batch, total, heads, 128)
    q = F.normalize(torch.randn(shape, device=device, dtype=torch.float32), dim=-1)
    k = F.normalize(torch.randn_like(q), dim=-1)
    # Synthetic negative per-token log2 gates, cumulative within each BT=64
    # chunk and each sequence. This is input preparation, not a timed baseline.
    gk = torch.empty_like(q)
    offset = 0
    for length in lengths:
        for start in range(0, length, 64):
            end = min(start + 64, length)
            part = gk[:, offset + start : offset + end]
            part.copy_((-torch.rand_like(part) * 0.1).cumsum(dim=1))
        offset += length
    beta = torch.randn(shape[:-1], device=device, dtype=torch.float32).sigmoid()
    aqk = torch.zeros((*shape[:-1], 64), device=device, dtype=dtype)
    akk = torch.zeros((*shape[:-1], 16), device=device, dtype=torch.float32)
    cu = None
    if packed:
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        cu = torch.tensor(boundaries, device=device, dtype=torch.int64)
    return (
        q.to(dtype).contiguous(),
        k.to(dtype).contiguous(),
        gk.contiguous(),
        beta.contiguous(),
        aqk,
        akk,
        128**-0.5,
    ), {"cu_seqlens": cu, "chunk_size": 64, "sub_chunk_size": 16, "use_graph": False}


def key_shape(binding):
    return [
        binding["B"],
        sum(binding["lengths"]),
        binding["H"],
        binding["HV"],
        binding["K"],
        binding["BT"],
        binding["BC"],
        list(binding["lengths"]),
    ]


def config(binding, dtype):
    """Describe actual tensors, scalars, output mutation and model provenance."""
    b, h, k, bt, bc = (binding[n] for n in ("B", "H", "K", "BT", "BC"))
    lengths = list(binding["lengths"])
    t = sum(lengths)
    qk_shape = [b, t, h, k]
    aqk = {
        "shape": [b, t, h, bt],
        "dtype": str(dtype),
        "note": "preallocated, written in place",
    }
    akk = {
        "shape": [b, t, h, bc],
        "dtype": str(torch.float32),
        "note": "preallocated, written in place",
    }
    result = {
        "inputs": {
            "q": {"shape": qk_shape, "dtype": str(dtype)},
            "k": {"shape": qk_shape, "dtype": str(dtype)},
            "gk": {
                "shape": qk_shape,
                "dtype": str(torch.float32),
                "domain": "chunk cumulative log2",
            },
            "beta": {"shape": [b, t, h], "dtype": str(torch.float32)},
            "Aqk": aqk,
            "Akk": akk,
            "scale": {"scalar": k**-0.5},
            "chunk_size": {"scalar": bt},
            "sub_chunk_size": {"scalar": bc},
            "use_graph": {"scalar": False},
            "cu_seqlens": (
                {"shape": [len(lengths) + 1], "dtype": str(torch.int64)}
                if len(lengths) > 1
                else None
            ),
        },
        "outputs": {"Aqk": dict(aqk), "Akk": dict(akk)},
        "dims": {
            "B": b,
            "T": t,
            "H": h,
            "HV": binding["HV"],
            "K": k,
            "BT": bt,
            "BC": bc,
            "lengths": lengths,
        },
        "layout": "contiguous seq-first",
        "model_profile": binding["profile"],
        "evidence": binding["evidence"],
        "tp": 1,
        "input_seed": _INPUT_SEED,
    }
    for section in ("inputs", "outputs"):
        for tensor in result[section].values():
            if isinstance(tensor, dict) and "shape" in tensor:
                shape = tensor["shape"]
                tensor["stride"] = [
                    math.prod(shape[i + 1 :]) for i in range(len(shape))
                ]
    result["graph"] = False
    result["outputs"]["Aqk"]["aliases"] = "Aqk"
    result["outputs"]["Akk"]["aliases"] = "Akk"
    result["accumulate_dtype"] = "torch.float32"
    return result
