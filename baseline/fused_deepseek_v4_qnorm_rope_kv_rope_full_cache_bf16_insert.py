# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Workloads for the DeepSeek V4 full BF16 cache insert native operator."""

import torch

OP_NAME = "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0, 2)
NATIVE_UNAVAILABLE_REASON = None
SEED = 20261126
EPS = 1e-6
D = 512
R = 64
FIELDS = ("N", "I", "H", "S", "apply_q_norm", "negative_slots", "page_padding")
CORE = (
    (1, 1, 1, 16, True, "none", 0),
    (1, 0, 8, 16, False, "none", 0),
    (5, 3, 8, 4, True, "alternate", 16),
    (5, 3, 8, 4, False, "alternate", 16),
    (5, 3, 8, 4, True, "none", 16),
    (5, 3, 8, 4, True, "alternate", 0),
    (5, 5, 17, 16, False, "all", 32),
    (256, 256, 17, 64, True, "alternate", 64),
)
EXTRA = (
    (1024, 1024, 8, 64, False, "none", 0),
    (1024, 769, 17, 16, True, "alternate", 32),
)


def native():
    global NATIVE_UNAVAILABLE_REASON
    try:
        op = getattr(torch.ops._C, OP_NAME, None)
        if op is None:
            import vllm._custom_ops  # noqa: F401 - registers torch.ops._C

            op = getattr(torch.ops._C, OP_NAME)
        if not callable(op):
            raise TypeError(f"torch.ops._C.{OP_NAME} is not callable")
    except Exception as exc:
        NATIVE_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return op


def grid():
    return [dict(zip(FIELDS, shape)) for shape in (*CORE, *EXTRA)]


def _page_layout(b):
    block_size = b["S"]
    pages = (b["I"] + block_size - 1) // block_size + 2
    return pages, block_size * D + b["page_padding"]


def key_shape(b):
    _, stride = _page_layout(b)
    return (
        f"N{b['N']}_I{b['I']}_H{b['H']}_S{b['S']}_norm{int(b['apply_q_norm'])}"
        f"_neg-{b['negative_slots']}_page-stride{stride}"
    )


def unsupported_reason(b, dtype):
    if not b["apply_q_norm"]:
        op = getattr(torch.ops._C, OP_NAME, None)
        schema = getattr(getattr(op, "default", None), "_schema", None)
        if schema is not None and [a.name for a in schema.arguments] == [
            "q",
            "kv",
            "k_cache",
            "slot_mapping",
            "position_ids",
            "cos_sin_cache",
            "eps",
            "cache_block_size",
        ]:
            return f"installed eight-argument native always normalizes Q: {schema}"
    # The native full-cache kernel uses uint4 stores (16-byte alignment).
    pages, page_stride = _page_layout(b)
    step = 2 if b["negative_slots"] == "alternate" else 1
    if (
        b["negative_slots"] != "all"
        and page_stride % 8
        and any((t % pages) * page_stride % 8 for t in range(0, b["I"], step))
    ):
        return "native uint4 stores require 16-byte-aligned BF16 cache page starts"
    return None


def cache_backing(cache):
    """Whole benchmark allocation [pages, page_stride], including page gaps."""
    return cache.as_strided((cache.shape[0], cache.stride(0)), (cache.stride(0), 1))


def build_inputs(b, dtype, device):
    n, i, h, s = b["N"], b["I"], b["H"], b["S"]
    if n <= 0 or h <= 0 or s <= 0 or not 0 <= i <= n or b["page_padding"] < 0:
        raise ValueError(f"invalid workload dimensions: {b!r}")
    distribution = b["negative_slots"]
    if distribution not in {"none", "alternate", "all"}:
        raise ValueError(f"unknown negative slot distribution: {distribution}")
    pages, page_stride = _page_layout(b)
    # Seed on CPU to reproduce identical input bytes across devices.
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    q = torch.randn(n, h, D, dtype=dtype, device="cpu", generator=gen).to(device)
    kv = torch.randn(n, D, dtype=dtype, device="cpu", generator=gen).to(device)
    storage = torch.full((pages, page_stride), -123.0, dtype=dtype, device=device)
    cache = storage.as_strided((pages, s, D), (page_stride, D, 1))
    t = torch.arange(i, dtype=torch.int64, device="cpu")
    # Cycle across pages before advancing to the next offset in each page.
    slots = (t % pages) * s + s - 1 - t // pages
    if distribution == "alternate":
        slots[1::2] = -1
    elif distribution == "all":
        slots.fill_(-1)
    max_pos = n + 32
    t = torch.arange(n, dtype=torch.int64, device="cpu")
    positions = 1 + (17 * t + 23) % (max_pos - 1)
    freq = 10000.0 ** (-torch.arange(0, R, 2, device="cpu").float() / R)
    angles = torch.arange(max_pos, device="cpu").float()[:, None] * freq
    cis = torch.cat((angles.cos(), angles.sin()), dim=1)
    args = (
        q,
        kv,
        cache,
        slots.to(device),
        positions.to(device),
        cis.to(device),
        EPS,
        s,
    )
    # Norm is enabled by default, including the original eight-argument ABI.
    if not b["apply_q_norm"]:
        args += (False,)
    return args, {}


def _tensor(shape, dtype, stride):
    return {"shape": list(shape), "dtype": str(dtype), "stride": list(stride)}


def config(b, dtype):
    n, i, h, s = b["N"], b["I"], b["H"], b["S"]
    pages, page_stride = _page_layout(b)
    q = _tensor((n, h, D), dtype, (h * D, D, 1))
    cache = _tensor((pages, s, D), dtype, (page_stride, D, 1))
    return {
        "inputs": {
            "q": q,
            "kv": _tensor((n, D), dtype, (D, 1)),
            "k_cache": cache,
            "slot_mapping": _tensor((i,), torch.int64, (1,)),
            "position_ids": _tensor((n,), torch.int64, (1,)),
            "cos_sin_cache": _tensor((n + 32, R), torch.float32, (R, 1)),
            "eps": {"scalar": EPS},
            "cache_block_size": {"scalar": s},
            "apply_q_norm": {"scalar": b["apply_q_norm"]},
        },
        "outputs": {
            "q": {**q, "aliases": "q"},
            "k_cache": {**cache, "aliases": "k_cache"},
        },
        "returns": None,
        "mutation": ["q", "k_cache"],
        "negative_slots": b["negative_slots"],
        "page_padding": b["page_padding"],
        "seed": SEED,
    }
