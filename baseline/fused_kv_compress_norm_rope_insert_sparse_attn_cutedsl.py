# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Installed vLLM fused C4 entry and its reproducible paged-cache workloads."""

from contextlib import contextmanager
from importlib import import_module

import torch

OP_NAME = "fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl"
NATIVE_MODULE = "vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl"
DTYPES = [torch.bfloat16, torch.float32]  # RMSNorm weight dtype
IS_INPLACE = True
MUTATED_INPUTS = (9,)
NATIVE_UNAVAILABLE_REASON = None

FIELDS = (
    "tokens",
    "state_slots",
    "destination_slots",
    "active_boundaries",
    "block_size",
    "kv_block_size",
    "page_padding",
    "store",
    "scale",
    "token_stride",
    "scale_dim",
)
_SMALL = (1, 1, 1, 1, 8, 8, 0)
_CROSS = (64, 48, 40, 16, 4, 8, 32)
_CROSS_CONTIGUOUS = (*_CROSS[:-1], 0)
_FORMATS = (
    ("packed", None),
    ("full_bf16", None),
    ("full_fp8", None),
    ("full_fp8", 0.375),
)
CORE = (
    (*_SMALL, "packed", None, 576, 8),
    *((*_CROSS, store, scale, 576, 8) for store, scale in _FORMATS),
)
EXTRA = (
    *((*_CROSS_CONTIGUOUS, store, scale, 576, 8) for store, scale in _FORMATS),
    *((*_SMALL, store, scale, 576, 8) for store, scale in _FORMATS[1:]),
    (128, 128, 128, 128, 16, 16, 0, "packed", None, 576, 8),
    (64, 16, 8, 4, 4, 8, 32, "packed", None, 576, 8),
    (*_CROSS, "packed", None, 608, 16),
)


def native():
    global NATIVE_UNAVAILABLE_REASON
    try:
        op = getattr(import_module(NATIVE_MODULE), OP_NAME)
        if not callable(op) or getattr(op, "__module__", "").startswith(
            "flaggems_vllm"
        ):
            raise TypeError(f"{NATIVE_MODULE}.{OP_NAME} is not an independent callable")
    except Exception as exc:
        NATIVE_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return op


def grid():
    return [dict(zip(FIELDS, row)) for row in (*CORE, *EXTRA)]


def _strides(binding):
    padded = binding["page_padding"]
    packed = binding["store"] == "packed"
    width = binding["token_stride"] + binding["scale_dim"] if packed else 512
    state_row = 2048 + (16 if padded else 0)
    state_page = binding["block_size"] * state_row + padded
    kv_row = width + (16 if padded and not packed else 0)
    kv_page = binding["kv_block_size"] * kv_row + padded
    return width, state_row, state_page, kv_row, kv_page


def key_shape(binding):
    _, _, state_page, _, kv_page = _strides(binding)
    fields = "_".join(str(binding[name]) for name in FIELDS)
    return f"{fields}_statepage{state_page}_kvpage{kv_page}"


def config(binding, dtype):
    t, src, dst, active, block, page, pad, store, scale, token_stride, scale_dim = (
        binding[name] for name in FIELDS
    )
    if (
        dtype not in DTYPES
        or not all(
            type(i) is int
            for i in (t, src, dst, active, block, page, pad, token_stride, scale_dim)
        )
        or not 0 <= active <= min(src, dst) <= max(src, dst) <= t
        or t <= 0
        or block < 4
        or block % 4
        or page <= 0
        or pad < 0
        or pad % 16
        or (store, scale) not in _FORMATS
        or (store != "packed" and (token_stride, scale_dim) != (576, 8))
        or token_stride < 576
        or token_stride % 8
        or scale_dim < 8
        or scale_dim % 8
    ):
        raise ValueError(f"invalid C4 benchmark configuration: {binding} {dtype}")

    def tensor(shape, stride, dt, offset=0):
        return dict(
            shape=list(shape), stride=list(stride), dtype=str(dt), storage_offset=offset
        )

    width, state_row, state_page, kv_row, kv_page = _strides(binding)
    requests = min(t, 16)
    pages = (t + page - 1) // page + 2
    cache_dtype = {
        "packed": torch.uint8,
        "full_bf16": torch.bfloat16,
        "full_fp8": torch.float8_e4m3fn,
    }[store]
    offset = 16 if pad else 0
    inputs = dict(
        state_cache=tensor(
            (3 * requests, block, 2048),
            (state_page, state_row, 1),
            torch.float32,
            offset,
        ),
        token_to_req_indices=tensor((t,), (1,), torch.int32),
        positions=tensor((t,), (1,), torch.int64),
        slot_mapping=tensor((t,), (1,), torch.int64),
        block_table=tensor((requests, 3), (3, 1), torch.int32),
        block_size={"scalar": block},
        rms_norm_weight=tensor((512,), (1,), dtype),
        rms_norm_eps={"scalar": 1e-6},
        cos_sin_cache=tensor((3 * block, 64), (72 if pad else 64, 1), torch.float32),
        k_cache=tensor((pages, page, width), (kv_page, kv_row, 1), cache_dtype, offset),
        kv_slot_mapping=tensor((t,), (1,), torch.int64),
        kv_cache_block_size={"scalar": page},
        kv_block_stride={"scalar": kv_page},
        head_size={"scalar": 512},
        state_width={"scalar": 1024},
        rope_head_dim={"scalar": 64},
        fp8_max={"scalar": 448.0},
        quant_block={"scalar": 64},
        token_stride={"scalar": token_stride},
        scale_dim={"scalar": scale_dim},
        compress_ratio={"scalar": 4},
        overlap={"scalar": True},
        store_full_kv={"scalar": store != "packed"},
        store_full_fp8={"scalar": store == "full_fp8"},
        fp8_scale=(
            tensor((1,), (1,), torch.float32) if scale is not None else {"scalar": None}
        ),
    )
    return dict(
        case=key_shape(binding),
        binding=dict(binding),
        inputs=inputs,
        outputs={"return": None, "mutated": ["k_cache"]},
        mutation=list(MUTATED_INPUTS),
        alias="independent argument allocations",
    )


def native_layout_reason(op, cfg, dtype):
    """Some releases derive packed geometry internally instead of using the arguments."""
    b = cfg["binding"]
    if b["store"] != "packed" or (b["token_stride"], b["scale_dim"]) == (576, 8):
        return None
    namespace = getattr(op, "__globals__", {})
    store = namespace.get("_SPARSE_ATTN_COMPRESS_NORM_ROPE_STORE_C4_KERNEL")
    if store is None:
        # The explicit-geometry implementation passes both arguments to compile().
        # The benchmark also verifies the actual write locations against the oracle.
        return None
    key = store.dispatch(
        compress_ratio=4,
        norm_weight_dtype=namespace["torch_to_cute_dtype"](dtype),
        head_size=512,
        rope_head_dim=64,
    )
    for name in ("token_stride", "scale_dim"):
        actual = getattr(key, name)
        if actual != b[name]:
            return f"installed packed C4 store derives {name}={actual}, requested {b[name]}"
    return None


def unsupported_reason(binding, dtype):
    op = native()
    if op is None:
        return NATIVE_UNAVAILABLE_REASON
    cfg = config(binding, dtype)
    try:
        return native_layout_reason(op, cfg, dtype)
    except (AttributeError, KeyError, TypeError) as exc:
        return f"cannot determine installed packed C4 geometry: {exc}"


@contextmanager
def single_cpu_thread():
    previous = torch.get_num_threads()
    if previous != 1:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if previous != 1:
            torch.set_num_threads(previous)


@single_cpu_thread()
def build_inputs(binding, dtype, device):
    spec = config(binding, dtype)["inputs"]
    rng = torch.Generator().manual_seed(4408)
    t, src, dst, active = (binding[name] for name in FIELDS[:4])
    block = binding["block_size"]

    def allocate(name):
        data = spec[name]
        shape, stride, offset = data["shape"], data["stride"], data["storage_offset"]
        dt = getattr(torch, data["dtype"].split(".")[-1])
        size = offset + shape[0] * stride[0] + 16
        if name == "k_cache":
            storage = (
                torch.full((size * dt.itemsize,), 0xA5, dtype=torch.uint8)
                .to(device)
                .view(dt)
            )
        else:
            storage = torch.full((size,), -17.25, dtype=dt)
            if name == "state_cache":
                # Keep all logical values and the downstream RNG sequence identical
                # for contiguous and padded versions of the same workload.
                storage.as_strided(shape, stride, offset).copy_(
                    torch.randn(shape, generator=rng)
                )
            storage = storage.to(device)
        return storage.as_strided(shape, stride, offset)

    state = allocate("state_cache")
    requests, columns = spec["block_table"]["shape"]
    table = torch.randperm(
        requests * columns, generator=rng, dtype=torch.int32
    ).reshape(requests, columns)
    rows = torch.randperm(t, generator=rng)
    req = torch.empty(t, dtype=torch.int32)
    req[rows] = torch.arange(t, dtype=torch.int32) % requests
    # Only the first 'active' shared slots are boundaries. Later valid source
    # and destination indices occupy opposite ends of a shuffled token order.
    src_rows = rows[:src]
    dst_rows = torch.cat((rows[:active], rows[t - (dst - active) :]))
    pos = torch.full((t,), 2, dtype=torch.int64)
    boundary = torch.tensor((3, block + 3, 2 * block + 3, 7), dtype=torch.int64)
    pos[rows[:active]] = boundary[torch.arange(active) % len(boundary)]
    slots = torch.full((t,), -1, dtype=torch.int64)
    slots[src_rows] = 0
    pages, page, _ = spec["k_cache"]["shape"]
    kv_slots = torch.full((t,), -1, dtype=torch.int64)
    kv_slots[dst_rows] = torch.randperm(pages * page, generator=rng)[:dst]
    if t == 1 and dst:
        kv_slots[0] = pages * page - 1
    # Exercise both negative-slot gates on otherwise valid C4 boundaries.
    invalid = (slots < 0) | (kv_slots < 0)
    pos[invalid] = 3
    physical_state = table[req[src_rows].long(), pos[src_rows] // block].long()
    slots[src_rows] = physical_state * block + pos[src_rows] % block
    weight = (torch.rand(512, generator=rng) + 0.5).to(dtype).to(device)
    cos_sin = allocate("cos_sin_cache")
    angles = torch.outer(
        torch.arange(3 * block, dtype=torch.float32), 10000 ** (-torch.arange(32) / 32)
    )
    cos_sin.copy_(torch.cat((angles.cos(), angles.sin()), dim=1).to(device))
    cache = allocate("k_cache")
    args = (
        state,
        req.to(device),
        pos.to(device),
        slots.to(device),
        table.to(device),
        block,
        weight,
        1e-6,
        cos_sin,
        cache,
        kv_slots.to(device),
        page,
        cache.stride(0),
    )
    kwargs = {
        k: v["scalar"] for k, v in list(spec.items())[len(args) :] if "scalar" in v
    }
    if binding["scale"] is not None:
        kwargs["fp8_scale"] = torch.tensor([binding["scale"]], dtype=torch.float32).to(
            device
        )
    return args, kwargs
