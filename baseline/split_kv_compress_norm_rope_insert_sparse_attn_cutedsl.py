# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Public vLLM C128 split entry and reproducible paged workloads."""

from contextlib import contextmanager
from importlib import import_module

import torch

OP_NAME = "split_kv_compress_norm_rope_insert_sparse_attn_cutedsl"
NATIVE_MODULE = "vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl"
DTYPES = [torch.bfloat16, torch.float32]  # RMSNorm weight precision
IS_INPLACE = True
MUTATED_INPUTS = (6, 10)
NATIVE_UNAVAILABLE_REASON = None

FIELDS = ("tokens", "active", "page_size", "format", "layout", "scale")
CORE = (
    (1, 1, 8, "packed", "dense", None),
    (32, 8, 16, "packed", "padded", None),
    (8, 4, 4, "bf16", "padded", None),
    (16, 8, 8, "e4m3fn", "dense", 0.375),
)
EXTRA = (
    (32, 16, 32, "packed", "dense", None),
    (8, 4, 4, "packed", "explicit", None),
    (16, 8, 8, "e4m3fn", "padded", None),
    (16, 4, 8, "bf16", "workspace_pad", None),
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


def key_shape(binding):
    return "_".join(str(binding[field]) for field in FIELDS)


def config(binding, dtype):
    t, active, page, cache_format, layout, scale = (binding[f] for f in FIELDS)
    if (
        dtype not in DTYPES
        or type(t) is not int
        or type(active) is not int
        or t < 1
        or not 0 <= active <= t
        or type(page) is not int
        or page < 1
        or cache_format not in ("packed", "bf16", "e4m3fn")
        or layout not in ("dense", "padded", "explicit", "workspace_pad")
        or (layout == "explicit" and cache_format != "packed")
        or (scale is not None and (cache_format != "e4m3fn" or scale <= 0))
    ):
        raise ValueError(f"invalid C128 configuration: {binding}, {dtype}")

    def tensor(shape, stride, dt, offset=0):
        return dict(
            shape=list(shape), stride=list(stride), storage_offset=offset, dtype=str(dt)
        )

    padded = layout != "dense"
    requests = max(1, min(active, 16))
    # Two independent 128-token windows per request, each spanning 16 state pages.
    state_row = 1040 if padded else 1024
    state_page = 8 * state_row + (16 if padded else 0)
    workspace_row = 528 if layout == "workspace_pad" else 512
    token_stride, scale_dim = (608, 16) if layout == "explicit" else (576, 8)
    cache_width = token_stride + scale_dim if cache_format == "packed" else 512
    cache_row = cache_width + (16 if padded and cache_format != "packed" else 0)
    cache_page = page * cache_row + (32 if padded else 0)
    cache_dtype = {
        "packed": torch.uint8,
        "bf16": torch.bfloat16,
        "e4m3fn": torch.float8_e4m3fn,
    }[cache_format]
    # All rows permit destination slots on different pages, even for one token.
    pages = (t + page - 1) // page + 2
    inputs = dict(
        state_cache=tensor(
            (32 * requests, 8, 1024),
            (state_page, state_row, 1),
            torch.float32,
            16 if padded else 0,
        ),
        token_to_req_indices=tensor((t,), (1,), torch.int32),
        positions=tensor((t,), (1,), torch.int64),
        slot_mapping=tensor((t,), (1,), torch.int64),
        block_table=tensor((requests, 32), (32, 1), torch.int32),
        block_size={"scalar": 8},
        compressed_kv=tensor(
            (t, 512), (workspace_row, 1), torch.float32, 4 if padded else 0
        ),
        rms_norm_weight=tensor((512,), (1,), dtype),
        rms_norm_eps={"scalar": 1e-6},
        cos_sin_cache=tensor(
            (256, 64),
            (72 if padded and cache_format != "packed" else 64, 1),
            torch.float32,
        ),
        k_cache=tensor(
            (pages, page, cache_width),
            (cache_page, cache_row, 1),
            cache_dtype,
            16 if padded else 0,
        ),
        kv_slot_mapping=tensor((t,), (1,), torch.int64),
        kv_cache_block_size={"scalar": page},
        kv_block_stride={"scalar": cache_page},
        head_size={"scalar": 512},
        state_width={"scalar": 512},
        rope_head_dim={"scalar": 64},
        fp8_max={"scalar": 448.0},
        quant_block={"scalar": 64},
        token_stride={"scalar": token_stride},
        scale_dim={"scalar": scale_dim},
        compress_ratio={"scalar": 128},
        overlap={"scalar": False},
        store_full_kv={"scalar": cache_format != "packed"},
        store_full_fp8={"scalar": cache_format == "e4m3fn"},
        fp8_scale=(
            tensor((1,), (1,), torch.float32) if scale is not None else {"scalar": None}
        ),
    )
    return dict(
        case=key_shape(binding),
        binding=dict(binding),
        inputs=inputs,
        state_pages="shuffled, 8 slots/page",
        active_compress=active,
        active_store=active,
        outputs={"return": None, "mutated": ["compressed_kv", "k_cache"]},
        mutation=list(MUTATED_INPUTS),
        alias="independent argument allocations",
    )


def native_layout_reason(op, cfg, dtype):
    """When the installed wrapper derives its packed geometry, compare its actual compile key."""
    spec = cfg["inputs"]
    if spec["store_full_kv"]["scalar"] or (
        spec["token_stride"]["scalar"] == 576 and spec["scale_dim"]["scalar"] == 8
    ):
        # All generated rows pass the view's actual page size and stride.
        return None
    namespace = getattr(op, "__globals__", {})
    store = namespace.get("_SPARSE_ATTN_NORM_ROPE_STORE_KERNEL")
    if store is None:
        # The explicit-geometry release passes the scalars to its compiler.
        # All generated buffers can hold both the default and the requested
        # encoding; qualification checks every output and padding byte.
        return None
    try:
        key = store.dispatch(
            compress_ratio=128,
            cache_block_size=spec["k_cache"]["shape"][1] * 128,
            cache_alignment=576,
            norm_weight_dtype=namespace["torch_to_cute_dtype"](dtype),
            head_size=512,
            rope_head_dim=64,
            runtime_kv_block_stride=spec["k_cache"]["stride"][0],
        )
        for name in (
            "token_stride",
            "scale_dim",
            "kv_block_stride",
            "kv_cache_block_size",
        ):
            value = spec[name]["scalar"]
            actual = getattr(key, name)
            if actual != value:
                return (
                    f"installed packed store derives {name}={actual}, requested {value}"
                )
    except (AttributeError, KeyError, TypeError):
        # A different private dispatch ABI can still support this public call.
        # The independent output check on guarded storage decides eligibility.
        return None
    return None


def unsupported_reason(binding, dtype):
    # Used by the standalone collector before launching a selected row.
    op = native()
    return (
        native_layout_reason(op, config(binding, dtype), dtype)
        if op
        else NATIVE_UNAVAILABLE_REASON
    )


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
    rng = torch.Generator().manual_seed(12808)
    t, active = binding["tokens"], binding["active"]
    requests = spec["block_table"]["shape"][0]

    def allocate(name, fill=None):
        data = spec[name]
        shape, stride, offset = data["shape"], data["stride"], data["storage_offset"]
        dt = getattr(torch, data["dtype"].split(".")[-1])
        n = offset + shape[0] * stride[0] + 16
        if fill is None:
            storage = torch.randn(n, generator=rng).to(dt)
        elif name == "k_cache":
            # Byte initialization also works on backends without FP8 arithmetic.
            storage = torch.full((n * dt.itemsize,), fill, dtype=torch.uint8).to(device)
            return storage.view(dt).as_strided(shape, stride, offset)
        else:
            storage = torch.full((n,), fill, dtype=dt)
        return storage.to(device).as_strided(shape, stride, offset)

    state = allocate("state_cache")
    table = torch.randperm(32 * requests, generator=rng, dtype=torch.int32).reshape(
        requests, 32
    )
    rows = torch.randperm(t, generator=rng)
    req = torch.arange(t, dtype=torch.int32) % requests
    req[rows[:active]] = torch.arange(active, dtype=torch.int32) % requests
    # Inactive rows alternate non-boundaries and negative source slot IDs.
    pos = torch.full((t,), 126, dtype=torch.int64)
    pos[rows[:active]] = 127 + 128 * (torch.arange(active) % 2)
    pos[rows[active::2]] = 127
    slots = torch.arange(t, dtype=torch.int64)
    slots[rows[active::2]] = -1
    pages, page, _ = spec["k_cache"]["shape"]
    dst = torch.randperm(pages * page, generator=rng)[:t]
    if t == 1:
        dst[0] = pages * page - 1
    weight = (torch.rand(512, generator=rng) + 0.5).to(dtype)
    cos_sin = allocate("cos_sin_cache", 0)
    angles = torch.outer(
        torch.arange(256, dtype=torch.float32), 10000 ** (-torch.arange(32) / 32)
    )
    cos_sin.copy_(torch.cat((angles.cos(), angles.sin()), dim=1).to(device))
    cache = allocate("k_cache", 0xA5)
    workspace = allocate("compressed_kv", -19.75)
    args = (
        state,
        req.to(device),
        pos.to(device),
        slots.to(device),
        table.to(device),
        8,
        workspace,
        weight.to(device),
        1e-6,
        cos_sin,
        cache,
        dst.to(device),
        page,
        cache.stride(0),
    )
    kwargs = {
        name: data["scalar"]
        for name, data in list(spec.items())[14:]
        if "scalar" in data
    }
    if binding["scale"] is not None:
        kwargs["fp8_scale"] = torch.tensor([binding["scale"]], dtype=torch.float32).to(
            device
        )
    return args, kwargs
