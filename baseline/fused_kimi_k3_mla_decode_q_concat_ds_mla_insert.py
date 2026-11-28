# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Installed vLLM native operator and shared fp8_ds_mla decode workloads."""

import importlib
from contextlib import contextmanager
from importlib import metadata

import torch

OP_NAME = "fused_kimi_k3_mla_decode_q_concat_ds_mla_insert"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (4, 5)
NATIVE_REF = "a17e79abde45715e79a00deb99ad34be49162231"
NATIVE_SOURCE = (
    f"https://github.com/vllm-project/vllm/blob/{NATIVE_REF}/csrc/"
    "fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu"
)
NATIVE_PROVENANCE = None
NATIVE_UNAVAILABLE_REASON = None
FIELDS = ("B", "H", "S", "layout", "rope", "slots")
CASES = (
    (1, 1, 16, "dense", False, "valid"),
    (1, 128, 16, "padded", True, "valid"),
    (3, 7, 2, "padded", True, "mixed"),
    (8, 16, 4, "dense", False, "valid"),
    (16, 32, 8, "padded", True, "mixed"),
    (31, 7, 16, "padded", False, "mixed"),
    (32, 64, 16, "dense", True, "valid"),
    (33, 7, 8, "padded", True, "valid"),
    (64, 96, 16, "padded", True, "valid"),
    (64, 128, 32, "padded", True, "mixed"),
)


@contextmanager
def single_cpu_thread():
    previous = torch.get_num_threads()
    try:
        if previous != 1:
            torch.set_num_threads(1)
        yield
    finally:
        if previous != 1:
            torch.set_num_threads(previous)


def native():
    """Resolve the installed C++ overload and verify its mutable schema."""
    global NATIVE_PROVENANCE, NATIVE_UNAVAILABLE_REASON
    errors, loaded = {}, {}
    for module in ("vllm._C", "vllm._C_stable_libtorch"):
        try:
            loaded[module] = importlib.import_module(module).__file__
        except Exception as exc:
            errors[module] = repr(exc)
    NATIVE_PROVENANCE = {
        "source": NATIVE_SOURCE,
        "installed_revision_verified": False,
        "extensions": loaded,
        "import_errors": errors,
    }
    try:
        NATIVE_PROVENANCE["vllm_version"] = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        pass
    try:
        op = getattr(torch.ops._C, OP_NAME).default
        schema = op._schema
        NATIVE_PROVENANCE.update(entry=str(op), schema=str(schema))
        mutations = tuple(
            i
            for i, arg in enumerate(schema.arguments)
            if arg.alias_info is not None and arg.alias_info.is_write
        )
        if len(schema.arguments) != 10 or schema.returns or mutations != MUTATED_INPUTS:
            raise ValueError(f"unexpected native schema: {schema}")
    except Exception as exc:
        NATIVE_UNAVAILABLE_REASON = (
            f"native resolution: {exc!r} (extensions: {errors!r})"
        )
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return op


def grid():
    return [dict(zip(FIELDS, row)) for row in CASES]


def key_shape(binding):
    b, h, s, layout, rope, slots = (binding[k] for k in FIELDS)
    return f"B{b}_H{h}_S{s}_{layout}_rope{int(rope)}_{slots}"


def config(binding, dtype):
    b, h, s, layout, rope, slots = (binding[k] for k in FIELDS)
    if dtype not in DTYPES or min(b, h, s) < 1:
        raise ValueError("fp8_ds_mla decode benchmark requires BF16 and positive B/H/S")
    if layout not in ("dense", "padded") or type(rope) is not bool:
        raise ValueError(f"invalid decode layout or RoPE setting: {binding}")
    if slots not in ("valid", "mixed") or (slots == "mixed" and b < 3):
        raise ValueError(f"invalid decode slot configuration: {binding}")
    padded = layout == "padded"
    pages = (b + s - 1) // s + 2
    h_stride = 584
    q_stride = (h * h_stride + 16, h_stride, 1)
    row_stride = 688 if padded else 656
    page_stride = row_stride * s + (32 if padded else 0)

    def tensor(shape, stride, offset=0, dt=dtype):
        return dict(
            shape=list(shape), stride=list(stride), offset=offset, dtype=str(dt)
        )

    inputs = {
        "ql_nope": tensor((b, h, 512), q_stride if padded else (h * 512, 512, 1)),
        "q_pe": tensor(
            (b, h, 64), q_stride if padded else (h * 64, 64, 1), 512 if padded else 0
        ),
        "kv_c_normed": tensor((b, 512), (512, 1)),
        "k_pe": tensor((b, 64), (576 if padded else 64, 1), 512 if padded else 0),
        "mqa_q": tensor((b, h, 576), (h * 576, 576, 1)),
        "k_cache": tensor(
            (pages, s, 656),
            (page_stride, row_stride, 1),
            16 if padded else 0,
            torch.uint8,
        ),
        "slot_mapping": tensor((b,), (1,), dt=torch.int64),
        "cache_block_size": {"scalar": s},
        "position_ids": (
            tensor((b,), (1,), dt=torch.int64) if rope else {"scalar": None}
        ),
        "cos_sin_cache": (
            tensor((b + 17, 64), (64, 1), dt=torch.float32)
            if rope
            else {"scalar": None}
        ),
    }
    return {
        "case": key_shape(binding),
        "inputs": inputs,
        "outputs": {"return": None, "preallocated": ["mqa_q", "k_cache"]},
        "mutation": list(MUTATED_INPUTS),
        "alias": "ql_nope/q_pe share disjoint slices" if padded else "separate tensors",
        "slot_pattern": slots,
        "cache_format": "4x128 E4M3FN RNE/448, 4 FP32 scales, 64 BF16 RoPE",
        "graph": False,
    }


@single_cpu_thread()
def build_inputs(binding, dtype, device):
    inputs = config(binding, dtype)["inputs"]
    b, h, s = (binding[k] for k in ("B", "H", "S"))
    generator = torch.Generator(device="cpu").manual_seed(20261126)

    def payload(shape):
        # Dyadic BF16 values and a dyadic RoPE table avoid FMA contraction
        # differences at BF16 rounding boundaries in the bytewise checks.
        values = torch.randint(-64, 65, shape, generator=generator, dtype=torch.int16)
        return values.to(dtype).mul_(1 / 32)

    def view(name, base):
        spec = inputs[name]
        return base.as_strided(spec["shape"], spec["stride"], spec["offset"])

    if binding["layout"] == "padded":
        q = payload((b * inputs["ql_nope"]["stride"][0],)).to(device)
        ql, qp = view("ql_nope", q), view("q_pe", q)
        kp = view("k_pe", payload((b, 576)).to(device))
    else:
        ql, qp = payload((b, h, 512)).to(device), payload((b, h, 64)).to(device)
        kp = payload((b, 64)).to(device)

    kv = payload((b, 4, 128))
    kv.mul_(torch.tensor([2**-16, 2**-4, 1, 16], dtype=dtype).view(1, 4, 1))
    out = torch.full((b, h, 576), -73.0, dtype=dtype, device=device)
    spec = inputs["k_cache"]
    backing = torch.full(
        (spec["shape"][0] * spec["stride"][0] + 2 * spec["offset"],),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )
    cache = view("k_cache", backing)

    # The first two active tokens hit different pages in descending order.
    # Slot zero stays active, including in every mixed row.
    order = [0] if b == 1 else [s, 0, s - 1]
    slots = torch.tensor(list(dict.fromkeys(order + list(range(b + s))))[:b])
    if binding["slots"] == "mixed":
        slots[2::4] = -1

    pos = table = None
    if binding["rope"]:
        pos = ((torch.arange(b) * 7 + 3) % (b + 17)).to(device)
        angles = torch.outer(
            torch.arange(b + 17, dtype=torch.float32),
            50000 ** (-torch.arange(32, dtype=torch.float32) / 32),
        )
        table = torch.cat((angles.cos(), angles.sin()), dim=-1)
        table = table.mul_(1024).round_().div_(1024).to(device)
    return (
        ql,
        qp,
        kv.reshape(b, 512).to(device),
        kp,
        out,
        cache,
        slots.to(device),
        s,
        pos,
        table,
    ), {}
