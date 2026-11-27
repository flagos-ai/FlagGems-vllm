# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Installed vLLM native entry and shared Kimi-K3 decode workloads."""

import importlib
from contextlib import contextmanager
from importlib import metadata

import torch

OP_NAME = "fused_kimi_k3_mla_decode_q_concat_kv_cache_insert"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = True
MUTATED_INPUTS = (4, 5)
NATIVE_REF = "a17e79abde45715e79a00deb99ad34be49162231"
NATIVE_SOURCE = (
    f"https://github.com/vllm-project/vllm/blob/{NATIVE_REF}/csrc/"
    "fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu"
)
NATIVE_UNAVAILABLE_REASON = None
NATIVE_PROVENANCE = None
FIELDS = ("B", "H", "S", "layout", "rope", "slots")
CASES = (
    (1, 1, 16, "contiguous", False, "valid"),
    (1, 128, 16, "packed", True, "valid"),
    (63, 7, 16, "packed", True, "mixed"),
    (64, 7, 16, "packed", True, "mixed"),
    (65, 7, 16, "packed", True, "mixed"),
    (64, 7, 16, "packed", True, "negative"),
    (64, 7, 16, "packed", False, "mixed"),
    (64, 7, 16, "contiguous", True, "mixed"),
    (64, 96, 16, "packed", True, "valid"),
    (64, 128, 16, "packed", True, "valid"),
)


@contextmanager
def single_cpu_thread():
    # Avoid host thread pools dominating small reference and input operations.
    previous = torch.get_num_threads()
    try:
        if previous != 1:
            torch.set_num_threads(1)
        yield
    finally:
        if previous != 1:
            torch.set_num_threads(previous)


def native():
    """Return only the installed, mutable-output C++ operator overload."""
    global NATIVE_PROVENANCE, NATIVE_UNAVAILABLE_REASON
    NATIVE_UNAVAILABLE_REASON = None
    NATIVE_PROVENANCE = {"source": NATIVE_SOURCE, "installed_revision_verified": False}
    extensions, errors = {}, {}
    for name in ("vllm._C", "vllm._C_stable_libtorch"):
        try:
            extensions[name] = importlib.import_module(name).__file__
        except Exception as exc:
            # Vendors may register the operator under either extension.
            errors[name] = repr(exc)
    NATIVE_PROVENANCE.update(extensions=extensions, import_errors=errors)
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
        return op
    except Exception as exc:
        NATIVE_UNAVAILABLE_REASON = f"native resolution: {exc!r}"
        if errors:
            NATIVE_UNAVAILABLE_REASON += f" (extensions: {errors!r})"
        return None


def grid():
    return [dict(zip(FIELDS, case)) for case in CASES]


def key_shape(binding):
    b, h, s, layout, rope, slots = (binding[field] for field in FIELDS)
    return f"B{b}_H{h}_S{s}_{layout}_rope{int(rope)}_{slots}"


def config(binding, dtype):
    b, h, s, layout, rope, slots = (binding[field] for field in FIELDS)
    if dtype not in DTYPES or min(b, h, s) < 1:
        raise ValueError("decode benchmarks require FP16/BF16 and positive B/H/S")
    if (
        layout not in ("packed", "contiguous")
        or slots
        not in (
            "valid",
            "mixed",
            "negative",
        )
        or type(rope) is not bool
    ):
        raise ValueError(f"invalid decode configuration: {binding}")
    packed = layout == "packed"
    pages = (b + s - 1) // s + 2
    head_stride = 576 + 8
    token_stride = h * head_stride + 16
    row_stride = 576 + (16 if packed else 0)
    page_stride = row_stride * s + (32 if packed else 0)

    def tensor(shape, stride, offset=0, dt=dtype):
        return dict(
            shape=list(shape), stride=list(stride), offset=offset, dtype=str(dt)
        )

    inputs = {
        "ql_nope": tensor(
            (b, h, 512), (token_stride, head_stride, 1) if packed else (h * 512, 512, 1)
        ),
        "q_pe": tensor(
            (b, h, 64),
            (token_stride, head_stride, 1) if packed else (h * 64, 64, 1),
            512 if packed else 0,
        ),
        "kv_c_normed": tensor((b, 512), (512, 1)),
        "k_pe": tensor((b, 64), (576 if packed else 64, 1), 512 if packed else 0),
        "mqa_q": tensor((b, h, 576), (h * 576, 576, 1)),
        "k_cache": tensor(
            (pages, s, 576), (page_stride, row_stride, 1), 8 if packed else 0
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
        "mutation": list(MUTATED_INPUTS),
        "outputs": {"return": None, "preallocated": ["mqa_q", "k_cache"]},
        "alias": "ql_nope/q_pe share disjoint slices" if packed else "separate tensors",
        "slot_pattern": slots,
        "graph": False,
    }


@single_cpu_thread()
def build_inputs(binding, dtype, device):
    """Seeded CPU generation preserves input values and aliasing across providers."""
    inputs = config(binding, dtype)["inputs"]
    b, h, s = (binding[field] for field in ("B", "H", "S"))
    generator = torch.Generator(device="cpu").manual_seed(20261126)

    def payload(shape):
        # Dyadic values make the FP32 rotate exact across FMA contraction modes.
        values = torch.randint(-64, 65, shape, generator=generator, dtype=torch.int16)
        return values.to(dtype).mul_(1 / 32).to(device)

    def view(name, base):
        spec = inputs[name]
        return base.as_strided(spec["shape"], spec["stride"], spec["offset"])

    if binding["layout"] == "packed":
        q = payload((b * inputs["ql_nope"]["stride"][0],))
        ql_nope, q_pe = view("ql_nope", q), view("q_pe", q)
        k_pe = view("k_pe", payload((b, 576)))
    else:
        ql_nope, q_pe = payload((b, h, 512)), payload((b, h, 64))
        k_pe = payload((b, 64))
    kv = payload((b, 512))
    out = torch.full((b, h, 576), -73.0, dtype=dtype, device=device)
    spec = inputs["k_cache"]
    storage = torch.full(
        (spec["shape"][0] * spec["stride"][0] + 2 * spec["offset"],),
        97.0,
        dtype=dtype,
        device=device,
    )
    cache = view("k_cache", storage)
    # Valid slots start with 0, S-1, S. Unused pages and rows retain the sentinel.
    order = [0, s - 1, s] if s > 1 else [0, 1]
    slots = torch.tensor(list(dict.fromkeys(order + list(range(b + s))))[:b])
    if binding["slots"] == "negative":
        slots.fill_(-1)
    elif binding["slots"] == "mixed":
        slots[3::4] = -1
    positions = table = None
    if binding["rope"]:
        positions = (torch.arange(b) * 7 + 3) % (b + 17)
        angles = torch.outer(
            torch.arange(b + 17, dtype=torch.float32),
            50000 ** (-torch.arange(32, dtype=torch.float32) / 32),
        )
        table = torch.cat((angles.cos(), angles.sin()), dim=-1)
        table = (table.mul_(1024).round_().div_(1024)).to(device)
        positions = positions.to(device)
    return (
        ql_nope,
        q_pe,
        kv,
        k_pe,
        out,
        cache,
        slots.to(device),
        s,
        positions,
        table,
    ), {}
