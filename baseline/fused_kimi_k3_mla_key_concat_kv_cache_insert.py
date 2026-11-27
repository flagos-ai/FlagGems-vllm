# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Installed vLLM prefill epilogue and its deterministic benchmark inputs."""

import importlib
from contextlib import contextmanager
from importlib import metadata

import torch

OP_NAME = "fused_kimi_k3_mla_key_concat_kv_cache_insert"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = True
MUTATED_INPUTS = (0, 4, 5)
NATIVE_SOURCE = (
    "https://github.com/vllm-project/vllm/blob/a17e79abde45715e79a00deb99ad34be49162231/"
    "csrc/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu"
)
NATIVE_UNAVAILABLE_REASON = None
NATIVE_PROVENANCE = None

FIELDS = ("T", "H", "S", "layout", "rope", "slots")
CASES = (
    (1, 1, 16, "contiguous", False, "all"),
    (7, 3, 8, "padded", True, "mixed"),
    (33, 12, 16, "qkv_a", False, "all"),
    (33, 12, 16, "qkv_a", True, "all"),
    (33, 12, 16, "qkv_gate", True, "all"),
    (33, 12, 16, "kv_a", True, "all"),
    (33, 12, 16, "qkv_a", True, "mixed"),
    (33, 12, 16, "qkv_a", True, "none"),
    (128, 96, 16, "qkv_gate", False, "all"),
    (128, 96, 16, "qkv_gate", True, "mixed"),
    (513, 12, 32, "qkv_gate", False, "all"),
    (64, 128, 16, "qkv_a", True, "mixed"),
)


@contextmanager
def single_cpu_thread():
    previous = torch.get_num_threads()
    if previous == 1:
        yield
        return
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def native():
    """Resolve the installed mutable-output overload; extension names vary by build."""
    global NATIVE_PROVENANCE, NATIVE_UNAVAILABLE_REASON
    NATIVE_UNAVAILABLE_REASON = None
    NATIVE_PROVENANCE = {
        "reference_source": NATIVE_SOURCE,
        "installed_ref_verified": False,
    }
    errors = {}
    for name in ("vllm._C", "vllm._C_stable_libtorch"):
        try:
            NATIVE_PROVENANCE[name] = importlib.import_module(name).__file__
        except Exception as exc:
            errors[name] = repr(exc)
    if errors:
        NATIVE_PROVENANCE["import_errors"] = errors
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
        if (
            not callable(op)
            or len(schema.arguments) != 10
            or schema.returns
            or mutations != MUTATED_INPUTS
        ):
            raise ValueError(f"unexpected prefill native schema: {schema}")
        return op
    except Exception as exc:
        NATIVE_UNAVAILABLE_REASON = (
            f"native resolution: {exc!r}; extension imports: {errors}"
        )
        return None


def grid():
    return [dict(zip(FIELDS, row)) for row in CASES]


def key_shape(binding):
    t, h, s, layout, rope, slots = (binding[k] for k in FIELDS)
    return f"T{t}_H{h}_S{s}_{layout}_rope{int(rope)}_{slots}"


def config(binding, dtype):
    t, h, s, layout, rope, slots = (binding[k] for k in FIELDS)
    if dtype not in DTYPES or any(type(n) is not int or n <= 0 for n in (t, h, s)):
        raise ValueError("expected BF16/FP16 and positive integer T, H and S")
    if slots not in ("all", "mixed", "none") or type(rope) is not bool:
        raise ValueError(f"invalid slot pattern or RoPE selection: {binding}")
    # Kimi-K3 q-LoRA rank 1536, latent 512, RoPE 64. With the output gate
    # the packed row also contains H * 128 gate columns following k_pe.
    pe_layouts = {
        "contiguous": (64, 0),
        "qkv_a": (2112, 2048),
        "qkv_gate": (2112 + 128 * h, 2048),
        "kv_a": (576, 512),
        "padded": (592, 8),
    }
    pe_stride, pe_offset = pe_layouts[layout]
    padded = layout == "padded"
    q_head = 200 if padded else 192
    k_head = 128 if layout == "contiguous" else (264 if padded else 256)
    gap = 16 if padded else 0
    cache_row = 592 if padded else 576
    pages = (t + s - 1) // s + 1

    def tensor(shape, stride, offset=0, dt=dtype):
        return dict(
            shape=list(shape), stride=list(stride), storage_offset=offset, dtype=str(dt)
        )

    inputs = {
        "q": tensor((t, h, 192), (h * q_head + gap, q_head, 1), 8 if padded else 0),
        "k_nope": tensor(
            (t, h, 128), (h * k_head + gap, k_head, 1), 8 if padded else 0
        ),
        "k_pe": tensor((t, 64), (pe_stride, 1), pe_offset),
        "kv_c_normed": tensor((t, 512), (512, 1)),
        "k_out": tensor((t, h, 192), (h * 192, 192, 1)),
        "k_cache": tensor(
            (pages, s, 576), (s * cache_row + gap, cache_row, 1), 8 if padded else 0
        ),
        "slot_mapping": tensor((t,), (1,), dt=torch.int64),
        "cache_block_size": {"scalar": s},
        "position_ids": (
            tensor((t,), (1,), dt=torch.int64) if rope else {"scalar": None}
        ),
        "cos_sin_cache": (
            tensor((t + 17, 64), (64, 1), dt=torch.float32)
            if rope
            else {"scalar": None}
        ),
    }
    return dict(
        case=key_shape(binding),
        inputs=inputs,
        mutation=list(MUTATED_INPUTS),
        outputs={"return": None, "preallocated": ["q", "k_out", "k_cache"]},
        alias="separate argument allocations",
        slot_pattern=slots,
        graph=False,
    )


@single_cpu_thread()
def build_inputs(binding, dtype, device):
    inputs = config(binding, dtype)["inputs"]
    generator = torch.Generator().manual_seed(20261126)
    t, s = binding["T"], binding["S"]

    def allocate(name, fill=None):
        spec = inputs[name]
        shape, stride, offset = spec["shape"], spec["stride"], spec["storage_offset"]
        # Allocate complete packed projection rows (including the unused V/gate
        # payload) and include the last row's padding in numerical qualification.
        elements = shape[0] * stride[0]
        if fill is None:
            values = torch.randint(
                -64, 65, (elements,), generator=generator, dtype=torch.int16
            )
            storage = values.to(dtype).mul_(1 / 32).to(device)
        else:
            storage = torch.full((elements,), fill, dtype=dtype, device=device)
        return storage.as_strided(shape, stride, offset)

    q = allocate("q")
    k_nope = allocate("k_nope")
    k_pe = allocate("k_pe")
    latent = allocate("kv_c_normed")
    output = allocate("k_out", -37)
    cache = allocate("k_cache", 73)
    # Include slot zero, the end of page zero and the start of page one.
    order = list(dict.fromkeys([0, s - 1, s, *range(t)]))[:t]
    slots = torch.tensor(order, dtype=torch.int64)
    if binding["slots"] == "none":
        slots.fill_(-1)
    elif binding["slots"] == "mixed":
        slots[3::4] = -1

    positions = table = None
    if binding["rope"]:
        positions = ((torch.arange(t) * 7 + 3) % (t + 17)).to(device)
        angles = torch.outer(
            torch.arange(t + 17, dtype=torch.float32),
            50000 ** (-torch.arange(32, dtype=torch.float32) / 32),
        )
        table = torch.cat((angles.cos(), angles.sin()), dim=-1)
        # Dyadic values make the FP32 products/sums exact on FMA and non-FMA paths.
        table = (table.mul_(1024).round_().div_(1024)).to(device)

    return (
        q,
        k_nope,
        k_pe,
        latent,
        output,
        cache,
        slots.to(device),
        s,
        positions,
        table,
    ), {}
