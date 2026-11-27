# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""vLLM LoRA-B projection workloads. DTYPES parameterizes the adapter weights."""

import math
import sys
from collections import Counter
from itertools import accumulate

import torch

OP_NAME = "lora_expand"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (2,)
NATIVE_UNAVAILABLE_REASON = None
SOURCE = "vllm/lora/ops/triton_ops/lora_expand_op.py"
SOURCE_PIN = "a17e79abde45715e79a00deb99ad34be49162231"
SEED = 20261125
CANARY = 0.75


def native():
    global NATIVE_UNAVAILABLE_REASON
    try:
        from vllm.lora.ops.triton_ops.lora_expand_op import lora_expand
    except Exception as exc:
        NATIVE_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
        return None
    if not callable(lora_expand):
        NATIVE_UNAVAILABLE_REASON = "Installed vLLM lora_expand entry is not callable"
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return lora_expand


def case(
    name,
    tokens,
    rank,
    widths,
    slots,
    *,
    offset=0,
    suffix=0,
    add=False,
    x_fp32=False,
    out_fp32=False,
    cross_low=False,
    strided=False,
    padded=False,
    distribution="decode",
):
    return dict(
        case=name,
        tokens=tokens,
        rank=rank,
        widths=tuple(widths),
        slots=slots,
        offset=offset,
        suffix=suffix,
        add=add,
        x_fp32=x_fp32,
        out_fp32=out_fp32,
        cross_low=cross_low,
        strided=strided,
        padded=padded,
        distribution=distribution,
    )


def grid():
    return [
        case("decode_one", 1, 8, [4096], 4),
        case("decode_one_add", 1, 8, [4096], 4, add=True),
        case(
            "decode_gqa",
            7,
            16,
            [4096, 512, 512],
            8,
            offset=11,
            suffix=7,
            add=True,
            x_fp32=True,
            padded=True,
        ),
        case("decode_qkv", 16, 32, [1024] * 3, 4),
        case(
            "strided_tail",
            17,
            4,
            [127, 31],
            5,
            offset=5,
            suffix=3,
            add=True,
            x_fp32=True,
            strided=True,
            padded=True,
        ),
        case("rank128", 32, 128, [8192], 4, add=True, x_fp32=True),
        case(
            "prefill_gqa",
            128,
            64,
            [4096, 512, 512],
            8,
            add=True,
            x_fp32=True,
            padded=True,
            distribution="prefill",
        ),
        case(
            "prefill_uneven",
            257,
            16,
            [2048],
            8,
            offset=7,
            suffix=13,
            distribution="prefill",
        ),
        case(
            "round_fp32_out",
            5,
            4,
            [33],
            3,
            offset=1,
            suffix=2,
            add=True,
            x_fp32=True,
            out_fp32=True,
        ),
        case("multi_fp32_out", 3, 8, [64, 16], 3, x_fp32=True, out_fp32=True),
        case("mixed_low_out", 11, 8, [33], 4, add=True, cross_low=True),
        case("no_lora", 4, 8, [4096], 4, distribution="disabled", padded=True),
    ]


def key_shape(binding):
    b = binding
    return [
        b["tokens"],
        b["rank"],
        b["slots"],
        len(b["widths"]),
        *b["widths"],
        b["offset"],
        b["suffix"],
        int(b["add"]),
        int(b["x_fp32"]),
        int(b["out_fp32"]),
        int(b["cross_low"]),
        int(b["strided"]),
        int(b["padded"]),
        ("decode", "prefill", "disabled").index(b["distribution"]),
    ]


def unsupported_reason(binding, dtype):
    if len(binding["widths"]) > 1 and binding["out_fp32"]:
        return (
            "Multi-slice vLLM do_expand_kernel casts each B pointer to the "
            "output pointer type. An FP32 output would read 16-bit weights "
            "with 32-bit addressing, beyond the allocation."
        )
    return None


def _metadata(binding):
    t, slots = binding["tokens"], binding["slots"]
    no_lora = binding["distribution"] == "disabled"
    if no_lora:
        mapping = [-1] * t
    elif binding["distribution"] == "prefill":
        # A long request followed by short requests, including a base-model one.
        mapping = [slots - 1] * (t - 7) + [0, 0, -1, -1, slots // 2, 0, -1]
    else:
        pattern = (slots - 1, -1, 0, slots // 2, slots - 1, 0, 0, 0)
        mapping = [pattern[i % len(pattern)] for i in range(t)]

    counts = Counter(mapping) if not no_lora else {}
    ids = sorted(counts)
    sizes = [counts[i] for i in ids]
    capacity = slots + 1
    pad = capacity - len(ids)
    return {
        "token_lora_mapping": mapping,
        "token_indices_sorted_by_lora_ids": sorted(range(t), key=mapping.__getitem__),
        "num_tokens_per_lora": sizes + [0] * pad,
        "lora_token_start_loc": [0, *accumulate(sizes)] + [0] * pad,
        "lora_ids": ids + [-1] * pad,
        "no_lora_flag_cpu": [no_lora],
        "num_active_loras": [capacity if binding["padded"] else len(ids)],
    }


def _initial_value(binding, dtype):
    if binding["case"] == "mixed_low_out":
        return -64 if dtype == torch.bfloat16 else 65536
    return CANARY


def _output_dtype(binding, dtype):
    if binding["out_fp32"]:
        return torch.float32
    if binding["cross_low"]:
        return torch.bfloat16 if dtype == torch.float16 else torch.float16
    return dtype


def _tensors(binding, dtype):
    b = binding
    t, r, slots = b["tokens"], b["rank"], b["slots"]
    widths = b["widths"]
    out_width = b["offset"] + sum(widths) + b["suffix"]

    def tensor(shape, dt, *, stride=None, storage_offset=0, device="device"):
        return dict(
            shape=list(shape),
            dtype=str(dt),
            device=device,
            stride=(
                list(stride)
                if stride
                else [
                    math.prod(max(1, x) for x in shape[i + 1 :])
                    for i in range(len(shape))
                ]
            ),
            storage_offset=storage_offset,
        )

    stride = (max(t, 1) * (2 * r + 1), 2 * r + 1, 2) if b["strided"] else None
    inputs = {
        "inputs": tensor(
            (len(widths), t, r),
            torch.float32 if b["x_fp32"] else dtype,
            stride=stride,
            storage_offset=int(b["strided"]),
        )
    }
    for s, width in enumerate(widths):
        inputs[f"lora_b_weights[{s}]"] = tensor((slots, width, r), dtype)
    inputs["output_tensor"] = tensor((t, out_width), _output_dtype(b, dtype))
    meta = _metadata(b)
    for name, values in meta.items():
        dt = torch.bool if name == "no_lora_flag_cpu" else torch.int32
        device = "cpu" if name in ("no_lora_flag_cpu", "num_active_loras") else "device"
        inputs[name] = tensor((len(values),), dt, device=device)
    return inputs, meta


def config(binding, dtype):
    inputs, values = _tensors(binding, dtype)
    inputs["offset_start"] = {"scalar": binding["offset"]}
    inputs["add_inputs"] = {"scalar": binding["add"]}
    return {
        "case": binding["case"],
        "key": key_shape(binding),
        "inputs": inputs,
        "outputs": {"output_tensor": {**inputs["output_tensor"], "aliases": "args[2]"}},
        "return": None,
        "mutation": ["output_tensor"],
        "metadata_values": values,
        "output_initial_value": _initial_value(binding, dtype),
        "seed": SEED,
        "distribution": binding["distribution"],
        "source": SOURCE,
        "reference_commit": SOURCE_PIN,
    }


def build_inputs(binding, dtype, device):
    b = binding
    widths, r, slots = b["widths"], b["rank"], b["slots"]
    t = b["tokens"]
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.randn((len(widths), t, r), generator=gen, device="cpu")
    weights = [
        torch.randn((slots, h, r), generator=gen, device="cpu") * r**-0.5
        for h in widths
    ]
    if b["case"] == "round_fp32_out":
        # Both the input cast and the post-dot cast affect the exact FP32 result.
        delta = torch.finfo(dtype).eps * 0.375
        x.zero_()
        x[..., 0] = 1 + delta
        x[..., 1] = 1
        for w in weights:
            w.zero_()
            w[..., 0] = 1
            w[..., 1] = delta
    if b["case"] == "mixed_low_out":
        # BF16 can represent 65536 exactly. It overflows on promotion to
        # FP16 before adding an FP16 residual, even if the exact sum fits.
        x.zero_()
        x[..., 0] = 1
        for w in weights:
            w.zero_()
            w[..., 0] = 65536 if dtype == torch.bfloat16 else -65504
    x = x.to(device=device, dtype=torch.float32 if b["x_fp32"] else dtype)
    if b["strided"]:
        backing = x.new_full((len(widths), t, 2 * r + 1), -13)
        backing[..., 1::2].copy_(x)
        x = backing[..., 1::2]
    weights = [w.to(device=device, dtype=dtype) for w in weights]
    # vLLM caches strides and the starting offset using only B addresses.
    # A freshly allocated binding may reuse a freed binding's addresses.
    module = sys.modules.get("vllm.lora.ops.triton_ops.utils")
    cache = getattr(module, "_LORA_B_PTR_DICT", None)
    if isinstance(cache, dict):
        cache.pop(tuple(w.data_ptr() for w in weights), None)

    out = torch.full(
        (t, b["offset"] + sum(widths) + b["suffix"]),
        _initial_value(b, dtype),
        dtype=_output_dtype(b, dtype),
        device=device,
    )
    args = [x, weights, out]
    for name, values in _metadata(b).items():
        dt = torch.bool if name == "no_lora_flag_cpu" else torch.int32
        host = name in ("no_lora_flag_cpu", "num_active_loras")
        args.append(torch.tensor(values, dtype=dt, device="cpu" if host else device))
    return (*args, b["offset"], b["add"]), {}
