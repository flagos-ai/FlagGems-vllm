# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Workloads for the public vLLM LoRA-A projection, including its host flags."""

import math
import os
import sys
from collections import Counter
from itertools import accumulate

import torch

OP_NAME = "lora_shrink"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (2,)
NATIVE_UNAVAILABLE_REASON = None
SEED = 20261125
DIMS = ("T", "K", "R", "S", "L")

# name, tokens, hidden, rank, slices, adapter slots, output type, scale, padded grid
_CASES = (
    ("no_lora", 4, 4096, 8, 1, 4, "input", 1.0, False),
    ("decode_one", 1, 4096, 8, 1, 4, "input", 1.0, False),
    ("decode_qkv", 7, 4096, 16, 3, 4, "fp32", 0.5, True),
    ("tail_zero_scale", 3, 257, 7, 2, 3, "input", 0.0, True),
    ("decode_grouped", 32, 4096, 16, 1, 8, "fp32", -0.75, False),
    ("prefill_qkv", 128, 4096, 32, 3, 4, "fp32", 1.0, True),
    ("prefill_uneven", 257, 2048, 16, 2, 8, "input", 0.5, False),
    ("long_k", 16, 8192, 64, 1, 4, "fp32", 1.25, True),
)


def native():
    global NATIVE_UNAVAILABLE_REASON
    try:
        from vllm.lora.ops.triton_ops.lora_shrink_op import lora_shrink
    except Exception as exc:
        NATIVE_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
        return None
    if not callable(lora_shrink):
        NATIVE_UNAVAILABLE_REASON = "Installed vLLM lora_shrink entry is not callable"
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return lora_shrink


def _native_bi_enabled():
    # The installed entry captures its BI flag at import. Older vLLM can read
    # the same environment setting while still using the ordinary shrink kernel.
    module = sys.modules.get("vllm.lora.ops.triton_ops.lora_shrink_op")
    return bool(getattr(module, "is_batch_invariant", False)) and callable(
        getattr(module, "_lora_shrink_reduce_kernel", None)
    )


def unsupported_reason(binding, dtype):
    if key_shape(binding)[-1] and not _native_bi_enabled():
        return (
            "BI NOT_RUN: installed vLLM shrink does not expose the active "
            "fixed-partials reduction for this process"
        )
    return None


def grid():
    fields = ("case", *DIMS, "out", "scaling", "padded_grid")
    return [dict(zip(fields, case)) for case in _CASES]


def key_shape(binding):
    # Batch invariance changes the reduction and is selected at process startup.
    return [*(binding[d] for d in DIMS), int(os.getenv("VLLM_BATCH_INVARIANT", "0"))]


def _metadata(binding):
    t, slots = binding["T"], binding["L"]
    case = binding["case"]
    if case == "no_lora":
        mapping = [-1] * t
    elif case == "decode_one":
        mapping = [slots - 1]
    elif case.startswith("prefill"):
        # Four requests of unequal lengths, including one base-model request.
        boundaries = (t // 2, 5 * t // 8, 7 * t // 8, t)
        ids = (slots - 1, -1, 0, slots // 2)
        mapping = [
            adapter
            for lo, hi, adapter in zip((0, *boundaries), boundaries, ids)
            for _ in range(hi - lo)
        ]
    else:
        pattern = (slots - 1, 0, -1, slots // 2, 0, 0, slots - 1, 0)
        mapping = [pattern[i % len(pattern)] for i in range(t)]

    # LoRAKernelMeta stores one extra slot for -1. prepare_tensors() leaves
    # the padding at its reset values, including zeros after the final offset.
    counts = Counter(mapping) if case != "no_lora" else {}
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
        "no_lora_flag_cpu": [case == "no_lora"],
        "num_active_loras": [capacity if binding["padded_grid"] else len(ids)],
    }


def _output_dtype(binding, dtype):
    return torch.float32 if binding["out"] == "fp32" else dtype


def build_inputs(binding, dtype, device):
    t, k, r, s, slots = (binding[d] for d in DIMS)
    gen = torch.Generator(device="cpu").manual_seed(SEED)

    def rand(shape, std=1.0):
        data = torch.randn(shape, generator=gen, device="cpu") * std
        return data.to(device=device, dtype=dtype)

    x = rand((t, k))
    weights = [rand((slots, r, k), k**-0.5) for _ in range(s)]
    out = torch.full((s, t, r), 7.0, dtype=_output_dtype(binding, dtype), device=device)
    meta = _metadata(binding)
    args = [x, weights, out]
    for name, values in meta.items():
        host = name in ("no_lora_flag_cpu", "num_active_loras")
        dt = torch.bool if name == "no_lora_flag_cpu" else torch.int32
        args.append(torch.tensor(values, dtype=dt, device="cpu" if host else device))
    return (*args, binding["scaling"]), {}


def config(binding, dtype):
    def tensor(shape, dt, device="device"):
        return {
            "shape": list(shape),
            "stride": [math.prod(shape[i + 1 :]) for i in range(len(shape))],
            "dtype": str(dt),
            "device": device,
        }

    t, k, r, s, slots = (binding[d] for d in DIMS)
    out = tensor((s, t, r), _output_dtype(binding, dtype))
    inputs = {"inputs": tensor((t, k), dtype)}
    for i in range(s):
        inputs[f"lora_a_weights[{i}]"] = tensor((slots, r, k), dtype)
    inputs["output_tensor"] = out
    values = _metadata(binding)
    for name, data in values.items():
        host = name in ("no_lora_flag_cpu", "num_active_loras")
        dt = torch.bool if name == "no_lora_flag_cpu" else torch.int32
        inputs[name] = tensor((len(data),), dt, "cpu" if host else "device")
    inputs["scaling"] = {"scalar": binding["scaling"]}
    return {
        "case": binding["case"],
        "key": key_shape(binding),
        "inputs": inputs,
        "outputs": {"output_tensor": {**out, "aliases": "args[2]"}},
        "return": None,
        "metadata_values": values,
        "batch_invariant_requested": bool(key_shape(binding)[-1]),
        "native_fixed_bi_enabled": _native_bi_enabled(),
        "output_initial_value": 7.0,
        "input_seed": SEED,
        "inputs_std": 1.0,
        "weights_std": k**-0.5,
        "graph": False,
    }
