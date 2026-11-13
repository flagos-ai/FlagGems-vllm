# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Three-argument vLLM _C baseline and the single frozen workload matrix."""

import importlib

import torch

OP_NAME = "fused_kimi_k3_mla_kv_concat"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (2,)
NATIVE_REF = "a17e79abde45715e79a00deb99ad34be49162231"
MODEL_CONFIG_REF = "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
MODEL_CONFIG_URL = (
    f"https://huggingface.co/moonshotai/Kimi-K3/blob/{MODEL_CONFIG_REF}/config.json"
)
NATIVE_SOURCE = (
    f"https://github.com/vllm-project/vllm/blob/{NATIVE_REF}/csrc/"
    "fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu"
)
NATIVE_UNAVAILABLE_REASON = None
NATIVE_PROVENANCE = None


def native():
    """Resolve the registered _C op, never the allocating Python wrapper."""
    global NATIVE_UNAVAILABLE_REASON, NATIVE_PROVENANCE
    NATIVE_UNAVAILABLE_REASON = None
    NATIVE_PROVENANCE = {"requested_ref": NATIVE_REF, "installed_ref_verified": False}
    import_reason = None
    try:
        extension = importlib.import_module("vllm._C")
        NATIVE_PROVENANCE["extension"] = extension.__file__
    except (ImportError, OSError) as error:
        import_reason = f"vllm._C import: {error!r}"
        NATIVE_PROVENANCE["import_error"] = import_reason
    try:
        schema = torch._C._dispatch_find_schema_or_throw(f"_C::{OP_NAME}", "").schema()
        NATIVE_PROVENANCE["schema"] = str(schema)
    except RuntimeError as error:
        NATIVE_UNAVAILABLE_REASON = f"_C schema missing: {error!r}"
        if import_reason:
            NATIVE_UNAVAILABLE_REASON = import_reason + "; " + NATIVE_UNAVAILABLE_REASON
        NATIVE_PROVENANCE["schema_error"] = repr(error)
        return None
    if (
        len(schema.arguments) != 3
        or len(schema.returns) != 0
        or schema.arguments[2].alias_info is None
        or not schema.arguments[2].alias_info.is_write
    ):
        raise RuntimeError(f"unexpected native schema: {schema}")
    if not torch._C._dispatch_has_kernel_for_dispatch_key(f"_C::{OP_NAME}", "CUDA"):
        NATIVE_UNAVAILABLE_REASON = "registered schema has no CUDA device kernel"
        return None
    op = getattr(torch.ops._C, OP_NAME).default
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = "registered native overload is not callable"
        return None
    NATIVE_PROVENANCE.update(
        {
            "schema": str(schema),
            "dispatch": torch._C._dispatch_dump_table(f"_C::{OP_NAME}"),
            "requested_ref": NATIVE_REF,
            "installed_ref_verified": False,
            "entry": f"torch.ops._C.{OP_NAME}",
        }
    )
    return op


def grid():
    """Synthetic decode/chunk/prefill sweeps; H=96 from the Kimi-K3 checkpoint."""
    cases = [
        (0, 128, "packed"),
        (1, 1, "contiguous"),
        (1, 128, "packed"),
        (7, 3, "gather"),
        (33, 128, "packed"),
        (128, 128, "packed"),
        (513, 128, "gather"),
        (2048, 128, "packed"),
        (8192, 128, "contiguous"),
        (128, 3, "contiguous"),
        (128, 3, "packed"),
        (128, 3, "gather"),
        (1, 96, "packed"),
        (33, 96, "gather"),
        (128, 96, "packed"),
        (513, 96, "padded"),
        (2048, 96, "gather"),
        (8192, 96, "packed"),
        (7, 12, "gather"),
    ]
    return [dict(zip(("T", "H", "layout"), case)) for case in cases]


def strides(binding):
    heads, layout = binding["H"], binding["layout"]
    if layout == "contiguous":
        return (heads * 128, 128, 1), (64, 1)
    if layout == "packed":
        # kv_b_proj holds a 128-d key and a 128-d value per head.
        return (heads * 256, 256, 1), (576, 1)
    if layout == "gather":
        return (heads * 256, 256, 1), (576, 1)
    if layout == "padded":
        # Padding between tokens/heads; deliberately not vector aligned.
        return (heads * 273 + 19, 273, 1), (593, 1)
    raise ValueError(f"unknown layout: {layout}")


def build_inputs(binding, dtype, device):
    """Copy CPU integer bit patterns, retaining full packed backing storage."""
    if dtype not in DTYPES:
        raise TypeError("expected FP16/BF16")
    tokens, heads = binding["T"], binding["H"]
    ns, ps = strides(binding)
    generator = torch.Generator().manual_seed(20261112)
    # Head/token gaps and offsets are initialized and checked by accuracy tests.
    nbits = torch.randint(
        -32768,
        32768,
        (max(tokens * ns[0] + 8, 8),),
        dtype=torch.int16,
        generator=generator,
    )
    pbits = torch.randint(
        -32768,
        32768,
        (max(tokens * ps[0] + 576, 576),),
        dtype=torch.int16,
        generator=generator,
    )
    # Include both signs of zero/Inf, signaling/quiet NaNs and subnormals in
    # both encodings, in addition to the uniformly sampled 16-bit patterns.
    special = torch.tensor(
        [0, -32768, 31744, -1024, 32257, 31745, 32640, -128, 32705, 32641, 1, -32767],
        dtype=torch.int16,
    )
    n_offset = 1 if binding["layout"] == "padded" else 0
    p_offset = (
        1
        if binding["layout"] == "padded"
        else (512 if binding["layout"] in ("packed", "gather") else 0)
    )
    for bits, offset in ((nbits, n_offset), (pbits, p_offset)):
        count = min(bits.numel() - offset, special.numel())
        bits[offset : offset + count] = special[:count]
    nbase = nbits.to(device).view(dtype)
    pbase = pbits.to(device).view(dtype)
    if binding["layout"] in ("packed", "gather"):
        # Exactly the projection split and the gathered latent workspace slice
        # used by the pinned Kimi-K3 chunked-context call site.
        nope = nbase[: tokens * heads * 256].view(tokens, heads, 256)[..., :128]
        pe = pbase[: tokens * 576].view(tokens, 576)[:, 512:]
    else:
        nope = torch.as_strided(
            nbase, (tokens, heads, 128), ns, storage_offset=n_offset
        )
        pe = torch.as_strided(pbase, (tokens, 64), ps, storage_offset=p_offset)
    out = torch.empty((tokens, heads, 192), dtype=dtype, device=device)
    out.view(torch.int16).fill_(0x5A5A)
    return (nope, pe, out), {}


def key_shape(binding):
    return f"T{binding['T']}_H{binding['H']}_{binding['layout']}"


def config(binding, dtype):
    tokens, heads = binding["T"], binding["H"]
    ns, ps = strides(binding)
    source_bytes = 2 * tokens * (heads * 128 + 64)
    output_bytes = 2 * tokens * heads * 192
    return {
        "inputs": {
            "k_nope": {
                "shape": [tokens, heads, 128],
                "stride": list(ns),
                "storage_offset": 1 if binding["layout"] == "padded" else 0,
                "dtype": str(dtype),
            },
            "k_pe": {
                "shape": [tokens, 64],
                "stride": list(ps),
                "storage_offset": (
                    1
                    if binding["layout"] == "padded"
                    else (512 if binding["layout"] in ("packed", "gather") else 0)
                ),
                "dtype": str(dtype),
            },
            "k_out": {
                "shape": [tokens, heads, 192],
                "stride": [heads * 192, 192, 1],
                "dtype": str(dtype),
            },
        },
        "outputs": {"return": None, "mutated_arg": 2},
        "mutation": [2],
        "alias": "output/input byte spans disjoint",
        "layout": binding["layout"],
        "graph": False,
        "source_unique_bytes": source_bytes,
        "output_bytes": output_bytes,
        "logical_minimum_bytes": source_bytes + output_bytes,
        "streaming_pe_load_bytes": 2 * tokens * heads * 64,
        "streaming_logical_bytes": output_bytes * 2,
        "physical_hbm_bytes": "unknown; PE cache reuse requires profiling",
        "workload_origin": (
            "synthetic T sweep; packed kv_b_proj and [T,576] gathered-cache PE view; "
            "padded is adversarial, not a model trace"
        ),
        "model_config": {
            "ref": MODEL_CONFIG_REF,
            "url": MODEL_CONFIG_URL,
            "num_attention_heads": 96,
            "H12": "TP=8 local heads",
            "H128": "synthetic stress",
        },
    }


def unsupported_reason(binding, dtype):
    if binding["T"] and binding["layout"] == "padded":
        return (
            "NOT_RUN: pinned native writeFullKeyPack dereferences uint4; "
            "this padded case has unaligned source addresses/strides"
        )
    return None
