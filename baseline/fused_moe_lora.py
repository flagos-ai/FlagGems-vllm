# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Installed MoE LoRA native entry and deterministic routing workloads."""

from math import prod

import torch

OP_NAME = "fused_moe_lora"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0,)
NATIVE_UNAVAILABLE_REASON = None
NATIVE_API = "vllm.lora.ops.triton_ops.fused_moe_lora_op.fused_moe_lora"
SCHEDULE = (
    "block_size_m",
    "block_size_n",
    "block_size_k",
    "group_size_m",
    "num_warps",
    "num_stages",
    "split_k",
)
FIELDS = (
    "case",
    "t",
    "topk",
    "e",
    "l",
    "s",
    "r",
    "k",
    "h",
    "sorted",
    "offset",
    "add",
    "weighted",
    "layout",
    "block",
    "tp",
    "world",
    "route",
)


def native():
    global NATIVE_UNAVAILABLE_REASON
    NATIVE_UNAVAILABLE_REASON = None
    try:
        from vllm.lora.ops.triton_ops.fused_moe_lora_op import fused_moe_lora
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
        NATIVE_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
        return None
    if not callable(fused_moe_lora) or getattr(
        fused_moe_lora, "__module__", ""
    ).startswith("flaggems"):
        NATIVE_UNAVAILABLE_REASON = f"{NATIVE_API} did not resolve to a native callable"
        return None
    return fused_moe_lora


def _case(
    case,
    t,
    topk,
    r,
    k,
    h,
    *,
    e=4,
    l=5,
    s=2,
    sorted=False,
    offset=0,
    add=True,
    weighted=False,
    layout="dense",
    block=16,
    tp="none",
    world=1,
    route=0,
):
    return dict(
        zip(
            FIELDS,
            (
                case,
                t,
                topk,
                e,
                l,
                s,
                r,
                k,
                h,
                sorted,
                offset,
                add,
                weighted,
                layout,
                block,
                tp,
                world,
                route,
            ),
        )
    )


def grid():
    return [
        _case("decode_single", 1, 1, 4, 2048, 1408, e=1, l=1, s=1),
        _case("decode_pairs", 3, 8, 8, 192, 768, weighted=True, add=False),
        _case(
            "decode_strided_offset",
            5,
            2,
            7,
            37,
            19,
            weighted=True,
            layout="strided",
            offset=3,
        ),
        _case("threshold_small", 8, 2, 64, 128, 192),
        _case("threshold_one_shot", 9, 2, 64, 128, 192),
        _case(
            "sorted_skew",
            37,
            2,
            17,
            95,
            73,
            sorted=True,
            offset=5,
            add=False,
            weighted=True,
            block=32,
        ),
        _case(
            "sorted_prefill",
            512,
            8,
            16,
            2048,
            1408,
            e=8,
            l=4,
            s=2,
            sorted=True,
            block=64,
        ),
        _case("rank128", 17, 2, 128, 128, 192, sorted=True),
        _case("rank129", 7, 2, 129, 64, 80, s=1, weighted=True, add=False),
        _case("rank256", 41, 2, 256, 128, 192, sorted=True),
        _case("empty", 0, 2, 16, 48, 64, s=1),
    ]


def tp_grid(world=2):
    if world < 2:
        raise ValueError("tensor-parallel workloads require at least two ranks")
    return [
        _case(
            "tp_reduce",
            7,
            2,
            16,
            64,
            32,
            tp="reduce",
            world=world,
            sorted=True,
            weighted=True,
            s=1,
        ),
        _case("tp_gather", 5, 2, 16, 96, 48, tp="gather", world=world, s=2),
    ]


def key_shape(binding):
    return [binding[name] for name in FIELDS]


def routes(binding):
    """Logical routes before block alignment. A route is indexed by token or pair."""
    t, topk, l, e = (binding[n] for n in ("t", "topk", "l", "e"))
    phase = binding["route"]
    choices = (l - 1, 0, -1, l // 2, 0, l - 1, 0) if l > 1 else (0,)
    adapters = [choices[(i + phase) % len(choices)] for i in range(t)]
    experts = []
    for p in range(t * topk):
        i = p + phase * 3
        experts.append(-1 if i % 13 == 6 else ((i % e) if i % 3 else 0))
    enabled = [int(l == 1 or a != (l // 2 if phase % 2 == 0 else 0)) for a in range(l)]
    return adapters, experts, enabled


def _tables(binding):
    adapters, experts, enabled = routes(binding)
    l, e, topk, block = (binding[k] for k in ("l", "e", "topk", "block"))
    # A sentinel lane and trailing inactive slots exercise the CPU active count.
    active = sorted(set(adapters), reverse=True)
    ids = torch.tensor(active + [-1] * (l + 1 - len(active)), dtype=torch.int32)
    common = (
        torch.tensor(adapters, dtype=torch.int32),
        ids,
        torch.tensor([len(active)], dtype=torch.int32),
        torch.tensor(enabled, dtype=torch.int32),
    )
    if not binding["sorted"]:
        return (None, torch.tensor(experts, dtype=torch.int32), None), common
    pairs = len(experts)
    width = pairs * block if pairs < e else pairs + e * (block - 1)
    tokens = torch.full((l, width), pairs, dtype=torch.int32)
    blocks = torch.full((l, (width + block - 1) // block), -1, dtype=torch.int32)
    counts = torch.zeros(l, dtype=torch.int32)
    for a in active:
        if a < 0 or not enabled[a]:
            continue
        tail = 0
        for expert in range(e):
            members = [
                p
                for p, ep in enumerate(experts)
                if ep == expert and adapters[p // topk] == a
            ]
            length = len(members)
            if not length:
                continue
            # Within-expert order is unconstrained. Reverse it to expose a decoder
            # that accidentally uses lane or sorted-table position as the pair.
            end = tail + (length + block - 1) // block * block
            tokens[a, tail : tail + length] = torch.tensor(
                members[::-1], dtype=torch.int32
            )
            blocks[a, tail // block : end // block] = expert
            tail = end
        counts[a] = tail
    return (tokens, blocks, counts), common


def _shape(binding, rank):
    t, topk, l, e, s, r, k, h = (
        binding[n] for n in ("t", "topk", "l", "e", "s", "r", "k", "h")
    )
    world, tp = binding["world"], binding["tp"]
    offset = rank * h if tp == "reduce" else binding["offset"]
    out_width = h * world if tp == "reduce" else offset + s * h + (3 if offset else 0)
    xrows = t * topk if binding["weighted"] else t
    return (
        dict(
            output=(t, topk, out_width),
            qcurr_hidden_states=(xrows, k),
            lora_a_stacked=(l, e, r, k),
            lora_b_stacked=(l, e, h, r * world if tp == "gather" else r),
            topk_weights=(t, topk),
        ),
        offset,
    )


def _strides(shape, strided):
    stride = [0] * len(shape)
    step = 2 if strided else 1
    for axis in reversed(range(len(shape))):
        stride[axis] = step
        step *= max(1, shape[axis])
        if axis == len(shape) - 1 and strided:
            step += 5
    return tuple(stride)


def _schedule(binding):
    return (binding["block"], 32, 32, 1, 4, 2, 1)


def _make_inputs(binding, dtype, device, rank):
    if dtype not in DTYPES:
        raise TypeError("expected float16 or bfloat16")
    shapes, offset = _shape(binding, rank)
    gen = torch.Generator(device="cpu")
    strided = binding["layout"] == "strided"

    def tensor(name, seed, scale, shift=0):
        shape = shapes[name]
        gen.manual_seed(seed)
        values = torch.randn(shape, generator=gen, dtype=torch.float32) * scale + shift
        if not strided:
            return values.to(dtype=dtype, device=device)
        # Nonzero storage offsets and padding in both last axes.
        strides = _strides(shape, True)
        capacity = 8 + sum(
            max(0, size - 1) * stride for size, stride in zip(shape, strides)
        )
        base = torch.full((capacity,), -91, dtype=dtype, device=device)
        view = base.as_strided(shape, strides, 2)
        view.copy_(values)
        return view

    # Reduce uses different input and K-weight shards. Gather uses replicated
    # activations and different A rank shards. Every rank owns distinct B/output.
    output = tensor("output", 700 + rank, 0.2, 0.42)
    x = tensor(
        "qcurr_hidden_states", 17 + (rank if binding["tp"] == "reduce" else 0), 0.5
    )
    a = [
        tensor("lora_a_stacked", 130 + 41 * i + rank, 1 / binding["k"] ** 0.5)
        for i in range(binding["s"])
    ]
    b = [
        tensor(
            "lora_b_stacked",
            317 + 53 * i + rank,
            1 / shapes["lora_b_stacked"][-1] ** 0.5,
        )
        for i in range(binding["s"])
    ]
    topk = torch.arange(prod(shapes["topk_weights"]), dtype=torch.int64)
    palette = torch.tensor([1.0, 0.0, 0.25, 0.75, 0.125, 0.375, 0.5])
    weights = (
        palette[(topk + binding["route"]) % len(palette)]
        .reshape(shapes["topk_weights"])
        .to(device)
    )
    aligned, common = _tables(binding)
    sorted_ids, experts, counts = (
        v.to(device) if v is not None else None for v in aligned
    )
    mapping, ids, active, enabled = common
    sched = _schedule(binding)
    args = (
        output,
        x,
        a,
        b,
        weights,
        sorted_ids,
        experts,
        counts,
        mapping.to(device),
        binding["r"],
        binding["topk"],
        ids.to(device),
        active,
        enabled.to(device),
        *sched,
        *sched,
        binding["weighted"],
        binding["tp"] != "none",
        offset,
        binding["add"],
    )
    return args, {}


def build_inputs(binding, dtype, device):
    if binding["tp"] != "none":
        raise ValueError("use build_tp_inputs with an explicit group rank")
    return _make_inputs(binding, dtype, device, 0)


def build_tp_inputs(binding, dtype, device, rank):
    if binding["tp"] == "none" or not 0 <= rank < binding["world"]:
        raise ValueError("invalid tensor-parallel workload or rank")
    return _make_inputs(binding, dtype, device, rank)


def unsupported_reason(binding, dtype):
    # Rank assertions and offset errors can safely be diagnosed by running native.
    # The strided workload's complete backing allocation contains every address
    # a contiguous-column native kernel might write.
    return None


def config(binding, dtype, rank=0):
    shapes, offset = _shape(binding, rank)

    def descriptor(shape, dt, strided=False, cpu=False):
        return dict(
            shape=list(shape),
            dtype=str(dt),
            stride=list(_strides(shape, strided)),
            storage_offset=2 if strided else 0,
            device="cpu" if cpu else "accelerator",
        )

    inputs = {}
    for name, shape in shapes.items():
        is_weight = name in ("lora_a_stacked", "lora_b_stacked")
        strided = name != "topk_weights" and binding["layout"] == "strided"
        desc = descriptor(
            shape, torch.float32 if name == "topk_weights" else dtype, strided
        )
        inputs[name] = [dict(desc) for _ in range(binding["s"])] if is_weight else desc
    aligned, common = _tables(binding)
    for name, tensor in zip(
        (
            "sorted_token_ids",
            "expert_ids",
            "num_tokens_post_padded",
            "token_lora_mapping",
            "lora_ids",
            "num_active_loras",
            "adapter_enabled",
        ),
        (*aligned, *common),
    ):
        inputs[name] = (
            descriptor(tensor.shape, tensor.dtype, cpu=name == "num_active_loras")
            if tensor is not None
            else {"scalar": None}
        )
    scalars = dict(
        max_lora_rank=binding["r"],
        top_k_num=binding["topk"],
        mul_routed_weight=binding["weighted"],
        fully_sharded=binding["tp"] != "none",
        offset=offset,
        add_inputs=binding["add"],
    )
    for stage in ("shrink", "expand"):
        scalars.update(
            {
                f"{stage}_{key}": value
                for key, value in zip(SCHEDULE, _schedule(binding))
            }
        )
    inputs.update({name: {"scalar": value} for name, value in scalars.items()})
    return dict(
        dims=dict(binding),
        tp_rank=rank,
        inputs=inputs,
        returns=None,
        outputs={"output": dict(inputs["output"])},
        mutated_inputs=[0],
        adapter_enabled=common[-1].tolist(),
        active_adapter_ids=common[1][: common[2].item()].tolist(),
        native_api=NATIVE_API,
        graph=False,
    )
