# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Inputs and independent reference for the Ascend MiniMax benchmark."""

from importlib import import_module

OP_NAME = "minimax_qknorm_rope_cache"
TOKENS = (1, 4, 8, 16, 32, 64, 128, 192, 256, 512)
Q_HEADS = (8, 16)
CACHE_NAMES = ("key_cache", "value_cache", "index_cache")
REPLAY_ORIGINAL_SHA256 = (
    "6942c357a5fd3d908a8bd3312b1a81beafa347247ca15bdc5c5f767107b6faaf"
)


def load_baseline(kind):
    module = (
        "vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache"
        if kind == "native"
        else "baseline.minimax_qknorm_rope_cache_replay"
    )
    return import_module(module).minimax_qknorm_rope_cache


def build_case(tokens, heads, device, *, layout="public", edge=False):
    import torch

    generator = torch.Generator().manual_seed(1701 + tokens + heads)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(shape, dtype=dtype, generator=generator)

    def copy_to_device(source, row_padding=0):
        if row_padding:
            target = torch.empty(
                (source.shape[0], source.shape[1] + row_padding),
                dtype=source.dtype,
                device=device,
            )[:, : source.shape[1]]
            target.copy_(source)
            return target
        return source.to(device)

    width = (heads + 2) * 128
    main, index = randn(tokens, width), randn(tokens, 256)
    positions = torch.arange(tokens, dtype=torch.int64)
    table = randn(1024, 64)
    weights = [
        (
            randn(128, dtype=torch.float32 if layout == "padded" else torch.bfloat16)
            * 0.1
            if edge or layout == "padded"
            else torch.zeros(128, dtype=torch.bfloat16)
        )
        for _ in range(4)
    ]
    blocks, block_size = (5392, 128) if layout == "public" else (16, 128)
    capacity = blocks * block_size
    main_slots = torch.arange(capacity - tokens, capacity, dtype=torch.int64)
    index_slots = main_slots.clone()
    actual = max(0, tokens - 3) if edge else tokens
    if edge:
        main_slots[::5] = -1
        index_slots[::7] = -1

    if layout == "padded":
        packed = copy_to_device(torch.cat((main, index), dim=1), row_padding=32)
        index_packed = None
        position_storage = torch.empty((tokens, 2), dtype=torch.int64, device=device)
        position_input = position_storage[:, 0]
        position_input.copy_(positions)
        cos_sin = copy_to_device(table, row_padding=16)
    else:
        packed, index_packed = copy_to_device(main), copy_to_device(index)
        position_input, cos_sin = copy_to_device(positions), copy_to_device(table)

    main_cache_shape = (blocks, block_size, 1, 128)
    cache_shapes = (main_cache_shape, main_cache_shape, (blocks, block_size, 128))
    kwargs = dict(
        packed=packed,
        index_packed=index_packed,
        cos_sin_cache=cos_sin,
        positions=position_input,
        q_weight=weights[0].to(device),
        k_weight=weights[1].to(device),
        index_q_weight=weights[2].to(device),
        index_k_weight=weights[3].to(device),
        slot_mapping=main_slots.to(device),
        index_slot_mapping=index_slots.to(device),
        num_actual_tokens=actual,
        num_q_heads=heads,
        num_kv_heads=1,
        num_index_heads=1,
        eps=1e-6,
    )
    for name, shape in zip(CACHE_NAMES, cache_shapes):
        kwargs[name] = torch.empty(shape, dtype=torch.bfloat16, device=device)

    phase = table[positions].float().unsqueeze(1)
    c, s = phase.split(32, dim=-1)

    def norm_rope(x, weight):
        x = x.reshape(tokens, x.shape[1] // 128, 128).float()
        norm = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + kwargs["eps"])
        norm = (norm * (1 + weight.float())).bfloat16().float()
        a, b, tail = norm.split((32, 32, 64), dim=-1)
        return torch.cat((a * c - b * s, b * c + a * s, tail), dim=-1).bfloat16()

    q, k, v = main.split((heads * 128, 128, 128), dim=1)
    iq, ik = index.split(128, dim=1)
    query = norm_rope(q, weights[0]).reshape(tokens, heads * 128)
    index_query = norm_rope(iq, weights[2]).reshape(tokens, 128)
    expected_writes = []
    for slots, values in zip(
        (main_slots, main_slots, index_slots),
        (norm_rope(k, weights[1]), v, norm_rope(ik, weights[3])),
    ):
        valid = slots[:actual] >= 0
        expected_writes.append(
            (slots[:actual][valid], values.reshape(tokens, 128)[:actual][valid])
        )

    return kwargs, (query, index_query), expected_writes


def check_call(op, case, *, check_untouched=True):
    import torch

    kwargs, expected_queries, expected_writes = case
    for name in CACHE_NAMES:
        kwargs[name].fill_(-7)
    queries = op(**kwargs)
    for output, expected in zip(queries, expected_queries, strict=True):
        actual = output.cpu()
        assert torch.isfinite(actual).all().item()
        torch.testing.assert_close(actual, expected, atol=0.032, rtol=0.01)
    for name, (slots, expected) in zip(CACHE_NAMES, expected_writes, strict=True):
        cache = kwargs[name].view(-1, 128)
        if check_untouched:
            cache = cache.cpu()
            keep = torch.ones(cache.shape[0], dtype=torch.bool)
            keep[slots] = False
            assert (
                (cache[keep] == -7).all().item()
            ), f"{name}: untouched row was modified"
        actual = cache[slots.to(cache.device)].cpu()
        assert torch.isfinite(actual).all().item()
        torch.testing.assert_close(actual, expected, atol=0.032, rtol=0.01)
