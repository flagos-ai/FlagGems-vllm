# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Token/pair reference for MoE LoRA, independent of aligned block metadata."""

import os
from contextlib import contextmanager
from datetime import timedelta

import pytest
import torch

from baseline import fused_moe_lora as workload


def _groups(binding, args):
    # Logical top-k experts are captured before alignment in the workload.
    # Never use the kernel's sorted IDs, block leaders or padded counts here.
    adapters, experts, _ = workload.routes(binding)
    enabled = args[13].cpu().tolist()
    groups = {}
    for pair, expert in enumerate(experts):
        adapter = adapters[pair // binding["topk"]]
        if expert >= 0 and adapter >= 0 and enabled[adapter]:
            groups.setdefault((adapter, expert), []).append(pair)
    return groups


def _small(binding):
    return (
        not binding["sorted"]
        and binding["tp"] == "none"
        and binding["r"] <= 64
        and binding["t"] * binding["topk"] * binding["r"] <= 1024
    )


def _reference(binding, shards, rank):
    args = shards[rank]
    initial, _, _, b_list, weights = args[:5]
    dtype = initial.dtype
    small = _small(binding)
    expected = initial.detach().cpu().clone()
    if not expected.numel():
        return expected
    out = expected.view(binding["t"] * binding["topk"], expected.shape[-1])
    offset, add = args[30:32]
    b_list = [v.detach().cpu().float() for v in b_list]
    inputs = [
        (a[1].detach().cpu().float(), [v.detach().cpu().float() for v in a[2]])
        for a in shards
    ]
    weights = weights.detach().cpu().view(-1, 1)
    for (adapter, expert), pairs in _groups(binding, args).items():
        pairs = torch.tensor(pairs, dtype=torch.long)
        rows = pairs if binding["weighted"] else pairs // binding["topk"]
        start = offset
        for s, b in enumerate(b_list):
            partials = []
            for x, a_list in inputs:
                low = x[rows] @ a_list[s][adapter, expert].T
                partials.append(low if small else low.to(dtype))
            if binding["tp"] == "reduce":
                low = torch.stack(partials).float().sum(0).to(dtype)
            elif binding["tp"] == "gather":
                low = torch.cat(partials, dim=1)
            else:
                low = partials[0]
            delta = low.float() @ b[adapter, expert].T
            if binding["weighted"]:
                delta *= weights[pairs]
            if not small:
                delta = delta.to(dtype).float()
            columns = slice(start, start + b.shape[2])
            if add:
                delta += out[pairs, columns].float()
            out[pairs, columns] = delta.to(dtype)
            start += b.shape[2]
    return expected


def reference(binding, args):
    return _reference(binding, [args], 0)


def reference_tp(binding, all_rank_args, rank):
    if len(all_rank_args) != binding["world"] or len(all_rank_args) < 2:
        raise ValueError(
            "reference requires the actual group size and every input shard"
        )
    return _reference(binding, all_rank_args, rank)


def assert_result(binding, args, result, expected, *, exact=False):
    assert result is None
    actual = args[0].detach().cpu()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    pairs = [p for group in _groups(binding, args).values() for p in group]
    # Untouched lanes and before/after columns are exact even for add_inputs=False.
    touched = torch.zeros(actual.shape[:2], dtype=torch.bool)
    touched.view(-1)[pairs] = True
    torch.testing.assert_close(actual[~touched], expected[~touched], atol=0, rtol=0)
    start = args[30]
    stop = start + sum(b.shape[2] for b in args[3])
    for columns in (slice(None, start), slice(stop, None)):
        torch.testing.assert_close(
            actual[..., columns], expected[..., columns], atol=0, rtol=0
        )
    # Two output ULPs near unity accommodate different dot reduction orders.
    eps = torch.finfo(actual.dtype).eps
    torch.testing.assert_close(
        actual, expected, rtol=0 if exact else 2 * eps, atol=0 if exact else eps
    )


def _readonly(args):
    for arg in args[1:]:
        if isinstance(arg, torch.Tensor):
            yield arg
        elif isinstance(arg, (list, tuple)):
            yield from arg


def snapshot_inputs(args):
    return [t.detach().cpu().clone() for t in _readonly(args)]


def assert_inputs_unchanged(args, before):
    for t, old in zip(_readonly(args), before):
        torch.testing.assert_close(t.cpu(), old, rtol=0, atol=0)


def _check(binding, args, *, exact=False, expected=None):
    import flaggems_vllm as gems

    before = snapshot_inputs(args)
    if expected is None:
        expected = reference(binding, args)
    physical = args[0]._base
    if physical is not None:
        guard = torch.ones_like(physical, dtype=torch.bool)
        guard.as_strided(
            args[0].shape, args[0].stride(), args[0].storage_offset()
        ).fill_(False)
        untouched = physical[guard].clone()
    actual = gems.fused_moe_lora(*args)
    assert_result(binding, args, actual, expected, exact=exact)
    assert_inputs_unchanged(args, before)
    if physical is not None:
        torch.testing.assert_close(physical[guard], untouched, rtol=0, atol=0)


@pytest.mark.fused_moe_lora
@pytest.mark.parametrize("binding", workload.grid(), ids=lambda b: b["case"])
@pytest.mark.parametrize("dtype", workload.DTYPES)
def test_fused_moe_lora(binding, dtype):
    import flaggems_vllm as gems

    args, _ = workload.build_inputs(binding, dtype, gems.device)
    _check(binding, args)


@pytest.mark.fused_moe_lora
@pytest.mark.parametrize("dtype", workload.DTYPES)
def test_fused_moe_lora_weighted_sorted_prefill(dtype):
    import flaggems_vllm as gems

    binding = workload._case(
        "weighted_sorted_prefill",
        160,
        8,
        16,
        2048,
        128,
        e=4,
        l=4,
        s=1,
        sorted=True,
        offset=3,
        add=False,
        weighted=True,
        layout="strided",
        block=64,
        route=1,
    )
    original, _ = workload.build_inputs(binding, dtype, gems.device)
    args = list(original)
    args[12] = args[12].to(gems.device)
    _check(binding, args)


@pytest.mark.fused_moe_lora
@pytest.mark.parametrize("routed_dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_lora_strided_metadata_and_unequal_slices(routed_dtype):
    import flaggems_vllm as gems

    binding = workload._case(
        "slice_layout",
        11,
        2,
        8,
        31,
        17,
        sorted=True,
        offset=5,
        weighted=True,
        add=False,
    )
    original, _ = workload.build_inputs(binding, torch.float16, gems.device)
    args = list(original)
    # The second slice has a different width and reads disjoint physical columns.
    args[3] = [args[3][0], args[3][1][:, :, ::2, :]]
    for i in (4, 5, 6, 7, 8, 11, 13):
        src = args[i]
        dtype = routed_dtype if i == 4 else (torch.bool if i == 13 else torch.int64)
        padded = torch.zeros((*src.shape, 3), dtype=dtype, device=src.device)
        args[i] = padded[..., 1]
        args[i].copy_(src)
    args[12] = args[12].to(gems.device)
    _check(binding, tuple(args))


@pytest.mark.fused_moe_lora
@pytest.mark.parametrize("sorted_mode", [False, True])
def test_fused_moe_lora_route_update(sorted_mode):
    import flaggems_vllm as gems

    base = workload._case(
        "route_update", 35, 2, 7, 32, 37, sorted=sorted_mode, weighted=True, offset=2
    )
    args, _ = workload.build_inputs(base, torch.float16, gems.device)
    original = args[0].clone()
    _check(base, args)
    updated = dict(base, route=1)
    new, _ = workload.build_inputs(updated, args[0].dtype, gems.device)
    # Same buffers and addresses across calls, including the output reset.
    args[0].copy_(original)
    for i in (4, 5, 6, 7, 8, 11, 12, 13):
        if args[i] is not None:
            args[i].copy_(new[i])
    _check(updated, args)


@pytest.mark.fused_moe_lora
@pytest.mark.parametrize("dtype", workload.DTYPES)
@pytest.mark.parametrize("rank,sorted_mode", [(4, False), (4, True), (129, False)])
@pytest.mark.parametrize("boundary", ["rank", "add"])
def test_fused_moe_lora_rounding(dtype, rank, sorted_mode, boundary):
    import flaggems_vllm as gems

    binding = workload._case(
        "round", 1, 1, rank, 32, 16, l=1, e=1, s=1, sorted=sorted_mode
    )
    args, _ = workload.build_inputs(binding, dtype, gems.device)
    small = _small(binding)
    y, x, (a,), (b,) = args[:4]
    eps = torch.finfo(dtype).eps
    # At 1, 1+eps/2 rounds to 1. Subtracting 1 recovers precisely
    # eps/2 when the addition is performed before the low-precision cast.
    x.fill_(1)
    a.zero_()
    b.zero_()
    if boundary == "rank":
        y.zero_()
        a[:, :, 0, 0] = 1
        a[:, :, 0, 1] = eps / 2
        a[:, :, 1, 0] = 1
        b[:, :, :, 0] = 1
        b[:, :, :, 1] = -1
    else:
        y.fill_(-1)
        a[:, :, 0, 0] = 1
        a[:, :, 1, 0] = 1
        b[:, :, :, 0] = 1
        b[:, :, :, 1] = eps / 2
    expected = torch.full_like(y, eps / 2 if small else 0, device="cpu")
    torch.testing.assert_close(reference(binding, args), expected, rtol=0, atol=0)
    _check(binding, args, exact=True, expected=expected)


@pytest.mark.fused_moe_lora
@pytest.mark.parametrize("dtype", workload.DTYPES)
@pytest.mark.parametrize("sorted_mode", [False, True])
def test_fused_moe_lora_nonfinite_and_signed_zero(dtype, sorted_mode):
    import flaggems_vllm as gems

    binding = workload._case(
        "nonfinite",
        4,
        1,
        3,
        1,
        1,
        l=1,
        e=1,
        s=1,
        sorted=sorted_mode,
        offset=1,
        add=False,
    )
    args, _ = workload.build_inputs(binding, dtype, gems.device)
    out, x, (a,), (b,) = args[:4]
    out.fill_(-0.0)
    x.copy_(
        torch.tensor(
            [[float("nan")], [float("inf")], [-float("inf")], [2.0]], dtype=dtype
        )
    )
    a.fill_(1)
    b.fill_(1)
    args[6].zero_()
    args[8].zero_()
    args[8][-1] = -1
    if sorted_mode:
        args[5].masked_fill_(args[5] == 3, 4)
    before = [
        t.detach().cpu().contiguous().view(torch.uint8).clone() for t in _readonly(args)
    ]
    bits = out.cpu().view(torch.int16).clone()
    result = gems.fused_moe_lora(*args)
    assert result is None
    values = out.cpu()[:, 0, 1]
    assert values[0].isnan() and values[1].isposinf() and values[2].isneginf()
    after = out.cpu().view(torch.int16)
    keep = torch.ones_like(bits, dtype=torch.bool)
    keep[:3, 0, 1] = False
    torch.testing.assert_close(after[keep], bits[keep], rtol=0, atol=0)
    for tensor, original in zip(_readonly(args), before):
        torch.testing.assert_close(
            tensor.cpu().contiguous().view(torch.uint8), original, rtol=0, atol=0
        )


@pytest.mark.fused_moe_lora
@pytest.mark.parametrize(
    "trigger",
    [
        "zero_active",
        "zero_padded",
        "negative_block",
        "negative_pair",
        "out_of_range_adapter",
        "out_of_range_expert",
        "int64_adapter_wraparound",
        "int64_expert_wraparound",
    ],
)
def test_fused_moe_lora_invalid_lanes(trigger):
    import flaggems_vllm as gems

    sorted_mode = trigger in (
        "zero_active",
        "zero_padded",
        "negative_block",
        "negative_pair",
        "int64_adapter_wraparound",
        "int64_expert_wraparound",
    )
    binding = workload._case(
        "invalid_lanes", 1, 1, 8, 32, 17, e=1, l=1, s=1, sorted=sorted_mode, add=False
    )
    args, _ = workload.build_inputs(binding, torch.float16, gems.device)
    if trigger == "zero_active":
        args[12].zero_()
    elif trigger == "zero_padded":
        args[7].zero_()
    elif trigger == "negative_block":
        args[6].fill_(-1)
    elif trigger == "negative_pair":
        args[5][0, 0] = -1
    elif trigger in ("int64_adapter_wraparound", "int64_expert_wraparound"):
        # Narrowing this invalid ID to int32 would select valid adapter/expert 0.
        index = 11 if trigger == "int64_adapter_wraparound" else 6
        args = list(args)
        args[index] = torch.full_like(args[index], 2**32, dtype=torch.int64)
    elif trigger == "out_of_range_adapter":
        args[8].fill_(binding["l"])
    else:
        args[6].fill_(binding["e"])
    _check(binding, args, exact=True, expected=args[0].cpu().clone())


@pytest.mark.fused_moe_lora
@pytest.mark.parametrize(
    "kind",
    [
        "output_width",
        "output_dtype",
        "x_rows",
        "slice_count",
        "rank",
        "topk",
        "offset",
        "naive_experts",
        "sorted_counts",
        "block_m",
        "block_m_mismatch",
        "tp_overwrite",
    ],
)
def test_fused_moe_lora_invalid_shapes(kind):
    import flaggems_vllm as gems

    sorted_mode = kind in ("sorted_counts", "block_m", "block_m_mismatch")
    binding = workload._case(
        "invalid_shapes", 3, 2, 8, 32, 16, weighted=True, sorted=sorted_mode
    )
    raw, _ = workload.build_inputs(binding, torch.float16, gems.device)
    args = list(raw)
    if kind == "output_width":
        args[0] = args[0][..., :-1]
    elif kind == "output_dtype":
        args[0] = args[0].float()
    elif kind == "x_rows":
        args[1] = args[1][:-1]
    elif kind == "slice_count":
        args[2] = args[2][:-1]
    elif kind == "rank":
        args[9] += 1
    elif kind == "topk":
        args[10] += 1
    elif kind == "offset":
        args[30] = -1
    elif kind == "naive_experts":
        args[6] = args[6][:-1]
    elif kind == "sorted_counts":
        args[7] = None
    elif kind == "block_m":
        args[14] = args[21] = 17
    elif kind == "block_m_mismatch":
        args[21] = 32
    else:
        args[29], args[31] = True, False
    before = args[0].clone()
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        gems.fused_moe_lora(*args)
    torch.testing.assert_close(args[0], before, rtol=0, atol=0)


@contextmanager
def tp_context():
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world < 2:
        pytest.skip("requires torchrun with two or more accelerator ranks")
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform

    import flaggems_vllm as gems

    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    device = torch.device(torch.device(gems.device).type, local)
    accelerator = torch.get_device_module(device)
    visible = accelerator.device_count()
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    if not (0 <= rank < world and 0 <= local < local_world <= visible):
        raise ValueError(
            f"invalid torchrun topology: rank={rank}/{world}, local={local}/{local_world}, "
            f"visible {device.type} devices={visible}"
        )
    accelerator.set_device(device)
    assert accelerator.current_device() == local
    backend = current_platform.dist_backend
    if backend == "gloo" or not torch.distributed.is_backend_available(backend):
        raise RuntimeError(f"accelerator TP backend unavailable: {backend}")
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=world, distributed_executor_backend="external_launcher"
        )
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world, rank, "env://", local, backend, timeout=timedelta(minutes=5)
        )
        try:
            initialize_model_parallel(tensor_model_parallel_size=world, backend=backend)
            try:
                group = get_tp_group()
                assert group.world_size == world and group.rank_in_group == rank
                yield rank, world, device
            finally:
                destroy_model_parallel()
        finally:
            destroy_distributed_environment()


def tp_errors(local_error):
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    results = [None] * group.world_size
    torch.distributed.all_gather_object(results, local_error, group=group.cpu_group)
    return [
        f"rank {rank}: {err}" for rank, err in enumerate(results) if err is not None
    ]


@pytest.fixture(scope="module")
def moe_lora_tp_group():
    with tp_context() as group:
        yield group


@contextmanager
def _tp_stage(name):
    local_error = None
    try:
        yield
    except Exception as exc:
        local_error = f"{name}: {type(exc).__name__}: {exc}"
    failures = tp_errors(local_error)
    if failures:
        pytest.fail("\n".join(failures), pytrace=False)


@pytest.mark.fused_moe_lora
@pytest.mark.parametrize(
    "collective",
    ["reduce", "gather"],
    ids=lambda mode: f"{mode}-rank{os.environ.get('RANK', '0')}",
)
@pytest.mark.parametrize("dtype", workload.DTYPES, ids=str)
def test_fused_moe_lora_tensor_parallel(moe_lora_tp_group, collective, dtype):
    import flaggems_vllm as gems

    rank, world, device = moe_lora_tp_group
    binding = next(b for b in workload.tp_grid(world) if b["tp"] == collective)
    with _tp_stage("prepare shards and oracle"):
        args, _ = workload.build_tp_inputs(binding, dtype, device, rank)
        host = [
            workload.build_tp_inputs(binding, dtype, "cpu", i)[0] for i in range(world)
        ]
        expected = reference_tp(binding, host, rank)
        initial = args[0].clone()
        before = snapshot_inputs(args)
    for replay in range(2):
        with _tp_stage(f"restore {replay}"):
            args[0].copy_(initial)
            torch.get_device_module(device).synchronize(device)
        with _tp_stage(f"operator {replay}"):
            got = gems.fused_moe_lora(*args)
            torch.get_device_module(device).synchronize(device)
        with _tp_stage(f"compare {replay}"):
            assert_result(binding, args, got, expected)
            assert_inputs_unchanged(args, before)
