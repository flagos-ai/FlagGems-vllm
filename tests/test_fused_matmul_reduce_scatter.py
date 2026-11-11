# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Run communication tests with torchrun --nproc-per-node=2 -m pytest ... ."""

import importlib
import inspect
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import flaggems_vllm
from flaggems_vllm.ops._fused_matmul_reduce_scatter_runtime import (
    _FlagCXAllocation,
    _validate_topology,
)
from flaggems_vllm.ops.fused_matmul_reduce_scatter import (
    GemmReduceScatterConfig,
    _validate_shape,
    clear_fused_matmul_reduce_scatter_cache,
    configure_fused_matmul_reduce_scatter,
    fused_matmul_reduce_scatter,
)

pytestmark = pytest.mark.fused_matmul_reduce_scatter


def test_signature_and_export():
    assert list(inspect.signature(fused_matmul_reduce_scatter).parameters) == [
        "A",
        "B",
        "reduce_op",
        "scatter_dim",
        "group_name",
    ]
    assert flaggems_vllm.fused_matmul_reduce_scatter is fused_matmul_reduce_scatter
    assert "fused_matmul_reduce_scatter" in flaggems_vllm._LIBRARY_ONLY_OPS


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"num_scatter_sms": 0}, ValueError),
        ({"block_m": 96}, ValueError),
        ({"rs_block_n": 256}, ValueError),
        ({"num_warps": 3}, ValueError),
        ({"block_k": True}, ValueError),
        ({"epilogue_subtile": 1}, ValueError),
    ],
)
def test_invalid_config(kwargs, exception):
    with pytest.raises(exception):
        GemmReduceScatterConfig(**kwargs).validate()


@pytest.mark.parametrize(
    "shape,world,exception",
    [
        ((512, 256, 256), 1, NotImplementedError),
        ((513, 256, 256), 2, ValueError),
        ((256, 256, 256), 2, NotImplementedError),
        ((768, 255, 256), 2, NotImplementedError),
        ((768, 256, 255), 2, NotImplementedError),
    ],
)
def test_unsupported_shape(shape, world, exception):
    with pytest.raises(exception):
        _validate_shape(*shape, world, GemmReduceScatterConfig())


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.complex64])
def test_unsupported_dtype(dtype):
    a = torch.empty((512, 64), dtype=dtype)
    b = torch.empty((64, 256), dtype=dtype)
    with pytest.raises(NotImplementedError, match="FP16 and BF16"):
        fused_matmul_reduce_scatter(a, b, "sum", 0, "unused")


@pytest.mark.parametrize(
    "reduce_op,dim,error", [("max", 0, ValueError), ("sum", 1, NotImplementedError)]
)
def test_unsupported_semantics(reduce_op, dim, error):
    a = torch.empty((512, 64), dtype=torch.float16)
    b = torch.empty((64, 256), dtype=torch.float16)
    with pytest.raises(error):
        fused_matmul_reduce_scatter(a, b, reduce_op, dim, "unused")


def _records(hosts):
    return [
        {"host": host, "device": f"gpu-{rank}", "signature": [512, 256, 64]}
        for rank, host in enumerate(hosts)
    ]


def test_group_topology_validation():
    assert _validate_topology(_records(["node0", "node0", "node1", "node1"])) == 2
    assert _validate_topology(_records(["node0", "node1"])) == 1
    with pytest.raises(NotImplementedError):
        _validate_topology(_records(["node0", "node1", "node0", "node1"]))
    with pytest.raises(NotImplementedError):
        _validate_topology(_records(["node0", "node0", "node1"]))
    mismatch = _records(["node0", "node0"])
    mismatch[1]["signature"] = [1024, 256, 64]
    with pytest.raises(ValueError, match="identical"):
        _validate_topology(mismatch)
    duplicate = _records(["node0", "node0"])
    duplicate[1]["device"] = duplicate[0]["device"]
    with pytest.raises(ValueError, match="distinct GPU"):
        _validate_topology(duplicate)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flagcx_allocation_view_is_zero_copy():
    # Exercise the ownership/view contract using a real CUDA allocation without
    # needing FlagCX or a second GPU. Production uses flagcxMemAlloc instead.
    import ctypes

    original = torch.arange(1024, dtype=torch.int32, device="cuda")
    freed = []
    library = SimpleNamespace(
        flagcxMemAlloc=lambda size: ctypes.c_void_p(original.data_ptr()),
        flagcxMemFree=lambda ptr: freed.append(ptr.value),
    )
    device = original.device
    owner = _FlagCXAllocation(
        library, original.numel() * original.element_size(), device
    )
    view = torch.as_tensor(owner, device=device).view(torch.int32)
    assert view.data_ptr() == original.data_ptr()
    torch.testing.assert_close(view, original)
    owner.release()
    owner.release()
    assert freed == [original.data_ptr()]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("subtile", [False, True])
def test_tma_producer_and_final_reduction(dtype, subtile):
    # Producer readiness and the final TMA reduction can be tested on one GPU.
    # This deliberately does not claim to validate remote transfers/barriers.
    pytest.importorskip("triton.experimental.tle.language")
    kernels = importlib.import_module(
        "flaggems_vllm.ops._fused_matmul_reduce_scatter_tle"
    )
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires SM90+")
    device = torch.device("cuda", torch.cuda.current_device())
    kernels.triton.set_allocator(
        lambda size, alignment, stream: torch.empty(
            size, device=device, dtype=torch.int8
        )
    )
    torch.manual_seed(41)
    a = torch.randn((512, 256), dtype=dtype, device=device) * 0.1
    weight = torch.randn((256, 256), dtype=dtype, device=device) * 0.1
    output = torch.empty((512, 256), dtype=dtype, device=device)
    signals = torch.zeros((2,), dtype=torch.int32, device=device)
    counters = torch.zeros_like(signals)
    kernels.gemm_rs_producer_persistent(
        a,
        weight,
        output,
        signals,
        counters,
        2,
        2,
        0,
        32,
        BLOCK_SIZE_N=128,
        STAGES=2,
        EPILOGUE_SUBTILE=subtile,
    )
    torch.testing.assert_close(output, a @ weight.T, rtol=0.01, atol=0.005)
    torch.testing.assert_close(signals, torch.ones_like(signals))
    torch.testing.assert_close(counters, torch.full_like(counters, 4))
    reduced = torch.empty((256, 256), dtype=dtype, device=device)
    kernels._ring_reduce_tma_kernel[(8,)](
        output,
        reduced,
        256,
        256,
        LOCAL_RANK=0,
        WORLD_SIZE=2,
        BLOCK_M=128,
        BLOCK_N=64,
        SCALE=0.5,
        num_warps=4,
    )
    expected = ((output[:256] + output[256:]).float() * 0.5).to(dtype)
    torch.testing.assert_close(reduced, expected)


@pytest.fixture(scope="module")
def distributed_context():
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("launch with torchrun and at least two ranks")
    assert torch.cuda.is_available(), "distributed launch requires CUDA"
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    assert torch.cuda.get_device_capability(device)[0] >= 9
    # An explicit distributed launch should fail visibly if dependencies are absent.
    importlib.import_module("triton.experimental.tle.language")
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", device_id=device)
    with flaggems_vllm.use_symm_mem():
        yield dist.group.WORLD, device
    clear_fused_matmul_reduce_scatter_cache()
    assert dist.is_initialized()
    if owns_group:
        dist.destroy_process_group()


def _assert_all_ranks_close(actual, expected, group):
    failure = None
    try:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.04 if actual.dtype == torch.bfloat16 else 0.01,
            atol=0.005 if actual.dtype == torch.bfloat16 else 0.001,
        )
    except AssertionError as exc:
        failure = str(exc)
    failed = torch.tensor(int(failure is not None), device=actual.device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
    assert not failed.item(), failure or "a peer rank failed correctness validation"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduce_op", ["sum", "avg"])
@pytest.mark.parametrize("layout", ["contiguous", "transposed"])
@pytest.mark.parametrize("rows_per_rank", [256, 384])
def test_fused_matmul_reduce_scatter(
    distributed_context, dtype, reduce_op, layout, rows_per_rank
):
    group, device = distributed_context
    m = rows_per_rank * group.size()
    torch.manual_seed(1234 + group.rank())
    if layout == "transposed":
        a = torch.randn((256, m), device=device, dtype=dtype).T
        b = torch.randn((256, 256), device=device, dtype=dtype).T
    else:
        a = torch.randn((m, 256), device=device, dtype=dtype)
        b = torch.randn((256, 256), device=device, dtype=dtype)
    a.mul_(0.1)
    b.mul_(0.1)
    saved = []
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    for iteration in range(3):
        # Alternate caller streams while reusing exactly the same context.
        selected = stream if iteration % 2 else torch.cuda.current_stream(device)
        with torch.cuda.stream(selected):
            a.add_(0.001)
            actual = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                a, b, reduce_op, 0, group.group_name
            )
            partial = a @ b
            reference = torch.empty_like(actual)
            dist.reduce_scatter_tensor(reference, partial, group=group)
            if reduce_op == "avg":
                reference.div_(group.size())
        torch.cuda.current_stream(device).wait_stream(selected)
        stream.wait_stream(torch.cuda.current_stream(device))
        _assert_all_ranks_close(actual, reference, group)
        for previous, expected in saved:
            _assert_all_ranks_close(previous, expected, group)
        saved.append((actual, reference))
    clear_fused_matmul_reduce_scatter_cache(group.group_name)
    assert dist.get_world_size(group) == group.size()


def test_subgroup_and_configuration(distributed_context):
    world, device = distributed_context
    if world.size() < 4 or world.size() % 2:
        pytest.skip("subgroup isolation test requires an even world size >=4")
    groups = [
        dist.new_group(list(range(start, start + 2)))
        for start in range(0, world.size(), 2)
    ]
    group = groups[world.rank() // 2]
    configure_fused_matmul_reduce_scatter(
        group.group_name, GemmReduceScatterConfig(block_n=128)
    )
    a = torch.full(
        (512, 64), 0.01 * (world.rank() + 1), device=device, dtype=torch.float16
    )
    b = torch.full((64, 256), 0.02, device=device, dtype=torch.float16)
    actual = fused_matmul_reduce_scatter(a, b, "sum", 0, group.group_name)
    expected = torch.empty_like(actual)
    dist.reduce_scatter_tensor(expected, a @ b, group=group)
    _assert_all_ranks_close(actual, expected, group)
    with pytest.raises(RuntimeError, match="clear"):
        configure_fused_matmul_reduce_scatter(
            group.group_name, GemmReduceScatterConfig()
        )
    clear_fused_matmul_reduce_scatter_cache(group.group_name)
    assert world.size() == dist.get_world_size()
    dist.barrier()
    dist.destroy_process_group(group)


def test_rank_metadata_mismatch(distributed_context):
    group, device = distributed_context
    m = (256 if group.rank() == 0 else 512) * group.size()
    a = torch.empty((m, 64), device=device, dtype=torch.float16)
    b = torch.empty((64, 256), device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="identical"):
        fused_matmul_reduce_scatter(a, b, "sum", 0, group.group_name)
