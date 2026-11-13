# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Correctness and dispatcher tests; launch collective cases with torchrun."""

import importlib
import inspect
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import flaggems_vllm
from flaggems_vllm.ops._fused_matmul_reduce_scatter_runtime import (
    _FlagCXAllocation,
    _validate_topology,
)
from flaggems_vllm.ops.fused_matmul_reduce_scatter import (
    GemmReduceScatterConfig,
    _default_config,
    _get_backend,
    _validate_device,
    _validate_shape,
    clear_fused_matmul_reduce_scatter_cache,
    configure_fused_matmul_reduce_scatter,
    fused_matmul_reduce_scatter,
)
from flaggems_vllm.testing.fused_matmul_reduce_scatter import (
    assert_close as assert_rs_close,
)
from flaggems_vllm.testing.fused_matmul_reduce_scatter import (
    error_metrics,
    make_exact_case,
    make_reference,
    roundoff_tolerance,
)

pytestmark = pytest.mark.fused_matmul_reduce_scatter


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp32_reference_budget_rejects_corruption_and_nonfinite_values(dtype):
    generator = torch.Generator().manual_seed(71)
    partials = torch.randn((8, 128, 64), generator=generator)
    expected = partials.sum(dim=0)
    tolerance = roundoff_tolerance(expected, partials.abs().sum(dim=0), dtype)
    # Independent arithmetic example with three low-precision storage rounds.
    stored = partials.to(dtype).float()
    subtotals = stored.reshape(2, 4, 128, 64).sum(dim=1).to(dtype).float()
    rounded = subtotals.sum(dim=0).to(dtype)
    assert error_metrics(rounded, expected, tolerance)["outside_budget"] == 0
    for corrupted in (torch.zeros_like(rounded), rounded.roll(1, 0), rounded / 8):
        metrics = error_metrics(corrupted, expected, tolerance)
        assert metrics["outside_budget"] > rounded.numel() // 2
    for invalid in (float("nan"), float("inf"), -float("inf")):
        corrupted = rounded.clone()
        corrupted[0, 0] = invalid
        assert error_metrics(corrupted, expected, tolerance)["outside_budget"] > 0


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


def test_hcu_default_and_device_routing(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.3")
    monkeypatch.setitem(
        sys.modules, "triton._flagtree_backend", SimpleNamespace(FLAGTREE_BACKEND="hcu")
    )
    assert _get_backend() == "hcu"
    config = _default_config()
    config.validate()
    assert (config.block_m, config.block_n, config.block_k) == (64, 64, 32)
    assert config.estimated_shared_memory() <= 65536
    _validate_shape(512, 256, 256, 2, config)

    def cuda_only_query(*args, **kwargs):
        pytest.fail("HCU routing must not apply an NVIDIA compute capability gate")

    monkeypatch.setattr(torch.cuda, "get_device_capability", cuda_only_query)
    assert _validate_device(torch.device("cuda", 0)) == "hcu"


def test_other_hip_compilers_are_not_silently_accepted(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.3")
    monkeypatch.setitem(
        sys.modules, "triton._flagtree_backend", SimpleNamespace(FLAGTREE_BACKEND="amd")
    )
    with pytest.raises(NotImplementedError, match="FLAGTREE_BACKEND=hcu"):
        _get_backend()


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
    if torch.version.hip is not None:
        pytest.skip("this partial TMA check is NVIDIA-only; use torchrun for HCU")
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
    expected = ((output[:256].float() + output[256:].float()) * 0.5).to(dtype)
    torch.testing.assert_close(reduced, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("local_rank", range(4))
@pytest.mark.parametrize("scale", [1.0, 0.25])
def test_reduction_preserves_small_terms_during_cancellation(dtype, local_rank, scale):
    pytest.importorskip("triton.experimental.tle.language")
    kernels = importlib.import_module(
        "flaggems_vllm.ops._fused_matmul_reduce_scatter_tle"
    )
    if torch.version.hip is not None:
        pytest.skip(
            "this descriptor unit check requires NVIDIA; validate HCU with torchrun"
        )
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires SM90+")
    device = torch.device("cuda", torch.cuda.current_device())
    kernels.triton.set_allocator(
        lambda size, alignment, stream: torch.empty(
            size, device=device, dtype=torch.int8
        )
    )
    # All terms and the exact result are representable in either input dtype.
    # A low-precision running sum loses unit contributions for some rank orders.
    terms = torch.tensor([4096.0, 1.0, -4096.0, 1.0], dtype=dtype, device=device)
    source = terms[:, None, None].expand(4, 256, 64).contiguous().view(1024, 64)
    output = torch.empty((256, 64), dtype=dtype, device=device)
    kernels._ring_reduce_tma_kernel[(2,)](
        source,
        output,
        256,
        64,
        LOCAL_RANK=local_rank,
        WORLD_SIZE=4,
        BLOCK_M=128,
        BLOCK_N=64,
        SCALE=scale,
        num_warps=4,
    )
    torch.testing.assert_close(
        output, torch.full_like(output, 2.0 * scale), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduce_op", ["sum", "avg"])
def test_exact_rank_and_shard_identity(gemm_rs_context, dtype, reduce_op):
    group, device = gemm_rs_context
    if group.size() not in (2, 4, 8):
        pytest.skip("exact encoding covers 2, 4 or 8 ranks")
    a, b, expected = make_exact_case(group, device, dtype, reduce_op)
    actual = torch.ops.symm_mem.fused_matmul_reduce_scatter(
        a, b, reduce_op, 0, group.group_name
    )
    assert_rs_close(actual, expected, torch.zeros_like(expected), group, dtype=dtype)
    clear_fused_matmul_reduce_scatter_cache(group.group_name)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduce_op", ["sum", "avg"])
@pytest.mark.parametrize("layout", ["contiguous", "transposed"])
@pytest.mark.parametrize("rows_per_rank", [256, 384])
def test_fused_matmul_reduce_scatter(
    gemm_rs_context, dtype, reduce_op, layout, rows_per_rank
):
    group, device = gemm_rs_context
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
            reference, tolerance = make_reference(a, b, reduce_op, group)
        torch.cuda.current_stream(device).wait_stream(selected)
        stream.wait_stream(torch.cuda.current_stream(device))
        assert_rs_close(actual, reference, tolerance, group, dtype=dtype)
        for previous, expected, budget in saved:
            assert_rs_close(previous, expected, budget, group, dtype=dtype)
        saved.append((actual, reference, tolerance))
    clear_fused_matmul_reduce_scatter_cache(group.group_name)
    assert dist.get_world_size(group) == group.size()


def test_subgroup_and_configuration(gemm_rs_context):
    world, device = gemm_rs_context
    if world.size() < 4 or world.size() % 2:
        pytest.skip("subgroup isolation test requires an even world size >=4")
    groups = [
        dist.new_group(list(range(start, start + 2)))
        for start in range(0, world.size(), 2)
    ]
    group = groups[world.rank() // 2]
    configure_fused_matmul_reduce_scatter(group.group_name, _default_config())
    a = torch.full(
        (512, 64), 0.01 * (world.rank() + 1), device=device, dtype=torch.float16
    )
    b = torch.full((64, 256), 0.02, device=device, dtype=torch.float16)
    actual = fused_matmul_reduce_scatter(a, b, "sum", 0, group.group_name)
    expected, tolerance = make_reference(a, b, "sum", group)
    assert_rs_close(actual, expected, tolerance, group, dtype=a.dtype)
    with pytest.raises(RuntimeError, match="clear"):
        configure_fused_matmul_reduce_scatter(
            group.group_name, GemmReduceScatterConfig()
        )
    clear_fused_matmul_reduce_scatter_cache(group.group_name)
    assert world.size() == dist.get_world_size()
    dist.barrier()
    dist.destroy_process_group(group)


def test_rank_metadata_mismatch(gemm_rs_context):
    group, device = gemm_rs_context
    m = (256 if group.rank() == 0 else 512) * group.size()
    a = torch.empty((m, 64), device=device, dtype=torch.float16)
    b = torch.empty((64, 256), device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="identical"):
        fused_matmul_reduce_scatter(a, b, "sum", 0, group.group_name)


@pytest.fixture
def dispatch_probe(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA to exercise the CUDA dispatcher")
    implementation = importlib.import_module(
        "flaggems_vllm.ops.fused_matmul_reduce_scatter"
    )
    a = torch.empty((512, 64), device="cuda", dtype=torch.float16)
    b = torch.empty((64, 128), device="cuda", dtype=torch.float16)
    original_result = torch.empty((1,), device="cuda")
    replacement_result = torch.empty((2,), device="cuda")
    calls = []

    def replacement(A, B, reduce_op, scatter_dim, group_name):
        calls.append((A, B, reduce_op, scatter_dim, group_name))
        return replacement_result

    monkeypatch.setattr(implementation, "fused_matmul_reduce_scatter", replacement)
    monkeypatch.setattr(symm_mem, "_is_test_mode", False)
    monkeypatch.setattr(
        symm_mem, "_fused_matmul_reduce_scatter_impl", lambda **kwargs: original_result
    )
    # Capture the overload before registration, as vLLM may do during imports.
    op = torch.ops.symm_mem.fused_matmul_reduce_scatter.default
    args = (a, b, "avg", 0, "probe-group")
    assert op(*args) is original_result
    yield op, args, original_result, replacement_result, calls
    flaggems_vllm.disable_symm_mem()
    assert op(*args) is original_result


def test_scoped_override_and_restoration(dispatch_probe):
    op, args, original, replacement, calls = dispatch_probe
    schema = str(op._schema)
    table = torch._C._dispatch_dump_table("symm_mem::fused_matmul_reduce_scatter")
    bookkeeping = "symm_mem/fused_matmul_reduce_scatter/CUDA"
    before = bookkeeping in torch.library._impls
    with flaggems_vllm.use_symm_mem():
        assert op(*args) is replacement
        assert str(op._schema) == schema
        assert len(calls) == 1
        assert calls[0][0] is args[0] and calls[0][1] is args[1]
        assert calls[0][2:] == args[2:]
        after = torch._C._dispatch_dump_table("symm_mem::fused_matmul_reduce_scatter")
        assert [line for line in table.splitlines() if line.startswith("Meta:")] == [
            line for line in after.splitlines() if line.startswith("Meta:")
        ]
    assert op(*args) is original
    assert (bookkeeping in torch.library._impls) == before


def test_hcu_selection_still_registers_the_cuda_dispatch_key(
    dispatch_probe, monkeypatch
):
    # HIP PyTorch tensors dispatch through CUDA. This checks registration routing
    # with a probe callback, not execution of an HCU GPU kernel on NVIDIA hardware.
    monkeypatch.setattr(torch.version, "hip", "6.3")
    monkeypatch.setitem(
        sys.modules, "triton._flagtree_backend", SimpleNamespace(FLAGTREE_BACKEND="hcu")
    )
    op, args, original, replacement, _ = dispatch_probe
    with flaggems_vllm.use_symm_mem():
        assert op(*args) is replacement
    assert op(*args) is original


def test_nested_scope_and_exception(dispatch_probe):
    op, args, original, replacement, _ = dispatch_probe
    with flaggems_vllm.use_symm_mem():
        assert op(*args) is replacement
        with pytest.raises(ValueError, match="scope failure"):
            with flaggems_vllm.use_symm_mem():
                assert op(*args) is replacement
                raise ValueError("scope failure")
        assert op(*args) is replacement
    assert op(*args) is original


def test_persistent_activation_is_idempotent(dispatch_probe):
    op, args, original, replacement, _ = dispatch_probe
    flaggems_vllm.enable_symm_mem()
    flaggems_vllm.enable_symm_mem()
    assert op(*args) is replacement
    with flaggems_vllm.use_symm_mem():
        flaggems_vllm.disable_symm_mem()
        flaggems_vllm.disable_symm_mem()
        assert op(*args) is replacement
    assert op(*args) is original


def test_persistent_activation_outlives_scope(dispatch_probe):
    op, args, original, replacement, _ = dispatch_probe
    with flaggems_vllm.use_symm_mem():
        flaggems_vllm.enable_symm_mem()
        assert op(*args) is replacement
    assert op(*args) is replacement
    flaggems_vllm.disable_symm_mem()
    assert op(*args) is original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_actual_operator_reached_and_unsupported_inputs_raise():
    a = torch.empty((512, 64), device="cuda", dtype=torch.float32)
    b = torch.empty((64, 128), device="cuda", dtype=torch.float32)
    # The real implementation rejects the dtype before requiring a process group
    # or TLE import. No replacement compute or original Torch fallback is used.
    with flaggems_vllm.use_symm_mem():
        with pytest.raises(NotImplementedError, match="FP16 and BF16"):
            torch.ops.symm_mem.fused_matmul_reduce_scatter(a, b, "sum", 0, "unused")


def test_unsupported_override_api_fails_without_registration(monkeypatch):
    def old_impl(self, name, fn, dispatch_key=""):
        raise AssertionError("registration should not be attempted")

    monkeypatch.setattr(torch.library.Library, "impl", old_impl)
    with pytest.raises(NotImplementedError, match="allow_override"):
        flaggems_vllm.enable_symm_mem()
    registration = importlib.import_module("flaggems_vllm.runtime.symm_mem")
    assert registration._REFERENCES == 0
    assert registration._LIBRARY is None
