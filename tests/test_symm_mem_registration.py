# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Real CUDA dispatcher tests; no process group or FlagCX required."""

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed._symmetric_memory as symm_mem

import flaggems_vllm

pytestmark = pytest.mark.fused_matmul_reduce_scatter


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
