# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Real public-entry dispatch and failure contracts for MetaX numerical modes."""

import importlib
from unittest.mock import patch

import pytest
import torch

import flaggems_vllm as fgv
from flaggems_vllm.ops.fused_moe import MoEActivation, apply_moe_activation
from flaggems_vllm.runtime import device

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or fgv.vendor_name != "metax",
    reason="MetaX GPU required",
)


class KernelSpy:
    """Observe real launches while forwarding to the unchanged Triton kernel."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.calls = []

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def call(*args, **kwargs):
            self.calls.append((grid, kwargs))
            return launch(*args, **kwargs)

        return call


class UnexpectedKernel:
    def __getitem__(self, grid):
        raise AssertionError("Default path entered a MetaX numerical mode")


def qkv_args(m=3):
    x = torch.randn(m, 768, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(128, device="cuda", dtype=x.dtype)
    cs = torch.randn(32, 64, device="cuda", dtype=x.dtype)
    pos = torch.arange(m, device="cuda", dtype=torch.int64)
    return x, w, w, cs, pos, 4, 1, 64, 1e-6


def test_public_modes_reach_metax_implementation():
    backend = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.fused.numerical_modes"
    )
    x = torch.randn(3, 384, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(3, 192, device="cuda", dtype=x.dtype)
    summands = torch.randn(3, 8, 192, device="cuda", dtype=x.dtype)
    sw = KernelSpy(backend._swigluoai_division_flat_kernel)
    sm = KernelSpy(backend._moe_sum_general_kernel)
    qk = KernelSpy(backend._staged_minimax_m3_kernel)
    with (
        patch.object(backend, "_swigluoai_division_flat_kernel", sw),
        patch.object(backend, "_moe_sum_general_kernel", sm),
        patch.object(backend, "_staged_minimax_m3_kernel", qk),
    ):
        apply_moe_activation(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            out,
            x,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=7.0,
            sigmoid_mode="division",
        )
        fgv.moe_sum(summands, out, reduction="sequential")
        fgv.fused_minimax_m3_qknorm_rope_kv_insert(*qkv_args(), numerics="bf16_staged")
        torch.cuda.synchronize()
        assert (len(sw.calls), len(sm.calls), len(qk.calls)) == (1, 1, 2)
        assert sm.calls[0][1]["USE_INT64_INDEX"] is False
        assert all(meta["num_warps"] == 4 for _, meta in qk.calls)


def test_default_paths_do_not_call_metax_modes():
    backend = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.fused.numerical_modes"
    )
    x = torch.randn(3, 384, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(3, 192, device="cuda", dtype=x.dtype)

    with (
        patch.object(backend, "_swigluoai_division_flat_kernel", UnexpectedKernel()),
        patch.object(backend, "_moe_sum_general_kernel", UnexpectedKernel()),
        patch.object(backend, "_staged_minimax_m3_kernel", UnexpectedKernel()),
    ):
        apply_moe_activation(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            out,
            x,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=7.0,
        )
        fgv.moe_sum(x.view(3, 2, 192), out)
        # The native CUDA warp-shuffle kernel is not executable on MetaX.
        # Check its real public launch selection without claiming GPU numerics.
        common = importlib.import_module(
            "flaggems_vllm.ops.fused_minimax_m3_qknorm_rope_kv_insert"
        )
        with patch.object(common, "_fused_minimax_m3_kernel") as kernel:
            fgv.fused_minimax_m3_qknorm_rope_kv_insert(*qkv_args())
            kwargs = kernel.__getitem__.return_value.call_args.kwargs
            assert kwargs == {"num_warps": 1, "enable_fp_fusion": False}
        torch.cuda.synchronize()


def test_other_vendor_rejects_before_writes(monkeypatch):
    x = torch.randn(3, 384, device="cuda", dtype=torch.bfloat16)
    out = torch.full((3, 192), -77, device="cuda", dtype=x.dtype)
    args = qkv_args()
    original = args[0].cpu().clone()
    monkeypatch.setattr(device, "vendor_name", "nvidia")
    with pytest.raises(NotImplementedError, match="MetaX"):
        apply_moe_activation(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            out,
            x,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=7.0,
            sigmoid_mode="division",
        )
    with pytest.raises(NotImplementedError, match="MetaX"):
        fgv.moe_sum(x.view(3, 2, 192), out, reduction="sequential")
    with pytest.raises(NotImplementedError, match="MetaX"):
        fgv.fused_minimax_m3_qknorm_rope_kv_insert(*args, numerics="bf16_staged")
    assert torch.equal(args[0].cpu(), original)
    assert torch.equal(out.cpu(), torch.full((3, 192), -77, dtype=x.dtype))


def test_launch_failure_is_not_retried():
    backend = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.fused.numerical_modes"
    )
    x = torch.randn(3, 384, device="cuda", dtype=torch.bfloat16)
    out = torch.zeros(3, 192, device="cuda", dtype=x.dtype)
    calls = []

    def fail_after_write(input, output, *args, **kwargs):
        calls.append(1)
        output.fill_(17)
        raise RuntimeError("injected failure after a write")

    class FailingKernel:
        def __getitem__(self, grid):
            return fail_after_write

    with patch.object(backend, "_swigluoai_division_flat_kernel", FailingKernel()):
        with pytest.raises(RuntimeError, match="injected failure"):
            apply_moe_activation(
                MoEActivation.SWIGLUOAI_UNINTERLEAVE,
                out,
                x,
                gemm1_alpha=1.702,
                gemm1_beta=1.0,
                gemm1_clamp_limit=7.0,
                sigmoid_mode="division",
            )
    torch.cuda.synchronize()
    assert len(calls) == 1
    assert torch.equal(out.cpu(), torch.full((3, 192), 17, dtype=x.dtype))


def test_empty_staged_qkv_does_not_launch():
    backend = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.fused.numerical_modes"
    )
    with patch.object(backend, "launch_staged_qkv") as launch:
        fgv.fused_minimax_m3_qknorm_rope_kv_insert(*qkv_args(0), numerics="bf16_staged")
        launch.assert_not_called()


@pytest.mark.parametrize("tokens,wide", [(43690, False), (43691, True)])
def test_sequential_sum_index_range_boundary(tokens, wide):
    backend = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.fused.numerical_modes"
    )
    x = torch.empty(tokens, 8, 6144, device="meta", dtype=torch.bfloat16)
    out = torch.empty(tokens, 6144, device="meta", dtype=x.dtype)
    with patch.object(backend, "_moe_sum_general_kernel") as kernel:
        backend.sequential_moe_sum(x, out)
        assert (
            kernel.__getitem__.return_value.call_args.kwargs["USE_INT64_INDEX"] is wide
        )


@pytest.mark.parametrize("strided", ["input", "output"])
def test_sequential_sum_large_stride_uses_wide_index(strided):
    backend = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.fused.numerical_modes"
    )
    x = torch.empty_strided(
        (2, 8, 4),
        (2**31, 4, 1) if strided == "input" else (32, 4, 1),
        device="meta",
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        (2, 4),
        (2**31, 1) if strided == "output" else (4, 1),
        device="meta",
        dtype=x.dtype,
    )
    with patch.object(backend, "_moe_sum_general_kernel") as kernel:
        backend.sequential_moe_sum(x, out)
        assert (
            kernel.__getitem__.return_value.call_args.kwargs["USE_INT64_INDEX"] is True
        )
