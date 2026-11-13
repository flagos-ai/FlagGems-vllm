# benchmark/test_mhc_pre_broadcast.py

import importlib
import warnings

import pytest
import torch

from flaggems_vllm.ops.mhc.mhc_pre_broadcast import (
    derive_fn_broadcast,
    mhc_pre_broadcast,
    mhc_pre_broadcast_ref,
)

import flaggems_vllm
from . import base

VENDOR = flaggems_vllm.vendor_name

# nvidia ships a dedicated tilelang kernel for the broadcast fast path
# (hc_mult == 4, all residual streams identical at model entry):
# vllm.model_executor.kernels.mhc.tilelang.mhc_pre_broadcast_tilelang.
# Checked and confirmed absent: hygon's aiter module (only exposes
# mhc_pre_aiter) and thead's vllm.model_executor.layers.mhc (only exposes
# MHCPreOp/MHCPostOp/MHCFusedPostPreOp/HCHeadOp, no broadcast variant).
# A vendor added here must expose the same positional signature as
# mhc_pre_broadcast (see its docstring for argument order).
_NATIVE_MHC_PRE_BROADCAST = {
    "nvidia": ("vllm.model_executor.kernels.mhc.tilelang", "mhc_pre_broadcast_tilelang"),
}


def _get_baseline_mhc_pre_broadcast_op():
    """Vendor-native mhc_pre_broadcast if available, else the PyTorch
    reference implementation already shipped alongside the Triton kernel
    (mirrors how hc_split_sinkhorn / mhc_bwd fall back to their own
    torch_ref / _ref when no vendor-native kernel exists)."""
    entry = _NATIVE_MHC_PRE_BROADCAST.get(VENDOR)
    if entry is not None:
        module_name, attr = entry
        try:
            return getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            warnings.warn(
                f"mhc_pre_broadcast benchmark: {VENDOR} declares a native "
                f"baseline at {module_name}.{attr} but it could not be "
                f"loaded ({exc!r}); falling back to the PyTorch reference.",
                RuntimeWarning,
                stacklevel=2,
            )
    return mhc_pre_broadcast_ref


class MHCPreBroadcastBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, hidden_size"

    def __init__(self, *args, hc_mult=4, sinkhorn_repeat=10, **kwargs):
        self.hc_mult = hc_mult
        self.sinkhorn_repeat = sinkhorn_repeat
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 1280),
            (8, 1280),
            (512, 1280),
            (512, 2560),
            (512, 4096),
            (1024, 1280),
            (1024, 2560),
            (1024, 4096),
            (2048, 1280),
            (2048, 2560),
            (2048, 4096),
            (8192, 1280),
            (8192, 2560),
            (8192, 4096),
        ]

    def get_input_iter(self, dtype):
        for n, hidden_size in self.shapes:
            hc_mult = self.hc_mult
            hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
            device = self.device

            torch.manual_seed(42)
            residual = torch.randn(
                (n, hidden_size), dtype=torch.float, device=device
            ).bfloat16()
            fn = torch.randn(
                (hc_mult3, hc_mult * hidden_size), dtype=torch.float, device=device
            ) * 1e-4
            fn_broadcast = derive_fn_broadcast(fn, hc_mult)
            hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
            hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1
            norm_weight = torch.randn(
                (hidden_size,), dtype=torch.float, device=device
            ).bfloat16()

            yield (
                residual,
                fn_broadcast,
                hc_scale,
                hc_base,
                1e-6,  # rms_eps
                1e-6,  # hc_pre_eps
                1e-6,  # hc_sinkhorn_eps
                1.0,  # hc_post_mult_value
                self.sinkhorn_repeat,
                norm_weight,
            )


@pytest.mark.mhc_pre_broadcast
def test_mhc_pre_broadcast():
    bench = MHCPreBroadcastBenchmark(
        op_name="mhc_pre_broadcast",
        torch_op=_get_baseline_mhc_pre_broadcast_op(),
        gems_op=mhc_pre_broadcast,
        dtypes=[torch.bfloat16],
    )
    bench.run()

