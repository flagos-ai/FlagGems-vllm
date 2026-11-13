# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

# ruff: noqa: I001
os.environ["FLAGTREE_AABS"] = "0"

from itertools import product  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

import flaggems_vllm  # noqa: E402
from benchmark.base import Benchmark  # noqa: E402

VENDOR = flaggems_vllm.vendor_name


# -----------------------------------------------------------------------------
# Vendor-specific baseline import
# -----------------------------------------------------------------------------

# A platform build may ship FlagGems without the vendor-native vocabulary
# package (e.g. lightop on Hygon). A failed import leaves the handle None and
# degrades the run to latency-only mode instead of failing collection.
flashinfer_gemma_rmsnorm = None
hygon_op = None
npu_gemma_rms_norm = None
mthreads_gemma_rmsnorm = None
iluvatar_gemma_rms_norm = None
_BASELINE_IMPORT_ERROR = None

if VENDOR in ("nvidia", "thead"):
    try:
        os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        from flashinfer.norm import gemma_rmsnorm as flashinfer_gemma_rmsnorm
    except ImportError as exc:
        _BASELINE_IMPORT_ERROR = exc

elif VENDOR == "hygon":
    try:
        from lightop import op as hygon_op
    except ImportError as exc:
        _BASELINE_IMPORT_ERROR = exc

elif VENDOR == "ascend":
    try:
        from torch_npu import npu_gemma_rms_norm as npu_gemma_rms_norm
    except ImportError as exc:
        _BASELINE_IMPORT_ERROR = exc

elif VENDOR == "mthreads":
    try:
        from vllm_musa.jit_kernel.csrc import gemma_rmsnorm as mthreads_gemma_rmsnorm
    except ImportError as exc:
        _BASELINE_IMPORT_ERROR = exc

elif VENDOR == "iluvatar":
    try:
        import vllm_iluvatar

        vllm_iluvatar.register_platform()
        vllm_iluvatar.register_ops()

        from vllm_iluvatar.custom_kernels.gemma_rms_norm import (
            gemma_rms_norm as iluvatar_gemma_rms_norm,
        )
    except Exception as exc:  # noqa: BLE001 - platform registration may fail
        _BASELINE_IMPORT_ERROR = exc


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------


def _flashinfer_native_op(x, w, eps):
    """flashinfer gemma_rmsnorm baseline (NVIDIA and T-Head)."""
    return flashinfer_gemma_rmsnorm(x, w, eps)


def _hygon_native_op(x, w, eps):
    """Hygon lightop native gemma_rmsnorm baseline."""
    out = torch.empty_like(x)
    hygon_op.gemma_rmsnorm(out, x, w, eps)
    return out


def _ascend_native_op(x, w, eps):
    """Ascend torch_npu native gemma_rms_norm baseline."""
    return npu_gemma_rms_norm(x, w, eps)


def _mthreads_native_op(x, w, eps):
    """vLLM-MUSA native gemma_rmsnorm baseline."""
    out = torch.empty_like(x)
    mthreads_gemma_rmsnorm(x, w, eps, out, True)
    return out


def _iluvatar_native_op(x, w, eps):
    """Iluvatar vllm_iluvatar native gemma_rms_norm baseline."""
    out = torch.empty_like(x)
    iluvatar_gemma_rms_norm(out, x, w, eps)
    return out


def _get_baseline_op():
    """Select the baseline implementation for the current vendor.

    Returns None when the vendor has no native op or when its library is not
    installed in this environment; the caller then runs latency-only.
    """
    if VENDOR in ("nvidia", "thead"):
        return _flashinfer_native_op if flashinfer_gemma_rmsnorm is not None else None

    if VENDOR == "hygon":
        return _hygon_native_op if hygon_op is not None else None

    if VENDOR == "ascend":
        return _ascend_native_op if npu_gemma_rms_norm is not None else None

    if VENDOR == "mthreads":
        return _mthreads_native_op if mthreads_gemma_rmsnorm is not None else None

    if VENDOR == "iluvatar":
        return _iluvatar_native_op if iluvatar_gemma_rms_norm is not None else None

    return None


# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------


class GemmaRmsNormBenchmark(Benchmark):
    """Benchmark FlagGems-vllm gemma_rms_norm against the vendor-native baseline.

    Shapes are aligned to powers of two and the hidden dimensions of the
    Gemma-series models, with batch size in {1, 256, 1024}.

    NVIDIA / T-Head:
        flashinfer gemma_rmsnorm

    Ascend:
        torch_npu npu_gemma_rms_norm

    Hygon:
        lightop op.gemma_rmsnorm

    Iluvatar:
        vllm_iluvatar custom gemma_rms_norm

    MThreads:
        vLLM-MUSA jit gemma_rmsnorm

    Other vendors, or a listed vendor whose native library is not installed:
        no native baseline: only FlagGems latency is measured, and when
        --base-data is given the H800 latency scaled by _scaling_factors is
        reported as the 'vs Base' comparison instead.
    """

    # Metrics measured by running torch_op. Without a vendor-native baseline
    # they cannot be produced, so they are dropped instead of failing the run.
    BASELINE_METRICS = ("latency_base", "speedup")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.torch_op is None:
            self.metrics = [
                metric for metric in self.metrics if metric not in self.BASELINE_METRICS
            ]
            self.to_bench_metrics = self.metrics

    # Shapes aligned to powers of two and the hidden dimensions of the Gemma‑series models
    _gemma_rms_norm_ns = [128, 256, 1024, 16384] + [
        1152,
        2048,
        2560,
        3072,
        3584,
        3840,
        4608,
        5376,
    ]
    # Batch size
    _gemma_rms_norm_ms = [1, 256, 1024]
    _gemma_rms_norm_shapes = list(product(_gemma_rms_norm_ms, _gemma_rms_norm_ns))

    def set_shapes(self, shape_file_path=None):
        self.shapes = GemmaRmsNormBenchmark._gemma_rms_norm_shapes

    def get_input_iter(self, dtype):
        device = flaggems_vllm.runtime.device.name
        for shape in self.shapes:
            N = shape[-1]
            x = torch.randn(shape, dtype=dtype, device=device)
            w = torch.randn((N,), dtype=dtype, device=device)
            eps = 1e-5
            yield x, w, eps


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------


@pytest.mark.gemma_rms_norm
def test_gemma_rms_norm():
    baseline_op = _get_baseline_op()

    if baseline_op is None:
        # Second comparison mode: this vendor has no native gemma_rms_norm on
        # the machine, either because the platform has no native op or because
        # its library is not installed. FlagGems latency is still measured,
        # and passing --base-data=<h800_baseline.json> additionally reports
        # the H800 latency scaled by the bandwidth ratio as 'compared_speedup'.
        if _BASELINE_IMPORT_ERROR is not None:
            reason = (
                f"native baseline library for vendor {VENDOR!r} is not "
                f"installed ({_BASELINE_IMPORT_ERROR})"
            )
        else:
            reason = f"vendor {VENDOR!r} has no native gemma_rms_norm baseline"
        print(
            f"[gemma_rms_norm] {reason}; measuring FlagGems latency only. Pass "
            "--base-data to also report the scaled H800 comparison ('vs Base')."
        )

    elif VENDOR in ("nvidia", "thead"):
        assert callable(
            flashinfer_gemma_rmsnorm,
        ), "flashinfer gemma_rmsnorm is not available"

    elif VENDOR == "hygon":
        assert hasattr(
            hygon_op,
            "gemma_rmsnorm",
        ), "lightop op.gemma_rmsnorm is not available"

    elif VENDOR == "ascend":
        assert callable(
            npu_gemma_rms_norm,
        ), "torch_npu.npu_gemma_rms_norm is not available"

    elif VENDOR == "mthreads":
        assert callable(
            mthreads_gemma_rmsnorm,
        ), "vLLM-MUSA gemma_rmsnorm is not available"

    elif VENDOR == "iluvatar":
        assert callable(
            iluvatar_gemma_rms_norm,
        ), "Iluvatar gemma_rms_norm is not available"

    dtypes = [torch.float16]
    bench = GemmaRmsNormBenchmark(
        op_name="gemma_rms_norm",
        # Benchmark calls this field torch_op, but here it is always the
        # vendor-native baseline; there is no PyTorch fallback.
        torch_op=baseline_op,
        dtypes=dtypes,
    )
    bench.set_gems(flaggems_vllm.gemma_rms_norm)
    bench.run()
