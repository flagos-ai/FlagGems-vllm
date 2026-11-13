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

import pytest
import torch

import flaggems_vllm

from . import base

VENDOR = flaggems_vllm.vendor_name


# -----------------------------------------------------------------------------
# Vendor-specific baseline import
# -----------------------------------------------------------------------------

# A platform build may ship FlagGems without the vendor-native vocabulary
# package. A failed import leaves the handle None and degrades the run to
# latency-only mode instead of failing collection.
vendor_ops = None
torch_npu = None
vllm_fused_q_kv_rmsnorm = None
_BASELINE_IMPORT_ERROR = None

if VENDOR == "mthreads":
    try:
        from vllm_musa import _custom_ops as vendor_ops
    except ImportError as exc:
        _BASELINE_IMPORT_ERROR = exc

elif VENDOR == "ascend":
    try:
        import torch_npu
    except ImportError as exc:
        _BASELINE_IMPORT_ERROR = exc

elif VENDOR in ("iluvatar", "thead", "nvidia", "hygon"):
    try:
        from vllm.models.deepseek_v4.common.ops import (
            fused_q_kv_rmsnorm as vllm_fused_q_kv_rmsnorm,
        )
    except ImportError as exc:
        _BASELINE_IMPORT_ERROR = exc


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------


def _mthreads_native_op(qr, kv, q_weight, kv_weight, eps):
    """vLLM-MUSA native fused_q_kv_rmsnorm baseline."""
    return vendor_ops.deepseek_v4_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)


def _ascend_native_op(qr, kv, q_weight, kv_weight, eps):
    """Ascend npu_rms_norm baseline used as the fused q/kv norm.

    Same math as the CUDA kernel: RMS in fp32, one cast at store. Mirrors
    vllm-ascend patch_triton.py:
    https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/patch/worker/patch_triton.py#L112
    """
    q_out, _ = torch_npu.npu_rms_norm(qr, q_weight, eps)
    kv_out, _ = torch_npu.npu_rms_norm(kv, kv_weight, eps)
    return q_out, kv_out


def _get_baseline_op():
    """Select the baseline implementation for the current vendor.

    Returns None when the vendor has no native op or when its library is not
    installed in this environment; the caller then runs latency-only.
    """
    if VENDOR in ("nvidia", "thead", "iluvatar", "hygon"):
        return vllm_fused_q_kv_rmsnorm if callable(vllm_fused_q_kv_rmsnorm) else None

    if VENDOR == "ascend":
        return _ascend_native_op if torch_npu is not None else None

    if VENDOR == "mthreads":
        return _mthreads_native_op if vendor_ops is not None else None

    return None


# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------


class FusedQKVRMSNormBenchmark(base.Benchmark):
    """Benchmark FlagGems-vllm fused_q_kv_rmsnorm against the vendor-native
    baseline.

    Shapes are the DeepSeek-V4 attention workload: (num_tokens, q_size,
    kv_size), covering both the MLA q/kv layout and the NSA sparse layout.

    NVIDIA / T-Head / Iluvatar / Hygon:
        vLLM native fused_q_kv_rmsnorm

    Ascend:
        torch_npu npu_rms_norm on q and kv separately

    MThreads:
        vLLM-MUSA deepseek_v4_fused_q_kv_rmsnorm

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

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [
            (1, 1536, 512),
            (32, 1536, 512),
            (128, 1536, 512),
            (512, 1536, 512),
            (2048, 1536, 512),
            (32, 64 * 576, 576),
            (128, 64 * 576, 576),
        ]

    def get_input_iter(self, dtype):
        device = flaggems_vllm.runtime.device.name
        for tokens, qdim, kvdim in self.shapes:
            qr = torch.randn(
                (tokens, qdim),
                device=device,
                dtype=dtype,
            )
            kv = torch.randn(
                (tokens, kvdim),
                device=device,
                dtype=dtype,
            )
            q_weight = torch.randn(
                (qdim,),
                device=device,
                dtype=dtype,
            )
            kv_weight = torch.randn(
                (kvdim,),
                device=device,
                dtype=dtype,
            )

            yield (
                qr,
                kv,
                q_weight,
                kv_weight,
                1e-6,
            )


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------


@pytest.mark.fused_q_kv_rmsnorm
def test_fused_q_kv_rmsnorm_benchmark():
    baseline_op = _get_baseline_op()

    if baseline_op is None:
        # Second comparison mode: this vendor has no native fused_q_kv_rmsnorm
        # on the machine, either because the platform has no native op or
        # because its library is not installed. FlagGems latency is still
        # measured, and passing --base-data=<h800_baseline.json> additionally
        # reports the H800 latency scaled by the bandwidth ratio as
        # 'compared_speedup'.
        if _BASELINE_IMPORT_ERROR is not None:
            reason = (
                f"native baseline library for vendor {VENDOR!r} is not "
                f"installed ({_BASELINE_IMPORT_ERROR})"
            )
        else:
            reason = f"vendor {VENDOR!r} has no native fused_q_kv_rmsnorm baseline"
        print(
            f"[fused_q_kv_rmsnorm] {reason}; measuring FlagGems latency only. "
            "Pass --base-data to also report the scaled H800 comparison "
            "('vs Base')."
        )

    elif VENDOR in ("nvidia", "thead", "iluvatar", "hygon"):
        assert callable(
            vllm_fused_q_kv_rmsnorm,
        ), "vLLM native fused_q_kv_rmsnorm is not available"

    elif VENDOR == "ascend":
        assert hasattr(
            torch_npu,
            "npu_rms_norm",
        ), "torch_npu.npu_rms_norm is not available"

    elif VENDOR == "mthreads":
        assert hasattr(
            vendor_ops,
            "deepseek_v4_fused_q_kv_rmsnorm",
        ), "vLLM-MUSA deepseek_v4_fused_q_kv_rmsnorm is not available"

    bench = FusedQKVRMSNormBenchmark(
        "fused_q_kv_rmsnorm",
        # Benchmark calls this field torch_op, but here it is always the
        # vendor-native baseline; there is no PyTorch fallback.
        baseline_op,
        [torch.bfloat16],
        # Use the top-level API so vendor-specific backend
        # overrides are respected.
        gems_op=flaggems_vllm.fused_q_kv_rmsnorm,
    )
    bench.run()
