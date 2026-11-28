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

"""Benchmark for the A5 MX-FP8 grouped matmul.

gems op: ``flaggems_vllm.grouped_matmul_mxfp8`` (A5 registrar entry).
vendor baseline: ``torch_npu.npu_grouped_matmul`` with e8m0 scales, called the
way vllm-ascend's ``apply_gmm1`` calls it (``w8a8_mxfp8.py:585``).
SpeedUp = latency_vendor / latency_flaggems (repo convention).

  ASCEND_RT_VISIBLE_DEVICES=7 pytest -q benchmark/test_grouped_matmul_mxfp8.py

``--mode kernel`` is the default and the only mode whose SpeedUp describes this
operator: device kernel time, summed over every kernel one call launches.  Host
launch overhead cannot be optimized and is gone once the model runs as a
captured graph, so the host-timed modes measure something deployment never
pays; their ~120us of Triton launch overhead buries the 14us Qwen3.6 kernel.
They still run through the common runner, but read the kernel-mode rows.

The shared ``do_bench_npu`` cannot produce that number.  It needs a constant
kernel count per iteration and an exact multiple of them in the trace, while
one call here is the MX-MAC kernel plus a ZerosLike on the output buffer, and
the vendor operator has its own set; it saw 33 and 138 rows against the 35 it
expected, rejected the trace and returned ``inf`` for both paths.

Every deployment shape is measured twice, with an ND weight and with the NZ
weight of a real deployment; ``weight_nz`` in the recorded shape tells the two
rows apart.  ``benchmark/bench_grouped_matmul_mxfp8.py`` stays alongside this
file for the per-kernel breakdown, which this runner does not report.
"""

import os

# TMPDIR must point to /tmp before importing torch_npu, or the NPU profiler
# cannot write kernel_details.csv and the device time silently comes back None.
if not os.environ.get("TMPDIR", "").startswith("/tmp"):
    os.environ["TMPDIR"] = "/tmp"

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("torch_npu")

import flaggems_vllm  # noqa: E402
from tests.test_grouped_matmul_mxfp8 import build_inputs, to_nz, vendor  # noqa: E402

from . import base  # noqa: E402
from .bench_grouped_matmul_mxfp8 import device_time_us  # noqa: E402
from .conftest import Config  # noqa: E402
from .consts import BenchMode  # noqa: E402

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend"
    or not torch.npu.is_available()
    or not hasattr(flaggems_vllm, "grouped_matmul_mxfp8"),
    reason="grouped_matmul_mxfp8 is registered on Ascend 950 (A5) only",
)

# (M, K, N, E) as the operator receives them, i.e. after the weight transpose
# in process_weights_after_loading.  The comment is the operator's share of
# model time measured on the deployment profile.
SHAPES = [
    (64, 3072, 6144, 16),  # MiniMax M3.1, 21.59%
    (64, 2048, 6144, 32),  # GLM W8A8C8, 3.37%
    (16, 512, 2048, 128),  # Qwen3.6, 4.19%
]


class GroupedMatmulMxfp8Benchmark(base.GenericBenchmark):
    """Use the common benchmark runner with the fixed deployment shapes."""

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = SHAPES
        self.shape_desc = "M, K, N, E"

    def get_latency(self, op, *args, **kwargs):
        """Device kernel time per call, summed over the kernels of one call.

        Device time is the metric: host launch overhead is not optimizable and
        does not exist once the model runs as a captured graph.  ``do_bench_npu``
        cannot report it here because a call launches more than one kernel and
        the trace row count is not an exact multiple of the iteration count,
        which it answers with ``inf``.  Summing the profiler's kernel durations
        also keeps this operator's output zero-init kernel and the vendor
        operator's own extra kernels inside both numbers.
        """
        if Config.mode != BenchMode.KERNEL:
            return super().get_latency(op, *args, **kwargs)
        # The common runner's warmup/repetition are durations in other modes;
        # here they bound the iteration counts, as on the other NPU benchmarks.
        warmup = min(int(Config.warm_up), 5)
        iters = min(int(Config.repetition), 30)
        total_us, _ = device_time_us(
            lambda: op(*args, **kwargs), warmup=warmup, iters=iters
        )
        if total_us is None:
            raise RuntimeError("the NPU profiler produced no kernel_details.csv")
        return total_us / 1e3


def _grouped_matmul_mxfp8_vendor(
    x,
    weight,
    weight_scale,
    per_token_scale,
    group_list,
    *,
    group_list_type=1,
    weight_nz=False,
):
    """Vendor baseline. It reads the NZ format from the tensor, not the hint."""
    _ = weight_nz
    return vendor(
        x,
        weight,
        weight_scale,
        per_token_scale,
        group_list,
        group_list_type=group_list_type,
    )


def _grouped_matmul_mxfp8_gems(
    x,
    weight,
    weight_scale,
    per_token_scale,
    group_list,
    *,
    group_list_type=1,
    weight_nz=False,
):
    """Production defaults: variant=2 (hardware MX-MAC), BM=16, BN=1024, BK=64.

    ``weight_nz`` is passed explicitly, matching the format the input builder
    produced, so the per-call ACL format query stays out of the timed region.
    """
    return flaggems_vllm.grouped_matmul_mxfp8(
        x,
        weight,
        weight_scale,
        per_token_scale,
        group_list,
        group_list_type,
        weight_nz=weight_nz,
    )


def _grouped_matmul_mxfp8_input_fn(shape, dtype, device):
    """Yield the ND and the NZ workload for one deployment shape.

    ``build_inputs`` quantises through ``npu_dynamic_mx_quant``, so the scale
    exponents keep the distribution of real inference.  ``dtype`` is the BF16
    output of the production path; both paths produce it natively.
    """
    _ = device
    if dtype != torch.bfloat16:
        raise ValueError(f"the MX-FP8 grouped matmul outputs bfloat16, not {dtype}")
    M, K, N, E = shape
    x, weight, weight_scale, per_token_scale, group_list = build_inputs(M, K, N, E)
    for w, weight_nz in ((weight, False), (to_nz(weight), True)):
        args = (x, w, weight_scale, per_token_scale, group_list)
        kwargs = {"group_list_type": 1, "weight_nz": weight_nz}
        # Both paths must reach the same MX-MAC instruction: a row that is not
        # bitwise equal to the vendor is not timing the same computation.
        ours = _grouped_matmul_mxfp8_gems(*args, **kwargs)
        expected = _grouped_matmul_mxfp8_vendor(*args, **kwargs)
        bad = int((ours != expected).sum().item())
        del ours, expected
        assert bad == 0, (
            f"[M,K]=[{M},{K}] [E,K,N]=[{E},{K},{N}] weight_nz={weight_nz}: "
            f"{bad} elements differ from the vendor operator"
        )
        yield (*args, kwargs)


@pytest.mark.grouped_matmul_mxfp8
def test_grouped_matmul_mxfp8_perf():
    bench = GroupedMatmulMxfp8Benchmark(
        input_fn=_grouped_matmul_mxfp8_input_fn,
        op_name="grouped_matmul_mxfp8",
        torch_op=_grouped_matmul_mxfp8_vendor,
        gems_op=_grouped_matmul_mxfp8_gems,
        dtypes=[torch.bfloat16],
    )
    bench.run()
