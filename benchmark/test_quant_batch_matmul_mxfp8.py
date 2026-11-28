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

"""Benchmark for the A5 MX-FP8 quant matmul (QuantBatchMatmulV3).

gems op: ``flaggems_vllm.quant_batch_matmul_mxfp8`` (A5 registrar entry).
vendor baseline: ``torch_npu.npu_quant_matmul`` with e8m0 scales, called the way
vllm-ascend's ``AscendW8A8MXFP8LinearMethod.apply`` calls it
(``w8a8_mxfp8.py:143``).  SpeedUp = latency_vendor / latency_flaggems (repo
convention).

  ASCEND_RT_VISIBLE_DEVICES=7 pytest -q benchmark/test_quant_batch_matmul_mxfp8.py

``--mode kernel`` is the default and the only mode whose SpeedUp describes this
operator: device time, summed over every kernel one call launches.  Host launch
overhead cannot be optimized and is gone once the model runs as a captured
graph, so the host-timed modes measure something deployment never pays, and
here they measure it twice over: this side reaches the kernel through Python
(shape checks, format query, tiling choice) while the vendor baseline is a
single C++ dispatch.  Those modes still run through the common runner, but read
the kernel-mode rows.

The shared ``do_bench_npu`` cannot produce that number.  It needs a constant
kernel count per iteration and an exact multiple of them in the trace, while a
split-K call here is the MX-MAC kernel plus the combine epilogue and the vendor
operator has its own set; it rejects such a trace and returns ``inf``.

Device time comes from the profiler's ``task_time.csv``, not
``kernel_details.csv``: the latter excludes memcpy, so any host-side input
rearrangement on this side would go uncounted while the vendor's single
dispatch has no memcpy to lose.  ``device_time_us`` in
``benchmark/bench_quant_batch_matmul_mxfp8.py`` holds that reader, and that
script stays alongside this file for the per-kernel breakdown, the tiling
report and the share-weighted summary, none of which this runner prints.

Every shape is measured in the weight layout the deployment profile recorded
for it, so the rows are directly comparable to that summary; ``weight_nz`` in
the recorded kwargs says which layout a row used.
"""

import os

# TMPDIR must point to /tmp before importing torch_npu, or the NPU profiler
# cannot write task_time.csv and the device time silently comes back None.
if not os.environ.get("TMPDIR", "").startswith("/tmp"):
    os.environ["TMPDIR"] = "/tmp"

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("torch_npu")

import flaggems_vllm  # noqa: E402
from tests.test_quant_batch_matmul_mxfp8 import (  # noqa: E402
    TOL,
    build_inputs,
    max_rel,
    vendor,
)

from . import base  # noqa: E402
from .bench_quant_batch_matmul_mxfp8 import device_time_us  # noqa: E402
from .conftest import Config  # noqa: E402
from .consts import BenchMode  # noqa: E402

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend"
    or not torch.npu.is_available()
    or not hasattr(flaggems_vllm, "quant_batch_matmul_mxfp8"),
    reason="quant_batch_matmul_mxfp8 is registered on Ascend 950 (A5) only",
)

# (M, K, N, weight_nz) as the operator receives them, i.e. K and N after the
# weight transpose in process_weights_after_loading, and the weight layout the
# model was profiled with.  The comment is the operator's share of model time;
# it ranks in the top three of all five models.  GLM W4 and W8 share one shape,
# so the row below carries both shares (12.41% + 11.54%).
SHAPES = [
    (1, 1024, 32768, 0),  # DeepSeek V4, 12.01%
    (1, 2048, 16384, 1),  # GLM W4 + W8, 23.95%
    (8, 2048, 6144, 1),  # MiniMax M3.1, 9.49%
    (1, 2048, 512, 0),  # Qwen3.6, 8.63%
]

# DSV4's profiled weight is [N,K] with an [N,K/64,2] scale while this operator
# takes [K,N]; its row is built as [K,N], which is the same arithmetic over the
# same bytes but not that call's byte layout, so read it as a same-scale
# reference rather than a drop-in replacement.


class QuantBatchMatmulMxfp8Benchmark(base.GenericBenchmark):
    """Use the common benchmark runner with the fixed deployment shapes."""

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = SHAPES
        self.shape_desc = "M, K, N, weight_nz"

    def get_latency(self, op, *args, **kwargs):
        """Device time per call, summed over the kernels of one call.

        Device time is the metric: host launch overhead is not optimizable and
        does not exist once the model runs as a captured graph.
        ``do_bench_npu`` cannot report it here because a split-K call launches
        the MX-MAC kernel and the combine epilogue, which leaves the trace row
        count short of an exact multiple of the iteration count and makes it
        answer ``inf``.  Summing the profiler's per-entry device time keeps
        that epilogue, and the vendor operator's own extra kernels, inside both
        numbers.
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
            raise RuntimeError("the NPU profiler produced no task_time.csv")
        return total_us / 1e3


def _quant_batch_matmul_mxfp8_vendor(
    x, weight, weight_scale, pertoken_scale, *, weight_nz=False, output_dtype=None
):
    """Vendor baseline. It reads the NZ format from the tensor, not the hint.

    The argument order is this operator's, not ``npu_quant_matmul``'s, so both
    sides can be called with one argument list.
    """
    _ = weight_nz
    return vendor(
        x,
        pertoken_scale,
        weight,
        weight_scale,
        output_dtype=output_dtype or torch.bfloat16,
    )


def _quant_batch_matmul_mxfp8_gems(
    x, weight, weight_scale, pertoken_scale, *, weight_nz=False, output_dtype=None
):
    """Production defaults: BLOCK_M=16, BLOCK_N/SPLIT_K from ``_pick_tiling``,
    BLOCK_K=64, and B double-buffering wherever the UB budget allows it.

    ``weight_nz`` is passed explicitly, matching the layout the input builder
    produced, so the per-call ACL format query stays out of the timed region.
    """
    return flaggems_vllm.quant_batch_matmul_mxfp8(
        x,
        weight,
        weight_scale,
        pertoken_scale,
        output_dtype=output_dtype or torch.bfloat16,
        weight_nz=weight_nz,
    )


def _quant_batch_matmul_mxfp8_input_fn(shape, dtype, device):
    """Yield one deployment shape in its profiled weight layout.

    ``build_inputs`` quantises through ``npu_dynamic_mx_quant``, so the scale
    exponents keep the distribution of real inference.  ``dtype`` is the output
    precision of the production path; both sides produce it natively.
    """
    _ = device
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"the MX-FP8 quant matmul outputs bf16/fp16, not {dtype}")
    M, K, N, nz = shape
    weight_nz = bool(nz)
    x, sa, w, _, sb = build_inputs(M, K, N, nz=weight_nz)
    args = (x, w, sb, sa)
    kwargs = {"weight_nz": weight_nz, "output_dtype": dtype}

    # Both sides must reach the same MX-MAC instruction: a row that does not
    # match the vendor is not timing the same computation.  Equality is bitwise
    # whenever split-K is off; the split-K shapes reorder the K accumulation,
    # so they are held to the output tolerance instead.
    ours = _quant_batch_matmul_mxfp8_gems(*args, **kwargs)
    expected = _quant_batch_matmul_mxfp8_vendor(*args, **kwargs)
    rel = max_rel(ours, expected)
    del ours, expected
    assert rel <= TOL, (
        f"[M,K]=[{M},{K}] [K,N]=[{K},{N}] weight_nz={weight_nz}: "
        f"max_rel={rel:.3e} against the vendor operator exceeds {TOL:g}"
    )
    yield (*args, kwargs)


@pytest.mark.quant_batch_matmul_mxfp8
def test_quant_batch_matmul_mxfp8_perf():
    bench = QuantBatchMatmulMxfp8Benchmark(
        input_fn=_quant_batch_matmul_mxfp8_input_fn,
        op_name="quant_batch_matmul_mxfp8",
        torch_op=_quant_batch_matmul_mxfp8_vendor,
        gems_op=_quant_batch_matmul_mxfp8_gems,
        dtypes=[torch.bfloat16],
    )
    bench.run()
