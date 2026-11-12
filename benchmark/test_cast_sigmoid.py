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

"""cast_sigmoid benchmark.

Two baselines, selected at collection time like ``test_fused_add_rms_norm``:

* vLLM ships ``vllm.models.glm5next.common.kda._cast_sigmoid`` (v0.33.0+):
  that native kernel is the baseline, and the FlagGems op is pinned to the same
  FP32 output contract so both sides write the same number of bytes.
* older vLLM builds have no such symbol: ``torch.sigmoid`` becomes the
  baseline, which still yields a same-dtype speed-up instead of a N/A row.

Shapes (small / medium / large) come from ``benchmark/core_shapes.yaml`` through
``op_name``.
"""

import logging

import pytest
import torch

import flaggems_vllm
from baseline import cast_sigmoid as workloads
from tests.accuracy_utils import gems_assert_close

from . import base, conftest, consts, utils

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------


def _torch_reference_op(x):
    """Fallback baseline: plain same-dtype sigmoid."""
    return torch.sigmoid(x)


def _vllm_native_op(x):
    """vLLM native ``_cast_sigmoid``: fused FP32 cast + sigmoid, FP32 out."""
    native = workloads.native()
    assert native is not None, workloads.NATIVE_UNAVAILABLE_REASON
    return native(x)


def _gems_same_dtype_op(x):
    """FlagGems op paired with the torch baseline (input dtype out)."""
    return flaggems_vllm.cast_sigmoid(x)


def _gems_native_contract_op(x):
    """FlagGems op paired with the native baseline (FP32 out)."""
    return flaggems_vllm.cast_sigmoid(x, out_dtype=workloads.NATIVE_OUT_DTYPE)


def _select_ops():
    """Pick the (baseline, gem) pair for the installed vLLM."""
    if workloads.native() is None:
        LOGGER.info(
            "cast_sigmoid: no vLLM native baseline (%s); "
            "falling back to torch.sigmoid",
            workloads.NATIVE_UNAVAILABLE_REASON,
        )
        return _torch_reference_op, _gems_same_dtype_op
    LOGGER.info(
        "cast_sigmoid: vLLM native baseline %s",
        workloads.NATIVE_ARTIFACT.get("resolved_api", workloads.NATIVE_NAME),
    )
    return _vllm_native_op, _gems_native_contract_op


# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------


class CastSigmoidBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # core_shapes.yaml already spans small/medium/large, and the generic
        # 2**28 shapes only stress the allocator for a bandwidth-bound kernel.
        return []

    def get_input_iter(self, dtype):
        out_dtype = (
            workloads.NATIVE_OUT_DTYPE if self.torch_op is _vllm_native_op else dtype
        )
        verified = False
        for inputs in super().get_input_iter(dtype):
            if not verified:
                # Guard against a silently broken kernel benchmarking fast.
                gems_assert_close(
                    self.gems_op(*inputs), self.torch_op(*inputs), out_dtype
                )
                verified = True
            yield inputs


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------


@pytest.mark.cast_sigmoid
def test_cast_sigmoid():
    if conftest.Config.mode != consts.BenchMode.KERNEL:
        raise ValueError("cast_sigmoid benchmark requires --mode kernel (no Graph)")

    baseline_op, gems_op = _select_ops()

    bench = CastSigmoidBenchmark(
        input_fn=utils.unary_input_fn,
        op_name=workloads.OP_NAME,
        # GenericBenchmark calls this field torch_op; here it carries whichever
        # baseline the installed vLLM allows (native kernel or torch.sigmoid).
        torch_op=baseline_op,
        gems_op=gems_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
