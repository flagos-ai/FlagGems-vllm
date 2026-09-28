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

"""End-to-end benchmark: mhc_post called with natural vs transposed mix
inputs.

Baseline: a legacy caller holding the transposed layout pays
``mix.transpose(-1, -2).contiguous()`` inside the timed loop and then runs
the fast path.  Candidate: the model-native (NATURAL) input is consumed in
place -- no copy -- through the strided dispatch.

Informational only: no timing assertions.  A bitwise equivalence check
(legacy == natural) gates each shape before its timings are collected.  The
framework reports ``speedup = latency_base / latency``: > 1.0 means the
natural-input call is faster than what the transposed-input caller pays.
"""

import pytest
import torch

from flaggems_vllm.ops.mhc import MixLayout, mhc_post
from tests.test_mhc_ops import MHC_POST_LAYOUT_CONFIGS

from . import base


class MhcPostLayoutBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [tuple(s) for s in MHC_POST_LAYOUT_CONFIGS]
        self.shape_desc = "n, H, hc_mult"


def _make_inputs(shape, _dtype, device):
    # dtype intentionally unused: mhc_post's contract requires bfloat16
    # residual/x and float32 mix/post regardless of the bench dtype slot.
    n, h, hc = shape
    x = torch.randn(n, h, dtype=torch.bfloat16, device=device)
    residual = torch.randn(n, hc, h, dtype=torch.bfloat16, device=device)
    post = torch.randn(n, hc, 1, dtype=torch.float32, device=device)
    mix = torch.randn(n, hc, hc, dtype=torch.float32, device=device)
    return x, residual, post, mix


def _legacy(x, residual, post, mix):
    """Baseline: transposed-input caller -- transpose-copy every call."""
    return mhc_post(
        x,
        residual,
        post,
        mix.transpose(-1, -2).contiguous(),
        mix_layout=MixLayout.TRANSPOSED,
    )


def _strided(x, residual, post, mix):
    """Candidate: natural-input caller -- read in place, no copy."""
    return mhc_post(x, residual, post, mix, mix_layout=MixLayout.NATURAL)


def _e2e_input(shape, dtype, device):
    x, residual, post, mix = _make_inputs(shape, dtype, device)
    out_legacy = _legacy(x, residual, post, mix)
    out_native = _strided(x, residual, post, mix)
    assert torch.equal(
        out_legacy, out_native
    ), f"STOP: transposed vs natural diverge bitwise at {shape}"
    yield x, residual, post, mix


@pytest.mark.mhc_post
def test_mhc_post_layout_e2e():
    bench = MhcPostLayoutBenchmark(
        input_fn=_e2e_input,
        op_name="mhc_post_layout_e2e",
        torch_op=_legacy,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_strided)
    bench.run()
