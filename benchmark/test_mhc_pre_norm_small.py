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

"""Native GLM5Next HcPre+RMSNorm vs three-kernel decode path."""
import pytest
import torch

import flaggems_vllm as gems

from . import base


class SmallBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1,), (2,), (3,), (4,)]
        self.shape_desc = "tokens; HC=4,hidden=4096"


def native(x, fn, scale, bias, gamma):
    value, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
        x, fn, scale, bias, 4, 20, 1e-5, 1e-6
    )
    return post.unsqueeze(-1), comb, torch.ops.npu.npu_rms_norm(value, gamma, 1e-5)[0]


def inputs(shape, dtype, device):
    n = shape[0]
    args = (
        torch.randn(n, 4, 4096, device=device, dtype=dtype),
        torch.randn(24, 16384, device=device) * 0.005,
        torch.tensor([0.1, 0.3, 0.2], device=device),
        torch.randn(24, device=device) * 0.1,
        torch.ones(4096, device=device, dtype=dtype),
    )
    for actual, expected in zip(gems.mhc_pre_norm_small(*args), native(*args)):
        torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.016)
    yield args


@pytest.mark.mhc_pre_norm_small
@pytest.mark.skipif(gems.vendor_name != "ascend", reason="Ascend only")
def test_mhc_pre_norm_small():
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op() or not hasattr(torch.ops._C_ascend, "npu_hc_pre_v2"):
        pytest.skip("native GLM5Next extension unavailable")
    bench = SmallBenchmark(
        input_fn=inputs,
        op_name="mhc_pre_norm_small",
        torch_op=native,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(gems.mhc_pre_norm_small)
    bench.run()
