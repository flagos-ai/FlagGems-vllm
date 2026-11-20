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

"""Native prepared H vs resident Triton H; milliseconds, baseline/candidate."""
import pytest
import torch

import flaggems_vllm as gems

from . import base


class HBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(4096, 16), (8192, 16), (16384, 32), (8193, 16)]
        self.shape_desc = "tokens,heads"


def native(k, w, u, **kw):
    from fla_npu.ops.ascendc import npu_chunk_fwd_h

    return npu_chunk_fwd_h(
        k,
        w,
        u,
        **kw,
        output_final_state=True,
        chunk_size=64,
        save_new_value=True,
        use_exp2=True,
        state_v_first=True,
    )


def inputs(shape, dtype, device):
    t, h = shape
    k = torch.randn(1, h, t, 128, device=device, dtype=dtype) * 0.04
    w = torch.randn_like(k) * 0.01
    u = torch.randn_like(k) * 0.3
    kw = dict(
        gk=-torch.rand(k.shape, device=device) * 0.2,
        initial_state=torch.randn(1, h, 128, 128, device=device) * 0.1,
        cu_seqlens=[0, t],
    )
    for actual, expected in zip(
        gems.chunk_kda_fwd_h(k, w, u, **kw), native(k, w, u, **kw)
    ):
        torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.002)
    yield k, w, u, kw


@pytest.mark.chunk_kda_fwd_h
@pytest.mark.skipif(gems.vendor_name != "ascend", reason="Ascend only")
def test_chunk_kda_fwd_h():
    pytest.importorskip("fla_npu")
    bench = HBenchmark(
        input_fn=inputs,
        op_name="chunk_kda_fwd_h",
        torch_op=native,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(gems.chunk_kda_fwd_h)
    bench.run()
