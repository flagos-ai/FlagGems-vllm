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

"""Kernel performance of the six validated SparseAttnSharedKV shapes.

Every case is measured against the AscendC operator on the same device with
`triton.backends.ascend.testing.do_bench_npu` at 5 warmups and 20 active samples,
which reports device kernel time (allocation and Python dispatch are excluded by
the profiler). The reported ratio is `AscendC / Triton`, so a value above 1 means
this implementation is faster.

The ratio is reported as measured and is not asserted: this operator is accepted
on accuracy, which is checked here for every case and more broadly in
`tests/test_sparse_attn_sharedkv.py`.
"""

from typing import Generator

import pytest
import torch

import flaggems_vllm
from tests.sparse_attn_sharedkv_utils import (
    CASES,
    make_inputs,
    official_operator,
    output_tensor,
)

from . import base

_IS_ASCEND = flaggems_vllm.vendor_name == "ascend" and hasattr(torch, "npu")
pytestmark = [
    pytest.mark.sparse_attn_sharedkv,
    pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU"),
]


class SparseAttnSharedKVBenchmark(base.Benchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_desc = "case"

    def set_shapes(self, shape_file_path=None):
        self.shapes = CASES

    def get_input_iter(self, dtype) -> Generator:
        for case in self.shapes:
            inputs = make_inputs(case, self.device)
            inputs["host_q_lens"] = (case.q_len,)
            inputs["host_kv_lens"] = (case.kv_len,)
            # Yield a 1-tuple: a bare dict would be iterated key-by-key by
            # unpack_to_args_kwargs instead of being merged into kwargs.
            yield (inputs,)


@torch.inference_mode()
def test_sparse_attn_sharedkv_perf():
    official = official_operator()
    if official is None:
        pytest.skip("requires the vLLM-Ascend SparseAttnSharedKV reference")

    def ascendc_op(**kwargs):
        # host_q_lens / host_kv_lens are host-length hints consumed by this
        # implementation only; the AscendC reference derives them on device.
        kwargs.pop("host_q_lens", None)
        kwargs.pop("host_kv_lens", None)
        return output_tensor(official(**kwargs))

    def gems_op(**kwargs):
        return output_tensor(flaggems_vllm.sparse_attn_sharedkv(**kwargs))

    SparseAttnSharedKVBenchmark(
        op_name="sparse_attn_sharedkv",
        torch_op=ascendc_op,
        gems_op=gems_op,
        dtypes=[torch.bfloat16],
    ).run()
