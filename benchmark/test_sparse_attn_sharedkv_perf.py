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

import pytest
import torch

import flaggems_vllm
from tests.sparse_attn_sharedkv_utils import (
    CASES,
    make_inputs,
    official_operator,
    output_tensor,
    shape_label,
)

WARMUP = 5
ACTIVE = 20

_IS_ASCEND = flaggems_vllm.vendor_name == "ascend" and hasattr(torch, "npu")
pytestmark = [
    pytest.mark.sparse_attn_sharedkv,
    pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU"),
]


def _kernel_time_us(fn) -> float:
    from triton.backends.ascend.testing import do_bench_npu

    return do_bench_npu(fn, warmup=WARMUP, active=ACTIVE) * 1000.0


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@torch.inference_mode()
def test_sparse_attn_sharedkv_kernel_perf(case):
    official = official_operator()
    if official is None:
        pytest.skip("requires the vLLM-Ascend SparseAttnSharedKV reference")
    inputs = make_inputs(case, flaggems_vllm.device)

    def candidate():
        return flaggems_vllm.sparse_attn_sharedkv(
            **inputs,
            host_q_lens=(case.q_len,),
            host_kv_lens=(case.kv_len,),
        )

    def reference():
        return output_tensor(official(**inputs))

    torch.testing.assert_close(
        output_tensor(candidate()).float().cpu(),
        reference().float().cpu(),
        atol=2e-2,
        rtol=2e-2,
    )
    reference_us = _kernel_time_us(reference)
    candidate_us = _kernel_time_us(candidate)
    print(
        f"\n{shape_label(case)}\n"
        f"  ascendc={reference_us:.3f}us triton={candidate_us:.3f}us "
        f"speedup={reference_us / candidate_us:.4f}x"
    )
