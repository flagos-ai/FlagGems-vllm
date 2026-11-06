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

"""Kernel performance of the FP8 SparseAttnSharedKV decode shapes.

Each case is measured against the AscendC operator on the same device with
`triton.backends.ascend.testing.do_bench_npu` at 5 warmups and 20 active
samples, which reports device kernel time. The reported ratio is
`AscendC / Triton`, so a value above 1 means this implementation is faster.
The AscendC operator reads the exact BF16 decoding of the pages this path
reads as FP8, so both attend to identical numbers; only the bytes differ.

The tiling metadata is built once per case, outside the measured region,
because a decode step builds it once and every layer of the step reuses it --
the same reason the AscendC operator takes a metadata tensor built by its own
operator. Both are built by device kernels.

The ratio is reported as measured and is not asserted: accuracy is checked
here for every case and more broadly in `tests/test_sparse_attn_sharedkv_fp8.py`.
"""

import pytest
import torch

import flaggems_vllm
from tests.test_sparse_attn_sharedkv_fp8 import (
    ATOL,
    CASES,
    RTOL,
    WINDOW,
    make_inputs,
    official_operator,
    reference_call,
    shape_label,
)
from tests.sparse_attn_sharedkv_utils import metadata_operator

WARMUP = 5
ACTIVE = 20

_IS_ASCEND = flaggems_vllm.vendor_name == "ascend" and hasattr(torch, "npu")
pytestmark = [
    pytest.mark.sparse_attn_sharedkv_fp8,
    pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU"),
]


def _kernel_time_us(fn) -> float:
    from triton.backends.ascend.testing import do_bench_npu

    return do_bench_npu(fn, warmup=WARMUP, active=ACTIVE) * 1000.0


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@torch.inference_mode()
def test_sparse_attn_sharedkv_fp8_kernel_perf(case):
    if official_operator() is None or metadata_operator() is None:
        pytest.skip("requires the vLLM-Ascend SparseAttnSharedKV reference")
    module = __import__(
        "flaggems_vllm.runtime.backend._ascend.ops.sparse_attn_sharedkv_fp8",
        fromlist=["plan"],
    )
    inputs, bf16 = make_inputs(case, flaggems_vllm.device)
    schedule = module.plan(
        case.batch,
        inputs["seqused_kv"],
        case.cmp_ratio,
        WINDOW,
        inputs["cmp_sparse_indices"],
        inputs["cmp_block_table"],
    )

    def candidate():
        return flaggems_vllm.sparse_attn_sharedkv(**inputs, schedule=schedule)[0]

    def reference():
        return reference_call(case, inputs, bf16)

    torch.testing.assert_close(
        candidate().float().cpu(), reference().float().cpu(), atol=ATOL, rtol=RTOL
    )
    reference_us = _kernel_time_us(reference)
    candidate_us = _kernel_time_us(candidate)
    print(
        f"\n{shape_label(case)}\n"
        f"  ascendc={reference_us:.3f}us triton={candidate_us:.3f}us "
        f"speedup={reference_us / candidate_us:.4f}x"
    )
