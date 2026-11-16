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

"""Kernel performance of kv_rmsnorm_rope_cache against the AscendC operator.

Every case is measured against ``torch_npu.npu_kv_rmsnorm_rope_cache`` on the
same device with ``triton.backends.ascend.testing.do_bench_npu`` at 5 warmups
and 20 active samples, which reports device kernel time. The reported ratio is
``AscendC / Triton``, so a value above 1 means this implementation is faster.

The ratio is reported as measured and is not asserted: this operator is
accepted on accuracy, which is checked here for every case and more broadly in
``tests/test_kv_rmsnorm_rope_cache.py``. Run with ``pytest -s`` to see the
numbers.
"""

import os
from typing import Generator

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

import flaggems_vllm

from . import base

RMS_SIZE = 512
ROPE_SIZE = 64
PAGE_NUM = 493
PAGE_SIZE = 128

# Same fp16 shapes as tests/test_kv_rmsnorm_rope_cache.py: decode-aligned,
# decode-unaligned and prefill cover both kernel branches.
CASES = ((16384, 1), (3, 1), (64, 256))

_IS_ASCEND = flaggems_vllm.vendor_name == "ascend" and hasattr(torch, "npu")
pytestmark = [
    pytest.mark.kv_rmsnorm_rope_cache,
    pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU"),
]


@pytest.fixture(autouse=True)
def _sequential_block_scheduling():
    # importing flaggems_vllm exports TRITON_ALL_BLOCKS_PARALLEL=1, under which
    # the kernel's tle.dsa.parallel loops produce wrong results. Force the
    # default (sequential) scheduling while this operator runs, and restore the
    # environment afterwards so other operators in the same pytest process are
    # unaffected.
    saved = os.environ.pop("TRITON_ALL_BLOCKS_PARALLEL", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = saved


class KvRmsnormRopeCacheBenchmark(base.Benchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_desc = "batch,seq"

    def set_shapes(self, shape_file_path=None):
        self.shapes = CASES

    def get_input_iter(self, dtype) -> Generator:
        for i, (batch_size, seq_len) in enumerate(self.shapes):
            torch.manual_seed(i)
            kv = torch.randn(
                batch_size, 1, seq_len, RMS_SIZE + ROPE_SIZE, dtype=dtype
            ).npu()
            gamma = torch.ones(RMS_SIZE, dtype=dtype).npu()
            cos = torch.randn(batch_size, 1, seq_len, ROPE_SIZE, dtype=dtype).npu()
            sin = torch.randn(batch_size, 1, seq_len, ROPE_SIZE, dtype=dtype).npu()
            k_cache = torch.zeros(PAGE_NUM, PAGE_SIZE, 1, ROPE_SIZE, dtype=dtype).npu()
            ckv_cache = torch.zeros(PAGE_NUM, PAGE_SIZE, 1, RMS_SIZE, dtype=dtype).npu()
            index = torch.arange(batch_size * seq_len, dtype=torch.int64).npu()
            yield kv, gamma, cos, sin, index, k_cache, ckv_cache, {
                "k_rope_scale": None,
                "c_kv_scale": None,
                "k_rope_offset": None,
                "c_kv_offset": None,
                "epsilon": 1e-5,
                "cache_mode": "PA_BNSD",
                "is_output_kv": True,
            }


def torch_kv_rmsnorm_rope_cache(
    kv, gamma, cos, sin, index, k_cache, ckv_cache, **kwargs
):
    return torch_npu.npu_kv_rmsnorm_rope_cache(
        kv, gamma, cos, sin, index, k_cache, ckv_cache, **kwargs
    )


def test_kv_rmsnorm_rope_cache_perf():
    KvRmsnormRopeCacheBenchmark(
        op_name="kv_rmsnorm_rope_cache",
        torch_op=torch_kv_rmsnorm_rope_cache,
        gems_op=flaggems_vllm.kv_rmsnorm_rope_cache,
        dtypes=[torch.float16],
    ).run()
