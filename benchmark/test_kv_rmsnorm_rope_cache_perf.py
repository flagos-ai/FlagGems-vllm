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

import pytest
import torch
import torch_npu

import flaggems_vllm

RMS_SIZE = 512
ROPE_SIZE = 64
PAGE_NUM = 493
PAGE_SIZE = 128
ATOL, RTOL = 1e-2, 1e-3
WARMUP = 5
ACTIVE = 20

CASES = (("decode-fp16", 16384, 1, torch.float16),)

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


def _kernel_time_us(fn) -> float:
    from triton.backends.ascend.testing import do_bench_npu

    return do_bench_npu(fn, warmup=WARMUP, active=ACTIVE) * 1000.0


def _make_inputs(batch_size, seq_len, dtype):
    torch.manual_seed(0)
    kv = torch.randn(batch_size, 1, seq_len, RMS_SIZE + ROPE_SIZE, dtype=dtype).npu()
    gamma = torch.ones(RMS_SIZE, dtype=dtype).npu()
    cos = torch.randn(batch_size, 1, seq_len, ROPE_SIZE, dtype=dtype).npu()
    sin = torch.randn(batch_size, 1, seq_len, ROPE_SIZE, dtype=dtype).npu()
    k_cache = torch.zeros(PAGE_NUM, PAGE_SIZE, 1, ROPE_SIZE, dtype=dtype).npu()
    ckv_cache = torch.zeros(PAGE_NUM, PAGE_SIZE, 1, RMS_SIZE, dtype=dtype).npu()
    index = torch.arange(batch_size * seq_len, dtype=torch.int64).npu()
    return kv, gamma, cos, sin, k_cache, ckv_cache, index


_KWARGS = dict(
    k_rope_scale=None,
    c_kv_scale=None,
    k_rope_offset=None,
    c_kv_offset=None,
    epsilon=1e-5,
    cache_mode="PA_BNSD",
    is_output_kv=True,
)


@pytest.mark.parametrize(
    "name, batch_size, seq_len, dtype",
    CASES,
    ids=lambda v: v if isinstance(v, str) else "",
)
@torch.inference_mode()
def test_kv_rmsnorm_rope_cache_kernel_perf(name, batch_size, seq_len, dtype):
    kv, gamma, cos, sin, k_cache, ckv_cache, index = _make_inputs(
        batch_size, seq_len, dtype
    )
    ref_k_cache = torch.zeros_like(k_cache)
    ref_ckv_cache = torch.zeros_like(ckv_cache)

    def candidate():
        return flaggems_vllm.kv_rmsnorm_rope_cache(
            kv, gamma, cos, sin, index, k_cache, ckv_cache, **_KWARGS
        )

    def reference():
        return torch_npu.npu_kv_rmsnorm_rope_cache(
            kv,
            gamma,
            cos,
            sin,
            index,
            ref_k_cache,
            ref_ckv_cache,
            **_KWARGS,
        )

    k_cache_r, ckv_r, k_rope, c_kv = candidate()
    k_cache_ref, v_cache_ref, k_rope_ref, c_kv_ref = reference()
    for expected, actual in (
        (k_rope_ref, k_rope),
        (k_cache_ref, k_cache_r),
        (v_cache_ref, ckv_r),
        (c_kv_ref, c_kv),
    ):
        torch.testing.assert_close(expected, actual, atol=ATOL, rtol=RTOL)

    reference_us = _kernel_time_us(reference)
    candidate_us = _kernel_time_us(candidate)
    print(
        f"\nkv_rmsnorm_rope_cache[{name}] B={batch_size} S={seq_len} {str(dtype).split('.')[-1]}\n"
        f"  ascendc={reference_us:.3f}us triton={candidate_us:.3f}us "
        f"speedup={reference_us / candidate_us:.4f}x"
    )
