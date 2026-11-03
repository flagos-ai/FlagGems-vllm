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

"""Accuracy tests for kv_rmsnorm_rope_cache against the AscendC reference.

The four parametrized shapes cover both kernel branches: the aligned decode
fast path, the masked (IS_ALIGNED=False) path, a seq_len > 1 prefill layout,
and BF16 inputs.
"""

import os

import pytest
import torch

try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()
except (ImportError, AttributeError):
    _NPU_AVAILABLE = False

import flaggems_vllm

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend" or not _NPU_AVAILABLE,
    reason="kv_rmsnorm_rope_cache is only implemented for available Ascend NPUs",
)

RMS_SIZE = 512
ROPE_SIZE = 64
PAGE_NUM = 493
PAGE_SIZE = 128
ATOL, RTOL = 1e-2, 1e-3


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


@pytest.mark.parametrize(
    "batch_size, seq_len, dtype",
    [
        (16384, 1, torch.float16),
        (3, 1, torch.float16),
        (64, 256, torch.float16),
        (16384, 1, torch.bfloat16),
    ],
    ids=["decode-aligned-fp16", "decode-unaligned-fp16", "prefill-fp16", "decode-bf16"],
)
def test_kv_rmsnorm_rope_cache_accuracy(batch_size, seq_len, dtype):
    kv, gamma, cos, sin, k_cache, ckv_cache, index = _make_inputs(
        batch_size, seq_len, dtype
    )
    ref_k_cache = torch.zeros_like(k_cache)
    ref_ckv_cache = torch.zeros_like(ckv_cache)

    k_cache, ckv_cache, k_rope, c_kv = flaggems_vllm.kv_rmsnorm_rope_cache(
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache,
        ckv_cache,
        k_rope_scale=None,
        c_kv_scale=None,
        k_rope_offset=None,
        c_kv_offset=None,
        epsilon=1e-5,
        cache_mode="PA_BNSD",
        is_output_kv=True,
    )

    k_cache_ref, v_cache_ref, k_rope_ref, c_kv_ref = (
        torch_npu.npu_kv_rmsnorm_rope_cache(
            kv,
            gamma,
            cos,
            sin,
            index,
            ref_k_cache,
            ref_ckv_cache,
            k_rope_scale=None,
            c_kv_scale=None,
            k_rope_offset=None,
            c_kv_offset=None,
            epsilon=1e-5,
            cache_mode="PA_BNSD",
            is_output_kv=True,
        )
    )

    torch.testing.assert_close(k_rope_ref, k_rope, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(k_cache_ref, k_cache, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(v_cache_ref, ckv_cache, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(c_kv_ref, c_kv, atol=ATOL, rtol=RTOL)
