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

"""Benchmark for top_k_top_p (Ascend sampling-filter kernel).

Production sampling profile: vocab=129280 fp32 logits at temperature 0.6,
top-p=0.95 with and without a top-k=20 pre-filter, across decode batch sizes.
The baseline is torch_npu's native npu_top_k_top_p, falling back to the
pure-PyTorch reference when that op is absent.  Each shape is additionally
verified against the baseline before it is timed (mask positions must match
to within a 1e-3 position error rate and kept values to 1e-5, matching the
original operator's acceptance thresholds).
"""

from typing import Generator

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime.backend._ascend.ops.top_k_top_p import top_k_top_p
from tests.test_top_k_top_p import top_k_top_p_ref

from . import base

torch_npu = pytest.importorskip("torch_npu")

VOCAB_SIZE = 129280
BATCHES = (1, 4, 16, 24, 50, 64, 128)
TOP_K_CASES: tuple[int | None, ...] = (20, None)
TOP_P = 0.95
TEMPERATURE = 0.6

# Native baseline when the running torch_npu exposes it.
HAS_NATIVE_OP = hasattr(torch_npu, "npu_top_k_top_p")


def _native_top_k_top_p(logits, p, k, **_):
    return torch_npu.npu_top_k_top_p(logits, k=k, p=p)


def _ref_top_k_top_p(logits, p, k, **_):
    return top_k_top_p_ref(logits, p=p, k=k)


def _gems_top_k_top_p(logits, p, k, **kw):
    return top_k_top_p(logits, p=p, k=k, **kw)


class TopKTopPBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "batch, vocab_size, top_k"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (batch, VOCAB_SIZE, top_k) for top_k in TOP_K_CASES for batch in BATCHES
        ]

    def get_input_iter(self, dtype) -> Generator:
        for i, (batch, vocab_size, top_k) in enumerate(self.shapes):
            torch.manual_seed(20260730 + batch)
            logits = torch.randn(
                (batch, vocab_size), dtype=torch.float32, device=self.device
            )
            scaled_logits = logits / TEMPERATURE
            k = (
                None
                if top_k is None
                else torch.full((batch,), top_k, dtype=torch.int32, device=self.device)
            )
            p = torch.full((batch,), TOP_P, dtype=torch.float32, device=self.device)

            yield scaled_logits, p, k, {
                "topk_len_override": top_k,
                "p_is_one_override": (TOP_P == 1.0),
            }


@pytest.mark.top_k_top_p
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="the optimized top_k_top_p targets Ascend",
)
def test_top_k_top_p_perf():
    if HAS_NATIVE_OP:
        # the replaced operator: torch_npu.npu_top_k_top_p
        torch_op = _native_top_k_top_p
        op_name = "top_k_top_p"
    else:
        # extension-free environments: torch-chain reference instead
        torch_op = _ref_top_k_top_p
        op_name = "top_k_top_p_vs_torch_ref"

    benchmark = TopKTopPBenchmark(
        op_name=op_name,
        torch_op=torch_op,
        gems_op=_gems_top_k_top_p,
        dtypes=[torch.float32],
    )
    benchmark.run()
