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

"""Clone-inclusive comparisons against vLLM's PyTorch logits filter."""

import pytest
import torch

import flaggems_vllm
from baseline.apply_top_k_top_p import apply_top_k_top_p as reference

from . import base


def baseline(logits, k, p):
    return reference(logits.clone(), k, p)


def candidate(logits, k, p):
    return flaggems_vllm.top_k_top_p(logits, p=p, k=k)


class TopKTopPMetaXBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "batch, vocab, bf16_grid"

    def __init__(self, bf16_grid):
        self.bf16_grid = bf16_grid
        super().__init__(
            "top_k_top_p", baseline, dtypes=[torch.float32], gems_op=candidate
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (m, n, self.bf16_grid)
            for m, n in [
                (1, 4096),
                (1, 151936),
                (8, 151936),
                (31, 131072),
                (32, 131071),
                (32, 131072),
                (32, 151936),
                (64, 4096),
                (64, 151936),
                (64, 262144),
            ]
        ]

    def record_shapes(self, *args, **kwargs):
        return (tuple(args[0].shape), self.bf16_grid)

    def get_input_iter(self, dtype):
        generator = torch.Generator().manual_seed(1234)
        for m, n, _ in self.shapes:
            shape = (m, n)
            logits = torch.randn(shape, generator=generator)
            if self.bf16_grid:
                logits = logits.bfloat16().float()
            logits = logits.to(device=self.device, dtype=dtype)
            k = torch.full((shape[0],), 20, device=self.device, dtype=torch.int32)
            p = torch.full((shape[0],), 0.8, device=self.device)
            torch.testing.assert_close(
                candidate(logits, k, p), baseline(logits, k, p), rtol=0, atol=0
            )
            yield logits, k, p


@pytest.mark.top_k_top_p
@pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX implementation")
@pytest.mark.parametrize("bf16_grid", [True, False], ids=["bf16-grid", "fp32"])
def test_top_k_top_p_metax(bf16_grid):
    TopKTopPMetaXBenchmark(bf16_grid).run()
