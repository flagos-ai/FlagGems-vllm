# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm
from tests.test_topk_sample_from_logits import reference

from . import base


def baseline(logits, k):
    return reference(logits, k, torch.empty_like(logits).exponential_())


def candidate(logits, k):
    noise = torch.empty_like(logits).exponential_()
    return flaggems_vllm.topk_sample_from_logits(logits, k, noise)


class TopKSamplingBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "batch, vocab, k"

    def __init__(self):
        super().__init__(
            "topk_sample_from_logits", baseline, dtypes=[torch.float32], gems_op=candidate
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 17, 1), (1, 6761, 25), (16, 6761, 25), (4, 8192, 33)]

    def get_input_iter(self, dtype):
        for batch, vocab, top_k in self.shapes:
            logits = torch.randn((batch, vocab), dtype=dtype, device=self.device)
            k = torch.full((batch,), top_k, dtype=torch.int32, device=self.device)
            yield logits, k


@pytest.mark.topk_sample_from_logits
@pytest.mark.skipif(flaggems_vllm.device != "musa", reason="MUSA-specific fused sampler")
def test_topk_sample_from_logits():
    TopKSamplingBenchmark().run()
