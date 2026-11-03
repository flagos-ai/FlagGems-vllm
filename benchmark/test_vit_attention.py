# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm

from . import base, consts


def reference(q, k, v, batch_size):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


class OperatorBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:]

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 128, 16, 64), (1, 512, 16, 80), (2, 256, 16, 80)]
        self.shape_desc = "synthetic operator dimensions"

    def get_input_iter(self, dtype):
        for b, s, h, d in self.shapes:
            q = torch.randn((b, s, h, d), device=self.device, dtype=dtype)
            yield (q, torch.randn_like(q), torch.randn_like(q), b)


@pytest.mark.vit_attention
def test_perf_vit_attention():
    bench = OperatorBenchmark(
        op_name="vit_attention",
        torch_op=reference,
        gems_op=flaggems_vllm.vit_flash_attn_wrapper,
        dtypes=[torch.bfloat16, torch.float16],
    )
    bench.run()
