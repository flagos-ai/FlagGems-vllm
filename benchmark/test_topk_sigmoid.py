# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm

from . import base, consts


def reference(w, ids, source, logits, renormalize, bias):
    p = torch.nan_to_num(torch.sigmoid(logits.float()), nan=0.0)
    chosen = torch.argsort(p + bias, dim=-1, descending=True, stable=True)[
        :, : w.shape[1]
    ]
    selected = p.gather(1, chosen)
    if renormalize:
        selected = selected / selected.sum(-1, keepdim=True).clamp(min=1e-30)
    w.copy_(selected)
    ids.copy_(chosen)
    m, k = w.shape
    source.copy_(
        torch.arange(k, device=w.device)[None, :] * m
        + torch.arange(m, device=w.device)[:, None]
    )


class OperatorBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:]

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 128, 8),
            (4, 128, 8),
            (64, 128, 8),
            (4096, 128, 8),
            (8192, 128, 8),
        ]
        self.shape_desc = "synthetic operator dimensions"

    def get_input_iter(self, dtype):
        for m, e, k in self.shapes:
            yield (
                torch.empty((m, k), device=self.device),
                torch.empty((m, k), device=self.device, dtype=torch.int32),
                torch.empty((m, k), device=self.device, dtype=torch.int32),
                torch.randn((m, e), device=self.device, dtype=dtype),
                True,
                torch.randn((e,), device=self.device),
            )


@pytest.mark.topk_sigmoid
def test_perf_topk_sigmoid():
    bench = OperatorBenchmark(
        op_name="topk_sigmoid",
        torch_op=reference,
        gems_op=flaggems_vllm.topk_sigmoid,
        dtypes=[torch.float32],
    )
    bench.run()
