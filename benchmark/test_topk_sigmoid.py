# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
from functools import partial

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


def padded_topk_reference(
    weights,
    ids,
    source,
    logits,
    renormalize,
    e_score_correction_bias,
    is_padding,
    routed_scaling_factor=1.0,
    *,
    sigmoid,
):
    scores = logits.float()
    scores = scores.sigmoid() if sigmoid else scores.softmax(dim=-1)
    scores = scores.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    selection = scores + e_score_correction_bias
    chosen = selection.argsort(dim=-1, descending=True, stable=True)
    chosen = chosen[:, : weights.shape[1]]
    selected = scores.gather(1, chosen)
    if renormalize:
        denominator = selected.sum(dim=-1, keepdim=True)
        selected = selected / torch.where(denominator > 0, denominator, 1.0)
    if sigmoid:
        selected = selected * routed_scaling_factor
    weights.copy_(selected.masked_fill(is_padding[:, None], 0.0))
    ids.copy_(chosen.masked_fill(is_padding[:, None], -1))
    m, k = weights.shape
    rows = torch.arange(m, dtype=source.dtype, device=source.device)[:, None]
    slots = torch.arange(k, dtype=source.dtype, device=source.device)[None, :]
    source.copy_(slots * m + rows)


class MetaXExtendedTopKBenchmark(base.Benchmark):
    def __init__(self, *, sigmoid):
        name = "topk_sigmoid" if sigmoid else "topk_softmax"
        super().__init__(
            op_name=name,
            torch_op=partial(padded_topk_reference, sigmoid=sigmoid),
            gems_op=getattr(flaggems_vllm, name),
            dtypes=[torch.bfloat16],
        )
        self.sigmoid = sigmoid

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (m, e, 8) for m in (1, 8, 32, 128, 512, 4096) for e in (256, 257, 512)
        ]
        self.shape_desc = (
            "tokens, experts, selected experts (with correction bias and padding)"
        )

    def get_input_iter(self, dtype):
        for m, e, k in self.shapes:
            options = {
                "e_score_correction_bias": torch.randn(e, device=self.device) * 0.02,
                "is_padding": torch.arange(m, device=self.device).remainder(19).eq(18),
            }
            if self.sigmoid:
                options["routed_scaling_factor"] = 2.5
            yield (
                torch.empty((m, k), dtype=torch.float32, device=self.device),
                torch.empty((m, k), dtype=torch.int32, device=self.device),
                torch.empty((m, k), dtype=torch.int32, device=self.device),
                torch.randn((m, e), dtype=dtype, device=self.device),
                True,
                options,
            )


@pytest.mark.topk_sigmoid
@pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX workload")
def test_perf_metax_padded_topk_sigmoid():
    MetaXExtendedTopKBenchmark(sigmoid=True).run()
