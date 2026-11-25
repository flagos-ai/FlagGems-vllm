# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
import pytest
import torch

import flaggems_vllm

from . import base, consts


def torch_concat_mla_q(ql_nope, q_pe, q_out):
    width = ql_nope.shape[-1]
    q_out[..., :width].copy_(ql_nope)
    q_out[..., width:].copy_(q_pe)


class ConcatMLAQBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [(t, 64, 512, 64) for t in (1, 7, 128, 8192)]
    DEFAULT_SHAPE_DESC = "tokens, heads, nope_width, rope_width"

    def set_more_shapes(self):
        return [(1, 1, 3, 5), (7, 4, 31, 17)]


@pytest.mark.concat_mla_q
def test_concat_mla_q():
    def input_kwargs(shape, dtype, device):
        tokens, heads, nope, rope = shape
        yield (
            torch.randn(tokens, heads, nope, dtype=dtype, device=device),
            torch.randn(tokens, 1, rope, dtype=dtype, device=device),
            torch.empty(tokens, heads, nope + rope, dtype=dtype, device=device),
        )

    bench = ConcatMLAQBenchmark(
        op_name="concat_mla_q",
        input_fn=input_kwargs,
        torch_op=torch_concat_mla_q,
        gems_op=flaggems_vllm.concat_mla_q,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
