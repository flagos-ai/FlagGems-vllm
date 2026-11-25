# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm import ple_short_conv_

from . import base
from .ple_conv_reference import ple_conv_reference


class PLEConvBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (mode, rows, channels)
            for channels in (33, 10240)
            for mode, rows in (
                ("decode", 1),
                ("decode", 64),
                ("prefill", 33),
                ("prefill", 1024),
                ("spec", 64),
            )
        ]
        self.shape_desc = "mode,rows,channels"


def conv_inputs(shape, dtype, device):
    mode, rows, channels = shape
    query_len = 4 if mode == "spec" else 1
    tokens = rows * query_len
    indices = torch.arange(rows, dtype=torch.int32, device=device)
    kwargs = {"dilation": 3}
    if mode == "prefill":
        indices = indices[:1]
        kwargs["max_query_len"] = tokens
        kwargs["query_start_loc"] = torch.tensor(
            [0, tokens], dtype=torch.int32, device=device
        )
    elif mode == "spec":
        kwargs.update(
            query_start_loc=torch.arange(rows + 1, device=device, dtype=torch.int32)
            * query_len,
            accepted_tokens=torch.ones(rows, device=device, dtype=torch.int32),
            max_query_len=query_len,
        )
    x = torch.randn(tokens, channels, dtype=dtype, device=device)
    yield (
        x,
        torch.empty_like(x),
        torch.randn(rows + 1, 12, channels, dtype=dtype, device=device).transpose(1, 2),
        torch.randn(channels, 4, dtype=dtype, device=device),
        indices,
        kwargs,
    )


@pytest.mark.ple_fusion
def test_ple_conv_perf():
    PLEConvBenchmark(
        input_fn=conv_inputs,
        op_name="ple_short_conv_",
        torch_op=ple_conv_reference,
        gems_op=ple_short_conv_,
        dtypes=[torch.bfloat16, torch.float16],
    ).run()
