# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Perf coverage for FlagGems PLE short-conv APIs.

- ``ple_short_conv_`` keeps the historical eager/reference baseline.
- ``ple_conv`` compares against upstream vLLM native when available
  (``baseline.ple_conv.native()``), otherwise records gems latency only.
"""

import pytest
import torch

from baseline import ple_conv as native_workload
from flaggems_vllm import ple_conv, ple_short_conv_

from . import base
from .ple_conv_reference import ple_conv_reference


class PLEShortConvBenchmark(base.GenericBenchmark):
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


def short_conv_inputs(shape, dtype, device):
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


class PLEConvNativeBenchmark(base.Benchmark):
    """Upstream-compatible ``ple_conv`` vs installed/pinned vLLM native."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = [tuple(native_workload.key_shape(b)) for b in native_workload.grid()]
        self.shape_desc = "mode,rows,channels,fuse,kernel,dilation"

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            binding = dict(
                zip(
                    (
                        "mode",
                        "rows",
                        "channels",
                        "fuse_residual",
                        "kernel_size",
                        "dilation",
                    ),
                    shape,
                )
            )
            binding["fuse_residual"] = bool(binding["fuse_residual"])
            binding["spec_query_len"] = native_workload.SPEC_QUERY_LEN
            args, kwargs = native_workload.build_inputs(binding, dtype, self.device)
            yield (*args, kwargs)


@pytest.mark.ple_fusion
def test_ple_short_conv_perf():
    PLEShortConvBenchmark(
        input_fn=short_conv_inputs,
        op_name="ple_short_conv_",
        torch_op=ple_conv_reference,
        gems_op=ple_short_conv_,
        dtypes=[torch.bfloat16, torch.float16],
    ).run()


@pytest.mark.ple_conv
def test_ple_conv_perf():
    PLEConvNativeBenchmark(
        op_name="ple_conv",
        torch_op=native_workload.native(),
        gems_op=ple_conv,
        dtypes=native_workload.DTYPES,
        is_inplace=native_workload.IS_INPLACE,
    ).run()
