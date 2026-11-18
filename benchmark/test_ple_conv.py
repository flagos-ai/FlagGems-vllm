# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Perf coverage for FlagGems PLE short-conv APIs.

Formal NV baseline is ``ple_conv`` via ``baseline.ple_conv.native()``.
Legacy ``ple_short_conv_`` has no matching installed-vLLM entrypoint, so its
benchmark measures FlagGems latency only (no torch reference baseline).
"""

import pytest
import torch

from baseline import ple_conv as native_workload
from flaggems_vllm import ple_conv, ple_short_conv_

from . import base


class PLEShortConvBenchmark(base.GenericBenchmark):
    """Legacy FlagGems API smoke latency (no NV native counterpart)."""

    BASELINE_METRICS = ("latency_base", "speedup")

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

    def init_user_config(self):
        super().init_user_config()
        # No installed-vLLM API matches ple_short_conv_; do not use a torch
        # reference as latency_base (see test-capability doc §6.1).
        if self.torch_op is None:
            self.to_bench_metrics = [
                metric
                for metric in self.to_bench_metrics
                if metric not in self.BASELINE_METRICS
            ]


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
    """Upstream-compatible ``ple_conv`` vs installed vLLM native."""

    BASELINE_METRICS = ("latency_base", "speedup")

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            tuple(native_workload.key_shape(b)) for b in native_workload.grid()
        ]
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

    def init_user_config(self):
        super().init_user_config()
        if self.torch_op is None:
            self.to_bench_metrics = [
                metric
                for metric in self.to_bench_metrics
                if metric not in self.BASELINE_METRICS
            ]


@pytest.mark.ple_fusion
def test_ple_short_conv_perf():
    PLEShortConvBenchmark(
        input_fn=short_conv_inputs,
        op_name="ple_short_conv_",
        torch_op=None,
        gems_op=ple_short_conv_,
        dtypes=[torch.bfloat16, torch.float16],
    ).run()


@pytest.mark.ple_conv
def test_ple_conv_perf():
    baseline_op = native_workload.native()
    PLEConvNativeBenchmark(
        op_name="ple_conv",
        torch_op=baseline_op,
        gems_op=ple_conv,
        dtypes=native_workload.DTYPES,
        is_inplace=native_workload.IS_INPLACE,
    ).run()
