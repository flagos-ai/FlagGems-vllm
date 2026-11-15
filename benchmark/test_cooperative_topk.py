# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Separate torch and vLLM baselines; latency is ms, SpeedUp = base / gems.

Each shape has homogeneous full-row lengths. The torch baseline includes the
int64-to-int32 output copy and short-row padding required by the public API.
No device-to-host length reads occur in timed or CUDA-graph-captured code.
The core set covers the short-row path, the 16384 regression cases, padded
rows and million-column rows. Comprehensive adds the rows/width/k sweep.
Report each shape's latency and SpeedUp separately; the default kernel mode
uses median latency and does not mix native-vLLM and torch baseline results.
"""

import pytest
import torch

import flaggems_vllm

from . import base

pytestmark = [
    pytest.mark.cooperative_topk,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def torch_cooperative_topk(logits, lengths, output, workspace, k, max_seq_len):
    width = logits.shape[1]
    if width <= k:
        output.fill_(-1)
        output[:, :width].copy_(torch.arange(width, device=logits.device))
    else:
        output.copy_(torch.topk(logits, k, dim=-1, sorted=False).indices)


class CooperativeTopKBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "rows, width, k"
    DEFAULT_SHAPES = [
        (1, 127, 512),
        (4, 4097, 512),
        (16, 16383, 1024),
        (16, 16384, 1024),
        (16, 16385, 1024),
        (4, 65539, 2048),
        (1, 262144, 512),
        (64, 262144, 2048),
        (1, 1048576, 1024),
    ]

    def set_more_shapes(self):
        return [
            (rows, width, k)
            for rows in (1, 4, 16, 64)
            for width in (4096, 16384, 16385, 65536, 262144, 1048576)
            for k in (512, 1024, 2048)
        ]

    def get_input_iter(self, dtype):
        for rows, width, k in self.shapes:
            # Preserve arbitrary width while meeting upstream's row-alignment
            # constraint, so both baseline suites consume equivalent layouts.
            stride = (width + 3) // 4 * 4
            logits = torch.randn(rows, stride, dtype=dtype, device=self.device)[
                :, :width
            ]
            lengths = torch.full((rows,), width, dtype=torch.int32, device=self.device)
            output = torch.empty(rows, k, dtype=torch.int32, device=self.device)
            workspace = torch.empty(rows * 16384, dtype=torch.uint8, device=self.device)
            yield logits, lengths, output, workspace, k, width


def test_cooperative_topk_torch():
    bench = CooperativeTopKBenchmark(
        op_name="cooperative_topk",
        torch_op=torch_cooperative_topk,
        gems_op=flaggems_vllm.cooperative_topk,
        dtypes=[torch.float32],
    )
    bench.run()


def test_cooperative_topk_vllm():
    if flaggems_vllm.vendor_name != "nvidia":
        pytest.skip("vLLM cooperative_topk requires NVIDIA")
    major, _ = torch.cuda.get_device_capability()
    if major < 9 or major == 12:
        pytest.skip("vLLM cooperative_topk requires SM90+ excluding SM12x")
    pytest.importorskip("vllm._custom_ops")
    if not hasattr(torch.ops._C, "cooperative_topk"):
        pytest.skip("vLLM build does not expose cooperative_topk")
    bench = CooperativeTopKBenchmark(
        op_name="cooperative_topk_vllm",
        torch_op=torch.ops._C.cooperative_topk,
        gems_op=flaggems_vllm.cooperative_topk,
        dtypes=[torch.float32],
    )
    bench.run()
