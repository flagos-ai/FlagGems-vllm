# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Benchmark FlagGems-vllm cooperative_topk against the native kernel.

Each shape has homogeneous full-row lengths. The torch reference includes the
int64-to-int32 output copy and short-row padding required by the public API.
No device-to-host length reads occur in timed or graph-captured code.
The core set covers the short-row path, the 16384 regression cases, padded
rows and million-column rows. Comprehensive adds the rows/width/k sweep.

The baseline follows the repo convention: use vLLM's native
``torch.ops._C.cooperative_topk`` when it is available on this build, and fall
back to a vectorized torch reference otherwise (never skip).
"""

import pytest
import torch

import flaggems_vllm

from . import base

pytestmark = [
    pytest.mark.cooperative_topk,
    pytest.mark.skipif(
        flaggems_vllm.device not in ("cuda", "npu")
        or not flaggems_vllm.runtime.torch_device_fn.is_available(),
        reason="CUDA or NPU device required",
    ),
]


def torch_cooperative_topk(logits, lengths, output, workspace, k, max_seq_len):
    width = logits.shape[1]
    if width <= k:
        output.fill_(-1)
        output[:, :width].copy_(torch.arange(width, device=logits.device))
    else:
        output.copy_(torch.topk(logits, k, dim=-1, sorted=False).indices)
    return output


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
            for width in (4096, 8192, 8193, 16384, 16385, 65536, 262144, 1048576)
            for k in (512, 1024, 2048)
        ] + [(17, 16384, 1024)]

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


def _native_cooperative_topk():
    """Resolve torch.ops._C.cooperative_topk on a supported NVIDIA build."""
    if flaggems_vllm.vendor_name != "nvidia":
        return None
    if not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability()
    if major < 9 or major == 12:
        return None
    try:
        import vllm._custom_ops  # noqa: F401
    except Exception:
        return None
    op = getattr(getattr(torch.ops, "_C", None), "cooperative_topk", None)
    return op if callable(op) else None


def _vllm_native_op(logits, lengths, output, workspace, k, max_seq_len):
    torch.ops._C.cooperative_topk(logits, lengths, output, workspace, k, max_seq_len)
    return output


def _get_baseline_op():
    if _native_cooperative_topk() is not None:
        return _vllm_native_op
    return torch_cooperative_topk


def test_cooperative_topk():
    bench = CooperativeTopKBenchmark(
        op_name="cooperative_topk",
        torch_op=_get_baseline_op(),
        gems_op=flaggems_vllm.cooperative_topk,
        dtypes=[torch.float32],
    )
    bench.run()
