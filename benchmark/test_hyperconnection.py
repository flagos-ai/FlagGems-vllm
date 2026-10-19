# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm import qwen4_hc_combine_norm

from . import base
from .qwen4_references import _reference_combine_norm


class HCCombineBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(r, h) for r in (1, 8, 64) for h in (513, 2560)]
        self.shape_desc = "rows,hidden"


def inputs(shape, dtype, device):
    rows, hidden = shape
    yield (
        torch.randn(rows, 4 * hidden, device=device, dtype=dtype),
        torch.randn(rows, hidden, device=device, dtype=dtype),
        torch.randn(rows, 324, device=device, dtype=dtype)[:, 320:],
        torch.randn(4 * hidden, device=device, dtype=dtype) * 0.01,
        1e-6,
        4,
    )


@pytest.mark.hyperconnection
def test_hc_combine_norm_perf():
    HCCombineBenchmark(
        input_fn=inputs,
        op_name="qwen4_hc_combine_norm",
        torch_op=_reference_combine_norm,
        gems_op=qwen4_hc_combine_norm,
        dtypes=[torch.bfloat16, torch.float16],
    ).run()
