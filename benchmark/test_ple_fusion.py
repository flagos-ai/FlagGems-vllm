# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm import ple_gate_norm, ple_prefill_short_conv_

from . import base
from .qwen4_references import _gate_norm_reference, prefill_conv_reference


class PLEBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(t, h) for t in (33, 1024, 4096) for h in (129, 2560)]
        self.shape_desc = "tokens,hidden"


def gate_inputs(shape, dtype, device):
    tokens, hidden = shape
    x = torch.randn(tokens, 4, hidden, device=device, dtype=dtype)
    yield (
        x,
        torch.randn_like(x),
        torch.randn(tokens, hidden, device=device, dtype=dtype),
        *[torch.randn(4 * hidden, device=device, dtype=dtype) * 0.01 for _ in range(3)],
        4,
        1e-6,
    )


def gate_reference(key, query, value, kw, qw, cw, hc, eps):
    return _gate_norm_reference(key, query, value, kw, qw, cw, eps)


def conv_inputs(shape, dtype, device):
    tokens, hidden = shape
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    yield x, torch.empty_like(x), torch.randn(
        2, hidden, 9, device=device, dtype=dtype
    ), torch.randn(hidden, 4, device=device, dtype=dtype), torch.tensor(
        [0, tokens], device=device, dtype=torch.int32
    ), torch.tensor(
        [1], device=device, dtype=torch.int32
    ), torch.tensor(
        [False], device=device, dtype=torch.bool
    ), {
        "num_prefills": 1,
        "max_len": tokens,
        "state_len": 9,
        "kernel_width": 4,
        "dilation": 3,
        "null_block_id": -1,
    }


@pytest.mark.ple_fusion
def test_ple_gate_norm_perf():
    PLEBenchmark(
        input_fn=gate_inputs,
        op_name="ple_gate_norm",
        torch_op=gate_reference,
        gems_op=ple_gate_norm,
        dtypes=[torch.bfloat16, torch.float16, torch.float32],
    ).run()


@pytest.mark.ple_fusion
def test_ple_prefill_conv_perf():
    PLEBenchmark(
        input_fn=conv_inputs,
        op_name="ple_prefill_short_conv_",
        torch_op=prefill_conv_reference,
        gems_op=ple_prefill_short_conv_,
        dtypes=[torch.bfloat16, torch.float16, torch.float32],
    ).run()
