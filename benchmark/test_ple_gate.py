# SPDX-License-Identifier: Apache-2.0
import torch

import flaggems_vllm

from . import base


class PleGateBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            op_name="ple_gate",
            torch_op=None,
            gems_op=flaggems_vllm.ple_gate,
            dtypes=[torch.bfloat16, torch.float16],
            is_inplace=False,
        )

    def get_inputs(self, shape, dtype, device):
        T, HC, H = shape
        key = torch.randn(T, HC * H, dtype=dtype, device=device)
        hidden = torch.randn(T, HC * H, dtype=dtype, device=device)
        value = torch.randn(T, H, dtype=dtype, device=device)
        nk = torch.randn(HC * H, dtype=dtype, device=device)
        nq = torch.randn(HC * H, dtype=dtype, device=device)
        ncw = torch.randn(HC * H, dtype=dtype, device=device)
        return key, value, hidden, nk, nq, ncw

    def get_shapes(self):
        return [(T, 4, 2560) for T in [128, 512, 2048, 4096, 8192]]
