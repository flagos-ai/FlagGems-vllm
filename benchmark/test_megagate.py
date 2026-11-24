# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm
from baseline.megagate import GATE_PROFILES, generate_gate_inputs, torch_megagate

from .base import Benchmark


class MegaGateBenchmark(Benchmark):
    DEFAULT_SHAPES = [("dsv4-flash", 1028)]

    def get_input_iter(self, dtype):
        for profile, tokens in self.shapes:
            args, kwargs = generate_gate_inputs(GATE_PROFILES[profile], tokens, "cuda")
            yield (*args, kwargs)


@pytest.mark.megagate
@pytest.mark.skipif(flaggems_vllm.vendor_name != "thead", reason="requires T-Head PPU")
def test_megagate_perf():
    MegaGateBenchmark(
        op_name="megagate",
        torch_op=torch_megagate,
        gems_op=flaggems_vllm.megagate,
        dtypes=[torch.bfloat16],
    ).run()
