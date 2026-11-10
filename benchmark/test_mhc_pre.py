# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare the public MetaX path with the portable implementation, not itself."""
import importlib

import pytest
import torch

import flaggems_vllm as ops

from .test_mhc import MHCPreBenchmark


class MetaXMHCPreBenchmark(MHCPreBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Cover both sides of the fused/split crossing and the generic fallback.
        self.shapes = [
            (1, 4096),
            (64, 4096),
            (65, 4096),
            (128, 4096),
            (129, 4096),
            (512, 4096),
            (4096, 4096),
            (3, 128),
        ]


@pytest.mark.mhc_pre
@pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX backend comparison")
def test_mhc_pre_metax_vs_portable():
    portable = importlib.import_module("flaggems_vllm.ops.mhc.mhc_pre").mhc_pre
    assert ops.mhc_pre is not portable
    benchmark = MetaXMHCPreBenchmark(
        op_name="mhc_pre",
        torch_op=portable,
        gems_op=ops.mhc_pre,
        dtypes=[torch.bfloat16],
    )
    benchmark.run()
