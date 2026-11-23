# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm
from tests.test_d256_port_precision import (
    test_probability_quantization_accuracy_d256 as _cancel_case,
)
from tests.test_flash_attn_varlen_func_w8a8_int8 import _run_case

pytestmark = [
    pytest.mark.flash_attn_varlen_func_w8a8,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "metax", reason="MetaX D256 PV streaming"
    ),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mixed", [False, True])
def test_streaming_pv_output_buffer_and_lse(dtype, mixed):
    _run_case(
        [1, 1031] if mixed else [1024],
        [2051, 2051] if mixed else [2048],
        dim=256,
        heads=4,
        kvheads=2,
        paged=True,
        causal=True,
        dtype=dtype,
        broadcast_scales=True,
        return_lse=True,
    )


@pytest.mark.parametrize(
    "scale", [-0.125, 0.0, (0.125, 1.7) * 4], ids=["negative", "zero", "mixed-paths"]
)
def test_streaming_pv_cancellation_and_residual_heads(scale):
    _cancel_case(0, 1024, 2048, True, scale)


@pytest.mark.parametrize("key_bound", [43690, 43691])
def test_streaming_pv_workspace_boundary(key_bound):
    # B=1, Hkv=4, D=256: these capacities bracket 128 MiB of INT8 K
    # plus FP16 V. Only 2048 keys are active; padded capacity is not length.
    _run_case(
        [1024],
        [2048],
        dim=256,
        heads=4,
        kvheads=4,
        paged=True,
        causal=True,
        broadcast_scales=True,
        max_key_bound=key_bound,
        extra_cache_pages=(key_bound + 15) // 16 - 128,
        return_lse=True,
    )
