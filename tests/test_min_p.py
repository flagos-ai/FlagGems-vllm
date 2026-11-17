# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Compare the public min-p entry points on identical inputs."""

import pytest
import torch

try:
    from vllm.v1.worker.gpu.sample.min_p import apply_min_p as vllm_reference
except ImportError as exc:
    raise ImportError(
        "Failed to import apply_min_p from vLLM. "
        "Check that the installed vLLM version provides this API "
        "and its dependencies are installed correctly."
    ) from exc

from flaggems_vllm import apply_min_p
from flaggems_vllm import device as DEVICE
from flaggems_vllm import runtime

from . import conftest as cfg

pytestmark = [
    pytest.mark.apply_min_p,
    pytest.mark.skipif(
        DEVICE == "cpu" or not runtime.torch_device_fn.is_available(),
        reason="requires the active accelerator",
    ),
]


@pytest.mark.parametrize(
    "rows,vocab",
    (
        [(50, 1024)]
        if cfg.QUICK_MODE
        else [(50, 1024), (1, 32000), (32, 128256), (128, 152065)]
    ),
)
@pytest.mark.parametrize("probability", [0.1, 0.0])
def test_min_p(rows, vocab, probability):
    logits = torch.full((rows, vocab), 1e-2, dtype=torch.float32, device=DEVICE)
    logits[:, 0] = 10.0
    mapping = torch.arange(rows, dtype=torch.int32, device=DEVICE)
    min_p = torch.full((rows,), probability, dtype=torch.float32, device=DEVICE)
    expected = logits.clone()

    vllm_reference(expected, mapping, min_p)
    apply_min_p(logits, mapping, min_p)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
