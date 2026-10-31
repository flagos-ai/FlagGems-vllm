# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise optional compiler features through real package imports and kernels."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("feature", ["pow", "assume"])
def test_optional_math_features(feature):
    code = r"""
import importlib
import sys
import torch
import triton.language as tl
feature = sys.argv[1]
if feature == "pow":
    for path in ("triton.language.math", "triton.language.extra.cuda.libdevice"):
        try:
            module = importlib.import_module(path)
        except ImportError:
            continue
        if hasattr(module, "pow"):
            delattr(module, "pow")
else:
    if hasattr(tl, "assume"):
        delattr(tl, "assume")
import flaggems_vllm as gems
assert callable(gems.concat_mla_q)
if feature == "pow":
    x = torch.randn(3, 254, device=gems.device)
    a, b = x.chunk(2, dim=-1)
    torch.testing.assert_close(gems.geglu(x), torch.nn.functional.gelu(a, approximate="tanh") * b)
else:
    from flaggems_vllm.ops.top_k_per_row_decode import top_k_per_row_decode
    from flaggems_vllm.ops.top_k_per_row_prefill import top_k_per_row_prefill
    logits = torch.randperm(4096, device=gems.device).float().view(1, -1)
    lengths = torch.tensor([4096], dtype=torch.int32, device=gems.device)
    starts = torch.zeros_like(lengths)
    output = torch.empty((1, 64), dtype=torch.int32, device=gems.device)
    expected = torch.topk(logits, 64).values.sort().values
    for op, args in ((top_k_per_row_decode, (1, lengths)),
                     (top_k_per_row_prefill, (starts, lengths))):
        op(logits, *args, output, 1, logits.stride(0), logits.stride(1), 64)
        actual = logits.gather(1, output.long()).sort().values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, feature],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
