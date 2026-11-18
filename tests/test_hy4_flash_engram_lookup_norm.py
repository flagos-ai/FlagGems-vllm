# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.hy4_flash_engram_lookup_norm,
    pytest.mark.skipif(
        not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
        reason="Hy4 Flash Engram requires NVIDIA CUDA",
    ),
]
op = flaggems_vllm.hy4_flash_engram_lookup_norm


def make_inputs(shape, *, fp8):
    m, h, n = shape
    rows = torch.randn(shape, device="cuda").to(
        torch.float8_e4m3fn if fp8 else torch.bfloat16
    )
    scales = torch.rand(m, h, dtype=torch.bfloat16, device="cuda") if fp8 else None
    gamma = torch.randn(h, n, dtype=torch.bfloat16, device="cuda")
    # Simulate TP-local heads sliced out of a larger hash matrix.
    hashes = torch.arange(m * 2 * h, device="cuda", dtype=torch.int32).view(m, 2 * h)
    hashes = hashes[:, h:]
    hashes[::2, 0] = -1
    return rows, scales, gamma, hashes, torch.bfloat16


def reference(rows, scales, gamma, hashes, out_dtype):
    x = rows.float()
    if scales is not None:
        x = x * scales.float().unsqueeze(-1)
    x = x.masked_fill(hashes.unsqueeze(-1) < 0, 0)
    rms = (x.square().mean(-1, keepdim=True) + torch.finfo(torch.float32).eps).rsqrt()
    return (x * rms * gamma.float()).to(out_dtype)


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("shape", [(0, 2, 128), (1, 1, 7), (4, 3, 128), (11, 4, 257)])
@pytest.mark.hy4_flash_engram_lookup_norm
def test_engram_lookup_norm(shape, fp8):
    args = make_inputs(shape, fp8=fp8)
    torch.testing.assert_close(op(*args), reference(*args), atol=0.0157, rtol=0.008)


@pytest.mark.hy4_flash_engram_lookup_norm
def test_lookup_norm_masks_nan_scale_and_replays():
    rows, scales, gamma, hashes, out_dtype = make_inputs((8, 4, 128), fp8=True)
    scales[0, 0] = float("nan")  # Negative hash must mask even invalid numeric data.
    args = rows, scales, gamma, hashes, out_dtype
    op(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args)
    for h in (-1, 1):
        hashes.fill_(h)
        graph.replay()
        torch.testing.assert_close(
            out, reference(*args), atol=0.0157, rtol=0.008, equal_nan=True
        )
    with pytest.raises(ValueError, match="row scales"):
        op(rows, None, gamma, hashes, out_dtype)
