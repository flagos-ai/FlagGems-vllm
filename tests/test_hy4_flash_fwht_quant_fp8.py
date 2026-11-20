# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
        reason="Hy4 Flash IndexPool requires NVIDIA CUDA",
    ),
]

op = flaggems_vllm.hy4_flash_fwht_quant_fp8


def make_inputs(shape, dtype=torch.bfloat16):
    return (torch.randn((*shape, 128), device="cuda", dtype=dtype),)


def reference(vectors):
    output = vectors.float()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        pairs = output.reshape(vectors.shape[0], 128 // (2 * stride), 2, stride)
        left, right = pairs.unbind(dim=2)
        output = torch.stack((left + right, left - right), dim=2).reshape(vectors.shape)
    output = output * (128**-0.5)
    scale = (output.abs().amax(-1).clamp_min(1e-4) / 448).log2().ceil().exp2()
    q = (output / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale


@pytest.mark.parametrize("rows", [0, 1, 17])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.hy4_flash_fwht_quant_fp8
def test_fwht_quantized_bytes(rows, dtype):
    (x,) = make_inputs((rows,), dtype)
    if rows:
        x[0].zero_()
    q, scales = op(x)
    ref_q, ref_scale = reference(x)
    torch.testing.assert_close(q.view(torch.uint8), ref_q.view(torch.uint8))
    torch.testing.assert_close(scales, ref_scale, atol=0, rtol=0)


@pytest.mark.hy4_flash_fwht_quant_fp8
def test_fwht_rejects_wrong_width():
    with pytest.raises(ValueError, match="128"):
        op(torch.empty((1, 127), device="cuda"))
