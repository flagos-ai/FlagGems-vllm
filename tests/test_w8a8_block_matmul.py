# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils

IS_INT8 = flaggems_vllm.vendor_name == "thead"
MNK_SHAPES = [
    (1, 65, 130),
    (7, 129, 257),
    (32, 64, 384),
    (7, 129, 4225),
    (128, 256, 512),
    (64, 128, 128),
    (256, 256, 256),
    (83, 7748, 3884),
    (84, 7168, 3884),
]

# Retain INT8 absolute tolerances without scaling them by K. The common
# assertion supplies the relative tolerance for BF16 output.
INT8_ATOL = 2e-2


def _make_inputs(m, n, k, dtype, block_size=(128, 128), scale_offset=0.0):
    device = flaggems_vllm.device
    block_n, block_k = block_size
    if dtype == torch.int8:
        a = torch.randint(-127, 128, (m, k), device=device, dtype=dtype)
        b = torch.randint(-127, 128, (n, k), device=device, dtype=dtype)
    else:
        a = torch.randn((m, k), device=device).to(dtype)
        b = torch.randn((n, k), device=device).to(dtype)
    k_groups = (k + block_k - 1) // block_k
    n_groups = (n + block_n - 1) // block_n
    a_scale = (
        torch.rand((m, k_groups), device=device, dtype=torch.float32) * 0.01
        + scale_offset
    )
    b_scale = (
        torch.rand((n_groups, k_groups), device=device, dtype=torch.float32) * 0.01
        + scale_offset
    )
    return a, b, a_scale, b_scale


def _reference(a, b, a_scale, b_scale, block_n, block_k):
    if a.dtype != torch.int8:
        # Expand block scales without changing the FP8 reference's scale-before-dot order.
        k = a.shape[1]
        n = b.shape[0]
        a_scale = a_scale.float().repeat_interleave(block_k, dim=1)[:, :k]
        b_scale = b_scale.float().repeat_interleave(block_n, dim=0)
        b_scale = b_scale.repeat_interleave(block_k, dim=1)[:n, :k]
        result = (a.float() * a_scale) @ (b.float() * b_scale).T
        return utils.to_reference(result, True)

    result = torch.zeros((a.shape[0], b.shape[0]), device="cpu", dtype=torch.float32)
    a_cpu = a.cpu().float()
    b_cpu = b.cpu().float()
    a_scale = a_scale.cpu()
    b_scale = b_scale.cpu()
    for k_tile in range((a.shape[1] + block_k - 1) // block_k):
        ks = slice(k_tile * block_k, min((k_tile + 1) * block_k, a.shape[1]))
        for n_tile in range((b.shape[0] + block_n - 1) // block_n):
            ns = slice(n_tile * block_n, min((n_tile + 1) * block_n, b.shape[0]))
            result[:, ns] += (
                (a_cpu[:, ks] @ b_cpu[ns, ks].T)
                * a_scale[:, k_tile, None]
                * b_scale[n_tile, k_tile]
            )
    return result


@pytest.mark.w8a8_block_matmul
@pytest.mark.skipif(
    flaggems_vllm.vendor_name not in ("thead", "nvidia", "mthreads"),
    reason="No W8A8 block matmul backend",
)
@pytest.mark.parametrize("M,N,K", MNK_SHAPES)
def test_w8a8_block_matmul(M, N, K):
    output_dtype = torch.bfloat16
    if not hasattr(flaggems_vllm, "w8a8_block_matmul"):
        pytest.skip("W8A8 block matmul is unavailable on this architecture")
    if IS_INT8:
        dtype = torch.int8
    elif flaggems_vllm.vendor_name == "mthreads":
        if hasattr(torch, "float8_e4m3fn"):
            dtype = torch.float8_e4m3fn
        else:
            dtype = torch.float32
    else:
        if not torch.cuda.is_available():
            pytest.skip("w8a8_block_fp8_matmul test requires CUDA or mthreads")
        major, _ = torch.cuda.get_device_capability()
        if major > 8:
            dtype = torch.float8_e4m3fn
        elif major == 8:
            dtype = torch.float8_e5m2
        else:
            dtype = torch.float32

    block_n, block_k = 128, 128
    block_size = [block_n, block_k]
    A, B, As, Bs = _make_inputs(
        M, N, K, dtype, block_size, scale_offset=0.0 if IS_INT8 else 0.005
    )
    ref_out = _reference(A, B, As, Bs, block_n, block_k).to(output_dtype)
    with nullcontext() if IS_INT8 else flaggems_vllm.use_gems():
        res_out = flaggems_vllm.w8a8_block_matmul(
            A, B, As, Bs, block_size, output_dtype=output_dtype
        )
    assert res_out.dtype == output_dtype
    utils.gems_assert_close(
        res_out.cpu() if IS_INT8 else res_out,
        ref_out,
        dtype=output_dtype,
        reduce_dim=1 if IS_INT8 else K,
        atol=INT8_ATOL if IS_INT8 else 1e-4,
    )
