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

import importlib

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

pytestmark = [
    pytest.mark.w8a8_block_int8_matmul,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "thead",
        reason="T-Head PPU W8A8 Block INT8 backend",
    ),
]

OUTPUT_DTYPES = (
    [torch.float32] if QUICK_MODE else [torch.float16, torch.bfloat16, torch.float32]
)

INT8_MNK_SHAPES = [
    (1, 65, 130),
    (7, 129, 257),
    (32, 64, 384),
    (7, 129, 4225),
]

# Retain INT8 absolute tolerances without scaling them by K. The common
# assertion supplies the relative tolerance for each output dtype.
INT8_ATOL = {torch.float16: 2e-2, torch.bfloat16: 2e-2, torch.float32: 1e-3}


def _reference(a, b, a_scale, b_scale, block_n, block_k):
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


@pytest.mark.parametrize("M,N,K", INT8_MNK_SHAPES)
@pytest.mark.parametrize("output_dtype", OUTPUT_DTYPES)
def test_w8a8_block_int8_matmul(M, N, K, output_dtype):
    device = flaggems_vllm.device
    block_n, block_k = 128, 128
    block_size = [block_n, block_k]

    A = torch.randint(-127, 128, (M, K), device=device, dtype=torch.int8)
    B = torch.randint(-127, 128, (N, K), device=device, dtype=torch.int8)

    num_k_groups = (K + block_k - 1) // block_k
    num_n_groups = (N + block_n - 1) // block_n
    As = torch.rand((M, num_k_groups), device=device, dtype=torch.float32) * 0.01
    Bs = (
        torch.rand((num_n_groups, num_k_groups), device=device, dtype=torch.float32)
        * 0.01
    )

    ref_out = _reference(A, B, As, Bs, block_n, block_k).to(output_dtype)
    res_out = flaggems_vllm.w8a8_block_int8_matmul(
        A, B, As, Bs, block_size, output_dtype=output_dtype
    )
    utils.gems_assert_close(
        res_out.cpu(),
        ref_out,
        dtype=output_dtype,
        reduce_dim=1,
        atol=INT8_ATOL[output_dtype],
    )


def test_w8a8_block_int8_empty_shapes():
    for m, n in ((0, 65), (3, 0)):
        k = 130
        a = torch.empty((m, k), device="cuda", dtype=torch.int8)
        b = torch.empty((n, k), device="cuda", dtype=torch.int8)
        a_scale = torch.empty((m, 2), device="cuda", dtype=torch.float32)
        b_scale = torch.empty((1 if n else 0, 2), device="cuda", dtype=torch.float32)
        out = flaggems_vllm.w8a8_block_int8_matmul(
            a, b, a_scale, b_scale, [128, 128], torch.float32
        )
        assert out.shape == (m, n)


def test_w8a8_block_int8_block_k256_scales():
    m, n, k = 7, 129, 512
    block_n, block_k = 128, 256
    a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    a_scale = torch.rand((m, 2), device="cuda", dtype=torch.float32) * 0.01
    b_scale = torch.rand((2, 2), device="cuda", dtype=torch.float32) * 0.01
    output = flaggems_vllm.w8a8_block_int8_matmul(
        a, b, a_scale, b_scale, [block_n, block_k], torch.float32
    )
    utils.gems_assert_close(
        output.cpu(),
        _reference(a, b, a_scale, b_scale, block_n, block_k),
        dtype=torch.float32,
        reduce_dim=1,
        atol=INT8_ATOL[torch.float32],
    )


def test_w8a8_block_int8_rejects_invalid_inputs():
    a = torch.zeros((1, 128), device="cuda", dtype=torch.int8)
    b = torch.zeros((64, 128), device="cuda", dtype=torch.int8)
    a_scale = torch.ones((1, 1), device="cuda", dtype=torch.float32)
    b_scale = torch.ones((1, 1), device="cuda", dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="block_k"):
        flaggems_vllm.w8a8_block_int8_matmul(
            a, b, a_scale, b_scale, [128, 16], torch.float32
        )
    with pytest.raises(NotImplementedError, match="contiguous"):
        flaggems_vllm.w8a8_block_int8_matmul(
            a[:, ::2], b[:, ::2], a_scale, b_scale, [128, 64], torch.float32
        )


@pytest.mark.parametrize(
    "splits",
    [2, 4, 8, 16],
)
@pytest.mark.parametrize("output_dtype", OUTPUT_DTYPES)
@pytest.mark.parametrize("group_n", [64])
def test_w8a8_block_int8_splitk_tails(splits, output_dtype, group_n):
    # Unused workspace slices remain NaN.  A correct reduction must exclude
    # them instead of relying on zero initialization or a previous invocation.
    backend = importlib.import_module(
        f"flaggems_vllm.runtime.backend._{flaggems_vllm.vendor_name}"
        ".ops.w8a8_block_int8_matmul"
    )
    m, n, k = 7, 129, 4225
    group_k = 128
    a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    groups = (k + group_k - 1) // group_k
    a_scale = torch.rand((m, groups), device="cuda") * 0.01
    b_scale = torch.rand(((n + group_n - 1) // group_n, groups), device="cuda") * 0.01
    workspace = torch.full(
        (backend._MAX_SPLITS.value * m * n + 1,), float("nan"), device="cuda"
    )
    output = torch.empty((m, n), device="cuda", dtype=output_dtype)
    backend._w8a8_block_int8_kernel_splitk.jit_function[((n + 31) // 32, splits)](
        a,
        b,
        workspace,
        output,
        a_scale,
        b_scale,
        m,
        n,
        k,
        GROUP_N=group_n,
        GROUP_K=group_k,
        BLOCK_M=16,
        BLOCK_N=32,
        GROUP_M=4,
        SPLITS=splits,
        SWAP=True,
        num_warps=2,
        num_stages=2,
    )
    backend._w8a8_block_int8_reduce_splitk[((m * n + 255) // 256,)](
        workspace, output, m, n, 256, num_warps=4
    )
    expected = _reference(a, b, a_scale, b_scale, group_n, group_k).to(output_dtype)
    utils.gems_assert_close(
        output.cpu(),
        expected,
        dtype=output_dtype,
        reduce_dim=1,
        atol=INT8_ATOL[output_dtype],
    )
