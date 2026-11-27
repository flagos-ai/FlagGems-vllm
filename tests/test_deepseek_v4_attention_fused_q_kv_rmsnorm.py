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

import pytest
import torch

import flaggems_vllm
import flaggems_vllm.testing as fg_testing

pytestmark = pytest.mark.fused_q_kv_rmsnorm


def _rmsnorm_ref(x, weight, eps):
    y = x.float() * torch.rsqrt(
        torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps
    )
    return (y * weight.float()).to(x.dtype)


@pytest.mark.parametrize(
    ("tokens", "qdim", "kvdim", "dtype"),
    [
        (4, 256, 128, torch.float32),
        (4, 256, 128, torch.float32),
        (8, 1536, 512, torch.bfloat16),
        (8, 1536, 512, torch.bfloat16),
    ],
)
def test_fused_q_kv_rmsnorm_accuracy(tokens, qdim, kvdim, dtype):
    device = flaggems_vllm.runtime.device.name
    qr = torch.randn((tokens, qdim), device=device, dtype=dtype)
    kv = torch.randn((tokens, kvdim), device=device, dtype=dtype)
    q_weight = torch.randn((qdim,), device=device, dtype=dtype)
    kv_weight = torch.randn((kvdim,), device=device, dtype=dtype)
    eps = 1e-6

    q_out, kv_out = flaggems_vllm.fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)

    fg_testing.assert_close(
        q_out,
        _rmsnorm_ref(qr, q_weight, eps),
        dtype=dtype,
        equal_nan=True,
    )
    fg_testing.assert_close(
        kv_out,
        _rmsnorm_ref(kv, kv_weight, eps),
        dtype=dtype,
        equal_nan=True,
    )
