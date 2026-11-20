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
from flaggems_vllm.ops.deepseek_v4_attention_fused_q_kv_rmsnorm import (
    fused_q_kv_rmsnorm,
)

pytestmark = pytest.mark.fused_q_kv_rmsnorm


def _rmsnorm_ref(x, weight, eps):
    y = x.float() * torch.rsqrt(
        torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps
    )
    return (y * weight.float()).to(x.dtype)


@pytest.mark.parametrize(
    ("tokens", "qdim", "kvdim"),
    [
        (4, 256, 128),
        (8, 1536, 512),
    ],
)
def test_fused_q_kv_rmsnorm_accuracy(tokens, qdim, kvdim):
    device = flaggems_vllm.runtime.device.name
    qr = torch.randn((tokens, qdim), device=device, dtype=torch.bfloat16)
    kv = torch.randn((tokens, kvdim), device=device, dtype=torch.bfloat16)
    q_weight = torch.randn((qdim,), device=device, dtype=torch.bfloat16)
    kv_weight = torch.randn((kvdim,), device=device, dtype=torch.bfloat16)
    eps = 1e-6

    q_out, kv_out = fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)

    fg_testing.assert_close(
        q_out,
        _rmsnorm_ref(qr, q_weight, eps),
        dtype=torch.bfloat16,
        equal_nan=True,
    )
    fg_testing.assert_close(
        kv_out,
        _rmsnorm_ref(kv, kv_weight, eps),
        dtype=torch.bfloat16,
        equal_nan=True,
    )
