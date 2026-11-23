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

import math

import pytest
import torch

import flaggems_vllm
from tests.accuracy_utils import gems_assert_close
from tests.test_flash_attn_varlen_func_w8a8_int8 import _lse_atol, _reference


@pytest.mark.parametrize(
    "seed,query_length,kv_length,is_cancellation,value_scale",
    [
        (0, 512, 512, False, 1.7),
        (1, 512, 512, False, 1.7),
        (2, 512, 512, False, 1.7),
        (3, 512, 512, False, 1.7),
        (0, 1024, 2048, True, 1.7),
        (0, 2048, 4096, True, 1.7),
        (0, 128, 2048, True, 0.125),
        (0, 128, 2048, True, 0.25),
        (0, 128, 2048, True, -1.7),
        (0, 128, 2048, True, 0.0),
        (0, 128, 65536, True, 0.125),
        (0, 128, 2048, True, (0.125, 1.7) * 4),
    ],
    ids=[
        "random-0",
        "random-1",
        "random-2",
        "random-3",
        "cancel-short",
        "cancel-long",
        "cancel-below-bound",
        "cancel-above-bound",
        "cancel-negative-scale",
        "cancel-zero-scale",
        "cancel-long-kv",
        "cancel-mixed-head-scales",
    ],
)
def test_probability_quantization_accuracy_d256(
    seed, query_length, kv_length, is_cancellation, value_scale
):
    if is_cancellation:
        # Opposite V rows expose probability rounding when two logits almost agree.
        score_gap = 0.000244
        device = flaggems_vllm.device
        q = torch.full((query_length, 32, 256), 127, dtype=torch.int8, device=device)
        q[:, :, 0] = 1
        k = torch.full(
            (kv_length // 16, 16, 8, 256), -127, dtype=torch.int8, device=device
        )
        k[0, :2] = 0
        k[0, 1, :, 0] = 1
        v = torch.zeros_like(k)
        v[0, 0] = 127
        v[0, 1] = -127
        qs = torch.ones((1, 32, 1), device=device).expand(1, 32, query_length // 128)
        ks = torch.full((1, 8, 1), score_gap * math.sqrt(256), device=device).expand(
            1, 8, kv_length // 128
        )
        v_scales = torch.as_tensor(
            value_scale, dtype=torch.float32, device=device
        ).expand(8)
        vs = v_scales[None, :, None].expand_as(ks)
        cuq = torch.tensor([0, query_length], dtype=torch.int32, device=device)
        used = torch.tensor([kv_length], dtype=torch.int32, device=device)
        table = torch.arange(kv_length // 16, dtype=torch.int32, device=device).reshape(
            1, -1
        )
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=device)
        actual = flaggems_vllm.flash_attn_varlen_func(
            q,
            k,
            v,
            query_length,
            cuq,
            kv_length,
            seqused_k=used,
            block_table=table,
            q_descale=qs,
            k_descale=ks,
            v_descale=vs,
            causal=True,
            out=out,
        )
        expected = (-127 * v_scales.repeat_interleave(4) * math.tanh(score_gap / 2))[
            None, :, None
        ].expand_as(actual)
        gems_assert_close(actual.float(), expected, dtype=torch.float32, atol=0.025)
        return
    torch.manual_seed(seed)
    tensors, descales, references = [], [], []
    for heads in (16, 8, 8):
        x = torch.randn(
            (512, heads, 256), device=flaggems_vllm.device, dtype=torch.bfloat16
        )
        scale = x.float().abs().amax((0, 2)) / 127
        quant = (x.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        tensors.append(quant)
        descales.append(scale[None, :, None].expand(1, heads, 4))
        references.append(quant.float() * scale[:, None])
    cu = torch.tensor([0, 512], dtype=torch.int32, device=flaggems_vllm.device)
    out, lse = flaggems_vllm.flash_attn_varlen_func(
        *tensors,
        512,
        cu,
        512,
        cu,
        causal=True,
        return_softmax_lse=True,
        q_descale=descales[0],
        k_descale=descales[1],
        v_descale=descales[2],
    )
    expected, expected_lse = _reference(*references, [512], [512], True)
    gems_assert_close(out.float(), expected, dtype=(out.float()).dtype, atol=0.025)
    gems_assert_close(lse, expected_lse, dtype=(lse).dtype, atol=_lse_atol())
