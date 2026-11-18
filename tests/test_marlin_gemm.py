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
from flaggems_vllm.ops.marlin_gemm import UINT4B8_ID, marlin_gemm

from .accuracy_utils import gems_assert_close

GROUP_SIZE = 128


def _pack_marlin_int4(codes, out_dim, in_dim):
    """Scatter uint4b8 codes into the Marlin 16x64-unit int32 layout.

    Inverts the kernel addressing: unit (kb, pid_n) holds 128 words; word
    waddr = 4*((n%8)*4 + (k%8)//2) + n//16, nibble J = 4*(k%2)+2*(k//8)+(n//8)%2.
    Output shape (K//16, N*2), linear layout equals (K//16)*(N//64) units.
    """
    words = torch.zeros(
        (in_dim // 16) * (out_dim // 64) * 128, dtype=torch.int32, device=codes.device
    )
    kk = torch.arange(16, device=codes.device)
    n_loc = torch.arange(64, device=codes.device)
    waddr = 4 * (((n_loc % 8) * 4).unsqueeze(0) + ((kk % 8) // 2).unsqueeze(1)) + (
        n_loc // 16
    ).unsqueeze(0)
    J = (4 * (kk % 2) + 2 * (kk // 8)).unsqueeze(1) + ((n_loc // 8) % 2).unsqueeze(0)
    for kb in range(in_dim // 16):
        for pid_n in range(out_dim // 64):
            base = (kb * (out_dim // 64) + pid_n) * 128
            tile = codes[pid_n * 64 : (pid_n + 1) * 64, kb * 16 : (kb + 1) * 16].T
            words.index_put_(
                (base + waddr.reshape(-1).long(),),
                (tile << (4 * J)).reshape(-1).to(torch.int32),
                accumulate=True,
            )
    return words.view(in_dim // 16, out_dim * 2)


def _marlin_permute_scales(s_kn, out_dim):
    """Apply vLLM get_scale_perm to (K//gs, N) scales: 64-column chunks."""
    perm = torch.tensor(
        [i + 8 * j for i in range(8) for j in range(8)], device=s_kn.device
    )
    return s_kn.reshape(-1, 64)[:, perm].reshape(-1, out_dim).contiguous()


def _quantize_uint4b8(w, group_size=GROUP_SIZE):
    """Symmetric per-group uint4b8 quantization (torch, test-only).

    Returns (w_ref, q_marlin, scales): dequantized reference, Marlin-repacked
    int32 (K//16, N*2) weights and marlin-permuted (K//gs, N) scales.
    """
    out_dim, in_dim = w.shape
    grouped = w.reshape(out_dim, in_dim // group_size, group_size).float()
    scales = (grouped.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7.0).squeeze(-1)
    q = torch.round(grouped / scales.unsqueeze(-1)).clamp(-7, 7) + 8
    w_ref = ((q - 8) * scales.unsqueeze(-1)).to(w.dtype).reshape(out_dim, in_dim)
    q_marlin = _pack_marlin_int4(
        q.to(torch.int32).reshape(out_dim, in_dim), out_dim, in_dim
    )
    scales_kn = scales.t().to(w.dtype).contiguous()
    scales_perm = _marlin_permute_scales(scales_kn, out_dim)
    return w_ref.contiguous(), q_marlin.contiguous(), scales_perm


def _make_inputs(size_m, size_n, size_k, dtype, device, with_bias=False):
    w = torch.randn((size_n, size_k), dtype=dtype, device=device) * 0.1
    w_ref, b_q_weight, b_scales = _quantize_uint4b8(w)
    a = torch.randn((size_m, size_k), dtype=dtype, device=device)
    b_bias = torch.randn((size_n,), dtype=dtype, device=device) if with_bias else None
    workspace = torch.empty(0, dtype=dtype, device=device)
    ref = a.float() @ w_ref.float().T
    if with_bias:
        ref += b_bias.float()
    ref = ref.to(dtype)
    return a, b_q_weight, b_scales, b_bias, workspace, ref


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.marlin_gemm
class TestMarlinGemmW4A16:

    @pytest.mark.parametrize("size_m", [1, 16, 48, 128])
    @pytest.mark.parametrize("size_n", [256, 512])
    @pytest.mark.parametrize("size_k", [256, 512])
    def test_w4a16_accuracy(self, size_m, size_n, size_k):
        device = flaggems_vllm.device
        a, b_q, s, _, ws, ref = _make_inputs(
            size_m, size_n, size_k, torch.float16, device
        )
        out = marlin_gemm(
            a=a,
            c=None,
            b_q_weight=b_q,
            b_bias=None,
            b_scales=s,
            a_scales=None,
            global_scale=None,
            b_zeros=None,
            g_idx=None,
            perm=None,
            workspace=ws,
            b_q_type=UINT4B8_ID,
            size_m=size_m,
            size_n=size_n,
            size_k=size_k,
        )
        assert out.shape == (size_m, size_n)
        gems_assert_close(out, ref, torch.float16, atol=5e-3)

    @pytest.mark.parametrize("size_m", [1, 16, 48, 128])
    @pytest.mark.parametrize("size_n", [256, 512])
    @pytest.mark.parametrize("size_k", [256, 512])
    def test_w4a16_accuracy_bf16(self, size_m, size_n, size_k):
        device = flaggems_vllm.device
        a, b_q, s, _, ws, ref = _make_inputs(
            size_m, size_n, size_k, torch.bfloat16, device
        )
        out = marlin_gemm(
            a=a,
            c=None,
            b_q_weight=b_q,
            b_bias=None,
            b_scales=s,
            a_scales=None,
            global_scale=None,
            b_zeros=None,
            g_idx=None,
            perm=None,
            workspace=ws,
            b_q_type=UINT4B8_ID,
            size_m=size_m,
            size_n=size_n,
            size_k=size_k,
        )
        gems_assert_close(out, ref, torch.bfloat16, atol=5e-2)

    def test_w4a16_with_bias(self):
        device = flaggems_vllm.device
        a, b_q, s, bias, ws, ref = _make_inputs(
            32, 256, 256, torch.float16, device, with_bias=True
        )
        out = marlin_gemm(
            a=a,
            c=None,
            b_q_weight=b_q,
            b_bias=bias,
            b_scales=s,
            a_scales=None,
            global_scale=None,
            b_zeros=None,
            g_idx=None,
            perm=None,
            workspace=ws,
            b_q_type=UINT4B8_ID,
            size_m=32,
            size_n=256,
            size_k=256,
        )
        gems_assert_close(out, ref, torch.float16, atol=5e-3)

    def test_out_buffer(self):
        device = flaggems_vllm.device
        a, b_q, s, _, ws, ref = _make_inputs(16, 256, 256, torch.float16, device)
        c = torch.empty((16, 256), dtype=torch.float16, device=device)
        out = marlin_gemm(
            a=a,
            c=c,
            b_q_weight=b_q,
            b_bias=None,
            b_scales=s,
            a_scales=None,
            global_scale=None,
            b_zeros=None,
            g_idx=None,
            perm=None,
            workspace=ws,
            b_q_type=UINT4B8_ID,
            size_m=16,
            size_n=256,
            size_k=256,
        )
        assert out is c
        gems_assert_close(out, ref, torch.float16, atol=5e-3)

    def test_unsupported_quant_type(self):
        device = flaggems_vllm.device
        a, b_q, s, _, ws, _ = _make_inputs(16, 256, 256, torch.float16, device)
        with pytest.raises(NotImplementedError):
            marlin_gemm(
                a=a,
                c=None,
                b_q_weight=b_q,
                b_bias=None,
                b_scales=s,
                a_scales=None,
                global_scale=None,
                b_zeros=None,
                g_idx=None,
                perm=None,
                workspace=ws,
                b_q_type=UINT4B8_ID + 1,
                size_m=16,
                size_n=256,
                size_k=256,
            )

    def test_unsupported_act_order(self):
        device = flaggems_vllm.device
        a, b_q, s, _, ws, _ = _make_inputs(16, 256, 256, torch.float16, device)
        g_idx = torch.zeros(256, dtype=torch.int32, device=device)
        with pytest.raises(NotImplementedError):
            marlin_gemm(
                a=a,
                c=None,
                b_q_weight=b_q,
                b_bias=None,
                b_scales=s,
                a_scales=None,
                global_scale=None,
                b_zeros=None,
                g_idx=g_idx,
                perm=None,
                workspace=ws,
                b_q_type=UINT4B8_ID,
                size_m=16,
                size_n=256,
                size_k=256,
            )
