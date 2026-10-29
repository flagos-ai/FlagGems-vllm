# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import flaggems_vllm
from benchmark.base import Benchmark
from flaggems_vllm.ops.gdn_w8a16 import quantize_gdn_state_fp8


def _vllm_bf16(q, k, v, g, beta, state_bf16, state_fp8, state_scale, cu, indices, t):
    from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule

    return fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=128**-0.5,
        initial_state=state_bf16,
        inplace_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=True,
    )


def _fp8(q, k, v, g, beta, state_bf16, state_fp8, state_scale, cu, indices, t):
    return flaggems_vllm.gdn_w8a16(
        q,
        k,
        v,
        g,
        beta,
        128**-0.5,
        state_fp8,
        state_scale,
        cu_seqlens=cu,
        ssm_state_indices=indices[:, 0],
        use_qk_l2norm_in_kernel=True,
        max_sequence_length=t,
    )


class GdnW8A16Benchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_DESC = "N, T, H, HV, K, V"
    DEFAULT_SHAPES = [
        (n, t, 2, hv, 128, 128)
        for hv in (4, 6)
        for n, t in (
            [
                (n, 1)
                for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
            ]
            + [
                (1, 2),
                (1, 8),
                (1, 32),
                (4, 4),
                (4, 16),
                (16, 4),
                (1, 256),
                (1, 512),
                (2, 192),
                (4, 256),
                (8, 128),
                (16, 128),
                (32, 64),
                (64, 32),
                (128, 24),
            ]
        )
    ]

    def set_more_shapes(self):
        return self.DEFAULT_SHAPES

    def get_input_iter(self, cur_dtype):
        for n, t, h, hv, key_dim, value_dim in self.shapes:
            if key_dim != 128 or value_dim != 128:
                raise ValueError("This benchmark covers K=V=128")
            torch.manual_seed(42)
            device = flaggems_vllm.device
            q = torch.randn(1, n * t, h, key_dim, device=device, dtype=cur_dtype)
            k = torch.randn_like(q)
            v = 0.125 * torch.randn(
                1, n * t, hv, value_dim, device=device, dtype=cur_dtype
            )
            g = torch.empty(1, n * t, hv, device=device, dtype=cur_dtype).uniform_(
                math.log(0.98), math.log(0.995)
            )
            beta = torch.rand(1, n * t, hv, device=device, dtype=cur_dtype)
            state = torch.zeros(
                n + 1, hv, key_dim, value_dim, device=device, dtype=cur_dtype
            )
            state_fp8, state_scale = quantize_gdn_state_fp8(state)
            cu = torch.arange(0, n * t + 1, t, device=device, dtype=torch.int32)
            indices = (
                torch.arange(1, n + 1, device=device, dtype=torch.int64)[:, None]
                .expand(n, t)
                .contiguous()
            )
            inputs = q, k, v, g, beta, state, state_fp8, state_scale, cu, indices, t
            baseline, _ = _vllm_bf16(*inputs)
            torch.cuda.synchronize()
            # A zero initial state gives a closed-form first-token output.
            query = q.reshape(n, t, h, key_dim)[:, 0].float()
            key = k.reshape(n, t, h, key_dim)[:, 0].float()
            query *= torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
            key *= torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
            correlation = (query * key).sum(-1).repeat_interleave(hv // h, -1)
            expected = (
                correlation[..., None]
                * key_dim**-0.5
                * v.reshape(n, t, hv, value_dim)[:, 0].float()
                * beta.reshape(n, t, hv)[:, 0].float()[..., None]
            )
            first_output = baseline.reshape(n, t, hv, value_dim)[:, 0]
            torch.testing.assert_close(
                first_output.float(), expected, atol=2e-3, rtol=5e-2
            )
            state.zero_()
            yield inputs


@pytest.mark.gdn_w8a16
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 9,
    reason="FP8 GDN requires NVIDIA SM90 or newer",
)
def test_gdn_w8a16():
    bench = GdnW8A16Benchmark(op_name="gdn_w8a16", torch_op=_vllm_bf16)
    bench.set_gems(_fp8)
    bench.run()
