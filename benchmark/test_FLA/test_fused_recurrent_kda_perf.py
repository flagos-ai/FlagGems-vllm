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
import triton

import flaggems_vllm
from benchmark.base import Benchmark
from flaggems_vllm.ops.FLA import fused_recurrent
from flaggems_vllm.ops.FLA.fused_recurrent_kda import fused_recurrent_kda_decode


def _vllm_024_kda_baseline(
    q,
    k,
    v,
    g,
    beta,
    scale,
    baseline_state,
    candidate_state,
    cu_seqlens,
    state_indices,
    baseline_out,
    candidate_out,
):
    """Launch the existing vLLM-style recurrent KDA Triton configuration."""
    del candidate_state, candidate_out
    batch, token_count, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    num_sequences = cu_seqlens.numel() - 1
    block_k = triton.next_power_of_2(key_dim)
    block_v = min(triton.next_power_of_2(value_dim), 8)
    grid = (
        triton.cdiv(key_dim, block_k),
        triton.cdiv(value_dim, block_v),
        num_sequences * value_heads,
    )
    fused_recurrent.fused_recurrent_gated_delta_rule_large_t_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=baseline_out,
        h0=baseline_state,
        ht=baseline_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=None,
        scale=scale,
        N=num_sequences,
        T=token_count,
        B=batch,
        H=heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        stride_init_state_token=baseline_state.stride(0),
        stride_final_state_token=baseline_state.stride(0),
        stride_indices_seq=state_indices.stride(0),
        stride_indices_tok=1,
        USE_INITIAL_STATE=True,
        INPLACE_FINAL_STATE=True,
        IS_BETA_HEADWISE=False,
        USE_QK_L2NORM_IN_KERNEL=True,
        IS_VARLEN=True,
        IS_CONTINUOUS_BATCHING=True,
        IS_SPEC_DECODING=False,
        IS_KDA=True,
        num_warps=1,
        num_stages=3,
    )
    return baseline_out, baseline_state


def _optimized_kda_candidate(
    q,
    k,
    v,
    g,
    beta,
    scale,
    baseline_state,
    candidate_state,
    cu_seqlens,
    state_indices,
    baseline_out,
    candidate_out,
):
    del baseline_state, baseline_out
    return fused_recurrent_kda_decode(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=candidate_state,
        ssm_state_indices=state_indices,
        enable_decode_optimization=True,
        cu_seqlens=cu_seqlens,
        out=candidate_out,
    )


class FusedRecurrentKDABenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    # Serving batch sizes exercised by the fixed SM90 schedule in PR #6015.
    DEFAULT_SHAPES = [(1,), (32,), (64,), (96,), (128,)]
    DEFAULT_SHAPE_DESC = "N"

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype):
        device = flaggems_vllm.device
        for (num_sequences,) in self.shapes:
            torch.manual_seed(2026 + num_sequences)
            q = torch.randn(1, num_sequences, 4, 128, dtype=cur_dtype, device=device)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            g = -5.0 * torch.sigmoid(
                torch.randn(
                    1,
                    num_sequences,
                    4,
                    128,
                    dtype=torch.float32,
                    device=device,
                )
            )
            beta = torch.sigmoid(
                torch.randn(1, num_sequences, 4, dtype=torch.float32, device=device)
            )
            state = 0.01 * torch.randn(
                num_sequences + 1,
                4,
                128,
                128,
                dtype=torch.float32,
                device=device,
            )
            baseline_state = state.clone()
            candidate_state = state.clone()
            cu_seqlens = torch.arange(
                num_sequences + 1, dtype=torch.int32, device=device
            )
            state_indices = torch.arange(
                1, num_sequences + 1, dtype=torch.int32, device=device
            )
            baseline_out = torch.empty_like(v)
            candidate_out = torch.empty_like(v)
            yield (
                q,
                k,
                v,
                g,
                beta,
                128**-0.5,
                baseline_state,
                candidate_state,
                cu_seqlens,
                state_indices,
                baseline_out,
                candidate_out,
            )


@pytest.mark.fused_recurrent_kda
def test_perf_fused_recurrent_kda():
    benchmark = FusedRecurrentKDABenchmark(
        op_name="fused_recurrent_kda",
        torch_op=_vllm_024_kda_baseline,
    )
    benchmark.set_gems(_optimized_kda_candidate)
    benchmark.run()
