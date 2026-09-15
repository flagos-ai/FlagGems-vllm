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
import torch.nn.functional as F

import flaggems_vllm
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name == "nvidia",
    reason=(
        "NVIDIA H20 CUDA illegal memory access: "
        "https://github.com/flagos-ai/FlagGems-vllm/issues/690"
    ),
)


def _normalized_randn(*shape, device, dtype):
    return F.normalize(
        torch.randn(*shape, device=device, dtype=torch.float32),
        p=2.0,
        dim=-1,
        eps=1e-6,
    ).to(dtype)


def naive_chunk_gated_delta_rule_fwd(
    q, k, v, g, beta, scale, initial_state, cu_seqlens=None
):
    """
    Naive reference implementation of chunk_gated_delta_rule_fwd.
    Implements the gated delta rule recurrence token-by-token:
        S_t = exp(g_t) * S_{t-1} + beta_t * k_t^T (v_t - k_t @ S_{t-1})
        o_t = q_t @ S_t * scale

    Supports optional GQA layouts (q/k may have fewer heads than v) and
    packed variable-length sequences via cu_seqlens, in which case one
    final state per sequence is returned.
    """
    B, T, Hg, K = q.shape
    H, V = v.shape[-2:]
    assert H % Hg == 0
    qk_head_group_size = H // Hg

    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()

    if cu_seqlens is None:
        segments = [(batch_idx, batch_idx, 0, T) for batch_idx in range(B)]
        num_sequences = B
    else:
        boundaries = cu_seqlens.cpu().tolist()
        segments = [
            (sequence_idx, 0, boundaries[sequence_idx], boundaries[sequence_idx + 1])
            for sequence_idx in range(len(boundaries) - 1)
        ]
        num_sequences = len(segments)

    final_state = torch.empty(
        num_sequences, H, K, V, device=q.device, dtype=torch.float32
    )
    o = torch.empty(B, T, H, V, device=q.device, dtype=torch.float32)
    for sequence_idx, batch_idx, bos, eos in segments:
        S = (
            initial_state[sequence_idx].float().clone()
            if initial_state is not None
            else torch.zeros(H, K, V, device=q.device, dtype=torch.float32)
        )
        for token_idx in range(bos, eos):
            q_t = q[batch_idx, token_idx].repeat_interleave(qk_head_group_size, dim=0)
            k_t = k[batch_idx, token_idx].repeat_interleave(qk_head_group_size, dim=0)
            v_t = v[batch_idx, token_idx]
            g_t = g[batch_idx, token_idx]
            beta_t = beta[batch_idx, token_idx]

            S = torch.exp(g_t)[:, None, None] * S
            kS = torch.einsum("hk,hkv->hv", k_t, S)
            delta = v_t - kS
            S = S + torch.einsum("hk,hv->hkv", k_t, delta) * beta_t[:, None, None]
            o[batch_idx, token_idx] = torch.einsum("hk,hkv->hv", q_t, S) * scale
        final_state[sequence_idx] = S
    return o, final_state


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.skipif(
    not (flaggems_vllm.device == "cuda" and has_triton_tle(3, 6, 0)),
    reason="Triton 3.6.0 compilation error on Hopper: 'ttng.warp_group_dot' op pipeliner issue",
)
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [64, 128])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("K", [64])
@pytest.mark.parametrize("V", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_chunk_gated_delta_rule_fwd_accuracy(B, T, H, K, V, dtype):
    device = flaggems_vllm.device
    torch.manual_seed(42)

    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5
    initial_state = torch.zeros(B, H, K, V, device=device, dtype=dtype)

    ref_o, ref_final_state = naive_chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, scale, initial_state
    )

    result = flaggems_vllm.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
    )
    # result is (g_cumsum, o, A, final_state, w_or_None, h_or_None, v_new_or_None)
    res_o = result[1]
    res_final_state = result[3]

    torch.testing.assert_close(res_o.float(), ref_o, rtol=1e-1, atol=2e-1)
    torch.testing.assert_close(
        res_final_state.float(), ref_final_state, rtol=1.5, atol=1.0
    )


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.skipif(
    not (flaggems_vllm.device == "cuda" and has_triton_tle(3, 6, 0)),
    reason="Triton 3.6.0 compilation error on Hopper: 'ttng.warp_group_dot' op pipeliner issue",
)
@pytest.mark.parametrize("T", [64, 128, 256])
def test_chunk_gated_delta_rule_fwd_no_initial_state(T):
    device = flaggems_vllm.device
    dtype = torch.bfloat16
    torch.manual_seed(0)

    B, H, K, V = 1, 4, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5

    ref_o, _ = naive_chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, None)

    result = flaggems_vllm.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    res_o = result[1]

    torch.testing.assert_close(res_o.float(), ref_o, rtol=1e-1, atol=2e-1)


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.skipif(
    not (flaggems_vllm.device == "cuda" and has_triton_tle(3, 6, 0)),
    reason="Triton 3.6.0 compilation error on Hopper: 'ttng.warp_group_dot' op pipeliner issue",
)
@pytest.mark.parametrize("T", [64, 128])
def test_chunk_gated_delta_rule_fwd_with_cu_seqlens(T):
    device = flaggems_vllm.device
    dtype = torch.bfloat16
    torch.manual_seed(1)

    B, H, K, V = 1, 4, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5
    initial_state = torch.zeros(B, H, K, V, device=device, dtype=dtype)
    cu_seqlens = torch.arange(T + 1, device=device, dtype=torch.long)

    result = flaggems_vllm.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    # Verify output shapes are correct
    res_o = result[1]
    res_final_state = result[3]
    assert res_o.shape == (B, T, H, V)
    assert res_final_state.shape[1:] == (H, K, V)


# ---------------------------------------------------------------------------
# Ascend-only coverage. The Ascend backend specializes on chunk size 64 and
# K=V=128, returns fp32 outputs and a seven-value protocol whose third entry
# (KKT inverse A) is always None, so it gets its own test group below instead
# of sharing the CUDA tests above.
# ---------------------------------------------------------------------------


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.skipif(
    flaggems_vllm.device not in ("ascend", "npu"),
    reason="Ascend-only accuracy coverage; Ascend backend requires K=V=128",
)
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_chunk_gated_delta_rule_fwd_accuracy_ascend(B, T, dtype):
    device = flaggems_vllm.device
    H, K, V = 4, 128, 128
    torch.manual_seed(42)

    q = _normalized_randn(B, T, H, K, device=device, dtype=dtype)
    k = _normalized_randn(B, T, H, K, device=device, dtype=dtype)
    v = 0.125 * torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5
    initial_state = torch.zeros(B, H, K, V, device=device, dtype=dtype)

    ref_o, ref_final_state = naive_chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, scale, initial_state
    )

    result = flaggems_vllm.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
    )
    res_o = result[1]
    res_final_state = result[3]

    assert len(result) == 7
    assert result[0].dtype == torch.float32
    assert result[2] is None
    assert res_final_state.dtype == torch.float32
    assert torch.isfinite(res_o).all()
    assert torch.isfinite(res_final_state).all()
    torch.testing.assert_close(res_o.float(), ref_o, rtol=1e-1, atol=2e-1)
    torch.testing.assert_close(
        res_final_state.float(), ref_final_state, rtol=1.5, atol=1.0
    )


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.skipif(
    flaggems_vllm.device not in ("ascend", "npu"),
    reason="Ascend-only coverage",
)
@pytest.mark.parametrize("T", [64, 128, 256])
def test_chunk_gated_delta_rule_fwd_no_initial_state_ascend(T):
    device = flaggems_vllm.device
    dtype = torch.bfloat16
    torch.manual_seed(0)

    B, H, K, V = 1, 4, 128, 128
    q = _normalized_randn(B, T, H, K, device=device, dtype=dtype)
    k = _normalized_randn(B, T, H, K, device=device, dtype=dtype)
    v = 0.125 * torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5

    ref_o, _ = naive_chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, None)

    result = flaggems_vllm.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )
    res_o = result[1]

    assert result[3] is None
    assert torch.isfinite(res_o).all()
    torch.testing.assert_close(res_o.float(), ref_o, rtol=1e-1, atol=2e-1)


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.skipif(
    flaggems_vllm.device not in ("ascend", "npu"),
    reason="Ascend-only coverage",
)
@pytest.mark.parametrize("sequence_lengths", [(21, 43), (63, 63, 63), (65, 129, 17)])
def test_chunk_gated_delta_rule_fwd_with_cu_seqlens_ascend(sequence_lengths):
    device = flaggems_vllm.device
    dtype = torch.bfloat16
    torch.manual_seed(1)

    T = sum(sequence_lengths)
    B, H, K, V = 1, 4, 128, 128
    q = _normalized_randn(B, T, H, K, device=device, dtype=dtype)
    k = _normalized_randn(B, T, H, K, device=device, dtype=dtype)
    v = 0.125 * torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    scale = K**-0.5
    boundaries = [0]
    for sequence_length in sequence_lengths:
        boundaries.append(boundaries[-1] + sequence_length)
    cu_seqlens = torch.tensor(boundaries, device=device, dtype=torch.long)
    initial_state = torch.zeros(
        len(sequence_lengths), H, K, V, device=device, dtype=dtype
    )
    ref_o, ref_final_state = naive_chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, scale, initial_state, cu_seqlens
    )

    result = flaggems_vllm.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    res_o = result[1]
    res_final_state = result[3]
    assert res_o.shape == (B, T, H, V)
    assert res_final_state.shape == (len(sequence_lengths), H, K, V)
    assert result[2] is None
    assert res_final_state.dtype == torch.float32
    assert torch.isfinite(res_o).all()
    assert torch.isfinite(res_final_state).all()
    torch.testing.assert_close(res_o.float(), ref_o, rtol=1e-1, atol=2e-1)
    torch.testing.assert_close(
        res_final_state.float(), ref_final_state, rtol=1.5, atol=1.0
    )


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.skipif(
    flaggems_vllm.device not in ("ascend", "npu"), reason="Ascend-only GQA coverage"
)
def test_chunk_gated_delta_rule_fwd_gqa_ascend():
    device = flaggems_vllm.device
    dtype = torch.bfloat16
    torch.manual_seed(7)

    B, T, Hg, H, K, V = 1, 65, 2, 4, 128, 128
    q_seq = _normalized_randn(B, T, Hg, K, device=device, dtype=dtype)
    k_seq = _normalized_randn(B, T, Hg, K, device=device, dtype=dtype)
    v_seq = 0.125 * torch.randn(B, T, H, V, device=device, dtype=dtype)
    g_seq = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta_seq = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    initial_state = 0.01 * torch.randn(B, H, K, V, device=device, dtype=torch.float32)
    scale = K**-0.5

    # The low-level forward is seq-first only (no head_first flavor) and takes
    # an explicit scale; the public wrapper's head-first/layout handling is
    # CUDA-only coverage (test_chunk_gated_delta_rule.py).
    result = flaggems_vllm.chunk_gated_delta_rule_fwd(
        q=q_seq,
        k=k_seq,
        v=v_seq,
        g=g_seq,
        beta=beta_seq,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
    )
    actual, actual_final_state = result[1], result[3]
    expected, expected_final_state = naive_chunk_gated_delta_rule_fwd(
        q_seq,
        k_seq,
        v_seq,
        g_seq,
        beta_seq,
        scale,
        initial_state,
    )

    assert actual.shape == v_seq.shape
    assert actual_final_state.shape == (B, H, K, V)
    assert actual_final_state.dtype == torch.float32
    torch.testing.assert_close(actual.float(), expected, rtol=1e-1, atol=2e-1)
    torch.testing.assert_close(
        actual_final_state.float(), expected_final_state, rtol=1.5, atol=1.0
    )


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.skipif(
    flaggems_vllm.device not in ("ascend", "npu"),
    reason="Ascend-only non-contiguous input coverage",
)
def test_chunk_gated_delta_rule_fwd_non_contiguous_inputs_ascend():
    device = flaggems_vllm.device
    dtype = torch.bfloat16
    torch.manual_seed(11)
    B, T, Hg, H, K, V = 1, 65, 2, 4, 128, 128

    q = _normalized_randn(B, T, Hg, K * 2, device=device, dtype=dtype)[..., ::2]
    k = _normalized_randn(B, T, Hg, K * 2, device=device, dtype=dtype)[..., ::2]
    v = (0.125 * torch.randn(B, T, H, V * 2, device=device, dtype=dtype))[..., ::2]
    g = F.logsigmoid(torch.randn(B, T, H * 2, device=device, dtype=dtype))[..., ::2]
    beta = torch.rand(B, T, H * 2, device=device, dtype=dtype).sigmoid()[..., ::2]
    initial_state = 0.01 * torch.randn(B, H, K, V, device=device, dtype=torch.float32)

    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()
    assert not g.is_contiguous()
    assert not beta.is_contiguous()

    expected, expected_final_state = naive_chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, K**-0.5, initial_state
    )
    result = flaggems_vllm.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=initial_state,
        output_final_state=True,
    )

    torch.testing.assert_close(result[1].float(), expected, rtol=1e-1, atol=2e-1)
    torch.testing.assert_close(
        result[3].float(), expected_final_state, rtol=1.5, atol=1.0
    )
