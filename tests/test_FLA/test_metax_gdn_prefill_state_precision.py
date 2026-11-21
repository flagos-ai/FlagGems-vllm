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

pytestmark = [
    pytest.mark.chunk_gated_delta_rule_fwd,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "metax", reason="MetaX V-major runtime coverage"
    ),
]
ATOL, RTOL = 2e-3, 2e-2


def _reference(q, k, v, g, beta, initial, boundaries):
    """FP32 recurrence on the same quantized inputs, with V-major states."""
    groups = v.shape[-2] // k.shape[-2]
    q = q.cpu().float().repeat_interleave(groups, 2)
    k = k.cpu().float().repeat_interleave(groups, 2)
    v, g, beta = (x.cpu().float() for x in (v, g, beta))
    state = initial.cpu().float().clone()
    out = torch.empty_like(v)
    for n, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        h = state[n].clone()
        for t in range(begin, end):
            h *= g[0, t].exp()[:, None, None]
            delta = (v[0, t] - (h * k[0, t, :, None, :]).sum(-1)) * beta[0, t, :, None]
            h += delta[:, :, None] * k[0, t, :, None, :]
            out[0, t] = (h * q[0, t, :, None, :]).sum(-1) * k.shape[-1] ** -0.5
        state[n] = h
    return out, state


def _inputs(
    K, V, lengths, seed, *, gate=None, zero=False, padding=0, dtype=torch.bfloat16
):
    torch.manual_seed(seed)
    T, Hg, H = sum(lengths), 8, 24
    q = torch.randn(1, T, Hg, K, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    q = (q.float() / (q.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(dtype)
    k = (k.float() / (k.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(dtype)
    v = torch.randn(1, T, H, V, device="cuda", dtype=dtype)
    g = -torch.rand(1, T, H, device="cuda")
    beta = torch.rand(1, T, H, device="cuda", dtype=dtype)
    if gate is not None:
        g.fill_(gate)
        beta = beta.float()
    storage = torch.randn(len(lengths), H * V * K + padding, device="cuda") * 0.01
    state = storage[:, : H * V * K].view(len(lengths), H, V, K)
    if zero:
        state.zero_()
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    return q, k, v, g, beta, state, storage, cu, boundaries


@pytest.mark.parametrize(
    "K,V,lengths,seed,gate,zero,padding",
    [
        (128, 128, [64, 96], 42, None, False, 0),
        (96, 80, [64, 96], 43, None, False, 0),
        (64, 64, [1, 63], 44, None, False, 0),
        (128, 128, [1, 63, 65], 45, None, False, 0),
        (128, 128, [0, 64], 46, None, False, 0),
        (128, 128, [64], 47, None, True, 0),
        (128, 128, [64, 96], 48, None, False, 512),
        (128, 128, [257], 100, -0.001, False, 0),
        (128, 128, [257], 101, -0.01, False, 0),
    ],
)
def test_metax_prefill_fp32_final_state_and_graph(
    K, V, lengths, seed, gate, zero, padding
):
    q, k, v, g, beta, initial, storage, cu, boundaries = _inputs(
        K, V, lengths, seed, gate=gate, zero=zero, padding=padding
    )
    fn = flaggems_vllm.chunk_gated_delta_rule_fwd
    assert fn.__module__ == "flaggems_vllm.runtime.backend._metax.fused.gdn_chunk"

    def run():
        return fn(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=K**-0.5,
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=cu,
        )

    def check(result, before):
        expected, final = _reference(q, k, v, g, beta, initial, boundaries)
        assert len(result) == 7
        assert result[1].dtype == q.dtype
        assert result[3].dtype == torch.float32
        assert result[3].shape == initial.shape
        torch.testing.assert_close(
            result[1].cpu().float(), expected, atol=ATOL, rtol=RTOL
        )
        torch.testing.assert_close(result[3].cpu(), final, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(storage, before, atol=0, rtol=0)

    before = storage.clone()
    check(run(), before)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run()
    for _ in range(2):
        q.neg_()
        v.mul_(0.5)
        initial.neg_()
        before = storage.clone()
        graph.replay()
        torch.cuda.synchronize()
        check(captured, before)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_metax_prefill_output_dtype_without_final_state(dtype):
    q, k, v, g, beta, _, _, cu, boundaries = _inputs(64, 64, [65], 52, dtype=dtype)
    initial = torch.zeros(1, 24, 64, 64, device="cuda")
    expected, _ = _reference(q, k, v, g, beta, initial, boundaries)
    result = flaggems_vllm.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=64**-0.5,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu,
    )
    assert result[1].dtype == dtype
    assert result[3] is None
    torch.testing.assert_close(result[1].cpu().float(), expected, atol=ATOL, rtol=RTOL)
