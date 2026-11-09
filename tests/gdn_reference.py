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

"""PyTorch GDN references shared by accuracy tests and Graph benchmarks."""

import torch


def torch_gdn_decode(
    mixed,
    a,
    b,
    a_log,
    bias,
    scale,
    state,
    output,
    indices,
    normalize,
    *,
    round_beta=False,
):
    """Vectorized PyTorch recurrence with FP32 state and one token per slot."""
    _, heads, columns, reduction = state.shape
    if isinstance(mixed, tuple):
        q, k, v = mixed
        tokens, query_heads = q.shape[1:3]
    else:
        tokens, packed = mixed.shape
        if tokens == 0:
            return output, state
        query_heads = (packed - heads * columns) // (2 * reduction)
        query, key, value = mixed.split(
            (query_heads * reduction, query_heads * reduction, heads * columns),
            dim=-1,
        )
        q = query.reshape(1, tokens, query_heads, reduction)
        k = key.reshape(1, tokens, query_heads, reduction)
        v = value.reshape(1, tokens, heads, columns)
    q, k = q[0].float(), k[0].float()
    if normalize:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(heads // query_heads, 1) * scale
    k = k.repeat_interleave(heads // query_heads, 1)
    slots = indices.clamp_min(0)
    original = state[slots].float()
    decay = torch.exp(-a_log.exp() * torch.nn.functional.softplus(a.float() + bias))
    current = original * decay[..., None, None]
    beta = b.float().sigmoid()
    if round_beta:
        beta = beta.to(b.dtype).float()
    delta = (v[0].float() - (current * k[..., None, :]).sum(-1)) * beta[..., None]
    current = current + delta[..., None] * k[..., None, :]
    valid = indices > 0
    result = (current * q[..., None, :]).sum(-1)
    output.copy_(
        torch.where(valid[:, None, None], result, 0).reshape(tokens, 1, heads, columns)
    )
    state.index_copy_(
        0,
        slots,
        torch.where(valid[:, None, None, None], current, original).to(state.dtype),
    )
    return output, state


def torch_gdn_prefill(
    q, k, v, g, beta, initial_state, bounds, *, normalize=True, output_dtype=None
):
    """PyTorch BF16/FP16-activation reference with FP32 recurrent state.

    The chunk algebra follows FlagGems PR 7279's PyTorch reference. Sequence
    boundaries are supplied on the host so timing contains no device reads.
    """
    dtype = q.dtype
    if normalize:
        q = (
            q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
        ).to(dtype)
        k = (
            k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
        ).to(dtype)
    state = initial_state
    group = v.shape[2] // q.shape[2]
    q = q[0].float().repeat_interleave(group, 1).transpose(0, 1)
    k = k[0].float().repeat_interleave(group, 1).transpose(0, 1)
    v = v[0].float().transpose(0, 1)
    g = g[0].transpose(0, 1)
    beta = beta[0].float().transpose(0, 1)
    outputs, states = [], []
    for seq, (start, stop) in enumerate(zip(bounds, bounds[1:])):
        length = stop - start
        if length == 0:
            outputs.append(v[:, start:stop])
            states.append(state[seq])
            continue
        chunks = (length + 63) // 64
        padding = chunks * 64 - length
        qq = torch.nn.functional.pad(q[:, start:stop], (0, 0, 0, padding)).reshape(
            q.shape[0], chunks, 64, q.shape[-1]
        )
        kk = torch.nn.functional.pad(k[:, start:stop], (0, 0, 0, padding)).reshape(
            k.shape[0], chunks, 64, k.shape[-1]
        )
        vv = torch.nn.functional.pad(v[:, start:stop], (0, 0, 0, padding)).reshape(
            v.shape[0], chunks, 64, v.shape[-1]
        )
        gg = (
            torch.nn.functional.pad(g[:, start:stop], (0, padding))
            .reshape(g.shape[0], chunks, 64)
            .cumsum(-1)
        )
        bb = torch.nn.functional.pad(beta[:, start:stop], (0, padding)).reshape(
            beta.shape[0], chunks, 64
        )
        decay = (gg[..., :, None] - gg[..., None, :]).exp().tril()
        lhs = (kk @ kk.transpose(-1, -2) * decay * bb[..., None]).tril(-1)
        # Strictly lower 64x64 L is nilpotent: the finite geometric series
        # gives (I+L)^-1 exactly, using only PyTorch batched matmuls.
        power = -lhs
        inverse = torch.eye(64, device=q.device) + power
        for _ in range(5):
            power = power @ power
            inverse = inverse + inverse @ power
        rhs = bb[..., None] * torch.cat((vv, gg.exp()[..., None] * kk), dim=-1)
        solved = inverse @ rhs
        u, w = solved.split((vv.shape[-1], kk.shape[-1]), dim=-1)
        q_exp = qq * gg.exp()[..., None]
        qk = (qq @ kk.transpose(-1, -2)) * decay
        weighted_k = kk * (gg[..., -1:] - gg).exp()[..., None]
        last_decay = gg[..., -1].exp()
        s = state[seq].transpose(-1, -2)
        chunk_outputs = []
        for i in range(chunks):
            update = u[:, i] - w[:, i] @ s
            chunk_outputs.append(q_exp[:, i] @ s + qk[:, i] @ update)
            s = (
                last_decay[:, i, None, None] * s
                + weighted_k[:, i].transpose(-1, -2) @ update
            )
        outputs.append(
            torch.stack(chunk_outputs, dim=1).reshape(
                q.shape[0], chunks * 64, v.shape[-1]
            )[:, :length]
        )
        states.append(s.transpose(-1, -2))
    output = torch.cat(outputs, dim=1).transpose(0, 1).unsqueeze(0) * (
        q.shape[-1] ** -0.5
    )
    return output.to(output_dtype or dtype), torch.stack(states).contiguous()


def recurrent_gdn_reference(q, k, v, g, beta, initial, boundaries, normalize, scale):
    """Token recurrence independent of the chunked prefill reference."""
    group = v.shape[2] // q.shape[2]
    q, k = q[0].float(), k[0].float()
    if normalize:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(group, 1) * scale
    k = k.repeat_interleave(group, 1)
    output = torch.empty_like(v)
    final = []
    for sequence, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = initial[sequence].float().clone()
        for token in range(begin, end):
            state = state * g[0, token].exp()[:, None, None]
            key = k[token, :, None, :]
            residual = v[0, token].float() - (state * key).sum(-1)
            state = (
                state + (residual * beta[0, token].float()[:, None])[..., None] * key
            )
            output[0, token] = (state * q[token, :, None, :]).sum(-1)
        final.append(state)
    return output, torch.stack(final)
