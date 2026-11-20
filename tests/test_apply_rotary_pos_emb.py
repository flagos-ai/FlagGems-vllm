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

import random
import time
from typing import Optional

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)


# Copied from transformers.models.llama.modeling_llama.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.cohere.modeling_cohere.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere/modeling_cohere.py
def rotate_interleave(x):
    """Rotates interleave the hidden dims of the input."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _torch_apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.Tensor] = None,
    rotary_interleaved: bool = False,
):
    q = q.float()
    k = k.float()
    if position_ids is None:
        cos = cos[None, : q.size(-3), None, :]
        sin = sin[None, : q.size(-3), None, :]
    else:
        cos = cos[position_ids].unsqueeze(-2)  # [bs, seq_len, 1, dim/2]
        sin = sin[position_ids].unsqueeze(-2)  # [bs, seq_len, 1, dim/2]
    if rotary_interleaved:
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # [bs, seq_len, 1, dim]
        sin = torch.repeat_interleave(sin, 2, dim=-1)  # [bs, seq_len, 1, dim]
        rotate_fn = rotate_interleave
    else:
        cos = torch.cat([cos, cos], dim=-1)  # [bs, seq_len, 1, dim]
        sin = torch.cat([sin, sin], dim=-1)  # [bs, seq_len, 1, dim]
        rotate_fn = rotate_half

    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)

    return q_embed, k_embed


def _get_rope_cos_sin(max_seq_len, dim, dtype, base=10000, device=flaggems_vllm.device):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)

    return cos, sin


@pytest.mark.apply_rotary_pos_emb
@pytest.mark.parametrize("batch_size", [2] if cfg.TO_CPU else [4, 8])
@pytest.mark.parametrize("max_seq_len", [16] if cfg.TO_CPU else [512, 2048])
@pytest.mark.parametrize("q_heads,k_heads", [(8, 1), (6, 2), (1, 1), (8, 8)])
@pytest.mark.parametrize("head_dim", [8] if cfg.TO_CPU else [64, 96, 128, 256])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("rotary_interleaved", [True, False])
@pytest.mark.parametrize("has_pos_id", [True, False])
@pytest.mark.parametrize("inplace", [False, True])
def test_apply_rotary_pos_emb(
    batch_size,
    max_seq_len,
    q_heads,
    k_heads,
    head_dim,
    dtype,
    has_pos_id,
    rotary_interleaved,
    inplace,
):
    seq_len = torch.randint(1, max_seq_len, (1,)).item()
    q = torch.randn(
        (batch_size, seq_len, q_heads, head_dim),
        dtype=dtype,
        device=flaggems_vllm.device,
    )
    k = torch.randn(
        (batch_size, seq_len, k_heads, head_dim),
        dtype=dtype,
        device=flaggems_vllm.device,
    )

    position_ids = torch.randint(
        0, max_seq_len, (batch_size, seq_len), device=flaggems_vllm.device
    )
    cos, sin = _get_rope_cos_sin(
        max_seq_len, head_dim, dtype, device=flaggems_vllm.device
    )

    ref_q = utils.to_reference(q, True)
    ref_k = utils.to_reference(k, True)
    ref_cos = utils.to_reference(cos, True)
    ref_sin = utils.to_reference(sin, True)
    ref_position_ids = utils.to_reference(position_ids)

    q_embed_ref, k_embed_ref = _torch_apply_rotary_pos_emb(
        q=ref_q,
        k=ref_k,
        cos=ref_cos,
        sin=ref_sin,
        position_ids=ref_position_ids if has_pos_id else None,
        rotary_interleaved=rotary_interleaved,
    )
    q_embed_out, k_embed_out = flaggems_vllm.apply_rotary_pos_emb(
        q=q,
        k=k,
        cos=cos,
        sin=sin,
        position_ids=position_ids if has_pos_id else None,
        rotary_interleaved=rotary_interleaved,
        inplace=inplace,
    )

    utils.gems_assert_close(q_embed_out, q_embed_ref, dtype)
    utils.gems_assert_close(k_embed_out, k_embed_ref, dtype)


@pytest.mark.apply_rotary_pos_emb
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("rotary_interleaved", [False, True])
@pytest.mark.parametrize(
    "tokens,q_heads,k_heads,head_dim,rotary_dim,packed",
    [
        (1, 10, 10, 128, 128, True),
        (129, 8, 2, 96, 96, False),
        (551, 10, 10, 128, 128, True),
        (1917, 10, 10, 128, 128, True),
        (129, 8, 2, 256, 64, True),
    ],
)
@torch.inference_mode()
def test_apply_rotary_pos_emb_inplace_pairs(
    dtype, rotary_interleaved, tokens, q_heads, k_heads, head_dim, rotary_dim, packed
):
    generator = torch.Generator().manual_seed(817)
    source = torch.randn(
        tokens, q_heads + 2 * k_heads, head_dim, generator=generator
    ).to(dtype)
    angles = torch.randn(2048, rotary_dim // 2, generator=generator)
    cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
    positions = torch.arange(tokens).roll(3)
    references = _torch_apply_rotary_pos_emb(
        source[:, :q_heads, :rotary_dim],
        source[:, q_heads : q_heads + k_heads, :rotary_dim],
        cos,
        sin,
        positions,
        rotary_interleaved,
    )
    outputs = []
    for _ in range(2):
        owner = source.to(flaggems_vllm.device, copy=True)
        query = owner[:, :q_heads, :rotary_dim]
        key = owner[:, q_heads : q_heads + k_heads, :rotary_dim]
        if not packed:
            query, key = query.contiguous(), key.contiguous()
        actual = flaggems_vllm.apply_rotary_pos_emb(
            query,
            key,
            cos.to(flaggems_vllm.device),
            sin.to(flaggems_vllm.device),
            position_ids=positions.to(flaggems_vllm.device),
            rotary_interleaved=rotary_interleaved,
            inplace=True,
        )
        for result, original, reference in zip(actual, (query, key), references):
            assert result.data_ptr() == original.data_ptr()
            # view() may normalize the unused stride of a singleton axis.
            for size, actual_stride, original_stride in zip(
                original.shape, result.stride(), original.stride()
            ):
                assert size == 1 or actual_stride == original_stride
            torch.testing.assert_close(
                result.cpu(),
                reference.to(dtype),
                atol=1e-6 if dtype == torch.float32 else 0.004,
                rtol=1e-5 if dtype == torch.float32 else 0.01,
            )
            torch.testing.assert_close(original.cpu(), result.cpu(), atol=0, rtol=0)
        # K/V neighbors and partial-RoPE tails are outside the mutation scope.
        torch.testing.assert_close(
            owner[:, q_heads + k_heads :].cpu(),
            source[:, q_heads + k_heads :],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            owner[..., rotary_dim:].cpu(), source[..., rotary_dim:], atol=0, rtol=0
        )
        outputs.append(tuple(t.cpu() for t in actual))
    for first, second in zip(*outputs):
        torch.testing.assert_close(first, second, atol=0, rtol=0)


@pytest.mark.apply_rotary_pos_emb
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph API required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rotary_interleaved", [False, True])
@torch.inference_mode()
def test_apply_rotary_pos_emb_inplace_graph_replay(dtype, rotary_interleaved):
    tokens, q_heads, k_heads, head_dim, rotary_dim = 17, 8, 2, 128, 64
    generator = torch.Generator().manual_seed(718)
    angles = torch.randn(256, rotary_dim // 2, generator=generator)
    cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
    owner = torch.ones(
        tokens,
        q_heads + 2 * k_heads,
        head_dim,
        device=flaggems_vllm.device,
        dtype=dtype,
    )
    query = owner[:, :q_heads, :rotary_dim]
    key = owner[:, q_heads : q_heads + k_heads, :rotary_dim]
    positions = torch.arange(tokens, device=flaggems_vllm.device)
    cos_device, sin_device = cos.to(owner.device), sin.to(owner.device)

    def run():
        flaggems_vllm.apply_rotary_pos_emb(
            query, key, cos_device, sin_device, positions, rotary_interleaved, True
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for epoch in (1, 2):
        source = torch.randn(owner.shape, generator=generator).to(dtype)
        pos = torch.arange(tokens) + epoch * 19
        references = _torch_apply_rotary_pos_emb(
            source[:, :q_heads, :rotary_dim],
            source[:, q_heads : q_heads + k_heads, :rotary_dim],
            cos,
            sin,
            pos,
            rotary_interleaved,
        )
        owner.copy_(source)
        positions.copy_(pos)
        graph.replay()
        torch.cuda.synchronize()
        for actual, reference in zip((query, key), references):
            torch.testing.assert_close(
                actual.cpu(), reference.to(dtype), atol=0.004, rtol=0.01
            )
        torch.testing.assert_close(
            owner[:, q_heads + k_heads :].cpu(),
            source[:, q_heads + k_heads :],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            owner[..., rotary_dim:].cpu(), source[..., rotary_dim:], atol=0, rtol=0
        )
