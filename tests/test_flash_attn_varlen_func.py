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

from typing import List, Optional, Tuple

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils

device = flaggems_vllm.device
vendor_name = flaggems_vllm.vendor_name
IS_HOPPER = (
    vendor_name == "nvidia"
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 9
)


# Following varlen and paged attn tests are copied from
# https://github.com/vllm-project/flash-attention/blob/main/tests/test_vllm_flash_attn.py
def attn_bias_from_alibi_slopes(slopes, seqlen_q, seqlen_k, causal=False):
    device = slopes.device
    slopes = slopes.unsqueeze(-1).unsqueeze(-1)

    if causal:
        v = torch.arange(-seqlen_k + 1, 1, device=device, dtype=torch.float32)
        return v * slopes

    row_idx = torch.arange(seqlen_q, device=device, dtype=torch.long).unsqueeze(-1)
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    relative_pos = torch.abs(row_idx + seqlen_k - seqlen_q - col_idx)

    return -slopes * relative_pos.to(dtype=slopes.dtype)


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    attn_bias: torch.Tensor = None,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        # clone to avoid clobbering the query tensor
        q = query[start_idx : start_idx + query_len].clone()
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k)
        empty_mask = torch.ones(query_len, kv_len, device=q.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask,
                    diagonal=kv_len - (query_len + sliding_window) + 1,
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))

        if attn_bias is not None:
            attn = attn + attn_bias[i, :, :query_len, :kv_len]

        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def make_paged_kv_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    layout: str,
    device: str = device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Non-contiguous K/V caches: "interleaved" takes every other page, so the
    # page stride is still a multiple of 16; "padded" separates pages by one
    # element, so the page stride is odd.
    shape = (num_blocks, block_size, num_kv_heads, head_size)
    page_numel = block_size * num_kv_heads * head_size
    if layout == "interleaved":
        storage_shape = (num_blocks * 2, block_size, num_kv_heads, head_size)
        key_storage = torch.randn(*storage_shape, dtype=dtype, device=device)
        value_storage = torch.randn_like(key_storage)
        key_cache = key_storage[::2][:num_blocks]
        value_cache = value_storage[::2][:num_blocks]
    elif layout == "padded":
        storage_shape = (num_blocks, page_numel + 1)
        key_storage = torch.randn(*storage_shape, dtype=dtype, device=device)
        value_storage = torch.randn_like(key_storage)
        key_cache = key_storage[:, :page_numel].view(shape)
        value_cache = value_storage[:, :page_numel].view(shape)
    else:
        raise ValueError(f"unknown layout {layout}")

    assert key_cache.shape == shape
    assert value_cache.shape == shape
    assert key_cache.stride() == value_cache.stride()
    assert key_cache.stride(-1) == 1
    assert key_cache.stride(0) != page_numel
    return key_cache, value_cache


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2815: Not supported")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2816: Not working")
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("num_heads", [(4, 4), (8, 2), (16, 2)])
@pytest.mark.parametrize("head_size", [128, 192, 256])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", [32768, 2048])
# Hopper runs FA3 for optimize_init=False and the FA2 flash_attn_varlen_opt_func
# for optimize_init=True, so both implementations stay covered.
@pytest.mark.parametrize("optimize_init", [False, True])
@torch.inference_mode()
def test_flash_attn_varlen_func(
    monkeypatch,
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    alibi: bool,
    soft_cap: Optional[float],
    num_blocks: int,
    optimize_init: bool,
) -> None:
    # (Issue) numerical stability concern
    if alibi is True and soft_cap is not None:
        return

    with torch.device(flaggems_vllm.device):
        utils.init_seed(1234567890)

        if vendor_name == "cambricon":
            torch.manual_seed(123456)
            torch.mlu.manual_seed_all(123456)

        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads = num_heads[0]
        num_kv_heads = num_heads[1]
        assert num_query_heads % num_kv_heads == 0
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = (
            (sliding_window, sliding_window) if sliding_window is not None else (-1, -1)
        )
        scale = head_size**-0.5
        query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
        key_cache = torch.randn(
            num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
        )
        value_cache = torch.randn_like(key_cache)
        cu_query_lens = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )

        causal = True

        if alibi:
            # alibi_slopes = torch.rand(num_seqs, num_query_heads, device=device, dtype=torch.float32) * 0.3
            alibi_slopes = (
                torch.ones(
                    num_seqs,
                    num_query_heads,
                    device=device,
                    dtype=torch.float32,
                )
                * 0.3
            )
            attn_bias = attn_bias_from_alibi_slopes(
                alibi_slopes, max_query_len, max_kv_len, causal=causal
            )
        else:
            alibi_slopes, attn_bias = None, None

        if vendor_name == "cambricon" or (IS_HOPPER and not optimize_init):
            output = flaggems_vllm.flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seqused_k,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=causal,
                window_size=window_size,
                block_table=block_tables,
                softcap=soft_cap if soft_cap is not None else 0,
                alibi_slopes=alibi_slopes,
                fa_version=3 if IS_HOPPER else 2,
            )
        else:
            if optimize_init:
                output = flaggems_vllm.ops.flash_attn_varlen_opt_func(
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=cu_query_lens,
                    seqused_k=seqused_k,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_kv_len,
                    softmax_scale=scale,
                    causal=causal,
                    window_size=window_size,
                    block_table=block_tables,
                    softcap=soft_cap if soft_cap is not None else 0,
                    alibi_slopes=alibi_slopes,
                    fa_version=2,
                )
            else:
                output = flaggems_vllm.ops.flash_attn_varlen_func(
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=cu_query_lens,
                    seqused_k=seqused_k,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_kv_len,
                    softmax_scale=scale,
                    causal=causal,
                    window_size=window_size,
                    block_table=block_tables,
                    softcap=soft_cap if soft_cap is not None else 0,
                    alibi_slopes=alibi_slopes,
                    fa_version=2,
                )

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
            attn_bias=attn_bias,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
        )

        msg = f"{torch.max(torch.abs(output - ref_output))}"
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2, msg=msg)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2815: Not supported")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2816: Not working")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# flash_attn_varlen_opt_func only supports fa_version=2.
@pytest.mark.parametrize(
    "optimize_init, fa_version",
    [(False, 2), (True, 2)] + ([(False, 3)] if IS_HOPPER else []),
)
@pytest.mark.parametrize("layout", ["interleaved", "padded"])
@torch.inference_mode()
def test_flash_attn_varlen_func_noncontiguous_kv_cache(
    dtype: torch.dtype,
    optimize_init: bool,
    fa_version: int,
    layout: str,
) -> None:
    with torch.device(flaggems_vllm.device):
        utils.init_seed(1234567890)

        seq_lens = [(1, 1328), (5, 18), (129, 463)]
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_seqs = len(seq_lens)
        num_query_heads = 8
        num_kv_heads = 2
        head_size = 128
        block_size = 32
        num_blocks = 2048
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = (-1, -1)
        scale = head_size**-0.5

        query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
        key_cache, value_cache = make_paged_kv_cache(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            dtype=dtype,
            layout=layout,
            device=device,
        )
        cu_query_lens = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )

        op = (
            flaggems_vllm.ops.flash_attn_varlen_opt_func
            if optimize_init
            else flaggems_vllm.ops.flash_attn_varlen_func
        )
        kwargs = dict(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=seqused_k,
            max_seqlen_q=max_query_len,
            max_seqlen_k=max_kv_len,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            block_table=block_tables,
            softcap=0,
            fa_version=fa_version,
        )
        if fa_version == 3 and layout == "padded":
            # FA3 addresses paged K/V in whole token rows, so a page stride that
            # is not a multiple of the token stride must be rejected.
            with pytest.raises(NotImplementedError, match="page stride"):
                op(**kwargs)
            return

        output = op(**kwargs)

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
        )

        max_diff = torch.max(torch.abs(output - ref_output))
        msg = f"max_diff={max_diff}, k_stride={key_cache.stride()}"
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2, msg=msg)


@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2815: Not working")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2816: Not working")
@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (1, 18), (1, 463)]])
@pytest.mark.parametrize("num_heads", [(8, 2)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("soft_cap", [None, 10.0])
@pytest.mark.parametrize("num_blocks", [2048])
@torch.inference_mode()
def test_flash_attn_varlen_func_swap_qg(
    monkeypatch,
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
) -> None:
    with torch.device(flaggems_vllm.device):
        utils.init_seed(1234567890)
        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        num_query_heads = num_heads[0]
        num_kv_heads = num_heads[1]
        assert num_query_heads % num_kv_heads == 0
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = (
            (sliding_window, sliding_window) if sliding_window is not None else (-1, -1)
        )
        scale = head_size**-0.5
        query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
        key_cache = torch.randn(
            num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
        )
        value_cache = torch.randn_like(key_cache)
        cu_query_lens = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )

        if vendor_name == "cambricon" or IS_HOPPER:
            output = flaggems_vllm.flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seqused_k,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=window_size,
                block_table=block_tables,
                softcap=soft_cap if soft_cap is not None else 0,
                fa_version=3 if IS_HOPPER else 2,
            )
        else:
            output = flaggems_vllm.ops.flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_query_lens,
                seqused_k=seqused_k,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_kv_len,
                softmax_scale=scale,
                causal=True,
                window_size=window_size,
                block_table=block_tables,
                softcap=soft_cap if soft_cap is not None else 0,
                fa_version=2,
            )

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
        )

        torch.testing.assert_close(
            output, ref_output, atol=2e-2, rtol=1e-2
        ), f"{torch.max(torch.abs(output - ref_output))}"


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(not IS_HOPPER, reason="FA3 requires Hopper")
# GQA ratio 1 and 5 are not packed, 4 is packed into one head group.
@pytest.mark.parametrize("num_heads", [(4, 4), (10, 2), (8, 2)])
@pytest.mark.parametrize("window_left", [-1, 0])
@torch.inference_mode()
def test_flash_attn_varlen_func_fa3_attention_sink(
    num_heads: Tuple[int, int],
    window_left: int,
) -> None:
    with torch.device(flaggems_vllm.device):
        utils.init_seed(1234567890)

        query_lens = [256, 64]
        kv_lens = [2256, 2064]
        num_query_heads, num_kv_heads = num_heads
        head_size = 128
        dtype = torch.bfloat16
        scale = head_size**-0.5

        query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
        key = torch.randn(sum(kv_lens), num_kv_heads, head_size, dtype=dtype)
        value = torch.randn_like(key)
        s_aux = torch.randn(num_query_heads, dtype=torch.bfloat16)
        cu_query_lens = torch.tensor(
            [0] + query_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)
        cu_kv_lens = torch.tensor(
            [0] + kv_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)

        output, lse = flaggems_vllm.flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_query_lens,
            cu_seqlens_k=cu_kv_lens,
            max_seqlen_q=max(query_lens),
            max_seqlen_k=max(kv_lens),
            softmax_scale=scale,
            causal=True,
            window_size=(window_left, -1),
            s_aux=s_aux,
            return_softmax_lse=True,
            fa_version=3,
        )

        # The sink adds exp(s_aux) to every softmax denominator.
        ref_outputs, ref_lses = [], []
        q_start = k_start = 0
        for query_len, kv_len in zip(query_lens, kv_lens):
            q = query[q_start : q_start + query_len].float()
            k = key[k_start : k_start + kv_len].float()
            v = value[k_start : k_start + kv_len].float()
            k = torch.repeat_interleave(k, num_query_heads // num_kv_heads, dim=1)
            v = torch.repeat_interleave(v, num_query_heads // num_kv_heads, dim=1)
            attn = torch.einsum("qhd,khd->hqk", q, k) * scale
            q_pos = torch.arange(query_len)[:, None] + kv_len - query_len
            k_pos = torch.arange(kv_len)[None, :]
            mask = k_pos > q_pos
            if window_left >= 0:
                mask |= k_pos < q_pos - window_left
            attn.masked_fill_(mask, float("-inf"))
            sink = s_aux.float()[:, None, None].expand(-1, query_len, 1)
            attn = torch.cat([attn, sink], dim=-1)
            probs = torch.softmax(attn, dim=-1)[..., :-1]
            ref_outputs.append(torch.einsum("hqk,khd->qhd", probs, v))
            ref_lses.append(torch.logsumexp(attn, dim=-1))
            q_start += query_len
            k_start += kv_len

        ref_output = torch.cat(ref_outputs).to(dtype)
        ref_lse = torch.cat(ref_lses, dim=-1)
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name != "nvidia", reason="Large dense KV regression")
@pytest.mark.parametrize("fa_version", [2, 3] if IS_HOPPER else [2])
# "later_start": the second sequence starts past 2**31 elements of the packed
# K/V. "within_sequence": a single sequence spans more than 2**31 elements.
@pytest.mark.parametrize("layout", ["later_start", "within_sequence"])
@torch.inference_mode()
def test_flash_attn_varlen_func_large_dense_kv_offset(
    fa_version: int, layout: str
) -> None:
    if torch.cuda.mem_get_info()[0] < 10 * 1024**3:
        pytest.skip("large dense KV regression requires 10 GiB free")
    with torch.device(flaggems_vllm.device):
        num_kv_heads, head_size = 8, 128
        long_kv_len = 2**31 // (num_kv_heads * head_size) + 64
        kv_lens = [long_kv_len, 512] if layout == "later_start" else [long_kv_len]
        dtype = torch.bfloat16
        scale = head_size**-0.5

        # Only the trailing tokens of the last sequence are non-zero, so the
        # reference attends over the full last sequence cheaply.
        key = torch.zeros(sum(kv_lens), num_kv_heads, head_size, dtype=dtype)
        value = torch.zeros_like(key)
        key[-512:].normal_()
        value[-512:].normal_()
        query = torch.randn(len(kv_lens), num_kv_heads, head_size, dtype=dtype)
        cu_kv_lens = torch.tensor(
            [0] + kv_lens, dtype=torch.int32, device=device
        ).cumsum(dim=0, dtype=torch.int32)

        output = flaggems_vllm.flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=torch.arange(
                len(kv_lens) + 1, dtype=torch.int32, device=device
            ),
            cu_seqlens_k=cu_kv_lens,
            max_seqlen_q=1,
            max_seqlen_k=max(kv_lens),
            softmax_scale=scale,
            causal=True,
            fa_version=fa_version,
        )

        k = key[-kv_lens[-1] :].float()
        v = value[-kv_lens[-1] :].float()
        attn = torch.einsum("hd,khd->hk", query[-1].float(), k) * scale
        ref_output = torch.einsum("hk,khd->hd", torch.softmax(attn, dim=-1), v)
        torch.testing.assert_close(
            output[-1], ref_output.to(dtype), atol=2e-2, rtol=1e-2
        )


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(not IS_HOPPER, reason="FA3 requires Hopper")
# Non-causal MHA short queries that stay on direct, plus unpacked multi-row
# queries (MHA and GQA ratio 5) large enough to leave the BM16 decode buckets.
@pytest.mark.parametrize(
    "batch_size, query_len, kv_len, num_heads, causal",
    [
        (64, 1, 2048, (32, 32), False),
        (16, 16, 4096, (32, 32), False),
        (16, 64, 4096, (32, 32), True),
        (16, 64, 4096, (40, 8), True),
        (4, 512, 8192, (40, 8), False),
    ],
)
@pytest.mark.parametrize("paged", [False, True])
@torch.inference_mode()
def test_flash_attn_varlen_func_fa3_matches_fa2(
    batch_size: int,
    query_len: int,
    kv_len: int,
    num_heads: Tuple[int, int],
    causal: bool,
    paged: bool,
) -> None:
    with torch.device(flaggems_vllm.device):
        utils.init_seed(1234567890)

        num_query_heads, num_kv_heads = num_heads
        head_size, block_size = 128, 16
        dtype = torch.bfloat16
        query = torch.randn(
            batch_size * query_len, num_query_heads, head_size, dtype=dtype
        )
        cu_query_lens = (
            torch.arange(batch_size + 1, dtype=torch.int32, device=device) * query_len
        )
        kwargs = dict(
            q=query,
            cu_seqlens_q=cu_query_lens,
            max_seqlen_q=query_len,
            max_seqlen_k=kv_len,
            softmax_scale=head_size**-0.5,
            causal=causal,
            return_softmax_lse=True,
        )
        if paged:
            num_blocks = batch_size * kv_len // block_size
            key = torch.randn(
                num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
            )
            block_tables = torch.randperm(num_blocks, device=device).to(torch.int32)
            kwargs.update(
                block_table=block_tables.view(batch_size, -1),
                seqused_k=torch.full(
                    (batch_size,), kv_len, dtype=torch.int32, device=device
                ),
            )
        else:
            key = torch.randn(batch_size * kv_len, num_kv_heads, head_size, dtype=dtype)
            kwargs.update(
                cu_seqlens_k=torch.arange(
                    batch_size + 1, dtype=torch.int32, device=device
                )
                * kv_len
            )
        kwargs.update(k=key, v=torch.randn_like(key))

        output, lse = flaggems_vllm.flash_attn_varlen_func(**kwargs, fa_version=3)
        ref_output, ref_lse = flaggems_vllm.flash_attn_varlen_func(
            **kwargs, fa_version=2
        )
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)
