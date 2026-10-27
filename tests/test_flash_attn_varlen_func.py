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


# FP32 support reuses the MetaX direct kernel from #808. Unlike paged #759,
# this path accepts packed encoder inputs and preserves IEEE products.
def _metax_fp32_oracle(q, k, v, q_lens, k_lens, scale):
    q, k, v = [x.detach().cpu().double() for x in (q, k, v)]
    outputs, lses = [], []
    iq = ik = 0
    for nq, nk in zip(q_lens, k_lens):
        qi = q[iq : iq + nq].transpose(0, 1)
        ki, vi = [x[ik : ik + nk].transpose(0, 1) for x in (k, v)]
        groups = qi.shape[0] // ki.shape[0]
        ki, vi = [x.repeat_interleave(groups, dim=0) for x in (ki, vi)]
        if nk == 0:
            outputs.append(torch.zeros_like(qi).transpose(0, 1))
            lses.append(torch.full(qi.shape[:2], -torch.inf, dtype=torch.float64))
        else:
            out_chunks, lse_chunks = [], []
            for start in range(0, nq, 128):
                scores = qi[:, start : start + 128] @ ki.transpose(-1, -2) * scale
                out_chunks.append(scores.softmax(-1) @ vi)
                lse_chunks.append(scores.logsumexp(-1))
            outputs.append(
                torch.cat(out_chunks, dim=1).transpose(0, 1)
                if out_chunks
                else torch.empty_like(q[iq : iq + nq])
            )
            lses.append(
                torch.cat(lse_chunks, dim=1)
                if lse_chunks
                else torch.empty(q.shape[1], 0, dtype=torch.float64)
            )
        iq += nq
        ik += nk
    return torch.cat(outputs), torch.cat(lses, dim=1)


def _metax_fp32_inputs(q_lens, k_lens, heads, dim, layout, dtype=torch.float32):
    torch.manual_seed(1234)
    tensors = []
    for total, num_heads in (
        (sum(q_lens), heads[0]),
        (sum(k_lens), heads[1]),
        (sum(k_lens), heads[1]),
    ):
        if layout == "packed":
            x = torch.randn(total, 3, num_heads, dim, device=device, dtype=dtype)[:, 1]
        elif layout == "last_stride":
            x = torch.randn(total, num_heads, dim * 2, device=device, dtype=dtype)[
                :, :, ::2
            ]
        else:
            x = torch.randn(total, num_heads, dim, device=device, dtype=dtype)
        tensors.append(x)
    cuq = torch.tensor([0] + q_lens, dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )
    cuk = torch.tensor([0] + k_lens, dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )
    kwargs = dict(
        q=tensors[0],
        k=tensors[1],
        v=tensors[2],
        cu_seqlens_q=cuq,
        cu_seqlens_k=cuk,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        causal=False,
        return_softmax_lse=True,
    )
    return kwargs


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name != "metax", reason="MetaX FP32 extension")
@pytest.mark.parametrize("dim", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("heads", [(1, 1), (4, 2), (16, 16)])
@pytest.mark.parametrize("layout", ["dense", "packed", "last_stride"])
def test_metax_fp32_varlen_reference(dim, heads, layout):
    q_lens, k_lens = [1, 2, 15, 17, 53], [3, 2, 19, 13, 67]
    args = _metax_fp32_inputs(q_lens, k_lens, heads, dim, layout)
    out_storage = torch.full((sum(q_lens), 2, heads[0], dim), 12345.0, device=device)
    args["out"] = out_storage[:, 0]
    args["softmax_scale"] = 0.37
    out, lse = flaggems_vllm.flash_attn_varlen_func(**args)
    ref, ref_lse = _metax_fp32_oracle(
        args["q"], args["k"], args["v"], q_lens, k_lens, 0.37
    )
    assert out.data_ptr() == args["out"].data_ptr()
    torch.testing.assert_close(out.cpu().double(), ref, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(lse.cpu().double(), ref_lse, atol=2e-5, rtol=2e-5)
    assert (out_storage[:, 1] == 12345.0).all()


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name != "metax", reason="MetaX FP32 extension")
@pytest.mark.parametrize(
    "q_lens,k_lens", [([0, 0], [0, 3]), ([0, 2, 5], [3, 0, 7]), ([1, 1], [2, 3])]
)
def test_metax_fp32_empty_and_single_query(q_lens, k_lens):
    args = _metax_fp32_inputs(q_lens, k_lens, (4, 2), 64, "dense")
    out, lse = flaggems_vllm.flash_attn_varlen_func(**args)
    ref, ref_lse = _metax_fp32_oracle(
        args["q"], args["k"], args["v"], q_lens, k_lens, 0.125
    )
    torch.testing.assert_close(out.cpu().double(), ref, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(lse.cpu().double(), ref_lse, atol=2e-5, rtol=2e-5)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name != "metax", reason="MetaX FP32 extension")
@pytest.mark.parametrize("length", [1024, 4096, 8192])
@pytest.mark.parametrize("layout", ["dense", "packed"])
def test_metax_fp32_long_encoder(length, layout):
    args = _metax_fp32_inputs([length, 2], [length, 2], (16, 16), 64, layout)
    out, lse = flaggems_vllm.flash_attn_varlen_func(**args)
    ref, ref_lse = _metax_fp32_oracle(
        args["q"], args["k"], args["v"], [length, 2], [length, 2], 0.125
    )
    torch.testing.assert_close(out.cpu().double(), ref, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(lse.cpu().double(), ref_lse, atol=2e-5, rtol=2e-5)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name != "metax", reason="MetaX FP32 extension regression")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [16, 32, 64, 128, 256])
def test_metax_fp32_half_regression(dtype, dim):
    args = _metax_fp32_inputs([1, 17, 53], [3, 19, 67], (4, 2), dim, "packed", dtype)
    out, lse = flaggems_vllm.flash_attn_varlen_func(**args)
    ref, ref_lse = _metax_fp32_oracle(
        args["q"], args["k"], args["v"], [1, 17, 53], [3, 19, 67], dim**-0.5
    )
    torch.testing.assert_close(out.cpu().double(), ref, atol=0.02, rtol=0.02)
    torch.testing.assert_close(lse.cpu().double(), ref_lse, atol=0.02, rtol=0.02)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name != "metax", reason="MetaX FP32 extension")
@pytest.mark.parametrize(
    "change",
    [
        dict(causal=True),
        dict(window_size=(4, 4)),
        dict(softcap=1.0),
        dict(num_splits=2),
        dict(dropout_p=0.1),
    ],
)
def test_metax_fp32_unsupported(change):
    args = _metax_fp32_inputs([2], [2], (1, 1), 64, "dense")
    args.update(change)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.flash_attn_varlen_func(**args)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name != "metax", reason="MetaX FP32 extension")
def test_metax_fp32_no_alias_or_backward():
    args = _metax_fp32_inputs([2], [2], (1, 1), 64, "dense")
    args["out"] = args["q"]
    with pytest.raises(ValueError, match="alias"):
        flaggems_vllm.flash_attn_varlen_func(**args)
    del args["out"]
    args["q"].requires_grad_(True)
    with pytest.raises(NotImplementedError, match="forward-only"):
        flaggems_vllm.flash_attn_varlen_func(**args)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name != "metax", reason="MetaX FP32 resource coverage")
@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("heads", [(1, 1), (16, 16)])
@pytest.mark.parametrize("length", [127, 257, 1024])
def test_metax_fp32_wide_launches(dim, heads, length):
    args = _metax_fp32_inputs([length], [length], heads, dim, "packed")
    out, lse = flaggems_vllm.flash_attn_varlen_func(**args)
    ref, ref_lse = _metax_fp32_oracle(
        args["q"], args["k"], args["v"], [length], [length], dim**-0.5
    )
    torch.testing.assert_close(out.cpu().double(), ref, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(lse.cpu().double(), ref_lse, atol=2e-5, rtol=2e-5)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name != "metax", reason="MetaX FP32 extension")
@pytest.mark.parametrize(
    "feature", ["alibi", "paged", "seqused", "out_layout", "dimension", "grad_stride"]
)
def test_metax_fp32_contract_rejections(feature):
    args = _metax_fp32_inputs([2], [2], (1, 1), 64, "dense")
    error = NotImplementedError
    if feature == "alibi":
        args["alibi_slopes"] = torch.ones(1, device=device)
    elif feature in ("paged", "seqused"):
        args["cu_seqlens_k"] = None
        args["seqused_k"] = torch.tensor([2], device=device, dtype=torch.int32)
        if feature == "paged":
            args["block_table"] = torch.zeros(1, 1, device=device, dtype=torch.int32)
            args["k"] = args["k"].unsqueeze(0)
            args["v"] = args["v"].unsqueeze(0)
    elif feature == "out_layout":
        args["out"] = torch.empty(1, 1, 64, device=device).expand(2, -1, -1)
        error = ValueError
    elif feature == "dimension":
        for key in ("q", "k", "v"):
            args[key] = args[key][:, :, :24]
    elif feature == "grad_stride":
        args = _metax_fp32_inputs([2], [2], (1, 1), 64, "last_stride")
        args["q"].requires_grad_(True)
    with pytest.raises(error):
        flaggems_vllm.flash_attn_varlen_func(**args)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name != "metax", reason="MetaX FP32 extension")
def test_metax_fp32_document_isolation():
    args = _metax_fp32_inputs([17, 53], [17, 53], (4, 2), 64, "packed")
    first, first_lse = flaggems_vllm.flash_attn_varlen_func(**args)
    args["k"][17:] += 3
    args["v"][17:] *= -7
    second, second_lse = flaggems_vllm.flash_attn_varlen_func(**args)
    torch.testing.assert_close(first[:17], second[:17], atol=0, rtol=0)
    torch.testing.assert_close(first_lse[:, :17], second_lse[:, :17], atol=0, rtol=0)
    assert not torch.equal(first[17:], second[17:])


@pytest.mark.parametrize(
    "head_size,length,self_attention,expected",
    [
        (64, 127, True, "mha_block_128"),
        (64, 128, True, "mha_block_64"),
        (64, 129, True, "mha_block_64"),
        (64, 1024, True, "mha_block_64"),
        (64, 1025, True, "mha_fp32_encoder_long"),
        (64, 8192, True, "mha_fp32_encoder_long"),
        (64, 4096, False, "mha_block_128"),
        (32, 4096, True, "mha_block_128"),
        (128, 4096, True, "mha_block_128"),
        (256, 4096, True, "mha_block_128"),
    ],
)
def test_metax_fp32_context_policy(head_size, length, self_attention, expected):
    from flaggems_vllm.runtime.backend._metax.heuristics_config_utils import (
        select_fp32_varlen_config,
    )

    assert (
        select_fp32_varlen_config(
            "mha_block_128",
            head_size=head_size,
            max_query_len=length,
            self_attention=self_attention,
        )
        == expected
    )


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name != "metax", reason="MetaX FP32 context policy")
@pytest.mark.parametrize("lengths", [[127], [128], [129], [1024], [4096], [2, 17, 129]])
@pytest.mark.parametrize("heads", [1, 16])
def test_metax_fp32_selected_context_path(monkeypatch, lengths, heads):
    from flaggems_vllm.runtime.backend._metax.ops.flash_attention import direct

    seen = []
    original = direct.flash_varlen_fwd_kernel

    class RecordLaunch:
        def __getitem__(self, grid):
            def call(*args, **kwargs):
                args[3].fill_(float("nan"))
                args[5].fill_(float("nan"))
                result = original[grid](*args, **kwargs)
                seen.append(kwargs)
                return result

            return call

    monkeypatch.setattr(direct, "flash_varlen_fwd_kernel", RecordLaunch())
    args = _metax_fp32_inputs(lengths, lengths, (heads, heads), 64, "packed")
    args["cu_seqlens_k"] = args["cu_seqlens_q"]
    out, lse = flaggems_vllm.flash_attn_varlen_func(**args)
    ref, ref_lse = _metax_fp32_oracle(
        args["q"], args["k"], args["v"], lengths, lengths, 0.125
    )
    torch.testing.assert_close(out.cpu().double(), ref, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(lse.cpu().double(), ref_lse, atol=2e-5, rtol=2e-5)
    assert len(seen) == 1
    if max(lengths) >= 128:
        assert (
            seen[0]["BLOCK_M"],
            seen[0]["BLOCK_N"],
            seen[0]["num_warps"],
            seen[0]["num_stages"],
        ) == ((64, 16, 4, 2) if max(lengths) > 1024 and heads == 16 else (64, 32, 4, 1))


@pytest.mark.parametrize("default", ["mha_block_16", "mha_block_32", "mha_block_64"])
def test_metax_fp32_long_preserves_other_native_profiles(default):
    from flaggems_vllm.runtime.backend._metax.heuristics_config_utils import (
        select_fp32_varlen_config,
    )

    assert (
        select_fp32_varlen_config(
            default, head_size=64, max_query_len=4096, self_attention=True
        )
        == default
    )
