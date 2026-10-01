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

from itertools import product
from typing import List, Optional, Tuple, Union

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils
from .test_flash_attn_varlen_func_w8a8_int8 import _inputs, _reference, _run_case

device = flaggems_vllm.device
vendor_name = flaggems_vllm.vendor_name
DTYPES = [torch.float16, torch.bfloat16]
ATTENTION_CONFIGS = list(
    product(DTYPES, [(4, 4), (8, 2), (16, 2)], [128, 192, 256], [False, True])
)
if vendor_name == "metax":
    ATTENTION_CONFIGS += [
        pytest.param(torch.int8, (4, 4), 64, False, id="int8-mha-d64"),
        pytest.param(torch.int8, (8, 2), 128, False, id="int8-gqa-d128"),
    ]
    DTYPES = [*DTYPES, pytest.param(torch.int8, id="int8")]


@pytest.fixture(autouse=True)
def exact_reference_matmul(monkeypatch):
    if vendor_name != "metax":
        return
    # TF32 rounding exceeds the FP32 LSE tolerance for quantized inputs.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def attention_inputs(
    query_lens, kv_lens, num_heads, head_size, block_size, num_blocks, dtype
):
    shapes = [
        (sum(query_lens), num_heads[0], head_size),
        (num_blocks, block_size, num_heads[1], head_size),
        (num_blocks, block_size, num_heads[1], head_size),
    ]
    if dtype == torch.int8:
        tensors = [torch.randint(-127, 128, shape, dtype=dtype) for shape in shapes]
        descales = {
            name: torch.rand(len(query_lens), heads, -(-length // 128)) * 0.015 + 0.002
            for name, heads, length in (
                ("q_descale", num_heads[0], max(query_lens)),
                ("k_descale", num_heads[1], max(kv_lens)),
                ("v_descale", num_heads[1], max(kv_lens)),
            )
        }
    else:
        tensors = [torch.randn(shape, dtype=dtype) for shape in shapes]
        descales = {}
    return *tensors, descales


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
    causal: bool = True,
    window_size: Optional[Tuple[int, int]] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    lses: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        # clone to avoid clobbering the query tensor
        q = query[start_idx : start_idx + query_len].clone()
        if q_descale is not None:
            q_blocks = torch.arange(query_len, device=q.device) // 128
            q = q.float() * q_descale[i][:, q_blocks].transpose(0, 1).unsqueeze(-1)
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]
        if k_descale is not None:
            kv_blocks = torch.arange(kv_len, device=q.device) // 128
            k = k.float() * k_descale[i][:, kv_blocks].transpose(0, 1).unsqueeze(-1)
            v = v.float() * v_descale[i][:, kv_blocks].transpose(0, 1).unsqueeze(-1)

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k)
        empty_mask = torch.ones(query_len, kv_len, device=q.device)
        mask = (
            torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
            if causal
            else torch.zeros_like(empty_mask, dtype=torch.bool)
        )
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
        if window_size is not None or alibi_slopes is not None:
            position = torch.arange(query_len, device=q.device) + kv_len - query_len
            columns = torch.arange(kv_len, device=q.device)
            if window_size is not None:
                left, right = window_size
                if left >= 0:
                    mask |= columns < position[:, None] - left
                if right >= 0:
                    mask |= columns > position[:, None] + right
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        if alibi_slopes is not None:
            slopes = alibi_slopes if alibi_slopes.ndim == 1 else alibi_slopes[i]
            attn -= slopes[:, None, None] * (position[:, None] - columns).abs()
        attn.masked_fill_(mask, float("-inf"))

        if attn_bias is not None:
            attn = attn + attn_bias[i, :, :query_len, :kv_len]

        if return_softmax_lse:
            lse = attn.logsumexp(dim=-1)
            lses.append(lse.masked_fill(mask.all(dim=-1), float("inf")))
        attn = torch.softmax(attn, dim=-1).nan_to_num().to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    output = torch.cat(outputs, dim=0)
    return (output, torch.cat(lses, dim=1)) if return_softmax_lse else output


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2815: Not supported")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2816: Not working")
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (5, 18), (129, 463)]])
@pytest.mark.parametrize("dtype,num_heads,head_size,optimize_init", ATTENTION_CONFIGS)
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", [32768, 2048])
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
    if dtype != torch.int8 and alibi is True and soft_cap is not None:
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
        query, key_cache, value_cache, descales = attention_inputs(
            query_lens, kv_lens, num_heads, head_size, block_size, num_blocks, dtype
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

        if vendor_name == "cambricon":
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
                fa_version=2,
                return_softmax_lse=dtype == torch.int8,
                **descales,
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
                    return_softmax_lse=dtype == torch.int8,
                    **descales,
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
                    return_softmax_lse=dtype == torch.int8,
                    **descales,
                )

        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
            attn_bias=attn_bias if dtype != torch.int8 else None,
            alibi_slopes=alibi_slopes if dtype == torch.int8 else None,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            return_softmax_lse=dtype == torch.int8,
            **descales,
        )

        if dtype == torch.int8:
            output, lse = output
            ref_output, ref_lse = ref_output
            torch.testing.assert_close(
                output.float(), ref_output, atol=0.025, rtol=0.025
            )
            torch.testing.assert_close(lse, ref_lse, atol=2e-5, rtol=2e-5)
        else:
            msg = f"{torch.max(torch.abs(output - ref_output))}"
            torch.testing.assert_close(
                output, ref_output, atol=2e-2, rtol=1e-2, msg=msg
            )


@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2815: Not working")
@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2816: Not working")
@pytest.mark.flash_attn_varlen_func
@pytest.mark.parametrize("seq_lens", [[(1, 1328), (1, 18), (1, 463)]])
@pytest.mark.parametrize("num_heads", [(8, 2)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", DTYPES)
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
        query, key_cache, value_cache, descales = attention_inputs(
            query_lens, kv_lens, num_heads, head_size, block_size, num_blocks, dtype
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

        if vendor_name == "cambricon":
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
                fa_version=2,
                return_softmax_lse=dtype == torch.int8,
                **descales,
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
                return_softmax_lse=dtype == torch.int8,
                **descales,
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
            return_softmax_lse=dtype == torch.int8,
            **descales,
        )

        if dtype == torch.int8:
            output, lse = output
            ref_output, ref_lse = ref_output
            torch.testing.assert_close(
                output.float(), ref_output, atol=0.025, rtol=0.025
            )
            torch.testing.assert_close(lse, ref_lse, atol=2e-5, rtol=2e-5)
        else:
            torch.testing.assert_close(
                output, ref_output, atol=2e-2, rtol=1e-2
            ), f"{torch.max(torch.abs(output - ref_output))}"


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name != "metax", reason="INT8 dtype coverage")
@pytest.mark.parametrize(
    "output_dtype",
    [
        pytest.param(torch.float16, id="int8-fp16-output"),
        pytest.param(torch.bfloat16, id="int8-bf16-output"),
    ],
)
@pytest.mark.parametrize(
    "qlens,klens,options",
    [
        ([31, 65], [63, 129], dict(dim=8, causal=True)),
        ([129, 513], [257, 769], dict(dim=96, window=(-1, 17), cap=3)),
        ([129, 513], [257, 769], dict(dim=96, window=(31, -1))),
        ([1024, 513], [1024, 769], dict(dim=192, causal=True)),
        ([513, 129], [513, 257], dict(dim=256, causal=True)),
        ([0], [0], {}),
        ([131, 0, 3], [1, 0, 0], dict(causal=True)),
        (
            [129, 0, 3],
            [0, 0, 0],
            dict(dim=128, heads=8, kvheads=2, causal=True, paged=True),
        ),
        (
            [0, 1, 16, 17, 513, 2, 0],
            [0, 33, 257, 19, 1025, 65, 0],
            dict(dim=128, heads=8, kvheads=2, causal=True, paged=True, strided=True),
        ),
        (
            [1, 4, 8, 16],
            [1, 17, 129, 257],
            dict(heads=6, kvheads=2, paged=True, window=(17, 3), cap=5),
        ),
        (
            [129, 513],
            [257, 1025],
            dict(dim=128, heads=8, kvheads=2, paged=True, broadcast_scales=True),
        ),
    ],
)
def test_flash_attn_varlen_func_boundaries(qlens, klens, options, output_dtype):
    _run_case(qlens, klens, dtype=output_dtype, **options)


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(vendor_name != "metax", reason="INT8 dtype coverage")
@pytest.mark.parametrize(
    "window,cap",
    [
        pytest.param((31, -1), 0, id="int8-left"),
        pytest.param((-1, 17), 3, id="int8-right"),
    ],
)
@pytest.mark.parametrize("head_size", [64, 96, 128])
@pytest.mark.parametrize("query_length", [128, 256])
def test_flash_attn_varlen_func_single_sided_window(
    window, cap, head_size, query_length
):
    qlens, klens = [query_length] * 2, [256] * 2
    q, qs, qr, cuq = _inputs(qlens, 4, head_size)
    k, ks, kr, cuk = _inputs(klens, 4, head_size)
    v, vs, vr, _ = _inputs(klens, 4, head_size)
    v, vs, vr = -v, vs * 1.7, -vr * 1.7
    # Omitting LSE exercises the existing dense dispatcher for D64/D128.
    actual = flaggems_vllm.flash_attn_varlen_func(
        q,
        k,
        v,
        query_length,
        cuq,
        256,
        cuk,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
        window_size=window,
        softcap=cap,
    )
    expected, _ = _reference(qr, kr, vr, qlens, klens, False, window, cap)
    torch.testing.assert_close(actual.float(), expected, atol=0.025, rtol=0.025)
