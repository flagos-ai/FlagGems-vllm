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

import inspect
import math
from functools import wraps
from typing import Any, List, Optional

import pytest
import torch
import triton

import flaggems_vllm
from flaggems_vllm.ops import flash_attn_varlen_func_w8a8_fp8 as w8a8_varlen

from . import base, utils

vendor_name = flaggems_vllm.vendor_name


def _with_supported_kwargs(op):
    """Wrap ``op`` while omitting keyword arguments it does not support.

    vLLM has changed the order and availability of the optional FlashAttention
    arguments following ``out`` across releases.  Inspecting the signature up
    front avoids relying on a ``TypeError`` fallback, which could accidentally
    hide a real error raised from inside the operator.
    """
    try:
        parameters = inspect.signature(op).parameters
    except (TypeError, ValueError):
        # Some vendor extension callables do not expose a Python signature.
        # Every kwarg supplied by this benchmark carries the API default, so
        # omitting them is the safest compatibility fallback.
        parameters = {}
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    supported_kwargs = {
        name
        for name, parameter in parameters.items()
        if parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
    }

    @wraps(op)
    def wrapped(*args, **kwargs):
        if not accepts_var_kwargs:
            kwargs = {
                name: value
                for name, value in kwargs.items()
                if name in supported_kwargs
            }
        return op(*args, **kwargs)

    return wrapped


class FlashAttnVarlenBenchmark(base.Benchmark):
    """
    benchmark for flash_attn_varlen_func
    """

    def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
        # Collecting from qwen/Qwen3-1.7B
        # --random-input 512 --random-output 2048 --num-prompts 200 --request-rate inf
        # Format: (cu_seq_lens_q, seqused_k, num_heads, head_size, block_size,
        # num_blocks, alibi, soft_cap)

        all_cu_seq_lens_q = [
            (
                0,
                512,
            ),
            (
                0,
                1,
                2,
                72,
            ),
            tuple(range(0, 45))
            + (
                105,
                121,
                137,
                153,
                169,
                185,
                201,
                217,
                233,
                249,
                265,
            ),
            tuple(range(0, 196))
            + (
                211,
                226,
                240,
                253,
                265,
            ),
        ]
        all_seqused_k = [
            (512,),
            (
                1,
                1,
                70,
            ),
            (515,) + (514,) * 20 + (513,) * 20 + (512,) * 14,
            (2333,)
            + (2331,) * 20
            + (2330,) * 20
            + (2329,) * 14
            + (2328,) * 18
            + (2327,) * 15
            + (2326,) * 17
            + (2325,) * 18
            + (2324,) * 21
            + (2323,) * 22
            + (2322,) * 24
            + (2321,) * 5
            + (
                2320,
                2319,
                2318,
                2317,
                2316,
            ),
        ]

        num_heads = 16
        num_heads_k = 8
        head_dim = 128
        block_size = 16
        num_blocks = 2000
        alibi = False
        soft_cap = None

        all_configs = [
            (
                cu_seq_lens_q,
                seqused_k,
                num_heads,
                num_heads_k,
                head_dim,
                block_size,
                num_blocks,
                alibi,
                soft_cap,
            )
            for cu_seq_lens_q, seqused_k in zip(all_cu_seq_lens_q, all_seqused_k)
        ]

        self.shapes = all_configs

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield self.flash_attn_varlen_input_fn(config, dtype, self.device)

    def flash_attn_varlen_input_fn(self, config, dtype, device):
        """Input function for flash attention varlen benchmark"""
        (
            cu_query_lens,
            seqused_k,
            num_query_heads,
            num_kv_heads,
            head_size,
            block_size,
            num_blocks,
            alibi,
            soft_cap,
        ) = config

        if alibi is True and soft_cap is not None:
            return

        num_seqs = len(cu_query_lens) - 1
        max_query_len = max(
            map(lambda x, y: x - y, cu_query_lens[1:], cu_query_lens[:-1])
        )
        max_kv_len = max(seqused_k)
        window_size = (-1, -1)
        scale = head_size**-0.5

        assert num_seqs == len(seqused_k)

        with torch.device(device):
            query = torch.randn(
                cu_query_lens[-1],
                num_query_heads,
                head_size,
                dtype=dtype,
                device=device,
            )
            out = torch.empty_like(query)
            key_cache = torch.randn(
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
                dtype=dtype,
                device=device,
            )
            value_cache = torch.randn_like(key_cache)
            cu_query_lens = torch.tensor(
                cu_query_lens, dtype=torch.int32, device=device
            )
            seqused_k = torch.tensor(seqused_k, dtype=torch.int32, device=device)

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
                alibi_slopes = (
                    torch.ones(
                        num_seqs,
                        num_query_heads,
                        device=device,
                        dtype=torch.float32,
                    )
                    * 0.3
                )
            else:
                alibi_slopes = None

        return (
            query,
            key_cache,
            value_cache,
            max_query_len,
            cu_query_lens,
            max_kv_len,
            None,
            seqused_k,
            None,
            0.0,
            scale,
            causal,
            window_size,
            soft_cap if soft_cap is not None else 0,
            alibi_slopes,
            False,
            False,
            block_tables,
            False,
            out,
            {
                "scheduler_metadata": None,
                "q_descale": None,
                "k_descale": None,
                "v_descale": None,
                "s_aux": None,
                "num_splits": 0,
                "cp_world_size": 1,
                "cp_rank": 0,
                "cp_tot_seqused_k": None,
                "fa_version": 2,
            },
        )


def flash_attn_varlen_legacy(*args, **kwargs):
    """
    Compatibility wrapper for running old flash_attn_varlen_func.
    """
    (
        query,
        key_cache,
        value_cache,
        max_query_len,
        cu_query_lens,
        max_kv_len,
        _,
        seqused_k,
        _,
        dropout_p,
        scale,
        causal,
        window_size,
        soft_cap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_tables,
        _,
        out,
        *_,
    ) = args

    k_flat = key_cache.reshape(-1, key_cache.shape[2], key_cache.shape[3])
    v_flat = value_cache.reshape(-1, value_cache.shape[2], value_cache.shape[3])
    cu_seqlens_k = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=seqused_k.device),
            torch.cumsum(seqused_k, dim=0),
        ]
    ).to(torch.int32)

    from flash_attn import flash_attn_varlen_func

    result = flash_attn_varlen_func(
        query,  # q
        k_flat,  # k (flattened from key_cache)
        v_flat,  # v (flattened from value_cache)
        cu_query_lens,  # cu_seqlens_q
        cu_seqlens_k,  # cu_seqlens_k (constructed from seqused_k)
        max_query_len,  # max_seqlen_q
        max_kv_len,  # max_seqlen_k
        dropout_p,  # dropout_p
        scale,  # softmax_scale
        causal,  # causal
        tuple(window_size),  # window_size
        float(soft_cap),  # softcap
        alibi_slopes,  # alibi_slopes
        deterministic,  # deterministic
        return_attn_probs,  # return_attn_probs
        block_tables,  # block_table
        alibi_slopes is not None,  # use_alibi (derived from alibi_slopes)
        0,  # alibi_mode
        1,  # imp_mode
        out=out,  # out
        bias=None,  # bias
    )
    return result


@pytest.mark.skipif(
    utils.SkipVersion("vllm", "<0.9"),
    reason="vLLM version prior to 0.9 does not include the flash_attn_varlen_func API.",
)
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.7"),
    reason="Torch version prior to 2.7 is not compatible with VLLM.",
)
@pytest.mark.skipif(vendor_name == "hygon", reason="#2816: RuntimeError")
@pytest.mark.skipif(vendor_name == "cambricon", reason="#2886: TypeError")
@pytest.mark.flash_attn_varlen_func
def test_flash_attn_varlen_func(monkeypatch):
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")

    if vendor_name == "iluvatar":
        # iluvatar does not have updated vllm_flash_attn, use conversion wrapper
        flash_attn_varlen_func = flash_attn_varlen_legacy
    else:
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

        flash_attn_varlen_func = _with_supported_kwargs(flash_attn_varlen_func)

    bench = FlashAttnVarlenBenchmark(
        op_name="flash_attn_varlen_func",
        torch_op=flash_attn_varlen_func,
        gems_op=flaggems_vllm.ops.flash_attn_varlen_func,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()


def _supports_hopper_fp8() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _get_fp8_dtype():
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        pytest.skip("torch.float8_e4m3fn is not available")
    return dtype


def _hadamard_matrix(dim, device):
    assert dim > 0 and dim & (dim - 1) == 0, "head_size must be a power of two"
    matrix = torch.tensor([[1.0]], device=device)
    while matrix.shape[0] < dim:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(dim)


def _apply_incoherent_qk(x):
    matrix = _hadamard_matrix(x.shape[-1], x.device).to(torch.float32)
    return torch.matmul(x.float(), matrix).to(x.dtype)


def _quantize_varlen_per_block_fp8(x, seq_lens, fp8_dtype, block_size=128):
    total_tokens, num_heads, _ = x.shape
    assert total_tokens == sum(seq_lens)
    fp8_max = float(torch.finfo(fp8_dtype).max)
    num_blocks = triton.cdiv(max(seq_lens), block_size)
    quantized = torch.empty_like(x, dtype=fp8_dtype)
    descale = torch.ones(
        (len(seq_lens), num_heads, num_blocks),
        device=x.device,
        dtype=torch.float32,
    )

    token_offset = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        for block_idx in range(triton.cdiv(seq_len, block_size)):
            lo = token_offset + block_idx * block_size
            hi = min(token_offset + seq_len, lo + block_size)
            tile = x[lo:hi].float()
            scale = (tile.abs().amax(dim=(0, 2)) / fp8_max).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            quantized[lo:hi] = torch.clamp(
                tile / scale[None, :, None], -fp8_max, fp8_max
            ).to(fp8_dtype)
            descale[batch_idx, :, block_idx] = scale
        token_offset += seq_len

    return quantized.contiguous(), descale.contiguous()


def _quantize_qkv_w8a8(q, k, v, q_seq_lens, kv_seq_lens):
    fp8_dtype = _get_fp8_dtype()
    q_fp8, q_descale = _quantize_varlen_per_block_fp8(
        _apply_incoherent_qk(q), q_seq_lens, fp8_dtype
    )
    k_fp8, k_descale = _quantize_varlen_per_block_fp8(
        _apply_incoherent_qk(k), kv_seq_lens, fp8_dtype
    )
    v_fp8, v_descale = _quantize_varlen_per_block_fp8(v, kv_seq_lens, fp8_dtype)
    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale


def baseline_flash_attn_varlen_func_w8a8_fp8(
    q,
    k,
    v,
    q_fp8,
    k_fp8,
    v_fp8,
    q_descale,
    k_descale,
    v_descale,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k,
    scale,
    causal,
    baseline_out,
    w8a8_out,
):
    return flaggems_vllm.flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        cu_seqlens_k,
        softmax_scale=scale,
        causal=causal,
        out=baseline_out,
    )


def gems_flash_attn_varlen_func_w8a8_fp8(
    q,
    k,
    v,
    q_fp8,
    k_fp8,
    v_fp8,
    q_descale,
    k_descale,
    v_descale,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k,
    scale,
    causal,
    baseline_out,
    w8a8_out,
):
    return w8a8_varlen(
        q_fp8,
        k_fp8,
        v_fp8,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        cu_seqlens_k,
        softmax_scale=scale,
        causal=causal,
        out=w8a8_out,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )


class FlashAttnVarlenFuncW8A8FP8Benchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        all_shapes = []

        for batch in (1, 2, 4, 8):
            all_shapes.extend(
                [
                    (batch, 512, 16, 128, False),
                    (batch, 512, 32, 64, False),
                    (batch, 512, 16, 128, True),
                    (batch, 512, 32, 64, True),
                ]
            )

        for batch in (1, 2, 4, 8):
            for seq_len in (1024, 2048, 4096, 8192):
                all_shapes.extend(
                    [
                        (batch, seq_len, 16, 128, False),
                        (batch, seq_len, 32, 64, False),
                    ]
                )

        all_shapes.extend(
            [
                (8, 8192, 16, 128, True),
                (8, 8192, 32, 64, True),
            ]
        )

        ragged_q = (32, 128, 512, 4096)
        ragged_kv = (1, 17, 129, 8192)
        for num_heads, head_size in ((32, 64), (16, 128)):
            for causal in (False, True):
                all_shapes.append((ragged_q, ragged_kv, num_heads, head_size, causal))

        core_shapes = [
            (1, 512, 16, 128, False),
            (1, 512, 32, 64, False),
            (2, 512, 16, 128, True),
            (1, 2048, 32, 64, False),
            (4, 4096, 32, 64, False),
            (8, 8192, 16, 128, True),
            (ragged_q, ragged_kv, 32, 64, False),
        ]
        self.shapes = (
            all_shapes
            if base.Config.bench_level == base.consts.BenchLevel.COMPREHENSIVE
            else core_shapes
        )

    def set_more_shapes(self):
        return []


def _make_deterministic_ragged_lengths(batch, max_seq_len):
    if batch == 1:
        return (max_seq_len,), (max_seq_len,)

    step = max(1, max_seq_len // (2 * batch))
    q_seq_lens = tuple(
        max(1, max_seq_len - batch_idx * step - batch_idx % 3)
        for batch_idx in range(batch)
    )
    kv_seq_lens = q_seq_lens[1:] + q_seq_lens[:1]
    return q_seq_lens, kv_seq_lens


def _make_cu_seqlens(seq_lens, device):
    return torch.tensor(
        (0,) + seq_lens,
        device=device,
        dtype=torch.int32,
    ).cumsum(dim=0, dtype=torch.int32)


def flash_attn_varlen_func_w8a8_fp8_input_fn(config, dtype, device):
    batch_or_q_lens, seq_len_or_kv_lens, num_heads, head_size, causal = config
    if isinstance(batch_or_q_lens, (list, tuple)):
        q_seq_lens = tuple(batch_or_q_lens)
        kv_seq_lens = tuple(seq_len_or_kv_lens)
        assert len(q_seq_lens) == len(kv_seq_lens)
    else:
        q_seq_lens, kv_seq_lens = _make_deterministic_ragged_lengths(
            batch_or_q_lens, seq_len_or_kv_lens
        )

    q = torch.empty(
        (sum(q_seq_lens), num_heads, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    k = torch.empty(
        (sum(kv_seq_lens), num_heads, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    v = torch.empty_like(k).uniform_(-0.05, 0.05)
    (
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
    ) = _quantize_qkv_w8a8(q, k, v, q_seq_lens, kv_seq_lens)
    cu_seqlens_q = _make_cu_seqlens(q_seq_lens, device)
    cu_seqlens_k = _make_cu_seqlens(kv_seq_lens, device)
    baseline_out = torch.empty_like(q)
    w8a8_out = torch.empty_like(q)
    torch.cuda.synchronize()

    yield (
        q,
        k,
        v,
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
        max(q_seq_lens),
        cu_seqlens_q,
        max(kv_seq_lens),
        cu_seqlens_k,
        1.0 / math.sqrt(head_size),
        causal,
        baseline_out,
        w8a8_out,
    )


@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.7"),
    reason="Torch version prior to 2.7 is not compatible with vLLM.",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(flaggems_vllm.device == "cpu", reason="Unsupported in CPU mode")
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
@pytest.mark.skipif(
    getattr(torch, "float8_e4m3fn", None) is None,
    reason="FP8 is not available",
)
@pytest.mark.flash_attn_varlen_func_w8a8_fp8
def test_flash_attn_varlen_func_w8a8_fp8():
    bench = FlashAttnVarlenFuncW8A8FP8Benchmark(
        op_name="flash_attn_varlen_func_w8a8_fp8",
        input_fn=flash_attn_varlen_func_w8a8_fp8_input_fn,
        torch_op=baseline_flash_attn_varlen_func_w8a8_fp8,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(gems_flash_attn_varlen_func_w8a8_fp8)
    bench.run()
