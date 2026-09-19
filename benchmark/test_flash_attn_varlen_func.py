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
from functools import wraps
from typing import Any, List, NamedTuple, Optional

import pytest
import torch

import flaggems_vllm

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


class _FlashAttnVarlenConfig(NamedTuple):
    cu_query_lens: tuple
    seqused_k: tuple
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    num_blocks: int
    alibi: bool
    soft_cap: Optional[float]
    max_seqlen_q: Optional[int] = None
    max_seqlen_k: Optional[int] = None
    block_table_width: Optional[int] = None
    num_splits: int = 0


def _qwen36_configs():
    # Qwen3.6-35B-A3B TP1/TP4 trace shapes. Preserve full KV caches and
    # sequence-length upper bounds; page-table values are generated below.
    # MetaX also accepts the recorded split hints through the FA2 interface.
    return [
        # TP1: short prefill (qwen36_tp1_p1024d1024_l0002).
        _FlashAttnVarlenConfig(
            (0, 1035),
            (1035,),
            16,
            2,
            256,
            32,
            73920,
            False,
            None,
            1035,
            1035,
            2310,
            0,
        ),
        # TP1: batched decode, loose KV upper bound (qwen36_tp1_p1024d1024_l0081).
        _FlashAttnVarlenConfig(
            tuple(range(257)),
            (1,) * 256,
            16,
            2,
            256,
            32,
            73920,
            False,
            None,
            1,
            73728,
            2310,
            32,
        ),
        # TP1: mixed short KV (qwen36_tp1_p4096d1024_l0002).
        _FlashAttnVarlenConfig(
            (0, 1, 2, 3, 4, 46, 4152, 8258, 12364, 16384),
            (4110, 4108, 4107, 4107, 4106, 4106, 4106, 4106, 4020),
            16,
            2,
            256,
            32,
            73920,
            False,
            None,
            4106,
            4110,
            2310,
            0,
        ),
        # TP1: mixed long KV (qwen36_tp1_p32768d1024_l0005).
        _FlashAttnVarlenConfig(
            (0, 12, 16384),
            (32780, 16372),
            16,
            2,
            256,
            32,
            73920,
            False,
            None,
            16372,
            32780,
            2310,
            0,
        ),
        # TP1: short query, long KV (qwen36_tp1_p65536d6144_l0018).
        _FlashAttnVarlenConfig(
            (0, 1, 2, 3, 67),
            (65560, 65555, 65550, 65546),
            16,
            2,
            256,
            32,
            73920,
            False,
            None,
            64,
            65560,
            2310,
            32,
        ),
        # TP1: large query, long KV (qwen36_tp1_p65536d6144_l0005).
        _FlashAttnVarlenConfig(
            (0, 16384),
            (65536,),
            16,
            2,
            256,
            32,
            73920,
            False,
            None,
            16384,
            65536,
            2310,
            0,
        ),
        # TP4: short prefill (qwen36_tp4_p1024d1024_l0146).
        _FlashAttnVarlenConfig(
            (0, 1036),
            (1036,),
            4,
            1,
            256,
            16,
            605550,
            False,
            None,
            1036,
            1036,
            4620,
            0,
        ),
        # TP4: batched decode, loose KV upper bound (qwen36_tp4_p1024d1024_l0272).
        _FlashAttnVarlenConfig(
            tuple(range(513)),
            (32,) * 512,
            4,
            1,
            256,
            16,
            16896,
            False,
            None,
            1,
            73728,
            4620,
            32,
        ),
        # TP4: mixed short KV (qwen36_tp4_p1024d1024_l0173).
        _FlashAttnVarlenConfig(
            tuple(range(17))
            + (
                182,
                1217,
                2253,
                3287,
                4321,
                5355,
                6390,
                7424,
                8458,
                9494,
                10528,
                11562,
                12596,
                13631,
                14667,
                15702,
                16384,
            ),
            (
                1038,
                1035,
                1035,
                1037,
                1035,
                1035,
                1035,
                1035,
                1035,
                1035,
                1036,
                1035,
                1037,
                1035,
                1035,
                1035,
                1034,
                1035,
                1036,
                1034,
                1034,
                1034,
                1035,
                1034,
                1034,
                1036,
                1034,
                1034,
                1034,
                1035,
                1036,
                1035,
                682,
            ),
            4,
            1,
            256,
            16,
            605550,
            False,
            None,
            1036,
            1038,
            4620,
            0,
        ),
        # TP4: mixed long KV (qwen36_tp4_p65536d6144_l0139).
        _FlashAttnVarlenConfig(
            (0, 12, 16384),
            (65548, 16372),
            4,
            1,
            256,
            16,
            605550,
            False,
            None,
            16372,
            65548,
            4620,
            0,
        ),
        # TP4: short query, long KV (qwen36_tp4_p32768d1024_l0143).
        _FlashAttnVarlenConfig(
            (0, 12),
            (32780,),
            4,
            1,
            256,
            16,
            605550,
            False,
            None,
            12,
            32780,
            4620,
            32,
        ),
        # TP4: large query, long KV (qwen36_tp4_p65536d6144_l0138).
        _FlashAttnVarlenConfig(
            (0, 16384),
            (65536,),
            4,
            1,
            256,
            16,
            605550,
            False,
            None,
            16384,
            65536,
            4620,
            0,
        ),
    ]


class FlashAttnVarlenBenchmark(base.Benchmark):
    """
    benchmark for flash_attn_varlen_func
    """

    def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
        self.shapes = _qwen36_configs()

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield self.flash_attn_varlen_input_fn(config, dtype, self.device)

    def flash_attn_varlen_input_fn(self, config, dtype, device):
        """Input function for flash attention varlen benchmark"""
        config = _FlashAttnVarlenConfig(*config)
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
        ) = config[:9]

        if alibi is True and soft_cap is not None:
            return

        num_seqs = len(cu_query_lens) - 1
        max_query_len = max(
            map(lambda x, y: x - y, cu_query_lens[1:], cu_query_lens[:-1])
        )
        max_kv_len = max(seqused_k)
        if config.max_seqlen_q is not None:
            assert config.max_seqlen_q >= max_query_len
            max_query_len = config.max_seqlen_q
        if config.max_seqlen_k is not None:
            assert config.max_seqlen_k >= max_kv_len
            max_kv_len = config.max_seqlen_k
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
            if config.block_table_width is not None:
                assert config.block_table_width >= max_num_blocks_per_seq
                max_num_blocks_per_seq = config.block_table_width
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
                "num_splits": config.num_splits if vendor_name == "metax" else 0,
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


def flash_attn_varlen_metax(*args, **kwargs):
    """Adapt vLLM arguments to MetaX FlashAttention's paged-KV interface.

    Cumulative-length conversion is included in the baseline timing. The
    native interface selects its own splits and does not accept ``out``.
    """
    (
        query,
        key_cache,
        value_cache,
        max_query_len,
        cu_query_lens,
        max_kv_len,
        cu_seqlens_k,
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
        _,
        *_,
    ) = args

    if cu_seqlens_k is None:
        cu_seqlens_k = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=seqused_k.device),
                torch.cumsum(seqused_k, dim=0),
            ]
        ).to(torch.int32)

    from flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func(
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        cu_seqlens_k,
        max_query_len,
        max_kv_len,
        dropout_p=dropout_p,
        softmax_scale=scale,
        causal=causal,
        window_size=tuple(window_size),
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        softcap=float(soft_cap),
        block_table=block_tables,
    )


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

    if vendor_name == "metax":
        flash_attn_varlen_func = flash_attn_varlen_metax
    elif vendor_name == "iluvatar":
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
