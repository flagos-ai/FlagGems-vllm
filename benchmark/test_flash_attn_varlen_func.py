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
from itertools import accumulate
from typing import Any, List, Optional

import pytest
import torch

import flaggems_vllm

from . import base, utils
from .conftest import Config

vendor_name = flaggems_vllm.vendor_name
DESCALE_BLOCK = 128


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

    @staticmethod
    def int8_shapes():
        # vllm/benchmarks/attention_benchmarks/configs/standard_attention.yaml
        # Each group is (request count, query length, total KV length).
        # Keep all 18 workloads in both core and comprehensive modes.
        workloads = [
            ((1, 512, 512),),
            ((1, 2048, 2048),),
            ((1, 4096, 4096),),
            ((1, 8192, 8192),),
            ((8, 1, 1024),),
            ((16, 1, 2048),),
            ((32, 1, 1024),),
            ((64, 1, 4096),),
            ((2, 2048, 2048), (8, 1, 1024)),
            ((4, 1024, 1024), (16, 1, 2048)),
            ((2, 4096, 4096), (32, 1, 1024)),
            ((16, 2, 1024),),
            ((16, 4, 1024),),
            ((16, 8, 1024),),
            ((32, 4, 2048),),
            ((8, 8, 4096),),
            ((1, 1024, 2048),),
            ((2, 1024, 4096),),
        ]
        shapes = []
        for groups in workloads:
            qlens = tuple(q for count, q, kv in groups for _ in range(count))
            klens = tuple(kv for count, q, kv in groups for _ in range(count))
            cuq = (0, *accumulate(qlens))
            num_blocks = len(klens) * ((max(klens) + 15) // 16)
            shapes.append((cuq, klens, 32, 8, 128, 16, num_blocks, False, None))

        return shapes

    def set_shapes(self, shape_file_path: Optional[List[Any]] = None):
        if self.to_bench_dtypes == [torch.int8]:
            self.shapes = self.int8_shapes()
            return
        # Collected from Qwen3.6-35B-A3B: six TP1 cases followed by six TP4 cases.
        all_cu_seq_lens_q = [
            # TP1: short prefill (qwen36_tp1_p1024d1024_l0002).
            (0, 1035),
            # TP1: batched decode (qwen36_tp1_p1024d1024_l0081).
            tuple(range(257)),
            # TP1: mixed short KV (qwen36_tp1_p4096d1024_l0002).
            (0, 1, 2, 3, 4, 46, 4152, 8258, 12364, 16384),
            # TP1: mixed long KV (qwen36_tp1_p32768d1024_l0005).
            (0, 12, 16384),
            # TP1: short query, long KV (qwen36_tp1_p65536d6144_l0018).
            (0, 1, 2, 3, 67),
            # TP1: large query, long KV (qwen36_tp1_p65536d6144_l0005).
            (0, 16384),
            # TP4: short prefill (qwen36_tp4_p1024d1024_l0146).
            (0, 1036),
            # TP4: batched decode (qwen36_tp4_p1024d1024_l0272).
            tuple(range(513)),
            # TP4: mixed short KV (qwen36_tp4_p1024d1024_l0173).
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
            # TP4: mixed long KV (qwen36_tp4_p65536d6144_l0139).
            (0, 12, 16384),
            # TP4: short query, long KV (qwen36_tp4_p32768d1024_l0143).
            (0, 12),
            # TP4: large query, long KV (qwen36_tp4_p65536d6144_l0138).
            (0, 16384),
        ]
        all_seqused_k = [
            (1035,),
            (1,) * 256,
            (4110, 4108, 4107, 4107, 4106, 4106, 4106, 4106, 4020),
            (32780, 16372),
            (65560, 65555, 65550, 65546),
            (65536,),
            (1036,),
            (32,) * 512,
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
            (65548, 16372),
            (32780,),
            (65536,),
        ]
        all_num_heads = [(16, 2)] * 6 + [(4, 1)] * 6
        all_block_sizes = [32] * 6 + [16] * 6
        all_num_blocks = [73920] * 6 + [605550, 16896] + [605550] * 4

        head_dim = 256
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
            for (
                cu_seq_lens_q,
                seqused_k,
                (num_heads, num_heads_k),
                block_size,
                num_blocks,
            ) in zip(
                all_cu_seq_lens_q,
                all_seqused_k,
                all_num_heads,
                all_block_sizes,
                all_num_blocks,
            )
        ]

        self.shapes = all_configs

    def get_input_iter(self, dtype):
        if dtype != torch.int8:
            for config in self.shapes:
                yield self.flash_attn_varlen_input_fn(config, dtype, self.device)
            return

        torch.manual_seed(0)
        shapes = (
            self.shapes if self.to_bench_dtypes == [torch.int8] else self.int8_shapes()
        )
        for config in shapes:
            bf16_args = list(
                self.flash_attn_varlen_input_fn(config, torch.bfloat16, self.device)
            )
            # Native FA2 needs cumulative lengths even for a paged cache.
            bf16_args[6] = torch.tensor(
                (0, *accumulate(config[1])), dtype=torch.int32, device=self.device
            )
            bf16_args, int8_args, reference_args = prepare_int8_inputs(bf16_args)
            torch.testing.assert_close(
                self.gems_op(bf16_args, int8_args),
                self.torch_op(reference_args, int8_args),
                atol=0.03,
                rtol=0.03,
            )
            yield bf16_args, int8_args

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


def prepare_int8_inputs(bf16_args):
    bf16_args = list(bf16_args)
    q, k, v = bf16_args[:3]
    batch = bf16_args[4].numel() - 1
    # Match vLLM's sequential cache blocks, including unused shorter-request slots.
    bf16_args[17] = torch.arange(
        k.shape[0], dtype=torch.int32, device=k.device
    ).reshape(batch, -1)
    quantized, descales, dequantized = [], [], []
    for tensor, max_len in ((q, bf16_args[3]), (k, bf16_args[5]), (v, bf16_args[5])):
        # Broadcast head scales keep shared physical pages consistent across requests.
        axes = tuple(i for i in range(tensor.ndim) if i != tensor.ndim - 2)
        scale = tensor.float().abs().amax(axes).clamp_min(1e-8) / 127
        quant = (
            (tensor.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        )
        quantized.append(quant)
        dequantized.append((quant.float() * scale[:, None]).to(q.dtype))
        descales.append(
            scale[None, :, None].expand(
                batch, tensor.shape[-2], -(-max_len // DESCALE_BLOCK)
            )
        )
    int8_args = list(bf16_args)
    int8_args[:3] = quantized
    int8_args[6] = None
    int8_args[19] = torch.empty_like(q)
    int8_args[-1] = dict(
        bf16_args[-1],
        q_descale=descales[0],
        k_descale=descales[1],
        v_descale=descales[2],
        scheduler_metadata=None,
        fa_version=2,
    )
    reference_args = (*dequantized, *bf16_args[3:])
    return tuple(bf16_args), tuple(int8_args), reference_args


def flash_attn_varlen_gems(*args, **kwargs):
    if len(args) == 2:
        _, int8_args = args
        return flaggems_vllm.ops.flash_attn_varlen_func(
            *int8_args[:-1], **int8_args[-1]
        )
    return flaggems_vllm.ops.flash_attn_varlen_func(*args, **kwargs)


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

    The INT8 benchmark prepares cumulative lengths before timing. Floating-point
    workloads retain their existing timed conversion. Native FA2 chooses splits.
    """
    if len(args) == 2:
        baseline_args, _ = args
        from vllm_metax.v1.attention.backends.fa_utils import flash_attn_varlen_func
    else:
        baseline_args = args
        from flash_attn import flash_attn_varlen_func
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
    ) = baseline_args

    if cu_seqlens_k is None:
        cu_seqlens_k = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=seqused_k.device),
                torch.cumsum(seqused_k, dim=0),
            ]
        ).to(torch.int32)

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
@pytest.mark.parametrize(
    "input_dtype",
    [pytest.param(None, id="floating")]
    + ([pytest.param(torch.int8, id="int8")] if vendor_name == "metax" else []),
)
def test_flash_attn_varlen_func(monkeypatch, input_dtype):
    if input_dtype is None:
        dtypes = [torch.float16, torch.bfloat16]
    else:
        if Config.user_desired_dtypes and input_dtype not in Config.user_desired_dtypes:
            pytest.skip("INT8 benchmark excluded by --dtypes")
        pytest.importorskip("vllm_metax.v1.attention.backends.fa_utils")
        dtypes = [input_dtype]
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
        gems_op=(
            flash_attn_varlen_gems
            if vendor_name == "metax"
            else flaggems_vllm.ops.flash_attn_varlen_func
        ),
        dtypes=dtypes,
    )
    bench.run()
