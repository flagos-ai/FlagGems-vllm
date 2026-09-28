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

from itertools import accumulate

import pytest
import torch

import flaggems_vllm

from .test_flash_attn_varlen_func import FlashAttnVarlenBenchmark

try:
    from vllm.vllm_flash_attn.flash_attn_interface import (
        flash_attn_varlen_func as vllm_flash_attn_varlen_func,
    )
    from vllm.vllm_flash_attn.flash_attn_interface import (
        get_scheduler_metadata,
        is_fa_version_supported,
    )
except ImportError:
    vllm_flash_attn_varlen_func = None
    get_scheduler_metadata = None
    is_fa_version_supported = None

DESCALE_BLOCK = 128

vendor_name = flaggems_vllm.vendor_name


class FlashAttnVarlenInt8Benchmark(FlashAttnVarlenBenchmark):
    """vLLM v0.19.0 standard attention workloads with the Gems benchmark runner."""

    def set_shapes(self, shape_file_path=None):
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
        self.shapes = []
        for groups in workloads:
            qlens = tuple(q for count, q, kv in groups for _ in range(count))
            klens = tuple(kv for count, q, kv in groups for _ in range(count))
            cuq = (0, *accumulate(qlens))
            num_blocks = len(klens) * ((max(klens) + 15) // 16)
            self.shapes.append((cuq, klens, 32, 8, 128, 16, num_blocks, False, None))

    def flash_attn_varlen_input_fn(self, config, dtype, device):
        args = list(super().flash_attn_varlen_input_fn(config, dtype, device))
        # Match vLLM runner._build_common_attn_metadata: distinct sequential
        # cache blocks per request, including unused slots for shorter requests.
        batch = len(config[1])
        args[17] = torch.arange(config[6], dtype=torch.int32, device=device).reshape(
            batch, -1
        )
        return tuple(args)

    def get_input_iter(self, dtype):
        for bf16_args in super().get_input_iter(dtype):
            q, k, v = bf16_args[:3]
            batch = bf16_args[4].numel() - 1
            # vLLM creates this metadata before attention. Exclude its creation
            # and input quantization from both timed calls.
            scheduler_metadata = get_scheduler_metadata(
                batch_size=batch,
                max_seqlen_q=bf16_args[3],
                max_seqlen_k=bf16_args[5],
                num_heads_q=q.shape[-2],
                num_heads_kv=k.shape[-2],
                headdim=q.shape[-1],
                cache_seqlens=bf16_args[7],
                qkv_dtype=dtype,
                cu_seqlens_q=bf16_args[4],
                page_size=k.shape[1],
                causal=bf16_args[11],
                window_size=bf16_args[12] or (-1, -1),
                num_splits=0,
            )
            bf16_args = (
                *bf16_args[:-1],
                dict(bf16_args[-1], scheduler_metadata=scheduler_metadata),
            )
            quantized, descales, dequantized = [], [], []
            for x, max_len in ((q, bf16_args[3]), (k, bf16_args[5]), (v, bf16_args[5])):
                # A head-wise scale shared by logical blocks also handles shared
                # physical cache pages in the upstream random block tables.
                axes = tuple(i for i in range(x.ndim) if i != x.ndim - 2)
                scale = x.float().abs().amax(axes).clamp_min(1e-8) / 127
                quantized.append(
                    (x.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
                )
                dequantized.append((quantized[-1].float() * scale[:, None]).to(dtype))
                descales.append(
                    scale[None, :, None].expand(
                        batch, x.shape[-2], -(-max_len // DESCALE_BLOCK)
                    )
                )
            int8_args = list(bf16_args)
            int8_args[:3] = quantized
            int8_args[19] = torch.empty_like(q)
            int8_args[-1] = dict(
                bf16_args[-1],
                q_descale=descales[0],
                k_descale=descales[1],
                v_descale=descales[2],
                scheduler_metadata=None,
                fa_version=2,
            )
            int8_args = tuple(int8_args)
            # Check the same quantized values, separating kernel error from
            # input quantization error. Timing still uses the original BF16 inputs.
            reference_args = (*dequantized, *bf16_args[3:])
            torch.testing.assert_close(
                _varlen_int8(bf16_args, int8_args),
                _varlen_fa3_baseline(reference_args, int8_args),
                atol=0.03,
                rtol=0.03,
            )
            yield bf16_args, int8_args


def _varlen_fa3_baseline(bf16_args, int8_args):
    return vllm_flash_attn_varlen_func(
        *bf16_args[:-1],
        scheduler_metadata=bf16_args[-1]["scheduler_metadata"],
        fa_version=3,
    )


def _varlen_int8(bf16_args, int8_args):
    return flaggems_vllm.flash_attn_varlen_func(*int8_args[:-1], **int8_args[-1])


@pytest.mark.skipif(vendor_name != "thead", reason="PPU-only API")
@pytest.mark.flash_attn_varlen_func_w8a8_int8
def test_flash_attn_varlen_func_w8a8_int8():
    if vllm_flash_attn_varlen_func is None or not is_fa_version_supported(3):
        pytest.skip("PPU vLLM FA3 is unavailable")
    print("Baseline: vLLM BF16 FA3; scheduler setup and quantization excluded.")
    bench = FlashAttnVarlenInt8Benchmark(
        op_name="flash_attn_varlen_func_w8a8_int8",
        torch_op=_varlen_fa3_baseline,
        gems_op=_varlen_int8,
        dtypes=[torch.bfloat16],
    )
    bench.run()
