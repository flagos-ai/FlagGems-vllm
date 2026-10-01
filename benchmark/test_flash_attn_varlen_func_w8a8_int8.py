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

from .test_flash_attn_varlen_func import FlashAttnVarlenBenchmark, prepare_int8_inputs

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

vendor_name = flaggems_vllm.vendor_name


class FlashAttnVarlenInt8Benchmark(FlashAttnVarlenBenchmark):
    """vLLM v0.19.0 standard attention workloads with the Gems benchmark runner."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.int8_shapes()

    def get_input_iter(self, dtype):
        for bf16_args in super().get_input_iter(dtype):
            q, k, v = bf16_args[:3]
            batch = bf16_args[4].numel() - 1
            if vendor_name == "thead":
                # vLLM creates scheduler metadata before attention. Exclude its
                # creation and input quantization from both timed calls.
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
                metadata_kwargs = {"scheduler_metadata": scheduler_metadata}
            else:
                metadata_kwargs = {}
            bf16_args = (
                *bf16_args[:-1],
                dict(bf16_args[-1], **metadata_kwargs),
            )
            bf16_args, int8_args, reference_args = prepare_int8_inputs(bf16_args)
            if vendor_name == "thead":
                baseline = _varlen_fa3_baseline(reference_args, int8_args)
            else:
                baseline = _varlen_bf16_baseline(reference_args, int8_args)
            torch.testing.assert_close(
                _varlen_int8(bf16_args, int8_args),
                baseline,
                atol=0.03,
                rtol=0.03,
            )
            yield bf16_args, int8_args


def _varlen_bf16_baseline(bf16_args, int8_args):
    return flaggems_vllm.flash_attn_varlen_func(*bf16_args[:-1], **bf16_args[-1])


def _varlen_fa3_baseline(bf16_args, int8_args):
    return vllm_flash_attn_varlen_func(
        *bf16_args[:-1],
        scheduler_metadata=bf16_args[-1]["scheduler_metadata"],
        fa_version=3,
    )


def _varlen_int8(bf16_args, int8_args):
    return flaggems_vllm.flash_attn_varlen_func(*int8_args[:-1], **int8_args[-1])


@pytest.mark.skipif(vendor_name not in ("hygon", "thead"), reason="Hygon/PPU API")
@pytest.mark.flash_attn_varlen_func_w8a8_int8
def test_flash_attn_varlen_func_w8a8_int8():
    if vendor_name == "thead":
        if vllm_flash_attn_varlen_func is None or not is_fa_version_supported(3):
            pytest.skip("PPU vLLM FA3 is unavailable")
        print("Baseline: vLLM BF16 FA3; scheduler setup and quantization excluded.")
        baseline = _varlen_fa3_baseline
    else:
        print(
            "Baseline: FlagGems-vllm BF16; input quantization is excluded from timing."
        )
        baseline = _varlen_bf16_baseline
    bench = FlashAttnVarlenInt8Benchmark(
        op_name="flash_attn_varlen_func_w8a8_int8",
        torch_op=baseline,
        gems_op=_varlen_int8,
        dtypes=[torch.bfloat16],
    )
    bench.run()
