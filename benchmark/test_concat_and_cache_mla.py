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

import pytest
import torch

import flaggems_vllm

from . import base, consts

# Use the vLLM native C++ kernel as the performance baseline when available.
# The pure-Python loop reference performs one .item() per token (triggering a
# CPU-GPU sync each time), which makes it ~10,000x slower than a real GPU
# kernel and produces misleading SpeedUp numbers.
try:
    import vllm._custom_ops  # noqa: F401

    _HAS_VLLM_NATIVE = hasattr(torch.ops._C_cache_ops, "concat_and_cache_mla")
except (ImportError, RuntimeError):
    _HAS_VLLM_NATIVE = False


def _vllm_concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    scale: torch.Tensor | None = None,
) -> None:
    torch.ops._C_cache_ops.concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    )


class ConcatAndCacheMLABenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return None


@pytest.mark.skipif(flaggems_vllm.vendor_name == "hygon", reason="#2885: RuntimeError")
@pytest.mark.concat_and_cache_mla
def test_concat_and_cache_mla():
    def input_kwargs(shape, dtype, device):
        (
            kv_lora_rank,
            qk_rope_head_dim,
            num_tokens,
            block_size,
            num_blocks,
        ) = shape
        total_slots = num_blocks * block_size
        slot_mapping_lst = random.sample(range(total_slots), num_tokens)
        slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

        kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
        k_pe = torch.randn(num_tokens, qk_rope_head_dim, dtype=dtype, device=device)
        entry_size = kv_lora_rank + qk_rope_head_dim

        scale = torch.tensor(0.1, dtype=torch.float32, device=device)

        kv_cache = torch.zeros(
            num_blocks,
            block_size,
            entry_size,
            dtype=dtype,
            device=device,
        )

        yield (
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            {"kv_cache_dtype": "auto", "scale": scale},
        )

    bench = ConcatAndCacheMLABenchmark(
        op_name="concat_and_cache_mla",
        input_fn=input_kwargs,
        torch_op=_vllm_concat_and_cache_mla if _HAS_VLLM_NATIVE else None,
        gems_op=flaggems_vllm.concat_and_cache_mla,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
