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

from . import accuracy_utils as utils

device = flaggems_vllm.device
vendor_name = flaggems_vllm.vendor_name

CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]


def _create_mla_cache(
    num_blocks: int,
    block_size: int,
    entry_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> torch.Tensor:
    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    return torch.zeros(
        num_blocks, block_size, entry_size, dtype=cache_dtype, device=device
    )


# Custom implementation for FP8 conversion (only for testing)
def convert_fp8(
    dst: torch.Tensor, src: torch.Tensor, scale: float, kv_dtype: str
) -> None:
    if kv_dtype == "fp8":
        dst_ = (src / scale).to(torch.float8_e4m3fn).view(dst.dtype)
        dst.copy_(dst_)
    else:
        dst.copy_(src)


@pytest.mark.skipif(vendor_name == "hygon", reason="Issue #2800: RuntimeError")
@pytest.mark.concat_and_cache_mla
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("num_tokens", [42])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_blocks", [8])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16, torch.float])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize(
    "device",
    [flaggems_vllm.device] if vendor_name == "mthreads" else CUDA_DEVICES,
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto"])
@torch.inference_mode()
def test_concat_and_cache_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    with torch.device(device):
        total_slots = num_blocks * block_size
        slot_mapping_lst = random.sample(range(total_slots), num_tokens)
        slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

        kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
        k_pe = torch.randn(num_tokens, qk_rope_head_dim, dtype=dtype, device=device)
        entry_size = kv_lora_rank + qk_rope_head_dim

        scale = torch.tensor(0.1, dtype=torch.float32, device=device)
        kv_cache = _create_mla_cache(
            num_blocks, block_size, entry_size, dtype, kv_cache_dtype, device
        )
        ref_temp = utils.to_reference(
            torch.zeros(*kv_cache.shape, dtype=dtype, device=device)
        )

        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            block_idx = slot // block_size
            block_offset = slot % block_size
            ref_temp[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
            ref_temp[block_idx, block_offset, kv_lora_rank:] = k_pe[i]

        if kv_cache_dtype == "fp8":
            ref_kv_cache = utils.to_reference(
                torch.empty_like(ref_temp, dtype=kv_cache.dtype)
            )
            convert_fp8(ref_kv_cache, ref_temp, scale.item(), kv_dtype=kv_cache_dtype)
        else:
            ref_kv_cache = utils.to_reference(ref_temp)
        with flaggems_vllm.use_gems():
            flaggems_vllm.concat_and_cache_mla(
                kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
            )

        if kv_cache_dtype == "fp8":
            result_temp = torch.empty_like(kv_cache, dtype=torch.uint8)
            convert_fp8(
                result_temp,
                kv_cache.contiguous(),
                scale.item(),
                kv_dtype=kv_cache_dtype,
            )
            expected_temp = utils.to_reference(
                torch.empty_like(ref_kv_cache, dtype=torch.uint8)
            )
            convert_fp8(
                expected_temp,
                ref_kv_cache,
                scale.item(),
                kv_dtype=kv_cache_dtype,
            )
            dtype = torch.float8_e4m3fn
            # TODO: RuntimeError: Comparing
            # maybe a bug in torch.testing.assert_close
            # utils.gems_assert_close(kv_cache.view(dtype), ref_kv_cache.view(dtype), dtype)
            torch.testing.assert_close(result_temp, expected_temp, atol=0.001, rtol=0.1)
        else:
            utils.gems_assert_close(kv_cache, ref_kv_cache, kv_cache.dtype)


@pytest.mark.concat_and_cache_mla
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16, torch.float])
@pytest.mark.parametrize("tokens", [0, 1, 4])
@pytest.mark.parametrize("rope_stride", [(0, 0), (1536, 1), (9, 17)])
@pytest.mark.parametrize("singleton_head", [False, True])
@torch.inference_mode()
def test_empty_rope_strides(dtype, tokens, rope_stride, singleton_head):
    # Match the zero-stride empty RoPE emitted by Inductor, and also allow other
    # unused strides. The latent slice retains its larger parent row stride.
    kv_c = torch.randn(tokens, 1536, dtype=dtype, device=device)[:, -512:]
    k_pe = torch.empty_strided((tokens, 0), rope_stride, dtype=dtype, device=device)
    slots = torch.tensor([3, -1, 6, 0][:tokens], dtype=torch.int64, device=device)
    cache = torch.full((3, 4, 512), -17, dtype=dtype, device=device)
    expected = cache.clone()
    for row, slot in enumerate(slots.tolist()):
        if slot >= 0:
            expected.view(-1, 512)[slot].copy_(kv_c[row])
    cache_view = cache.unsqueeze(2) if singleton_head else cache
    flaggems_vllm.concat_and_cache_mla(
        kv_c, k_pe, cache_view, slots, "auto", torch.ones((), device=device)
    )
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


@pytest.mark.concat_and_cache_mla
@pytest.mark.parametrize("strided_input", ["kv", "rope", "cache", "slots"])
@torch.inference_mode()
def test_nonempty_feature_stride_rejected(strided_input):
    kv_c = torch.randn(4, 512, dtype=torch.bfloat16, device=device)
    k_pe = torch.randn(4, 64, dtype=kv_c.dtype, device=device)
    cache = torch.full((3, 4, 576), -17, dtype=kv_c.dtype, device=device)
    slots = torch.tensor([3, -1, 6, 0], dtype=torch.int64, device=device)
    if strided_input == "kv":
        kv_c = torch.empty_strided(
            kv_c.shape, (1024, 2), dtype=kv_c.dtype, device=device
        )
    elif strided_input == "rope":
        k_pe = torch.empty_strided(
            k_pe.shape, (128, 2), dtype=k_pe.dtype, device=device
        )
    elif strided_input == "cache":
        cache = torch.full((3, 4, 1152), -17, dtype=kv_c.dtype, device=device)[..., ::2]
    else:
        slots = torch.tensor(
            [3, 0, -1, 0, 6, 0, 0, 0], dtype=slots.dtype, device=device
        )[::2]
    before = cache.clone()
    with pytest.raises(ValueError, match="matching contiguous feature dimensions"):
        flaggems_vllm.concat_and_cache_mla(
            kv_c, k_pe, cache, slots, "auto", torch.ones((), device=device)
        )
    torch.testing.assert_close(cache, before, rtol=0, atol=0)
