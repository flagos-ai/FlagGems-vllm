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

import importlib

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops import indexer_k_quant_and_cache

from . import accuracy_utils as utils


def _default_fp8_dtype():
    try:
        from vllm.platforms import current_platform

        return current_platform.fp8_dtype()
    except ImportError:
        pass

    if getattr(torch.version, "hip", None) is not None and hasattr(
        torch, "float8_e4m3fnuz"
    ):
        return torch.float8_e4m3fnuz
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    pytest.skip("float8_e4m3fn is required for indexer_k_quant_and_cache")


def _is_fp8_fnuz(dtype):
    return hasattr(torch, "float8_e4m3fnuz") and dtype == torch.float8_e4m3fnuz


def _load_vllm_cuda_op():
    try:
        import vllm._custom_ops as ops  # noqa: F401
    except Exception:
        return None, False

    if not hasattr(torch.ops._C_cache_ops, "indexer_k_quant_and_cache"):
        return None, False

    def vllm_indexer(k, kv_cache, slot_mapping, quant_block_size, scale_fmt):
        torch.ops._C_cache_ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    return vllm_indexer, True


def _f32_to_fp8_e4m3fn(y):
    """Bit-exact f32 -> e4m3fn conversion (RNE); `y` must be finite and
    pre-clamped to [-448, 448].
    """
    b = y.view(torch.int32)
    a = b & 0x7FFFFFFF
    t = a - 0x3C000000
    t += 0x0007FFFF + ((t >> 20) & 1)
    r_norm = t >> 20
    r_sub = (a.view(torch.float32) * 512.0 + 8388608.0).view(torch.int32) - 0x4B000000
    r = torch.where(a >= 0x3C800000, r_norm, r_sub)
    return (r | ((b >> 24) & 0x80)).to(torch.uint8)


def torch_npu_indexer(k, kv_cache, slot_mapping, quant_block_size, scale_fmt):
    num_blocks = kv_cache.shape[0]
    block_size = kv_cache.shape[1]
    head_dim = k.shape[-1]
    num_quant_blocks = head_dim // quant_block_size
    scale_divisor = 448.0

    flat_cache = kv_cache.view(num_blocks, -1)
    cache_values = flat_cache[:, : block_size * head_dim].view(torch.uint8)
    cache_scales = flat_cache[:, block_size * head_dim :].view(torch.float32)

    for token_idx in range(slot_mapping.numel()):
        slot_id = int(slot_mapping[token_idx].item())
        if slot_id < 0:
            continue

        block_id = slot_id // block_size
        block_offset = slot_id % block_size
        for quant_block_id in range(num_quant_blocks):
            start = quant_block_id * quant_block_size
            end = start + quant_block_size
            val = k[token_idx, start:end]
            amax = val.abs().to(torch.float32).amax()

            scale = (
                torch.maximum(
                    amax,
                    torch.tensor(1e-4, dtype=torch.float32, device=k.device),
                )
                / scale_divisor
            )
            if scale_fmt == "ue8m0":
                scale = torch.exp2(torch.ceil(torch.log2(scale)))

            value_start = block_offset * head_dim + start
            value_end = value_start + quant_block_size
            cache_values[block_id, value_start:value_end].copy_(
                _f32_to_fp8_e4m3fn(val.to(torch.float32) / scale)
            )
            cache_scales[
                block_id,
                block_offset * num_quant_blocks + quant_block_id,
            ] = scale


def torch_indexer(
    k, kv_cache, slot_mapping, quant_block_size, scale_fmt, fp8_dtype=None
):
    num_blocks = kv_cache.shape[0]
    block_size = kv_cache.shape[1]
    head_dim = k.shape[-1]
    num_quant_blocks = head_dim // quant_block_size
    fp8_dtype = _default_fp8_dtype() if fp8_dtype is None else fp8_dtype
    scale_divisor = 224.0 if _is_fp8_fnuz(fp8_dtype) else 448.0

    flat_cache = kv_cache.view(num_blocks, -1)
    cache_values = flat_cache[:, : block_size * head_dim].view(fp8_dtype)
    cache_scales = flat_cache[:, block_size * head_dim :].view(torch.float32)

    for token_idx in range(slot_mapping.numel()):
        slot_id = int(slot_mapping[token_idx].item())
        if slot_id < 0:
            continue

        block_id = slot_id // block_size
        block_offset = slot_id % block_size
        for quant_block_id in range(num_quant_blocks):
            start = quant_block_id * quant_block_size
            end = start + quant_block_size
            val = k[token_idx, start:end]
            amax = val.abs().to(torch.float32).amax()

            scale = (
                torch.maximum(
                    amax,
                    torch.tensor(1e-4, dtype=torch.float32, device=k.device),
                )
                / scale_divisor
            )
            if scale_fmt == "ue8m0":
                scale = torch.exp2(torch.ceil(torch.log2(scale)))

            value_start = block_offset * head_dim + start
            value_end = value_start + quant_block_size
            cache_values[block_id, value_start:value_end].copy_(
                (val.to(torch.float32) / scale).to(fp8_dtype)
            )
            cache_scales[
                block_id,
                block_offset * num_quant_blocks + quant_block_id,
            ] = scale


def _make_cache(
    num_blocks, block_size, head_dim, quant_block_size, device, fp8_dtype=None
):
    cache_stride = head_dim + head_dim * 4 // quant_block_size
    fp8_dtype = _default_fp8_dtype() if fp8_dtype is None else fp8_dtype
    k_cache = torch.empty(
        (num_blocks, block_size, cache_stride),
        dtype=torch.uint8,
        device="cpu",
    )
    num_quant_blocks = head_dim // quant_block_size
    flat_cache = k_cache.view(num_blocks, -1)
    values = flat_cache[:, : block_size * head_dim].view(fp8_dtype)
    values.copy_(torch.randn(values.shape).to(fp8_dtype))
    scales = flat_cache[:, block_size * head_dim :].view(torch.float32)
    scales.copy_(
        torch.rand(
            num_blocks,
            block_size * num_quant_blocks,
            device="cpu",
            dtype=torch.float32,
        )
        + 0.01
    )
    return k_cache.to(device)


def _make_slot_mapping(num_tokens, num_blocks, block_size, device):
    slot_mapping = torch.randperm(num_blocks * block_size, device=device)[:num_tokens]
    slot_mapping = slot_mapping.to(torch.long)
    slot_mapping[-1] = -1
    return slot_mapping


@pytest.mark.indexer_k_quant_and_cache
@pytest.mark.parametrize(
    "dtype,num_tokens,num_blocks,block_size,head_dim,quant_block_size,scale_fmt",
    [
        (torch.bfloat16, 19, 4, 16, 128, 128, "ue8m0"),
        (torch.float16, 23, 5, 16, 512, 128, "ue8m0"),
        (torch.float16, 29, 6, 16, 384, 128, "ue8m0"),
        (torch.float16, 31, 7, 16, 640, 128, "ue8m0"),
        (torch.bfloat16, 17, 4, 64, 512, 128, "ue8m0"),
    ],
)
@torch.inference_mode()
def test_indexer_k_quant_and_cache_matches_reference(
    dtype,
    num_tokens,
    num_blocks,
    block_size,
    head_dim,
    quant_block_size,
    scale_fmt,
):
    vllm_op, has_vllm = _load_vllm_cuda_op()

    torch.manual_seed(0)
    device = flaggems_vllm.device
    k = torch.randn(num_tokens, head_dim, device=device, dtype=dtype)
    slot_mapping = _make_slot_mapping(num_tokens, num_blocks, block_size, device)

    gems_cache = _make_cache(
        num_blocks,
        block_size,
        head_dim,
        quant_block_size,
        device,
    )
    reference_cache = gems_cache.clone()

    # MetaX: the vLLM op is broken on this backend (stock mcoplib ships it as
    # a silent no-op), so use the torch composition as the reference.
    use_vllm_ref = (
        has_vllm
        and dtype != torch.float16
        and flaggems_vllm.vendor_name == "nvidia"
        and torch.cuda.get_device_capability() >= (8, 9)
    )

    if use_vllm_ref:
        vllm_op(k, reference_cache, slot_mapping, quant_block_size, scale_fmt)
    elif flaggems_vllm.vendor_name == "ascend":
        torch_npu_indexer(k, reference_cache, slot_mapping, quant_block_size, scale_fmt)
    else:
        cpu_cache = reference_cache.cpu()
        torch_indexer(
            k.cpu(), cpu_cache, slot_mapping.cpu(), quant_block_size, scale_fmt
        )
        reference_cache.copy_(cpu_cache)
    flaggems_vllm.indexer_k_quant_and_cache(
        k,
        gems_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )
    if flaggems_vllm.vendor_name == "ascend":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()

    utils.gems_assert_equal(gems_cache, utils.to_reference(reference_cache))


@pytest.mark.indexer_k_quant_and_cache
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA-compatible graph API required"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale_fmt", ["ue8m0", "float32"])
@pytest.mark.parametrize("head_dim", [128, 512])
def test_software_fp8_cache_graph_matches_cpu(monkeypatch, dtype, scale_fmt, head_dim):
    module = importlib.import_module("flaggems_vllm.ops.indexer_k_quant_and_cache")
    monkeypatch.setattr(module, "kernel_supports_fp8_e4m3", lambda: False)
    # Exercise this wire format even on backends whose default is FNUZ.
    monkeypatch.setattr(module, "_get_fp8_dtype", lambda: torch.float8_e4m3fn)
    # Exact cache-byte comparison includes untouched storage and negative slots.
    torch.manual_seed(29)
    k = torch.randn(19, head_dim, device="cuda", dtype=dtype)
    k[0].zero_()
    k[1].fill_(-0.0)
    slots = _make_slot_mapping(19, 4, 16, "cuda")
    cache = _make_cache(4, 16, head_dim, 128, "cuda", torch.float8_e4m3fn)

    def run():
        indexer_k_quant_and_cache(k, cache, slots, 128, scale_fmt)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for factor in (0.5, -1.25):
        k.mul_(factor)
        slots.copy_(slots.roll(3))
        cache.fill_(0xA5)
        expected = cache.cpu()
        torch_indexer(
            k.cpu(), expected, slots.cpu(), 128, scale_fmt, torch.float8_e4m3fn
        )
        graph.replay()
        torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)
