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

from . import base

USE_SOFT_CAST = flaggems_vllm.vendor_name == "ascend"


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


_CVT_CONSTS = {}


def _cvt_consts(device):
    """Cache device scalars used by the Ascend software FP8 conversion."""
    # NPU notes: python-int scalar ops take a slow path, and both int32 shifts
    # and int32 div-floor (which is routed through float) are slow or inexact.
    # All constants are therefore pre-materialized as device tensor scalars,
    # and shifts are emulated by masking off the low bits (making the value an
    # exactly-representable multiple of 2**k) followed by div-floor.
    if device not in _CVT_CONSTS:
        i32 = torch.int32
        _CVT_CONSTS[device] = {
            "m_abs": torch.tensor(0x7FFFFFFF, dtype=i32, device=device),
            "c_sub": torch.tensor(0x3C000000, dtype=i32, device=device),
            "c_rnd": torch.tensor(0x0007FFFF, dtype=i32, device=device),
            "c_bit20": torch.tensor(0x00100000, dtype=i32, device=device),
            "c_low20": torch.tensor(0x000FFFFF, dtype=i32, device=device),
            "c_low24": torch.tensor(0x00FFFFFF, dtype=i32, device=device),
            "c_ge": torch.tensor(0x3C800000, dtype=i32, device=device),
            "c_sub2": torch.tensor(0x4B000000, dtype=i32, device=device),
            "c_2p20": torch.tensor(1 << 20, dtype=i32, device=device),
            "c_2p24": torch.tensor(1 << 24, dtype=i32, device=device),
            "c_sign": torch.tensor(0x80, dtype=i32, device=device),
        }
    return _CVT_CONSTS[device]


def _f32_to_fp8_e4m3fn(y):
    """Bit-exact f32 -> e4m3fn conversion (RNE); `y` must be finite and
    pre-clamped to [-448, 448].
    """
    c = _cvt_consts(y.device)
    b = y.view(torch.int32)
    a = b & c["m_abs"]
    t = a - c["c_sub"]
    # (t >> 20) & 1; the dividend is a single bit so the div is exact.
    t += c["c_rnd"] + torch.div(t & c["c_bit20"], c["c_bit20"], rounding_mode="floor")
    # t >> 20, exact: strip the low 20 bits first so the dividend is a multiple
    # of 2**20 (exactly representable in f32, so the float-routed div is exact).
    r_norm = torch.div(t - (t & c["c_low20"]), c["c_2p20"], rounding_mode="floor")
    r_sub = (a.view(torch.float32) * 512.0 + 8388608.0).view(torch.int32) - c["c_sub2"]
    r = torch.where(a >= c["c_ge"], r_norm, r_sub)
    # (b >> 24) & 0x80, exact: strip the low 24 bits first.
    sign = (
        torch.div(b - (b & c["c_low24"]), c["c_2p24"], rounding_mode="floor")
        & c["c_sign"]
    )
    return (r | sign).to(torch.uint8)


def torch_indexer(k, kv_cache, slot_mapping, quant_block_size, scale_fmt):
    # Vectorized reference: slot_id uniquely identifies the (block, row) since
    # slot_id = block_id * block_size + block_offset, so the quant values and
    # scales can be scattered with one advanced-indexing write each.
    num_blocks = kv_cache.shape[0]
    block_size = kv_cache.shape[1]
    head_dim = k.shape[-1]
    num_quant_blocks = head_dim // quant_block_size
    fp8_dtype = _default_fp8_dtype()
    scale_divisor = 224.0 if _is_fp8_fnuz(fp8_dtype) else 448.0

    valid = slot_mapping >= 0
    slots = slot_mapping[valid]
    if slots.numel() == 0:
        return
    block_id = slots // block_size
    block_offset = slots % block_size

    val = k[valid]
    num_valid = val.shape[0]
    # abs/max selects an input magnitude exactly for fp16/bf16; cast only
    # the reduced tensor rather than an entire extra copy of the input.
    amax = (
        val.abs()
        .view(num_valid, num_quant_blocks, quant_block_size)
        .amax(-1)
        .to(torch.float32)
    )
    # amax is a fresh temporary; updating it cannot mutate the inputs.
    scale = amax.clamp_min_(1e-4).div_(scale_divisor)
    if scale_fmt == "ue8m0":
        scale.log2_().ceil_().exp2_()

    scaled_val = val.to(torch.float32).view(
        num_valid, num_quant_blocks, quant_block_size
    ) / scale.unsqueeze(-1)

    flat_cache = kv_cache.view(num_blocks, -1)
    if USE_SOFT_CAST:
        fp8_val = _f32_to_fp8_e4m3fn(scaled_val.reshape(num_valid, head_dim))
        if head_dim % 4 == 0:
            # uint8 scatters hit a slow NPU kernel; scatter as packed int32.
            fp8_val = fp8_val.view(torch.int32)
            values_view = (
                flat_cache[:, : block_size * head_dim]
                .view(torch.int32)
                .unflatten(1, (block_size, head_dim // 4))
            )
        else:
            values_view = flat_cache[:, : block_size * head_dim].unflatten(
                1, (block_size, head_dim)
            )
    else:
        fp8_val = scaled_val.reshape(num_valid, head_dim).to(fp8_dtype)
        values_view = (
            flat_cache[:, : block_size * head_dim]
            .view(fp8_dtype)
            .unflatten(1, (block_size, head_dim))
        )
    values_view[block_id, block_offset] = fp8_val

    scales_view = (
        flat_cache[:, block_size * head_dim :]
        .view(torch.float32)
        .unflatten(1, (block_size, num_quant_blocks))
    )
    scales_view[block_id, block_offset] = scale


def vllm_indexer(k, kv_cache, slot_mapping, quant_block_size, scale_fmt):
    torch.ops._C_cache_ops.indexer_k_quant_and_cache(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )


try:
    import vllm._custom_ops as ops  # noqa: F401

    if hasattr(torch.ops._C_cache_ops, "indexer_k_quant_and_cache"):
        ref_indexer = vllm_indexer
    else:
        ref_indexer = torch_indexer
except Exception:
    ref_indexer = torch_indexer


class IndexerKQuantAndCacheBenchmark(base.Benchmark):
    def __init__(self, vllm_op):
        super().__init__(
            op_name="indexer_k_quant_and_cache",
            torch_op=vllm_op,
            dtypes=[torch.float16, torch.bfloat16],  # vLLM supports both K dtypes.
        )
        self.set_gems(flaggems_vllm.indexer_k_quant_and_cache)
        self.shape_desc = (
            "num_tokens, num_blocks, block_size, head_dim, quant_block_size"
        )

    def set_shapes(self, shape_file_path=None):
        head_dim = 512
        quant_block_size = 128
        block_size = 16
        token_sweep = (
            1,
            2,
            4,
            8,
            16,
            17,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        )
        self.shapes = [
            (
                num_tokens,
                max(1, (2 * num_tokens + block_size - 1) // block_size),
                block_size,
                head_dim,
                quant_block_size,
            )
            for num_tokens in token_sweep
        ]
        block_size = 64
        self.shapes += [
            (
                num_tokens,
                max(1, (2 * num_tokens + block_size - 1) // block_size),
                block_size,
                head_dim,
                quant_block_size,
            )
            for num_tokens in (8192, 32768, 65536)
        ]

    def get_input_iter(self, dtype):
        for (
            num_tokens,
            num_blocks,
            block_size,
            head_dim,
            quant_block_size,
        ) in self.shapes:
            k = torch.randn(
                num_tokens,
                head_dim,
                dtype=dtype,
                device=self.device,
            )
            slot_mapping = torch.randperm(
                num_blocks * block_size,
                device=self.device,
            )[
                :num_tokens
            ].to(torch.long)
            cache_stride = head_dim + head_dim * 4 // quant_block_size
            kv_cache = torch.empty(
                num_blocks,
                block_size,
                cache_stride,
                dtype=torch.uint8,
                device=self.device,
            )
            yield k, kv_cache, slot_mapping, quant_block_size, {"scale_fmt": "ue8m0"}


@pytest.mark.indexer_k_quant_and_cache
def test_indexer_k_quant_and_cache_benchmark():
    bench = IndexerKQuantAndCacheBenchmark(ref_indexer)
    bench.run()
