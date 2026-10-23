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
import importlib.util
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pytest
import torch

import flaggems_vllm
from benchmark.test_flash_mla_with_kvcache_fwd import (
    assert_dense_fp8_close,
    assert_dense_fp8_replay,
)
from flaggems_vllm.ops.flash_mla_with_kvcache_fwd_w8a8_fp8 import (
    HAS_TLE,
    flash_mla_with_kvcache_fwd_w8a8_fp8,
    get_mla_fp8_metadata,
    prepare_flash_mla_with_kvcache_fwd_w8a8_fp8,
)

from .accuracy_utils import gems_assert_equal, to_reference

HAS_INT8_MLA = "flash_mla_with_kvcache_fwd_w8a8_int8" in dict(
    flaggems_vllm.runtime.backend.get_customized_ops()
)
HAS_FP8_MLA = (
    HAS_TLE and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
)


@lru_cache(maxsize=1)
def dense_cuda_reference():
    if HAS_INT8_MLA:
        return importlib.import_module("flash_mla")
    source_path = os.environ.get("FLAGGEMS_FLASHMLA_REFERENCE_PATH")
    if not source_path:
        try:
            return importlib.import_module(
                "vllm.third_party.flashmla.flash_mla_interface"
            )
        except ImportError as error:
            raise RuntimeError(
                "Official vLLM CUDA FlashMLA is required; set "
                "FLAGGEMS_FLASHMLA_REFERENCE_PATH to its unmodified interface"
            ) from error
    source = Path(source_path).resolve()
    package_root = source.parents[2]
    created_parent = "vllm" not in sys.modules
    if created_parent:
        parent = ModuleType("vllm")
        parent.__path__ = [str(package_root)]
        sys.modules["vllm"] = parent
    for module_name in ("_flashmla_C", "_flashmla_extension_C"):
        qualified = f"vllm.{module_name}"
        if qualified in sys.modules:
            continue
        candidates = sorted(package_root.glob(f"{module_name}*.so"))
        if not candidates:
            raise RuntimeError(f"Missing CUDA reference extension: {qualified}")
        spec = importlib.util.spec_from_file_location(qualified, candidates[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    name = "flaggems_vllm_test_dense_flashmla_reference"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if created_parent:
        del sys.modules["vllm"]
    return module


def _dense_cuda_reference_or_skip():
    try:
        return dense_cuda_reference()
    except (ImportError, RuntimeError):
        pytest.skip(
            "vLLM CUDA FlashMLA reference is unavailable; set "
            "FLAGGEMS_FLASHMLA_REFERENCE_PATH to its unmodified interface"
        )


def run_cuda(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
    reference = (
        dense_cuda_reference() if HAS_INT8_MLA else _dense_cuda_reference_or_skip()
    )
    if HAS_INT8_MLA:
        metadata, splits = reference.get_mla_metadata(
            cache_seqlens, q.shape[1] * q.shape[2], 1
        )
        return reference.flash_mla_with_kvcache(
            q,
            k_cache,
            block_table,
            cache_seqlens,
            head_dim_v,
            metadata,
            splits,
            **kwargs,
        )
    metadata, _ = reference.get_mla_metadata()
    return reference.flash_mla_with_kvcache(
        q, k_cache, block_table, cache_seqlens, head_dim_v, metadata, **kwargs
    )


def _quantize_dense_content(x, d_nope=512):
    """Per-token content plus scaled BF16 RoPE, and its BF16 dequantization."""
    peak = 127.0 if HAS_INT8_MLA else 448.0
    scale = x[..., :d_nope].float().abs().amax(-1, keepdim=True) / peak
    if HAS_INT8_MLA:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
        nope = (x[..., :d_nope].float() / scale).round().to(torch.int8)
    else:
        nope = (x[..., :d_nope].float() / scale).to(torch.float8_e4m3fn)
    rope = (x[..., d_nope:].float() / scale).to(torch.bfloat16)
    dequantized = torch.cat((nope.float(), rope.float()), -1) * scale
    return nope, rope, scale, dequantized.to(torch.bfloat16)


@(
    pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
    if HAS_INT8_MLA
    else pytest.mark.flash_mla_with_kvcache_fwd
)
@pytest.mark.skipif(
    not (HAS_FP8_MLA or HAS_INT8_MLA), reason="Requires registered dense MLA"
)
@pytest.mark.parametrize("seq_q", (1, 2))
@pytest.mark.parametrize("h_q", (16, 32, 64, 128))
def test_dense_decode(seq_q, h_q):
    batch, d_qk = 4, 576
    h_k = 1
    head_dim_v = 512
    page_block_size = 64
    seqlen = 256
    if HAS_INT8_MLA:
        dense_cuda_reference()
    else:
        _dense_cuda_reference_or_skip()

    torch.manual_seed(45)
    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device="cuda")
    max_pages_per_seq = math.ceil(seqlen / page_block_size) + 4
    total_pages = batch * max_pages_per_seq
    kv_cache = (
        torch.randn(
            total_pages, page_block_size, h_k, d_qk, dtype=torch.bfloat16, device="cuda"
        )
        * 0.1
    )
    block_table = torch.arange(total_pages, dtype=torch.int32, device="cuda").view(
        batch, max_pages_per_seq
    )
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device="cuda")
    cache_seqlens[0] = seqlen // 2
    cache_seqlens[-1] = seqlen + page_block_size
    q_nope, q_rope, q_scale, q = _quantize_dense_content(q)
    k_nope, k_rope, k_scale, kv_cache = _quantize_dense_content(kv_cache)

    cuda_out, cuda_lse = run_cuda(
        q, kv_cache, block_table, cache_seqlens, head_dim_v, causal=True
    )
    cuda_out, cuda_lse = to_reference(cuda_out), to_reference(cuda_lse)
    quantized_inputs = (
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
    )
    if HAS_INT8_MLA:
        out, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
            *quantized_inputs, causal=True
        )
        assert_dense_fp8_close((out, lse), (cuda_out, cuda_lse))
        handle, _ = flaggems_vllm.prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
            *quantized_inputs, causal=True
        )
        assert_dense_fp8_replay(handle, (cuda_out, cuda_lse))
        return
    out, lse = flash_mla_with_kvcache_fwd_w8a8_fp8(*quantized_inputs, causal=True)
    assert_dense_fp8_close((out, lse), (cuda_out, cuda_lse))
    # Precomputed GPU split metadata must reproduce the internal plan.
    metadata, num_splits = get_mla_fp8_metadata(cache_seqlens, seq_q * h_q)
    planned_out, planned_lse = flash_mla_with_kvcache_fwd_w8a8_fp8(
        *quantized_inputs, metadata, num_splits, causal=True
    )
    gems_assert_equal(planned_out, to_reference(out))
    gems_assert_equal(planned_lse, to_reference(lse))
    host_lengths = (seqlen // 2, seqlen, seqlen, seqlen + page_block_size)
    handle, (prepared_out, prepared_lse) = prepare_flash_mla_with_kvcache_fwd_w8a8_fp8(
        *quantized_inputs,
        causal=True,
        initial_cache_seqlens=host_lengths,
        max_cache_seqlens=host_lengths,
    )
    assert_dense_fp8_replay(
        lambda: handle(out=prepared_out, lse=prepared_lse), (cuda_out, cuda_lse)
    )
