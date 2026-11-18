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
import random
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pytest
import torch

import flaggems_vllm
from tests.test_flash_mla_sparse_fwd import (
    Flashmla_Sparse_Test_Param,
    FlashmlaSparseTestKit,
)
from tests.test_flash_mla_with_kvcache_w8a8 import SUPPORTED as HAS_CUDA_FP8_MLA
from tests.test_flash_mla_with_kvcache_w8a8 import quantize_dense_fp8

from .accuracy_utils import gems_assert_close, gems_assert_equal, to_cpu, to_reference

MTHREADS_FP8_MLA = flaggems_vllm.vendor_name == "mthreads" and (
    "flash_mla_sparse_fwd_w8a8"
    in dict(flaggems_vllm.runtime.backend.get_customized_ops())
)
HAS_FP8_MLA = HAS_CUDA_FP8_MLA or MTHREADS_FP8_MLA


@lru_cache(maxsize=1)
def flashmla_reference() -> ModuleType:
    """Load the explicitly selected, unmodified vLLM CUDA reference for tests only."""
    reference_path = os.environ.get("FLAGGEMS_FLASHMLA_REFERENCE_PATH")
    if reference_path:
        # Initialize Torch shared libraries before loading the CUDA extension.
        importlib.import_module("torch")
        source = Path(reference_path).resolve()
        package_root = source.parents[2]
        parent = sys.modules.get("vllm")
        if parent is None:
            # The approved CUDA source only imports its extension; do not initialize serving.
            parent = ModuleType("vllm")
            parent.__path__ = [str(package_root)]
            sys.modules["vllm"] = parent
            created_parent = True
        else:
            created_parent = False

        for module_name in ("_flashmla_C", "_flashmla_extension_C"):
            qualified = f"vllm.{module_name}"
            if qualified in sys.modules:
                continue
            candidates = sorted(package_root.glob(f"{module_name}*.so"))
            if not candidates:
                raise RuntimeError(
                    f"missing CUDA reference extension: {package_root}/{module_name}"
                )
            spec = importlib.util.spec_from_file_location(qualified, candidates[0])
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            spec.loader.exec_module(module)
        name = "flaggems_vllm_test_flashmla_reference"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if created_parent:
            del sys.modules["vllm"]
        return module
    try:
        module = importlib.import_module("vllm.v1.attention.ops.flashmla")
    except ImportError as error:
        raise RuntimeError(
            "vLLM CUDA SparseMLA reference is unavailable; set "
            "FLAGGEMS_FLASHMLA_REFERENCE_PATH to its unmodified CUDA interface"
        ) from error
    supported, reason = module.is_flashmla_sparse_supported()
    if not supported:
        raise RuntimeError(
            f"vLLM SparseMLA reference unavailable: {reason}. "
            "Set FLAGGEMS_FLASHMLA_REFERENCE_PATH to the approved CUDA reference "
            "interface; required benchmarks must not be skipped."
        )
    return module


def make_sparse_inputs(shape, dtype=torch.bfloat16):
    s_q, h_q, d_qk, s_kv, h_kv, topk, d_v = shape[:7]
    sink, length, seed = shape[7:] if len(shape) == 10 else (False, False, 0)
    random.seed(seed)
    torch.manual_seed(seed)
    device = flaggems_vllm.device
    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device) / 10
    q = (q + (random.random() - 0.5) / 10).clamp_(-10, 10)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device) / 10
    kv = (kv + (random.random() - 0.5) / 10).clamp_(-10, 10)

    # Coprime strides select unique tokens without allocating an SQ-by-SKV matrix.
    positions = torch.arange(min(s_kv, topk), dtype=torch.int64)
    indices = torch.full((s_q, topk), -1, dtype=torch.int32)
    for row in range(s_q):
        offset = random.randrange(s_kv)
        step = random.randrange(1, s_kv) if s_kv > 1 else 1
        while math.gcd(step, s_kv) != 1:
            step = step + 1 if step + 1 < s_kv else 1
        indices[row, : positions.numel()] = (offset + step * positions) % s_kv
    indices = indices[:, None].to(device)

    attn_sink = None
    if sink:
        attn_sink = torch.randn((h_q,), dtype=torch.float32, device=device)
        mask = torch.randn((h_q,), dtype=torch.float32, device=device)
        attn_sink[mask < -0.5] = -torch.inf
        attn_sink[mask > 0.5] = torch.inf
    else:
        pass
    topk_length = (
        torch.randint(
            0, max(topk + 1, 64), (s_q,), dtype=torch.int32, device=device
        ).clamp_max(topk)
        if length
        else None
    )
    return q, kv, indices, 0.5, d_v, attn_sink, topk_length


def quantize_sparse_fp8_fixture(
    q, kv, indices, sm_scale, value_dim, attn_sink=None, topk_length=None
):
    if value_dim != 512:
        raise ValueError("FP8 sparse MLA requires value_dim=512")
    q_nope, q_rope, q_scale, _ = quantize_dense_fp8(q, dequantize=False)
    pages = math.ceil(kv.shape[0] / 64)
    cache = torch.zeros((pages * 64, kv.shape[-1]), dtype=kv.dtype, device=kv.device)
    cache[: kv.shape[0]].copy_(kv[:, 0])
    k_nope, k_rope, k_scale, _ = quantize_dense_fp8(cache, dequantize=False)
    return (
        q_nope[:, None],
        q_rope[:, None],
        k_nope.view(pages, 64, 512),
        k_rope.view(pages, 64, q_rope.shape[-1]),
        q_scale[:, None],
        k_scale.view(pages, 64, 1),
        torch.where((indices >= 0) & (indices < kv.shape[0]), indices, -1),
        sm_scale,
        attn_sink,
        topk_length,
    )


def assert_sparse_fixture_close(output, lse, reference, reference_lse):
    reference = to_reference(reference)
    reference_lse = to_reference(reference_lse)
    output = to_cpu(output, reference)
    lse = to_cpu(lse, reference_lse)
    relative = torch.linalg.vector_norm(
        output.float() - reference.float()
    ) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    gems_assert_close(relative, torch.zeros_like(relative), torch.float32, atol=0.05)
    gems_assert_equal(torch.isposinf(lse), torch.isposinf(reference_lse))
    gems_assert_equal(torch.isneginf(lse), torch.isneginf(reference_lse))
    finite = torch.isfinite(reference_lse)
    scaled_error = (lse[finite].float() - reference_lse[finite].float()) / (
        0.025 + 0.002 * reference_lse[finite].float().abs()
    )
    gems_assert_close(
        scaled_error, torch.zeros_like(scaled_error), torch.float32, atol=1.0
    )


def _mthreads_quantize(tensor, *, dequantize=True):
    scale = (
        tensor[..., :512].float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
    )
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    content = (tensor[..., :512].float() / scale).to(torch.float8_e4m3fn)
    rope = (tensor[..., 512:].float() / scale).to(torch.bfloat16)
    if not dequantize:
        return content, rope, scale
    native = torch.empty(tensor.shape, dtype=torch.bfloat16, device=tensor.device)
    row_elements = math.prod(tensor.shape[1:-1]) * 512
    chunk = max(1, (1 << 23) // max(1, row_elements))
    for first in range(0, tensor.shape[0], chunk):
        last = min(first + chunk, tensor.shape[0])
        values = content[first:last].float()
        if rope.shape[-1]:
            values = torch.cat((values, rope[first:last].float()), -1)
        native[first:last] = (values * scale[first:last]).to(torch.bfloat16)
    return content, rope, scale, native


def make_mthreads_sparse_inputs(batch, heads, topk, tokens, dtype, rope_dim, seed):
    pages = math.ceil(tokens / 64)
    param = Flashmla_Sparse_Test_Param(batch, tokens, topk, heads, d_qk=576)
    query, cache, indices, _, _ = FlashmlaSparseTestKit.make_input_flashmla(
        param, seed=seed
    )
    q, qr, qs, native_q = _mthreads_quantize(query.view(batch, 1, heads, 576))
    if tokens % 64:
        padded = torch.zeros(
            (pages * 64, 1, 576), dtype=cache.dtype, device=cache.device
        )
        padded[:tokens].copy_(cache)
        cache = padded
    kv, kr, ks = _mthreads_quantize(cache.view(pages, 64, 1, 576), dequantize=False)
    if rope_dim == 0:
        qr, kr = qr[..., :0].contiguous(), kr[..., :0].contiguous()
        native_q = native_q[..., :512].contiguous()
    inputs = [
        q,
        qr.to(dtype),
        kv,
        kr.to(dtype),
        qs,
        ks,
        indices.view(batch, 1, topk),
    ]
    return inputs, native_q


def mthreads_sparse_reference(
    inputs,
    *,
    attn_sink=None,
    topk_length=None,
    softmax_scale=None,
    reference_device="cpu",
    chunk_rows=16,
):
    q, qr, kv, kr, qs, ks, indices = (tensor.to(reference_device) for tensor in inputs)
    attn_sink = None if attn_sink is None else attn_sink.to(reference_device)
    topk_length = (
        None if topk_length is None else topk_length.to(reference_device).flatten()
    )
    batch, queries, heads, _ = q.shape
    dims = 512 + qr.shape[-1]
    rows = batch * queries
    q = q.view(rows, heads, 512)
    qr = qr.view(rows, heads, qr.shape[-1])
    qs = qs.view(rows, heads, 1)
    tokens = kv.shape[0] * 64
    kv = kv.view(tokens, 1, 512)
    kr = kr.view(tokens, 1, kr.shape[-1])
    ks = ks.view(tokens, 1, 1)
    cache = torch.empty((tokens, 1, dims), dtype=torch.float32, device=reference_device)
    # Keep FP8 casts bounded: large unary datacast launches fail in muDNN.
    for first in range(0, tokens, 16384):
        last = min(first + 16384, tokens)
        values = kv[first:last].float()
        if dims > 512:
            values = torch.cat((values, kr[first:last].float()), -1)
        cache[first:last] = values * ks[first:last]
    selected = indices.view(rows, 1, -1)
    output = torch.empty(
        (batch * queries, heads, 512), dtype=torch.float32, device=reference_device
    )
    lse = torch.empty(
        (batch * queries, heads), dtype=torch.float32, device=reference_device
    )
    for begin in range(0, batch * queries, chunk_rows):
        end = min(begin + chunk_rows, batch * queries)
        query = q[begin:end].float()
        if dims > 512:
            query = torch.cat((query, qr[begin:end].float()), -1)
        query = query * qs[begin:end]
        output[begin:end], _, lse[begin:end] = (
            FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
                end - begin,
                cache.shape[0],
                heads,
                1,
                dims,
                indices.shape[-1],
                query,
                cache,
                selected[begin:end],
                dims**-0.5 if softmax_scale is None else softmax_scale,
                512,
                attn_sink,
                None if topk_length is None else topk_length[begin:end],
                output_dtype=torch.float32,
            )
        )
    return output.view(batch, queries, heads, 512), lse.view(
        batch, queries, heads
    ).transpose(1, 2)


def assert_mthreads_sparse_accuracy(actual, expected, *, output_atol=4e-3):
    ref_output, ref_lse = expected
    output, lse = actual
    assert output.shape == ref_output.shape and lse.shape == ref_lse.shape
    # Compare every row while bounding casts and copies to the same small size.
    for first in range(0, output.shape[0], 16):
        last = min(first + 16, output.shape[0])
        value = output[first:last].to(ref_output.device)
        logsum = lse[first:last].to(ref_lse.device)
        reference_value = ref_output[first:last]
        reference_logsum = ref_lse[first:last]
        gems_assert_close(value, reference_value, output.dtype, atol=output_atol)
        gems_assert_equal(torch.isposinf(logsum), torch.isposinf(reference_logsum))
        finite = torch.isfinite(reference_logsum)
        gems_assert_close(
            logsum[finite], reference_logsum[finite], torch.float32, atol=2e-4
        )


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(not HAS_FP8_MLA, reason="Sparse FP8 MLA backend is unavailable")
@pytest.mark.parametrize(
    "shape",
    [
        (4, 64, 512, 95, 1, 128, 512, False, False, 0),
        (4, 128, 512, 592, 1, 256, 512, True, True, 1),
        (4, 64, 576, 95, 1, 128, 512, False, False, 2),
        (4, 128, 576, 1592, 1, 512, 512, True, True, 3),
        (62, 64, 512, 1521, 1, 512, 512, True, False, 4),
        (62, 128, 576, 98816, 1, 256, 512, False, True, 5),
        (547, 64, 512, 98816, 1, 8320, 512, False, False, 195),
    ],
)
def test_flash_mla_sparse_flashmla(shape):
    inputs = make_sparse_inputs(shape)
    inputs[2][0].fill_(-1)
    packed = quantize_sparse_fp8_fixture(*inputs)
    if MTHREADS_FP8_MLA:
        expected = mthreads_sparse_reference(
            packed[:7],
            softmax_scale=packed[7],
            attn_sink=packed[8],
            topk_length=packed[9],
            reference_device=packed[0].device,
        )
        assert_mthreads_sparse_accuracy(
            flaggems_vllm.flash_mla_sparse_fwd_w8a8(*packed), expected
        )
        return
    reference, _, reference_lse = flashmla_reference().flash_mla_sparse_fwd(*inputs)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8(*packed)
    assert_sparse_fixture_close(output[:, 0], lse[:, :, 0], reference, reference_lse)
