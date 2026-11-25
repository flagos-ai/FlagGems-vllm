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
from tests.test_flash_mla_with_kvcache_w8a8 import SUPPORTED as HAS_FP8_MLA
from tests.test_flash_mla_with_kvcache_w8a8 import quantize_dense_fp8

from .accuracy_utils import gems_assert_close, gems_assert_equal, to_cpu, to_reference


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
        if not created_parent:
            return module
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

    if sink:
        attn_sink = torch.randn((h_q,), dtype=torch.float32, device=device)
        mask = torch.randn((h_q,), dtype=torch.float32, device=device)
        attn_sink[mask < -0.5] = -torch.inf
        attn_sink[mask > 0.5] = torch.inf
    else:
        attn_sink = None
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


PAGE_TOKENS = 64
NOPE = 448
ROPE = 64
GROUP = 64
SCALE_BYTES = 8
DATA_BYTES = NOPE + ROPE * 2
TOKEN_BYTES = DATA_BYTES + SCALE_BYTES
PAGE_PAYLOAD = PAGE_TOKENS * TOKEN_BYTES
PAGE_STRIDE = ((PAGE_PAYLOAD + DATA_BYTES - 1) // DATA_BYTES) * DATA_BYTES


def pack_v4_cache(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native packed pages and the corresponding BF16 reference KV."""
    tokens = kv.shape[0]
    assert tokens > 0 and kv.shape == (tokens, 1, NOPE + ROPE)
    assert kv.dtype == torch.bfloat16

    pages = (tokens + PAGE_TOKENS - 1) // PAGE_TOKENS
    padded = torch.zeros(
        (pages * PAGE_TOKENS, NOPE + ROPE), dtype=kv.dtype, device=kv.device
    )
    padded[:tokens].copy_(kv[:, 0])
    padded = padded.view(pages, PAGE_TOKENS, NOPE + ROPE)
    groups = (
        padded[..., :NOPE].float().reshape(pages, PAGE_TOKENS, NOPE // GROUP, GROUP)
    )
    exponent = (groups.abs().amax(-1) / 448.0).clamp_min(1.0e-4).log2().ceil()
    scales = torch.exp2(exponent)
    fp8 = (groups / scales[..., None]).to(torch.float8_e4m3fn)

    page_bytes = torch.zeros((pages, PAGE_STRIDE), dtype=torch.uint8, device=kv.device)
    data_end = PAGE_TOKENS * DATA_BYTES
    payload = page_bytes[:, :data_end].view(pages, PAGE_TOKENS, DATA_BYTES)
    payload[..., :NOPE] = fp8.view(torch.uint8).reshape(pages, PAGE_TOKENS, NOPE)
    payload[..., NOPE:] = padded[..., NOPE:].contiguous().view(torch.uint8)
    scale = page_bytes[:, data_end:PAGE_PAYLOAD].view(pages, PAGE_TOKENS, SCALE_BYTES)
    scale[..., : NOPE // GROUP] = (exponent + 127).to(torch.uint8)

    reference = padded.clone()
    reference[..., :NOPE] = (fp8.float() * scales[..., None]).reshape(
        pages, PAGE_TOKENS, NOPE
    )
    packed = page_bytes[:, :PAGE_PAYLOAD].view(pages, PAGE_TOKENS, 1, TOKEN_BYTES)
    return packed, reference.view(-1, 1, NOPE + ROPE)[:tokens]


def quantize_v4_query(q: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Prepare the native V4 448 FP8 Q and 64 BF16 RoPE."""
    batch, heads, width = q.shape
    assert width == NOPE + ROPE and q.dtype == torch.bfloat16
    blocks = q[..., :NOPE].float().reshape(batch, heads, NOPE // GROUP, GROUP)
    exponent = (blocks.abs().amax(-1) / 448.0).clamp_min(1.0e-4).log2().ceil()
    scales = torch.exp2(exponent)
    fp8 = (blocks / scales[..., None]).to(torch.float8_e4m3fn)
    rope = q[..., NOPE:].contiguous()
    reference = q.clone()
    reference[..., :NOPE] = (fp8.float() * scales[..., None]).reshape(
        batch, heads, NOPE
    )
    return fp8.reshape(batch, heads, NOPE), rope, scales, reference


def native_v4_inputs(inputs):
    query, kv, indices, scale, value_dim, sink, length = inputs
    if sink is not None or length is not None:
        raise ValueError("Native V4 benchmark requires fixed top-k without sinks")
    fp8, rope, scales, query_reference = quantize_v4_query(query)
    cache, kv_reference = pack_v4_cache(kv)
    baseline = query_reference, kv_reference, indices, scale, value_dim, sink, length
    packed = (
        fp8[:, None],
        rope[:, None],
        cache,
        None,
        scales[:, None],
        None,
        indices,
        scale,
    )
    return baseline, packed


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(not HAS_FP8_MLA, reason="FP8 MLA requires NVIDIA Hopper and TLE")
@pytest.mark.parametrize("topk", [95, 193, 641])
def test_native_v4_sparse_attention(topk):
    inputs = make_sparse_inputs((5, 64, 512, 665, 1, topk, 512, False, False, 29))
    query, kv, indices, scale, _, _, _ = inputs
    for group in range(7):
        columns = slice(group * 64, (group + 1) * 64)
        query[..., columns].mul_(2.0 ** (2 - group))
        kv[..., columns].mul_(2.0 ** (group - 2))
    indices[0].fill_(-1)
    indices[1, 0, ::2] = 669
    baseline, packed = native_v4_inputs(inputs)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8(*packed, cache_tokens=665)
    for row in range(query.shape[0]):
        physical = indices[row, 0]
        valid = physical[(physical >= 0) & (physical < kv.shape[0])].long()
        if not valid.numel():
            gems_assert_equal(output[row], torch.zeros_like(output[row]))
            gems_assert_equal(lse[row], torch.full_like(lse[row], torch.inf))
            continue
        keys = baseline[1][valid, 0].float()
        # Elementwise FP32 products avoid the global TF32 matmul policy.
        scores = (baseline[0][row, :, None, :].float() * keys).sum(-1) * scale
        expected = (scores.softmax(-1)[:, :, None] * keys).sum(1)
        gems_assert_close(
            output[row, 0], expected.to(torch.bfloat16), torch.bfloat16, atol=0.015
        )
        gems_assert_close(
            lse[row, :, 0], scores.logsumexp(-1), torch.float32, atol=0.002
        )


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(not HAS_FP8_MLA, reason="FP8 MLA requires NVIDIA Hopper and TLE")
def test_native_v4_long_value_reduction():
    # A nonzero value mean exposes bias when many FP8 MMA tiles share an accumulator.
    inputs = make_sparse_inputs((240, 64, 512, 8448, 1, 8320, 512, False, False, 212))
    baseline, packed = native_v4_inputs(inputs)
    reference, _, reference_lse = flashmla_reference().flash_mla_sparse_fwd(*baseline)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8(*packed)
    assert_sparse_fixture_close(output[:, 0], lse[:, :, 0], reference, reference_lse)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(not HAS_FP8_MLA, reason="FP8 MLA requires NVIDIA Hopper and TLE")
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
    reference, _, reference_lse = flashmla_reference().flash_mla_sparse_fwd(*inputs)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8(
        *quantize_sparse_fp8_fixture(*inputs)
    )
    assert_sparse_fixture_close(output[:, 0], lse[:, :, 0], reference, reference_lse)
