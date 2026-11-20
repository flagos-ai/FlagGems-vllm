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


PPU_SPARSE_SUPPORTED = (
    "flash_mla_sparse_fwd_w8a8"
    in dict(flaggems_vllm.runtime.backend.get_customized_ops())
    and flaggems_vllm.vendor_name == "thead"
)


def quantize_sparse_int8(x):
    scale = x[..., :512].float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
    content = (x[..., :512].float() / scale).round().clamp(-127, 127).to(torch.int8)
    rope = (x[..., 512:].float() / scale).to(x.dtype)
    return content, rope, scale


def make_sparse_decode_inputs(lengths, queries, heads, topk, dtype, rope_dim=64):
    torch.manual_seed(0)
    batch = len(lengths)
    padded = math.ceil(max(lengths) / 256) * 256
    device = flaggems_vllm.device
    q = torch.randn(batch, queries, heads, 512 + rope_dim, dtype=dtype, device=device)
    q, qr, qs = quantize_sparse_int8(q)
    native_q = (torch.cat((q.float(), qr.float()), -1) * qs).to(dtype)
    pages = batch * padded // 64
    kv = torch.empty((pages, 64, 512), dtype=torch.int8, device=device)
    kr = torch.empty((pages, 64, rope_dim), dtype=dtype, device=device)
    ks = torch.empty((pages, 64, 1), dtype=torch.float32, device=device)
    native_kv = torch.empty((pages, 64, 1, 512 + rope_dim), dtype=dtype, device=device)
    # Chunk fixture construction to bound temporary memory on long ragged cases.
    for start in range(0, pages, 1024):
        part = slice(start, min(start + 1024, pages))
        source = torch.randn_like(native_kv[part, :, 0])
        v, r, s = quantize_sparse_int8(source)
        kv[part], kr[part], ks[part] = v, r, s
        native_kv[part, :, 0] = (torch.cat((v.float(), r.float()), -1) * s).to(dtype)
    ids = torch.full((batch, queries, topk), -1, dtype=torch.int32)
    generator = torch.Generator().manual_seed(0)
    for i, length in enumerate(lengths):
        if length < padded:
            kr.view(batch, padded, rope_dim)[i, length:] = torch.nan
            ks.view(batch, padded)[i, length:] = torch.nan
            native_kv.view(batch, padded, 512 + rope_dim)[i, length:] = torch.nan
        for j in range(queries):
            visible = length - queries + j + 1
            chosen = torch.randperm(visible, generator=generator)[:topk]
            ids[i, j, : len(chosen)] = i * padded + chosen
    return (q, qr, kv, kr, qs, ks, ids.to(device)), native_q, native_kv


def check_sparse_decode_sample(inputs, native_q, native_cache, candidate, baseline):
    from tests.test_flash_mla_sparse_fwd import FlashmlaSparseTestKit

    q, qr, kv, kr, qs, ks, indices = inputs
    # Check both precisions against their own FP32 logits on one complete request.
    ids = indices[0].long()
    selected = ids.clamp_min(0)
    sparse_q = torch.cat((q[0].float(), qr[0].float()), -1) * qs[0]
    v = kv.view(-1, 512)[selected].float()
    r = kr.view(-1, 64)[selected].float()
    sparse_k = torch.cat((v, r), -1) * ks.view(-1)[selected].unsqueeze(-1)
    native_k = native_cache.view(-1, 576)[selected]
    for query, keys, (out, lse) in (
        (sparse_q, sparse_k, candidate),
        (native_q[0], native_k, baseline),
    ):
        for j in range(query.shape[0]):
            topk = ids.shape[1]
            local_ids = torch.arange(topk).masked_fill(ids[j].cpu() < 0, -1)
            y, _, logsum = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
                1,
                topk,
                query.shape[1],
                1,
                576,
                topk,
                query[j : j + 1].float().cpu(),
                keys[j, :, None].float().cpu(),
                local_ids.to(torch.int32).view(1, 1, topk),
                576**-0.5,
                512,
                None,
                None,
            )
            gems_assert_close(out[0, j].cpu(), y[0], torch.bfloat16, atol=0.004)
            gems_assert_close(lse[0, :, j].cpu(), logsum[0], torch.float32, atol=2e-4)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(not PPU_SPARSE_SUPPORTED, reason="Requires PPU INT8 sparse MLA")
@pytest.mark.parametrize(
    "queries,heads,topk", [(1, 16, 1), (2, 32, 129), (2, 65, 137), (1, 128, 257)]
)
@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sparse_decode_ppu(queries, heads, topk, rope_dim, dtype):
    from tests.test_flash_mla_sparse_fwd import FlashmlaSparseTestKit

    inputs, _, _ = make_sparse_decode_inputs(
        [79, 215, 302], queries, heads, topk, dtype, rope_dim
    )
    q, qr, kv, kr, qs, ks, indices = inputs
    indices[0] = -1
    indices[2, :, 3::5] = kv.shape[0] * 64
    if topk > 1:
        indices[2, :, 1] = indices[2, :, 0]
    lengths = torch.tensor(
        [topk, 0, max(1, topk - 5)], device=q.device, dtype=torch.int32
    )
    sink = torch.randn(heads, device=q.device)
    sink[0], sink[1] = torch.inf, -torch.inf
    out, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8(
        *inputs, softmax_scale=0.0625, attn_sink=sink, topk_length=lengths
    )
    rows = 3 * queries
    dim = 512 + rope_dim
    expected, _, expected_lse = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
        rows,
        kv.shape[0] * 64,
        heads,
        1,
        dim,
        topk,
        (torch.cat((q.float(), qr.float()), -1) * qs).view(rows, heads, dim).cpu(),
        (torch.cat((kv.float(), kr.float()), -1) * ks).view(-1, 1, dim).cpu(),
        indices.view(rows, 1, topk).cpu(),
        0.0625,
        512,
        sink.cpu(),
        lengths.repeat_interleave(queries).cpu(),
    )
    gems_assert_close(
        out.float().view_as(expected).cpu(), expected.float(), torch.float32, atol=0.008
    )
    actual_lse = lse.transpose(1, 2).reshape(rows, heads).cpu()
    gems_assert_equal(torch.isposinf(actual_lse), torch.isposinf(expected_lse))
    mask = torch.isfinite(expected_lse)
    gems_assert_close(actual_lse[mask], expected_lse[mask], torch.float32, atol=2e-4)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(not PPU_SPARSE_SUPPORTED, reason="Requires PPU INT8 sparse MLA")
@pytest.mark.parametrize("empty", ["indices", "pages", "batch"])
def test_sparse_decode_ppu_empty(empty):
    inputs, _, _ = make_sparse_decode_inputs([8], 2, 16, 7, torch.bfloat16)
    q, qr, kv, kr, qs, ks, ids = inputs
    if empty == "indices":
        ids = ids[..., :0]
    elif empty == "pages":
        kv, kr, ks = kv[:0], kr[:0], ks[:0]
    else:
        q, qr, qs, ids = q[:0], qr[:0], qs[:0], ids[:0]
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8(
        q, qr, kv.unsqueeze(2), kr.unsqueeze(2), qs, ks.unsqueeze(2), ids
    )
    assert output.shape == q.shape and lse.shape == (q.shape[0], 16, 2)
    gems_assert_equal(output, torch.zeros_like(output))
    gems_assert_equal(lse, torch.full_like(lse, torch.inf))
