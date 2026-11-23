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
from .conftest import QUICK_MODE

INT8_OP_NAME = "flash_mla_sparse_fwd_w8a8_int8"
SPARSE_INT8_OP = dict(
    flaggems_vllm.runtime.backend.get_customized_ops(flaggems_vllm.vendor_name)
).get(INT8_OP_NAME)
HAS_SPARSE_W8A8 = HAS_FP8_MLA or SPARSE_INT8_OP is not None


@lru_cache(maxsize=1)
def flashmla_reference() -> ModuleType:
    """Load the native BF16 reference on the active backend."""
    if SPARSE_INT8_OP is not None:
        return importlib.import_module("flash_mla")
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
    sm_scale = 0.5 if d_qk == 512 else d_qk**-0.5
    return q, kv, indices, sm_scale, d_v, attn_sink, topk_length


def quantize_sparse_fixture(
    q, kv, indices, sm_scale, value_dim, attn_sink=None, topk_length=None, *, int8=False
):
    if value_dim != 512:
        raise ValueError("Sparse W8A8 MLA requires value_dim=512")

    def quantize(tensor):
        if not int8:
            return quantize_dense_fp8(tensor, dequantize=False)[:3]
        tensor = tensor.float()
        scale = (tensor[..., :512].abs().amax(-1, keepdim=True) / 127).clamp_min(1e-30)
        nope = (tensor[..., :512] / scale).round().clamp(-127, 127).to(torch.int8)
        rope = (tensor[..., 512:] / scale).to(q.dtype).contiguous()
        return nope, rope, scale

    q_nope, q_rope, q_scale = quantize(q)
    skv = kv.shape[0]
    pages = math.ceil(skv / 64)
    cache = torch.zeros((pages * 64, kv.shape[-1]), dtype=kv.dtype, device=kv.device)
    cache[:skv].copy_(kv[:, 0])
    k_nope, k_rope, k_scale = quantize(cache)
    if int8 and pages * 64 != skv:
        k_scale[skv:].fill_(1)
    return (
        q_nope[:, None],
        q_rope[:, None],
        k_nope.view(pages, 64, 512),
        k_rope.view(pages, 64, q_rope.shape[-1]),
        q_scale[:, None],
        k_scale.view(pages, 64, 1),
        torch.where((indices >= 0) & (indices < skv), indices, -1),
        sm_scale,
        attn_sink,
        topk_length,
    )


def dequantize_sparse_int8_fixture(packed, skv, dtype):
    q_nope, q_rope, kv_nope, kv_rope, q_scale, k_scale, indices, sm, sink, length = (
        packed
    )
    q = torch.cat((q_nope.float(), q_rope.float()), -1) * q_scale
    q = q.flatten(0, 1)
    indices = indices.flatten(0, 1)[:, None]
    k_nope = kv_nope.view(-1, 512)[:skv]
    k_rope = kv_rope.view(kv_nope.shape[0] * 64, q_rope.shape[-1])[:skv]
    k = torch.cat((k_nope.float(), k_rope.float()), -1) * k_scale.view(-1, 1)[:skv]
    return q.to(dtype), k[:, None].to(dtype), indices, sm, 512, sink, length


def assert_sparse_int8_close(output, lse, reference, reference_lse, *, lse_atol=2e-4):
    reference = to_reference(reference)
    reference_lse = to_reference(reference_lse)
    output = to_cpu(output, reference)
    lse = to_cpu(lse, reference_lse)
    gems_assert_close(output, reference, output.dtype, atol=2e-3)
    gems_assert_equal(torch.isposinf(lse), torch.isposinf(reference_lse))
    finite = torch.isfinite(reference_lse)
    gems_assert_close(lse[finite], reference_lse[finite], torch.float32, atol=lse_atol)


def assert_sparse_int8_reference(packed, skv, output, lse, *, row_indices=None):
    from tests.test_flash_mla_sparse_fwd import FlashmlaSparseTestKit

    q, kv, indices, sm, dv, sink, length = dequantize_sparse_int8_fixture(
        packed, skv, torch.float32
    )
    output = output.flatten(0, 1)
    lse = lse.transpose(1, 2).flatten(0, 1)
    if row_indices is not None:
        q = q[row_indices]
        indices = indices[row_indices]
        output = output[row_indices]
        lse = lse[row_indices]
        if length is not None:
            length = length[row_indices]
    q_ref = to_reference(q)
    kv_ref = to_reference(kv)
    indices_ref = to_reference(indices)
    sink_ref = to_reference(sink)
    length_ref = to_reference(length)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for begin in range(0, q_ref.shape[0], 8):
            part = slice(begin, begin + 8)
            reference, _, reference_lse = (
                FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
                    q_ref[part].shape[0],
                    kv_ref.shape[0],
                    q_ref.shape[1],
                    kv_ref.shape[1],
                    q_ref.shape[2],
                    indices_ref.shape[2],
                    q_ref[part],
                    kv_ref,
                    indices_ref[part],
                    sm,
                    dv,
                    sink_ref,
                    length_ref[part] if length_ref is not None else None,
                )
            )
            assert_sparse_int8_close(output[part], lse[part], reference, reference_lse)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


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


SPARSE_FLASHMLA_SHAPES = (
    [
        (4, 64, 512, 95, 1, 128, 512, False, False, 0),
        (4, 128, 576, 1592, 1, 512, 512, True, True, 3),
        (62, 64, 512, 1521, 1, 512, 512, True, False, 4),
    ]
    if QUICK_MODE
    else [
        (4, 64, 512, 95, 1, 128, 512, False, False, 0),
        (4, 128, 512, 592, 1, 256, 512, True, True, 1),
        (4, 64, 576, 95, 1, 128, 512, False, False, 2),
        (4, 128, 576, 1592, 1, 512, 512, True, True, 3),
        (62, 64, 512, 1521, 1, 512, 512, True, False, 4),
        (62, 128, 576, 98816, 1, 256, 512, False, True, 5),
        (547, 64, 512, 98816, 1, 8320, 512, False, False, 195),
    ]
) + (
    (
        [(16, 64, 512, 675, 1, 640, 512, True, True, 7)]
        if QUICK_MODE
        else [
            (16, 64, 512, 675, 1, 640, 512, True, True, 7),
            (16, 64, 512, 8461, 1, 8320, 512, True, True, 8),
        ]
    )
    if SPARSE_INT8_OP is not None
    else []
)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(not HAS_SPARSE_W8A8, reason="Sparse W8A8 MLA is not registered")
@pytest.mark.parametrize("shape", SPARSE_FLASHMLA_SHAPES)
def test_flash_mla_sparse_flashmla(shape):
    inputs = make_sparse_inputs(shape)
    inputs[2][0].fill_(-1)
    if SPARSE_INT8_OP is not None:
        inputs[2][1, 0, :2] = 1
        packed = quantize_sparse_fixture(*inputs, int8=True)
        packed[6][1, 0, 2] = packed[2].shape[0] * 64
        output, lse = SPARSE_INT8_OP(*packed)
        assert_sparse_int8_reference(packed, inputs[1].shape[0], output, lse)
        return
    reference, _, reference_lse = flashmla_reference().flash_mla_sparse_fwd(*inputs)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8(
        *quantize_sparse_fixture(*inputs)
    )
    assert_sparse_fixture_close(output[:, 0], lse[:, :, 0], reference, reference_lse)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(SPARSE_INT8_OP is None, reason="Sparse INT8 MLA is not registered")
@pytest.mark.parametrize("topk", [0, 65])
def test_sparse_int8_multiquery_fp16(topk):
    # SQ=2 also exercises the head-major LSE layout.
    inputs = make_sparse_inputs((4, 16, 576, 71, 1, max(topk, 1), 512), torch.float16)
    packed = quantize_sparse_fixture(*inputs, int8=True)
    q, rope, k, kr, qs, ks, indices, sm, sink, length = packed
    packed = (
        q.view(2, 2, 16, 512),
        rope.view(2, 2, 16, 64),
        k,
        kr,
        qs.view(2, 2, 16, 1),
        ks,
        indices[..., :topk].view(2, 2, topk),
        sm,
        sink,
        length,
    )
    output, lse = SPARSE_INT8_OP(*packed)
    assert output.dtype == torch.float16 and output.shape == (2, 2, 16, 512)
    assert lse.dtype == torch.float32 and lse.shape == (2, 16, 2)
    if topk == 0:
        expected_out = to_reference(torch.zeros_like(output))
        expected_inf = to_reference(torch.ones_like(lse, dtype=torch.bool))
        gems_assert_equal(to_cpu(output, expected_out), expected_out)
        gems_assert_equal(to_cpu(torch.isposinf(lse), expected_inf), expected_inf)
    else:
        assert_sparse_int8_reference(packed, inputs[1].shape[0], output, lse)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(SPARSE_INT8_OP is None, reason="Sparse INT8 MLA is not registered")
@pytest.mark.parametrize(
    "batch,sq,heads,skv,dtype,clustered,seed",
    [
        # Exercises staged V32 with tile-location ordering, BV=256, SQ=2, and FP16.
        (32, 2, 64, 65536, torch.float16, True, 11),
        # Exercises staged V32 with H=128, BV=128, and bitcast P14 rounding.
        (32, 1, 128, 4096, torch.bfloat16, False, 12),
    ],
)
def test_sparse_int8_v32_staged(batch, sq, heads, skv, dtype, clustered, seed):
    rows = batch * sq
    inputs = make_sparse_inputs(
        (rows, heads, 576, skv, 1, 2048, 512, True, True, seed), dtype=dtype
    )
    q, kv, indices, sm, dv, sink, length = inputs
    if clustered:
        # Nearby indices per tile trigger the KV-location sorting branch.
        offsets = torch.arange(2048, dtype=torch.int32, device=indices.device)
        starts = (
            torch.arange(rows, dtype=torch.int32, device=indices.device) * 31
        ) % skv
        indices = ((starts[:, None] + offsets[None, :]) % skv)[:, None]
        inputs = (q, kv, indices, sm, dv, sink, length)
    inputs[2][0].fill_(-1)
    inputs[2][1, 0, :2] = 7
    packed = quantize_sparse_fixture(*inputs, int8=True)
    q_nope, q_rope, k_lora, k_rope, q_scale, k_scale, idx, sm, sink, length = packed
    idx[1, 0, 2] = k_lora.shape[0] * 64
    packed = (
        q_nope.view(batch, sq, heads, 512),
        q_rope.view(batch, sq, heads, 64),
        k_lora,
        k_rope,
        q_scale.view(batch, sq, heads, 1),
        k_scale,
        idx.view(batch, sq, 2048),
        sm,
        sink,
        length,
    )
    output, lse = SPARSE_INT8_OP(*packed)
    assert output.dtype == dtype and output.shape == (batch, sq, heads, 512)
    assert lse.dtype == torch.float32 and lse.shape == (batch, heads, sq)
    sample_rows = [0, 1, 2, 3, rows // 2, rows - 2, rows - 1]
    assert_sparse_int8_reference(packed, skv, output, lse, row_indices=sample_rows)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(SPARSE_INT8_OP is None, reason="Sparse INT8 MLA is not registered")
def test_sparse_int8_v32_vslice256():
    # H=64, D576, topk=2048 with SKV<65536 and 1024<=rows<4096 exercises VSlice=256.
    rows, skv = 1024, 4096
    inputs = make_sparse_inputs(
        (rows, 64, 576, skv, 1, 2048, 512, True, True, 13), dtype=torch.bfloat16
    )
    inputs[2][0].fill_(-1)
    inputs[2][1, 0, :2] = 5
    packed = quantize_sparse_fixture(*inputs, int8=True)
    packed[6][1, 0, 2] = packed[2].shape[0] * 64
    output, lse = SPARSE_INT8_OP(*packed)
    assert output.shape == (rows, 1, 64, 512)
    assert lse.shape == (rows, 64, 1)
    sample_rows = [0, 1, 2, 511, 512, rows - 2, rows - 1]
    assert_sparse_int8_reference(packed, skv, output, lse, row_indices=sample_rows)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(SPARSE_INT8_OP is None, reason="Sparse INT8 MLA is not registered")
def test_sparse_int8_invalid_softmax_scale():
    inputs = make_sparse_inputs((2, 16, 512, 64, 1, 64, 512, False, False, 14))
    packed = list(quantize_sparse_fixture(*inputs, int8=True))
    with pytest.raises(TypeError):
        SPARSE_INT8_OP(*packed[:7], True, *packed[8:])
    for bad_scale in (0.0, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            SPARSE_INT8_OP(*packed[:7], bad_scale, *packed[8:])
