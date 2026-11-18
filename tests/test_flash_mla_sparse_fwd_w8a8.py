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
from functools import lru_cache, partial
from pathlib import Path
from types import ModuleType

import pytest
import torch

import flaggems_vllm
from tests.test_flash_mla_with_kvcache_w8a8 import SUPPORTED as HAS_FP8_MLA
from tests.test_flash_mla_with_kvcache_w8a8 import quantize_dense_fp8

from .accuracy_utils import gems_assert_close, gems_assert_equal, to_cpu, to_reference

HAS_INT8_MLA = (
    flaggems_vllm.vendor_name == "metax"
    and "flash_mla_sparse_fwd_w8a8"
    in dict(flaggems_vllm.runtime.backend.get_customized_ops())
    and torch.cuda.is_available()
    and "C550" in torch.cuda.get_device_name()
)
requires_int8_mla = pytest.mark.skipif(
    not HAS_INT8_MLA, reason="INT8 MLA requires MetaX C550"
)


def prepare_sparse_int8(*args, **kwargs):
    from flaggems_vllm.runtime.backend._metax.ops.flash_mla_sparse_fwd_w8a8 import (
        _prepare_flash_mla_sparse_fwd_w8a8,
    )

    return _prepare_flash_mla_sparse_fwd_w8a8(*args, **kwargs)


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


def quantize_sparse_fixture(
    q, kv, indices, sm_scale, value_dim, attn_sink=None, topk_length=None
):
    if value_dim != 512 or kv.shape[1] != 1:
        raise ValueError("sparse W8A8 MLA requires value_dim=512 and one KV head")
    tokens, _, dims = kv.shape
    pages = math.ceil(tokens / 64)
    cache = kv[:, 0]
    if tokens % 64:
        cache = torch.nn.functional.pad(cache, (0, 0, 0, pages * 64 - tokens))
    quantize = (
        quantize_w8a8 if HAS_INT8_MLA else partial(quantize_dense_fp8, dequantize=False)
    )
    q_nope, q_rope, q_scale, native_q = quantize(q)
    k_nope, k_rope, k_scale, native_kv = quantize(cache)
    packed = (
        q_nope[:, None],
        q_rope[:, None],
        k_nope.view(pages, 64, 512),
        k_rope.view(pages, 64, dims - 512),
        q_scale[:, None],
        k_scale.view(pages, 64, 1),
        torch.where((indices >= 0) & (indices < tokens), indices, -1),
        sm_scale,
        attn_sink,
        topk_length,
    )
    return packed, native_q, None if native_kv is None else native_kv[:, None]


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
@pytest.mark.skipif(
    not (HAS_FP8_MLA or HAS_INT8_MLA), reason="Sparse MLA requires Hopper or C550"
)
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
    packed, _, _ = quantize_sparse_fixture(*inputs)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8(*packed)
    if HAS_INT8_MLA:
        check_sparse_int8(packed, (output, lse))
        return
    reference, _, reference_lse = flashmla_reference().flash_mla_sparse_fwd(*inputs)
    assert_sparse_fixture_close(output[:, 0], lse[:, :, 0], reference, reference_lse)


def check_sparse_int8(packed, *outputs):
    torch.backends.cuda.matmul.allow_tf32 = False
    q, qr, kv, kr, qs, ks, indices, scale, sink, lengths = packed
    rows = torch.linspace(
        0, q.shape[0] - 1, min(16, q.shape[0]), device=q.device
    ).long()
    expected = reference(
        (q[rows], qr[rows], kv, kr, qs[rows], ks, indices[rows]),
        attn_sink=sink,
        topk_length=None if lengths is None else lengths[rows],
        softmax_scale=scale,
    )
    for out, lse in outputs:
        assert_accuracy((out[rows], lse[rows]), expected)


def quantize_w8a8(x: torch.Tensor):
    scale = x[..., :512].float().abs().amax(-1, keepdim=True) / 127.0
    scale = torch.where(scale > 0, torch.exp2(torch.ceil(torch.log2(scale))), 1.0)
    nope = (x[..., :512].float() / scale).round().to(torch.int8)
    rope = (x[..., 512:].float() / scale).to(torch.bfloat16)
    native = (torch.cat((nope.float(), rope.float()), -1) * scale).to(torch.bfloat16)
    return nope, rope, scale, native


def make_int8_inputs(
    batch, queries, heads, topk, pages=32, dtype=torch.bfloat16, rope_dim=64
):
    shape = (
        batch * queries,
        heads,
        512 + rope_dim,
        pages * 64,
        1,
        topk,
        512,
        False,
        False,
        42,
    )
    packed, _, _ = quantize_sparse_fixture(*make_sparse_inputs(shape))
    q, qr, kv, kr, qs, ks, indices = packed[:7]
    return [
        q.view(batch, queries, heads, 512),
        qr.view(batch, queries, heads, rope_dim).to(dtype),
        kv,
        kr.to(dtype),
        qs.view(batch, queries, heads, 1),
        ks,
        indices.view(batch, queries, topk),
    ]


def reference(inputs, *, attn_sink=None, topk_length=None, softmax_scale=None):
    q, qr, kv, kr, qs, ks, indices = inputs
    batch, queries, heads, _ = q.shape
    dims = 512 + qr.shape[-1]
    query = (torch.cat((q.float(), qr.float()), -1) * qs).view(-1, heads, dims)
    cache = (torch.cat((kv.float(), kr.float()), -1) * ks).view(-1, dims)
    indices = indices.flatten(0, 1)
    valid = (indices >= 0) & (indices < cache.shape[0])
    if topk_length is not None:
        positions = torch.arange(indices.shape[-1], device=indices.device)
        valid &= positions < topk_length.reshape(-1, 1)
    gathered = cache.index_select(0, indices.masked_fill(~valid, 0).flatten()).view(
        batch * queries, indices.shape[-1], dims
    )
    logits = query @ gathered.transpose(1, 2)
    logits *= dims**-0.5 if softmax_scale is None else softmax_scale
    logits.masked_fill_(~valid[:, None, :], float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    normalizer = lse if attn_sink is None else torch.logaddexp(lse, attn_sink)
    normalizer = normalizer.masked_fill(torch.isneginf(normalizer), float("inf"))
    output = torch.exp(logits - normalizer.unsqueeze(-1)) @ gathered[..., :512]
    lse = lse.masked_fill(torch.isneginf(lse), float("inf"))
    return output.view(batch, queries, heads, 512), lse.view(
        batch, queries, heads
    ).transpose(1, 2)


def assert_accuracy(actual, expected):
    output, lse = actual
    ref_output, ref_lse = map(to_reference, expected)
    output, lse = to_cpu(output, ref_output), to_cpu(lse, ref_lse)
    gems_assert_close(output, ref_output, output.dtype, atol=2e-3)
    gems_assert_equal(torch.isposinf(lse), torch.isposinf(ref_lse))
    finite = torch.isfinite(ref_lse)
    gems_assert_close(lse[finite], ref_lse[finite], torch.float32, atol=2e-4)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@requires_int8_mla
@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("queries", [1, 2])
@pytest.mark.parametrize("heads", [16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sparse_decode(queries, heads, dtype, rope_dim):
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = make_int8_inputs(4, queries, heads, 129, dtype=dtype, rope_dim=rope_dim)
    inputs[4].mul_(1.3)
    inputs[4][2].mul_(256)
    inputs[5].mul_(0.7)
    indices = inputs[-1]
    indices[0].fill_(-1)
    indices[1, :, ::3] = inputs[2].shape[0] * 64
    indices[2, :, 1] = indices[2, :, 0]
    lengths = torch.full((4, queries), 97, dtype=torch.int32, device=inputs[0].device)
    lengths[-1] = 0
    sink = torch.randn(heads, dtype=torch.float32, device=inputs[0].device)
    sink[0], sink[1] = float("inf"), -float("inf")
    out = torch.empty((4, queries, heads, 512), dtype=dtype, device=inputs[0].device)
    _, actual = prepare_sparse_int8(
        *inputs,
        topk_length=lengths,
        attn_sink=sink,
        out=out,
        softmax_scale=0.125,
    )
    assert actual[0] is out
    assert_accuracy(
        actual,
        reference(inputs, topk_length=lengths, attn_sink=sink, softmax_scale=0.125),
    )


@pytest.mark.flash_mla_sparse_fwd_w8a8
@requires_int8_mla
@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("topk", [0, 1, 64, 65, 512, 2048, 8320])
def test_sparse_decode_lengths(topk, rope_dim):
    inputs = make_int8_inputs(2, 1, 64, topk, rope_dim=rope_dim)
    if topk > 0:
        inputs[-1][0].fill_(-1)
        inputs[-1][1, :, ::4] = 2147483647
    if topk == 8320:
        inputs[4].mul_(256)
    handle, _ = prepare_sparse_int8(*inputs)
    actual = handle()
    if topk == 0:
        gems_assert_equal(actual[0], torch.zeros_like(actual[0]))
        gems_assert_equal(actual[1], torch.full_like(actual[1], float("inf")))
    else:
        assert_accuracy(actual, reference(inputs))


@pytest.mark.flash_mla_sparse_fwd_w8a8
@requires_int8_mla
@pytest.mark.parametrize("empty", ["batch", "heads", "cache"])
def test_sparse_decode_empty(empty):
    inputs = make_int8_inputs(2, 1, 64, 65)
    if empty == "batch":
        for index in (0, 1, 4, 6):
            inputs[index] = inputs[index][:0]
    elif empty == "heads":
        for index in (0, 1, 4):
            inputs[index] = inputs[index][:, :, :0]
    else:
        for index in (2, 3, 5):
            inputs[index] = inputs[index][:0]
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8(*inputs)
    gems_assert_equal(output, torch.zeros_like(output))
    gems_assert_equal(lse, torch.full_like(lse, float("inf")))


@pytest.mark.flash_mla_sparse_fwd_w8a8
@requires_int8_mla
@pytest.mark.parametrize(
    ("topk", "has_lengths"), [(65, True), (128, False), (192, False)]
)
def test_sparse_decode_large_batch(topk, has_lengths):
    batch = 257
    inputs = make_int8_inputs(batch, 1, 64, topk, rope_dim=0)
    indices = inputs[-1]
    indices[0].fill_(-1)
    indices[1, :, ::3] = inputs[2].shape[0] * 64
    indices[2, :, 1] = indices[2, :, 0]
    if topk == 192:
        # Exercise an empty final page and a query with only a valid final page.
        indices[3, :, -64:].fill_(-1)
        indices[4, :, :-64].fill_(-1)
    lengths = (
        torch.full((batch, 1), topk - 1, dtype=torch.int32, device=indices.device)
        if has_lengths
        else None
    )
    actual = flaggems_vllm.flash_mla_sparse_fwd_w8a8(*inputs, topk_length=lengths)
    assert_accuracy(actual, reference(inputs, topk_length=lengths))


@pytest.mark.flash_mla_sparse_fwd_w8a8
@requires_int8_mla
@pytest.mark.parametrize("batch, topk", [(8, 4096), (16, 640), (257, 4096)])
def test_sparse_raw_selected_partitions(batch, topk):
    inputs = make_int8_inputs(batch, 1, 64, topk, pages=max(32, topk // 32), rope_dim=0)
    indices = inputs[-1]
    indices[0].fill_(-1)
    indices[1, :, :-64].fill_(-1)
    indices[2, :, 1] = indices[2, :, 0]
    indices[2, :, -4] = inputs[2].shape[0] * 64
    inputs[4][2].mul_(8)
    actual = flaggems_vllm.flash_mla_sparse_fwd_w8a8(*inputs, softmax_scale=0.5)
    first_rows = [
        item[:4] if i in (0, 1, 4, 6) else item for i, item in enumerate(inputs)
    ]
    assert_accuracy(
        (actual[0][:4], actual[1][:4]), reference(first_rows, softmax_scale=0.5)
    )


@pytest.mark.flash_mla_sparse_fwd_w8a8
@requires_int8_mla
@pytest.mark.parametrize("topk, variable_lengths", [(8192, False), (4096, True)])
def test_sparse_value_split_merge(topk, variable_lengths):
    # For D512, 33 rows use the cached path instead of the fused path.
    batch = 33
    pages = 32 if variable_lengths else 128
    inputs = make_int8_inputs(batch, 1, 64, topk, pages=pages, rope_dim=0)
    lengths = None
    if variable_lengths:
        lengths = torch.full(
            (batch,), pages * 64, dtype=torch.int32, device=inputs[0].device
        )
        lengths[0], lengths[1] = 1023, 0
    packed = (*inputs, 0.5, None, lengths)
    check_sparse_int8(packed, flaggems_vllm.flash_mla_sparse_fwd_w8a8(*packed))
