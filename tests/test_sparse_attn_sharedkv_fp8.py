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

"""Accuracy and input gates of the FP8 SparseAttnSharedKV decode path.

Each case is built twice from one set of values: as the packed FP8 cache this
path reads, and as the exact BF16 decoding of those same pages, which the
vLLM-Ascend AscendC operator reads as the reference. An e4m3 value times a
power-of-two scale is representable in BF16, so both see identical numbers and
only the bytes differ -- 584 per token against 1024.

The batches straddle the dispatch boundaries so every shape is exercised:
split-K below batch four, the request pipelines to batch 31, the dense pipeline
from batch 32, and the window pipeline from batch eight. One case carries a
compressed prefix wider than the 512-wide index list, which only a CFA
call can have.

`benchmark/test_sparse_attn_sharedkv_fp8_perf.py` measures the same cases and
takes them, and the inputs they build, from here.
"""

from __future__ import annotations

import dataclasses
import math
from functools import lru_cache

import pytest
import torch

import flaggems_vllm
from tests.sparse_attn_sharedkv_utils import metadata_operator

SEED = 43
Q_HEADS = 64
KV_HEADS = 1
HEAD_DIM = 512
NOPE_BYTES = 448
# e4m3's largest finite magnitude, the divisor the ue8m0 scale rule uses.
# Numerically equal to NOPE_BYTES, which is a byte count: keep them apart.
FP8_MAX = 448.0
SCALE_GROUPS = 7
PAGE_SIZE = 128
TOKEN_RECORD = 576
PAGE_STRIDE = (
    (PAGE_SIZE * (TOKEN_RECORD + 8) + TOKEN_RECORD - 1) // TOKEN_RECORD * TOKEN_RECORD
)
SOFTMAX_SCALE = HEAD_DIM**-0.5
WINDOW = 128
ATOL = RTOL = 2e-2


@dataclasses.dataclass(frozen=True)
class Fp8DecodeCase:
    name: str
    mode: str
    batch: int
    kv_len: int
    cmp_ratio: int = 1
    topk: int = 512

    @property
    def has_cmp(self) -> bool:
        return self.cmp_ratio > 1

    @property
    def has_sparse(self) -> bool:
        return self.mode == "SCFA"


CASES = (
    Fp8DecodeCase("swa_b001", "SWA", 1, 8193),
    Fp8DecodeCase("swa_b008", "SWA", 8, 8193),
    Fp8DecodeCase("swa_b032", "SWA", 32, 8193),
    Fp8DecodeCase("scfa_b002", "SCFA", 2, 8193, 4),
    Fp8DecodeCase("scfa_b008", "SCFA", 8, 8193, 4),
    Fp8DecodeCase("scfa_b032", "SCFA", 32, 8193, 4),
    Fp8DecodeCase("cfa_b008", "CFA", 8, 8193, 128),
    Fp8DecodeCase("cfa_b032", "CFA", 32, 8193, 128),
    # A prefix wider than the index list CFA does not use: 70000 / 128 = 546
    # columns, so a path that sized itself by the index list would drop the tail.
    Fp8DecodeCase("cfa_long_prefix", "CFA", 2, 70000, 128),
    Fp8DecodeCase("short_window", "SWA", 4, 70),
    Fp8DecodeCase("mixed_lengths", "SCFA", 5, 4096, 4),
)


@lru_cache(maxsize=1)
def official_operator():
    """The vLLM-Ascend AscendC operator used as the reference."""
    try:
        import vllm_ascend.utils

        if not vllm_ascend.utils.enable_custom_op():
            return None
        return torch.ops._C_ascend.npu_sparse_attn_sharedkv
    except (ImportError, AttributeError):
        return None


def pack_fp8(values: torch.Tensor) -> torch.Tensor:
    """[pages, PAGE_SIZE, HEAD_DIM] BF16 -> the packed FP8 pages, on the host.

    The NoPE half is quantised per 64-value group with a power-of-two (ue8m0)
    scale; the RoPE half is already BF16 and is copied. The scales go in the
    page's trailing block, eight bytes per token.
    """
    pages = values.shape[0]
    groups = (
        values[:, :, :NOPE_BYTES].float().reshape(pages, PAGE_SIZE, SCALE_GROUPS, 64)
    )
    peak = torch.clamp_min(groups.abs().amax(-1) / FP8_MAX, 1e-4)
    scales = torch.exp2(peak.log2().ceil())
    quantized = (groups / scales[..., None]).reshape(pages, PAGE_SIZE, NOPE_BYTES)
    raw = torch.zeros((pages, PAGE_STRIDE), dtype=torch.uint8)
    body = raw[:, : PAGE_SIZE * TOKEN_RECORD].view(pages, PAGE_SIZE, TOKEN_RECORD)
    body[:, :, :NOPE_BYTES] = quantized.to(torch.float8_e4m3fn).view(torch.uint8)
    body[:, :, NOPE_BYTES:] = values[:, :, NOPE_BYTES:].contiguous().view(torch.uint8)
    scale_block = raw[:, PAGE_SIZE * TOKEN_RECORD : PAGE_SIZE * (TOKEN_RECORD + 8)]
    scale_block.view(pages, PAGE_SIZE, 8)[:, :, :SCALE_GROUPS] = (
        scales.log2() + 127
    ).to(torch.uint8)
    return raw


def unpack_fp8(raw: torch.Tensor) -> torch.Tensor:
    """The exact BF16 decoding of packed pages, with integer operations only.

    e4m3's sign, exponent and mantissa land in BF16's fields unchanged, and the
    ue8m0 scale is a power of two, so nothing rounds.
    """
    pages = raw.shape[0]
    body = raw[:, : PAGE_SIZE * TOKEN_RECORD].reshape(pages, PAGE_SIZE, TOKEN_RECORD)
    scale_block = raw[:, PAGE_SIZE * TOKEN_RECORD : PAGE_SIZE * (TOKEN_RECORD + 8)]
    exponent = scale_block.reshape(pages, PAGE_SIZE, 8)[:, :, :SCALE_GROUPS]
    scale = (exponent.to(torch.int32) << 23).view(torch.float32)
    code = body[:, :, :NOPE_BYTES].to(torch.int32)
    bits = ((code & 0x80) << 24) | ((code & 0x7F) << 20)
    nope = bits.view(torch.float32).reshape(pages, PAGE_SIZE, SCALE_GROUPS, 64)
    nope = nope * 2.0**120
    nope = (nope * scale[..., None]).reshape(pages, PAGE_SIZE, NOPE_BYTES)
    rope = body[:, :, NOPE_BYTES:].contiguous().view(torch.bfloat16)
    return torch.cat((nope.to(torch.bfloat16), rope), -1)


def _paged_values(token_count: int, generator: torch.Generator) -> torch.Tensor:
    page_count = max(1, math.ceil(token_count / PAGE_SIZE))
    values = torch.randn(
        (page_count, PAGE_SIZE, HEAD_DIM), generator=generator, dtype=torch.float32
    )
    return (values / 10).clamp_(-1, 1).to(torch.bfloat16)


def make_inputs(case: Fp8DecodeCase, device: str = "npu") -> dict[str, object]:
    """The FP8 inputs, plus `bf16_caches` for the reference call."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    lengths = [case.kv_len - 37 * index for index in range(case.batch)]
    lengths = [max(1, length) for length in lengths]

    q = torch.randn(
        (case.batch, Q_HEADS, HEAD_DIM), generator=generator, dtype=torch.float32
    )
    q = q.clamp_(-1, 1).to(torch.bfloat16).to(device)

    win_values = _paged_values(PAGE_SIZE * 2 * case.batch, generator)
    win_table = torch.zeros(
        (case.batch, max(1, math.ceil(max(lengths) / PAGE_SIZE))), dtype=torch.int32
    )
    for request, length in enumerate(lengths):
        first = max(0, length - WINDOW) // PAGE_SIZE
        last = (length - 1) // PAGE_SIZE
        win_table[request, first] = 2 * request
        win_table[request, last] = 2 * request + 1

    cmp_kv = cmp_table = cmp_indices = None
    cmp_values = None
    if case.has_cmp:
        cmp_lengths = [length // case.cmp_ratio for length in lengths]
        cmp_tokens = PAGE_SIZE * 2 * case.batch + max(cmp_lengths)
        cmp_values = _paged_values(cmp_tokens, generator)
        columns = max(1, math.ceil(max(cmp_lengths) / PAGE_SIZE))
        cmp_table = torch.zeros((case.batch, columns), dtype=torch.int32)
        indices = torch.full((case.batch, KV_HEADS, case.topk), -1, dtype=torch.int32)
        for request, length in enumerate(cmp_lengths):
            needed = max(1, math.ceil(length / PAGE_SIZE))
            cmp_table[request, :needed] = torch.randperm(
                cmp_values.shape[0], generator=generator, dtype=torch.int32
            )[:needed]
            count = min(case.topk, length)
            if count and case.has_sparse:
                picked = torch.randperm(length, generator=generator, dtype=torch.int32)
                indices[request, 0, :count] = picked[:count].sort().values
        if case.has_sparse:
            cmp_indices = indices.to(device)
        cmp_kv = pack_fp8(cmp_values).to(device)

    sinks = torch.randn((Q_HEADS,), generator=generator, dtype=torch.float32)
    inputs = {
        "q": q,
        "ori_kv": pack_fp8(win_values).to(device),
        "ori_block_table": win_table.to(device),
        "cmp_kv": cmp_kv,
        "cmp_block_table": None if cmp_table is None else cmp_table.to(device),
        "cmp_sparse_indices": cmp_indices,
        "cu_seqlens_q": torch.arange(case.batch + 1, dtype=torch.int32, device=device),
        "seqused_kv": torch.tensor(lengths, dtype=torch.int32, device=device),
        "sinks": sinks.to(device),
        "softmax_scale": SOFTMAX_SCALE,
        "cmp_ratio": case.cmp_ratio,
        "ori_mask_mode": 4,
        "cmp_mask_mode": 3,
        "ori_win_left": WINDOW - 1,
        "ori_win_right": 0,
        "layout_q": "TND",
        "layout_kv": "PA_ND",
    }
    bf16 = {"ori_kv": unpack_fp8(pack_fp8(win_values)).unsqueeze(2).to(device)}
    if case.has_cmp:
        bf16["cmp_kv"] = unpack_fp8(pack_fp8(cmp_values)).unsqueeze(2).to(device)
    return inputs, bf16


def shape_label(case: Fp8DecodeCase) -> str:
    return (
        f"{case.mode}/TND/FP8; Q=[{case.batch},{Q_HEADS},{HEAD_DIM}], "
        f"KV={case.kv_len}, ratio={case.cmp_ratio}, topk={case.topk}"
    )


def reference_call(case: Fp8DecodeCase, inputs, bf16):
    """The AscendC operator on the exact BF16 decoding of the same pages."""
    metadata = metadata_operator()(
        num_heads_q=Q_HEADS,
        num_heads_kv=KV_HEADS,
        head_dim=HEAD_DIM,
        cu_seqlens_q=inputs["cu_seqlens_q"],
        cu_seqlens_ori_kv=inputs["cu_seqlens_q"],
        seqused_q=torch.ones(case.batch, dtype=torch.int32, device=inputs["q"].device),
        seqused_kv=inputs["seqused_kv"],
        batch_size=case.batch,
        max_seqlen_q=1,
        max_seqlen_kv=int(inputs["seqused_kv"].max()),
        cmp_topk=case.topk if case.has_sparse else 0,
        cmp_ratio=case.cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=inputs["ori_win_left"],
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=case.has_cmp,
        device=str(inputs["q"].device),
    )
    arguments = dict(inputs)
    arguments.update(bf16, metadata=metadata)
    if not case.has_sparse:
        arguments.pop("cmp_sparse_indices", None)
    return official_operator()(arguments.pop("q"), **arguments)[0]


_IS_ASCEND = flaggems_vllm.vendor_name == "ascend" and hasattr(torch, "npu")
pytestmark = [
    pytest.mark.sparse_attn_sharedkv_fp8,
    pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU"),
]
npu_only = pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU")


def _module():
    return __import__(
        "flaggems_vllm.runtime.backend._ascend.ops.sparse_attn_sharedkv_fp8",
        fromlist=["sparse_attn_sharedkv_fp8"],
    )


@npu_only
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@torch.inference_mode()
def test_sparse_attn_sharedkv_fp8_accuracy(case):
    if official_operator() is None or metadata_operator() is None:
        pytest.skip("requires the vLLM-Ascend SparseAttnSharedKV reference")
    inputs, bf16 = make_inputs(case, flaggems_vllm.device)
    expected = reference_call(case, inputs, bf16)
    actual, lse = flaggems_vllm.sparse_attn_sharedkv(**inputs)
    torch.npu.synchronize()
    torch.testing.assert_close(
        actual.float().cpu(), expected.float().cpu(), atol=ATOL, rtol=RTOL
    )
    assert lse.numel() == 0 and lse.dtype == torch.float32
    assert actual.shape == inputs["q"].shape and actual.dtype == torch.bfloat16


@npu_only
@torch.inference_mode()
def test_sparse_attn_sharedkv_fp8_is_selected_by_the_cache():
    """A packed FP8 cache reaches the FP8 path through the public entry."""
    module = _module()
    inputs, _ = make_inputs(CASES[0], flaggems_vllm.device)
    assert module.is_packed_fp8_cache(inputs["ori_kv"])
    assert not module.is_packed_fp8_cache(
        torch.empty((2, 128, 1, 512), dtype=torch.bfloat16, device=flaggems_vllm.device)
    )


def test_pack_and_unpack_are_exact():
    """e4m3 times a power-of-two scale survives the round trip in BF16."""
    generator = torch.Generator(device="cpu").manual_seed(7)
    values = torch.randn((3, 128, 512), generator=generator)
    values = (values / 10).clamp_(-1, 1).to(torch.bfloat16)
    packed = pack_fp8(values)
    assert packed.shape == (3, PAGE_STRIDE) and packed.dtype == torch.uint8
    decoded = unpack_fp8(packed)
    # The RoPE half is copied, so it must come back bit for bit.
    assert torch.equal(decoded[:, :, 448:], values[:, :, 448:])
    assert torch.equal(decoded, unpack_fp8(pack_fp8(decoded)))


def test_sparse_attn_sharedkv_fp8_rejects_cpu():
    module = _module()
    with pytest.raises(NotImplementedError, match="Ascend NPU"):
        module.sparse_attn_sharedkv_fp8(
            torch.empty((1, 64, 512), dtype=torch.bfloat16),
            ori_kv=torch.empty((1, PAGE_STRIDE), dtype=torch.uint8),
            ori_block_table=torch.zeros((1, 1), dtype=torch.int32),
            cu_seqlens_q=torch.zeros((2,), dtype=torch.int32),
            seqused_kv=torch.ones((1,), dtype=torch.int32),
            sinks=torch.zeros((64,), dtype=torch.float32),
        )


@npu_only
@torch.inference_mode()
def test_sparse_attn_sharedkv_fp8_rejects_lse_request():
    inputs, _ = make_inputs(CASES[0], flaggems_vllm.device)
    with pytest.raises(NotImplementedError, match="LSE"):
        flaggems_vllm.sparse_attn_sharedkv(**inputs, return_softmax_lse=True)
