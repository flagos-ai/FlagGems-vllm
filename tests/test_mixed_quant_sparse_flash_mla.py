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

"""Accuracy tests for the Ascend A5 sparse MLA decode attention op.

The reference is a torch implementation of the op's own definition — online
softmax in fp32 over the dequantized cache, with the probabilities cast to
bf16 before the PV product — evaluated on the host.  The vendor op cannot serve
as a value reference in this call form: its ``quant_mode=1`` cache is
608B/token with group size 64 and a bf16 rope segment, so it dequantizes
different bytes.  ``benchmark/bench_mixed_quant_sparse_flash_mla.py`` compares
the two on latency only, for the same reason.

Run on an idle 950DT, for example:

    ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=src pytest -q \
        tests/test_mixed_quant_sparse_flash_mla.py
"""

import math

import pytest
import torch

from . import conftest as cfg

torch_npu = pytest.importorskip("torch_npu")

from flaggems_vllm.runtime.backend._ascend.a5.ops.mixed_quant_sparse_flash_mla import (  # noqa: E402,E501
    MixedQuantSparseFlashMlaPlan,
    mixed_quant_sparse_flash_mla,
)

D = 512
N1 = 64
N_GROUPS = 16
GROUP_SIZE = 32
KV_ROW_BYTES = 544
FP8_MAX = 448.0

# T1 selects the dispatch path: 1 -> scattered split-K, 2 and 4 -> gather plus
# split-K attention, 8 -> gather plus plain contiguous attention.
if cfg.QUICK_MODE:
    T1_CASES = [1, 8]
else:
    T1_CASES = [1, 2, 4, 8]

_PATHS = ["gather_contig", "gather_contig_splitk", "scattered_splitk", "scattered"]

# The op and the reference run the same arithmetic in a different tile order,
# so they differ only by fp32 reassociation and the bf16 probability cast.
_MAX_REL = 0.05
_MIN_COS = 0.999


def _quantize(kv_f32):
    """Per-group-of-32 FP8 E4M3 quantization with bf16 scales."""
    s = kv_f32.shape[0]
    grp = kv_f32.reshape(s, N_GROUPS, GROUP_SIZE)
    amax = grp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (amax / FP8_MAX).to(torch.bfloat16)
    normalized = (grp / scale.float()).clamp(-FP8_MAX, FP8_MAX)
    return normalized.to(torch.float8_e4m3fn).reshape(s, D), scale.reshape(s, N_GROUPS)


def _dequantize(fp8_data, scale_bf16):
    s = scale_bf16.shape[0]
    k = fp8_data.float().to(torch.bfloat16).reshape(s, N_GROUPS, GROUP_SIZE)
    deq = k * scale_bf16.float().reshape(s, N_GROUPS, 1)
    return deq.to(torch.bfloat16).reshape(s, D)


def _pack_cache(fp8_data, scale_bf16, block_num, block_size, perm):
    """Scatter the quantized rows into a PA_BBND uint8 cache."""
    cache = torch.zeros(block_num, block_size, 1, KV_ROW_BYTES, dtype=torch.uint8)
    flat = cache.view(-1, KV_ROW_BYTES)
    rows = (
        perm.long()[:, None] * block_size + torch.arange(block_size)[None, :]
    ).reshape(-1)
    rows = rows[: fp8_data.shape[0]]
    flat[rows, :D] = fp8_data.view(torch.uint8)
    flat[rows, D : D + N_GROUPS * 2] = scale_bf16.contiguous().view(torch.uint8)
    return cache


def build_inputs(
    t1,
    k=2048,
    block_size=64,
    seqlen_kv=8192,
    *,
    with_sinks=True,
    with_topk=True,
    topk_width=None,
    negative_indices=False,
    seed=42,
):
    """Build one ORI_SPARSE decode call on the host."""
    gen = torch.Generator().manual_seed(seed)
    n_blk = seqlen_kv // block_size
    block_num = n_blk + 16

    q = (torch.randn(t1, N1, D, generator=gen) * 0.5).to(torch.bfloat16)
    sinks = torch.randn(N1, generator=gen).float() if with_sinks else None
    kv_f32 = torch.randn(seqlen_kv, D, generator=gen) * 0.5
    fp8_data, scale_bf16 = _quantize(kv_f32)

    perm = torch.randperm(block_num, generator=gen)[:n_blk].to(torch.int32)
    cache = _pack_cache(fp8_data, scale_bf16, block_num, block_size, perm)
    block_table = perm.reshape(1, n_blk).expand(t1, -1).contiguous()

    idx_gen = torch.Generator().manual_seed(seed * 1000003 + 17)
    indices = (torch.rand(t1, k, generator=idx_gen) * seqlen_kv).to(torch.int32)
    if negative_indices:
        # A skipped slot inside the valid prefix, which the kernels mask out.
        indices[:, 1::7] = -1

    if with_topk:
        width = k if topk_width is None else topk_width
        topk = torch.full((t1,), width, dtype=torch.int32)
    else:
        topk = None

    return {
        "q": q,
        "ori_kv": cache,
        "ori_sparse_indices": indices,
        "ori_block_table": block_table,
        "ori_topk_length": topk,
        "sinks": sinks,
        "decoded_kv": _dequantize(fp8_data, scale_bf16),
        "softmax_scale": 1.0 / math.sqrt(D),
    }


def golden(data, tile=128):
    """Online-softmax reference over the dequantized cache, on the host."""
    q = data["q"]
    decoded_kv = data["decoded_kv"]
    indices = data["ori_sparse_indices"]
    topk = data["ori_topk_length"]
    sinks = data["sinks"]
    scale = data["softmax_scale"]

    t1, n1, d = q.shape
    k = indices.shape[1]
    out = torch.empty_like(q)
    lse = torch.empty(t1, n1, dtype=torch.float32)

    for row in range(t1):
        qr = q[row].float()
        if sinks is None:
            row_max = torch.full((n1,), float("-inf"), dtype=torch.float32)
            row_sum = torch.zeros(n1, dtype=torch.float32)
        else:
            row_max = sinks.clone()
            row_sum = torch.ones(n1, dtype=torch.float32)
        acc = torch.zeros((n1, d), dtype=torch.float32)

        width = k if topk is None else int(topk[row])
        taken = indices[row, :width].long()
        taken = taken[taken >= 0]

        for start in range(0, len(taken), tile):
            kv = decoded_kv[taken[start : start + tile]].float()
            scores = (qr @ kv.T) * scale
            tile_max = scores.amax(dim=-1).clamp_min(0.0)
            new_max = torch.maximum(row_max, tile_max)
            new_max_safe = torch.where(
                new_max == float("-inf"), torch.zeros_like(new_max), new_max
            )
            rescale = torch.exp(row_max - new_max_safe)
            rescale = torch.where(
                row_max == float("-inf"), torch.zeros_like(rescale), rescale
            )
            probs = torch.exp(scores - new_max_safe[:, None])
            row_sum = rescale * row_sum + probs.sum(dim=-1)
            acc = acc * rescale[:, None] + probs.to(torch.bfloat16).float() @ kv
            row_max = new_max

        safe_sum = torch.where(row_sum > 0.0, row_sum, torch.ones_like(row_sum))
        out[row] = (acc / safe_sum[:, None]).to(torch.bfloat16)
        lse[row] = row_max + torch.log(safe_sum)

    return out, lse


def _to_device(data, device):
    moved = {}
    for name in (
        "q",
        "ori_kv",
        "ori_sparse_indices",
        "ori_block_table",
        "ori_topk_length",
        "sinks",
    ):
        value = data[name]
        moved[name] = None if value is None else value.to(device)
    return moved


def _check(actual, expected, name="out", max_rel=_MAX_REL, min_cos=_MIN_COS):
    actual = actual.cpu().float()
    expected = expected.cpu().float()
    assert actual.shape == expected.shape, name
    assert torch.isfinite(actual).all(), f"{name}: non-finite values"

    mask = expected.abs() > 0.01
    rel = (
        ((actual[mask] - expected[mask]).abs() / expected[mask].abs()).max().item()
        if mask.any()
        else 0.0
    )
    cos = torch.nn.functional.cosine_similarity(
        actual.reshape(1, -1), expected.reshape(1, -1)
    ).item()
    assert rel <= max_rel, f"{name}: max relative diff {rel:.4%}"
    assert cos >= min_cos, f"{name}: cosine similarity {cos:.6f}"


def _run(data, **kwargs):
    args = _to_device(data, "npu")
    return mixed_quant_sparse_flash_mla(
        args["q"],
        ori_kv=args["ori_kv"],
        ori_sparse_indices=args["ori_sparse_indices"],
        ori_block_table=args["ori_block_table"],
        ori_topk_length=args["ori_topk_length"],
        sinks=args["sinks"],
        softmax_scale=data["softmax_scale"],
        **kwargs,
    )


@pytest.mark.parametrize("t1", T1_CASES)
def test_dispatch_paths(t1):
    data = build_inputs(t1)
    expected, _ = golden(data)
    out, lse = _run(data)
    _check(out, expected)
    assert lse.numel() == 0


@pytest.mark.parametrize("path", _PATHS)
def test_every_path_agrees(path):
    data = build_inputs(2)
    expected, _ = golden(data)
    out, _ = _run(data, path=path)
    _check(out, expected, name=path)


@pytest.mark.parametrize("block_size", [16, 64, 128])
def test_page_sizes(block_size):
    data = build_inputs(4, block_size=block_size)
    expected, _ = golden(data)
    out, _ = _run(data)
    _check(out, expected)


@pytest.mark.parametrize("k", [256, 1024, 2048])
def test_sparse_widths(k):
    data = build_inputs(2, k=k)
    expected, _ = golden(data)
    out, _ = _run(data)
    _check(out, expected)


@pytest.mark.parametrize(
    "with_sinks,with_topk,topk_width,negative_indices",
    [
        (False, True, None, False),  # no sink bias
        (True, False, None, False),  # no valid-prefix length
        (True, True, 1024, False),  # half the slots unused
        (True, True, None, True),  # skipped slots inside the prefix
        (False, False, None, True),  # neither optional input
    ],
)
def test_optional_inputs(with_sinks, with_topk, topk_width, negative_indices):
    data = build_inputs(
        4,
        with_sinks=with_sinks,
        with_topk=with_topk,
        topk_width=topk_width,
        negative_indices=negative_indices,
    )
    expected, _ = golden(data)
    out, _ = _run(data)
    _check(out, expected)


@pytest.mark.parametrize("t1", [1, 4])
def test_softmax_lse(t1):
    data = build_inputs(t1)
    expected, expected_lse = golden(data)
    out, lse = _run(data, return_softmax_lse=True)
    _check(out, expected)
    assert lse.shape == (1, t1, N1)
    assert lse.dtype == torch.float32
    _check(lse.reshape(t1, N1), expected_lse, name="lse", max_rel=1e-3)


def test_all_slots_masked_out():
    # Every index skipped: only the sink's own term remains, which carries no
    # value, so the output is exactly zero rather than undefined.
    data = build_inputs(2, k=256)
    data["ori_sparse_indices"] = torch.full_like(data["ori_sparse_indices"], -1)
    expected, _ = golden(data)
    assert torch.count_nonzero(expected) == 0
    out, _ = _run(data)
    assert torch.count_nonzero(out) == 0


def test_reused_plan_is_stable():
    data = build_inputs(2)
    expected, _ = golden(data)
    args = _to_device(data, "npu")
    plan = MixedQuantSparseFlashMlaPlan(
        args["q"],
        ori_kv=args["ori_kv"],
        ori_sparse_indices=args["ori_sparse_indices"],
        ori_block_table=args["ori_block_table"],
        ori_topk_length=args["ori_topk_length"],
        sinks=args["sinks"],
        softmax_scale=data["softmax_scale"],
    )
    assert plan.path == "gather_contig_splitk"
    for _ in range(3):
        out, _ = plan(
            args["q"],
            ori_kv=args["ori_kv"],
            ori_sparse_indices=args["ori_sparse_indices"],
            ori_block_table=args["ori_block_table"],
            ori_topk_length=args["ori_topk_length"],
            sinks=args["sinks"],
        )
        _check(out, expected)


def test_caller_supplied_output_buffer():
    data = build_inputs(2)
    expected, _ = golden(data)
    args = _to_device(data, "npu")
    plan = MixedQuantSparseFlashMlaPlan(
        args["q"],
        ori_kv=args["ori_kv"],
        ori_sparse_indices=args["ori_sparse_indices"],
        ori_block_table=args["ori_block_table"],
        ori_topk_length=args["ori_topk_length"],
        sinks=args["sinks"],
        softmax_scale=data["softmax_scale"],
    )
    out = torch.empty(2, N1, D, dtype=torch.bfloat16, device="npu")
    returned, _ = plan(
        args["q"],
        ori_kv=args["ori_kv"],
        ori_sparse_indices=args["ori_sparse_indices"],
        ori_block_table=args["ori_block_table"],
        ori_topk_length=args["ori_topk_length"],
        sinks=args["sinks"],
        out=out,
    )
    assert returned.data_ptr() == out.data_ptr()
    _check(out, expected)


def test_unsupported_call_forms_raise():
    data = build_inputs(1, k=256)
    args = _to_device(data, "npu")
    common = dict(
        ori_kv=args["ori_kv"],
        ori_sparse_indices=args["ori_sparse_indices"],
        ori_block_table=args["ori_block_table"],
        ori_topk_length=args["ori_topk_length"],
        sinks=args["sinks"],
    )

    with pytest.raises(NotImplementedError, match="quant_mode 2"):
        mixed_quant_sparse_flash_mla(args["q"], quant_mode=2, **common)
    with pytest.raises(NotImplementedError, match="compressed KV side"):
        mixed_quant_sparse_flash_mla(args["q"], cmp_kv=args["ori_kv"], **common)
    with pytest.raises(NotImplementedError, match="544B FP8 rows|544 bytes"):
        narrow = dict(common, ori_kv=args["ori_kv"][..., :512].contiguous())
        mixed_quant_sparse_flash_mla(args["q"], **narrow)


def test_invalid_input_raises():
    data = build_inputs(1, k=256)
    args = _to_device(data, "npu")
    common = dict(
        ori_kv=args["ori_kv"],
        ori_sparse_indices=args["ori_sparse_indices"],
        ori_block_table=args["ori_block_table"],
        ori_topk_length=args["ori_topk_length"],
        sinks=args["sinks"],
    )

    with pytest.raises(ValueError, match="q must be bfloat16"):
        mixed_quant_sparse_flash_mla(args["q"].to(torch.float16), **common)
    with pytest.raises(ValueError, match="sinks must be float32"):
        mixed_quant_sparse_flash_mla(
            args["q"], **dict(common, sinks=args["sinks"].to(torch.bfloat16))
        )
    with pytest.raises(ValueError, match="multiple of block_s"):
        mixed_quant_sparse_flash_mla(args["q"], block_s=100, **common)
    with pytest.raises(ValueError, match="multiple of block_h"):
        mixed_quant_sparse_flash_mla(args["q"], block_h=24, **common)
    with pytest.raises(ValueError, match="path must be one of"):
        mixed_quant_sparse_flash_mla(args["q"], path="cv", **common)
    with pytest.raises(ValueError, match="ori_topk_length must be int32"):
        mixed_quant_sparse_flash_mla(
            args["q"], **dict(common, ori_topk_length=args["ori_topk_length"].long())
        )
    with pytest.raises(ValueError, match="plan is for a different call"):
        plan = MixedQuantSparseFlashMlaPlan(args["q"], **common)
        plan(torch.cat([args["q"], args["q"]]), **common)
