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

"""Accuracy tests for the BNSD FlashAttention forward router.

Correctness is gated against an fp32 head-by-head reference (the ground truth),
not against the vendor fused op: two fp16 kernels can disagree within fp16 noise
while both being correct, so comparing to fp32 is the only oracle that does not
encode one implementation's rounding. The vendor op
(`torch_npu.npu_fusion_attention` -> aclnnFlashAttentionScoreV4) is additionally
checked as a cross-implementation sanity bound.

Lives next to the kernel it tests rather than in the repo's ``tests/`` tree,
because the kernel is 950DT-specific and ships under ``_ascend/a5``. The a5
path is listed in ``pytest.ini``'s ``testpaths`` so a default ``pytest`` run
still collects it; this directory has no ``__init__.py``, so setuptools does
not package it into the wheel.

Run (pick an idle card):

    ASCEND_RT_VISIBLE_DEVICES=7 pytest \\
        src/flaggems_vllm/runtime/backend/_ascend/a5/tests/test_flash_attention_bnsd.py

The benchmark that produces the numbers (latency, speedup, TFLOPS) is
``bench_flash_attention_bnsd.py`` beside this file; its last full sweep is
``20261114_fa_clc_480.csv``.
"""

import math

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime.backend._ascend.a5.ops import flash_attention_fwd
from flaggems_vllm.runtime.backend._ascend.a5.ops.flash_attention_bnsd.fa_naive import (
    SUPPORTED_DIMS,
)

try:
    import torch_npu  # noqa: F401  registers npu device + npu_fusion_attention

    _HAS_NPU_FA = hasattr(torch_npu, "npu_fusion_attention")
except ImportError:  # pragma: no cover - non-Ascend host
    torch_npu = None
    _HAS_NPU_FA = False

device = flaggems_vllm.device

# The full clc.yaml sweep grid, reduced to the kernel's hard constraints: fp16,
# D == 64, BNSD, Sq == Skv, Hq % Hkv == 0, S divisible by the 128 block. The
# raw grid is B x seqlen_pairs x head_dims x head_pairs x causal; filtering to
# D == 64 and symmetric Sq == Skv leaves 240 compatible cases
# (5 B x 6 S x 4 head_pairs x 2 causal), which is what the 950DT report swept.
BATCHES = [1, 4, 8, 16, 32]
SEQ_LENS = [256, 1024, 2048, 4096, 8192, 16384]
HEAD_PAIRS = [(16, 16), (16, 8), (16, 4), (16, 1)]  # Hq, Hkv (GQA ratios)
D = 64

# D=128 runs a *subset of the same grid* rather than a second axis on it: the
# full sweep would double a suite whose fp32 reference is already the slow part
# (at B=32/S=16384 it shrinks the query block to a few rows). The kernels take
# `D` as a `tl.constexpr` and no body is specialised to 64, so covering the dim
# does not need every B/S/GQA combination re-swept.
#
# The subset is drawn from BATCHES/HEAD_PAIRS/SEQ_LENS above and asserted to be
# on-grid, which is the point: an off-grid shape makes the D=128 result
# incomparable to both the D=64 numbers and the clc.yaml report, so "which
# shapes regress at 128" becomes unanswerable. An earlier revision of this list
# was off-grid in 3 of its 4 entries (Hq=8 and B=2 appear nowhere in the grid).
# Verified on 950DT card 7 at max|ours - fp32| = 5.09e-06, ~380x inside the gate.
#
# Derived from the kernel's own list so a newly supported dim cannot be
# added without either getting covered here or failing this import.
D_EXTRA = [d for d in SUPPORTED_DIMS if d != D]
# B, (Hq, Hkv), S — GQA-equal and the widest GQA ratio, S spanning the dense
# autotune gate at 4096 and one value below it. B stays small because the fp32
# reference is what costs here, not the kernel.
DIM_SUBSET_SHAPES = [
    (1, (16, 16), 1024),
    (1, (16, 4), 2048),
    (1, (16, 16), 4096),
    (4, (16, 16), 4096),
]
assert all(
    b in BATCHES and hp in HEAD_PAIRS and s in SEQ_LENS
    for (b, hp, s) in DIM_SUBSET_SHAPES
), "DIM_SUBSET_SHAPES must be a subset of the D=64 grid to stay comparable"

# fp16 attention tolerance. The full 240-case sweep on 950DT peaked at
# max|ours - fp32| = 1.95e-03; 2e-2 is the project's fp16 attention gate and
# leaves comfortable headroom without masking a real regression.
ATOL = 2e-2
RTOL = 2e-2


def _make_input(B, Hq, Hkv, S, dim=None):
    dim = D if dim is None else dim
    g = torch.Generator(device="cpu").manual_seed(0)
    shp_q = (B, Hq, S, dim)
    shp_kv = (B, Hkv, S, dim)
    q = (torch.randn(shp_q, generator=g, dtype=torch.float32) * 0.1).to(torch.float16)
    k = (torch.randn(shp_kv, generator=g, dtype=torch.float32) * 0.1).to(torch.float16)
    v = (torch.randn(shp_kv, generator=g, dtype=torch.float32) * 0.1).to(torch.float16)
    return (t.to(device) for t in (q, k, v))


def _reference_fp32(q, k, v, is_causal):
    """fp32 ground truth, GQA expanded.

    Streams over query blocks so the full S x S score matrix is never
    materialized: at B=32/S=16384 a dense scores tensor would be hundreds of
    GB. Each query block does a numerically-stable softmax over all (causal:
    all visible) keys, which is the exact math, not an online approximation --
    the whole key dimension is reduced at once per block.
    """
    B, Hq, S, Dh = q.shape
    Hkv = k.shape[1]
    scale = 1.0 / math.sqrt(Dh)
    qf, kf, vf = (t.float() for t in (q, k, v))
    if Hq != Hkv:
        rep = Hq // Hkv
        kf = kf.repeat_interleave(rep, dim=1)
        vf = vf.repeat_interleave(rep, dim=1)

    out = torch.empty_like(qf)
    # Bound peak scores memory (B*Hq*qblk*S*4 bytes) to ~2 GB by shrinking the
    # query block for large B*Hq*S; at B=32/Hq=16/S=16384 that forces qblk down
    # to a few rows, which is slow but never OOMs.
    budget = 512 * 1024 * 1024  # 512M fp32 elements ~ 2 GB
    qblk = max(1, min(1024, budget // (B * Hq * S)))
    qblk = (qblk // 128) * 128 or 128  # keep it a block multiple, >= 128
    row_idx = torch.arange(S, device=q.device)
    for s0 in range(0, S, qblk):
        s1 = min(s0 + qblk, S)
        scores = torch.matmul(qf[:, :, s0:s1], kf.transpose(-1, -2)) * scale
        if is_causal:
            q_pos = row_idx[s0:s1][:, None]
            k_pos = row_idx[None, :]
            scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[:, :, s0:s1] = torch.matmul(probs, vf)
    return out


_MASK_CACHE: dict = {}


def _causal_mask_2048(dev):
    key = str(dev)
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = torch.triu(
            torch.ones(2048, 2048, dtype=torch.bool, device=dev), diagonal=1
        )
    return _MASK_CACHE[key]


def _vendor_fa(q, k, v, is_causal):
    """aclnnFlashAttentionScoreV4. Causal uses sparse_mode=2 + compressed mask
    (mode 0 is 1.5-2.2x slower for identical output)."""
    B, Hq, S, Dh = q.shape
    scale = 1.0 / math.sqrt(Dh)
    if not is_causal:
        return torch_npu.npu_fusion_attention(
            q, k, v, head_num=Hq, input_layout="BNSD", scale=scale
        )[0]
    return torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num=Hq,
        input_layout="BNSD",
        scale=scale,
        atten_mask=_causal_mask_2048(q.device),
        sparse_mode=2,
    )[0]


@pytest.mark.parametrize("B", BATCHES)
@pytest.mark.parametrize("Hq,Hkv", HEAD_PAIRS)
@pytest.mark.parametrize("S", SEQ_LENS)
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attention_bnsd_accuracy(B, Hq, Hkv, S, is_causal):
    q, k, v = _make_input(B, Hq, Hkv, S)
    out = flash_attention_fwd(q, k, v, is_causal=is_causal)
    ref = _reference_fp32(q, k, v, is_causal)
    assert out.shape == ref.shape
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("dim", D_EXTRA)
@pytest.mark.parametrize("B,head_pair,S", DIM_SUBSET_SHAPES)
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attention_bnsd_accuracy_dim(B, head_pair, S, is_causal, dim):
    """Same accuracy gate as the D=64 sweep, at the other supported head dim.

    D=64 covers these same shapes through the full 240-case sweep above, so
    only the extra dims are run here -- but on the same shapes, so the two
    dims' results are directly comparable.
    """
    Hq, Hkv = head_pair
    q, k, v = _make_input(B, Hq, Hkv, S, dim=dim)
    out = flash_attention_fwd(q, k, v, is_causal=is_causal)
    ref = _reference_fp32(q, k, v, is_causal)
    assert out.shape == ref.shape
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("dim", SUPPORTED_DIMS)
def test_flash_attention_bnsd_tile_configs(dim):
    """Every tile config must be correct at every supported dim, not just the
    one autotune happens to pick.

    `(64, 128)` dispatches to `_attn_fwd_dense_grp2` (GROUP=2); the other two
    go to `_attn_fwd_dense`. Pinning each one keeps a tile-specific miscompile
    from hiding behind the autotune choice -- the grouped kernel writes its two
    KV blocks out as flat code, so it is the arm most likely to drift.

    This runs at **D=64 as well**, which the 240-case sweep above does not
    cover: that sweep always takes the autotune's pick, so until this was
    parametrized over `SUPPORTED_DIMS` the shipping dim's non-default tiles
    (including the grp2 arm, which `S % 256 == 0` selects at D=64 too) were
    never pinned by any test.
    """
    B, Hq, Hkv, S = 1, 16, 16, 1024
    q, k, v = _make_input(B, Hq, Hkv, S, dim=dim)
    ref = _reference_fp32(q, k, v, False)
    for bm, bn in ((128, 128), (64, 256), (64, 128)):
        if S % bm or S % bn:
            continue
        out = flash_attention_fwd(q, k, v, is_causal=False, block_m=bm, block_n=bn)
        torch.testing.assert_close(
            out.float(),
            ref,
            atol=ATOL,
            rtol=RTOL,
            msg=lambda m, bm=bm, bn=bn: f"D={dim} BLOCK_M={bm} BLOCK_N={bn}: {m}",
        )


def test_flash_attention_bnsd_rejects_unsupported_dim():
    """An unsupported head dim must be rejected by name, not by an assert deep
    in the kernel or a silent wrong answer."""
    q, k, v = _make_input(1, 8, 8, 256, dim=32)
    with pytest.raises((ValueError, AssertionError)):
        flash_attention_fwd(q, k, v, is_causal=False)


@pytest.mark.skipif(
    not _HAS_NPU_FA, reason="torch_npu.npu_fusion_attention unavailable"
)
@pytest.mark.parametrize("dim", SUPPORTED_DIMS)
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attention_bnsd_vs_vendor_dim(is_causal, dim):
    """Cross-check against the vendor op at every supported dim, so a shared
    misreading of the layout cannot pass by matching our own reference.

    D=64 is included rather than left to `..._vs_vendor` below: that one sweeps
    its own (B, heads, S) list, so running the same single shape across dims
    keeps this an apples-to-apples dim comparison on its own.
    """
    q, k, v = _make_input(1, 16, 16, 2048, dim=dim)
    out = flash_attention_fwd(q, k, v, is_causal=is_causal)
    vendor = _vendor_fa(q, k, v, is_causal)
    ref = _reference_fp32(q, k, v, is_causal)
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(vendor.float(), ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(out, vendor, atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(
    not _HAS_NPU_FA, reason="torch_npu.npu_fusion_attention unavailable"
)
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("Hq,Hkv", [(16, 16), (16, 4)])
@pytest.mark.parametrize("S", [512, 2048])
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attention_bnsd_vs_vendor(B, Hq, Hkv, S, is_causal):
    q, k, v = _make_input(B, Hq, Hkv, S)
    out = flash_attention_fwd(q, k, v, is_causal=is_causal)
    vendor = _vendor_fa(q, k, v, is_causal)
    # Both are fp16; compare against fp32 reference individually and to each
    # other within fp16 noise.
    ref = _reference_fp32(q, k, v, is_causal)
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(vendor.float(), ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(out, vendor, atol=ATOL, rtol=RTOL)


def test_flash_attention_bnsd_rejects_combine_batch():
    q, k, v = _make_input(1, 16, 16, 256)
    with pytest.raises(TypeError, match="combine_batch"):
        flash_attention_fwd(q, k, v, is_causal=False, combine_batch=True)
