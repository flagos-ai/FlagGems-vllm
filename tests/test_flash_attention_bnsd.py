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

Run (device 5 is the idle card on this host; adjust as needed):

    ASCEND_RT_VISIBLE_DEVICES=5 pytest tests/test_flash_attention_bnsd.py
"""

import math

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime.backend._ascend.ops import flash_attention_fwd

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

# fp16 attention tolerance. The full 240-case sweep on 950DT peaked at
# max|ours - fp32| = 1.95e-03; 2e-2 is the project's fp16 attention gate and
# leaves comfortable headroom without masking a real regression.
ATOL = 2e-2
RTOL = 2e-2


def _make_input(B, Hq, Hkv, S):
    g = torch.Generator(device="cpu").manual_seed(0)
    shp_q = (B, Hq, S, D)
    shp_kv = (B, Hkv, S, D)
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
            torch.ones(2048, 2048, dtype=torch.bool, device=dev), diagonal=1)
    return _MASK_CACHE[key]


def _vendor_fa(q, k, v, is_causal):
    """aclnnFlashAttentionScoreV4. Causal uses sparse_mode=2 + compressed mask
    (mode 0 is 1.5-2.2x slower for identical output)."""
    B, Hq, S, Dh = q.shape
    scale = 1.0 / math.sqrt(Dh)
    if not is_causal:
        return torch_npu.npu_fusion_attention(
            q, k, v, head_num=Hq, input_layout="BNSD", scale=scale)[0]
    return torch_npu.npu_fusion_attention(
        q, k, v, head_num=Hq, input_layout="BNSD", scale=scale,
        atten_mask=_causal_mask_2048(q.device), sparse_mode=2)[0]


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


@pytest.mark.skipif(not _HAS_NPU_FA, reason="torch_npu.npu_fusion_attention unavailable")
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
