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

import pytest
import torch

import flaggems_vllm

torch_npu = pytest.importorskip("torch_npu")

HEAD_DIM = 128
DEVICE = "npu"

# Production: lens = recorded per-step token mixes ([8192] full block,
# [8184] spec boundary, [4096,4090] two-request block, [56] chunk tail) over
# the [6220,4,256] compressor cache; hist>0 emulates chunked prefill whose
# window dips into the previous chunk (kernel_details).
# Stress: doubled block, four-way split, ragged pairs, small blocks, T=1.
CASES = (
    # lens, kpool, hist_prefill, pad_slots
    # ---- production shapes (kernel_details) ----
    ([8192], 4, 0, False),  # (lens,kpool,hist,pad) full prefill block
    ([8184], 4, 0, False),  # spec-boundary block
    ([4096, 4090], 4, 0, False),  # two-request block with ragged second
    ([56], 4, 0, False),  # chunked-prefill tail
    # ---- stress shapes ----
    ([16384], 4, 0, False),  # stress: doubled block
    ([2048, 2048, 2048, 2048], 4, 0, False),  # stress: four-way request split
    ([100, 37], 4, 64, True),  # stress: ragged pair, prev-chunk window, PAD slots
    ([64], 4, 0, False),  # stress: small block
    ([1], 4, 0, False),  # stress: single token
    ([8188], 4, 96, False),  # stress: spec boundary + prev-chunk window
)

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="the optimized kpool state compress targets Ascend",
)


def _reference(
    state_cache,
    indexer_cache,
    k,
    gate,
    ape,
    positions,
    cum_q,
    seq_lens,
    state_slots,
    state_bt,
    indexer_slots,
    kpool,
):
    """Vectorised torch reference (the small-op flood the kernel replaced).

    NOTE on provenance: unlike the epilogue UT (whose replaced chain is
    liftable verbatim from the wrapper), the kpool compress flood lives
    inline inside the big vllm-main CompressorStateCache.forward (linear /
    norm / ape / state-cache interleaved with forward_context plumbing) and
    cannot be lifted as a callable.  This reference was written from the
    kernel's semantic contract, cross-checked against an independent
    per-token python reference on the edge cases, and doubles as the perf
    baseline in benchmark/test_kpool_state_compress_perf.py.
    """
    num_tokens, D = k.shape
    state_bs = state_cache.shape[1]
    state_rows = state_cache.view(-1, 2 * HEAD_DIM)
    valid_state = (state_slots >= 0) & (state_slots < state_rows.shape[0])
    safe_state = state_slots.clamp(min=0)
    rows2w = torch.cat([k.float(), gate.float()], dim=-1)
    state_rows[safe_state[valid_state]] = rows2w[valid_state]

    t_idx = torch.arange(num_tokens, device=k.device)
    req = torch.bucketize(t_idx, cum_q.to(torch.int64), right=True).clamp_max(
        cum_q.shape[0] - 1
    )
    prev = torch.where(req > 0, cum_q[req - 1], torch.zeros_like(cum_q[req]))
    rqs = seq_lens[req] - (cum_q[req] - prev)
    j = torch.arange(kpool, device=k.device)
    pool_pos = positions[:, None] - (kpool - 1 - j)[None, :]
    eff = pool_pos.clamp_min(0)
    is_hist = pool_pos < rqs[:, None]
    page = (eff // state_bs).clamp_max(state_bt.shape[1] - 1)
    phys = (
        state_bt[req[:, None], page]
        .long()
        .clamp_min(0)
        .clamp_max(state_cache.shape[0] - 1)
    )
    hist_row = (phys * state_bs + eff % state_bs).view(-1)
    cur_row = (prev[:, None] + eff - rqs[:, None]).clamp(0, num_tokens - 1)
    hist_k = state_rows[hist_row, :HEAD_DIM].view(num_tokens, kpool, HEAD_DIM)
    hist_g = state_rows[hist_row, HEAD_DIM:].view(num_tokens, kpool, HEAD_DIM)
    cur_k = k[cur_row.view(-1)].view(num_tokens, kpool, HEAD_DIM).float()
    cur_g = gate[cur_row.view(-1)].view(num_tokens, kpool, HEAD_DIM).float()
    pk = torch.where(is_hist[..., None], hist_k, cur_k)
    pg = torch.where(is_hist[..., None], hist_g, cur_g)
    w = torch.softmax(pg + ape[:kpool].float(), dim=1)
    out = (w * pk).sum(dim=1)
    islot = indexer_slots
    valid = (islot >= 0) & (islot < indexer_cache.shape[0] * indexer_cache.shape[1])
    indexer_rows = indexer_cache.view(-1, HEAD_DIM)
    indexer_rows[islot[valid]] = out[valid].to(indexer_cache.dtype)


def _make_case(
    lens, kpool=4, state_bs=4, indexer_bs=16, hist_prefill=0, pad_slots=False, seed=0
):
    torch.manual_seed(seed)
    num_tokens = sum(lens)
    state_cache = torch.zeros(
        64, state_bs, 2 * HEAD_DIM, dtype=torch.float32, device=DEVICE
    )
    indexer_cache = torch.zeros(
        64, indexer_bs, 1, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
    )
    k = torch.randn(num_tokens, HEAD_DIM, dtype=torch.float32, device=DEVICE)
    gate = torch.randn(num_tokens, HEAD_DIM, dtype=torch.float32, device=DEVICE)
    ape = torch.randn(kpool, HEAD_DIM, dtype=torch.float32, device=DEVICE)
    positions = torch.cat(
        [torch.arange(length, device=DEVICE) + hist_prefill for length in lens]
    ).long()
    cum_q = torch.tensor(
        list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=DEVICE
    )
    seq_lens = torch.tensor(
        [hist_prefill + length for length in lens], dtype=torch.int32, device=DEVICE
    )
    # slot base skips the hist pre-written rows (a production allocator never
    # overlaps them with the current batch's window reads)
    state_slots = (
        torch.arange(num_tokens, dtype=torch.int64, device=DEVICE) + hist_prefill + 7
    )
    indexer_slots = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE) + 3
    if pad_slots:
        state_slots[0] = -1
        indexer_slots[1] = -1
    state_bt = (
        torch.arange(64, dtype=torch.int32, device=DEVICE)
        .unsqueeze(0)
        .repeat(len(lens), 1)
    )
    for p in range(hist_prefill):
        state_cache[p // state_bs, p % state_bs] = torch.randn(
            2 * HEAD_DIM, device=DEVICE
        )
    return (
        state_cache,
        indexer_cache,
        k,
        gate,
        ape,
        positions,
        cum_q,
        seq_lens,
        state_slots,
        state_bt,
        indexer_slots,
        kpool,
    )


# Production context: GLM-5.3-Flash-W8A8 serving profile (random 16k-in /
# 1k-out @ 4-way concurrency, TP16 + EP16, MTP spec=3).  Shape source:
# kernel_details.csv of the per-rank baseline profiles (aggregated in
# analysis/op_catalog.csv).
# Shape provenance: [8192] / [4096, 4090] are the production prefill token
# mixes over the [6220, 4, 256] compressor cache; the hist_prefill cases
# emulate chunked prefill whose window dips into the previous chunk.


@pytest.mark.kpool_state_compress
@pytest.mark.parametrize("lens,kpool,hist,pad", CASES)
def test_kpool_state_compress_accuracy(lens, kpool, hist, pad):
    args = _make_case(lens, kpool=kpool, hist_prefill=hist, pad_slots=pad)
    (sc, ic, k, g, ape, pos, cq, sl, ss, bt, islt, kp) = args
    ref_sc, ref_ic = sc.clone(), ic.clone()
    _reference(ref_sc, ref_ic, k, g, ape, pos, cq, sl, ss, bt, islt, kp)
    new_sc, new_ic = sc.clone(), ic.clone()
    flaggems_vllm.kpool_state_compress(
        new_sc, new_ic, k, g, ape, pos, cq, sl, ss, bt, islt, kp
    )
    torch.npu.synchronize()
    assert torch.equal(new_sc, ref_sc), "state cache rows must be bitwise"
    torch.testing.assert_close(new_ic, ref_ic, rtol=2e-2, atol=2e-2)
