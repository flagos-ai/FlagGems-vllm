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

from .conftest import QUICK_MODE

torch_npu = pytest.importorskip("torch_npu")

# Production context: GLM-5.3-Flash-W8A8 serving profile (random 16k-in /
# 1k-out @ 4-way concurrency, TP16 + EP16, MTP spec=3).  Shape source:
# kernel_details.csv of the per-rank baseline profiles (aggregated in
# analysis/op_catalog.csv).
# NOTE: unlike the perf benchmark (3-tuple nreq/maxtok/bs), the accuracy UT
# pins per-request tlen so the varlen packing arithmetic is exercised.
FULL_CASES = (
    # ---- production shapes (kernel_details) ----
    # reqs, tokens, maxtokens, blocksize
    (4, 200, 8192, 128),  # 4 reqs x 200 tok, MLA KV table (bs=128), maxtok 8192
    (8, 200, 8192, 128),  # 8 reqs, recorded upper end of concurrent requests
    (12, 200, 8192, 128),  # 12 reqs, recorded mid point
    (16, 200, 8192, 128),  # 16 reqs, full c16 concurrency point
    (4, 200, 8192, 16),  # compressor-state table (bs=16) at c4
    # ---- stress shapes ----
    (32, 200, 8192, 128),  # 32 reqs: max_num_seqs boundary
    (128, 64, 8192, 128),  # 128 reqs, decode-phase form (128x64=8192 tokens)
    (8, 200, 8192, 16),  # compressor table at higher concurrency
    (8, 200, 16384, 128),  # doubled maxtok deployment
    (1, 200, 8192, 128),  # single request
)
CASES = (FULL_CASES[0], FULL_CASES[2]) if QUICK_MODE else FULL_CASES

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="the optimized slot mapping targets Ascend",
)


def _ref_slot_mapping(
    num_tokens, qsl_host, positions, block_table, block_size, pad_id=-1
):
    """Per-request torch reference of the stock kernel's semantics."""
    out = torch.full((len(positions),), pad_id, dtype=torch.int64)
    nreq = len(qsl_host) - 1
    for r in range(nreq):
        for t in range(qsl_host[r], qsl_host[r + 1]):
            pos = int(positions[t])
            blk = pos // block_size
            num = int(block_table[r][blk])
            out[t] = num * block_size + (pos % block_size)
    return out


def _load_upstream_kernel() -> bool:
    """Load the exact replaced per-request kernel from the vllm package when
    present; extension-free environments fall back to the torch reference."""
    global _upstream_slot_kernel
    try:
        from vllm.v1.worker.block_table import _compute_slot_mapping_kernel

        _upstream_slot_kernel = _compute_slot_mapping_kernel
        return True
    except Exception:
        return False


HAS_UPSTREAM = _load_upstream_kernel()


@pytest.mark.slot_mapping
@pytest.mark.parametrize("nreq,tlen,maxtok,bs", CASES)
def test_slot_mapping_accuracy(nreq, tlen, maxtok, bs):
    torch.manual_seed(0)
    lens = [tlen] * nreq
    qsl_host = [0]
    for n in lens:
        qsl_host.append(qsl_host[-1] + n)
    qsl = torch.tensor(qsl_host, dtype=torch.int32, device="npu")
    pos = torch.cat([torch.arange(tlen, dtype=torch.int32) for _ in range(nreq)]).to(
        "npu"
    )
    # block-table depth must cover the largest position any case reaches
    # (nreq=128 stress reaches position ~25.6k at tlen=200)
    max_pos = nreq * tlen
    maxb = (max_pos + bs - 1) // bs + 1
    btab = (torch.arange(nreq * maxb, dtype=torch.int32).reshape(nreq, maxb) % 900).to(
        "npu"
    )
    expected = _ref_slot_mapping(sum(lens), qsl_host, pos.cpu(), btab.cpu(), bs).to(
        "npu"
    )

    out = torch.full((maxtok,), -1, dtype=torch.int64, device="npu")
    flaggems_vllm.compute_slot_mapping_parallel(
        sum(lens), maxtok, qsl, nreq, pos, btab, out, block_size=bs
    )
    torch.npu.synchronize()
    assert torch.equal(out[: sum(lens)], expected)


@pytest.mark.slot_mapping
@pytest.mark.skipif(not HAS_UPSTREAM, reason="vllm package not available")
@pytest.mark.parametrize("nreq,tlen,maxtok,bs", CASES)
def test_slot_mapping_matches_upstream_kernel(nreq, tlen, maxtok, bs):
    """Bitwise parity against the exact replaced per-request kernel."""
    torch.manual_seed(0)
    lens = [tlen] * nreq
    qsl_host = [0]
    for n in lens:
        qsl_host.append(qsl_host[-1] + n)
    qsl = torch.tensor(qsl_host, dtype=torch.int32, device="npu")
    pos = torch.cat([torch.arange(tlen, dtype=torch.int32) for _ in range(nreq)]).to(
        "npu"
    )
    # block-table depth must cover the largest position any case reaches
    # (nreq=128 stress reaches position ~25.6k at tlen=200)
    max_pos = nreq * tlen
    maxb = (max_pos + bs - 1) // bs + 1
    btab = (torch.arange(nreq * maxb, dtype=torch.int32).reshape(nreq, maxb) % 900).to(
        "npu"
    )

    old = torch.full((maxtok,), -1, dtype=torch.int64, device="npu")
    _upstream_slot_kernel[(nreq + 1,)](
        sum(lens),
        maxtok,
        qsl,
        pos,
        btab,
        btab.stride(0),
        bs,
        old,
        BLOCK_SIZE=1024,
        TOTAL_CP_WORLD_SIZE=1,
        TOTAL_CP_RANK=0,
        CP_KV_CACHE_INTERLEAVE_SIZE=1,
        PAD_ID=-1,
    )
    new = torch.full((maxtok,), -1, dtype=torch.int64, device="npu")
    flaggems_vllm.compute_slot_mapping_parallel(
        sum(lens), maxtok, qsl, nreq, pos, btab, new, block_size=bs
    )
    torch.npu.synchronize()
    assert torch.equal(old[: sum(lens)], new[: sum(lens)])
