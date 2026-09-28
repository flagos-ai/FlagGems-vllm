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

try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()
except (ImportError, AttributeError):
    _NPU_AVAILABLE = False

DEVICE = "npu"
HEAD_DIM = 128
POOL_BLOCK = 96  # production cache layout [max_pools, 96, 1, 128]
TRITON_POOL_CHUNK_SIZE = 2048
TRITON_POOL_SUBTILE_SIZE = 128

# Production context: GLM-5.3-Flash-W8A8 serving profile (random 16k-in /
# 1k-out @ 4-way concurrency, TP16 + EP16, MTP spec=3).  Shape source:
# kernel_details.csv of the per-rank baseline profiles (aggregated in
# analysis/op_catalog.csv).
FULL_CASES = (
    # tokens_per_req, n_req, max_pool, pos_base
    # ---- production shapes (kernel_details) ----
    (2048, 4, 6220, 8192),  # (tpr,nreq,pools,pos) production prefill scorer, high pos
    (2048, 1, 6220, 8192),  # single-request production variant
    # ---- stress shapes ----
    # (2048,8,...)/(4096,4,...) would mean 16384 tokens in one step, past the
    # scheduler's 8192 cap and a shape production never issues -- it also
    # faults the vector cores; the 1024x8 / 4096x2 forms keep the high-nreq
    # and high-pos intent within the cap.
    (1024, 8, 6220, 8192),  # stress: 8 requests at high pos
    (4096, 2, 6220, 16384),  # stress: 8192 tokens at very high pos
    (2048, 4, 12440, 8192),  # stress: doubled pools
    (2048, 4, 2049, 8192),  # stress: non-pow2 pools (tail tile partial)
    (300, 3, 777, 1024),  # stress: ragged small pool at mid pos
    (128, 2, 2048, 4096),  # stress: short seqs at high pos
    (64, 4, 1024, 0),  # stress: short seqs from zero pos
    (4, 4, 6220, 8192),  # 16 rows total -> matvec fallback path (decode/draft)
)
CASES = (FULL_CASES[0], FULL_CASES[2], FULL_CASES[6]) if QUICK_MODE else FULL_CASES

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend" or not _NPU_AVAILABLE,
    reason="the optimized indexer GEMM scorer targets Ascend",
)


def build_inputs(tokens_per_req, n_req, max_pool, pos_base=0, seed=0):
    """Construct the full scorer layout: cache pages + block table + varlen."""
    torch.manual_seed(seed)
    total_q = tokens_per_req * n_req
    num_pages = (max_pool + POOL_BLOCK - 1) // POOL_BLOCK
    cache = (
        torch.empty((max_pool, POOL_BLOCK, 1, HEAD_DIM), device=DEVICE)
        .uniform_(-1.0, 1.0)
        .to(torch.bfloat16)
    )
    query = (
        torch.empty((total_q, 4, HEAD_DIM), device=DEVICE)
        .uniform_(-1.0, 1.0)
        .to(torch.bfloat16)
    )
    weights = torch.rand(total_q, 4, device=DEVICE)
    bt = (
        torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
        .unsqueeze(0)
        .repeat(n_req, 1)
        .clone()
    )
    for r in range(n_req):
        bt[r] = (bt[r] + r * 7) % num_pages
    cum_q = torch.arange(
        tokens_per_req,
        (n_req + 1) * tokens_per_req,
        tokens_per_req,
        dtype=torch.int64,
        device=DEVICE,
    )
    seq_lens = torch.full((n_req,), max_pool, dtype=torch.int64, device=DEVICE)
    positions = (
        torch.arange(tokens_per_req, dtype=torch.int64, device=DEVICE)
        .unsqueeze(0)
        .repeat(n_req, 1)
        .reshape(-1)
        + pos_base
    )
    qbar = (query.float() * weights.unsqueeze(-1)).sum(1).contiguous()
    return qbar, cache, cum_q, seq_lens, bt, positions


def _matvec_reference(qbar, cache, cum_q, seq_lens, bt, positions, kpool=4):
    """The exact replaced per-token scorer kernel (same module; doubles as
    the accuracy reference and the perf baseline).  A pure-python loop
    reference is infeasible at the production shape (8192 x 6220)."""
    import importlib

    m = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.ops.indexer_gemm_score"
    )

    num_tokens, head_dim = qbar.shape
    max_pool = int(seq_lens.max().item())
    num_reqs = cum_q.shape[0]
    out = torch.full(
        (num_tokens, max_pool), float("-inf"), dtype=torch.float32, device=qbar.device
    )
    num_chunks = (max_pool + m.TRITON_POOL_CHUNK_SIZE - 1) // m.TRITON_POOL_CHUNK_SIZE
    m._glm5_next_lightning_indexer_score_kernel[(num_tokens, num_chunks)](
        qbar,
        cache,
        cum_q,
        seq_lens,
        bt,
        positions,
        out,
        0,
        max_pool,
        num_reqs,
        cache.shape[0],
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        bt.stride(0),
        bt.stride(1),
        cache.shape[1],
        m._next_power_of_2(max(1, num_reqs)),
        head_dim,
        kpool,
        m.TRITON_POOL_CHUNK_SIZE,
        m.TRITON_POOL_SUBTILE_SIZE,
    )
    return out


def _assert_single(tokens_per_req, n_req, max_pool, pos_base):
    """The actual parity assertions; executed in a fresh interpreter per case."""
    qbar, cache, cum_q, seq_lens, bt, positions = build_inputs(
        tokens_per_req, n_req, max_pool, pos_base
    )
    expected = _matvec_reference(qbar, cache, cum_q, seq_lens, bt, positions)
    actual = flaggems_vllm.indexer_gemm_score(
        qbar, cache, cum_q, seq_lens, bt, positions
    )
    torch.npu.synchronize()
    # -inf mask must match exactly (visibility window)
    assert torch.equal(actual == float("-inf"), expected == float("-inf"))
    m = expected != float("-inf")
    if m.any():
        # finite cells are reduction-order dependent; require top-64 sets to agree
        topk = min(64, int(m.sum(dim=1).max()))
        if topk > 0:
            i1 = torch.topk(expected, topk, dim=1).indices
            i2 = torch.topk(actual, topk, dim=1).indices
            s1 = torch.sort(i1, dim=1).values
            s2 = torch.sort(i2, dim=1).values
            inter = (s1[:, :, None] == s2[:, None, :]).any(-1).sum(1)
            flips = (topk - inter).clamp_min(0)
            assert (
                int(flips.max()) <= 1
            ), f"topk set mismatch: min inter {int(inter.min())}"
        torch.testing.assert_close(actual[m], expected[m], rtol=2e-2, atol=2e-2)


@pytest.mark.indexer_gemm_score
@pytest.mark.parametrize("tokens_per_req,n_req,max_pool,pos_base", CASES)
def test_indexer_gemm_score_accuracy(tokens_per_req, n_req, max_pool, pos_base):
    """In-process by default; set FGEMMS_SUBPROCESS=1 for per-case isolation.

    The triton-ascend runtime on some setups mis-executes the scorer for a
    small shape when a large-pool shape already ran in the same process
    (every case passes individually).  On such a machine export
    FGEMMS_SUBPROCESS=1; each case then runs in a fresh interpreter, at the
    cost of one interpreter startup per case.
    """
    import os

    if not os.environ.get("FGEMMS_SUBPROCESS"):
        _assert_single(tokens_per_req, n_req, max_pool, pos_base)
        return

    import subprocess
    import sys
    from pathlib import Path

    repo_root = str(Path(__file__).resolve().parents[1])
    # tle is absent in some triton builds; stub it before flaggems imports
    code = (
        "import sys, types;"
        "import triton.experimental as _te;"
        "_m = types.ModuleType('triton.experimental.tle');"
        "_l = types.ModuleType('triton.experimental.tle.language');"
        "_m.language = _l;"
        "sys.modules['triton.experimental.tle'] = _m;"
        "sys.modules['triton.experimental.tle.language'] = _l;"
        "_te.tle = _m;"
        "from tests.test_indexer_gemm_score import _assert_single;"
        "_assert_single(*(int(x) for x in sys.argv[1:]))"
    )
    env = dict(os.environ)
    # pytest's `pythonpath = src` lives only inside the pytest process, so
    # hand the child both the repo root (tests package) and src explicitly.
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root, os.path.join(repo_root, "src")]
        + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(tokens_per_req),
            str(n_req),
            str(max_pool),
            str(pos_base),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        cwd=repo_root,
    )
    assert (
        r.returncode == 0
    ), f"case ({tokens_per_req},{n_req},{max_pool},{pos_base}) failed:\n{r.stderr[-800:]}"
