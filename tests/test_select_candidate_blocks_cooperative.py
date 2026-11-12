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

import inspect

import pytest
import torch
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.ops.select_candidate_blocks_cooperative import (
    supports_cooperative_candidates,
)

from . import conftest as cfg

select_candidate_blocks_cooperative = flaggems_vllm.select_candidate_blocks_cooperative
device = flaggems_vllm.device


def _has_histogram_mask():
    # The top-k step reuses persistent_topk, which needs tl.histogram(mask=...).
    if not hasattr(tl, "histogram"):
        return False
    try:
        return "mask" in inspect.signature(tl.histogram).parameters
    except (ValueError, TypeError):
        return False


pytestmark = pytest.mark.skipif(
    not _has_histogram_mask(),
    reason="tl.histogram with mask parameter not available",
)

try:
    from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.candidate_topk import (  # noqa: E501
        select_candidate_blocks_cooperative as _vllm_op,
    )
    from vllm.models.deepseek_v41.nvidia.specialized_decode.attention.candidate_topk import (  # noqa: E501
        supports_cooperative_candidates as _vllm_supports,
    )

    HAS_VLLM = True
except Exception:  # ImportError, missing _C ops, missing workspace manager...
    _vllm_op = None
    _vllm_supports = None
    HAS_VLLM = False


def _torch_ref(logits, row_ends, block_size, k):
    """Reference semantics of vLLM's select_candidate_blocks (decode, no starts)."""
    rows, width = logits.shape
    nblocks = (width + block_size - 1) // block_size
    out = torch.full((rows, k), -1, dtype=torch.int32, device=logits.device)
    scores = torch.full(
        (rows, nblocks), float("-inf"), dtype=torch.float32, device=logits.device
    )
    for r in range(rows):
        end = min(int(row_ends[r]), width)
        if end <= 0:
            continue
        nvalid = (end + block_size - 1) // block_size
        row = torch.full(
            (nvalid * block_size,), float("-inf"), dtype=torch.float32, device=device
        )
        row[:end] = logits[r, :end]
        blk = row.view(nvalid, block_size)
        s = blk.amax(dim=1)
        s = torch.where(blk.isnan().any(dim=1), torch.full_like(s, float("nan")), s)
        s[(int(row_ends[r]) - 1) // block_size] = float("inf")
        scores[r, :nvalid] = s
        take = min(k, nblocks)
        vals, idx = scores[r].topk(take)
        idx = torch.where(vals != float("-inf"), idx, torch.full_like(idx, -1))
        out[r, :take] = idx.to(torch.int32)
    return out, scores


def _make_inputs(rows, width, ends, data="random", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    if data == "random":
        logits = torch.randn(rows, width, generator=g, device=device)
    elif data == "many_ties":
        logits = (
            torch.randint(0, 8, (rows, width), generator=g, device=device).float() / 8
        )
    else:
        raise ValueError(data)
    # Junk past the row end must be ignored.
    for r, e in enumerate(ends):
        logits[r, max(e, 0) :] = 1e30
    row_ends = torch.tensor(ends, dtype=torch.int32, device=device)
    return logits, row_ends


def _run_gems(logits, row_ends, block_size, k):
    out = torch.full((logits.shape[0], k), 12345, dtype=torch.int32, device=device)
    select_candidate_blocks_cooperative(logits, row_ends, block_size, out)
    return out


def _check_against_ref(out, ref, scores, row_ends, block_size, k):
    nblocks = scores.shape[1]
    for r in range(out.shape[0]):
        got = out[r][out[r] >= 0].long()
        exp = ref[r][ref[r] >= 0].long()
        # Ids are unique and in range.
        assert got.unique().numel() == got.numel(), f"row {r}: duplicate ids"
        assert bool((got < nblocks).all()), f"row {r}: id out of range"
        # Same set size; same multiset of selected scores (ties at the
        # boundary may pick different ids with equal score).
        assert got.numel() == exp.numel(), f"row {r}: {got.numel()} vs {exp.numel()}"
        sg = scores[r].gather(0, got).sort(descending=True)[0]
        se = scores[r].gather(0, exp).sort(descending=True)[0]
        torch.testing.assert_close(sg, se, equal_nan=True, rtol=0, atol=0)
        # Newest block is always kept.
        end = int(row_ends[r])
        if end > 0:
            assert (end - 1) // block_size in got.tolist(), f"row {r}: newest dropped"


# DeepSeek-V4.1 decode uses candidate_block_size=8 over max_model_len-wide
# logits (131072 blocks at a 1M context) and k=candidate_topk_blocks=2048; see
# vLLM specialized_decode/attention/indexer.py and its test_kernels.py.
if cfg.QUICK_MODE:
    SHAPES = [
        # rows, width, block_size, k, row_ends kind
        (4, 131072 * 8, 8, 2048, "mixed"),
        (32, 16384 * 8, 8, 1024, "long"),
        (64, 32768 * 8, 8, 512, "mixed"),
    ]
    DATA = ["random"]
else:
    SHAPES = [
        (1, 131072 * 8, 8, 2048, "long"),
        (1, 131072 * 8, 8, 2048, "short"),
        (6, 131072 * 8, 8, 2048, "mixed"),
        (4, 131072 * 8, 8, 512, "mixed"),
        (3, 131072 * 8, 8, 1024, "mixed"),
        (16, 32768 * 8, 8, 2048, "long"),
        (32, 16384 * 8, 8, 1024, "long"),
        (33, 65536 * 8, 8, 2048, "mixed"),
        (64, 131072 * 8, 8, 2048, "mixed"),
        (64, 4096 * 8, 8, 512, "long"),
        # other block sizes, incl. a non-power-of-two one
        (8, 65536, 32, 1024, "mixed"),
        (4, 96000, 48, 512, "mixed"),
    ]
    DATA = ["random", "many_ties"]


def _ends(kind, rows, width, block_size, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    if kind == "short":
        return [min(width, block_size * k // 2 + 7)] * rows
    if kind == "long":
        return [width - int(torch.randint(0, block_size, (1,), generator=g))] * rows
    ends = []
    for r in range(rows):
        choice = r % 5
        if choice == 0:
            ends.append(0)  # empty row
        elif choice == 1:
            ends.append(1)  # single token
        elif choice == 2:
            ends.append(block_size * k - 3)  # just under k blocks
        elif choice == 3:
            ends.append(block_size * (k + 1) + 5)  # just over k blocks
        else:
            ends.append(int(torch.randint(1, width + 1, (1,), generator=g)))
    return [min(e, width) for e in ends]


@pytest.mark.select_candidate_blocks_cooperative
@pytest.mark.parametrize("rows, width, block_size, k, kind", SHAPES)
@pytest.mark.parametrize("data", DATA)
@torch.inference_mode()
def test_select_candidate_blocks_cooperative_vs_ref(
    rows, width, block_size, k, kind, data
):
    ends = _ends(kind, rows, width, block_size, k)
    logits, row_ends = _make_inputs(rows, width, ends, data)
    out = _run_gems(logits, row_ends, block_size, k)
    ref, scores = _torch_ref(logits, row_ends, block_size, k)
    _check_against_ref(out, ref, scores, row_ends, block_size, k)


@pytest.mark.select_candidate_blocks_cooperative
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM candidate_topk unavailable")
@pytest.mark.parametrize("rows, width, block_size, k, kind", SHAPES)
@torch.inference_mode()
def test_select_candidate_blocks_cooperative_vs_vllm(rows, width, block_size, k, kind):
    ends = _ends(kind, rows, width, block_size, k)
    logits, row_ends = _make_inputs(rows, width, ends, "random")
    nblocks = (width + block_size - 1) // block_size
    if not _vllm_supports(rows, nblocks, k):
        pytest.skip("shape outside vLLM support")
    expected = torch.empty((rows, k), dtype=torch.int32, device=device)
    try:
        _vllm_op(logits, row_ends, block_size, expected)
    except (RuntimeError, AssertionError, NotImplementedError) as e:
        pytest.skip(f"vLLM op not runnable here: {e}")
    out = _run_gems(logits, row_ends, block_size, k)
    _, scores = _torch_ref(logits, row_ends, block_size, k)
    _check_against_ref(out, expected, scores, row_ends, block_size, k)


NATIVE_CASES = [
    # Long rows around the 16384-block register-path limit and a short one.
    ((16384 * 8, 16384 * 8 - 3, 16385 * 8 + 1, 3000 * 8 + 5, 20000 * 8, 1), 2048),
    ((0, 100 * 8 + 3, 2048 * 8, 512 * 8), 512),
    ((131072 * 8, 131070, 9000 * 8 + 2), 1024),
]


def _capture(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


@pytest.mark.select_candidate_blocks_cooperative
@pytest.mark.parametrize("ends, k", NATIVE_CASES)
def test_select_candidate_blocks_cooperative_native_cases(ends, k):
    """Mirror of vLLM's test_cooperative_candidates_match_stock: exact set
    equality with the stock selection, pinned lowest block, -inf blocks as -1,
    stale in-range ids overwritten, and CUDA-graph replay on fresh logits."""
    block, nblocks = 8, 131072
    rows = len(ends)
    logits = torch.empty(rows, nblocks * block, device=device)
    row_ends = torch.tensor(ends, dtype=torch.int32, device=device)
    workspace = torch.empty(1 << 20, dtype=torch.uint8, device=device)

    def fill_logits(seed):
        torch.manual_seed(seed)
        logits.normal_()
        logits[0, : 50 * block] = float("-inf")
        end = ends[-1]
        logits[-1, (end - 1) // block * block : end] = -100.0  # lowest, but pinned

    def cooperative():
        out = torch.full((rows, k), 7, dtype=torch.int32, device=device)
        select_candidate_blocks_cooperative(
            logits, row_ends, block, out, workspace=workspace
        )
        return out

    def assert_same_sets(got):
        want, _ = _torch_ref(logits, row_ends, block, k)
        assert torch.equal(got.sort(dim=1).values, want.sort(dim=1).values)

    with torch.inference_mode():
        fill_logits(0)
        assert_same_sets(cooperative())
    graph, out = _capture(cooperative)
    fill_logits(1)
    graph.replay()
    torch.cuda.synchronize()
    assert_same_sets(out)


@pytest.mark.select_candidate_blocks_cooperative
@torch.inference_mode()
def test_select_candidate_blocks_cooperative_nan_and_strided():
    rows, width, block_size, k = 4, 65536, 64, 512
    base = torch.randn(rows, width * 2, device=device)
    logits = base[:, ::2]  # non-unit column stride
    logits[0, 100] = float("nan")
    logits[1, 5000:5100] = float("nan")
    ends = [width, width - 17, block_size * 300, 64]
    row_ends = torch.tensor(ends, dtype=torch.int32, device=device)
    out = _run_gems(logits, row_ends, block_size, k)
    ref, scores = _torch_ref(logits.contiguous(), row_ends, block_size, k)
    _check_against_ref(out, ref, scores, row_ends, block_size, k)


@pytest.mark.select_candidate_blocks_cooperative
@torch.inference_mode()
def test_select_candidate_blocks_cooperative_short_rows_padding():
    rows, width, block_size, k = 3, 65536, 64, 512
    ends = [0, 1, 64 * 10 + 1]
    logits, row_ends = _make_inputs(rows, width, ends)
    out = _run_gems(logits, row_ends, block_size, k)
    assert bool((out[0] == -1).all())
    assert sorted(out[1][out[1] >= 0].tolist()) == [0]
    assert sorted(out[2][out[2] >= 0].tolist()) == list(range(11))
    assert int((out[2] == -1).sum()) == k - 11


@pytest.mark.select_candidate_blocks_cooperative
def test_select_candidate_blocks_cooperative_supports():
    assert supports_cooperative_candidates(1, 2048, 512)
    assert supports_cooperative_candidates(64, 16384, 2048)
    assert not supports_cooperative_candidates(0, 2048, 512)
    assert not supports_cooperative_candidates(65, 2048, 512)
    assert not supports_cooperative_candidates(1, 2048, 256)
    assert not supports_cooperative_candidates(1, 2046, 512)


@pytest.mark.select_candidate_blocks_cooperative
@pytest.mark.parametrize(
    "rows, width, k, dtype",
    [
        (65, 131072, 512, torch.float32),  # too many rows
        (4, 131072, 256, torch.float32),  # unsupported k
        (4, 131072 - 64 * 2, 512, torch.float32),  # nblocks % 4 != 0
        (4, 131072, 512, torch.bfloat16),  # unsupported dtype
    ],
)
def test_select_candidate_blocks_cooperative_unsupported(rows, width, k, dtype):
    logits = torch.zeros(rows, width, dtype=dtype, device=device)
    row_ends = torch.full((rows,), width, dtype=torch.int32, device=device)
    out = torch.empty((rows, k), dtype=torch.int32, device=device)
    with pytest.raises(NotImplementedError):
        select_candidate_blocks_cooperative(logits, row_ends, 64, out)


def test_select_candidate_blocks_cooperative_no_torch_compute():
    """Static audit: the production file only uses allowed torch calls."""
    import importlib
    import re
    import sys
    from pathlib import Path

    # The re-exported function shadows the submodule as a package attribute, so
    # fetch the module object from sys.modules rather than by attribute access.
    name = "flaggems_vllm.ops.select_candidate_blocks_cooperative"
    importlib.import_module(name)
    text = Path(sys.modules[name].__file__).read_text()
    calls = set(re.findall(r"torch\.([A-Za-z_][A-Za-z0-9_\.]*)\(", text))
    allowed = {"empty"}
    assert calls <= allowed, f"unexpected torch calls: {calls - allowed}"
    # `.to(` is not listed: it only appears as Triton tensor casts in kernels.
    for bad in (".contiguous(", ".clone(", ".sum(", ".max(", ".topk(", ".fill_("):
        assert bad not in text, bad
