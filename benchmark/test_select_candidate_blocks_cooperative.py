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

"""Benchmark select_candidate_blocks_cooperative against vLLM's version."""

import functools

import pytest
import torch

import flaggems_vllm

from . import base

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

try:
    from vllm.models.deepseek_v41.nvidia.specialized_decode.attention import (
        candidate_topk as _vllm_candidate_topk,
    )
except Exception:
    _vllm_candidate_topk = None


# (rows, seq_len, width, block_size, k). width is the CUDA-graph-static padded
# row (max_model_len-wide logits); seq_len is the live decode length.
# DeepSeek-V4.1 decode: candidate_block_size=8, candidate_topk_blocks=2048,
# 1M context -> 131072 blocks; at most 64 rows per rank.
_W = 131072 * 8
_SHAPES = [
    (1, 4096, _W, 8, 2048),
    (1, 131072, _W, 8, 2048),
    (1, _W, _W, 8, 2048),
    (4, 32768, _W, 8, 2048),
    (4, _W, _W, 8, 2048),
    (14, 262144, _W, 8, 2048),
    (16, 65536, _W, 8, 2048),
    (32, 131072, _W, 8, 2048),
    (64, 16384, _W, 8, 2048),
    (64, 524288, _W, 8, 2048),
    (16, 131072, _W, 8, 512),
    (16, 131072, _W, 8, 1024),
]


def _input_fn(shape, dtype, device):
    rows, seq_len, width, block_size, k = shape
    logits = torch.randn(rows, width, dtype=dtype, device=device)
    row_ends = torch.full((rows,), seq_len, dtype=torch.int32, device=device)
    out = torch.empty((rows, k), dtype=torch.int32, device=device)
    yield logits, row_ends, block_size, out


@functools.lru_cache(maxsize=1)
def _vllm_workspace_ready():
    """vLLM fetches its top-k scratch from the global workspace manager."""
    try:
        from vllm.v1.worker import workspace as ws

        ws.current_workspace_manager()
        return True
    except Exception:
        return False


def _vllm_op(logits, row_ends, block_size, out):
    _vllm_candidate_topk.select_candidate_blocks_cooperative(
        logits, row_ends, block_size, out
    )
    return out


_GEMS_WS = {}


def _gems_op(logits, row_ends, block_size, out):
    ws = _GEMS_WS.get(logits.device)
    if ws is None:
        ws = torch.empty(
            RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=logits.device
        )
        _GEMS_WS[logits.device] = ws
    flaggems_vllm.select_candidate_blocks_cooperative(
        logits, row_ends, block_size, out, workspace=ws
    )
    return out


def _torch_baseline(logits, row_ends, block_size, out):
    """Vectorized torch reference for the full candidate-block selection, used
    as the baseline when vLLM's native op is unavailable (benchmark-only; torch
    compute is allowed here, never in the production op). Mirrors vLLM's
    select_candidate_blocks: block max, newest block pinned, top-k, -inf -> -1.
    """
    rows, width = logits.shape
    k = out.shape[1]
    nblocks = (width + block_size - 1) // block_size
    padded = nblocks * block_size
    cols = torch.arange(padded, device=logits.device)
    valid = cols[None, :] < row_ends[:, None]
    if padded != width:
        lg = torch.full(
            (rows, padded), float("-inf"), dtype=logits.dtype, device=logits.device
        )
        lg[:, :width] = logits
    else:
        lg = logits
    lg = torch.where(valid, lg, torch.full_like(lg, float("-inf")))
    scores = lg.view(rows, nblocks, block_size).amax(dim=2)
    newest = (row_ends - 1).clamp(min=0) // block_size
    ar = torch.arange(nblocks, device=logits.device)
    pin = (ar[None, :] == newest[:, None]) & (row_ends[:, None] > 0)
    scores = torch.where(pin, torch.full_like(scores, float("inf")), scores)
    take = min(k, nblocks)
    vals, idx = scores.topk(take, dim=-1)
    idx = torch.where(
        vals != float("-inf"),
        idx.to(torch.int32),
        torch.full_like(idx, -1, dtype=torch.int32),
    )
    out[:, :take] = idx
    if take < k:
        out[:, take:] = -1
    return out


def _get_baseline_op():
    if _vllm_candidate_topk is not None and _vllm_workspace_ready():
        return _vllm_op
    # No native op on this build: compare against a vectorized torch reference.
    return _torch_baseline


class SelectCandidateBlocksBenchmark(base.GenericBenchmarkExcluse1D):
    DEFAULT_SHAPE_DESC = "rows, seq_len, width, block_size, k"

    def init_user_config(self):
        super().init_user_config()
        if self.torch_op is None:
            self.to_bench_metrics = [
                m for m in self.to_bench_metrics if m not in ("latency_base", "speedup")
            ]

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = list(_SHAPES)


@pytest.mark.select_candidate_blocks_cooperative
def test_select_candidate_blocks_cooperative():
    bench = SelectCandidateBlocksBenchmark(
        input_fn=_input_fn,
        op_name="select_candidate_blocks_cooperative",
        torch_op=_get_baseline_op(),
        gems_op=_gems_op,
        dtypes=[torch.float32],
    )
    bench.run()
