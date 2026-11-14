# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Measure the complete BF16 TP2 PPU window with CUDA Graph replay."""

import json
import os
import statistics
import sys

import pytest
import torch

import flaggems_vllm
from tests.thead.megapreattention_utils import run_tp2, summarize_round

from ..conftest import Config

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "thead" or torch.cuda.device_count() < 2,
    reason="requires two T-Head PPUs",
)


@pytest.mark.megapreattention
@pytest.mark.parametrize("tokens", [4096, 8192, 16384, 32768])
def test_megapreattention(tokens):
    warmup = (
        Config.warm_up if any(a.split("=")[0] == "--warmup" for a in sys.argv) else 20
    )
    iterations = (
        Config.repetition if any(a.split("=")[0] == "--iter" for a in sys.argv) else 100
    )
    if warmup < 1 or iterations < 1:
        raise ValueError(
            "MegaPreAttention benchmark needs positive warmup and sample counts"
        )
    fixture_path = os.getenv("MEGAPREATTENTION_FIXTURE")
    ranks_by_round = [
        run_tp2(tokens, fixture_path=fixture_path, warmup=warmup, iterations=iterations)
        for _ in range(5)
    ]
    names = ("baseline", "candidate")
    rounds = [
        {**{name: summarize_round(ranks, name) for name in names}, "ranks": ranks}
        for ranks in ranks_by_round
    ]
    latency = {
        name: statistics.mean(r[name]["rank_max_mean_ms"] for r in rounds)
        for name in names
    }
    result = {
        "global_tokens": tokens,
        "tp": 2,
        "dtype": "bfloat16",
        "model_dims": [2048, 16, 2],
        "fixture": fixture_path or "seeded synthetic",
        "mode": "cudagraph",
        "warmup": warmup,
        "samples_per_rank_per_round": iterations,
        "baseline": "SymmMem + separate FlagGems Gemma RMSNorm, MRoPE and KV store",
        "rank_max_mean_ms": latency,
        "speedup": latency["baseline"] / latency["candidate"],
        "rounds": rounds,
    }
    print("MEGAPREATTENTION_RESULT " + json.dumps(result))
