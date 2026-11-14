# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

"""Host-only tests of the production MetaX configuration selector.

Extract only host selector functions so these tests need neither a GPU nor the
vendor compiler. They do not validate kernel execution or backend registration.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def selectors(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    path = (
        Path(__file__).resolve().parents[1]
        / "src/flaggems_vllm/runtime/backend/_metax/ops/fused_moe.py"
    )
    names = {
        "_is_qwen_tp2_moe_shape",
        "_qwen_tp2_config",
        "_is_qwen_moe_shape",
        "_is_qwenvl_tp2_moe_shape",
        "_qwenvl_tp2_config",
        "get_default_config",
        "try_get_optimal_moe_config",
    }
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if getattr(node, "name", None) in names
    ]
    assert {node.name for node in nodes} == names
    namespace = {
        "Any": Any,
        "os": os,
        "_PLAIN_HALF_CONFIG_DTYPES": ("fp16", "bf16"),
        "_get_device_name": lambda: "MetaX_C550",
        "_generic_get_default_config": lambda *args: {"generic": True},
        "get_moe_configs": lambda *args: {},
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def select(namespace, m, stage, **kwargs):
    n, k = (512, 2048) if stage == "gemm1" else (2048, 256)
    return namespace["get_default_config"](
        m, 256, n, k, 8, kwargs.pop("dtype", "bf16"), gemm_stage=stage, **kwargs
    )


@pytest.mark.parametrize("stage", ["gemm1", "gemm2"])
@pytest.mark.parametrize("m", [64, 8192])
def test_retained_configurations(selectors, m, stage):
    expected = {
        "BLOCK_SIZE_M": 16 if m == 64 else 128,
        "BLOCK_SIZE_N": 64 if m == 64 else 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4 if m == 64 or stage == "gemm2" else 8,
        "num_stages": 1 if m == 8192 and stage == "gemm2" else 2,
    }
    if m == 64 and stage == "gemm2":
        expected["SWAP_AB"] = True
    if m == 8192:
        expected.update(SPLIT_K=1, PAIR_GATE_UP_DOT=False)
        if stage == "gemm2":
            expected.update(pipeline="basic", pipeline_load_num=-1)
    assert select(selectors, m, stage) == expected


@pytest.mark.parametrize("m", [0, 1, 8, 9, 63, 65, 384, 4096, 8191, 8193, 16384, 32768])
@pytest.mark.parametrize("stage", ["gemm1", "gemm2"])
@pytest.mark.parametrize("fast", [False, True])
def test_other_token_counts_keep_existing_profile(selectors, m, stage, fast):
    expected = selectors["_qwen_tp2_config"](m, stage, fast)
    assert select(selectors, m, stage, enable_gemm_fast_path=fast) == expected


@pytest.mark.parametrize("m", [64, 8192])
@pytest.mark.parametrize("stage", ["gemm1", "gemm2"])
@pytest.mark.parametrize(
    "exclusion", ["fp16", "blocked", "int8", "invariant", "device"]
)
def test_excluded_modes_keep_existing_profile(
    selectors, monkeypatch, m, stage, exclusion
):
    kwargs = {}
    if exclusion == "fp16":
        kwargs["dtype"] = "fp16"
    elif exclusion == "blocked":
        kwargs["block_shape"] = [128, 128]
    elif exclusion == "int8":
        kwargs["use_int8_w8a8"] = True
    elif exclusion == "invariant":
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    else:
        selectors["_get_device_name"] = lambda: "MetaX_other"
    assert select(selectors, m, stage, **kwargs) == selectors["_qwen_tp2_config"](
        m, stage
    )


@pytest.mark.parametrize("stage", ["gemm1", "gemm2"])
def test_new_large_m_fast_path_is_preserved(selectors, stage):
    assert select(selectors, 8192, stage, enable_gemm_fast_path=True) == selectors[
        "_qwen_tp2_config"
    ](8192, stage, True)


@pytest.mark.parametrize("stage", ["gemm1", "gemm2"])
def test_embedded_configuration_precedes_retained_profile(selectors, stage):
    embedded = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}
    selectors["get_moe_configs"] = lambda *args: {64: embedded}
    config, is_embedded = selectors["try_get_optimal_moe_config"](
        (256, 512, 2048),
        (256, 2048, 256),
        8,
        "bf16",
        64,
        256,
        gemm_stage=stage,
        return_is_embedded=True,
    )
    assert is_embedded and config == embedded
    config["BLOCK_SIZE_N"] = 1
    assert embedded["BLOCK_SIZE_N"] == 256


@pytest.mark.parametrize("m", [64, 8192])
def test_both_stages_share_alignment_block_m(selectors, m):
    assert (
        select(selectors, m, "gemm1")["BLOCK_SIZE_M"]
        == select(selectors, m, "gemm2")["BLOCK_SIZE_M"]
    )


@pytest.mark.parametrize(
    "e,n,k,topk", [(128, 512, 2048, 8), (256, 513, 2048, 8), (256, 512, 2048, 6)]
)
def test_unrelated_shapes_keep_generic_fallback(selectors, e, n, k, topk):
    assert selectors["get_default_config"](64, e, n, k, topk, "bf16") == {
        "generic": True
    }
