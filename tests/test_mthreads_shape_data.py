# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for original TP2 coverage and frozen input metadata.

Can run without GPU dependencies: pytest --noconftest -q <this file>.
"""

import gzip
import hashlib
import json
from collections import Counter

import pytest

from benchmark.mthreads_tp2_cases import DATA_FILE, SOURCE_SHA256
from benchmark.qwen36_flash_attn_cases import case_config, case_seed, load_cases


@pytest.mark.parametrize(
    "selection,count",
    [
        ("full", 1602),
        ("tp1-full", 275),
        ("tp2-full", 1040),
        ("tp4-full", 287),
        ("tp1-tp4-full", 562),
    ],
)
def test_selection_coverage(selection, count, monkeypatch):
    monkeypatch.delenv("MTHREADS_FA_CASE_IDS", raising=False)
    cases = load_cases(selection)
    assert len(cases) == len({case["case_id"] for case in cases}) == count


def test_tp2_preserves_original_workload(monkeypatch):
    monkeypatch.delenv("MTHREADS_FA_CASE_IDS", raising=False)
    raw = gzip.decompress(DATA_FILE.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256
    source = json.loads(raw)
    cases = load_cases("tp2-full")
    assert Counter(case["category"] for case in cases) == {
        "Decode": 1023,
        "Mixed": 15,
        "Prefill": 2,
    }
    for original, case in zip(source["cases"], cases):
        assert case["case_id"] == original["id"]
        assert case["cu_seqlens_q"] == original["cu_q"]
        assert case["seqused_k"] == original["klens"]
        assert case["q_shape"] == [original[f"q_dim{i}"] for i in range(3)]
        assert case["k_shape"] == [original[f"k_dim{i}"] for i in range(4)]
        assert case["v_shape"] == [original[f"v_dim{i}"] for i in range(4)]
        assert case["kv_storage_shape"] == source["kv_storage_shape"]
        assert case["kv_seed"] == source["seed"]
        assert case_seed(case) == source["seed"] + original["line"]
        assert case["block_table_shape"] == [len(original["qlens"]), 2112]
        assert case["max_seqlen_q"] == original["max_seqlen_q"]
        assert case["max_seqlen_k"] == original["max_seqlen_k"]
        assert case["num_splits"] == original["num_splits"] == 0
        assert case["observed_count"] == original["count"]
        assert case["category"] == original["source_category"].title()
        assert case["strict_category"] == original["category"].title()
        cu, used, hq, hk, dim, page, pages, alibi, softcap = case_config(case)
        assert (hq, hk, dim, page, pages, alibi, softcap) == (
            8,
            1,
            256,
            32,
            69333,
            False,
            None,
        )
        assert cu[-1] == case["q_shape"][0] and len(cu) == len(used) + 1
        assert sum((length + 31) // 32 for length in used) <= pages


def test_subset_source_order_and_unknown_ids(monkeypatch):
    monkeypatch.delenv("MTHREADS_FA_CASE_IDS", raising=False)
    cases = load_cases("tp2-full")
    ids = f"{cases[-1]['case_id']},{cases[0]['case_id']}"
    assert [case["case_id"] for case in load_cases("tp2-full", ids)] == [
        cases[0]["case_id"],
        cases[-1]["case_id"],
    ]
    monkeypatch.setenv("MTHREADS_FA_CASE_IDS", cases[3]["case_id"])
    assert load_cases("full") == [cases[3]]
    with pytest.raises(ValueError, match="Unknown or out-of-set"):
        load_cases("tp1-full", cases[0]["case_id"])
    with pytest.raises(ValueError, match="Unknown"):
        load_cases("typo")
