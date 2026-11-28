# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for route-specific MThreads attention tests."""

import pytest


@pytest.fixture(autouse=True)
def mthreads_attention_heuristic_seed(request, monkeypatch):
    """Route-specific tests use heuristics; full and tuner tests search normally."""
    stem = request.node.path.stem
    if stem.startswith("test_flash_attn_varlen_mthreads") and stem not in (
        "test_flash_attn_varlen_mthreads_autotune",
        "test_flash_attn_varlen_mthreads_full",
    ):
        monkeypatch.setenv("MTHREADS_FA_AUTOTUNE", "0")
