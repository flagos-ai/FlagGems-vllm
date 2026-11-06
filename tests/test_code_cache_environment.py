# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only cache selection checks; importing this test never initializes a GPU."""
import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "fgv,fg,expected",
    [
        ("/tmp/fgv", "/tmp/fg", "/tmp/fgv"),
        (None, "/tmp/fg", "/tmp/fg"),
        ("", "/tmp/fg", "/tmp/fg"),
        (None, None, str(Path.home() / ".flagdnn")),
    ],
)
def test_independent_cache_precedence(monkeypatch, fgv, fg, expected):
    for key, value in (("FLAGGEMS_VLLM_CACHE_DIR", fgv), ("FLAGGEMS_CACHE_DIR", fg)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    path = Path(__file__).parents[1] / "src/flaggems_vllm/utils/code_cache.py"
    spec = importlib.util.spec_from_file_location("fgv_cache_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.cache_dir_path() == Path(expected)
