# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import sys
from importlib import import_module, util
from types import ModuleType, SimpleNamespace

import pytest
import torch

from flaggems_vllm.utils.tle_capabilities import (
    supports_sparse_mla_tle,
    supports_topk_tle,
)


def _language():
    def api(*args, **kwargs):
        raise AssertionError("capability check executed an optional API")

    return SimpleNamespace(
        cumsum=api,
        pipe=api,
        gpu=SimpleNamespace(
            alloc=api, local_ptr=api, copy=api, warp_specialize=api, smem=object()
        ),
    )


@pytest.mark.parametrize(
    "predicate,missing",
    [
        (supports_topk_tle, "cumsum"),
        (supports_topk_tle, "gpu"),
        (supports_topk_tle, "gpu.alloc"),
        (supports_topk_tle, "gpu.local_ptr"),
        (supports_topk_tle, "gpu.smem"),
        (supports_sparse_mla_tle, "pipe"),
        (supports_sparse_mla_tle, "gpu"),
        (supports_sparse_mla_tle, "gpu.alloc"),
        (supports_sparse_mla_tle, "gpu.local_ptr"),
        (supports_sparse_mla_tle, "gpu.copy"),
        (supports_sparse_mla_tle, "gpu.warp_specialize"),
        (supports_sparse_mla_tle, "gpu.smem"),
    ],
)
def test_missing_used_api_rejects_tle_without_mutation(predicate, missing):
    language = _language()
    assert predicate(language)
    owner, _, attr = missing.rpartition(".")
    target = language.gpu if owner else language
    delattr(target, attr)
    before = vars(target).copy()
    assert not predicate(language)
    assert vars(target) == before


@pytest.mark.parametrize("predicate", [supports_topk_tle, supports_sparse_mla_tle])
def test_optional_language_absence(predicate):
    assert not predicate(None)


@pytest.mark.parametrize("attr", ["cumsum", "pipe"])
def test_non_callable_api_rejects_only_its_consumer(attr):
    language = _language()
    setattr(language, attr, object())
    assert supports_topk_tle(language) is (attr != "cumsum")
    assert supports_sparse_mla_tle(language) is (attr != "pipe")


@pytest.fixture(params=["prefill", "decode"])
def load_topk_module(request, monkeypatch):
    """Import fresh JIT functions without changing the already loaded operators."""
    original = import_module(f"flaggems_vllm.ops.top_k_per_row_{request.param}")
    version_utils = import_module("flaggems_vllm.utils.triton_version_utils")
    experimental = import_module("triton.experimental")
    monkeypatch.setattr(version_utils, "has_triton_tle", lambda *args: True)
    monkeypatch.setenv("FLAGGEMS_FORCE_TLE", "1")

    def load(language):
        package = ModuleType("triton.experimental.tle")
        package.language = language
        spec = util.spec_from_file_location(
            f"{original.__name__}_test", original.__file__
        )
        module = util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        # Keep the fake capabilities local to this operator import. The Triton
        # compiler itself imports the real TLE language when lowering tl.range.
        with monkeypatch.context() as import_patch:
            import_patch.setattr(experimental, "tle", package, raising=False)
            import_patch.setitem(sys.modules, package.__name__, package)
            import_patch.setitem(sys.modules, f"{package.__name__}.language", language)
            spec.loader.exec_module(module)
        return module

    return request.param, load


@pytest.mark.parametrize(
    "missing,non_callable",
    [
        (name, False)
        for name in ("cumsum", "gpu", "gpu.alloc", "gpu.local_ptr", "gpu.smem")
    ]
    + [(name, True) for name in ("cumsum", "gpu.alloc", "gpu.local_ptr")],
)
def test_incomplete_tle_does_not_break_non_tle_cache_key(
    load_topk_module, missing, non_callable
):
    phase, load = load_topk_module
    language = _language()
    owner, _, attr = missing.rpartition(".")
    target = language.gpu if owner else language
    if non_callable:
        setattr(target, attr, None)
    else:
        delattr(target, attr)
    before = vars(target).copy()
    module = load(language)
    assert module.HAS_TLE is False
    # Exercise Triton's real dependency walk, including disabled constexpr branches.
    assert getattr(module, f"non_tle_top_k_per_row_{phase}").cache_key
    assert module.tle is None
    assert vars(target) == before


def test_complete_tle_remains_enabled(load_topk_module):
    _, load = load_topk_module
    language = _language()
    module = load(language)
    assert module.HAS_TLE is True
    assert module.tle is language


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_incomplete_tle_topk_matches_reference(load_topk_module):
    phase, load = load_topk_module
    language = _language()
    del language.gpu.alloc
    module = load(language)
    rows, vocab, top_k = 2, 4096, 32
    logits = torch.randn(rows, vocab, device="cuda")
    indices = torch.full((rows, top_k), -1, device="cuda", dtype=torch.int32)
    if phase == "prefill":
        starts, ends = [0, 17], [vocab, 4001]
        module.top_k_per_row_prefill(
            logits,
            torch.tensor(starts, device="cuda", dtype=torch.int32),
            torch.tensor(ends, device="cuda", dtype=torch.int32),
            indices,
            rows,
            *logits.stride(),
            top_k,
        )
    else:
        starts, ends = [0, 0], [4000, 4001]
        module.top_k_per_row_decode(
            logits,
            2,
            torch.tensor([4001], device="cuda", dtype=torch.int32),
            indices,
            rows,
            *logits.stride(),
            top_k,
        )
    for row, (start, end) in enumerate(zip(starts, ends)):
        selected = indices[row].long()
        assert torch.all((selected >= 0) & (selected < end - start))
        assert selected.unique().numel() == top_k
        actual = logits[row, start + selected].sort(descending=True).values
        expected = logits[row, start:end].topk(top_k).values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
