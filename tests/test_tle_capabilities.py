# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import sys
from importlib import import_module, util
from types import ModuleType, SimpleNamespace

import pytest
import torch

from flaggems_vllm.utils.tle_capabilities import (
    _has_module_wide_named_barrier_support,
    supports_megagate_tle,
    supports_module_wide_named_barriers,
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
            alloc=api,
            local_ptr=api,
            copy=api,
            warp_specialize=api,
            alloc_barrier=api,
            barrier_wait=api,
            wgmma=api,
            wgmma_wait=api,
            smem=object(),
        ),
    )


def _real_tle_language():
    """Use real builtins when a test intentionally enables TLE.

    FlagTree's dependency walker rejects ordinary Python functions referenced
    by a JIT function.  ``_language`` is therefore suitable for capability
    probes and disabled-TLE imports only; an enabled import must bind the real
    TLE builtins that the compiler can hash.
    """

    return pytest.importorskip("triton.experimental.tle.language")


def _dependency_cache_key(kernel):
    """Trigger the real JIT dependency walk through wrapper APIs.

    NVIDIA operators are wrapped in ``LibEntry``, whose stable public handle
    to the underlying JIT function is ``jit_function``.  Older wrappers used
    to forward ``cache_key`` directly, while current FlagTree does not; keep
    this test independent of that incidental forwarding behavior.
    """

    jit_function = getattr(kernel, "jit_function", kernel)
    cache_key = getattr(jit_function, "cache_key", None)
    assert cache_key
    return cache_key


def _named_barrier_components():
    def api(*args, **kwargs):
        raise AssertionError("feature probe executed a compiler API")

    core = SimpleNamespace(
        _FIRST_VIRTUAL_NAMED_BARRIER_ID=16,
        _reserve_named_barrier_ids=api,
    )
    libtriton = SimpleNamespace(
        tle=SimpleNamespace(
            passes=SimpleNamespace(add_allocate_named_barriers=api),
        )
    )
    return core, libtriton


def test_module_wide_named_barrier_probe_requires_both_compiler_halves():
    core, libtriton = _named_barrier_components()
    assert _has_module_wide_named_barrier_support(core, libtriton)

    core._FIRST_VIRTUAL_NAMED_BARRIER_ID = 15
    assert not _has_module_wide_named_barrier_support(core, libtriton)
    core._FIRST_VIRTUAL_NAMED_BARRIER_ID = 16

    core._reserve_named_barrier_ids = object()
    assert not _has_module_wide_named_barrier_support(core, libtriton)
    core, libtriton = _named_barrier_components()

    libtriton.tle.passes.add_allocate_named_barriers = object()
    assert not _has_module_wide_named_barrier_support(core, libtriton)


def test_module_wide_named_barrier_public_probe_is_cached():
    supports_module_wide_named_barriers.cache_clear()
    before = supports_module_wide_named_barriers.cache_info()
    first = supports_module_wide_named_barriers()
    middle = supports_module_wide_named_barriers.cache_info()
    second = supports_module_wide_named_barriers()
    after = supports_module_wide_named_barriers.cache_info()

    assert first is second
    assert middle.misses == before.misses + 1
    assert after.hits == middle.hits + 1


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
        (supports_megagate_tle, "pipe"),
        (supports_megagate_tle, "gpu"),
        (supports_megagate_tle, "gpu.alloc"),
        (supports_megagate_tle, "gpu.local_ptr"),
        (supports_megagate_tle, "gpu.copy"),
        (supports_megagate_tle, "gpu.warp_specialize"),
        (supports_megagate_tle, "gpu.alloc_barrier"),
        (supports_megagate_tle, "gpu.barrier_wait"),
        (supports_megagate_tle, "gpu.wgmma"),
        (supports_megagate_tle, "gpu.wgmma_wait"),
        (supports_megagate_tle, "gpu.smem"),
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


@pytest.mark.parametrize(
    "predicate", [supports_topk_tle, supports_sparse_mla_tle, supports_megagate_tle]
)
def test_optional_language_absence(predicate):
    assert not predicate(None)


@pytest.mark.parametrize("attr", ["cumsum", "pipe"])
def test_non_callable_api_rejects_only_its_consumer(attr):
    language = _language()
    setattr(language, attr, object())
    assert supports_topk_tle(language) is (attr != "cumsum")
    assert supports_sparse_mla_tle(language) is (attr != "pipe")
    assert supports_megagate_tle(language) is (attr != "pipe")


@pytest.mark.parametrize(
    "attr",
    [
        "alloc",
        "local_ptr",
        "copy",
        "warp_specialize",
        "alloc_barrier",
        "barrier_wait",
        "wgmma",
        "wgmma_wait",
    ],
)
def test_non_callable_megagate_gpu_api_rejects_tle(attr):
    language = _language()
    setattr(language.gpu, attr, object())
    assert not supports_megagate_tle(language)


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
    _dependency_cache_key(getattr(module, f"non_tle_top_k_per_row_{phase}"))
    assert module.tle is None
    assert vars(target) == before


def test_complete_tle_remains_enabled(load_topk_module):
    _, load = load_topk_module
    language = _real_tle_language()
    module = load(language)
    assert module.HAS_TLE is True
    assert module.tle is language


def test_attn_res_missing_tle_load_uses_non_tle_cache_key(monkeypatch):
    original = import_module("flaggems_vllm.ops.attn_res")
    version_utils = import_module("flaggems_vllm.utils.triton_version_utils")
    experimental = import_module("triton.experimental")
    language = _language()
    package = ModuleType("triton.experimental.tle")
    package.language = language
    spec = util.spec_from_file_location(f"{original.__name__}_test", original.__file__)
    module = util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.setattr(version_utils, "_triton_version_at_least", lambda *args: True)

    with monkeypatch.context() as import_patch:
        import_patch.setattr(experimental, "tle", package, raising=False)
        import_patch.setitem(sys.modules, package.__name__, package)
        import_patch.setitem(sys.modules, f"{package.__name__}.language", language)
        spec.loader.exec_module(module)

    assert module._HAS_TLE_LOAD is False
    assert module.tle is None
    _dependency_cache_key(module._attn_res_post_kernel)


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
