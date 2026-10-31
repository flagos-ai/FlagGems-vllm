# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Timer failure recovery must measure completed work and propagate errors."""

import importlib
import importlib.util
import sys
from types import SimpleNamespace

import pytest

libentry = importlib.import_module("flaggems_vllm.utils.libentry")


def test_event_timer_result_is_preserved(monkeypatch):
    result = [0.2, 0.1, 0.3]
    monkeypatch.setattr(libentry.triton.testing, "do_bench", lambda *a, **kw: result)
    assert (
        libentry._do_bench_with_timer_fallback(
            lambda: None, warmup=1, rep=1, quantiles=(0.5, 0.2, 0.8)
        )
        is result
    )


@pytest.mark.parametrize("vendor", ["kunlunxin", "nvidia"])
def test_zero_event_timer_uses_synchronized_elapsed_work(monkeypatch, vendor):
    def broken_timer(*args, **kwargs):
        estimate_ms = 0.0
        return 1 / estimate_ms

    monkeypatch.setattr(libentry.triton.testing, "do_bench", broken_timer)
    monkeypatch.setattr(libentry.runtime, "device", SimpleNamespace(vendor_name=vendor))
    now = [0.0]
    calls = []

    def kernel():
        calls.append("kernel")
        now[0] += 0.001

    def synchronize():
        calls.append("sync")
        now[0] += 0.001

    monkeypatch.setattr(libentry.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(
        libentry, "torch_device_fn", SimpleNamespace(synchronize=synchronize)
    )
    if vendor == "nvidia":
        with pytest.raises(ZeroDivisionError):
            libentry._do_bench_with_timer_fallback(
                kernel, warmup=2, rep=4, quantiles=(0.5, 0.2, 0.8)
            )
        assert not calls
    else:
        result = libentry._do_bench_with_timer_fallback(
            kernel, warmup=2, rep=4, quantiles=(0.5, 0.2, 0.8)
        )
        assert result == pytest.approx([2.0, 2.0, 2.0])
        assert calls == ["sync", "kernel", "sync", "kernel", "sync", "kernel", "sync"]


def test_kernel_failure_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(
        libentry.runtime, "device", SimpleNamespace(vendor_name="kunlunxin")
    )

    def broken_kernel(*args, **kwargs):
        raise RuntimeError("compile failure")

    monkeypatch.setattr(libentry.triton.testing, "do_bench", broken_kernel)
    with pytest.raises(RuntimeError, match="compile failure"):
        libentry._do_bench_with_timer_fallback(
            lambda: None, warmup=1, rep=1, quantiles=(0.5, 0.2, 0.8)
        )


def test_kernel_zero_division_is_not_treated_as_zero_event_time(monkeypatch):
    monkeypatch.setattr(
        libentry.runtime, "device", SimpleNamespace(vendor_name="kunlunxin")
    )
    error = ZeroDivisionError("kernel host failure")
    calls = []

    def kernel():
        calls.append("kernel")
        raise error

    def timer(kernel_call, **kwargs):
        estimate_ms = 0.0
        kernel_call()
        return estimate_ms

    monkeypatch.setattr(libentry.triton.testing, "do_bench", timer)
    monkeypatch.setattr(
        libentry,
        "torch_device_fn",
        SimpleNamespace(synchronize=lambda: calls.append("sync")),
    )
    with pytest.raises(ZeroDivisionError) as caught:
        libentry._do_bench_with_timer_fallback(
            kernel, warmup=1, rep=1, quantiles=(0.5, 0.2, 0.8)
        )
    assert caught.value is error
    assert calls == ["kernel"]


@pytest.mark.parametrize(
    "module_name,kernel_names,bench_name",
    [
        (
            "mhc_post",
            ("mhc_post_kernel_hc_mult_4", "mhc_post_kernel_generic"),
            "_mhc_post_do_bench",
        ),
        (
            "mhc_pre",
            ("mhc_pre_fused_kernel_hc_mult_4", "mhc_pre_generic_kernel"),
            "_mhc_pre_do_bench",
        ),
    ],
)
def test_mhc_autotuners_share_timer_recovery(
    monkeypatch, module_name, kernel_names, bench_name
):
    module = importlib.import_module(f"flaggems_vllm.ops.mhc.{module_name}")
    bench = getattr(module, bench_name)
    for kernel_name in kernel_names:
        kernel = getattr(module, kernel_name)
        selected = getattr(kernel, "_do_bench", None)
        if module.runtime.device.vendor_name == "kunlunxin":
            if selected is None:
                selected = getattr(kernel, "do_bench", None)
            assert selected is bench
        else:
            assert selected is not bench
    result = [0.2, 0.1, 0.3]
    calls = []

    def timer(kernel, **kwargs):
        calls.append((kernel, kwargs))
        return result

    monkeypatch.setattr(module, "_do_bench_with_timer_fallback", timer)
    kernel = lambda: None
    assert bench(kernel, (0.5, 0.2, 0.8)) is result
    assert calls == [(kernel, {"warmup": 25, "rep": 100, "quantiles": (0.5, 0.2, 0.8)})]


@pytest.mark.parametrize("module_name", ["mhc_pre", "mhc_post"])
@pytest.mark.parametrize("vendor", ["metax", "nvidia", "mthreads", "ascend", "thead"])
def test_mhc_source_import_with_legacy_autotune(monkeypatch, module_name, vendor):
    module = importlib.import_module(f"flaggems_vllm.ops.mhc.{module_name}")
    native_autotune = module.triton.autotune
    calls = []

    # Real MetaX CI rejects do_bench, including None, during module import.
    # This strict constructor boundary delegates accepted arguments to the
    # installed Triton; it does not impersonate a particular SDK version.
    def legacy_autotune(configs, key, prune_configs_by=None):
        calls.append((configs, key))
        return native_autotune(
            configs=configs, key=key, prune_configs_by=prune_configs_by
        )

    monkeypatch.setattr(module.runtime, "device", SimpleNamespace(vendor_name=vendor))
    monkeypatch.setattr(module.triton, "autotune", legacy_autotune)
    name = f"flaggems_vllm.ops.mhc._legacy_import_{module_name}"
    spec = importlib.util.spec_from_file_location(name, module.__file__)
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, fresh)
    spec.loader.exec_module(fresh)
    assert len(calls) == 2
    kernel_names = (
        ("mhc_pre_fused_kernel_hc_mult_4", "mhc_pre_generic_kernel")
        if module_name == "mhc_pre"
        else ("mhc_post_kernel_hc_mult_4", "mhc_post_kernel_generic")
    )
    for kernel_name in kernel_names:
        kernel = getattr(fresh, kernel_name)
        assert isinstance(kernel, type(getattr(module, kernel_name)))
        assert getattr(kernel, "_do_bench", None) is None
