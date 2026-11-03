# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Call-site timing only: actual callback estimates and error propagation."""

import importlib
import importlib.util
import sys
from types import SimpleNamespace

import pytest
import triton

libentry = importlib.import_module("flaggems_vllm.utils.libentry")


class FakeDriver:

    def __init__(self, durations):
        self.durations = iter(durations)
        self.calls = []
        self.scratch_probe = lambda: pytest.fail("independent scratch probe used")

    def get_device_interface(self):
        return self

    def get_empty_cache_for_benchmark(self):
        self.calls.append("cache")
        return object()

    def clear_cache(self, cache):
        self.calls.append("flush")

    def synchronize(self):
        self.calls.append("sync")

    def Event(self, *, enable_timing):
        assert enable_timing
        owner = self

        class Event:

            def record(self):
                owner.calls.append("record")

            def elapsed_time(self, other):
                owner.calls.append("elapsed")
                value = next(owner.durations)
                if isinstance(value, Exception):
                    raise value
                return value

        return Event()


def install_driver(monkeypatch, durations):
    driver = FakeDriver(durations)
    monkeypatch.setattr(libentry.triton.runtime.driver, "_active", driver)
    monkeypatch.setattr(
        libentry,
        "torch_device_fn",
        SimpleNamespace(synchronize=driver.synchronize),
    )
    monkeypatch.setattr(
        libentry.runtime, "device", SimpleNamespace(vendor_name="kunlunxin")
    )
    return driver


@pytest.mark.parametrize("vendor", ["nvidia", "metax", "ascend", "mthreads"])
def test_other_vendor_uses_unchanged_triton_benchmark(monkeypatch, vendor):
    monkeypatch.setattr(libentry.runtime, "device", SimpleNamespace(vendor_name=vendor))
    result = [0.2, 0.1, 0.3]
    calls = []

    def do_bench(fn, **kwargs):
        calls.append((fn, kwargs))
        return result

    monkeypatch.setattr(libentry.triton.testing, "do_bench", do_bench)
    kernel = lambda: None
    quantiles = (0.5, 0.2, 0.8)
    assert (
        libentry._do_bench_with_timer_fallback(
            kernel, warmup=11, rep=22, quantiles=quantiles
        )
        is result
    )
    assert calls == [(kernel, {"warmup": 11, "rep": 22, "quantiles": quantiles})]


def test_event_benchmark_uses_actual_estimate_once(monkeypatch):
    driver = install_driver(monkeypatch, [10.0] + [0.3] * 5 + [1.25])
    monkeypatch.setattr(
        libentry.triton.testing,
        "do_bench",
        lambda *args, **kwargs: pytest.fail("would repeat Triton's estimate"),
    )
    calls = []
    result = libentry._do_bench_with_timer_fallback(
        lambda: calls.append("kernel"), warmup=0, rep=0, quantiles=(0.5, 0.2, 0.8)
    )
    assert result == pytest.approx([1.25, 1.25, 1.25])
    assert calls == ["kernel"] * 8
    assert driver.calls.count("flush") == 6


def test_event_quantiles_use_completed_real_samples(monkeypatch):
    driver = install_driver(monkeypatch, [10.0] + [0.3] * 5 + [3.0, 1.0])
    calls = []
    result = libentry._do_bench_with_timer_fallback(
        lambda: calls.append("kernel"), warmup=0, rep=5, quantiles=(0.5, 0.2, 0.8)
    )
    assert result == pytest.approx([2.0, 1.4, 2.6])
    assert calls == ["kernel"] * 9
    assert driver.calls.count("flush") == 7


def test_wall_path_honors_warmup_and_measurement_duration(monkeypatch):
    install_driver(monkeypatch, [0.0] + [0.3] * 5)
    now = [0]
    calls = []
    monkeypatch.setattr(libentry.time, "perf_counter", lambda: now[0] / 1000)

    def kernel():
        calls.append("kernel")
        now[0] += 1

    result = libentry._do_bench_with_timer_fallback(
        kernel, warmup=1.5, rep=2.5, quantiles=(0.5, 0.2, 0.8)
    )
    assert result == pytest.approx([1.0, 1.0, 1.0])
    assert calls == ["kernel"] * 11


def test_triton_32_cache_flush_api():
    calls = []
    cache = SimpleNamespace(zero_=lambda: calls.append("flush"))
    libentry._clear_benchmark_cache(SimpleNamespace(), cache)
    assert calls == ["flush"]


@pytest.mark.parametrize(
    "durations",
    [[0.0] + [0.3] * 5, [10.0, 0.3, 0.3, 0.0, 0.3, 0.3], [float("nan")] + [0.3] * 5],
)
def test_real_zero_selects_wall_without_scratch_probe(monkeypatch, durations):
    driver = install_driver(monkeypatch, durations)
    scratch_calls = []
    driver.scratch_probe = lambda: scratch_calls.append("scratch") or 3.0
    monkeypatch.setattr(
        libentry.triton.testing,
        "do_bench",
        lambda *args, **kwargs: pytest.fail("event benchmark called twice"),
    )
    calls = []
    result = libentry._do_bench_with_timer_fallback(
        lambda: calls.append("kernel"), warmup=0, rep=0, quantiles=(0.5, 0.2, 0.8)
    )
    assert len(result) == 3
    assert all((sample >= 0 for sample in result))
    assert calls == ["kernel"] * 7
    assert driver.calls.count("flush") == 5
    assert scratch_calls == []


def test_zero_timed_event_is_not_returned_as_zero_latency(monkeypatch):
    install_driver(monkeypatch, [10.0] + [0.3] * 5 + [0.0])
    with pytest.raises(RuntimeError, match="unusable sample"):
        libentry._do_bench_with_timer_fallback(
            lambda: None, warmup=0, rep=0, quantiles=(0.5, 0.2, 0.8)
        )


@pytest.mark.parametrize("failure_call", [1, 3, 7, 8])
def test_kernel_exceptions_propagate_without_retry(monkeypatch, failure_call):
    install_driver(monkeypatch, [10.0] + [0.3] * 5 + [1.25])
    calls = []
    error = ZeroDivisionError("kernel failure")

    def kernel():
        calls.append("kernel")
        if len(calls) == failure_call:
            raise error

    with pytest.raises(ZeroDivisionError) as caught:
        libentry._do_bench_with_timer_fallback(
            kernel, warmup=0, rep=0, quantiles=(0.5,)
        )
    assert caught.value is error
    assert len(calls) == failure_call


def test_kernel_failure_on_wall_path_propagates_without_retry(monkeypatch):
    install_driver(monkeypatch, [0.0] + [0.3] * 5)
    error = RuntimeError("wall kernel failed")
    calls = []

    def kernel():
        calls.append("kernel")
        if len(calls) == 7:
            raise error

    with pytest.raises(RuntimeError) as caught:
        libentry._do_bench_with_timer_fallback(
            kernel, warmup=0, rep=0, quantiles=(0.5,)
        )
    assert caught.value is error
    assert len(calls) == 7


def test_probe_event_failure_propagates_before_mode_is_cached(monkeypatch):
    error = RuntimeError("device event failed")
    install_driver(monkeypatch, [error])
    calls = []
    with pytest.raises(RuntimeError) as caught:
        libentry._do_bench_with_timer_fallback(
            lambda: calls.append("kernel"), warmup=0, rep=0, quantiles=(0.5,)
        )
    assert caught.value is error
    assert calls == ["kernel"] * 6


@pytest.mark.parametrize("fail", [False, True])
def test_triton_autotuner_pre_post_hooks_restore_mutated_state(monkeypatch, fail):
    install_driver(monkeypatch, [10.0] + [0.3] * 5 + [1.25])
    state = {"value": 10, "saved": None}
    calls = []
    error = RuntimeError("kernel failure")

    def pre_hook(args, reset_only=False):
        state["saved"] = state["value"]
        calls.append("pre")

    def run(*args, **kwargs):
        calls.append("kernel")
        state["value"] += 1
        if fail and calls.count("kernel") == 3:
            raise error

    def post_hook(args, exception):
        calls.append("post")
        state["value"] = state["saved"]
        if exception is not None:
            assert exception is error

    tuner = SimpleNamespace(
        nargs={},
        pre_hook=pre_hook,
        post_hook=post_hook,
        fn=SimpleNamespace(run=run),
        do_bench=lambda kernel, quantiles: libentry._do_bench_with_timer_fallback(
            kernel, warmup=0, rep=0, quantiles=quantiles
        ),
    )
    if fail:
        with pytest.raises(RuntimeError) as caught:
            triton.runtime.Autotuner._bench(tuner, config=triton.Config({}))
        assert caught.value is error
        expected = 3
    else:
        assert triton.runtime.Autotuner._bench(
            tuner, config=triton.Config({})
        ) == pytest.approx([1.25, 1.25, 1.25])
        expected = 8
    assert state["value"] == 10
    assert calls == ["pre", "kernel", "post"] * expected


@pytest.mark.parametrize(
    "quantiles, expected",
    [((0, 0.25, 0.5, 0.75, 1), [1, 1.5, 2, 2.5, 3]), ((0.5,), 2), ((), []), (None, 2)],
)
def test_quantile_contract(quantiles, expected):
    assert libentry._summarize_benchmark([3, 1, 2], quantiles) == pytest.approx(
        expected
    )


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
