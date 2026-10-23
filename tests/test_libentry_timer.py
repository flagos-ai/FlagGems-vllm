# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Timer failure recovery must measure completed work and propagate errors."""

import importlib
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
        raise ZeroDivisionError("warmup / estimate_ms")

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
