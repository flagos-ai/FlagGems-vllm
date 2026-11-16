# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Compare apply_logit_bias against the installed vLLM implementation.

Run with ``pytest -v -s benchmark/test_logit_bias.py``.

The installed vLLM baseline does not expose all-masked-row restoration, so
both implementations are measured with that option disabled. Device events
bracket the Python call and therefore include host-side dispatch gaps; the
reported speedup describes complete-call latency, not isolated kernel time.

Every parametrized case is collected and printed once as a single aligned
table when the module finishes. The device, the event timing and the
baseline module are resolved from the active backend, so this file carries
no hard-coded CUDA references.
"""

import importlib
import os
import statistics
import time

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.logit_bias import apply_logit_bias
from flaggems_vllm.runtime import torch_device_fn

from .conftest import Config
from .consts import BenchMode

device = flaggems_vllm.device
vendor_name = flaggems_vllm.vendor_name

_EVENT_CLS = getattr(torch_device_fn, "Event", None)
_TIMING = "device-event" if _EVENT_CLS is not None else "host-clock"
_REPETITIONS = 12

_probe = getattr(torch_device_fn, "is_available", None)
if _probe is None:
    _probe = getattr(torch_device_fn, "device_count", None)
_HAS_DEVICE = bool(_probe()) if _probe is not None else True


def _synchronize():
    sync = getattr(torch_device_fn, "synchronize", None)
    if sync is not None:
        sync()


def _import_upstream_baseline():
    """Resolve the vLLM baseline module for the active backend.

    Vendor vLLM images normally keep the upstream package layout. A backend
    image that ships the op elsewhere can point at it through the
    ``FLAGGEMS_VLLM_LOGIT_BIAS_BASELINE`` environment variable, without
    changing this file.
    """
    candidates = ["vllm.v1.worker.gpu.sample.logit_bias"]
    override = os.environ.get("FLAGGEMS_VLLM_LOGIT_BIAS_BASELINE")
    if override:
        candidates.insert(0, override)
    for name in candidates:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, "apply_logit_bias"):
            return module
    pytest.skip(
        "upstream vLLM logit_bias baseline unavailable on backend "
        f"{vendor_name}/{device} (tried: {', '.join(candidates)})",
        allow_module_level=True,
    )


_upstream = _import_upstream_baseline()

pytestmark = [
    pytest.mark.apply_logit_bias,
    pytest.mark.skipif(
        not _HAS_DEVICE, reason=f"requires an available {device} device"
    ),
]

# Table layout: (header, width, alignment). Latency columns carry the unit in
# the header so the data rows stay purely numeric and easy to compare.
_COLUMNS = (
    ("Status", 8, "<"),
    ("dtype", 9, "<"),
    ("batch", 6, ">"),
    ("vocab", 7, ">"),
    ("allowed", 8, ">"),
    ("bias", 6, ">"),
    ("vLLM (ms)", 12, ">"),
    ("FlagGems (ms)", 15, ">"),
    ("Speedup", 10, ">"),
    ("shape", 16, "<"),
)
_RESULTS = []


def _format_row(cells):
    return " ".join(
        format(value, f"{align}{width}")
        for value, (_, width, align) in zip(cells, _COLUMNS)
    ).rstrip()


def _record(status, dtype, batch, vocab, allowed_width, bias_width,
            reference_ms=None, implementation_ms=None):
    speedup = (
        reference_ms / implementation_ms
        if reference_ms is not None and implementation_ms
        else None
    )

    def fmt(value, spec):
        return format(value, spec) if value is not None else "N/A"

    cells = [
        status,
        str(dtype).replace("torch.", ""),
        batch,
        vocab,
        allowed_width,
        bias_width,
        fmt(reference_ms, ".4f"),
        fmt(implementation_ms, ".4f"),
        f"{speedup:.3f}x" if speedup is not None else "N/A",
        f"({batch}, {vocab})",
    ]
    _RESULTS.append({"cells": cells, "speedup": speedup})


def _format_table(rows):
    header = _format_row([column[0] for column in _COLUMNS])
    rule = "-" * len(header)
    body = "\n".join(_format_row(row["cells"]) for row in rows)
    speedups = [row["speedup"] for row in rows if row["speedup"] is not None]
    if speedups:
        summary = (
            f"{len(rows)} cases | speedup min={min(speedups):.3f}x "
            f"median={statistics.median(speedups):.3f}x "
            f"max={max(speedups):.3f}x"
        )
    else:
        summary = f"{len(rows)} cases | no successful measurements"
    mode = getattr(Config.mode, "value", Config.mode)
    title = (
        "Operator: apply_logit_bias  Performance Test "
        f"(backend={vendor_name}, device={device}, mode={mode}, "
        f"timing={_TIMING}, reps={_REPETITIONS}, metric=median)"
    )
    baseline = (
        f"Baseline: {_upstream.__name__}.apply_logit_bias "
        "(all-masked-row restoration disabled)"
    )
    return "\n".join(
        ["", title, baseline, header, rule, body, rule, summary, ""]
    )


@pytest.fixture(scope="module", autouse=True)
def _report_table():
    _RESULTS.clear()
    yield
    if _RESULTS:
        print(_format_table(_RESULTS))


def _time_call(fn, original):
    working = original.clone()
    fn(working)
    _synchronize()
    timings = []
    for _ in range(_REPETITIONS):
        working.copy_(original)
        if _EVENT_CLS is not None:
            start = _EVENT_CLS(enable_timing=True)
            end = _EVENT_CLS(enable_timing=True)
            start.record()
            fn(working)
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
        else:
            # Backends without a device event API fall back to a host clock
            # bracketing the same synchronized call.
            start = time.perf_counter()
            fn(working)
            _synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0
        timings.append(elapsed)
    return statistics.median(timings)


@pytest.mark.parametrize(
    "batch,vocab,allowed_width,bias_width",
    [
        (1, 32000, 0, 8),
        (16, 32000, 16, 8),
        (64, 128256, 0, 32),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_apply_logit_bias_latency(batch, vocab, allowed_width, bias_width, dtype):
    if Config.mode != BenchMode.KERNEL:
        raise NotImplementedError(
            "in-place device-event benchmark supports --mode kernel"
        )
    original = torch.randn((batch, vocab), device=device, dtype=dtype)
    mapping = torch.arange(batch, device=device, dtype=torch.int32)
    pos = torch.zeros(batch, device=device, dtype=torch.int32)
    allowed_count = torch.full(
        (batch,), allowed_width, device=device, dtype=torch.int32
    )
    allowed = torch.arange(
        max(1, allowed_width), device=device, dtype=torch.int32
    ).repeat(batch, 1)
    bias_count = torch.full(
        (batch,), bias_width, device=device, dtype=torch.int32
    )
    bias_ids = torch.arange(bias_width, device=device, dtype=torch.int32)
    bias_ids = (bias_ids + 64).repeat(batch, 1)
    bias = torch.full((batch, bias_width), 0.1, device=device)
    min_lens = torch.full((batch,), 10, device=device, dtype=torch.int32)
    stop_count = torch.ones(batch, device=device, dtype=torch.int32)
    stop_ids = torch.full(
        (batch, 1), vocab - 1, device=device, dtype=torch.int32
    )
    restore = torch.zeros(batch, device=device, dtype=torch.int32)
    shared = (
        mapping, pos, allowed_count, allowed, bias_count, bias_ids,
        bias, min_lens, stop_count,
    )
    try:
        reference_ms = _time_call(
            lambda logits: _upstream.apply_logit_bias(
                logits, *shared, stop_ids
            ),
            original,
        )
        implementation_ms = _time_call(
            lambda logits: apply_logit_bias(
                logits, *shared, restore, stop_ids, False
            ),
            original,
        )
    except Exception:
        _record(
            "FAILED", dtype, batch, vocab, allowed_width, bias_width
        )
        raise
    _record(
        "PASSED", dtype, batch, vocab, allowed_width, bias_width,
        reference_ms, implementation_ms,
    )
