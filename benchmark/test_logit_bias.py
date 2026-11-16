# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import statistics
import time

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.logit_bias import apply_logit_bias
from flaggems_vllm.runtime import torch_device_fn

from .conftest import Config
from .consts import BenchMode

try:
    from vllm.v1.worker.gpu.sample.logit_bias import (
        apply_logit_bias as vllm_apply_logit_bias,
    )
except ImportError:
    vllm_apply_logit_bias = None


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


pytestmark = [
    pytest.mark.apply_logit_bias,
    pytest.mark.skipif(
        not _HAS_DEVICE, reason=f"requires an available {device} device"
    ),
]

_COLUMNS = [
    ("Status", 8, "<"),
    ("dtype", 9, "<"),
    ("batch", 6, ">"),
    ("vocab", 7, ">"),
    ("allowed", 8, ">"),
    ("bias", 6, ">"),
    ("scenario", 11, "<"),
]
if vllm_apply_logit_bias is not None:
    _COLUMNS.append(("vLLM (ms)", 12, ">"))
_COLUMNS.append(("FlagGems (ms)", 15, ">"))
if vllm_apply_logit_bias is not None:
    _COLUMNS.append(("Speedup", 10, ">"))
_COLUMNS.append(("shape", 16, "<"))
_RESULTS = []


def _format_row(cells):
    return " ".join(
        format(value, f"{align}{width}")
        for value, (_, width, align) in zip(cells, _COLUMNS)
    ).rstrip()


def _record(
    status,
    dtype,
    batch,
    vocab,
    allowed_width,
    bias_width,
    scenario,
    reference_ms=None,
    implementation_ms=None,
):
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
        scenario,
    ]
    if vllm_apply_logit_bias is not None:
        cells.append(fmt(reference_ms, ".4f"))
    cells.append(fmt(implementation_ms, ".4f"))
    if vllm_apply_logit_bias is not None:
        cells.append(f"{speedup:.3f}x" if speedup is not None else "N/A")
    cells.append(f"({batch}, {vocab})")
    _RESULTS.append({"cells": cells, "speedup": speedup})


def _format_table(rows):
    header = _format_row([column[0] for column in _COLUMNS])
    rule = "-" * len(header)
    body = "\n".join(_format_row(row["cells"]) for row in rows)
    speedups = [row["speedup"] for row in rows if row["speedup"] is not None]
    if speedups:
        summary = (
            f"cases={len(rows)} | speedup min={min(speedups):.3f}x "
            f"median={statistics.median(speedups):.3f}x "
            f"max={max(speedups):.3f}x"
        )
    else:
        summary = f"cases={len(rows)}"
    mode = getattr(Config.mode, "value", Config.mode)
    title = (
        "Operator: apply_logit_bias  Performance Test "
        f"(backend={vendor_name}, device={device}, mode={mode}, "
        f"timing={_TIMING}, reps={_REPETITIONS}, metric=median)"
    )
    return "\n".join(["", title, header, rule, body, rule, summary, ""])


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
            start = time.perf_counter()
            fn(working)
            _synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0
        timings.append(elapsed)
    return statistics.median(timings)


@pytest.mark.apply_logit_bias
@pytest.mark.parametrize(
    "batch,vocab,allowed_width,bias_width,scenario",
    [
        (1, 32000, 0, 8, "bias_only"),
        (16, 32000, 16, 8, "allowlist"),
        (64, 128256, 0, 32, "bias_only"),
        (1, 32000, 1, 8, "restore"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_apply_logit_bias_latency(
    batch, vocab, allowed_width, bias_width, scenario, dtype
):
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
    if scenario == "restore":
        allowed.fill_(vocab - 1)
    bias_count = torch.full(
        (batch,), bias_width, device=device, dtype=torch.int32
    )
    bias_ids = torch.arange(bias_width, device=device, dtype=torch.int32)
    bias_ids = (bias_ids + 64).repeat(batch, 1)
    bias = torch.full((batch, bias_width), 0.1, device=device)
    min_lens = torch.full((batch,), 10, device=device, dtype=torch.int32)
    stop_count = torch.full(
        (batch,), int(scenario != "bias_only"), device=device, dtype=torch.int32
    )
    stop_ids = torch.full(
        (batch, 1), vocab - 1, device=device, dtype=torch.int32
    )
    restore = torch.full(
        (batch,), int(scenario == "restore"), device=device, dtype=torch.int32
    )
    shared = (
        mapping, pos, allowed_count, allowed, bias_count, bias_ids,
        bias, min_lens, stop_count,
    )
    reference_ms = None
    try:
        if vllm_apply_logit_bias is not None:
            reference_ms = _time_call(
                lambda logits: vllm_apply_logit_bias(
                    logits, *shared, restore, stop_ids, scenario == "restore"
                ),
                original,
            )
        implementation_ms = _time_call(
            lambda logits: apply_logit_bias(
                logits, *shared, restore, stop_ids, scenario == "restore"
            ),
            original,
        )
    except Exception:
        _record(
            "FAILED", dtype, batch, vocab, allowed_width, bias_width, scenario
        )
        raise
    _record(
        "PASSED", dtype, batch, vocab, allowed_width, bias_width, scenario,
        reference_ms, implementation_ms,
    )
