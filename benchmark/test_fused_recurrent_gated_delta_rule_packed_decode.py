# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""No-Graph event timing, with state reset outside every timed region."""
import json
import os
import statistics
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.testing.packed_decode import (
    assert_result,
    make_inputs,
    native_vllm024,
    reference,
)

OP = "fused_recurrent_gated_delta_rule_packed_decode"
SHAPES = [(b, 8, 16, 128, 128) for b in (1, 4, 16, 64, 256)] + [
    (9, 4, 12, 64, 32),
    (3, 2, 4, 17, 23),
    (1, 16, 64, 128, 128),
]


@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
@pytest.mark.parametrize("entry", ["generic", "top"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", SHAPES)
def test_packed_decode_performance(entry, dtype, shape):
    data = make_inputs(
        shape, dtype, torch.float32, flaggems_vllm.device, strided=True, invalid=False
    )
    state0 = data["initial_state"].clone()
    expected = reference(data, True)
    candidate = getattr(flaggems_vllm.ops if entry == "generic" else flaggems_vllm, OP)
    functions = {"gems": candidate}
    native_reason = "No validated equivalent same-device native packed interface"
    if flaggems_vllm.vendor_name == "nvidia":
        native = native_vllm024()
        functions["native"] = native
        native_reason = None
    for fn in functions.values():
        data["initial_state"].copy_(state0)
        fn(**data, use_qk_l2norm_in_kernel=True)
        assert_result(data, expected, dtype)
    # Compilation/warmup and every reset are explicitly outside timing events.
    for fn in functions.values():
        for _ in range(5):
            data["initial_state"].copy_(state0)
            fn(**data, use_qk_l2norm_in_kernel=True)
    torch_device_fn.synchronize()
    samples = {name: [] for name in functions}
    for i in range(25):
        names = list(functions) if i % 2 == 0 else list(reversed(functions))
        for name in names:
            data["initial_state"].copy_(state0)
            torch_device_fn.synchronize()
            start = torch_device_fn.Event(enable_timing=True)
            end = torch_device_fn.Event(enable_timing=True)
            start.record()
            functions[name](**data, use_qk_l2norm_in_kernel=True)
            end.record()
            torch_device_fn.synchronize()
            samples[name].append(start.elapsed_time(end))
    assert_result(data, expected, dtype)
    config = {
        k: (
            {"shape": list(v.shape), "stride": list(v.stride()), "dtype": str(v.dtype)}
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in data.items()
    }
    row = dict(
        op=OP,
        entry=entry,
        backend=flaggems_vllm.vendor_name,
        dtype=str(dtype),
        shape=shape,
        config=config,
        normalize=True,
        accuracy="PASS",
        performance="PASS",
        graph=False,
        unit="ms",
        protocol="device-event-reset-outside-alternating-median-v1",
        warmup=5,
        repetitions=25,
        samples=samples,
        median_ms={k: statistics.median(v) for k, v in samples.items()},
        native_reason=native_reason,
        candidate_module=candidate.__module__,
    )
    destination = os.environ.get("PACKED_DECODE_RESULTS")
    if not destination:
        raise RuntimeError("Set PACKED_DECODE_RESULTS to a fresh JSONL evidence path")
    with Path(destination).open("a") as f:
        f.write(json.dumps(row) + "\n")
