# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Exact in-place padding checks, including untouched slot and storage guards."""

import pytest
import torch

from benchmark.test_speculator import (
    check_case,
    make_case,
    run_candidate,
    run_reference,
)
from flaggems_vllm import device as DEVICE
from flaggems_vllm import pad_trailing_draft_slots, runtime

pytestmark = [
    pytest.mark.pad_trailing_draft_slots,
    pytest.mark.skipif(
        DEVICE == "cpu" or not runtime.torch_device_fn.is_available(),
        reason=f"requires an available {DEVICE} accelerator",
    ),
]


@pytest.mark.pad_trailing_draft_slots
@pytest.mark.parametrize(
    "shape",
    [
        (0, 0, 0, "decode"),
        (0, 7, 3, "decode"),
        (2, 0, 0, "decode"),
        (1, 1, 0, "decode"),
        (1, 1, 1, "decode"),
        (3, 7, 3, "ragged"),
        (4, 31, 15, "empty"),
        (2, 129, 31, "strided"),
        (4, 513, 7, "prefill"),
        (1, 17, 7, "overshoot"),
        (4, 129, 255, "ragged"),
        (4, 129, 256, "decode"),
        (4, 129, 257, "decode"),
        (2, 31, 1024, "ragged"),
        (1, 129, 4096, "decode"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_padding(shape, dtype):
    case = make_case(*shape, dtype=dtype)
    run_reference(case)
    check_case(case)
    case.storage.copy_(case.original)
    assert run_candidate(case) is None
    check_case(case)
    run_candidate(case)
    check_case(case)


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("query_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("last_dtype", [torch.int32, torch.int64])
def test_integer_types(slot_dtype, query_dtype, last_dtype):
    case = make_case(
        3,
        37,
        17,
        "strided",
        dtype=slot_dtype,
        query_dtype=query_dtype,
        last_dtype=last_dtype,
    )
    case.storage.add_(2**40 if slot_dtype == torch.int64 else 2**20)
    run_reference(case)
    expected = case.storage.clone()
    case.storage.copy_(case.original)
    case.storage.add_(2**40 if slot_dtype == torch.int64 else 2**20)
    run_candidate(case)
    torch.testing.assert_close(case.storage, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    runtime.device.vendor_name != "nvidia", reason="NVIDIA tuning seeds"
)
def test_all_tuning_tiles():
    for config in runtime.get_tuned_config("pad_trailing_draft_slots"):
        for tail in [0, 7, 257, 1024]:
            case = make_case(3, config.kwargs["BLOCK_ROWS"] + 1, tail, "ragged")
            run_candidate(case, config=config)
            check_case(case)


def test_cpu_unsupported():
    case = make_case(1, 1, 1, device="cpu")
    with pytest.raises(NotImplementedError, match="requires an accelerator"):
        pad_trailing_draft_slots(*case.args)


def test_inner_stride_unsupported():
    case = make_case(2, 7, 3)
    slots, q, last, n = case.args
    with pytest.raises(NotImplementedError, match="unit inner strides"):
        pad_trailing_draft_slots(slots[:, ::2], q, last, n)
