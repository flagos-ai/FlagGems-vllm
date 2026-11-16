# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Preallocated trailing-slot padding benchmarks, excluding input preparation."""

from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

from benchmark import base
from flaggems_vllm import device as DEVICE
from flaggems_vllm import pad_trailing_draft_slots, runtime
from flaggems_vllm.ops import speculator as impl

pytestmark = [
    pytest.mark.pad_trailing_draft_slots,
    pytest.mark.skipif(
        DEVICE == "cpu" or not runtime.torch_device_fn.is_available(),
        reason=f"requires an available {DEVICE} accelerator",
    ),
]


# Unchanged kernel from vLLM 947dd62c, multi_module_mtp/speculator.py.
@triton.jit
def _pad_trailing_draft_slots_kernel(
    slot_mappings_ptr,
    slot_mappings_stride0,
    query_start_loc_ptr,
    last_token_indices_ptr,
    PAD_ID,
    BLOCK_SIZE: tl.constexpr,
):
    group_idx = tl.program_id(0)
    req_idx = tl.program_id(1)
    # Slots computed from stale token positions in the range
    # [last_token_index + 1, query_end) can result in writes to blocks.
    # Pad these slot values so that attention kernels ignore them.
    start = tl.load(last_token_indices_ptr + req_idx) + 1
    end = tl.load(query_start_loc_ptr + req_idx + 1)
    base = slot_mappings_ptr + group_idx * slot_mappings_stride0
    for i in range(start, end, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        tl.store(base + offs, PAD_ID, mask=mask)


PRIMARY_SHAPES = [
    (g, n, t, "decode")
    for g in (1, 4, 8)
    for n in (1, 8, 32, 128, 512, 2048)
    for t in (0, 1, 3, 7, 15)
]
EXTRA_SHAPES = [
    (2, 31, 7, "ragged"),
    (3, 129, 15, "empty"),
    (4, 513, 31, "strided"),
    (1, 128, 7, "prefill"),
    (4, 512, 7, "prefill"),
    (4, 128, 63, "decode"),
    (4, 512, 255, "ragged"),
    (4, 512, 256, "decode"),
    (4, 512, 257, "ragged"),
    (4, 128, 1024, "decode"),
    (4, 512, 4096, "decode"),
    (1, 17, 7, "overshoot"),
]


def make_case(
    groups,
    n,
    tail,
    scenario="decode",
    *,
    dtype=torch.int64,
    query_dtype=torch.int32,
    last_dtype=torch.int64,
    device=DEVICE,
):
    prefix = 2048 if scenario == "prefill" else 5
    lengths = torch.full((n,), prefix + tail, dtype=torch.int64)
    tails = torch.full((n,), tail, dtype=torch.int64)
    if scenario == "ragged":
        tails = torch.arange(n, dtype=torch.int64) % (tail + 1)
    elif scenario == "empty":
        lengths[::3] = 0
        tails[::3] = 0
    query_cpu = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    last_cpu = query_cpu[1:] - tails - 1
    if scenario == "overshoot":
        last_cpu[::2] = query_cpu[1:][::2] + 2
    padded = triton.cdiv(int(query_cpu[-1]) + 19, 16) * 16
    stride = padded + (3 if scenario == "strided" else 16)
    offset = 3 if scenario == "strided" else 0
    original = (torch.arange(groups * stride + 23, dtype=torch.int64) % 100000).to(
        dtype
    )
    storage = original.to(device)
    slots = storage.as_strided((groups, padded), (stride, 1), offset)
    expected = original.clone()
    expected_view = expected.as_strided((groups, padded), (stride, 1), offset)
    for r in range(n):
        start, end = int(last_cpu[r]) + 1, int(query_cpu[r + 1])
        if start < end:
            expected_view[:, start:end] = -1
    query = query_cpu.to(query_dtype).to(device)
    last = last_cpu.to(last_dtype).to(device)
    return SimpleNamespace(
        storage=storage,
        original=original,
        expected=expected,
        args=(slots, query, last, n),
        shape=(groups, n, tail, scenario),
    )


def run_reference(case):
    slots, query, last, n = case.args
    if slots.shape[0] and n:
        _pad_trailing_draft_slots_kernel[(slots.shape[0], n)](
            slots,
            slots.stride(0),
            query,
            last,
            -1,
            BLOCK_SIZE=256,
        )


def run_candidate(case, config=None):
    if config is None:
        return pad_trailing_draft_slots(*case.args)
    slots, query, last, n = case.args
    if slots.shape[0] and n:
        impl._pad_trailing_draft_slots_kernel.jit_function[
            (slots.shape[0], triton.cdiv(n, config.kwargs["BLOCK_ROWS"]))
        ](
            slots,
            slots.stride(0),
            query,
            last,
            -1,
            NUM_REQS=n,
            **config.kwargs,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )


def check_case(case):
    torch.testing.assert_close(case.storage.cpu(), case.expected, rtol=0, atol=0)


class TrailingSlotsBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PRIMARY_SHAPES + EXTRA_SHAPES

    def record_shapes(self, *args, **kwargs):
        g, n, t, scenario = self._case.shape
        return {
            "groups": g,
            "requests": n,
            "tail_slots": t,
            "scenario": scenario,
            "dtype": str(self._case.args[0].dtype),
            "inputs": super().record_shapes(*self._case.args),
        }

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            self._case = make_case(*shape, dtype=dtype, device=self.device)
            run_reference(self._case)
            check_case(self._case)
            self._case.storage.copy_(self._case.original)
            run_candidate(self._case)
            check_case(self._case)
            yield ({"case": self._case},)


@pytest.mark.pad_trailing_draft_slots
def test_pad_trailing_draft_slots_perf():
    bench = TrailingSlotsBenchmark(
        op_name="pad_trailing_draft_slots",
        torch_op=run_reference,
        dtypes=[torch.int32, torch.int64],
    )
    bench.set_gems(run_candidate)
    bench.run()
