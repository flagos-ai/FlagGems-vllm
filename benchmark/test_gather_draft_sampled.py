# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Gather tokens and positions for speculative verification."""

import pytest
import torch

from benchmark import base
from flaggems_vllm import device as DEVICE
from flaggems_vllm import runtime

try:
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
        gather_draft_sampled as vllm_reference,
    )
except (ImportError, OSError):
    vllm_reference = None

pytestmark = pytest.mark.skipif(
    DEVICE == "cpu" or not runtime.torch_device_fn.is_available(),
    reason=f"requires an available {DEVICE} accelerator",
)
try:
    from vllm.v1.worker.gpu.buffer_utils import is_uva_available
except ImportError:
    _HAS_UVA = False
else:
    _HAS_UVA = runtime.device.vendor_name == "nvidia" and is_uva_available()
PRIMARY_STORAGE = "uva" if _HAS_UVA else "device"

# Requests, drafts per request (-1 means variable), token layout, request phase,
# state storage. UVA matches RequestState.prefill_len in the pinned vLLM caller.
SHAPES = (
    [
        (n, drafts, "dense", "decode", PRIMARY_STORAGE)
        for n in (1, 8, 32, 128, 512, 2048)
        for drafts in (0, 1, 3, 7, 15)
    ]
    + [
        (n, 0, "dense", "decode", PRIMARY_STORAGE)
        for n in (31, 33, 127, 129, 255, 257, 511, 513, 1023, 1025, 4097)
    ]
    + [
        (13, 4, "dense", "decode", PRIMARY_STORAGE),
        (129, 2, "dense", "decode", PRIMARY_STORAGE),
        (513, 6, "dense", "decode", PRIMARY_STORAGE),
    ]
    + [
        (n, 7, "dense", "decode", "device")
        for n in (1, 32, 128, 512, 2048)
        if PRIMARY_STORAGE == "uva"
    ]
    + [(n, 7, "gapped", "decode", PRIMARY_STORAGE) for n in (32, 128, 512, 2048)]
    + [(n, 7, "random", "decode", PRIMARY_STORAGE) for n in (32, 512)]
    + [
        (n, 7, "dense", phase, PRIMARY_STORAGE)
        for n in (32, 512)
        for phase in ("prefill", "mixed")
    ]
    + [(n, -1, "gapped", "mixed", PRIMARY_STORAGE) for n in (17, 129, 513)]
    + [(n, 7, "dense", "decode", PRIMARY_STORAGE, 4096) for n in (1, 32, 128, 512)]
)


def case_scenario(shape):
    n, drafts, layout, phase, storage, *state_capacity = shape
    if state_capacity:
        return "sparse_active_request_slots"
    if layout == "random":
        return "scattered_gather_stress"
    if drafts < 0:
        return "variable_draft_lengths"
    if phase != "decode":
        return f"{phase}_placeholders"
    if layout == "gapped":
        return "gapped_token_blocks"
    if storage == "device":
        return "device_state_table"
    if n in (31, 33, 127, 129, 255, 257, 511, 513, 1023, 1025, 4097) or drafts in (
        2,
        4,
        6,
    ):
        return "non_power_of_two_rows"
    return "uva_decode"


def make_inputs(shape, seed=20261107, device=None):
    device = DEVICE if device is None else device
    n, drafts, layout, phase, storage, *state_capacity = shape
    generator = torch.Generator().manual_seed(seed)
    counts = (
        (torch.arange(n, dtype=torch.int64) % 9 + 1)
        if drafts < 0
        else torch.full((n,), drafts + 1, dtype=torch.int64)
    )
    total = int(counts.sum())
    row = torch.repeat_interleave(torch.arange(n, dtype=torch.int64), counts)
    starts = counts.cumsum(0) - counts
    local = torch.arange(total, dtype=torch.int64) - starts[row]
    slots = torch.randperm(n, generator=generator).to(torch.int32) * 3 + 1
    if state_capacity:
        slots = torch.randperm(state_capacity[0], generator=generator)[:n].to(
            torch.int32
        )
    mapping = slots[row]
    gap = 64 if layout in ("gapped", "random") else 0
    token_capacity = total + gap * n + 7
    indices = torch.arange(total, dtype=torch.int64) + row * gap + 3
    if layout == "random":
        indices = torch.randperm(token_capacity, generator=generator)[:total]
    tokens = torch.randint(
        0, 32000, (token_capacity,), generator=generator, dtype=torch.int32
    )
    positions = torch.randint(
        0, 2048, (token_capacity,), generator=generator, dtype=torch.int64
    )
    prefill = torch.randint(
        32,
        2048,
        (state_capacity[0] if state_capacity else n * 3 + 5,),
        generator=generator,
        dtype=torch.int32,
    )
    boundary = prefill[mapping.long()].long()
    block_start = boundary + 11
    if phase == "prefill":
        block_start = boundary - 2
    elif phase == "mixed":
        offsets = torch.tensor([-2, 0, 11], dtype=torch.int64)
        block_start = boundary + offsets[row % 3]
    positions[indices] = block_start + local
    data = [tokens, positions, indices, mapping, local.to(torch.int32), prefill]
    data = [tensor.to(device) for tensor in data]
    if storage == "uva":
        if runtime.device.vendor_name != "nvidia":
            pytest.skip("mapped-host UVA state is NVIDIA-specific")
        buffer_utils = pytest.importorskip("vllm.v1.worker.gpu.buffer_utils")
        UvaBufferPool = buffer_utils.UvaBufferPool
        pool = UvaBufferPool(prefill.shape, torch.int32, max_concurrency=1)
        if not isinstance(pool._uva_bufs[0], buffer_utils.UvaBuffer):
            pytest.skip("mapped-host UVA is unavailable on this platform")
        data[-1] = pool.copy_to_uva(prefill)
        data[-1]._gather_state_owner = pool
    return tuple(data)


class GatherDraftSampledBenchmark(base.Benchmark):
    DEFAULT_SHAPES = SHAPES
    DEFAULT_SHAPE_DESC = "requests, drafts_per_request, token_layout, phase, state_storage, optional_state_capacity"

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def record_shapes(self, *args, **kwargs):
        return {
            "case": self._case,
            "scenario": case_scenario(self._case),
            "inputs": super().record_shapes(*args, **kwargs),
        }

    def set_metrics(self, user_desired_metrics):
        super().set_metrics(user_desired_metrics)
        if self.torch_op is None:
            self.to_bench_metrics = [
                m
                for m in self.to_bench_metrics
                if m not in ("latency_base", "speedup", "gbps_base")
            ]

    def get_input_iter(self, dtype):
        from flaggems_vllm import gather_draft_sampled

        for shape in self.shapes:
            self._case = shape
            data = make_inputs(shape, device=self.device)
            if vllm_reference is not None:
                expected = vllm_reference(*data)
            else:
                tokens, positions, indices, requests, local, prefill = (
                    t.cpu() for t in data
                )
                pos = positions[indices.long()]
                gathered = tokens[indices.long()].clone()
                gathered[(local > 0) & (pos - local < prefill[requests.long()])] = -1
                expected = (gathered.to(self.device), pos.to(self.device))
            actual = gather_draft_sampled(*data)
            for left, right in zip(actual, expected):
                assert torch.equal(left, right)
            yield data


@pytest.mark.gather_draft_sampled
def test_gather_draft_sampled_perf():
    from flaggems_vllm import gather_draft_sampled

    bench = GatherDraftSampledBenchmark(
        op_name="gather_draft_sampled", torch_op=vllm_reference, dtypes=[torch.int32]
    )
    bench.set_gems(gather_draft_sampled)
    bench.run()
