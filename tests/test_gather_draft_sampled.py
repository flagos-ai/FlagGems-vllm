# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Exact gather and prefill-placeholder checks."""

import importlib

import pytest
import torch

from flaggems_vllm import device as DEVICE
from flaggems_vllm import gather_draft_sampled, runtime

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


def reference(inputs):
    tokens, positions, indices, requests, local, prefill = (t.cpu() for t in inputs)
    indices = indices.long()
    gathered = tokens[indices].clone()
    pos = positions[indices].clone()
    placeholder = (local > 0) & (pos - local < prefill[requests.long()])
    gathered[placeholder] = -1
    return gathered, pos


def make_inputs(rows, dtype="caller", seed=17):
    generator = torch.Generator().manual_seed(seed)
    capacity = rows * 2 + 17
    tokens = torch.randint(
        -100, 32000, (capacity,), generator=generator, dtype=torch.int32
    )
    positions = torch.zeros(capacity, dtype=torch.int64)
    indices = torch.randperm(capacity, generator=generator)[:rows].to(torch.int64)
    requests = torch.randint(0, 37, (rows,), generator=generator, dtype=torch.int32)
    local = (torch.arange(rows) % 7).to(torch.int32)
    prefill = torch.randint(64, 2048, (37,), generator=generator, dtype=torch.int32)
    offsets = torch.tensor([-1, 0, 1], dtype=torch.int64)
    positions[indices] = (
        prefill[requests.long()].long() + local + offsets[torch.arange(rows) % 3]
    )
    tensors = [tokens, positions, indices, requests, local, prefill]
    if dtype != "caller":
        tensors = [t.to(dtype) for t in tensors]
    return tuple(t.to(DEVICE) for t in tensors)


def check(inputs):
    expected = reference(inputs)
    before = [t.clone() for t in inputs]
    actual = gather_draft_sampled(*inputs)
    assert len(actual) == 2
    for out, ref in zip(actual, expected):
        assert torch.equal(out.cpu(), ref)
        assert out.is_contiguous()
    assert actual[0].dtype == inputs[0].dtype
    assert actual[1].dtype == inputs[1].dtype
    for current, original in zip(inputs, before):
        assert torch.equal(current, original)
    if vllm_reference is not None and runtime.device.vendor_name == "nvidia":
        original = vllm_reference(*inputs)
        assert all(torch.equal(x, y) for x, y in zip(actual, original))
    again = gather_draft_sampled(*inputs)
    assert all(torch.equal(x, y) for x, y in zip(actual, again))


@pytest.mark.gather_draft_sampled
@pytest.mark.parametrize(
    "rows",
    [
        0,
        1,
        2,
        7,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
        1023,
        1024,
        1025,
        4097,
        16384,
    ],
)
@pytest.mark.parametrize("dtype", ["caller", torch.int32, torch.int64])
def test_gather_and_boundaries(rows, dtype):
    check(make_inputs(rows, dtype))


@pytest.mark.gather_draft_sampled
@pytest.mark.parametrize("seed", [3, 107, 20261107])
def test_other_inputs(seed):
    check(make_inputs(387, seed=seed))


@pytest.mark.gather_draft_sampled
@pytest.mark.parametrize("index", range(6))
def test_misaligned_pointers(index):
    inputs = list(make_inputs(129))
    source = inputs[index]
    padded = torch.empty(source.numel() + 1, dtype=source.dtype, device=DEVICE)
    padded[1:] = source
    inputs[index] = padded[1:]
    check(tuple(inputs))


@pytest.mark.gather_draft_sampled
def test_large_integer_values():
    inputs = list(make_inputs(257, torch.int64))
    inputs[0][::2] = 2**40 + 17
    inputs[0][1::2] = -(2**40) - 19
    inputs[1].add_(2**40)
    check(tuple(inputs))


@pytest.mark.gather_draft_sampled
@pytest.mark.parametrize("index", range(6))
def test_individual_integer_types(index):
    inputs = list(make_inputs(65))
    dtype = torch.int64 if inputs[index].dtype == torch.int32 else torch.int32
    inputs[index] = inputs[index].to(dtype)
    check(tuple(inputs))


@pytest.mark.gather_draft_sampled
@pytest.mark.skipif(
    runtime.device.vendor_name != "nvidia", reason="NVIDIA tuning candidate guards"
)
@pytest.mark.parametrize(
    "config",
    [
        (dict(c.kwargs), c.num_warps, c.num_stages)
        for c in runtime.get_tuned_config("gather_draft_sampled")
    ],
)
def test_all_tiles_and_output_guards(config):
    module = importlib.import_module("flaggems_vllm.ops.gather_draft_sampled")
    kernel = module._gather_draft_sampled_kernel.jit_function
    meta, warps, stages = config
    block = meta["BLOCK_SIZE"]
    for rows in (33, 129, 1025):
        inputs = make_inputs(rows)
        expected = reference(inputs)
        out = torch.full((rows + 16,), -777, dtype=inputs[0].dtype, device=DEVICE)
        pos = torch.full((rows + 16,), -999, dtype=inputs[1].dtype, device=DEVICE)
        kernel[((rows + block - 1) // block,)](
            out,
            pos,
            *inputs,
            rows,
            USE_LAYOUT=module._HAS_LAYOUT,
            **meta,
            num_warps=warps,
            num_stages=stages,
        )
        assert torch.equal(out[:rows].cpu(), expected[0])
        assert torch.equal(pos[:rows].cpu(), expected[1])
        assert (out[rows:] == -777).all()
        assert (pos[rows:] == -999).all()


@pytest.mark.gather_draft_sampled
def test_repeated_token_and_request_indices():
    inputs = list(make_inputs(129))
    inputs[2].fill_(3)
    inputs[3].fill_(7)
    check(tuple(inputs))


@pytest.mark.gather_draft_sampled
def test_prefill_boundary_and_first_row():
    inputs = tuple(
        t.to(DEVICE)
        for t in (
            torch.tensor([11, 22, 33, 44, 55, 66], dtype=torch.int32),
            torch.tensor([30, 31, 32, 32, 33, 34], dtype=torch.int64),
            torch.arange(6, dtype=torch.int64),
            torch.zeros(6, dtype=torch.int32),
            torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32),
            torch.tensor([32], dtype=torch.int32),
        )
    )
    actual = gather_draft_sampled(*inputs)
    assert actual[0].tolist() == [11, -1, -1, 44, 55, 66]
    check(inputs)


@pytest.mark.gather_draft_sampled
def test_plain_triton_path(monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.gather_draft_sampled")
    monkeypatch.setattr(module, "_HAS_LAYOUT", False)
    check(make_inputs(1025))


@pytest.mark.gather_draft_sampled
def test_uva_request_state():
    if runtime.device.vendor_name != "nvidia":
        pytest.skip("mapped-host UVA state is NVIDIA-specific")
    buffer_utils = pytest.importorskip("vllm.v1.worker.gpu.buffer_utils")
    UvaBufferPool = buffer_utils.UvaBufferPool

    inputs = list(make_inputs(1025))
    pool = UvaBufferPool(inputs[-1].shape, inputs[-1].dtype, max_concurrency=1)
    if not isinstance(pool._uva_bufs[0], buffer_utils.UvaBuffer):
        pytest.skip("mapped-host UVA is unavailable on this platform")
    inputs[-1] = pool.copy_to_uva(inputs[-1].cpu())
    check(tuple(inputs))


@pytest.mark.gather_draft_sampled
def test_unsupported_cpu():
    inputs = tuple(t.cpu() for t in make_inputs(1))
    with pytest.raises(NotImplementedError, match="accelerator"):
        gather_draft_sampled(*inputs)


@pytest.mark.gather_draft_sampled
@pytest.mark.parametrize("index", range(6))
def test_unsupported_strides(index):
    inputs = list(make_inputs(33))
    source = inputs[index]
    padded = torch.empty(source.numel() * 2, dtype=source.dtype, device=DEVICE)
    padded[::2] = source
    inputs[index] = padded[::2]
    with pytest.raises(NotImplementedError, match="unit-stride"):
        gather_draft_sampled(*inputs)


@pytest.mark.gather_draft_sampled
def test_invalid_metadata():
    inputs = list(make_inputs(3))
    wrong = inputs.copy()
    wrong[0] = wrong[0].float()
    with pytest.raises(TypeError, match="int32 or int64"):
        gather_draft_sampled(*wrong)
    wrong = inputs.copy()
    wrong[3] = wrong[3][:-1]
    with pytest.raises(ValueError, match="one element"):
        gather_draft_sampled(*wrong)
    wrong = inputs.copy()
    wrong[0] = wrong[0].unsqueeze(0)
    with pytest.raises(ValueError, match="one-dimensional"):
        gather_draft_sampled(*wrong)
