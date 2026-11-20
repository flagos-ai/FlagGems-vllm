# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Ascend V2 temperature scaling against independent CPU arithmetic."""

import pytest
import torch


def _ascend():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    import flaggems_vllm

    return flaggems_vllm, torch.device("npu", torch.npu.current_device())


@pytest.mark.v2_apply_temperature
@pytest.mark.parametrize(
    "vocab_size,dtype",
    [(1031, torch.bfloat16), (8193, torch.float16), (151936, torch.float32)],
)
@pytest.mark.parametrize("order", ([0, 1, 2, 3], [3, 0, 2, 1]))
def test_v2_apply_temperature(vocab_size, dtype, order):
    library, device = _ascend()
    generator = torch.Generator().manual_seed(631)
    temperatures = torch.tensor([0.7, 1.0, 0.0, 1.3])
    temperatures_device = temperatures.to(device)
    mapping = torch.tensor(order, dtype=torch.int64, device=device)
    original = (torch.randn(4, vocab_size, generator=generator) * 2).to(dtype)
    original[0, 0] = float("-inf")
    original[3, -1] = float("inf")
    storage = torch.full((4, vocab_size + 13), 125.0, dtype=dtype, device=device)
    values = storage[:, :vocab_size]
    values.copy_(original)
    expected = original.clone()
    for _ in range(3):
        for row, slot in enumerate(order):
            temperature = float(temperatures[slot])
            if temperature not in (0.0, 1.0):
                expected[row] = (expected[row].float() / temperature).to(dtype)
        library.v2_apply_temperature(values, mapping, temperatures_device)
        tolerance = 1e-6 if dtype == torch.float32 else torch.finfo(dtype).eps
        torch.testing.assert_close(
            values.cpu(), expected, rtol=tolerance, atol=tolerance
        )
        torch.testing.assert_close(
            storage[:, vocab_size:].cpu(),
            torch.full((4, 13), 125.0, dtype=dtype),
            rtol=0,
            atol=0,
        )
        for row, slot in enumerate(order):
            if float(temperatures[slot]) in (0.0, 1.0):
                torch.testing.assert_close(
                    values[row].cpu(), original[row], rtol=0, atol=0
                )


@pytest.mark.v2_apply_temperature
def test_temperature_handles_empty_and_rejects_noncontiguous_columns():
    library, device = _ascend()
    index = torch.zeros(4, dtype=torch.int32, device=device)
    temperatures = torch.ones(1, device=device)
    assert (
        library.v2_apply_temperature(
            torch.empty((4, 0), device=device), index, temperatures
        )
        is None
    )
    values = torch.ones((4, 12), device=device)[:, ::2]
    with pytest.raises(ValueError, match="contiguous columns"):
        library.v2_apply_temperature(values, index, temperatures)
