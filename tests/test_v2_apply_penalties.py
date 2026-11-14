# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Ascend sampling penalty semantics against an independent CPU reference."""

import pytest
import torch


def _penalty_fixture(vocab_size):
    rows = 9
    generator = torch.Generator().manual_seed(873)
    logits_storage = torch.full((rows, vocab_size + 7), 1234.0)
    logits = logits_storage[:, :vocab_size]
    logits.copy_(torch.randn(rows, vocab_size, generator=generator) * 2)
    logits[:, 31] = 3
    logits[:, 32] = 2
    logits[:, 1023] = 2
    logits[:, 1024] = 2
    logits[0, 7] = -2
    logits[2, 7] = -2
    logits[0, -1] = -4
    logits[-1, 0] = float("-inf")

    mapping = torch.tensor([2, 2, 2, 2, 0, 0, 0, 1, 3], dtype=torch.int32)
    local_pos = torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 0], dtype=torch.int32)
    token_ids = torch.tensor(
        [31, 31, 7, 31, 32, 31, vocab_size - 1, 33, 1024], dtype=torch.int32
    )
    repetition = torch.tensor([1.2, 0.7, 1.7, 1.0], dtype=torch.float32)
    frequency = torch.tensor([0.4, -0.25, 0.5, 0.0], dtype=torch.float32)
    presence = torch.tensor([-0.3, 0.6, 0.2, 0.0], dtype=torch.float32)
    prompts = (
        {31, 32, 1024, vocab_size - 1},
        {0, 31, 33, vocab_size - 2},
        {7, 31, 1023, 1024, vocab_size - 1},
        {31, 1024},
    )
    words = (vocab_size + 31) // 32
    packed_storage = torch.full((4, words + 3), 123, dtype=torch.int32)
    packed = packed_storage[:, :words]
    packed.zero_()
    for req, tokens in enumerate(prompts):
        bits = {}
        for token in tokens:
            word, bit = divmod(token, 32)
            bits[word] = bits.get(word, 0) | (1 << bit)
        for word, value in bits.items():
            packed[req, word] = value if value < (1 << 31) else value - (1 << 32)

    counts_storage = torch.full((4, vocab_size + 11), 71, dtype=torch.int32)
    counts = counts_storage[:, :vocab_size]
    counts.zero_()
    for req, tokens in enumerate(
        (
            {31: 2, 5: 1, vocab_size - 1: 3},
            {33: 4, vocab_size - 2: 1},
            {7: 2, 31: 3, vocab_size - 1: 1},
            {31: 5},
        )
    ):
        for token, count in tokens.items():
            counts[req, token] = count
    return {
        "logits_storage": logits_storage,
        "logits": logits,
        "mapping": mapping,
        "local_pos": local_pos,
        "tokens": token_ids,
        "repetition": repetition,
        "frequency": frequency,
        "presence": presence,
        "packed_storage": packed_storage,
        "packed": packed,
        "prompts": prompts,
        "counts_storage": counts_storage,
        "counts": counts,
    }


def _penalty_cpu_reference(case):
    result = case["logits"].clone()
    for row, request in enumerate(case["mapping"].tolist()):
        counts = case["counts"][request].clone()
        position = int(case["local_pos"][row])
        start = row - position
        for previous in range(1, position + 1):
            counts[int(case["tokens"][start + previous])] += 1
        output_seen = counts > 0
        seen = output_seen.clone()
        for token in case["prompts"][request]:
            seen[token] = True
        scale = torch.where(seen, case["repetition"][request], 1.0)
        result[row] *= torch.where(result[row] > 0, scale.reciprocal(), scale)
        result[row] -= case["frequency"][request] * counts
        result[row] -= case["presence"][request] * output_seen
    return result


def _verify_penalty_oracle(case, expected):
    # These values exercise independent prompt-only and repeated-draft cases.
    rep = float(case["repetition"][2])
    freq = float(case["frequency"][2])
    pres = float(case["presence"][2])
    checks = (
        (0, 31, 3 / rep - freq * 3 - pres),
        (1, 31, 3 / rep - freq * 4 - pres),
        (3, 31, 3 / rep - freq * 5 - pres),
        (0, 7, -2 * rep - freq * 2 - pres),
        (2, 7, -2 * rep - freq * 3 - pres),
        (0, 1023, 2 / rep),
        (0, 1024, 2 / rep),
        (0, 32, 2),
    )
    for row, token, value in checks:
        assert abs(float(expected[row, token]) - value) < 1e-6
    torch.testing.assert_close(expected[-1], case["logits"][-1], rtol=0, atol=0)
    assert int(case["packed"][2, 0]) < 0, "Packed bit 31 was not exercised"


def _ascend():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    import flaggems_vllm

    return flaggems_vllm, torch.device("npu", torch.npu.current_device())


@pytest.mark.v2_apply_penalties
@pytest.mark.parametrize(
    "vocab_size,dtype",
    [(1031, torch.bfloat16), (2057, torch.float16), (151939, torch.float32)],
)
def test_v2_apply_penalties(vocab_size, dtype):
    library, device = _ascend()
    case = _penalty_fixture(vocab_size)
    case["logits_storage"] = case["logits_storage"].to(dtype)
    case["logits"] = case["logits_storage"][:, :vocab_size]
    expected = _penalty_cpu_reference({**case, "logits": case["logits"].float()})
    if dtype == torch.float32:
        _verify_penalty_oracle(case, expected)
    logits_storage = case["logits_storage"].to(device)
    logits = logits_storage[:, :vocab_size]
    packed_storage = case["packed_storage"].to(device)
    packed = packed_storage[:, : case["packed"].shape[1]]
    counts_storage = case["counts_storage"].to(device)
    counts = counts_storage[:, :vocab_size]

    # Upstream may supply padding after the rows that produce logits.
    tokens = torch.cat((case["tokens"], torch.zeros(4, dtype=torch.int32)))
    library.v2_apply_penalties(
        logits,
        case["mapping"].to(device),
        tokens.to(device),
        case["local_pos"].to(device),
        case["repetition"].to(device),
        case["frequency"].to(device),
        case["presence"].to(device),
        packed,
        counts,
    )
    tolerance = 1e-5 if dtype == torch.float32 else torch.finfo(dtype).eps
    torch.testing.assert_close(
        logits.cpu(), expected.to(dtype), rtol=tolerance, atol=tolerance
    )
    for actual, reference in (
        (logits_storage[:, vocab_size:], case["logits_storage"][:, vocab_size:]),
        (packed_storage, case["packed_storage"]),
        (counts_storage, case["counts_storage"]),
    ):
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)


@pytest.mark.v2_apply_penalties
def test_empty_penalties_preserve_logits_shape():
    library, device = _ascend()
    logits = torch.empty((0, 1031), device=device)
    index = torch.empty((0,), device=device, dtype=torch.int32)
    coeff = torch.empty((0,), device=device)
    matrix = torch.empty((0, 0), device=device, dtype=torch.int32)
    assert (
        library.v2_apply_penalties(
            logits,
            index,
            index,
            index,
            coeff,
            coeff,
            coeff,
            matrix,
            matrix,
        )
        is None
    )
    assert logits.shape == (0, 1031)
