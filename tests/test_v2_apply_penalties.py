# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Shared V2 penalty contract for Ascend and Kunlunxin backends."""

import pytest
import torch


def _penalty_fixture(vocab_size, backend):
    rows = 10 if backend == "kunlunxin" else 9
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
    if backend == "kunlunxin":
        logits[-1, 1] = -0.0

    mapping = torch.tensor([2, 2, 2, 2, 0, 0, 0, 1, 3, 3][:rows], dtype=torch.int32)
    local_pos = torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 0, 1][:rows], dtype=torch.int32)
    token_ids = torch.tensor(
        [31, 31, 7, 31, 32, 31, vocab_size - 1, 33, 1024, 7][:rows], dtype=torch.int32
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
        if (
            case["repetition"][request] == 1
            and case["frequency"][request] == 0
            and case["presence"][request] == 0
        ):
            continue
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


def _sampling_device(backend):
    if backend == "ascend":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("Ascend NPU is unavailable")
    import flaggems_vllm

    if flaggems_vllm.vendor_name != backend:
        pytest.skip(f"{backend} backend required")
    device_type = "npu" if backend == "ascend" else "cuda"
    device_module = torch.npu if backend == "ascend" else torch.cuda
    return flaggems_vllm, torch.device(device_type, device_module.current_device())


@pytest.mark.v2_apply_penalties
@pytest.mark.parametrize(
    "backend,vocab_size,dtype",
    [
        ("ascend", 1031, torch.bfloat16),
        ("ascend", 2057, torch.float16),
        ("ascend", 151939, torch.float32),
        ("kunlunxin", 1031, torch.bfloat16),
        ("kunlunxin", 2057, torch.float16),
        ("kunlunxin", 151939, torch.float32),
        ("kunlunxin", 248320, torch.float16),
    ],
)
def test_v2_apply_penalties(backend, vocab_size, dtype):
    library, device = _sampling_device(backend)
    case = _penalty_fixture(vocab_size, backend)
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
    if backend == "kunlunxin":
        assert torch.signbit(logits[-1, 1].cpu())
    for actual, reference in (
        (logits_storage[:, vocab_size:], case["logits_storage"][:, vocab_size:]),
        (packed_storage, case["packed_storage"]),
        (counts_storage, case["counts_storage"]),
    ):
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)


@pytest.mark.v2_apply_penalties
@pytest.mark.parametrize("backend", ["ascend", "kunlunxin"])
def test_empty_penalties_preserve_logits_shape(backend):
    library, device = _sampling_device(backend)
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


@pytest.mark.v2_apply_penalties
@pytest.mark.parametrize("backend", ["ascend", "kunlunxin"])
def test_penalty_cpu_reference_covers_backend_fixtures(backend):
    case = _penalty_fixture(1031, backend)
    expected = _penalty_cpu_reference(case)
    _verify_penalty_oracle(case, expected)
    if backend == "kunlunxin":
        assert torch.signbit(expected[-1, 1])


@pytest.mark.v2_apply_penalties
@pytest.mark.parametrize("rows", [0, 1])
def test_v2_apply_penalties_is_explicitly_unsupported_on_nvidia(rows):
    import flaggems_vllm

    if flaggems_vllm.vendor_name != "nvidia":
        pytest.skip("NVIDIA generic contract")
    device = flaggems_vllm.device
    index = torch.zeros(rows, device=device, dtype=torch.int32)
    coeff = torch.zeros(rows, device=device)
    words = torch.zeros(rows, 33, device=device, dtype=torch.int32)
    counts = torch.zeros(rows, 1031, device=device, dtype=torch.int32)
    with pytest.raises(NotImplementedError, match="backend implementation"):
        flaggems_vllm.v2_apply_penalties(
            torch.empty(rows, 1031, device=device),
            index,
            index,
            index,
            torch.ones_like(coeff),
            coeff,
            coeff,
            words,
            counts,
        )
