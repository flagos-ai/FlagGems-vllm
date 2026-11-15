# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""CPU-only qualification oracle; never imported by production operators."""

import numpy as np

# Independent integer known answers frozen before the candidate implementation.
KAT_INPUTS = (
    (0, 0, 0, 0x8134CDF8, 0xD28AD962),
    (1, 2, 3, 0x23B53F91, 0xE3AAF818),
    (-1, 17, 1023, 0xC8C13C14, 0xD223C51C),
    (0x123456789ABCDEF, 123456789, 199999, 0x5DDE974F, 0x0547D2FE),
    (-(1 << 63), 1 << 30, 4096, 0xF1A9355B, 0x3AED160F),
    ((1 << 63) - 1, (1 << 30) + 7, 0xFFFFFFFF, 0xD952F9AA, 0x49E8EA7F),
)
MIN_WINNER_MARGIN = 2e-3


def murmur3(seed, position, offsets, domain=0):
    """Vectorized unsigned integer oracle, independent of any Triton helper."""
    mask = np.uint64(0xFFFFFFFF)
    h = np.full(np.shape(offsets), domain, dtype=np.uint64)
    seed = int(seed) & ((1 << 64) - 1)
    for word in (seed & 0xFFFFFFFF, seed >> 32, int(position) & 0xFFFFFFFF, offsets):
        k = np.asarray(word, dtype=np.uint64)
        k = (k * np.uint64(0xCC9E2D51)) & mask
        k = ((k << np.uint64(15)) | (k >> np.uint64(17))) & mask
        h ^= (k * np.uint64(0x1B873593)) & mask
        h = ((h << np.uint64(13)) | (h >> np.uint64(19))) & mask
        h = (h * np.uint64(5) + np.uint64(0xE6546B64)) & mask
    h ^= np.uint64(16)
    h ^= h >> np.uint64(16)
    h = (h * np.uint64(0x85EBCA6B)) & mask
    h ^= h >> np.uint64(13)
    h = (h * np.uint64(0xC2B2AE35)) & mask
    return (h ^ (h >> np.uint64(16))).astype(np.uint32)


def check_known_answers():
    for seed, pos, offset, first, second in KAT_INPUTS:
        for domain, expected in ((0, first), (0x9E3779B9, second)):
            assert int(murmur3(seed, pos, offset, domain)) == expected
            assert int(murmur3(seed, pos + (1 << 32), offset, domain)) == expected


def winners(logits, mapping, temperature, seeds, positions):
    """Return independent FP32 winners and mathematical winner margins.

    log1p is evaluated in FP64 and rounded into the FP32 score. The existing
    contract's 2e-3 stable-winner margin guards compiler log/div differences;
    qualification rejects an ambiguous row rather than dropping it.
    """
    check_known_answers()
    expected, margins = [], []
    keys = np.arange(logits.shape[1], dtype=np.uint64)
    for row, request in enumerate(mapping):
        values = np.asarray(logits[row], dtype=np.float32)
        temp = np.float32(temperature[int(request)])
        if temp == 0:
            # np.argmax returns the first maximum, including repeated maxima.
            expected.append(int(np.argmax(values)))
            margins.append(None)
            continue
        hashed = murmur3(int(seeds[int(request)]), int(positions[row]), keys)
        uniform = (hashed >> 16).astype(np.float32) * np.float32(2.0**-16)
        uniform += (
            (hashed & 0xFFFF).astype(np.float32) + np.float32(0.5)
        ) * np.float32(2.0**-32)
        with np.errstate(divide="ignore"):
            noise = -np.log(-np.log1p(-uniform.astype(np.float64)))
        scores = (values / temp + noise).astype(np.float32)
        winner = int(np.argmax(scores))
        top = np.partition(scores, -2)[-2:]
        margin = float(np.max(top) - np.min(top))
        if not margin > MIN_WINNER_MARGIN:
            raise ValueError(
                f"independent oracle has ambiguous row {row}: margin={margin}"
            )
        expected.append(winner)
        margins.append(margin)
    return expected, margins


def check_native_hashes(module, device):
    """Probe the resolved native's own hash helpers against independent KATs."""
    global triton, tl
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def probe(seed_ptr, pos_ptr, key_ptr, out_ptr, hash_fn: tl.constexpr):
        i = tl.arange(0, 8)
        seed = tl.load(seed_ptr + i, i < 6, 0)
        pos = tl.load(pos_ptr + i, i < 6, 0)
        key = tl.load(key_ptr + i, i < 6, 0)
        first = hash_fn(seed, pos, key)
        second = hash_fn(seed, pos, key, domain=0x9E3779B9)
        tl.store(out_ptr + i, first.to(tl.int64), i < 6)
        tl.store(out_ptr + 6 + i, second.to(tl.int64), i < 6)

    columns = [
        torch.tensor(
            [row[index] for row in KAT_INPUTS], dtype=torch.int64, device=device
        )
        for index in range(3)
    ]
    output = torch.empty(12, dtype=torch.int64, device=device)
    probe[(1,)](*columns, output, module.murmur3_hash32)
    expected = [row[3] for row in KAT_INPUTS] + [row[4] for row in KAT_INPUTS]
    if output.cpu().tolist() != expected:
        raise AssertionError("native Murmur3 known-answer check failed")
