# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Logical shifts and seeded uniform draws for the V2 Gumbel RNG."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class _Unsigned32WithArithmeticShift:
    """Model the observed NPU signed-shift lowering and uint32 arithmetic."""

    def __init__(self, value):
        self.bits = int(value) & 0xFFFFFFFF

    def __int__(self):
        return self.bits

    def __and__(self, other):
        return type(self)(self.bits & int(other))

    def __or__(self, other):
        return type(self)(self.bits | int(other))

    def __xor__(self, other):
        return type(self)(self.bits ^ int(other))

    def __mul__(self, other):
        return type(self)(self.bits * int(other))

    def __add__(self, other):
        return type(self)(self.bits + int(other))

    def __lshift__(self, shift):
        return type(self)(self.bits << shift)

    def __rshift__(self, shift):
        signed = self.bits if self.bits < 0x80000000 else self.bits - 0x100000000
        return type(self)(signed >> shift)


def _production_integer_helpers():
    path = (
        Path(__file__).resolve().parents[1] / "src/flaggems_vllm/ops/v2_gumbel_rng.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_murmur3_rotl32", "_murmur3_mix", "_murmur3_fmix32"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    namespace = {"tl": SimpleNamespace(constexpr=int)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _scalar_mix(h, key):
    key = (key * 0xCC9E2D51) & 0xFFFFFFFF
    key = ((key << 15) | (key >> 17)) & 0xFFFFFFFF
    key = (key * 0x1B873593) & 0xFFFFFFFF
    h ^= key
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    return (h * 5 + 0xE6546B64) & 0xFFFFFFFF


@pytest.mark.v2_gumbel_rng
def test_murmur_helpers_preserve_unsigned_bits_under_arithmetic_right_shift():
    functions = _production_integer_helpers()
    values = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xCC9E2D51]
    values += np.random.default_rng(13).integers(0, 1 << 32, 100).tolist()
    for value in values:
        wrapped = _Unsigned32WithArithmeticShift(value)
        for shift in (13, 15):
            expected = ((value << shift) | (value >> (32 - shift))) & 0xFFFFFFFF
            assert int(functions["_murmur3_rotl32"](wrapped, shift)) == expected
        assert int(functions["_murmur3_mix"](wrapped, wrapped)) == _scalar_mix(
            value, value
        )
        expected = value ^ (value >> 16)
        expected = (expected * 0x85EBCA6B) & 0xFFFFFFFF
        expected ^= expected >> 13
        expected = (expected * 0xC2B2AE35) & 0xFFFFFFFF
        expected ^= expected >> 16
        assert int(functions["_murmur3_fmix32"](wrapped)) == expected


def _cpu_draws(seed, position, vocab_size, domain=0):
    def mix(h, key):
        key = np.asarray(key, dtype=np.uint32) * np.uint32(0xCC9E2D51)
        key = (key << np.uint32(15)) | (key >> np.uint32(17))
        key = key * np.uint32(0x1B873593)
        h = np.asarray(h, dtype=np.uint32) ^ key
        h = (h << np.uint32(13)) | (h >> np.uint32(19))
        return h * np.uint32(5) + np.uint32(0xE6546B64)

    seed = int(seed) & ((1 << 64) - 1)
    with np.errstate(over="ignore"):
        h = mix(np.uint32(domain), np.uint32(seed & 0xFFFFFFFF))
        h = mix(h, np.uint32(seed >> 32))
        h = mix(h, np.uint32(position & 0xFFFFFFFF))
        h = mix(h, np.arange(vocab_size, dtype=np.uint32)) ^ np.uint32(16)
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x85EBCA6B)
        h ^= h >> np.uint32(13)
        h *= np.uint32(0xC2B2AE35)
        h ^= h >> np.uint32(16)
    hi = (h >> np.uint32(16)).astype(np.float32)
    lo = (h & np.uint32(0xFFFF)).astype(np.float32)
    u = hi * np.float32(2**-16) + (lo + np.float32(0.5)) * np.float32(2**-32)
    return h, u


@pytest.mark.v2_gumbel_rng
@pytest.mark.parametrize(
    "seed,position,vocab_size",
    [(7, 13, 1031), ((1 << 40) + 7, 5, 151939), (-31, 9, 151939)],
)
def test_v2_gumbel_rng_matches_cpu(seed, position, vocab_size):
    import torch

    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    import triton
    import triton.language as tl

    from flaggems_vllm import v2_gumbel_rng

    @triton.jit
    def sample_hash_and_uniform(
        seed_ptr, pos_ptr, hashes, uniforms, vocab_size, BLOCK_SIZE: tl.constexpr
    ):
        idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        seed_value = tl.load(seed_ptr)
        position_value = tl.load(pos_ptr)
        bits = v2_gumbel_rng.murmur3_hash32(seed_value, position_value, idx)
        uniform = v2_gumbel_rng.murmur3_uniform32(seed_value, position_value, idx)
        tl.store(hashes + idx, bits.to(tl.int32), mask=idx < vocab_size)
        tl.store(uniforms + idx, uniform, mask=idx < vocab_size)

    device = torch.device("npu", torch.npu.current_device())
    hashes = torch.empty(vocab_size, dtype=torch.int32, device=device)
    uniforms = torch.empty(vocab_size, dtype=torch.float32, device=device)
    seed_device = torch.tensor(seed, dtype=torch.int64, device=device)
    pos_device = torch.tensor(position, dtype=torch.int64, device=device)
    expected_hash, expected_uniform = _cpu_draws(seed, position, vocab_size)
    for _ in range(2):
        sample_hash_and_uniform[(triton.cdiv(vocab_size, 1024),)](
            seed_device, pos_device, hashes, uniforms, vocab_size, BLOCK_SIZE=1024
        )
        np.testing.assert_array_equal(
            hashes.cpu().numpy().view(np.uint32), expected_hash
        )
        np.testing.assert_array_equal(uniforms.cpu().numpy(), expected_uniform)
