# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-Text-01 TP8 cases and references from Kafka-Hatsune's LightningAttn branch."""
import torch

HEADS = 8
DIM = 128
CACHE_ROWS = 10749

MINIMAX_PREFIX_LENGTHS = (
    2,
    16,
    31,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    512,
    1023,
    1024,
    1025,
    1536,
    2048,
    4095,
    4096,
    4097,
    8191,
    8192,
    12185,
    16383,
    16384,
)

MINIMAX_DECODE_BATCHES = (
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    10,
    14,
    15,
    16,
    17,
    24,
    25,
    31,
    32,
    33,
    48,
    63,
    64,
    65,
    96,
    127,
    128,
)


def make_qkv(batch, tokens):
    return tuple(
        torch.randn(batch, HEADS, tokens, DIM, device="npu").mul_(0.125)
        for _ in range(3)
    )


def make_slope():
    # TP rank 0 measured slopes from Kafka's fixture.
    return torch.tensor(
        [
            0.91701323,
            0.84090483,
            0.77111316,
            0.70711386,
            0.64842629,
            0.59460950,
            0.54525936,
            0.50000501,
        ],
        device="npu",
    ).view(HEADS, 1, 1)


def make_initial(batch, dtype=torch.bfloat16, *, device="npu"):
    return torch.empty((batch, HEADS, DIM, DIM), device=device, dtype=dtype).normal_(
        std=0.125
    )


def make_decode_case(batch):
    qkv, slope = make_qkv(batch, 1), make_slope()
    cache = torch.empty(
        (CACHE_ROWS, HEADS, DIM, DIM), device="npu", dtype=torch.bfloat16
    )
    start = CACHE_ROWS - batch
    slots = torch.arange(start, CACHE_ROWS, device="npu", dtype=torch.int32)
    initial = make_initial(batch, device="cpu").to("npu")
    assert (CACHE_ROWS - 1) * cache.stride(0) * cache.element_size() > 2**31
    return qkv, slope, cache, slots, initial


def prefix_reference(q, k, v, slope, initial=None, dtype=torch.float64):
    """Dense causal reference inside each native 256-token history block."""
    q, k, v = (x.to(dtype) for x in (q, k, v))
    b, h, t, d = q.shape
    e = v.shape[-1]
    slope = slope.reshape(1, h, 1, 1).to(dtype)
    state = q.new_zeros((b, h, d, e)) if initial is None else initial.to(dtype)
    output, snapshots = [], []
    for start in range(0, t, 256):
        stop = min(start + 256, t)
        row = torch.arange(stop - start, device=q.device)
        age = row[:, None] - row[None, :]
        weight = torch.exp(-slope * age.clamp_min(0)) * (age >= 0)
        qs, ks, vs = (x[:, :, start:stop] for x in (q, k, v))
        inner = ((qs @ ks.transpose(-1, -2)) * weight) @ vs
        # vLLM's prefix read starts at power zero for a new 256-token block.
        output.append(inner + (qs @ state) * torch.exp(-slope * row[:, None]))
        snapshots.append(state)
        tail = torch.exp(-slope * (len(row) - 1 - row[:, None])) * vs
        state = torch.exp(-slope * len(row)) * state + ks.transpose(-1, -2) @ tail
    snapshots.append(state)
    y = torch.cat(output, dim=2) if output else q.new_empty((b, h, 0, e))
    return y, torch.stack(snapshots, dim=2)


def decode_reference(q, k, v, cache, slope, slots, dtype=torch.float32):
    """Compute live rows in the requested precision and round the persistent cache each step."""
    active = slots >= 0
    idx = slots[active].long()
    old = cache[idx].to(dtype)
    state = torch.exp(-slope.to(dtype).reshape(1, -1, 1, 1)) * old
    state = state + k[active, :, 0, :, None].to(dtype) * v[active, :, 0, None, :].to(
        dtype
    )
    result = torch.zeros(
        (q.shape[0], q.shape[1] * q.shape[-1]), device=q.device, dtype=q.dtype
    )
    result[active] = (
        (q[active].to(dtype) @ state).reshape(-1, result.shape[-1]).to(q.dtype)
    )
    cache[idx] = state.to(cache.dtype)
    return result


def assert_relative(actual, expected, limit):
    assert actual.shape == expected.shape
    if not actual.numel():
        return 0.0
    expected = expected.double()
    actual = actual.to(device=expected.device, dtype=torch.float64)
    error = (
        torch.linalg.vector_norm(actual - expected)
        / torch.linalg.vector_norm(expected).clamp_min(1e-12)
    ).item()
    assert error <= limit, f"relative L2 = {error:.7g}, limit={limit}"
    return error
