# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import torch


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
