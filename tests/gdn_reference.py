# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Independent float64 oracle and explicit workloads for recurrent GDN."""
import torch

CASES = [
    ("decode", [1] * 4, 4, 8, 128, 128, "index", True, False),
    ("rectangular", [1] * 3, 2, 4, 17, 23, "index", True, True),
    ("uniform", [4, 4], 2, 4, 32, 16, "plain", True, True),
    ("ragged", [0, 2, 1, 5], 2, 6, 32, 19, "table", True, True),
    ("spec", [1, 3, 2], 2, 4, 64, 32, "spec", True, True),
    ("outofplace", [2, 0, 5], 2, 4, 17, 23, "table", False, True),
    ("zero_initial", [3, 3], 1, 2, 32, 16, "none", False, False),
    ("vector_beta", [4, 2], 2, 4, 32, 23, "table", True, True),
    ("padding", [1, 1, 1, 1], 2, 4, 17, 23, "index", True, True),
    ("low_state", [6, 3], 2, 4, 32, 16, "table", True, True),
    ("long", [65], 1, 2, 32, 16, "table", False, True),
    ("empty", [0], 1, 2, 17, 23, "index", True, False),
]


def build(case, dtype, device, normalize=True):
    name, lengths, H, HV, K, V, mode, inplace, strided = case
    torch.manual_seed(2701)
    N = len(lengths)
    plain = mode in ("plain", "none")
    B, T = (N, lengths[0]) if plain else (1, sum(lengths))

    def rand(shape, dt=dtype):
        if strided:
            x = torch.randn((*shape[:-1], shape[-1] * 2), dtype=dt)[..., ::2]
        else:
            x = torch.randn(shape, dtype=dt)
        # Preserve the tested strides after device transfer.
        out = torch.empty_strided(x.shape, x.stride(), dtype=dt, device=device)
        out.copy_(x)
        return out

    q, k, v = rand((B, T, H, K)), rand((B, T, H, K)), rand((B, T, HV, V))
    k.mul_(0.05)
    g = rand((B, T, HV), torch.float32)
    g.copy_(-g.abs() * 0.1)
    beta = rand((B, T, HV, V) if name == "vector_beta" else (B, T, HV))
    beta.copy_(beta.sigmoid())
    width = max(max(lengths), 4)
    pool = N * width + 3
    state_dtype = dtype if name == "low_state" else torch.float32
    state = rand((pool, HV, V, K), state_dtype)
    state.mul_(0.05)
    indices = None
    if mode in ("index", "table", "spec"):
        if mode == "index":
            indices = torch.arange(1, N + 1, dtype=torch.int32)
            if name == "padding":
                indices[:2] = torch.tensor([0, -1], dtype=torch.int32)
        else:
            indices = torch.arange(1, N * width + 1, dtype=torch.int64).view(N, width)
        if strided:
            buf = torch.empty(
                (*indices.shape[:-1], indices.shape[-1] * 2), dtype=indices.dtype
            )
            buf[..., ::2] = indices
            indices = buf[..., ::2]
        d = torch.empty_strided(
            indices.shape, indices.stride(), dtype=indices.dtype, device=device
        )
        d.copy_(indices)
        indices = d
    cu = (
        None
        if plain
        else torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
    )
    accepted = (
        torch.tensor([3] * N, dtype=torch.int32, device=device)
        if mode == "spec"
        else None
    )
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=None if mode == "none" else state,
        inplace_final_state=inplace,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=normalize,
    )


def reference(data):
    q, k, v = (data[x].detach().cpu().double() for x in ("q", "k", "v"))
    g = data["g"].cpu().double() if data["g"] is not None else None
    beta = data["beta"].cpu().double() if data["beta"] is not None else None
    state = data["initial_state"]
    state = state.cpu().clone() if state is not None else None
    B, T, H, K = q.shape
    HV, V = v.shape[-2:]
    indices = data["ssm_state_indices"]
    indices = indices.cpu() if indices is not None else None
    cu = data["cu_seqlens"]
    bounds = (
        list(zip(cu.cpu().tolist()[:-1], cu.cpu().tolist()[1:]))
        if cu is not None
        else [(0, T)] * B
    )
    accepted = data["num_accepted_tokens"]
    output = torch.zeros(v.shape, dtype=torch.float64)
    final = (
        state.clone()
        if data["inplace_final_state"]
        else torch.zeros(
            (B * T, HV, V, K), dtype=state.dtype if state is not None else torch.float32
        )
    )
    for n, (start, end) in enumerate(bounds):
        batch = 0 if cu is not None else n
        slot = n
        if indices is not None:
            col = int(accepted[n].item()) - 1 if accepted is not None else 0
            slot = int(indices[n, col] if indices.ndim == 2 else indices[n])
            if slot <= 0 or slot >= len(state):
                continue
        acc = (
            state[slot].double().clone()
            if state is not None
            else torch.zeros((HV, V, K), dtype=torch.float64)
        )
        for t in range(start, end):
            qv, kv = q[batch, t], k[batch, t]
            if data["use_qk_l2norm_in_kernel"]:
                qv = qv / (qv.square().sum(-1, keepdim=True) + 1e-6).sqrt()
                kv = kv / (kv.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            qv = qv.repeat_interleave(HV // H, 0) * data["scale"]
            kv = kv.repeat_interleave(HV // H, 0)
            if g is not None:
                acc = acc * g[batch, t].exp()[:, None, None]
            residual = v[batch, t] - (acc * kv[:, None, :]).sum(-1)
            if beta is not None:
                bv = beta[batch, t]
                residual = residual * (bv[:, None] if bv.ndim == 1 else bv)
            acc = acc + residual[:, :, None] * kv[:, None, :]
            output[batch, t] = (acc * qv[:, None, :]).sum(-1)
            if data["inplace_final_state"]:
                if indices is not None and indices.ndim == 2:
                    dest = int(indices[n, t - start])
                    if 0 < dest < len(final):
                        final[dest] = acc
                elif t == end - 1:
                    final[slot] = acc
            else:
                final[batch * T + t] = acc
    return output.to(data["v"].dtype), final


def assert_result(result, expected, dtype):
    tol = {
        torch.float16: (3e-3, 3e-3),
        torch.bfloat16: (2e-2, 2e-2),
        torch.float32: (3e-5, 3e-5),
    }
    for got, ref in zip(result, expected):
        rtol, atol = tol[got.dtype]
        torch.testing.assert_close(got.detach().cpu(), ref, rtol=rtol, atol=atol)
