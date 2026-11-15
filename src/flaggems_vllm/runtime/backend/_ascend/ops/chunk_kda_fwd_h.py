# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the GLM5Next resident H implementation (Apache-2.0).

"""Ascend resident KDA H: fixed chunks, FP32 recurrence, native BF16 boundaries."""
from functools import lru_cache
from numbers import Integral
from operator import index

import torch
import torch_npu
import triton as tr
import triton.language as tl
from triton.language.extra.cann import extension as tlex


@lru_cache(maxsize=1)
def _launch_options():
    # FlagTree 3.5 has no graph-optimize option; triton-ascend 3.6 does.
    from triton.compiler.compiler import make_backend
    from triton.runtime import driver

    backend = make_backend(driver.active.get_current_target())
    supported = backend.parse_options({})
    return (
        {"enable_graph_optimize": False}
        if hasattr(supported, "enable_graph_optimize")
        else {}
    )


# The CANN dot extension was added after FlagTree's 3.5 backend. Both paths
# use BF16 multiplicands and FP32 accumulation (no TF32 input is involved).
_update_dot = getattr(tlex, "dot", tl.dot)

CHUNK = 64
DIM = 128
LN2 = 0.6931471805599453


@tr.jit
def _strict_h(
    K,
    W,
    U,
    G,
    INITIAL,
    SNAPSHOTS,
    VALUES,
    FINAL,
    T: tl.constexpr,
    H: tl.constexpr,
    CU: tl.constexpr,
    CHUNK_OFFSETS: tl.constexpr,
    TOTAL_CHUNKS: tl.constexpr,
    BV: tl.constexpr,
    CHUNK: tl.constexpr,
    DIM: tl.constexpr,
    LN2: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    value_tile = tl.program_id(2)
    start = tl.full((), 0, tl.int32)
    end = tl.full((), 0, tl.int32)
    first_chunk = tl.full((), 0, tl.int32)
    for s in tl.static_range(len(CU) - 1):
        start = tl.where(seq == s, CU[s], start)
        end = tl.where(seq == s, CU[s + 1], end)
        first_chunk = tl.where(seq == s, CHUNK_OFFSETS[s], first_chunk)
    kk = tl.arange(0, DIM)
    vv = value_tile * BV + tl.arange(0, BV)
    rr = tl.arange(0, CHUNK)
    state_offset = (seq * H + head) * DIM * DIM + vv[:, None] * DIM + kk[None, :]
    state = tl.load(INITIAL + state_offset).to(tl.float32)
    for c in range(tl.cdiv(end - start, CHUNK)):
        tokens = start + c * CHUNK + rr
        valid = tokens < end
        # Native H writes the pre-update state in BF16 VK layout for Finalize.
        rounded_state = state.to(tl.bfloat16)
        # Native VK snapshots are [B,total_chunks,H,V,K].
        # Write the native layout directly; no output transpose or copy.
        snap_offset = (
            ((first_chunk + c) * H + head) * DIM * DIM + vv[:, None] * DIM + kk[None, :]
        )
        tl.store(SNAPSHOTS + snap_offset, rounded_state)
        w = tl.load(
            W + (head * T + tokens[:, None]) * DIM + kk[None, :],
            valid[:, None],
            other=0,
        )
        # FP32 accumulation, BF16 multiplicands, exactly as native Stage 0.
        prediction = tl.dot(w, tl.trans(rounded_state), allow_tf32=False)
        u = tl.load(
            U + (head * T + tokens[:, None]) * DIM + vv[None, :],
            valid[:, None],
            other=0,
        ).to(tl.float32)
        values = (u - prediction).to(tl.bfloat16)
        tl.store(
            VALUES + (head * T + tokens[:, None]) * DIM + vv[None, :],
            values,
            valid[:, None],
        )
        kg = tl.load(
            K + (head * T + tokens[:, None]) * DIM + kk[None, :],
            valid[:, None],
            other=0,
        )
        # Padded rows must contribute zero, including U-P when valid is false.
        values = tl.where(valid[:, None], values, 0).to(tl.bfloat16)
        delta = _update_dot(tl.trans(values), kg)
        last = tl.minimum(start + (c + 1) * CHUNK, end) - 1
        gate = tl.load(G + (head * T + last) * DIM + kk)
        decay = tl.exp(gate * LN2)
        # enable_fp_fusion=False at launch preserves native Mul then Add.
        state = state * decay[None, :] + delta
    tl.store(FINAL + state_offset, state)


def _validate(k, w, u, gk, initial_state, cu_seqlens, value_tile):
    if not isinstance(value_tile, Integral) or value_tile not in (32, 64, 128):
        raise ValueError("value_tile must be 32, 64 or 128")
    if k.ndim != 4 or k.shape[0] != 1 or k.shape[-1] != DIM or k.shape[1] <= 0:
        raise ValueError("expected BNSD [1,H,T,128] with H > 0")
    if k.device.type != "npu":
        raise NotImplementedError("only Ascend NPU is supported")
    for name, tensor in (("k", k), ("w", w), ("u", u), ("gk", gk)):
        dtype = torch.float32 if name == "gk" else torch.bfloat16
        if tensor.shape != k.shape or tensor.device != k.device:
            raise ValueError(f"{name}: shape/device must match k")
        if tensor.dtype != dtype or not tensor.is_contiguous():
            raise NotImplementedError(f"{name}: requires contiguous {dtype}")
    if not isinstance(cu_seqlens, (list, tuple)):
        raise NotImplementedError("cu_seqlens must be a host list or tuple")
    if not all(isinstance(x, Integral) for x in cu_seqlens):
        raise NotImplementedError("cumulative lengths must contain host integers")
    cu = tuple(index(x) for x in cu_seqlens)
    if (
        len(cu) < 2
        or cu[0] != 0
        or cu[-1] != k.shape[2]
        or any(a > b for a, b in zip(cu, cu[1:]))
    ):
        raise ValueError("cumulative lengths must be nondecreasing and cover T")
    expected = (len(cu) - 1, k.shape[1], DIM, DIM)
    if initial_state.shape != expected or initial_state.device != k.device:
        raise ValueError("initial_state must have same-device shape [N,H,V,K]")
    if initial_state.dtype != torch.float32 or not initial_state.is_contiguous():
        raise NotImplementedError("initial_state must be contiguous FP32")
    for tensor in (k, w, u, gk, initial_state):
        if tensor.requires_grad:
            raise NotImplementedError("only forward inference is supported")
        if torch_npu.get_npu_format(tensor) not in (0, 2):
            raise NotImplementedError(
                "only physically standard NCHW/ND tensors are supported"
            )
    return cu


def chunk_kda_fwd_h(k, w, u, *, gk, initial_state, cu_seqlens, value_tile=128):
    """Return native-shaped (h BF16, v_new BF16, final_state FP32).

    Input state is never mutated. CPU metadata specialization avoids an extra
    host-to-device metadata copy; compilation is excluded only after warmup.
    Unsupported contracts raise; no framework compute fallback is used.
    """
    cu = _validate(k, w, u, gk, initial_state, cu_seqlens, value_tile)
    value_tile = index(value_tile)
    offsets = [0]
    for a, b in zip(cu, cu[1:]):
        offsets.append(offsets[-1] + tr.cdiv(b - a, CHUNK))
    total_chunks = offsets[-1]
    _, heads, tokens, _ = k.shape
    snapshots = torch.empty(
        (1, total_chunks, heads, DIM, DIM), device=k.device, dtype=k.dtype
    )
    values = torch.empty_like(u)
    final = torch.empty_like(initial_state)
    _strict_h[(len(cu) - 1, heads, DIM // value_tile)](
        k,
        w,
        u,
        gk,
        initial_state,
        snapshots,
        values,
        final,
        tokens,
        heads,
        cu,
        tuple(offsets),
        total_chunks,
        value_tile,
        CHUNK,
        DIM,
        LN2,
        num_warps=4,
        enable_fp_fusion=False,
        **_launch_options(),
    )
    return snapshots, values, final
