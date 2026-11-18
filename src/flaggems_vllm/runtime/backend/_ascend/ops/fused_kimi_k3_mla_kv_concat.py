# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Token/head tiles avoid vector integer division on Ascend and reuse PE."""

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fused_kimi_k3_mla_kv_concat_ascend"),
    key=["T", "H", "NS0", "NS1", "PS0"],
    restore_value=["OUT"],
)
@triton.jit
def _kv_concat_ascend_kernel(
    N, P, OUT, T, H: tl.constexpr, NS0, NS1, PS0, HEADS_PER_PROGRAM: tl.constexpr
):
    token = tl.program_id(1).to(tl.int64)
    head = tl.program_id(0) * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
    dim = tl.arange(0, 256)
    # Only small, power-of-two tile indices are vectors. All global address
    # arithmetic stays int64, with no large per-element integer division.
    n_offset = token * NS0 + head[:, None].to(tl.int64) * NS1 + dim[None, :]
    n = tl.load(N + n_offset, (head[:, None] < H) & (dim[None, :] < 128), other=0)
    p = tl.load(P + token * PS0 + dim - 128, (dim >= 128) & (dim < 192), other=0)
    value = tl.where(dim[None, :] < 128, n, p[None, :])
    out_offset = token * H * 192 + head[:, None].to(tl.int64) * 192 + dim[None, :]
    tl.store(OUT + out_offset, value, (head[:, None] < H) & (dim[None, :] < 192))


def _launch_ascend(k_nope, k_pe, k_out):
    """Internal launcher; the shared public entry validates and guards devices."""
    tokens, heads, _ = k_nope.shape
    _kv_concat_ascend_kernel[
        lambda meta: (triton.cdiv(heads, meta["HEADS_PER_PROGRAM"]), tokens)
    ](
        k_nope,
        k_pe,
        k_out,
        tokens,
        heads,
        k_nope.stride(0),
        k_nope.stride(1),
        k_pe.stride(0),
    )
