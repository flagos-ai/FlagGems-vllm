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

"""TLE launch ABI for the native SIMD row tiles."""

import triton
import triton.language as tl
from triton.experimental.tle import language as tle
from triton.language import core


@core.extern
def _simt_tile(*args, W, R, N, _semantic=None):
    w, r, n = W.value, R.value, N.value
    return core.extern_elementwise(
        "megamhc",
        "",
        list(args),
        {
            (tl.uint64,) * 16
            + (tl.uint32,) * 3
            + (tl.float32,) * 5
            + (tl.uint32,): (f"megamhc_simt_{w}_{r}_{n}", tl.int32)
        },
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def _simt_rows(
    pointers,
    scalars,
    first,
    T: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
):
    if T % N == 0 or first + N <= T:
        _simt_tile(*pointers, first, *scalars, W=W, R=R, N=N)
    else:
        for row in range(N):
            token = first + row
            if token < T:
                _simt_tile(*pointers, token, *scalars, W=W, R=R, N=1)


@triton.jit
def mega_mhc_simt_kernel(
    X,
    Residual,
    Post,
    Comb,
    Fn,
    Scale,
    Base,
    Norm,
    NewResidual,
    NewPost,
    NewComb,
    Y,
    comb_stride_t,
    comb_stride_src,
    comb_stride_dst,
    rms_eps,
    pre_eps,
    sinkhorn_eps,
    post_multiplier,
    sinkhorn_repeat,
    norm_eps,
    T: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    HAS_NORM: tl.constexpr,
    ROWS: tl.constexpr,
    WARPS: tl.constexpr,
    LARGE_BLOCKS: tl.constexpr = 0,
):
    shared = tle.gpu.alloc(
        [triton.next_power_of_2(2 * ROWS * 32 + ROWS * R + WARPS + ROWS)],
        dtype=tl.float32,
        scope=tle.gpu.smem,
        layout=None,
    )
    address = tle.gpu.local_ptr(shared, (0,)).to(tl.uint64).to(tl.uint32)
    pointers = (
        X.to(tl.uint64),
        Residual.to(tl.uint64),
        Post.to(tl.uint64),
        Comb.to(tl.uint64),
        Fn.to(tl.uint64),
        Scale.to(tl.uint64),
        Base.to(tl.uint64),
        Norm.to(tl.uint64),
        NewResidual.to(tl.uint64),
        NewPost.to(tl.uint64),
        NewComb.to(tl.uint64),
        Y.to(tl.uint64),
    )
    scalars = (
        tl.cast(comb_stride_t, tl.uint64),
        tl.cast(comb_stride_src, tl.uint64),
        tl.cast(comb_stride_dst, tl.uint64),
        tl.full((), H, tl.uint32),
        address,
        tl.cast(sinkhorn_repeat, tl.uint32),
        rms_eps,
        pre_eps,
        sinkhorn_eps,
        post_multiplier,
        norm_eps,
        tl.full((), int(HAS_NORM), tl.uint32),
    )
    pid = tl.program_id(0).to(tl.uint64)
    if LARGE_BLOCKS:
        if pid < LARGE_BLOCKS:
            _simt_rows(pointers, scalars, pid * ROWS, T, WARPS, R, ROWS)
        else:
            first = LARGE_BLOCKS * ROWS + (pid - LARGE_BLOCKS) * (ROWS // 2)
            _simt_rows(pointers, scalars, first, T, WARPS, R, ROWS // 2)
    else:
        _simt_rows(pointers, scalars, pid * ROWS, T, WARPS, R, ROWS)
