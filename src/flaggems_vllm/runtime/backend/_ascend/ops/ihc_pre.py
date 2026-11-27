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

"""iHC pre schedules for Ascend vector and cube cores."""

import inspect
import logging
import struct

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.ihc.ihc_pre import _ihc_pre, _launch_ihc_pre

logger = logging.getLogger(__name__)

# Bind only the tested FlagTree launcher ABI. Other installations use the normal
# Triton launch on every call.
_use_bound_launch = False
if triton.__version__ == "3.5.1":
    try:
        from torch._C._dynamo.guards import TensorGuards
        from torch_npu._C import _npu_getCurrentRawStreamNoWait, _npu_getDevice
        from triton import knobs
        from triton.backends.ascend import utils as backend_utils
        from triton.backends.ascend.driver import NPULauncher
    except ImportError:
        pass
    else:
        _use_bound_launch = (
            "do_not_specialize_on_alignment" in inspect.signature(triton.jit).parameters
            and hasattr(knobs, "HookChain")
            and hasattr(torch._C, "_len_torch_dispatch_stack")
            and hasattr(torch._C, "_is_torch_function_mode_enabled")
        )


# The cached runner can receive new tensors, including views with a different
# storage offset. Keep pointer alignment out of these kernels' specialization.
_pointer_options = (
    {"do_not_specialize_on_alignment": ("X", "W", "S", "B", "Y", "P")}
    if _use_bound_launch
    else {}
)


_weight_options = (
    {"do_not_specialize_on_alignment": ("W",)} if _use_bound_launch else {}
)


@triton.jit(**_pointer_options)
def _short_projection(
    X,
    W,
    S,
    B,
    Y,
    P,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BASE_STRIDE: tl.constexpr,
    MAG: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, BK)
    d = tl.arange(0, BD)
    length: tl.constexpr = H * D
    x = tl.load(X + row * length + k, k < length, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / length + NORM)
    sc = tl.load(S)
    ps = tl.load(S + SCALE_STRIDE)
    y = tl.full((BD,), 0, tl.float32)
    for j in tl.static_range(H):
        w = tl.load(W + j * length + k, k < length, 0)
        wp = tl.load(W + (H + j) * length + k, k < length, 0)
        gate = (
            tl.sigmoid(tl.sum(x * w, 0) * inv * sc + tl.load(B + j * BASE_STRIDE)) + EPS
        )
        post = (
            MAG
            * tl.sigmoid(
                tl.sum(x * wp, 0) * inv * ps + tl.load(B + (H + j) * BASE_STRIDE)
            )
            + EPS
        )
        xx = tl.load(X + (row * H + j) * D + d, d < D, 0).to(tl.float32)
        y += gate * xx
        tl.store(P + row * H + j, post)
    tl.store(Y + row * D + d, y, d < D)


@triton.jit
def _decode_tile(X, tile, D: tl.constexpr, WIDTH: tl.constexpr, STRIDED: tl.constexpr):
    if STRIDED:
        off = 2 * tile * WIDTH + tl.arange(0, 2 * WIDTH)
        # The unused lane of the last pair can fall beyond the input storage.
        words = tl.load(X + off, off < 2 * D - 1, 0)
        x, _ = tl.split(tl.reshape(words, (WIDTH, 2)))
    else:
        x = tl.load(X + tile * WIDTH + tl.arange(0, WIDTH))
    return x.to(tl.float32)


@triton.jit
def _decode_partial(product, BK: tl.constexpr):
    # Keep each short subtotal intact when distant K tiles cancel.
    partial = tl.sum(tl.reshape(product, (BK // 32, 32)), 1)
    high = (partial.to(tl.uint32, bitcast=True) & 0xFFFF0000).to(
        tl.float32, bitcast=True
    )
    low = tl.where(tl.abs(partial) == float("inf"), 0.0, partial - high)
    return tl.sum(high, 0), tl.sum(low, 0)


@triton.jit(**_pointer_options)
def _decode_projection(
    X,
    W,
    S,
    B,
    Y,
    P,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BASE_STRIDE: tl.constexpr,
    MAG: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    CHANNEL_STRIDE: tl.constexpr,
):
    # One output tile per vector core. The first H tiles also write a post gate,
    # using the activation and row norm already needed for their output.
    K: tl.constexpr = H * D
    TILES: tl.constexpr = D // BD
    pid = tl.program_id(0)
    if N == 1:
        row, tile = 0, pid
    else:
        row, tile = pid // TILES, pid % TILES
    post_channel = tile % H
    start = X + row * (TOKEN_STRIDE if CHANNEL_STRIDE else K)
    squares = 0.0
    post_high, post_low = 0.0, 0.0
    high, low = (), ()
    for j in tl.static_range(H):
        high += (0.0,)
        low += (0.0,)
    k = tl.arange(0, BK)
    for block in range(K // BK):
        col = block * BK + k
        if CHANNEL_STRIDE:
            channel = block // (D // BK)
            x = _decode_tile(
                start + channel * CHANNEL_STRIDE, block % (D // BK), D, BK, True
            )
        else:
            x = tl.load(start + col).to(tl.float32)
        squares += tl.sum(x * x, 0)
        post_weight = tl.load(W + (H + post_channel) * K + col)
        ph, pl = _decode_partial(x * post_weight, BK)
        post_high += ph
        post_low += pl
        next_high, next_low = (), ()
        for j in tl.static_range(H):
            weight = tl.load(W + j * K + col)
            wh, wl = _decode_partial(x * weight, BK)
            next_high += (high[j] + wh,)
            next_low += (low[j] + wl,)
        high, low = next_high, next_low
    inv = tl.rsqrt(squares / K + NORM)
    scale = tl.load(S)
    y = tl.full((BD,), 0, tl.float32)
    for j in tl.static_range(H):
        pre = tl.sigmoid(
            (high[j] + low[j]) * inv * scale + tl.load(B + j * BASE_STRIDE)
        )
        value = _decode_tile(
            start + j * (CHANNEL_STRIDE if CHANNEL_STRIDE else D),
            tile,
            D,
            BD,
            CHANNEL_STRIDE,
        )
        y += (pre + EPS) * value
    tl.store(Y + row * D + tile * BD + tl.arange(0, BD), y)
    if tile < H:
        post = (post_high + post_low) * inv * tl.load(S + SCALE_STRIDE)
        post = tl.sigmoid(post + tl.load(B + (H + post_channel) * BASE_STRIDE))
        tl.store(P + row * H + post_channel, MAG * post + EPS)


@triton.jit
def _load_stride_two(
    row, col, valid_rows, D: tl.constexpr, BM: tl.constexpr, WIDTH: tl.constexpr
):
    # Load contiguous pairs and discard the unused lane in local memory.
    # The last pair may end at the storage boundary of the original view.
    offset = 2 * col + tl.arange(0, 2 * WIDTH)
    lanes = tl.load(
        row + offset[None, :], valid_rows[:, None] & (offset[None, :] < 2 * D - 1), 0
    )
    even, _ = tl.split(tl.reshape(lanes, (BM, WIDTH, 2)))
    return even.to(tl.float32)


@triton.jit
def _write_outputs(
    X,
    S,
    B,
    Y,
    P,
    acc,
    sq,
    m,
    c,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BD: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BASE_STRIDE: tl.constexpr,
    MAG: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    CHANNEL_STRIDE: tl.constexpr,
):
    inv = tl.rsqrt(sq / (H * D) + NORM)
    sc = tl.load(S)
    ps = tl.load(S + SCALE_STRIDE)
    pre = (
        tl.sigmoid(
            acc * inv[:, None] * sc + tl.load(B + c * BASE_STRIDE, c < H, 0)[None, :]
        )
        + EPS
    )
    post = (
        MAG
        * tl.sigmoid(
            acc * inv[:, None] * ps
            + tl.load(B + c * BASE_STRIDE, c < 2 * H, 0)[None, :]
        )
        + EPS
    )
    tl.store(
        P + m[:, None] * H + c[None, :] - H,
        post,
        (m[:, None] < N) & (c[None, :] >= H) & (c[None, :] < 2 * H),
    )
    gates = ()
    for j in tl.static_range(H):
        gates += (tl.sum(tl.where(c[None, :] == j, pre, 0.0), 1),)
    for i in range(triton.cdiv(D, BD)):
        off = i * BD + tl.arange(0, BD)
        mask = (m[:, None] < N) & (off[None, :] < D)
        y = tl.full((BM, BD), 0, tl.float32)
        for j in tl.static_range(H):
            if CHANNEL_STRIDE:
                row = X + m[:, None] * TOKEN_STRIDE + j * CHANNEL_STRIDE
                x = _load_stride_two(row, i * BD, m < N, D, BM, BD)
            else:
                x = tl.load(X + (m[:, None] * H + j) * D + off[None, :], mask, 0).to(
                    tl.float32
                )
            y += gates[j][:, None] * x
        tl.store(Y + m[:, None] * D + off[None, :], y, mask)


@triton.jit(**_pointer_options)
def _project_and_gate(
    X,
    W,
    S,
    B,
    Y,
    P,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    BASE_STRIDE: tl.constexpr,
    MAG: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    CHANNEL_STRIDE: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    k = tl.arange(0, BK)
    NC: tl.constexpr = 2 * H if BM <= 8 else 16
    c = tl.arange(0, NC)
    # Keep the eight FP32 projections on the cube cores and reuse each input
    # tile for the row norm. Only the final outputs go to global memory.
    acc = tl.full((NC, BM), 0, tl.float32)
    sq = tl.full((BM,), 0, tl.float32)
    for i in range(triton.cdiv(H * D, BK)):
        off = i * BK + k
        if CHANNEL_STRIDE:
            channel = i // (D // BK)
            col = (i - channel * (D // BK)) * BK
            row = X + m[:, None] * TOKEN_STRIDE + channel * CHANNEL_STRIDE
            x = _load_stride_two(row, col, m < N, D, BM, BK)
        else:
            x = tl.load(
                X + m[:, None] * (H * D) + off[None, :],
                (m[:, None] < N) & (off[None, :] < H * D),
                0,
            ).to(tl.float32)
        w = tl.load(
            W + c[:, None] * (H * D) + off[None, :],
            (c[:, None] < 2 * H) & (off[None, :] < H * D),
            0,
        )
        acc += tl.dot(w, tl.trans(x), input_precision="ieee")
        sq += tl.sum(x * x, 1)
    _write_outputs(
        X,
        S,
        B,
        Y,
        P,
        tl.trans(acc),
        sq,
        m,
        c,
        N,
        H,
        D,
        BM,
        BD,
        SCALE_STRIDE,
        BASE_STRIDE,
        MAG,
        EPS,
        NORM,
        TOKEN_STRIDE,
        CHANNEL_STRIDE,
    )


@triton.jit(**_weight_options)
def _pack_weights(W, V, K: tl.constexpr, C: tl.constexpr, BK: tl.constexpr):
    # Three limbs retain the FP32 significand. Scaled residuals extend the
    # range of the small FP16 limbs. A flag selects IEEE projection when
    # a weight cannot be recovered exactly in the input dtype.
    block = tl.program_id(0)
    off = block * BK + tl.arange(0, BK)
    row = tl.arange(0, C)
    idx = row[:, None] * K + off[None, :]
    w = tl.load(W + idx)
    if V.dtype.element_ty == tl.float16:
        hi = w.to(tl.float16)
        rest = (w - hi.to(tl.float32)) * 4096.0
        mid = rest.to(tl.float16)
        tail = ((rest - mid.to(tl.float32)) * 4096.0).to(tl.float16)
        recovered = (hi.to(tl.float32) + mid.to(tl.float32) * 0.000244140625) + tail.to(
            tl.float32
        ) * 5.960464477539063e-8
    else:
        hi = (
            (w.to(tl.uint32, bitcast=True) & 0xFFFF0000)
            .to(tl.float32, bitcast=True)
            .to(tl.bfloat16)
        )
        rest = w - hi.to(tl.float32)
        mid = (
            (rest.to(tl.uint32, bitcast=True) & 0xFFFF0000)
            .to(tl.float32, bitcast=True)
            .to(tl.bfloat16)
        )
        tail = (rest - mid.to(tl.float32)).to(tl.bfloat16)
        recovered = (hi.to(tl.float32) + mid.to(tl.float32)) + tail.to(tl.float32)
    tl.store(V + idx, hi)
    tl.store(V + idx + C * K, mid)
    tl.store(V + idx + 2 * C * K, tail)
    tl.store(V + idx + 3 * C * K, tl.full((C, BK), 0, V.dtype.element_ty))
    loss = tl.max(tl.max((recovered != w).to(tl.int32), axis=1), axis=0)
    tl.store(V + 4 * C * K + block, loss)


@triton.jit
def _project_fp32(
    X,
    W,
    m,
    c,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    # Sum FP32 products in small K tiles, including weights outside the
    # representable range of the packed format.
    k = tl.arange(0, BK)
    sums = ()
    for j in tl.static_range(2 * H):
        sums += (tl.full((BM,), 0, tl.float32),)
    sq = tl.full((BM,), 0, tl.float32)
    for i in range(H * D // BK):
        off = i * BK + k
        xx = tl.load(X + m[:, None] * (H * D) + off[None, :], m[:, None] < N, 0).to(
            tl.float32
        )
        sq += tl.sum(xx * xx, 1)
        acc = ()
        for j in tl.static_range(2 * H):
            ww = tl.load(W + j * (H * D) + off)
            acc += (sums[j] + tl.sum(xx * ww[None, :], 1),)
        sums = acc
    proj = tl.full((BM, 2 * H), 0, tl.float32)
    for j in tl.static_range(2 * H):
        proj = tl.where(c[None, :] == j, sums[j][:, None], proj)
    return proj, sq


@triton.jit(**_pointer_options)
def _project_large(
    X,
    W,
    S,
    B,
    Y,
    P,
    V,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    SS: tl.constexpr,
    BS: tl.constexpr,
    MAG: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    CHANNEL_STRIDE: tl.constexpr,
    PACK_WIDTH: tl.constexpr,
):
    K: tl.constexpr = H * D
    C: tl.constexpr = 2 * H
    flags = tl.arange(0, triton.next_power_of_2(K // PACK_WIDTH))
    bad = tl.max(
        tl.load(V + 4 * C * K + flags, flags < K // PACK_WIDTH, 0).to(tl.int32), 0
    )
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.arange(0, C)
    if bad:
        projection, sq = _project_fp32(X, W, m, c, N, H, D, BM, 256)
    else:
        rows = tl.arange(0, 4 * C)
        k = tl.arange(0, BK)
        parts = tl.full((4 * C, BM), 0, tl.float32)
        gram = tl.full((BM, BM), 0, tl.float32)
        for i in range(K // BK):
            off = i * BK + k
            x = tl.load(X + m[:, None] * K + off[None, :], m[:, None] < N, 0)
            w = tl.load(V + rows[:, None] * K + off[None, :])
            parts = tl.dot(w, tl.trans(x), acc=parts)
            gram = tl.dot(x, tl.trans(x), acc=gram)
        local = tl.arange(0, BM)
        sq = tl.sum(tl.where(local[:, None] == local[None, :], gram, 0.0), 1)
        t = tl.trans(parts)
        idx = tl.broadcast_to(c[None, :], (BM, C))
        head = tl.gather(t, idx, 1)
        middle = tl.gather(t, idx + C, 1)
        tail = tl.gather(t, idx + 2 * C, 1)
        if X.dtype.element_ty == tl.float16:
            projection = (head + middle * 0.000244140625) + tail * 5.960464477539063e-8
        else:
            projection = (head + middle) + tail
    _write_outputs(
        X,
        S,
        B,
        Y,
        P,
        projection,
        sq,
        m,
        c,
        N,
        H,
        D,
        BM,
        BD,
        SS,
        BS,
        MAG,
        EPS,
        NORM,
        0,
        0,
    )


def _address_type(shape):
    # The Ascend launcher accepts a device address. Its profiler reads size()
    # to describe the argument even when profiling starts after compilation.
    class Pointer(int):
        __slots__ = ()

        def size(self):
            return shape

    return Pointer


# Only successful, fully validated calls populate this cache.
_tensor_types = (torch.Tensor, torch.nn.Parameter)
_launches = {}
_last_plan = None


def _key(x, magnitude, hc_eps, norm_eps):
    return x.shape, x.dtype, struct.pack("ddd", magnitude, hc_eps, norm_eps)


def _bind(kernel, grid, pointers, parameters):
    compiled = kernel.warmup(
        *pointers, *parameters, grid=grid, num_warps=4, enable_fp_fusion=False
    )
    runner = compiled[grid]
    backend = compiled.run
    runtime = knobs.runtime
    hook_type = knobs.HookChain
    device = pointers[0].device.index
    function, metadata = compiled.function, compiled.packed_metadata
    instrumented = (
        getattr(compiled.metadata, "debug_enabled", False)
        or getattr(backend, "enable_msprof_register_tensor", False)
        or getattr(backend, "compile_only", False)
    )

    x_arg, w_arg, s_arg, b_arg, y_arg, p_arg = (
        _address_type(tensor.shape) for tensor in pointers
    )
    blocks = grid[0]
    vendor_launch = backend.launch if type(backend) is NPULauncher else None

    def launch(x, weight, scale, base, y, post):
        stream = _npu_getCurrentRawStreamNoWait(device)
        enter, leave = runtime.launch_enter_hook, runtime.launch_exit_hook
        if (
            vendor_launch is not None
            and not instrumented
            and type(enter) is hook_type
            and not enter.calls
            and type(leave) is hook_type
            and not leave.calls
        ):
            # NPULauncher forwards to this kernel-specific entry and publishes
            # its profiler flag. Hooks and instrumentation use the full runner.
            registered = vendor_launch(
                blocks,
                1,
                1,
                stream,
                function,
                metadata,
                None,
                None,
                None,
                x_arg(x.data_ptr()),
                w_arg(weight.data_ptr()),
                s_arg(scale.data_ptr()),
                b_arg(base.data_ptr()),
                y_arg(y.data_ptr()),
                p_arg(post.data_ptr()),
                *parameters,
            )
            backend_utils.TRITON_PROFILER_REGISTERED = registered == 1
        else:
            runner(x, weight, scale, base, y, post, *parameters, stream=stream)

    return launch


def _bind_large(grid, pointers, parameters):
    # Cache compiler handles only. All three allocations belong to this call.
    _, h, d = parameters[:3]
    pack_width = 512
    k = h * d
    pack_grid = (k // pack_width, 1, 1)
    # Four groups of (2 * h) rows, then one validity flag per pack block.
    size = 8 * h * k + pack_grid[0]
    x, weight = pointers[:2]
    scratch = x.new(size)
    args = (k, 2 * h, pack_width)
    opts = dict(num_warps=4, enable_fp_fusion=False)
    pack = _pack_weights.warmup(weight, scratch, *args, grid=pack_grid, **opts)[
        pack_grid
    ]
    project = _project_large.warmup(
        *pointers, scratch, *parameters, pack_width, grid=grid, **opts
    )[grid]

    def launch(x, weight, scale, base, y, post):
        tmp = x.new(size)
        pack(weight, tmp, *args)
        project(x, weight, scale, base, y, post, tmp, *parameters, pack_width)

    return launch


def _launch(x, weight, scale, base, magnitude, hc_eps, norm_eps, pipeline):
    n, h, d = x.shape
    k = h * d
    if (
        not n
        or h > 4
        or x.numel() > 2**31 - 1
        or (k > 1024 and (h != 4 or d % 1024 or d > 8192))
    ):
        return _launch_ihc_pre(
            x, weight, scale, base, magnitude, hc_eps, norm_eps, pipeline
        )
    token_stride = channel_stride = 0
    if not x.is_contiguous():
        sn, sc, sd = x.stride()
        if k > 1024 and sd == 2 and sc > 0 and n * sn + h * sc + 2 * d < 2**31:
            token_stride, channel_stride = sn, sc
        else:
            x = x.contiguous()
    y = x.new_empty((n, d))
    post = weight.new_empty((n, h))
    pointers = (x, weight, scale, base, y, post)
    if k <= 1024:
        bm, bk, bd = 1, triton.next_power_of_2(k), triton.next_power_of_2(d)
        kernel = _short_projection
    elif n <= 8 and h == 4 and d >= 4096:
        bd = 1024
        bk = 2048 if d % 2048 == 0 else 1024
        kernel = _decode_projection
    else:
        if channel_stride:
            bm, bk, bd = 8, (2048 if d % 2048 == 0 else 1024), 512
        elif n <= 64:
            bm, bk, bd = (4 if n <= 8 else 8), (2048 if d % 2048 == 0 else 1024), 1024
        else:
            bm, bk, bd = 16, 1024, (1024 if n >= 512 else 512)
        kernel = _project_and_gate
    if kernel is _decode_projection:
        parameters = (n, h, d, bk, bd)
        blocks = n * (d // bd)
    else:
        parameters = (n, h, d, bm, bk, bd)
        blocks = triton.cdiv(n, bm)
    parameters += (scale.stride(0), base.stride(0), magnitude, hc_eps, norm_eps)
    if k > 1024:
        parameters += (token_stride, channel_stride)
    grid = (blocks, 1, 1)
    if (
        _use_bound_launch
        and all(type(t) in _tensor_types for t in pointers)
        and not torch._C._is_torch_function_mode_enabled()
        and not torch._C._len_torch_dispatch_stack()
    ):
        if k > 1024 and n >= 1024 and not channel_stride:
            run = _bind_large(grid, pointers, parameters)
        else:
            run = _bind(kernel, grid, pointers, parameters)
        run(*pointers)
        try:
            guard = TensorGuards(
                x,
                weight,
                scale,
                base,
                dynamic_dims_sizes=None,
                dynamic_dims_strides=None,
            ).check
        except (AttributeError, TypeError):
            return y, post
        device = x.device.index

        def guarded(x, weight, scale, base):
            if not guard(x, weight, scale, base) or _npu_getDevice() != device:
                return None
            y, post = x.new(n, d), weight.new(n, h)
            run(x, weight, scale, base, y, post)
            return y, post

        if len(_launches) >= 256:
            _launches.clear()
        key = _key(x, magnitude, hc_eps, norm_eps)
        _launches[key] = guarded
        global _last_plan
        _last_plan = key[2], guarded
    else:
        kernel[grid](*pointers, *parameters, num_warps=4, enable_fp_fusion=False)
    return y, post


def triton_ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce iHC channels using FP32 projections and gates."""
    logger.debug("GEMS_ASCEND TRITON_IHC_PRE")
    global _last_plan
    last = _last_plan
    if (
        last is not None
        and type(x) in _tensor_types
        and type(magnitude) is float
        and type(hc_eps) is float
        and type(norm_eps) is float
        and not torch._C._is_torch_function_mode_enabled()
        and not torch._C._len_torch_dispatch_stack()
    ):
        scalars = struct.pack("ddd", magnitude, hc_eps, norm_eps)
        if scalars == last[0]:
            outputs = last[1](x, weight, scale, base)
            if outputs is not None:
                return outputs
        run = _launches.get((x.shape, x.dtype, scalars))
        if run is not None and run is not last[1]:
            outputs = run(x, weight, scale, base)
            if outputs is not None:
                _last_plan = scalars, run
                return outputs
    return _ihc_pre(
        x,
        weight,
        scale,
        base,
        magnitude,
        hc_eps,
        norm_eps,
        _pipeline_stages=1,
        _launcher=_launch,
    )


def ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility entry for :func:`triton_ihc_pre`."""
    logger.debug("GEMS_ASCEND IHC_PRE")
    return triton_ihc_pre(x, weight, scale, base, magnitude, hc_eps, norm_eps)
