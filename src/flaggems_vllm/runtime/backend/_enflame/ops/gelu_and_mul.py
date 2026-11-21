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

import logging
import os

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import pointwise_dynamic, tl_extra_shim
from flaggems_vllm.utils.codegen_config_utils import CodeGenConfig, get_codegen_config

erf = tl_extra_shim.erf
pow = tl_extra_shim.pow
tanh = tl_extra_shim.tanh
logger = logging.getLogger(__name__)

# torch_gcu's ATen gelu silently swaps approximate="none" (erf) for the tanh
# approximation unless ENFLAME_PT_GELU_ERF_ENABLE is set, which would make any
# torch-side reference on GCU disagree with the erf kernels below. Turn the erf
# path on so torch.nn.functional.gelu keeps the requested approximate="none"
# semantics; an explicit user setting always wins (setdefault).
os.environ.setdefault("ENFLAME_PT_GELU_ERF_ENABLE", "true")

# The default (CUDA-oriented) codegen config caps grid.x at 65536, one above
# the GCU hardware limit (65535). The generated pointwise wrapper clamps
# num_ctas with max_grid_size[0] and lets each CTA iterate several tiles once
# the task space exceeds the cap, so clamping here keeps large core shapes
# launchable without changing small-shape behavior.
#
# max_tile_size is raised from the default 512 (measured on GCU400: tile 512
# tops out around 0.08x-0.3x of torch on core shapes; tile 4096 reaches
# 0.44x-0.69x on the large shapes and keeps ~1.8x on tiny ones). The grad
# kernels keep one more live buffer than the forward kernels (3 inputs + 2
# outputs), so their tile stays at 2048 to hold the same DSM budget
# (5 * 2048 * 4B = 40KB <= forward's validated 3 * 4096 * 4B = 48KB).
_codegen_base = get_codegen_config()
GCU_GRID_LIMITS = (65535, 255, 255)
GCU_FORWARD_CONFIG = CodeGenConfig(
    max_tile_size=4096,
    max_grid_size=GCU_GRID_LIMITS,
    max_num_warps_per_cta=_codegen_base.max_num_warps_per_cta,
    prefer_block_pointer=_codegen_base.prefer_block_pointer,
    prefer_1d_tile=_codegen_base.prefer_1d_tile,
)
GCU_GRAD_CONFIG = CodeGenConfig(
    max_tile_size=2048,
    max_grid_size=GCU_GRID_LIMITS,
    max_num_warps_per_cta=_codegen_base.max_num_warps_per_cta,
    prefer_block_pointer=_codegen_base.prefer_block_pointer,
    prefer_1d_tile=_codegen_base.prefer_1d_tile,
)

# ===========================================================================
# Flat streaming fast path for contiguous inputs
# ===========================================================================
# Measured on GCU400 (1G-element core shapes, do_bench kernel mode): the cost
# ladder of the previous deg-8 kernel at B=16384/W1 was
#   fp16: load+store 2.41ms -> +casts 0.08 -> +Horner(8 FMA) 2.42 -> +sigmoid
#         0.50 -> 5.41ms, i.e. the exposed fp32 chain dominates the 16-bit
#         kernel (memory floor is only 2.2-2.4ms); fp32 tensors cost barely
#         more (5.75ms) despite 2x traffic because their chain is shorter.
# This round therefore specializes degree and launch config per output dtype:
#
#   * 16-bit outputs (fp16/bf16): degree-6 fit, no s-clamp, B=65536/W1 for
#     large n. The fit error (2.4e-5 in gelu(x)) is ~2 orders below the fp16
#     ulp, so outputs keep the same 1-ulp behavior as deg-8 (verified
#     tol-aware vs the upcast reference on 32M randn samples: 0 violations).
#     The clamp is dropped because the degree-6 leading coefficient is
#     positive: for s > 32 the polynomial extrapolates monotonically to
#     +inf, so sigmoid saturates to exactly Phi's 0/1 tails. Verified at
#     x = +-inf, +-65504 (fp16), +-3e38 (bf16/fp32), |x| = 8: all match the
#     torch reference (gelu(-inf) = nan via -inf * 0, same as torch).
#   * 32-bit outputs: degree-7 fit WITH the s=min(x*x, 32) clamp, B=16384/W2.
#     Degree-7's leading coefficient is negative, so without the clamp the
#     extrapolation flips sign around |x| ~ 7.3 and returns 0 for large
#     positive x (measured: gelu(8.0) -> 0.0) - the clamp is mandatory.
#     Degree-6 is vetoed here: its 2.4e-5 fit error gets amplified by |y| in
#     the product and exceeds the 1e-4 atol on randn inputs (1.14e-4). The
#     degree-7 error stays at 2.8e-5 (3.5x margin). W2 hides the chain
#     behind the loads (4.58ms vs the 4.51ms load+store floor at 1G).
#   * sigmoid is already optimal: an A/B against a manual 1/(1+exp(-t))
#     lowered to the same latency (4.2236 vs 4.2244ms), and exp2-based
#     forms are known-slower on GCU (perf_playbook).
#   * small/mid sizes (n <= 2M): B=2048/W2 with grid=min(48, cdiv(n, B))
#     fills the SPs progressively (one tile per program while the grid is
#     below 48) and beats the large-config single-wave tiles up to ~2M
#     elements (12.3us vs 13.7us at 1M); above that the wide persistent
#     tiles win again (30.6us vs 37.9us at 4M).
#
# Headline numbers (1G elements, vs the previous deg-8 B=16384/W1 config):
#   fp16 5.40 -> 3.97ms, bf16 5.41 -> ~3.97ms, fp32 5.75 -> 4.58ms;
#   (4096,4096) fp16 98 -> ~79us, fp32 104 -> ~80us; (64,64) 3.8 -> 3.7us.
_FLAT_TILE_SMALL = 2048
_FLAT_SMALL_MAX = 2**21  # below this many elements: fill-the-grid small tiles
_FLAT_GRID_CAP = 48  # GCU400 SP count; persistent grid-stride beyond that
_FLAT_TILE_LARGE_16BIT = 65536
_FLAT_TILE_LARGE_32BIT = 16384

# Degree-6 Chebyshev least-squares fit of P(s) = 2*atanh(erf(sqrt(s/2)))/sqrt(s)
# over s in [0, 32] (gelu(x) = x * sigmoid(x * P(x^2))); 16-bit path, no clamp.
_E0 = tl.constexpr(1.5953784901812362)
_E1 = tl.constexpr(0.07338409181814284)
_E2 = tl.constexpr(-0.0003813880148825668)
_E3 = tl.constexpr(-5.428118007938764e-05)
_E4 = tl.constexpr(3.1787987321044684e-06)
_E5 = tl.constexpr(-7.568447796921953e-08)
_E6 = tl.constexpr(6.8404153739139e-10)

# Degree-7 fit of the same target; 32-bit path, clamped at s = 32 (the
# negative leading coefficient makes unclamped extrapolation unsafe).
_D0 = tl.constexpr(1.5956858496775363)
_D1 = tl.constexpr(0.07284617639058376)
_D2 = tl.constexpr(-0.00015445021611273802)
_D3 = tl.constexpr(-9.368046462698117e-05)
_D4 = tl.constexpr(6.564689983953184e-06)
_D5 = tl.constexpr(-2.280499144167936e-07)
_D6 = tl.constexpr(4.1228476561246845e-09)
_D7 = tl.constexpr(-3.0703626060118777e-11)


@triton.jit
def _gelu_none_mul_flat16(
    x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    # erf-gelu for 16-bit outputs: degree-6 polynomial, no clamp (positive
    # leading coefficient keeps the extrapolation monotone, see above).
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for start in range(pid * BLOCK, n, nprog * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(x_ptr + offs)
        else:
            x = tl.load(x_ptr + offs, mask=offs < n)
        xf = x.to(tl.float32)
        sq = xf * xf
        g = _E6
        g = _E5 + sq * g
        g = _E4 + sq * g
        g = _E3 + sq * g
        g = _E2 + sq * g
        g = _E1 + sq * g
        g = _E0 + sq * g
        gelu = xf * tl.sigmoid(xf * g)
        if EVEN:
            y = tl.load(y_ptr + offs)
            tl.store(o_ptr + offs, (gelu * y.to(tl.float32)).to(o_ptr.dtype.element_ty))
        else:
            m = offs < n
            y = tl.load(y_ptr + offs, mask=m)
            tl.store(
                o_ptr + offs,
                (gelu * y.to(tl.float32)).to(o_ptr.dtype.element_ty),
                mask=m,
            )


@triton.jit
def _gelu_none_mul_flat32(
    x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    # erf-gelu for 32-bit outputs: degree-7 polynomial, clamped at s = 32.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for start in range(pid * BLOCK, n, nprog * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(x_ptr + offs)
        else:
            x = tl.load(x_ptr + offs, mask=offs < n)
        xf = x.to(tl.float32)
        sq = tl.minimum(xf * xf, 32.0)
        g = _D7
        g = _D6 + sq * g
        g = _D5 + sq * g
        g = _D4 + sq * g
        g = _D3 + sq * g
        g = _D2 + sq * g
        g = _D1 + sq * g
        g = _D0 + sq * g
        gelu = xf * tl.sigmoid(xf * g)
        if EVEN:
            y = tl.load(y_ptr + offs)
            tl.store(o_ptr + offs, (gelu * y.to(tl.float32)).to(o_ptr.dtype.element_ty))
        else:
            m = offs < n
            y = tl.load(y_ptr + offs, mask=m)
            tl.store(
                o_ptr + offs,
                (gelu * y.to(tl.float32)).to(o_ptr.dtype.element_ty),
                mask=m,
            )


@triton.jit
def _gelu_tanh_mul_flat(
    x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for start in range(pid * BLOCK, n, nprog * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(x_ptr + offs)
        else:
            x = tl.load(x_ptr + offs, mask=offs < n)
        xf = x.to(tl.float32)
        # 0.5 * x * (1 + tanh(u)) == x * sigmoid(2 * u)
        u = xf * 0.79788456 * (1.0 + 0.044715 * (xf * xf))
        gelu = xf * tl.sigmoid(2.0 * u)
        if EVEN:
            y = tl.load(y_ptr + offs)
            tl.store(o_ptr + offs, (gelu * y.to(tl.float32)).to(o_ptr.dtype.element_ty))
        else:
            m = offs < n
            y = tl.load(y_ptr + offs, mask=m)
            tl.store(
                o_ptr + offs,
                (gelu * y.to(tl.float32)).to(o_ptr.dtype.element_ty),
                mask=m,
            )


def _prev_pow2(value):
    return 1 << (value.bit_length() - 1)


def _flat_config(n, wide):
    # wide=True -> 32-bit output elements. Small/mid sizes fill the grid with
    # narrow tiles (one tile per program while cdiv(n, B) < 48); large sizes
    # switch to wide persistent tiles, sized so ~48 programs cover the tensor.
    if n <= _FLAT_SMALL_MAX:
        block, warps = _FLAT_TILE_SMALL, 2
    elif wide:
        block = min(_FLAT_TILE_LARGE_32BIT, _prev_pow2(max(n // _FLAT_GRID_CAP, 1)))
        warps = 2
    else:
        block = min(_FLAT_TILE_LARGE_16BIT, _prev_pow2(max(n // _FLAT_GRID_CAP, 1)))
        warps = 1
    grid = min(_FLAT_GRID_CAP, triton.cdiv(n, block))
    return block, warps, grid, (n % block) == 0


def _use_flat_path(x, y):
    # int32 element offsets in the flat kernels: cap at 2^31 elements
    return (
        x.numel() < 2**31
        and x.shape == y.shape
        and x.is_contiguous()
        and y.is_contiguous()
    )


def _gelu_mul_flat(x, y, approximate):
    n = x.numel()
    out_dtype = torch.promote_types(x.dtype, y.dtype)
    out = torch.empty_like(x, dtype=out_dtype)
    if n == 0:
        return out
    wide = out_dtype not in (torch.float16, torch.bfloat16)
    if approximate == "tanh":
        kernel = _gelu_tanh_mul_flat
    elif wide:
        kernel = _gelu_none_mul_flat32
    else:
        kernel = _gelu_none_mul_flat16
    block, warps, nprog, even = _flat_config(n, wide)
    kernel[(nprog,)](x, y, out, n, BLOCK=block, EVEN=even, num_warps=warps)
    return out


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=GCU_FORWARD_CONFIG)
@triton.jit
def gelu_none_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    RCP_SQRT_2: tl.constexpr = 0.7071067811
    x_gelu = 0.5 * x_fp32 * (1 + erf(x_fp32 * RCP_SQRT_2))
    return x_gelu * y


@pointwise_dynamic(
    promotion_methods=[(0, 1, 2, "DEFAULT"), (0, 1, 2, "DEFAULT")],
    num_outputs=2,
    config=GCU_GRAD_CONFIG,
)
@triton.jit
def gelu_none_and_mul_grad_kernel(x, y, dgrad):
    RCP_SQRT_2: tl.constexpr = 0.7071067811
    COEFF: tl.constexpr = 0.7978845608028654

    x_fp32 = x.to(tl.float32)
    x_gelu = 0.5 * x_fp32 * (1 + erf(x_fp32 * RCP_SQRT_2))

    d_gelu = dgrad * y
    dx = (
        d_gelu
        * 0.5
        * (
            1.0
            + erf(x_fp32 * RCP_SQRT_2)
            + x_fp32 * COEFF * tl.exp(-0.5 * x_fp32 * x_fp32)
        )
    )

    dy = dgrad * x_gelu

    return dx, dy


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=GCU_FORWARD_CONFIG)
@triton.jit
def gelu_tanh_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_gelu = (
        0.5
        * x_fp32
        * (
            1
            + tanh(x_fp32 * 0.79788456 * (1 + 0.044715 * pow(x_fp32.to(tl.float32), 2)))
        )
    )
    return x_gelu * y


@pointwise_dynamic(
    promotion_methods=[(0, 1, 2, "DEFAULT"), (0, 1, 2, "DEFAULT")],
    num_outputs=2,
    config=GCU_GRAD_CONFIG,
)
@triton.jit
def gelu_tanh_and_mul_grad_kernel(x, y, dgrad):
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)

    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = x_fp32 * x_fp32 * x_fp32
    tanh_arg = sqrt_2_over_pi * (x_fp32 + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * x_fp32 * (1 + tanh_result)
    dy = geglu_a * dgrad

    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = (
        0.5
        * x_fp32
        * (1 - tanh_sq)
        * (sqrt_2_over_pi * (1 + 3 * 0.044715 * x_fp32 * x_fp32))
    )
    dx = dgrad * y_fp32 * (term1 + term2)

    return dx, dy


class GeluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, approximate="none"):
        logger.debug("GEMS GELU AND MUL FORWARD")
        ctx.save_for_backward(x, y)
        ctx.approximate = approximate
        if approximate == "none":
            if _use_flat_path(x, y):
                return _gelu_mul_flat(x, y, "none")
            return gelu_none_and_mul_kernel(x, y)
        elif approximate == "tanh":
            if _use_flat_path(x, y):
                return _gelu_mul_flat(x, y, "tanh")
            return gelu_tanh_and_mul_kernel(x, y)
        else:
            raise ValueError(f"Invalid approximate value: {approximate}")

    @staticmethod
    def backward(ctx, dgrad):
        logger.debug("GEMS GELU AND MUL BACKWARD")
        x, y = ctx.saved_tensors
        if ctx.approximate == "none":
            dx, dy = gelu_none_and_mul_grad_kernel(x, y, dgrad)
        else:
            dx, dy = gelu_tanh_and_mul_grad_kernel(x, y, dgrad)
        return dx, dy, None


def gelu_and_mul(x, y, approximate="none"):
    return GeluAndMul.apply(x, y, approximate)
