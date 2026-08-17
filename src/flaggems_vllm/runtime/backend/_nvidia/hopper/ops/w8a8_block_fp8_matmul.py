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

import functools
import logging
import os
from typing import Any, Dict, List, Optional

import torch
import triton
import triton.language as tl
import yaml

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

logger = logging.getLogger(__name__)
CACHE_USAGE_THRESHOLD = 0.8

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as _tle_probe

        HAS_TLE_2C = hasattr(_tle_probe.gpu, "alloc_barriers")
    except ImportError:
        HAS_TLE_2C = False
else:
    HAS_TLE_2C = False
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "w8a8_block_fp8_matmul_hopper_expand.yaml",
    )
)

_HAS_DEVICE_TMA = hasattr(tl, "make_tensor_descriptor")

_FP8_DTYPES = tuple(
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if hasattr(torch, name)
)

# Explicit-TLE 2-consumer GEMM tile configs (H20-3e PoC optima). n128 is the
# default; n64 halves each consumer's N width to raise occupancy on shapes with
# num_tiles << NUM_SMS. group_n stays 128 in both, so b_s is scalar per K-tile.
TLE2C_BLOCK_M = 64
TLE2C_BLOCK_N = 256
TLE2C_BLOCK_N_HALF = 128
TLE2C_NUM_BUFS = 4
TLE2C_C_REGS = 240
N64_BLOCK_M = 64
N64_BLOCK_N = 128
N64_BLOCK_N_HALF = 64
N64_NUM_BUFS = 4
N64_C_REGS = 240


def _can_use_tle_2c(a, b, M, N, K, group_n, group_k) -> bool:
    if not (HAS_TLE_2C and _HAS_DEVICE_TMA):
        return False
    if a.dtype not in _FP8_DTYPES or b.dtype not in _FP8_DTYPES:
        return False
    if 0 in a.stride() or 0 in b.stride():
        return False
    if group_k != 128 or K % group_k != 0:
        return False
    if N % TLE2C_BLOCK_N_HALF != 0:
        return False
    if group_n != TLE2C_BLOCK_N_HALF:
        return False
    return True


# The explicit-TLE machinery imports triton.experimental.tle, which is absent on
# many Triton builds; define it only behind a positive capability check.
if HAS_TLE_2C and _HAS_DEVICE_TMA:
    import triton.language.core as tlc
    import triton.experimental.tle.language as tle
    from triton.experimental.tle.language.gpu import types as tle_types
    from triton.tools.tensor_descriptor import TensorDescriptor

    def _torch_dtype_to_tl(dt: torch.dtype):
        if dt == torch.bfloat16:
            return tl.bfloat16
        if dt == torch.float16:
            return tl.float16
        return tl.float32

    @tlc.builtin
    def tle_subslice(buf, offsets, shape, _semantic=None):
        """Row/col subslice VIEW of a shared buffered_tensor, preserving the
        parent swizzle so a wgmma reading it sees the same layout. Safe as a
        wgmma operand and tl.store dest; NOT safe as a TMA-copy dest."""
        offsets = [int(tlc._unwrap_if_constexpr(o)) for o in offsets]
        shape = [int(tlc._unwrap_if_constexpr(s)) for s in shape]
        result_ty = tle_types.buffered_tensor_type(
            buf.dtype, shape, buf.type.storage, buf.type.layout, _semantic,
            alloc_shape=buf.type.alloc_shape,
        )
        handle = _semantic.builder.create_memdesc_subslice(
            result_ty.to_ir(_semantic.builder), buf.handle, offsets
        )
        return tle_types.buffered_tensor(
            handle, buf.dtype, shape, buf.type.storage, buf.type.layout, _semantic,
            alloc_shape=buf.type.alloc_shape,
        )

    @triton.jit
    def _producer(desc_a, desc_b, a_buf, b_buf, ab_empty, a_full, b_full, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, NUM_BUFS: tl.constexpr,
                  NUM_SMS: tl.constexpr):
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_tiles = tl.cdiv(M, BLOCK_M) * num_pid_n
        k_tiles = tl.cdiv(K, BLOCK_K)
        ctr = 0  # continuous ring counter across all tiles this CTA owns
        for tile in range(pid, num_tiles, NUM_SMS):
            pid_m = tile // num_pid_n
            pid_n = tile % num_pid_n
            for k in range(k_tiles):
                buf = ctr % NUM_BUFS
                cycle = ctr // NUM_BUFS
                tle.gpu.barrier_wait(ab_empty[buf], phaseIdx=cycle)
                # one barrier per TMA transfer (a single barrier can't cover two)
                tle.gpu.copy(desc_a, a_buf.slot(buf), [BLOCK_M, BLOCK_K],
                             [pid_m * BLOCK_M, k * BLOCK_K], barrier=a_full[buf])
                tle.gpu.copy(desc_b, b_buf.slot(buf), [BLOCK_N, BLOCK_K],
                             [pid_n * BLOCK_N, k * BLOCK_K], barrier=b_full[buf])
                ctr += 1

    @triton.jit
    def _consumer(c_ptr, As, Bs, M, N, K,
                  stride_cm, stride_cn, stride_As_m, stride_As_k,
                  stride_Bs_n, stride_Bs_k,
                  a_buf, b_buf, ab_empty, a_full, b_full,
                  OUT_DTYPE: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, BLOCK_N_HALF: tl.constexpr,
                  GROUP_N: tl.constexpr, NUM_BUFS: tl.constexpr,
                  NUM_SMS: tl.constexpr, HALF: tl.constexpr,
                  PARTITION_ID: tl.constexpr):
        """One warpgroup computes the HALF-th N-slice of each owned tile. A
        BLOCK_N_HALF-wide slice lies inside one GROUP_N scale group, so b_s is a
        scalar per K-tile."""
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_tiles = tl.cdiv(M, BLOCK_M) * num_pid_n
        k_tiles = tl.cdiv(K, BLOCK_K)

        ctr = 0
        for tile in range(pid, num_tiles, NUM_SMS):
            pid_m = tile // num_pid_n
            pid_n = tile % num_pid_n
            n_start = pid_n * BLOCK_N + HALF * BLOCK_N_HALF
            offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            # Ragged M: rows past M are TMA-zeroed in the A tile, but As has only
            # M rows -> mask the per-row scale load. half_valid drops a wholly-OOB
            # N half when N tiles unevenly into BLOCK_N.
            m_mask = offs_am < M
            As_ptrs = As + offs_am * stride_As_m
            half_valid = n_start < N
            Bs_scalar_ptr = Bs + (n_start // GROUP_N) * stride_Bs_n

            acc = tl.zeros([BLOCK_M, BLOCK_N_HALF], tl.float32)
            for k in range(k_tiles):
                buf = ctr % NUM_BUFS
                cycle = ctr // NUM_BUFS
                tle.gpu.barrier_wait(a_full[buf], phaseIdx=cycle)
                tle.gpu.barrier_wait(b_full[buf], phaseIdx=cycle)
                b_half = tle_subslice(b_buf.slot(buf), [HALF * BLOCK_N_HALF, 0],
                                      [BLOCK_N_HALF, BLOCK_K])
                prod = tle.gpu.wgmma(a_buf.slot(buf), b_half,
                                     out_dtype=tl.float32, trans_b=True)
                a_s = tl.load(As_ptrs + k * stride_As_k, mask=m_mask, other=0.0)
                b_s = tl.load(Bs_scalar_ptr + k * stride_Bs_k, mask=half_valid,
                              other=0.0)
                prod = tle.gpu.wgmma_wait(0, prod)
                tle.gpu.barrier_arrive(ab_empty[buf], phaseIdx=cycle)
                acc += prod * (a_s * b_s)[:, None]
                ctr += 1

            offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_cn = n_start + tl.arange(0, BLOCK_N_HALF)
            c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, acc.to(OUT_DTYPE), mask=c_mask)

    @triton.jit
    def _gemm_tle_kernel(desc_a, desc_b, c_ptr, As, Bs, M, N, K,
                         stride_cm, stride_cn, stride_As_m, stride_As_k,
                         stride_Bs_n, stride_Bs_k,
                         OUT_DTYPE: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr, BLOCK_N_HALF: tl.constexpr,
                         GROUP_N: tl.constexpr, NUM_BUFS: tl.constexpr,
                         NUM_SMS: tl.constexpr, C_REGS: tl.constexpr):
        a_buf = tle.gpu.alloc([NUM_BUFS, BLOCK_M, BLOCK_K], dtype=tl.float8e4nv,
                              scope=tle.gpu.smem)
        b_buf = tle.gpu.alloc([NUM_BUFS, BLOCK_N, BLOCK_K], dtype=tl.float8e4nv,
                              scope=tle.gpu.smem)
        # fill barriers: each TMA arrives once. empty barrier: BOTH consumers
        # release the slot (arrive_count=2) before the producer may refill.
        a_full = tle.gpu.alloc_barriers(
            num_barriers=NUM_BUFS, arrive_count=1, expect_bytes=BLOCK_M * BLOCK_K)
        b_full = tle.gpu.alloc_barriers(
            num_barriers=NUM_BUFS, arrive_count=1, expect_bytes=BLOCK_N * BLOCK_K)
        ab_empty = tle.gpu.alloc_barriers(
            num_barriers=NUM_BUFS, arrive_count=2, init=tle.gpu.READY)

        tle.gpu.warp_specialize(
            [
                (_producer, (desc_a, desc_b, a_buf, b_buf, ab_empty, a_full,
                             b_full, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                             NUM_BUFS, NUM_SMS)),
                (_consumer, (c_ptr, As, Bs, M, N, K, stride_cm, stride_cn,
                             stride_As_m, stride_As_k, stride_Bs_n, stride_Bs_k,
                             a_buf, b_buf, ab_empty, a_full, b_full, OUT_DTYPE,
                             BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_N_HALF, GROUP_N,
                             NUM_BUFS, NUM_SMS, 0, 1)),
                (_consumer, (c_ptr, As, Bs, M, N, K, stride_cm, stride_cn,
                             stride_As_m, stride_As_k, stride_Bs_n, stride_Bs_k,
                             a_buf, b_buf, ab_empty, a_full, b_full, OUT_DTYPE,
                             BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_N_HALF, GROUP_N,
                             NUM_BUFS, NUM_SMS, 1, 2)),
            ],
            [4, 4],            # two consumers, 4 warps each
            [C_REGS, C_REGS],  # per-consumer reg budget
        )

    def w8a8_block_fp8_matmul_tle(a, b, c, a_s, b_s, M, N, K, group_n, group_k,
                                  BLOCK_M, BLOCK_N, BLOCK_N_HALF, NUM_BUFS,
                                  C_REGS):
        """Explicit 2-consumer TLE launcher; numerics match the general kernel.
        Caller (_can_use_tle_2c) guarantees fp8 inputs, K % group_k == 0,
        N % BLOCK_N_HALF == 0, group_n == 128."""

        def _alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device=a.device)

        triton.set_allocator(_alloc_fn)
        out_tl = _torch_dtype_to_tl(c.dtype)
        block_k = group_k  # one K-tile == one K scale group

        desc_a = TensorDescriptor(
            a, shape=[M, K], strides=[a.stride(0), a.stride(1)],
            block_shape=[BLOCK_M, block_k])
        desc_b = TensorDescriptor(
            b, shape=[N, K], strides=[b.stride(0), b.stride(1)],
            block_shape=[BLOCK_N, block_k])

        num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
        num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
        grid = (min(num_sms, num_tiles),)

        _gemm_tle_kernel[grid](
            desc_a, desc_b, c, a_s, b_s, M, N, K,
            c.stride(0), c.stride(1), a_s.stride(0), a_s.stride(1),
            b_s.stride(0), b_s.stride(1),
            OUT_DTYPE=out_tl,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=block_k,
            BLOCK_N_HALF=BLOCK_N_HALF, GROUP_N=group_n, NUM_BUFS=NUM_BUFS,
            NUM_SMS=grid[0], C_REGS=C_REGS, num_warps=4,
        )
        return c


@functools.lru_cache
def get_w8a8_block_fp8_hopper_configs(N: int, K: int) -> Optional[Dict[int, Any]]:
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    file_name = "w8a8_block_fp8_matmul_hopper.yaml"

    cfg_file = os.path.join(os.path.dirname(__file__), "..", file_name)

    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            logger.info(
                "GEMS_NVIDIA Using config from %s for W8A8 block FP8 kernel.",
                cfg_file,
            )
            dev_data = yaml.safe_load(f).get(device_name, {})
            NK_data = dev_data.get(f"{N},{K}", {})

            result = {}
            for k, p in NK_data.items():
                result[int(k)] = {
                    "BLOCK_SIZE_M": p[0],
                    "BLOCK_SIZE_N": p[1],
                    "BLOCK_SIZE_K": p[2],
                    "GROUP_SIZE_M": p[3],
                    "num_warps": p[4],
                    "num_stages": p[5],
                }

            if not result:
                return None
            return result

    logger.warning(
        "GEMS_NVIDIA Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! "
        "Config file not found at %s",
        cfg_file,
    )
    return None


def _get_placeholder_tuner_configs(pre_hook=None):
    return [
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 128,
                "GROUP_M": 8,
            },
            num_stages=3,
            num_warps=4,
            pre_hook=pre_hook,
        )
    ]

def _get_swap_ab_placeholder_configs(pre_hook=None):
    return [
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 16,
                "BLOCK_K": 128,
                "GROUP_M": 8,
            },
            num_warps=4,
            num_stages=2,
            pre_hook=pre_hook,
        )
    ]

def _get_swap_ab_splitk_placeholder_configs(pre_hook=None):
    return [
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 16,
                "BLOCK_K": 128,
                "GROUP_M": 8,
                "SPLIT_K": 4,
            },
            num_warps=4,
            num_stages=2,
            pre_hook=pre_hook,
        )
    ]

def _get_short_k256_placeholder_configs(pre_hook=None):
    return [
        triton.Config(
            {
                "BLOCK_M": 16,
                "BLOCK_N": 64,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=pre_hook,
        )
    ]

@functools.lru_cache
def _get_fixed_matmul_meta(M: int, N: int, K: int, block_n: int, block_k: int):
    configs = get_w8a8_block_fp8_hopper_configs(N, K)
    if not configs:
        return {
            "BLOCK_M": 64,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": 32,
            "num_warps": 4,
            "num_stages": 2,
        }

    config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    return {
        "BLOCK_M": config["BLOCK_SIZE_M"],
        "BLOCK_N": config["BLOCK_SIZE_N"],
        "BLOCK_K": config["BLOCK_SIZE_K"],
        "GROUP_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }


@libentry()
@libtuner(
    configs=_get_placeholder_tuner_configs(pre_hook=None),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=5,
    flagtune_op_name="w8a8_block_fp8_matmul",
    flagtune_expand_op_name="w8a8_block_fp8_general",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_pre_hook=None,
)
@triton.jit
def w8a8_block_fp8_matmul_kernel_general(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    tl.static_assert(GROUP_K % BLOCK_K == 0)

    SCALAR_BS: tl.constexpr = (BLOCK_N <= GROUP_N) and (GROUP_N % BLOCK_N == 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    as_base = As + offs_am * stride_As_m
    if SCALAR_BS:
        b_scale_group = (pid_n * BLOCK_N) // GROUP_N
        bs_base = Bs + b_scale_group * stride_Bs_n
    else:
        offs_bsn = offs_bn // GROUP_N
        bs_base = Bs + offs_bsn * stride_Bs_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
            b = tl.load(b_ptrs, mask=offs_bn[None, :] < N, other=0.0)
        else:
            k_remaining = K - k * BLOCK_K
            k_mask = offs_k < k_remaining
            a = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < M) & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_bn[None, :] < N) & k_mask[:, None],
                other=0.0,
            )

        k_start = k * BLOCK_K
        offs_ks = k_start // GROUP_K
        a_s = tl.load(as_base + offs_ks * stride_As_k, mask=offs_am < M, other=0.0)

        if SCALAR_BS:
            b_s = tl.load(bs_base + offs_ks * stride_Bs_k)
            partial = tl.dot(a, b, out_dtype=tl.float32)
            ab_scale = a_s * b_s
            acc += partial * ab_scale[:, None]
        else:
            b_s = tl.load(bs_base + offs_ks * stride_Bs_k, mask=offs_bn < N, other=0.0)
            acc += tl.dot(a, b, out_dtype=tl.float32) * a_s[:, None] * b_s[None, :]

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = acc.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = acc.to(tl.float16)
    else:
        c = acc.to(tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@libentry()
@libtuner(
    configs=_get_swap_ab_placeholder_configs(pre_hook=None),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["default", "default", "default", "default", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="w8a8_block_fp8_matmul",
    flagtune_expand_op_name="w8a8_block_fp8_swap_ab",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_pre_hook=None,
)
@triton.jit
def w8a8_block_fp8_matmul_kernel_swap_ab(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Swap-AB matmul: C.T[N, M] = B[N, K] @ A[M, K].T.

    Output tile is [BLOCK_M, BLOCK_N] where BLOCK_M maps to original N and
    BLOCK_N to original M.
    """

    tl.static_assert(GROUP_K % BLOCK_K == 0)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_bm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_an = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    b_ptrs = (
        B
        + offs_bm[:, None] * stride_bn
        + offs_k[None, :] * stride_bk
    )

    a_t_ptrs = (
        A
        + offs_k[:, None] * stride_ak
        + offs_an[None, :] * stride_am
    )

    as_ptrs = As + offs_an * stride_As_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)

    for k_tile in range(0, num_k_tiles):
        if EVEN_K:
            b_tile = tl.load(
                b_ptrs,
                mask=offs_bm[:, None] < N,
                other=0.0,
            )

            a_t_tile = tl.load(
                a_t_ptrs,
                mask=offs_an[None, :] < M,
                other=0.0,
            )
        else:
            k_remaining = K - k_tile * BLOCK_K
            k_mask = offs_k < k_remaining

            b_tile = tl.load(
                b_ptrs,
                mask=(
                    (offs_bm[:, None] < N)
                    & k_mask[None, :]
                ),
                other=0.0,
            )

            a_t_tile = tl.load(
                a_t_ptrs,
                mask=(
                    k_mask[:, None]
                    & (offs_an[None, :] < M)
                ),
                other=0.0,
            )

        scale_k_idx = (k_tile * BLOCK_K) // GROUP_K

        a_scale = tl.load(
            as_ptrs + scale_k_idx * stride_As_k,
            mask=offs_an < M,
            other=0.0,
        )

        if BLOCK_M <= GROUP_N and GROUP_N % BLOCK_M == 0:
            b_scale_group = (pid_m * BLOCK_M) // GROUP_N

            b_scale = tl.load(
                Bs
                + scale_k_idx * stride_Bs_k
                + b_scale_group * stride_Bs_n
            )

            partial = tl.dot(
                b_tile,
                a_t_tile,
                out_dtype=tl.float32,
            )

            ab_scale = a_scale * b_scale
            acc += partial * ab_scale[None, :]

        else:
            b_scale_group = offs_bm // GROUP_N

            b_scale = tl.load(
                Bs
                + scale_k_idx * stride_Bs_k
                + b_scale_group * stride_Bs_n,
                mask=offs_bm < N,
                other=0.0,
            )

            partial = tl.dot(
                b_tile,
                a_t_tile,
                out_dtype=tl.float32,
            )

            acc += (
                partial
                * b_scale[:, None]
                * a_scale[None, :]
            )

        b_ptrs += BLOCK_K * stride_bk
        a_t_ptrs += BLOCK_K * stride_ak

    if C.dtype.element_ty == tl.bfloat16:
        output = acc.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        output = acc.to(tl.float16)
    else:
        output = acc

    c_ptrs = (
        C
        + offs_bm[:, None] * stride_cn
        + offs_an[None, :] * stride_cm
    )

    c_mask = (
        (offs_bm[:, None] < N)
        & (offs_an[None, :] < M)
    )

    tl.store(c_ptrs, output, mask=c_mask)

@libentry()
@libtuner(
    configs=_get_swap_ab_splitk_placeholder_configs(pre_hook=None),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["default", "default", "default", "default", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="w8a8_block_fp8_matmul",
    flagtune_expand_op_name="w8a8_block_fp8_swap_ab_splitk",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_pre_hook=None,
)
@triton.jit
def w8a8_block_fp8_matmul_kernel_swap_ab_splitk(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Swap-AB split-K matmul: C.T[N, M] = B[N, K] @ A[M, K].T, K split over
    program_id(axis=2) and accumulated into C with atomic_add.

    Output tile is [BLOCK_M, BLOCK_N] where BLOCK_M maps to original N and
    BLOCK_N to original M (same layout as w8a8_block_fp8_matmul_kernel_swap_ab).
    This is the skinny-GEMM (tiny M, large K) split-K variant: swap_ab keeps the
    MMA issue efficient while split-K spreads the long-K B read across SMs.
    """

    tl.static_assert(GROUP_K % BLOCK_K == 0)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    offs_bm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_an = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    total_k_tiles = tl.cdiv(K, BLOCK_K)
    k_per_split = tl.cdiv(total_k_tiles, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = min((pid_k + 1) * k_per_split, total_k_tiles)

    k_off0 = k_start * BLOCK_K
    b_ptrs = (
        B
        + offs_bm[:, None] * stride_bn
        + (k_off0 + offs_k)[None, :] * stride_bk
    )
    a_t_ptrs = (
        A
        + (k_off0 + offs_k)[:, None] * stride_ak
        + offs_an[None, :] * stride_am
    )

    as_ptrs = As + offs_an * stride_As_m

    SCALAR_BS: tl.constexpr = (BLOCK_M <= GROUP_N) and (GROUP_N % BLOCK_M == 0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(k_start, k_end):
        if EVEN_K:
            b_tile = tl.load(
                b_ptrs,
                mask=offs_bm[:, None] < N,
                other=0.0,
            )
            a_t_tile = tl.load(
                a_t_ptrs,
                mask=offs_an[None, :] < M,
                other=0.0,
            )
        else:
            k_remaining = K - k_tile * BLOCK_K
            k_mask = offs_k < k_remaining
            b_tile = tl.load(
                b_ptrs,
                mask=((offs_bm[:, None] < N) & k_mask[None, :]),
                other=0.0,
            )
            a_t_tile = tl.load(
                a_t_ptrs,
                mask=(k_mask[:, None] & (offs_an[None, :] < M)),
                other=0.0,
            )

        scale_k_idx = (k_tile * BLOCK_K) // GROUP_K

        a_scale = tl.load(
            as_ptrs + scale_k_idx * stride_As_k,
            mask=offs_an < M,
            other=0.0,
        )

        if SCALAR_BS:
            b_scale_group = (pid_m * BLOCK_M) // GROUP_N
            b_scale = tl.load(
                Bs + scale_k_idx * stride_Bs_k + b_scale_group * stride_Bs_n
            )
            partial = tl.dot(b_tile, a_t_tile, out_dtype=tl.float32)
            ab_scale = a_scale * b_scale
            acc += partial * ab_scale[None, :]
        else:
            b_scale_group = offs_bm // GROUP_N
            b_scale = tl.load(
                Bs + scale_k_idx * stride_Bs_k + b_scale_group * stride_Bs_n,
                mask=offs_bm < N,
                other=0.0,
            )
            partial = tl.dot(b_tile, a_t_tile, out_dtype=tl.float32)
            acc += partial * b_scale[:, None] * a_scale[None, :]

        b_ptrs += BLOCK_K * stride_bk
        a_t_ptrs += BLOCK_K * stride_ak

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = (
        C
        + offs_cm[:, None] * stride_cn
        + offs_cn[None, :] * stride_cm
    )
    c_mask = (offs_cm[:, None] < N) & (offs_cn[None, :] < M)

    if C.dtype.element_ty == tl.bfloat16:
        tl.atomic_add(c_ptrs, acc.to(tl.bfloat16), mask=c_mask, sem="relaxed")
    elif C.dtype.element_ty == tl.float16:
        tl.atomic_add(c_ptrs, acc.to(tl.float16), mask=c_mask, sem="relaxed")
    else:
        tl.atomic_add(c_ptrs, acc.to(tl.float32), mask=c_mask, sem="relaxed")

@libentry()
@libtuner(
    configs=_get_short_k256_placeholder_configs(pre_hook=None),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["default", "default", "default", "default", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="w8a8_block_fp8_matmul",
    flagtune_expand_op_name="w8a8_block_fp8_short_k256",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_pre_hook=None,
)
@triton.jit
def w8a8_block_fp8_matmul_kernel_short_k256(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,  # FlagTune key only; this kernel is called only when K == 256
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Specialized for K == 256, group_k == 128, group_n == 128.

    K is fixed to two 128-channel scale blocks (now loop-based).
    """

    tl.static_assert(128 % BLOCK_N == 0)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)

    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = (
        A
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        B
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    as_base = As + offs_m * stride_As_m

    b_scale_group = (pid_n * BLOCK_N) // 128

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(2):
        a_tile = tl.load(
            a_ptrs,
            mask=mask_m[:, None],
            other=0.0,
        )

        b_tile = tl.load(
            b_ptrs,
            mask=mask_n[None, :],
            other=0.0,
        )

        a_scale = tl.load(
            as_base + k_tile * stride_As_k,
            mask=mask_m,
            other=0.0,
        )

        b_scale = tl.load(
            Bs
            + k_tile * stride_Bs_k
            + b_scale_group * stride_Bs_n
        )

        ab_scale = a_scale * b_scale
        acc += (
            tl.dot(
                a_tile,
                b_tile,
                out_dtype=tl.float32,
            )
            * ab_scale[:, None]
        )

        a_ptrs += 128 * stride_ak
        b_ptrs += 128 * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        output = acc.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        output = acc.to(tl.float16)
    else:
        output = acc

    c_ptrs = (
        C
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )

    c_mask = mask_m[:, None] & mask_n[None, :]

    tl.store(
        c_ptrs,
        output,
        mask=c_mask,
    )

def general_w8a8_block_fp8_matmul(a, b, c, a_s, b_s, M, N, K, group_n, group_k):
    logger.debug(
        "GEMS_NVIDIA W8A8_BLOCK_FP8_MATMUL_HOPPER, [scenario]: general, "
        "[shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s",
        M,
        N,
        K,
        a.stride(0) == 1,
        b.stride(0) == 1,
    )

    use_flagtune = runtime.flagtune_enabled("w8a8_block_fp8_matmul")

    if M < 512 and N > 2112 and K == 256:
        if use_flagtune:
            short_k_grid = lambda META: (
                triton.cdiv(M, META["BLOCK_M"]),
                triton.cdiv(N, META["BLOCK_N"]),
            )
            with torch_device_fn.device(a.device):
                w8a8_block_fp8_matmul_kernel_short_k256[short_k_grid](
                    a,
                    b,
                    c,
                    a_s,
                    b_s,
                    M,
                    N,
                    K,
                    a.stride(0),
                    a.stride(1),
                    b.stride(1),
                    b.stride(0),
                    c.stride(0),
                    c.stride(1),
                    a_s.stride(0),
                    a_s.stride(1),
                    b_s.stride(1),
                    b_s.stride(0),
                )
        else:
            SHORT_K_META = {
                "BLOCK_M": 16,
                "BLOCK_N": 64,
                "num_warps": 4,
                "num_stages": 2,
            }
            short_k_grid = (
                triton.cdiv(M, SHORT_K_META["BLOCK_M"]),
                triton.cdiv(N, SHORT_K_META["BLOCK_N"]),
            )
            with torch_device_fn.device(a.device):
                w8a8_block_fp8_matmul_kernel_short_k256.fn.fn[short_k_grid](
                    a,
                    b,
                    c,
                    a_s,
                    b_s,
                    M,
                    N,
                    K,
                    a.stride(0),
                    a.stride(1),
                    b.stride(1),
                    b.stride(0),
                    c.stride(0),
                    c.stride(1),
                    a_s.stride(0),
                    a_s.stride(1),
                    b_s.stride(1),
                    b_s.stride(0),
                    **SHORT_K_META,
                )
        return c
    elif M < 512 and N > 2112 and K >= 1024:
        if use_flagtune:
            swap_grid = lambda META: (
                triton.cdiv(N, META["BLOCK_M"]),
                triton.cdiv(M, META["BLOCK_N"]),
            )
            with torch_device_fn.device(a.device):
                w8a8_block_fp8_matmul_kernel_swap_ab[swap_grid](
                    a,
                    b,
                    c,
                    a_s,
                    b_s,
                    M,
                    N,
                    K,
                    a.stride(0),
                    a.stride(1),
                    b.stride(1),
                    b.stride(0),
                    c.stride(0),
                    c.stride(1),
                    a_s.stride(0),
                    a_s.stride(1),
                    b_s.stride(1),
                    b_s.stride(0),
                    GROUP_N=group_n,
                    GROUP_K=group_k,
                    EVEN_K=(K % group_k == 0),
                )
        else:
            if group_n % 64 == 0 and N == 4096:
                _SWAP_BM = 64
            elif group_n % 64 == 0 and N == 8192:
                _SWAP_BM = 128
            elif group_n % 32 == 0:
                _SWAP_BM = 32
            else:
                _SWAP_BM = 16

            _SWAP_BN = 16 if M <= 16 else 32

            _SWAP_BK = 128
            _SWAP_STAGES = 4 if triton.cdiv(K, _SWAP_BK) >= 32 else 3

            SWAP_META = {
                "BLOCK_M": _SWAP_BM,
                "BLOCK_N": _SWAP_BN,
                "BLOCK_K": _SWAP_BK,
                "GROUP_M": 8,
                "num_warps": 4,
                "num_stages": _SWAP_STAGES,
            }

            swap_grid = (
                triton.cdiv(N, SWAP_META["BLOCK_M"]),
                triton.cdiv(M, SWAP_META["BLOCK_N"]),
            )

            with torch_device_fn.device(a.device):
                w8a8_block_fp8_matmul_kernel_swap_ab.fn.fn[swap_grid](
                    a,
                    b,
                    c,
                    a_s,
                    b_s,
                    M,
                    N,
                    K,
                    a.stride(0),
                    a.stride(1),
                    b.stride(1),
                    b.stride(0),
                    c.stride(0),
                    c.stride(1),
                    a_s.stride(0),
                    a_s.stride(1),
                    b_s.stride(1),
                    b_s.stride(0),
                    GROUP_N=group_n,
                    GROUP_K=group_k,
                    EVEN_K=(K % group_k == 0),
                    **SWAP_META,
                )
        return c
    elif M < 512 and N < 2112 and K >= 4096:
        if use_flagtune and M > 32:
            swap_splitk_grid = lambda META: (
                triton.cdiv(N, META["BLOCK_M"]),
                triton.cdiv(M, META["BLOCK_N"]),
                META["SPLIT_K"],
            )
            with torch_device_fn.device(a.device):
                c.zero_()
                w8a8_block_fp8_matmul_kernel_swap_ab_splitk[swap_splitk_grid](
                    a,
                    b,
                    c,
                    a_s,
                    b_s,
                    M,
                    N,
                    K,
                    a.stride(0),
                    a.stride(1),
                    b.stride(1),
                    b.stride(0),
                    c.stride(0),
                    c.stride(1),
                    a_s.stride(0),
                    a_s.stride(1),
                    b_s.stride(1),
                    b_s.stride(0),
                    GROUP_N=group_n,
                    GROUP_K=group_k,
                    EVEN_K=(K % group_k == 0),
                )
            return c
        else:
            dev_index = a.device.index
            if dev_index is None:
                dev_index = torch.cuda.current_device()

            if N == 512 and K == 4096:
                _SWAP_SPLITK_BLOCK_K = 64
            else:
                _SWAP_SPLITK_BLOCK_K = 128
            if group_n % 64 == 0:
                _SWAP_SPLITK_BLOCK_M = 64
            elif group_n % 32 == 0:
                _SWAP_SPLITK_BLOCK_M = 32
            else:
                _SWAP_SPLITK_BLOCK_M = 16
            _SWAP_SPLITK_BLOCK_N = 16 if M <= 16 else 32

            grid_m = triton.cdiv(N, _SWAP_SPLITK_BLOCK_M)
            grid_n = triton.cdiv(M, _SWAP_SPLITK_BLOCK_N)
            grid_mn = grid_m * grid_n
            total_k_iters = triton.cdiv(K, _SWAP_SPLITK_BLOCK_K)

            sm_count = torch.cuda.get_device_properties(
                dev_index
            ).multi_processor_count
            split_k = min(total_k_iters, max(4, 2 * sm_count // max(grid_mn, 1)))

            c.zero_()
            swap_splitk_grid = (grid_m, grid_n, split_k)

            with torch_device_fn.device(a.device):
                w8a8_block_fp8_matmul_kernel_swap_ab_splitk.fn.fn[swap_splitk_grid](
                    a,
                    b,
                    c,
                    a_s,
                    b_s,
                    M,
                    N,
                    K,
                    a.stride(0),
                    a.stride(1),
                    b.stride(1),
                    b.stride(0),
                    c.stride(0),
                    c.stride(1),
                    a_s.stride(0),
                    a_s.stride(1),
                    b_s.stride(1),
                    b_s.stride(0),
                    GROUP_N=group_n,
                    GROUP_K=group_k,
                    BLOCK_M=_SWAP_SPLITK_BLOCK_M,
                    BLOCK_N=_SWAP_SPLITK_BLOCK_N,
                    BLOCK_K=_SWAP_SPLITK_BLOCK_K,
                    GROUP_M=8,
                    SPLIT_K=split_k,
                    EVEN_K=(K % _SWAP_SPLITK_BLOCK_K == 0),
                )
            return c
    else:
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )

        if _can_use_tle_2c(a, b, M, N, K, group_n, group_k):
            num_sms = torch.cuda.get_device_properties(
                a.device
            ).multi_processor_count
            n128_tiles = triton.cdiv(M, TLE2C_BLOCK_M) * triton.cdiv(
                N, TLE2C_BLOCK_N
            )
            n64_tiles = triton.cdiv(M, N64_BLOCK_M) * triton.cdiv(N, N64_BLOCK_N)
            cost_n128 = triton.cdiv(n128_tiles, num_sms) * TLE2C_BLOCK_N
            cost_n64 = triton.cdiv(n64_tiles, num_sms) * N64_BLOCK_N

            # n64 wins only when it raises occupancy (num_tiles << NUM_SMS) and
            # K is deep enough to amortize its lower per-wgmma efficiency.
            k_tiles = triton.cdiv(K, group_k)
            use_n64 = cost_n64 < cost_n128 and k_tiles >= 8
            with torch_device_fn.device(a.device):
                if use_n64:
                    w8a8_block_fp8_matmul_tle(
                        a, b, c, a_s, b_s, M, N, K, group_n, group_k,
                        N64_BLOCK_M, N64_BLOCK_N, N64_BLOCK_N_HALF,
                        N64_NUM_BUFS, N64_C_REGS,
                    )
                else:
                    w8a8_block_fp8_matmul_tle(
                        a, b, c, a_s, b_s, M, N, K, group_n, group_k,
                        TLE2C_BLOCK_M, TLE2C_BLOCK_N, TLE2C_BLOCK_N_HALF,
                        TLE2C_NUM_BUFS, TLE2C_C_REGS,
                    )
            return c

        fixed_meta = (
            None
            if use_flagtune
            else _get_fixed_matmul_meta(M, N, K, block_n=group_n, block_k=group_k)
        )

        if use_flagtune:
            launch = lambda: w8a8_block_fp8_matmul_kernel_general[grid](
                a,
                b,
                c,
                a_s,
                b_s,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(1),
                b.stride(0),
                c.stride(0),
                c.stride(1),
                a_s.stride(0),
                a_s.stride(1),
                b_s.stride(1),
                b_s.stride(0),
                GROUP_N=group_n,
                GROUP_K=group_k,
                EVEN_K=(K % group_k == 0),
            )
        else:
            launch = lambda: w8a8_block_fp8_matmul_kernel_general.fn.fn[grid](
                a,
                b,
                c,
                a_s,
                b_s,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(1),
                b.stride(0),
                c.stride(0),
                c.stride(1),
                a_s.stride(0),
                a_s.stride(1),
                b_s.stride(1),
                b_s.stride(0),
                GROUP_N=group_n,
                GROUP_K=group_k,
                EVEN_K=(K % group_k == 0),
                **fixed_meta,
            )

        with torch_device_fn.device(a.device):
            launch()
        return c


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    device = A.device
    assert len(block_size) == 2
    block_n, block_k = block_size

    if A.ndim >= 2 and A.stride(-2) > 1 and A.stride(-1) > 1:
        A = A.contiguous()
    if B.ndim == 2 and B.stride(0) > 1 and B.stride(1) > 1:
        B = B.contiguous()
    if As.ndim >= 2 and As.stride(-2) > 1 and As.stride(-1) > 1:
        As = As.contiguous()
    if Bs.ndim == 2 and Bs.stride(0) > 1 and Bs.stride(1) > 1:
        Bs = Bs.contiguous()

    assert A.shape[-1] == B.shape[-1], "incompatible dimensions"
    assert A.shape[:-1] == As.shape[:-1], "A and As dimensions mismatch"
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1], "invalid As shape"
    assert B.ndim == 2 and Bs.ndim == 2, "B and Bs must be 2D"

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0], "invalid Bs N dimension"
    assert triton.cdiv(K, block_k) == Bs.shape[1], "invalid Bs K dimension"

    output_shape = A.shape[:-1] + (N,)
    c = torch.empty(output_shape, device=device, dtype=output_dtype)

    a_2d = A.reshape(M, K)
    as_2d = As.reshape(M, As.shape[-1])
    c_2d = c.reshape(M, N)

    return general_w8a8_block_fp8_matmul(
        a_2d,
        B,
        c_2d,
        as_2d,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
    ).reshape(c.shape)
