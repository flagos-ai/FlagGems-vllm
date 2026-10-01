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
import math

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner, tl_extra_shim
from flaggems_vllm.utils.device_info import get_device_capability
from flaggems_vllm.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)
_debug = False

# QK uses native INT8 tensor cores with INT32 accumulation. The varlen path
# keeps probabilities in FP16 and converts INT8 V exactly to avoid requantizing P.


@triton.jit
def u64_to_lohi(x):
    return (x >> 32).to(tl.uint32), (x & 0xFFFFFFFF).to(tl.uint32)


@triton.jit
def philox_(seed, subsequence, offset):
    kPhilox10A: tl.constexpr = 0x9E3779B9
    kPhilox10B: tl.constexpr = 0xBB67AE85
    k0, k1 = u64_to_lohi(seed.to(tl.uint64))
    c0, c1 = u64_to_lohi(offset.to(tl.uint64))
    c2, c3 = u64_to_lohi(subsequence.to(tl.uint64))

    # pragma unroll
    kPhiloxSA: tl.constexpr = 0xD2511F53
    kPhiloxSB: tl.constexpr = 0xCD9E8D57
    for _ in tl.static_range(6):
        res0 = kPhiloxSA * c0.to(tl.uint64)
        res1 = kPhiloxSB * c2.to(tl.uint64)
        res0_x, res0_y = u64_to_lohi(res0)
        res1_x, res1_y = u64_to_lohi(res1)
        c0, c1, c2, c3 = res1_y ^ c1 ^ k0, res1_x, res0_y ^ c3 ^ k1, res0_x
        k0 += kPhilox10A
        k1 += kPhilox10B

    res0 = kPhiloxSA * c0.to(tl.uint64)
    res1 = kPhiloxSB * c2.to(tl.uint64)
    res0_x, res0_y = u64_to_lohi(res0)
    res1_x, res1_y = u64_to_lohi(res1)
    c0, c1, c2, c3 = res1_y ^ c1 ^ k0, res1_x, res0_y ^ c3 ^ k1, res0_x

    return c0, c1, c2, c3


@triton.jit
def apply_dropout_mask(
    P,
    mask,
    encode_dropout_in_sign_bit: tl.constexpr,
):
    if encode_dropout_in_sign_bit:
        P = tl.where(mask, -P, P)
    else:
        P = tl.where(mask, (P * 0).to(P.dtype), P)
    return P


@triton.jit
def apply_dropout(
    P,
    row_start,
    col_start,
    n_cols,
    bid,
    hid,
    philox_seed,
    philox_offset,
    p_dropout_uint8: tl.constexpr,
    is_dropout: tl.constexpr,
    encode_dropout_in_sign_bit: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if is_dropout:
        row_start = tl.multiple_of(row_start, BLOCK_M)
        col_start = tl.multiple_of(col_start, BLOCK_N)
        row = row_start + tl.arange(0, BLOCK_M)[:, None]
        # Down scale col_idx by 4
        col = col_start // 4 + tl.arange(0, BLOCK_N // 4)[None, :]

        subsequence = row.to(tl.uint64) * n_cols + col.to(tl.uint64)

        offset = philox_offset + bid * NUM_HEADS + hid
        offset += subsequence * 0
        r0, r1, r2, r3 = philox_(philox_seed, subsequence, offset)

        r = tl.join(tl.join(r0, r1), tl.join(r2, r3)).reshape(BLOCK_M, BLOCK_N)

        mask = (r & 0xFF) >= p_dropout_uint8

        P = apply_dropout_mask(
            P, mask, encode_dropout_in_sign_bit=encode_dropout_in_sign_bit
        )
    return P


@triton.jit
def apply_alibi(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    is_causal: tl.constexpr,
    is_alibi: tl.constexpr,
    alibi_slope: tl.constexpr = None,
):
    if is_alibi:
        if is_causal:
            # The row independent alibi bias renders the same attention output
            # as with the standard alibi because softmax is shift invariant, i.e.,
            # softmax(A + bias + const) = softamx(A + bias). The following two
            # biases are no different if causal is true.
            bias = alibi_slope * (-max_seqlen_k + 1 + col_idx[None, :]).to(tl.float32)
            S += bias
        else:
            bias = -alibi_slope * tl.abs(
                col_idx[None, :] - max_seqlen_k + max_seqlen_q - row_idx[:, None]
            ).to(tl.float32)
            S += bias

    return S


@triton.jit
def apply_mask(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    window_size_left,
    window_size_right,
    is_even_mn: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
):
    need_mask = is_causal | is_local | (not is_even_mn)
    # need_mask: tl.constexpr = is_causal | is_local
    if need_mask:
        col_lb = tl.where(
            window_size_left < 0,
            0,
            tl.maximum(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left),
        )
        col_rb = tl.where(
            window_size_right < 0,
            max_seqlen_k - 1,
            tl.minimum(
                max_seqlen_k - 1,
                row_idx + max_seqlen_k - max_seqlen_q + window_size_right,
            ),
        )

        if is_causal:
            S = tl.where(col_idx[None, :] > col_rb[:, None], float("-inf"), S)

        if is_local:
            S = tl.where(
                (col_idx[None, :] > col_rb[:, None])
                | (col_idx[None, :] < col_lb[:, None]),
                float("-inf"),
                S,
            )

        if (not is_local) & (not is_causal) & (not is_even_mn):
            S = tl.where(col_idx[None, :] >= max_seqlen_k, float("-inf"), S)

    return S


@triton.jit
def softmax_rescale(
    O_acc,
    S,
    row_max,
    row_sum,
    softmax_scale_log2e: tl.constexpr,
    is_border: tl.constexpr,
    local_prob: tl.constexpr = runtime.device.vendor_name == "metax",
):
    prev_max = row_max
    S_tile_max = tl.max(S, 1)
    row_max = tl.maximum(row_max, S_tile_max)

    if is_border:
        cur_max = tl.where(row_max == float("-inf"), 0, row_max)
    else:
        cur_max = row_max

    p_scale = tl.math.exp2((prev_max - cur_max) * softmax_scale_log2e)
    row_sum *= p_scale
    O_acc *= p_scale[:, None]

    max_scaled = tl.where(row_max == float("-inf"), 0, row_max * softmax_scale_log2e)
    if local_prob:
        safe_tile_max = tl.where(S_tile_max == float("-inf"), 0, S_tile_max)
        P = tl.math.exp2((S - safe_tile_max[:, None]) * softmax_scale_log2e)
        p_tile_max = tl.math.exp2((S_tile_max - cur_max) * softmax_scale_log2e)
        row_sum = row_sum + tl.sum(P, 1) * p_tile_max
    else:
        P = tl.math.exp2(S * softmax_scale_log2e - max_scaled[:, None])
        p_tile_max = tl.math.exp2(S_tile_max * softmax_scale_log2e - max_scaled)
        row_sum = row_sum + tl.sum(P, 1)
    return O_acc, P, row_max, row_sum, p_tile_max


@triton.jit
def _load_int8_block_descales(
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    q_descale_batch_stride,
    q_descale_head_stride,
    q_descale_block_stride,
    k_descale_batch_stride,
    k_descale_head_stride,
    k_descale_block_stride,
    v_descale_batch_stride,
    v_descale_head_stride,
    v_descale_block_stride,
    bid,
    hid,
    kv_hid,
    row_start,
    col_start,
):

    q_block = row_start // 128
    kv_block = col_start // 128
    q_descale = tl.load(
        q_descale_ptr
        + bid * q_descale_batch_stride
        + hid * q_descale_head_stride
        + q_block * q_descale_block_stride
    ).to(tl.float32)
    k_descale = tl.load(
        k_descale_ptr
        + bid * k_descale_batch_stride
        + kv_hid * k_descale_head_stride
        + kv_block * k_descale_block_stride
    ).to(tl.float32)
    v_descale = tl.load(
        v_descale_ptr
        + bid * v_descale_batch_stride
        + kv_hid * v_descale_head_stride
        + kv_block * v_descale_block_stride
    ).to(tl.float32)
    return q_descale, k_descale, v_descale


@triton.jit
def _int8_pv_dot(
    P,
    P_tile_max,
    V,
    acc,
    v_descale,
    precise_p: tl.constexpr = False,
    transpose_pv: tl.constexpr = False,
    local_prob: tl.constexpr = runtime.device.vendor_name == "metax",
    half_pv: tl.constexpr = False,
    fold_v_descale: tl.constexpr = False,
):
    if half_pv:
        if fold_v_descale:
            if transpose_pv:
                return tl.trans(
                    tl.dot(
                        tl.trans(V.to(tl.float16)),
                        tl.trans(P.to(tl.float16)),
                        tl.trans(acc),
                    )
                )
            else:
                return tl.dot(P.to(tl.float16), V.to(tl.float16), acc)
        else:
            if transpose_pv:
                pv = tl.trans(
                    tl.dot(
                        tl.trans(V.to(tl.float16)),
                        tl.trans(P.to(tl.float16)),
                        out_dtype=tl.float32,
                    )
                )
            else:
                pv = tl.dot(P.to(tl.float16), V.to(tl.float16), out_dtype=tl.float32)
            return acc + pv * v_descale
    elif precise_p and local_prob:
        fixed = (P * (127.0 * 256.0) + 0.5).to(tl.int32)
        high = ((fixed + 128) >> 8).to(tl.int8)
        low = fixed.to(tl.int8)
        v_i8 = V.to(tl.int8)
        if transpose_pv:
            high_dot = tl.dot(tl.trans(v_i8), tl.trans(high), out_dtype=tl.int32)
            both = tl.dot(
                tl.trans(v_i8), tl.trans(low), high_dot << 8, out_dtype=tl.int32
            )
            pv = tl.trans(both).to(tl.float32)
        else:
            high_dot = tl.dot(high, v_i8, out_dtype=tl.int32)
            pv = tl.dot(low, v_i8, high_dot << 8, out_dtype=tl.int32).to(tl.float32)
        p_scale = P_tile_max * (1.0 / (127.0 * 256.0))
        return acc + pv * (p_scale[:, None] * v_descale)
    else:
        if local_prob:
            # P_tile_max carries beta, the tile-to-running softmax scale.
            p_scale = P_tile_max * (1.0 / 127.0)
            P_s = P * 127.0
        else:
            p_amax = tl.maximum(P_tile_max, 1.0 / (1 << 30))
            p_scale = p_amax * (1.0 / 127.0)
            P_s = P * (127.0 / p_amax)[:, None]
        P_i8 = (P_s + 0.5).to(tl.int8)
        V_i8 = V.to(tl.int8)
        if transpose_pv:
            pv = tl.trans(
                tl.dot(tl.trans(V_i8), tl.trans(P_i8), out_dtype=tl.int32)
            ).to(tl.float32)
        else:
            pv = tl.dot(P_i8, V_i8, out_dtype=tl.int32).to(tl.float32)
        if precise_p:
            R_s = P_s - P_i8.to(tl.float32)
            r_amax = tl.maximum(tl.max(tl.abs(R_s), 1), 1.0 / (1 << 30))
            R_sc = R_s * (127.0 / r_amax)[:, None]
            # Round half away from zero: the +0.5-trunc trick alone is wrong
            # for negative residuals (truncation heads to zero, not away).
            R_mag = tl.abs(R_sc) + 0.5
            R_i8 = tl.where(R_sc >= 0, R_mag.to(tl.int8), (-R_mag).to(tl.int8))
            if transpose_pv:
                residual = tl.trans(
                    tl.dot(tl.trans(V_i8), tl.trans(R_i8), out_dtype=tl.int32)
                ).to(tl.float32)
            else:
                residual = tl.dot(R_i8, V_i8, out_dtype=tl.int32).to(tl.float32)
            pv += residual * (r_amax * (1.0 / 127.0))[:, None]
        return acc + pv * (p_scale[:, None] * v_descale)


@triton.jit
def _int8_qk_tn_fragments(
    Q,
    K,
    q_rows,
    k_rows,
    q_len,
    k_len,
    q_stride,
    k_stride,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IS_BORDER: tl.constexpr,
):
    scores_t = tl.zeros((BN, BM), dtype=tl.int32)
    for fragment in tl.range(0, tl.cdiv(D, 64)):
        channel = fragment * 64 + tl.arange(0, 64)
        q_tile = tl.load(
            Q + channel[:, None] + q_rows[None, :] * q_stride,
            (channel[:, None] < D) & (q_rows[None, :] < q_len),
            other=0,
        )
        if IS_BORDER:
            k_tile = tl.load(
                K + k_rows[:, None] * k_stride + channel[None, :],
                (k_rows[:, None] < k_len) & (channel[None, :] < D),
                other=0,
            )
        else:
            k_tile = tl.load(
                K + k_rows[:, None] * k_stride + channel[None, :],
                channel[None, :] < D,
                other=0,
            )
        scores_t = tl.dot(k_tile, q_tile, scores_t, out_dtype=tl.int32)
    return tl.trans(scores_t).to(tl.float32)


@triton.jit
def apply_softcap(S, softcap, is_softcap: tl.constexpr):
    if is_softcap:
        S = tl_extra_shim.tanh(S * softcap)

    return S


def block_m_splitkv_heuristic(headdim):
    return 128 if headdim <= 128 else 64


def block_n_splitkv_heuristic(headdim):
    return 64 if headdim <= 64 else 32


def is_even_mn(M, N, BM, BN, WL, WR):
    if M % BM == 0 and N % BN == 0:
        if M % N == 0 or N % M == 0:
            if (WL == -1 or WL % BN == 0) and (WR == -1 or WR % BN == 0):
                return True
    return False


def block_m_splitkv_heuristic_spec_args(args):
    return 128 if args["d"] <= 128 else 64


def block_n_splitkv_heuristic_spec_args(args):
    return 64 if args["d"] <= 64 else 32


def is_even_mn_spec_args(args):
    if (
        args["seqlen_q"] % args["BLOCK_M"] == 0
        and args["seqlen_k"] % args["BLOCK_N"] == 0
    ):
        if (
            args["seqlen_q"] % args["seqlen_k"] == 0
            or args["seqlen_k"] % args["seqlen_q"] == 0
        ):
            if (
                args["window_size_left"] == -1
                or args["window_size_left"] % args["BLOCK_N"] == 0
            ) and (
                args["window_size_right"] == -1
                or args["window_size_right"] % args["BLOCK_N"] == 0
            ):
                return True
    return False


def fwd_configs_w8a8():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    ]


def prune_fwd_configs_w8a8(configs, nargs, **kwargs):
    d = nargs["d"]
    is_dropout = nargs["is_dropout"]
    out = []
    for cfg in configs:
        BM = cfg.kwargs["BLOCK_M"]
        BN = cfg.kwargs["BLOCK_N"]
        w = cfg.num_warps
        s = cfg.num_stages

        if is_dropout and (w != 4 or s > 3):
            continue

        if d <= 64:
            if (BM, BN, w, s) in {
                (128, 32, 4, 2),
                (128, 128, 8, 2),
                (128, 64, 4, 2),
                (128, 64, 4, 3),
                (128, 128, 4, 2),
                (128, 128, 4, 3),
            }:
                out.append(cfg)
        else:
            if (BM, BN, w, s) in {
                (128, 32, 8, 2),
                (128, 32, 8, 3),
                (128, 64, 8, 2),
                (128, 64, 8, 3),
                (128, 128, 4, 2),
                (128, 128, 8, 2),
                (64, 128, 8, 3),
            }:
                out.append(cfg)

    return out


def flash_fwd_kernel_heur_block_k(args):
    return triton.next_power_of_2(args["d"])


@libentry()
@triton.autotune(
    configs=fwd_configs_w8a8(),
    prune_configs_by={"early_config_prune": prune_fwd_configs_w8a8},
    key=["b", "h", "seqlen_q", "seqlen_k", "d", "is_dropout"],
)
@triton.heuristics(
    values={
        "BLOCK_K": flash_fwd_kernel_heur_block_k,
        "PRE_LOAD_V": lambda args: False,
        "IS_EVEN_MN": lambda args: is_even_mn(
            args["seqlen_q"],
            args["seqlen_k"],
            args["BLOCK_M"],
            args["BLOCK_N"],
            args["window_size_left"],
            args["window_size_right"],
        ),
    }
)
@triton.jit(
    do_not_specialize=["seqlen_q", "seqlen_k", "seqlen_q_rounded", "seqlen_k_rounded"]
)
def flash_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,
    is_cu_seqlens_q,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k,
    cu_seqlens_k_ptr,
    is_seqused_k,
    seqused_k_ptr,
    # sizes
    b: tl.constexpr,
    bk: tl.constexpr,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    d: tl.constexpr,
    d_rounded: tl.constexpr,
    # scaling factors
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    q_descale_batch_stride,
    q_descale_head_stride,
    q_descale_block_stride,
    k_descale_batch_stride,
    k_descale_head_stride,
    k_descale_block_stride,
    v_descale_batch_stride,
    v_descale_head_stride,
    v_descale_block_stride,
    # dropout
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    # causal and swa
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    is_paged: tl.constexpr,
    # alibi
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    # block table
    total_q: tl.constexpr,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    k_page_stride: tl.constexpr,
    precise_p: tl.constexpr,
    # kernel params
    IS_EVEN_MN: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLIT_D: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    m_block = tl.program_id(0)
    bh = tl.program_id(1)
    d_split = tl.program_id(2)
    d_start = d_split * BLOCK_D
    hid = bh % h
    bid = bh // h
    num_m_blocks = tl.cdiv(seqlen_q, BLOCK_M)

    col_min = 0
    if is_local and window_size_left >= 0:
        col_min = max(0, m_block * BLOCK_M + seqlen_k - seqlen_q - window_size_left)
        if not IS_EVEN_MN:
            # round left
            col_min = (col_min // BLOCK_N) * BLOCK_N

    col_max = seqlen_k
    if is_causal or (is_local and window_size_right >= 0):
        col_max += (m_block - num_m_blocks + 1) * BLOCK_M
        if is_local:
            col_max += window_size_right
        col_max = min(seqlen_k, col_max)

    if not IS_EVEN_MN:
        # round right
        col_max = tl.cdiv(col_max, BLOCK_N) * BLOCK_N

    if (not is_causal) and (not is_local):
        if IS_EVEN_MN:
            masking_cols: tl.constexpr = 0
        else:
            masking_cols: tl.constexpr = BLOCK_N
    elif (is_causal | is_local) and IS_EVEN_MN:
        masking_cols: tl.constexpr = tl.cdiv(BLOCK_M, BLOCK_N) * BLOCK_N
    else:
        # local
        masking_cols: tl.constexpr = (tl.cdiv(BLOCK_M, BLOCK_N) + 1) * BLOCK_N

    if is_dropout:
        philox_seed = tl.load(philox_args).to(tl.uint64)
        philox_offset = tl.load(philox_args + 1).to(tl.uint64)

    if is_alibi:
        alibi_offset = bid * alibi_slopes_batch_stride + hid
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
        alibi_slope /= scale_softmax
    else:
        alibi_slope = 0.0

    q_batch_stride = tl.multiple_of(q_batch_stride, d * h)
    q_ptr += bid * q_batch_stride + hid * q_head_stride
    row_start = m_block * BLOCK_M
    row_idx = row_start + tl.arange(0, BLOCK_M)
    q_off = row_idx[:, None] * q_row_stride + tl.arange(0, BLOCK_K)[None, :]
    dmask = tl.arange(0, BLOCK_K) < d
    qmask = dmask[None, :] & (row_idx[:, None] < seqlen_q)
    if IS_EVEN_MN & d == BLOCK_K:
        Q = tl.load(q_ptr + q_off, cache_modifier=".cg")
    else:
        Q = tl.load(q_ptr + q_off, mask=qmask, cache_modifier=".cg")

    if return_softmax:
        p_ptr += (
            (bid * h + hid) * seqlen_q_rounded + m_block * BLOCK_M
        ) * seqlen_k_rounded
        p_offset = tl.arange(0, BLOCK_M)[:, None] * seqlen_k_rounded + tl.arange(
            0, BLOCK_N
        )
        p_bp0 = p_ptr + p_offset

    # Use a [BLOCK_M, BLOCK_D] PV accumulator. For D128, BLOCK_D is 64,
    # so one CTA does not hold a [BM, 128] accumulator, reducing register pressure.
    acc_ = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    k_batch_stride = tl.multiple_of(k_batch_stride, d * hk)
    h_hk_ratio = h // hk
    kv_hid = hid // h_hk_ratio
    k_ptr += bid * k_batch_stride
    k_ptr += kv_hid * k_head_stride
    v_ptr += bid * k_batch_stride
    v_ptr += kv_hid * k_head_stride

    k_offset = (
        tl.arange(0, BLOCK_N)[None, :] * k_row_stride + tl.arange(0, BLOCK_K)[:, None]
    )
    v_d = d_start + tl.arange(0, BLOCK_D)
    v_dmask = v_d < d
    v_offset = tl.arange(0, BLOCK_N)[:, None] * k_row_stride + v_d[None, :]

    p_bk0 = k_ptr + k_offset
    p_bv0 = v_ptr + v_offset

    if is_causal | is_local | (not IS_EVEN_MN):
        # Cut short masking cols if there's not enough cols out there
        masking_cols = min(col_max - col_min, masking_cols)
        for col_shift in tl.range(0, masking_cols, step=BLOCK_N):
            col_start = col_max - col_shift - BLOCK_N
            col_start = tl.multiple_of(col_start, BLOCK_N)
            off = col_start * k_row_stride
            if IS_EVEN_MN & d == BLOCK_K:
                K = tl.load(p_bk0 + off, cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(p_bv0 + off, cache_modifier=".cg")
            elif d == BLOCK_K:
                col_idx = col_start + tl.arange(0, BLOCK_N)
                kvmask = col_idx < seqlen_k
                K = tl.load(p_bk0 + off, mask=kvmask[None, :], cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(p_bv0 + off, mask=kvmask[:, None], cache_modifier=".cg")
            else:
                col_idx = col_start + tl.arange(0, BLOCK_N)
                kvmask = col_idx < seqlen_k
                K = tl.load(
                    p_bk0 + off,
                    mask=kvmask[None, :] & dmask[:, None],
                    cache_modifier=".cg",
                )
                if PRE_LOAD_V:
                    V = tl.load(
                        p_bv0 + off,
                        mask=kvmask[:, None] & v_dmask[None, :],
                        cache_modifier=".cg",
                    )
            q_descale, k_descale, v_descale = _load_int8_block_descales(
                q_descale_ptr,
                k_descale_ptr,
                v_descale_ptr,
                q_descale_batch_stride,
                q_descale_head_stride,
                q_descale_block_stride,
                k_descale_batch_stride,
                k_descale_head_stride,
                k_descale_block_stride,
                v_descale_batch_stride,
                v_descale_head_stride,
                v_descale_block_stride,
                bid,
                hid,
                kv_hid,
                row_start,
                col_start,
            )
            S = tl.dot(Q, K, out_dtype=tl.int32).to(tl.float32)
            S *= q_descale * k_descale
            S = apply_softcap(S, softcap, is_softcap)
            col_idx = col_start + tl.arange(0, BLOCK_N)
            row_idx = row_start + tl.arange(0, BLOCK_M)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            S = apply_mask(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
                is_even_mn=IS_EVEN_MN,
                is_causal=is_causal,
                is_local=is_local,
            )

            acc_, P, rowmax_, rowsum_, p_tile_max = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=(is_causal or is_local),
            )

            if is_dropout:
                if return_softmax:
                    P_drop = apply_dropout(
                        P,
                        row_start,
                        col_start,
                        seqlen_k,
                        bid,
                        hid,
                        philox_seed,
                        philox_offset,
                        p_dropout_in_uint8_t,
                        is_dropout,
                        encode_dropout_in_sign_bit=True,
                        NUM_HEADS=h,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                    )
                    if IS_EVEN_MN:
                        tl.store(p_bp0 + col_start, P_drop, mask=d_split == 0)
                    else:
                        kvmask = col_idx < seqlen_k
                        tl.store(
                            p_bp0 + col_start,
                            P_drop,
                            mask=qmask & kvmask[None, :] & (d_split == 0),
                        )

                P = apply_dropout(
                    P,
                    row_start,
                    col_start,
                    seqlen_k,
                    bid,
                    hid,
                    philox_seed,
                    philox_offset,
                    p_dropout_in_uint8_t,
                    is_dropout,
                    encode_dropout_in_sign_bit=False,
                    NUM_HEADS=h,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                )
                p_tile_max = tl.max(P, 1)

            if not PRE_LOAD_V:
                off = col_start * k_row_stride
                if IS_EVEN_MN & d == BLOCK_K:
                    V = tl.load(p_bv0 + off, cache_modifier=".cg")
                elif d == BLOCK_K:
                    kvmask = col_idx < seqlen_k
                    V = tl.load(p_bv0 + off, mask=kvmask[:, None], cache_modifier=".cg")
                else:
                    kvmask = col_idx < seqlen_k
                    V = tl.load(
                        p_bv0 + off,
                        mask=kvmask[:, None] & v_dmask[None, :],
                        cache_modifier=".cg",
                    )
            acc_ = _int8_pv_dot(P, p_tile_max, V, acc_, v_descale, precise_p)

    for col_start in tl.range(
        col_min, col_max - masking_cols, step=BLOCK_N, num_stages=num_stages
    ):
        col_start = tl.multiple_of(col_start, BLOCK_N)
        off = col_start * k_row_stride
        if d == BLOCK_K:
            K = tl.load(p_bk0 + off, cache_modifier=".cg")
            if PRE_LOAD_V:
                V = tl.load(p_bv0 + off, cache_modifier=".cg")
        else:
            K = tl.load(p_bk0 + off, mask=dmask[:, None], cache_modifier=".cg")
            if PRE_LOAD_V:
                V = tl.load(p_bv0 + off, mask=v_dmask[None, :], cache_modifier=".cg")

        q_descale, k_descale, v_descale = _load_int8_block_descales(
            q_descale_ptr,
            k_descale_ptr,
            v_descale_ptr,
            q_descale_batch_stride,
            q_descale_head_stride,
            q_descale_block_stride,
            k_descale_batch_stride,
            k_descale_head_stride,
            k_descale_block_stride,
            v_descale_batch_stride,
            v_descale_head_stride,
            v_descale_block_stride,
            bid,
            hid,
            kv_hid,
            row_start,
            col_start,
        )
        S = tl.dot(Q, K, out_dtype=tl.int32).to(tl.float32)
        S *= q_descale * k_descale
        S = apply_softcap(S, softcap, is_softcap)
        col_idx = col_start + tl.arange(0, BLOCK_N)
        row_idx = row_start + tl.arange(0, BLOCK_M)
        S = apply_alibi(
            S,
            col_idx,
            row_idx,
            seqlen_q,
            seqlen_k,
            is_causal=is_causal,
            is_alibi=is_alibi,
            alibi_slope=alibi_slope,
        )
        S = apply_mask(
            S,
            col_idx,
            row_idx,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
            is_even_mn=True,
            is_causal=False,
            is_local=is_local,
        )

        acc_, P, rowmax_, rowsum_, p_tile_max = softmax_rescale(
            acc_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=scale_softmax_log2,
            is_border=is_local,
        )

        if is_dropout:
            if return_softmax:
                P_drop = apply_dropout(
                    P,
                    row_start,
                    col_start,
                    seqlen_k,
                    bid,
                    hid,
                    philox_seed,
                    philox_offset,
                    p_dropout_in_uint8_t,
                    is_dropout,
                    encode_dropout_in_sign_bit=True,
                    NUM_HEADS=h,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                )
                if IS_EVEN_MN:
                    tl.store(p_bp0 + col_start, P_drop, mask=d_split == 0)
                else:
                    kvmask = col_idx < seqlen_k
                    tl.store(
                        p_bp0 + col_start,
                        P_drop,
                        mask=qmask & kvmask[None, :] & (d_split == 0),
                    )

            P = apply_dropout(
                P,
                row_start,
                col_start,
                seqlen_k,
                bid,
                hid,
                philox_seed,
                philox_offset,
                p_dropout_in_uint8_t,
                is_dropout,
                encode_dropout_in_sign_bit=False,
                NUM_HEADS=h,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
            p_tile_max = tl.max(P, 1)

        if not PRE_LOAD_V:
            off = col_start * k_row_stride
            if d == BLOCK_K:
                V = tl.load(p_bv0 + off, cache_modifier=".cg")
            else:
                V = tl.load(p_bv0 + off, mask=v_dmask[None, :], cache_modifier=".cg")
        acc_ = _int8_pv_dot(P, p_tile_max, V, acc_, v_descale, precise_p)

    # LSE
    lse = tl.where(
        rowsum_ == 0 | (rowsum_ != rowsum_),
        float("inf"),
        rowmax_ * scale_softmax + tl.log(rowsum_),
    )
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)

    if is_dropout:
        acc_ *= inv_sum[:, None] * rp_dropout
    else:
        acc_ *= inv_sum[:, None]

    out = acc_.to(o_ptr.type.element_ty)  # noqa

    o_batch_stride = tl.multiple_of(o_batch_stride, d * h)
    o_ptr += bid * o_batch_stride
    o_ptr += hid * o_head_stride
    o_cols = d_start + tl.arange(0, BLOCK_D)
    o_dmask = o_cols < d
    o_offset = row_idx[:, None] * o_row_stride + o_cols[None, :]

    if IS_EVEN_MN & (d == BLOCK_K) & (not SPLIT_D):
        tl.store(o_ptr + o_offset, out)
    else:
        tl.store(
            o_ptr + o_offset, out, mask=(row_idx[:, None] < seqlen_q) & o_dmask[None, :]
        )

    # Write back lse
    p_lse = softmax_lse_ptr + (bid * h + hid) * seqlen_q
    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)

    lse_write_mask = d_split == 0
    if IS_EVEN_MN:
        tl.store(p_lse + row_idx, lse, mask=lse_write_mask)
    else:
        tl.store(p_lse + row_idx, lse, mask=(row_idx < seqlen_q) & lse_write_mask)


def flash_fwd_splitkv_kernel_heur_block_k(args):
    return triton.next_power_of_2(args["d"])


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": block_m_splitkv_heuristic_spec_args,
        "BLOCK_N": block_n_splitkv_heuristic_spec_args,
        "BLOCK_K": flash_fwd_splitkv_kernel_heur_block_k,
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 3,
        "PRE_LOAD_V": lambda args: True,
        "IS_EVEN_MN": is_even_mn_spec_args,
    }
)
@triton.jit(
    do_not_specialize=["seqlen_q", "seqlen_k", "seqlen_q_rounded", "seqlen_k_rounded"]
)
def flash_fwd_splitkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,
    is_cu_seqlens_q,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k: tl.constexpr,
    cu_seqlens_k_ptr,
    is_seqused_k: tl.constexpr,
    seqused_k_ptr,
    # sizes
    b: tl.constexpr,
    bk: tl.constexpr,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    d: tl.constexpr,
    d_rounded: tl.constexpr,
    # scaling factors
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    q_descale_batch_stride,
    q_descale_head_stride,
    q_descale_block_stride,
    k_descale_batch_stride,
    k_descale_head_stride,
    k_descale_block_stride,
    v_descale_batch_stride,
    v_descale_head_stride,
    v_descale_block_stride,
    # dropout
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    # causal and swa
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    is_paged: tl.constexpr,
    # alibi
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    # block table
    total_q,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    k_page_stride: tl.constexpr,
    precise_p: tl.constexpr,
    # kernel params
    IS_EVEN_MN: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    blocks_per_split: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    m_block = tl.program_id(0)
    split_id = tl.program_id(1)
    bid = tl.program_id(2) // h
    hid = tl.program_id(2) % h

    split_block_min = split_id * blocks_per_split
    split_block_max = split_block_min + blocks_per_split

    n_block_max = tl.cdiv(seqlen_k, BLOCK_N)
    if is_causal:
        n_block_max = min(
            n_block_max,
            tl.cdiv(
                (m_block + 1) * BLOCK_M + seqlen_k - seqlen_q + window_size_right,
                BLOCK_N,
            ),
        )

    if is_alibi:
        alibi_offset = bid * alibi_slopes_batch_stride + hid
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
        alibi_slope /= scale_softmax
    else:
        alibi_slope = 0

    if not is_causal:
        if IS_EVEN_MN:
            masking_block_min = n_block_max
        else:
            masking_block_min = n_block_max - 1
    elif is_causal and IS_EVEN_MN:  # causal implies window_size_right is zero
        masking_block_min = n_block_max - tl.cdiv(BLOCK_M, BLOCK_N)
    else:
        masking_block_min = n_block_max - tl.cdiv(BLOCK_M, BLOCK_N) - 1

    q_ptr += bid * q_batch_stride
    q_ptr += hid * q_head_stride
    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    q_off = row_idx[:, None] * q_row_stride + tl.arange(0, BLOCK_K)[None, :]
    p_qm = q_ptr + q_off
    dmask = tl.arange(0, BLOCK_K) < d
    qmask = dmask[None, :] & (row_idx[:, None] < seqlen_q)
    if IS_EVEN_MN & BLOCK_K == d:
        Q = tl.load(p_qm, cache_modifier=".cg")
    else:
        Q = tl.load(p_qm, mask=qmask, cache_modifier=".cg")

    h_hk_ratio = h // hk
    kv_hid = hid // h_hk_ratio
    k_ptr += bid * k_batch_stride
    k_ptr += kv_hid * k_head_stride
    v_ptr += bid * k_batch_stride
    v_ptr += kv_hid * k_head_stride

    k_offset = (
        tl.arange(0, BLOCK_N)[None, :] * k_row_stride + tl.arange(0, BLOCK_K)[:, None]
    )
    p_k0 = k_ptr + k_offset

    v_offset = (
        tl.arange(0, BLOCK_N)[:, None] * k_row_stride + tl.arange(0, BLOCK_K)[None, :]
    )
    p_v0 = v_ptr + v_offset

    acc_ = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    if split_block_max <= masking_block_min:
        # no masking needed
        for n_block in tl.range(
            split_block_min, split_block_max, num_stages=num_stages
        ):
            kv_off = n_block * BLOCK_N * k_row_stride
            if d == BLOCK_K:
                K = tl.load(p_k0 + kv_off, cache_modifier=".cg")
            else:
                K = tl.load(
                    p_k0 + kv_off, mask=dmask[:, None], cache_modifier=".cg", other=0.0
                )
            if PRE_LOAD_V:
                if d == BLOCK_K:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
                else:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :],
                        cache_modifier=".cg",
                        other=0.0,
                    )
            row_start = m_block * BLOCK_M
            col_start = n_block * BLOCK_N
            q_descale, k_descale, v_descale = _load_int8_block_descales(
                q_descale_ptr,
                k_descale_ptr,
                v_descale_ptr,
                q_descale_batch_stride,
                q_descale_head_stride,
                q_descale_block_stride,
                k_descale_batch_stride,
                k_descale_head_stride,
                k_descale_block_stride,
                v_descale_batch_stride,
                v_descale_head_stride,
                v_descale_block_stride,
                bid,
                hid,
                kv_hid,
                row_start,
                col_start,
            )
            S = tl.dot(Q, K, out_dtype=tl.int32).to(tl.float32)
            S *= q_descale * k_descale
            S = apply_softcap(S, softcap, is_softcap)
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            acc_, P, rowmax_, rowsum_, p_tile_max = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=False,
            )

            if not PRE_LOAD_V:
                if d == BLOCK_K:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
                else:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :],
                        cache_modifier=".cg",
                        other=0.0,
                    )
            # split-KV PV uses an in-kernel INT8 P tile and INT8 V.
            acc_ = _int8_pv_dot(P, p_tile_max, V, acc_, v_descale, precise_p)
    else:
        for n_block in tl.range(split_block_min, min(split_block_max, n_block_max)):
            kv_off = n_block * BLOCK_N * k_row_stride
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
            if IS_EVEN_MN & d == BLOCK_K:
                K = tl.load(p_k0 + kv_off, cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
            elif d == BLOCK_K:
                kvmask = col_idx < seqlen_k
                K = tl.load(p_k0 + kv_off, mask=kvmask[None, :], cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(
                        p_v0 + kv_off, mask=kvmask[:, None], cache_modifier=".cg"
                    )
            else:
                kvmask = col_idx < seqlen_k
                K = tl.load(
                    p_k0 + kv_off,
                    mask=dmask[:, None] & kvmask[None, :],
                    cache_modifier=".cg",
                    other=0.0,
                )
                if PRE_LOAD_V:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :] & kvmask[:, None],
                        cache_modifier=".cg",
                        other=0.0,
                    )

            row_start = m_block * BLOCK_M
            col_start = n_block * BLOCK_N
            q_descale, k_descale, v_descale = _load_int8_block_descales(
                q_descale_ptr,
                k_descale_ptr,
                v_descale_ptr,
                q_descale_batch_stride,
                q_descale_head_stride,
                q_descale_block_stride,
                k_descale_batch_stride,
                k_descale_head_stride,
                k_descale_block_stride,
                v_descale_batch_stride,
                v_descale_head_stride,
                v_descale_block_stride,
                bid,
                hid,
                kv_hid,
                row_start,
                col_start,
            )
            S = tl.dot(Q, K, out_dtype=tl.int32).to(tl.float32)
            S *= q_descale * k_descale
            S = apply_softcap(S, softcap, is_softcap)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            S = apply_mask(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
                is_even_mn=IS_EVEN_MN,
                is_causal=is_causal,
                is_local=False,
            )

            acc_, P, rowmax_, rowsum_, p_tile_max = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=(is_causal or is_local),
            )

            if not PRE_LOAD_V:
                if IS_EVEN_MN & d == BLOCK_K:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
                elif d == BLOCK_K:
                    V = tl.load(
                        p_v0 + kv_off, mask=kvmask[:, None], cache_modifier=".cg"
                    )
                else:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :] & kvmask[:, None],
                        cache_modifier=".cg",
                        other=0.0,
                    )
            acc_ = _int8_pv_dot(P, p_tile_max, V, acc_, v_descale, precise_p)

    lse = tl.where(
        rowsum_ == 0 | (rowsum_ != rowsum_),
        float("-inf"),
        rowmax_ * scale_softmax + tl.log(rowsum_),
    )
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)

    # Rescale output
    acc_ *= inv_sum[:, None]

    o_split_ptr = o_ptr
    o_split_ptr += (split_id * tl.num_programs(2) + tl.program_id(2)) * seqlen_q * d
    o_split_offset = row_idx[:, None] * d + tl.arange(0, BLOCK_K)
    o_split_ptr = tl.multiple_of(o_split_ptr, d)
    p_om = o_split_ptr + o_split_offset

    if IS_EVEN_MN & BLOCK_K == d:
        tl.store(p_om, acc_, cache_modifier=".cg")
    else:
        tl.store(p_om, acc_, mask=qmask, cache_modifier=".cg")

    lse_split_ptr = softmax_lse_ptr
    lse_split_ptr += (
        split_id * tl.num_programs(2) + tl.program_id(2)
    ) * seqlen_q + m_block * BLOCK_M

    if IS_EVEN_MN:
        tl.store(lse_split_ptr + tl.arange(0, BLOCK_M), lse, cache_modifier=".cg")
    else:
        tl.store(
            lse_split_ptr + tl.arange(0, BLOCK_M),
            lse,
            mask=row_idx < seqlen_q,
            cache_modifier=".cg",
        )


@libentry()
@triton.jit
def flash_fwd_splitkv_combine_kernel(
    out_ptr,
    lse_ptr,
    out_splits_ptr,
    lse_splits_ptr,
    head_size: tl.constexpr,
    out_split_stride,
    lse_split_stride,
    out_b_stride,
    out_s_stride,
    out_h_stride,
    n_splits,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    q_total,
    MAX_N_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    lse_splits_ptr += pid * BLOCK_M
    lse_ptr += pid * BLOCK_M
    out_splits_ptr += pid * BLOCK_M * head_size
    out_ptr += pid * BLOCK_M * head_size

    # Subtracting maximum from each of the split lse's for better numerical stability
    lse_split_offset = (
        tl.arange(0, BLOCK_M)[:, None]
        + tl.arange(0, MAX_N_SPLITS)[None, :] * lse_split_stride
    )
    lse_split_mask = (pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < q_total) & (
        tl.arange(0, MAX_N_SPLITS)[None, :] < n_splits
    )
    lse_splits = tl.load(
        lse_splits_ptr + lse_split_offset, mask=lse_split_mask, other=float("-inf")
    )
    max_lse = tl.max(lse_splits, 1)

    Zi_scaled = tl.exp(lse_splits - max_lse[:, None])
    Z_scaled = tl.sum(Zi_scaled, 1)
    Zi_Z = Zi_scaled / Z_scaled[:, None]

    lse = tl.log(Z_scaled) + max_lse
    out_mask = pid * BLOCK_M + tl.arange(0, BLOCK_M) < q_total
    tl.store(lse_ptr + tl.arange(0, BLOCK_M), lse, mask=out_mask)

    out_split_offset = (
        tl.arange(0, BLOCK_M)[:, None, None] * head_size
        + tl.arange(0, MAX_N_SPLITS)[None, :, None] * out_split_stride
        + tl.arange(0, BLOCK_K)[None, None, :]
    )
    out_split_mask = (
        (pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None, None] < q_total)
        & (tl.arange(0, MAX_N_SPLITS)[None, :, None] < n_splits)
        & (tl.arange(0, BLOCK_K)[None, None, :] < head_size)
    )
    out_splits = tl.load(
        out_splits_ptr + out_split_offset, mask=out_split_mask, other=0.0
    )
    out = tl.sum(Zi_Z[:, :, None] * out_splits, 1)
    out = out.to(out_ptr.type.element_ty)

    out_offset = tl.arange(0, BLOCK_M)[:, None] * out_s_stride + tl.arange(0, BLOCK_K)
    dmask = tl.arange(0, BLOCK_K) < head_size
    tl.store(out_ptr + out_offset, out, mask=out_mask[:, None] & dmask[None, :])


@triton.jit
def virtual_to_cache_offset(
    virtual_index,
    max_virtual_index,
    page_table_ptr,
    block_size,
    k_row_stride,
    k_page_stride,
    boundary_check: tl.constexpr = False,
):
    virtual_page_index = virtual_index // block_size
    page_offset = virtual_index % block_size
    if boundary_check:
        page_block_index = tl.load(
            page_table_ptr + virtual_page_index,
            mask=virtual_index < max_virtual_index,
            other=0,
        ).to(tl.int64)
    else:
        page_block_index = tl.load(page_table_ptr + virtual_page_index).to(tl.int64)
    return page_block_index * k_page_stride + page_offset * k_row_stride


@triton.jit
def load_from_kvcache(
    virtual_index,
    max_virtual_index,
    page_table_ptr,
    k_ptr_base,
    v_ptr_base,
    block_size,
    d: tl.constexpr,
    k_row_stride,
    k_page_stride,
    BLOCK_K: tl.constexpr,
    boundary_check: tl.constexpr = False,
):
    cache_offset = virtual_to_cache_offset(
        virtual_index,
        max_virtual_index,
        page_table_ptr,
        block_size,
        k_row_stride,
        k_page_stride,
        boundary_check,
    )
    k_offset = tl.arange(0, BLOCK_K)[:, None] + cache_offset[None, :]
    v_offset = tl.arange(0, BLOCK_K)[None, :] + cache_offset[:, None]
    if d == BLOCK_K:
        bK_mask = virtual_index[None, :] < max_virtual_index[None, :]
        bV_mask = virtual_index[:, None] < max_virtual_index[:, None]
        bK = tl.load(k_ptr_base + k_offset, mask=bK_mask, other=0.0)
        bV = tl.load(v_ptr_base + v_offset, mask=bV_mask, other=0.0)
    else:
        bK_mask = (tl.arange(0, BLOCK_K)[:, None] < d) & (
            virtual_index[None, :] < max_virtual_index[None, :]
        )
        bV_mask = (tl.arange(0, BLOCK_K)[None, :] < d) & (
            virtual_index[:, None] < max_virtual_index[:, None]
        )
        bK = tl.load(k_ptr_base + k_offset, mask=bK_mask, other=0.0)
        bV = tl.load(v_ptr_base + v_offset, mask=bV_mask, other=0.0)
    return bK, bV


@libentry()
@triton.jit(
    do_not_specialize=[
        "q_batch_stride",
        "k_batch_stride",
        "v_batch_stride",
        "o_batch_stride",
        "bk",
        "seqlen_q_rounded",
        "seqlen_k_rounded",
    ]
)
def flash_varlen_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,
    is_cu_seqlens_q: tl.constexpr,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k: tl.constexpr,
    cu_seqlens_k_ptr,
    is_seqused_k: tl.constexpr,
    seqused_k_ptr,
    # sizes
    b,
    bk,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    d: tl.constexpr,
    d_rounded: tl.constexpr,
    # scaling factors
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    q_descale_batch_stride,
    q_descale_head_stride,
    q_descale_block_stride,
    k_descale_batch_stride,
    k_descale_head_stride,
    k_descale_block_stride,
    v_descale_batch_stride,
    v_descale_head_stride,
    v_descale_block_stride: tl.constexpr,
    # dropout
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    # causal and swa
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    is_paged: tl.constexpr,
    # alibi
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    # block table
    total_q,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    k_page_stride,
    precise_p: tl.constexpr,
    # kernel params
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLIT_D: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    USE_TN: tl.constexpr = False,
    CHUNKED_QK: tl.constexpr = False,
):
    m_block = tl.program_id(0)
    bid = tl.program_id(1)
    hd = tl.program_id(2)
    hid = hd % h
    d_split = hd // h
    d_start = d_split * BLOCK_D
    # num_m_blocks = tl.cdiv(seqlen_q, BLOCK_M)

    if is_cu_seqlens_q:
        q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
        q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
        q_len = q_eos - q_bos
        # Current request's start offset in the batched Q
        q_offset = q_bos * q_row_stride
        o_offset = q_bos * o_row_stride
        lse_offset = q_bos * 1
    else:
        q_len = seqlen_q
        q_offset = bid * q_batch_stride
        o_offset = bid * o_batch_stride
        lse_offset = bid * seqlen_q

    if is_cu_seqlens_k:
        k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
        k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
        k_len_cache = k_eos - k_bos
        # k_offset = k_bos * k_row_stride
    else:
        k_len_cache = seqlen_k
        # k_offset = bid * k_batch_stride

    if is_seqused_k:
        k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
    else:
        k_len = k_len_cache

    # Noop CTA
    if m_block * BLOCK_M >= q_len:
        return

    # is_even_mn = (q_len % BLOCK_M == 0) and (k_len % BLOCK_N == 0)
    is_even_mn: tl.constexpr = False

    if is_local and window_size_left >= 0:
        n_block_min = max(
            0, (m_block * BLOCK_M + k_len - q_len - window_size_left) // BLOCK_N
        )
    else:
        n_block_min = 0

    n_block_max = tl.cdiv(k_len, BLOCK_N)
    if is_causal or (is_local and window_size_right >= 0):
        n_block_max = min(
            n_block_max,
            tl.cdiv(
                (m_block + 1) * BLOCK_M + k_len - q_len + window_size_right, BLOCK_N
            ),
        )

    if is_dropout:
        philox_seed = tl.load(philox_args).to(tl.uint64)
        philox_offset = tl.load(philox_args + 1).to(tl.uint64)

    # Locate the page table entry for the current batch element
    if is_paged:
        page_table_ptr += bid * page_table_batch_stride
    # Calculate the starting offset of q for the current head
    q_row_offset = hid * q_head_stride
    # Calculate the starting offset of k and v for the current head
    kv_hid = hid // h_hk_ratio
    k_row_offset = kv_hid * k_head_stride
    # Shift the k, v pointers to align with the current head
    k_ptr_base = k_ptr + k_row_offset
    v_ptr_base = v_ptr + k_row_offset

    gQ = tl.make_block_ptr(
        base=q_ptr + q_offset + q_row_offset,
        shape=(q_len, d),
        strides=(q_row_stride, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    bQ = tl.load(
        gQ.advance([m_block * BLOCK_M, 0]),
        boundary_check=(0, 1),
        padding_option="zero",
    )

    # Partition wide PV accumulators into D64 slices to bound register use.
    acc_ = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    if is_alibi:
        alibi_offset = bid * alibi_slopes_batch_stride + hid
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
        alibi_slope /= scale_softmax
    else:
        alibi_slope = 0.0

    MASK_N: tl.constexpr = min(BLOCK_M, BLOCK_N) if is_causal and USE_TN else BLOCK_N
    if is_local:
        n_masking_steps = min(n_block_max - n_block_min, tl.cdiv(BLOCK_M, BLOCK_N) + 1)
        full_blocks = n_block_max - n_masking_steps
        mask_block_max = n_block_max
    else:
        # Full tiles end at the first query's boundary; only the tail needs masking.
        if is_causal:
            full_keys = min(k_len, m_block * BLOCK_M + k_len - q_len + 1)
            visible_keys = min(k_len, (m_block + 1) * BLOCK_M + k_len - q_len)
        else:
            full_keys = k_len
            visible_keys = k_len
        full_blocks = max(0, full_keys) // BLOCK_N
        mask_block_max = tl.cdiv(max(0, visible_keys), MASK_N)
        n_masking_steps = mask_block_max - full_blocks * (BLOCK_N // MASK_N)

    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_block = mask_block_max - 1
    for step in tl.range(0, n_masking_steps):
        col_idx = n_block * MASK_N + tl.arange(0, MASK_N)
        if is_paged:
            bK, bV = load_from_kvcache(
                col_idx,
                k_len,
                page_table_ptr,
                k_ptr_base,
                v_ptr_base,
                block_size,
                d,
                k_row_stride,
                k_page_stride,
                BLOCK_K=BLOCK_K,
                boundary_check=True,
            )
        else:
            start_n = n_block * MASK_N
            k_ptr_seq = k_ptr_base + k_bos * k_row_stride
            v_ptr_seq = v_ptr_base + k_bos * k_row_stride
            gK = tl.make_block_ptr(
                base=k_ptr_seq,
                shape=(k_len, d),
                strides=(k_row_stride, 1),
                offsets=(start_n, 0),
                block_shape=(MASK_N, BLOCK_K),
                order=(0, 1),
            )
            gV = tl.make_block_ptr(
                base=v_ptr_seq,
                shape=(k_len, d),
                strides=(k_row_stride, 1),
                offsets=(start_n, d_start),
                block_shape=(MASK_N, BLOCK_D),
                order=(0, 1),
            )
            bK = tl.load(gK, boundary_check=(0, 1), padding_option="zero")
            bK = tl.trans(bK)
            bV = tl.load(gV, boundary_check=(0, 1), padding_option="zero")
        q_descale, k_descale, v_descale = _load_int8_block_descales(
            q_descale_ptr,
            k_descale_ptr,
            v_descale_ptr,
            q_descale_batch_stride,
            q_descale_head_stride,
            q_descale_block_stride,
            k_descale_batch_stride,
            k_descale_head_stride,
            k_descale_block_stride,
            v_descale_batch_stride,
            v_descale_head_stride,
            v_descale_block_stride,
            bid,
            hid,
            kv_hid,
            m_block * BLOCK_M,
            n_block * MASK_N,
        )
        if CHUNKED_QK:
            S = _int8_qk_tn_fragments(
                q_ptr + q_offset + q_row_offset,
                k_ptr_base + k_bos * k_row_stride,
                row_idx,
                col_idx,
                q_len,
                k_len,
                q_row_stride,
                k_row_stride,
                D=d,
                BM=BLOCK_M,
                BN=MASK_N,
                IS_BORDER=True,
            )
        elif USE_TN:
            S = tl.trans(tl.dot(tl.trans(bK), tl.trans(bQ), out_dtype=tl.int32)).to(
                tl.float32
            )
        else:
            S = tl.dot(bQ, bK, out_dtype=tl.int32).to(tl.float32)
        S *= q_descale * k_descale
        S = apply_softcap(S, softcap, is_softcap)
        S = apply_alibi(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            is_causal=is_causal,
            is_alibi=is_alibi,
            alibi_slope=alibi_slope,
        )
        S = apply_mask(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            window_size_left,
            window_size_right,
            is_even_mn=is_even_mn,
            is_causal=is_causal,
            is_local=is_local,
        )

        acc_, P, rowmax_, rowsum_, p_tile_max = softmax_rescale(
            acc_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=scale_softmax_log2,
            is_border=True,
            local_prob=False,
        )
        if is_dropout:
            P = apply_dropout(
                P,
                n_block * MASK_N,
                m_block * BLOCK_M,
                k_len,
                bid,
                hid,
                philox_seed,
                philox_offset,
                p_dropout_in_uint8_t,
                is_dropout,
                encode_dropout_in_sign_bit=False,
                NUM_HEADS=h,
                BLOCK_M=BLOCK_M,
                BLOCK_N=MASK_N,
            )
            p_tile_max = tl.max(P, 1)

        acc_ = _int8_pv_dot(
            P,
            p_tile_max,
            bV,
            acc_,
            v_descale,
            half_pv=True,
            fold_v_descale=v_descale_block_stride == 0,
            transpose_pv=USE_TN,
        )
        n_block -= 1

    for n_block in tl.range(
        full_blocks - 1,
        n_block_min - 1,
        step=-1,
        num_stages=1 if is_paged else num_stages,
    ):
        col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        if is_paged:
            bK, bV = load_from_kvcache(
                col_idx,
                k_len,
                page_table_ptr,
                k_ptr_base,
                v_ptr_base,
                block_size,
                d,
                k_row_stride,
                k_page_stride,
                BLOCK_K=BLOCK_K,
            )
        else:
            start_n = n_block * BLOCK_N
            k_ptr_seq = k_ptr_base + k_bos * k_row_stride
            v_ptr_seq = v_ptr_base + k_bos * k_row_stride
            gK = tl.make_block_ptr(
                base=k_ptr_seq,
                shape=(k_len, d),
                strides=(k_row_stride, 1),
                offsets=(start_n, 0),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(0, 1),
            )
            gV = tl.make_block_ptr(
                base=v_ptr_seq,
                shape=(k_len, d),
                strides=(k_row_stride, 1),
                # For non-paged varlen attention, load only the V slice for
                # the current D split.
                offsets=(start_n, d_start),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(0, 1),
            )
            # Sequence tiles are complete here, but head_dim may be padded.
            bK = tl.load(gK, boundary_check=(1,), padding_option="zero")
            bK = tl.trans(bK)
            bV = tl.load(gV, boundary_check=(1,), padding_option="zero")
        q_descale, k_descale, v_descale = _load_int8_block_descales(
            q_descale_ptr,
            k_descale_ptr,
            v_descale_ptr,
            q_descale_batch_stride,
            q_descale_head_stride,
            q_descale_block_stride,
            k_descale_batch_stride,
            k_descale_head_stride,
            k_descale_block_stride,
            v_descale_batch_stride,
            v_descale_head_stride,
            v_descale_block_stride,
            bid,
            hid,
            kv_hid,
            m_block * BLOCK_M,
            n_block * BLOCK_N,
        )
        # non-masking varlen QK uses the same int8 dot and block descale path.
        if CHUNKED_QK:
            S = _int8_qk_tn_fragments(
                q_ptr + q_offset + q_row_offset,
                k_ptr_base + k_bos * k_row_stride,
                row_idx,
                col_idx,
                q_len,
                k_len,
                q_row_stride,
                k_row_stride,
                D=d,
                BM=BLOCK_M,
                BN=BLOCK_N,
                IS_BORDER=False,
            )
        elif USE_TN:
            S = tl.trans(tl.dot(tl.trans(bK), tl.trans(bQ), out_dtype=tl.int32)).to(
                tl.float32
            )
        else:
            S = tl.dot(bQ, bK, out_dtype=tl.int32).to(tl.float32)
        S *= q_descale * k_descale
        S = apply_softcap(S, softcap, is_softcap)
        S = apply_alibi(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            is_causal=is_causal,
            is_alibi=is_alibi,
            alibi_slope=alibi_slope,
        )
        S = apply_mask(
            S,
            col_idx,
            row_idx,
            q_len,
            k_len,
            window_size_left,
            window_size_right,
            is_even_mn=True,
            is_causal=False,
            is_local=is_local,
        )

        acc_, P, rowmax_, rowsum_, p_tile_max = softmax_rescale(
            acc_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=scale_softmax_log2,
            is_border=is_local,
            local_prob=False,
        )
        if is_dropout:
            P = apply_dropout(
                P,
                m_block * BLOCK_M,
                n_block * BLOCK_N,
                k_len,
                bid,
                hid,
                philox_seed,
                philox_offset,
                p_dropout_in_uint8_t,
                is_dropout,
                encode_dropout_in_sign_bit=False,
                NUM_HEADS=h,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
            p_tile_max = tl.max(P, 1)
        acc_ = _int8_pv_dot(
            P,
            p_tile_max,
            bV,
            acc_,
            v_descale,
            half_pv=True,
            fold_v_descale=v_descale_block_stride == 0,
            transpose_pv=USE_TN,
        )

    # LSE
    lse = tl.where(
        rowsum_ == 0 | (rowsum_ != rowsum_),
        float("inf"),
        rowmax_ * scale_softmax + tl.log(rowsum_),
    )
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)

    if v_descale_block_stride == 0:
        # A head-wise V scale commutes with the complete weighted sum.
        final_v_descale = tl.load(
            v_descale_ptr
            + bid * v_descale_batch_stride
            + kv_hid * v_descale_head_stride
        )
    else:
        final_v_descale = 1.0
    acc_ *= (inv_sum * final_v_descale)[:, None]

    out = acc_.to(o_ptr.type.element_ty)  # noqa

    o_row_offset = hid * o_head_stride

    gO = tl.make_block_ptr(
        base=o_ptr + o_offset + o_row_offset,
        shape=(q_len, d),
        strides=(o_row_stride, 1),
        # Each varlen split-D CTA writes only its current D64 output slice.
        offsets=(0, d_start),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(gO.advance([m_block * BLOCK_M, 0]), out, boundary_check=(0, 1))

    # Write back lse
    # lse shape: [h, total_q]
    softmax_lse_ptr += hid * total_q
    lse_row_offset = lse_offset + m_block * BLOCK_M + tl.arange(0, BLOCK_M)

    tl.store(
        softmax_lse_ptr + lse_row_offset,
        lse,
        mask=(lse_row_offset < (lse_offset + q_len)) & (d_split == 0),
    )


def prune_varlen_int8_configs(configs, named_args, **kwargs):
    if named_args["d"] <= 128:
        return configs
    # A full D256 accumulator needs smaller tiles to bound registers and shared memory.
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_M"] <= 32 and config.kwargs["BLOCK_N"] <= 64
    ]


flash_varlen_fwd_tuned_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("flash_varlen_w8a8_int8"),
        key=[
            "seqlen_q",
            "seqlen_k",
            "total_q",
            "b",
            "h",
            "d",
            "q_row_stride",
            "k_row_stride",
            "q_head_stride",
            "k_head_stride",
            "is_causal",
            "v_descale_block_stride",
        ],
        prune_configs_by={"early_config_prune": prune_varlen_int8_configs},
        use_cuda_graph=True,
    )(flash_varlen_fwd_kernel.fn)
)


def CHECK_DEVICE(x):
    assert x.device.type == runtime.device.name


_HIGH_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def _normalize_dense_descale(descale, batch_size, num_heads, nblocks, device, name):
    if descale is None:
        raise ValueError(f"{name} is required for W8A8-INT8 attention")
    if descale.device != device:
        raise ValueError(f"{name} must be on the same device as q")
    if descale.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if descale.ndim == 0:
        descale = descale.reshape(1, 1, 1).expand(batch_size, num_heads, nblocks)
    elif descale.ndim == 1:
        if descale.numel() == 1:
            descale = descale.reshape(1, 1, 1).expand(batch_size, num_heads, nblocks)
        else:
            if descale.numel() != num_heads:
                raise ValueError(f"{name} 1D scale must have H elements")
            descale = descale.reshape(1, num_heads, 1).expand(
                batch_size, num_heads, nblocks
            )
    elif descale.ndim == 2:
        if descale.shape != (batch_size, num_heads):
            raise ValueError(f"{name} 2D scale must be [B, H]")
        descale = descale[:, :, None].expand(batch_size, num_heads, nblocks)
    else:
        if descale.ndim != 3 or descale.shape != (
            batch_size,
            num_heads,
            nblocks,
        ):
            raise ValueError(f"{name} must be [B, H, nblocks]")
    return descale


class fwd_params:
    __slots__ = (
        # pointers and strides
        "q_ptr",
        "k_ptr",
        "v_ptr",
        "o_ptr",
        "p_ptr",
        "softmax_lse_ptr",
        "q_row_stride",
        "k_row_stride",
        "v_row_stride",
        "q_head_stride",
        "k_head_stride",
        "v_head_stride",
        "o_row_stride",
        "o_head_stride",
        "q_batch_stride",
        "k_batch_stride",
        "v_batch_stride",
        "o_batch_stride",
        "is_cu_seqlens_q",
        "cu_seqlens_q_ptr",
        "is_cu_seqlens_k",
        "cu_seqlens_k_ptr",
        "is_seqused_k",
        "seqused_k_ptr",
        # sizes
        "b",
        "bk",
        "h",
        "hk",
        "h_hk_ratio",
        "seqlen_q",
        "seqlen_k",
        "seqlen_q_rounded",
        "seqlen_k_rounded",
        "d",
        "d_rounded",
        # scaling factors
        "is_softcap",
        "softcap",
        "scale_softmax",
        "scale_softmax_log2",
        "q_descale_ptr",
        "k_descale_ptr",
        "v_descale_ptr",
        "q_descale_batch_stride",
        "q_descale_head_stride",
        "q_descale_block_stride",
        "k_descale_batch_stride",
        "k_descale_head_stride",
        "k_descale_block_stride",
        "v_descale_batch_stride",
        "v_descale_head_stride",
        "v_descale_block_stride",
        # dropout
        "is_dropout",
        "p_dropout",
        "rp_dropout",
        "p_dropout_in_uint8_t",
        "philox_args",
        "return_softmax",
        # masking
        "is_causal",
        "is_local",
        "window_size_left",
        "window_size_right",
        "seqlenq_ngroups_swapped",
        "is_paged",
        # alibi
        "is_alibi",
        "alibi_slopes_ptr",
        "alibi_slopes_batch_stride",
        # block table
        "total_q",
        "page_table_ptr",
        "page_table_batch_stride",
        "block_size",
        "k_page_stride",
        "precise_p",
    )

    def __init__(
        self,
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        p_ptr,
        softmax_lse_ptr,
        q_row_stride,
        k_row_stride,
        v_row_stride,
        q_head_stride,
        k_head_stride,
        v_head_stride,
        o_row_stride,
        o_head_stride,
        q_batch_stride,
        k_batch_stride,
        v_batch_stride,
        o_batch_stride,
        is_cu_seqlens_q,
        cu_seqlens_q_ptr,
        is_cu_seqlens_k,
        cu_seqlens_k_ptr,
        is_seqused_k,
        seqused_k_ptr,
        # sizes
        b,
        bk,
        h,
        hk,
        h_hk_ratio,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        d,
        d_rounded,
        # scaling factors
        is_softcap,
        softcap,
        scale_softmax,
        scale_softmax_log2,
        q_descale_ptr,
        k_descale_ptr,
        v_descale_ptr,
        q_descale_batch_stride,
        q_descale_head_stride,
        q_descale_block_stride,
        k_descale_batch_stride,
        k_descale_head_stride,
        k_descale_block_stride,
        v_descale_batch_stride,
        v_descale_head_stride,
        v_descale_block_stride,
        # dropout
        is_dropout,
        p_dropout,
        rp_dropout,
        p_dropout_in_uint8_t,
        philox_args,
        return_softmax,
        # masking
        is_causal,
        is_local,
        window_size_left,
        window_size_right,
        seqlenq_ngroups_swapped,
        is_paged,
        # alibi
        is_alibi,
        alibi_slopes_ptr,
        alibi_slopes_batch_stride,
        # block table
        total_q,
        page_table_ptr,
        page_table_batch_stride,
        block_size,
        k_page_stride,
        precise_p,
    ):
        self.q_ptr = q_ptr
        self.k_ptr = k_ptr
        self.v_ptr = v_ptr
        self.o_ptr = o_ptr
        self.p_ptr = p_ptr
        self.softmax_lse_ptr = softmax_lse_ptr
        self.q_row_stride = q_row_stride
        self.k_row_stride = k_row_stride
        self.v_row_stride = v_row_stride
        self.q_head_stride = q_head_stride
        self.k_head_stride = k_head_stride
        self.v_head_stride = v_head_stride
        self.o_row_stride = o_row_stride
        self.o_head_stride = o_head_stride
        self.q_batch_stride = q_batch_stride
        self.k_batch_stride = k_batch_stride
        self.v_batch_stride = v_batch_stride
        self.o_batch_stride = o_batch_stride
        self.is_cu_seqlens_q = is_cu_seqlens_q
        self.cu_seqlens_q_ptr = cu_seqlens_q_ptr
        self.is_cu_seqlens_k = is_cu_seqlens_k
        self.cu_seqlens_k_ptr = cu_seqlens_k_ptr
        self.is_seqused_k = is_seqused_k
        self.seqused_k_ptr = seqused_k_ptr
        # sizes
        self.b = b
        self.bk = bk
        self.h = h
        self.hk = hk
        self.h_hk_ratio = h_hk_ratio
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.seqlen_q_rounded = seqlen_q_rounded
        self.seqlen_k_rounded = seqlen_k_rounded
        self.d = d
        self.d_rounded = d_rounded
        # scaling factors
        self.is_softcap = is_softcap
        self.softcap = softcap
        self.scale_softmax = scale_softmax
        self.scale_softmax_log2 = scale_softmax_log2
        self.q_descale_ptr = q_descale_ptr
        self.k_descale_ptr = k_descale_ptr
        self.v_descale_ptr = v_descale_ptr
        self.q_descale_batch_stride = q_descale_batch_stride
        self.q_descale_head_stride = q_descale_head_stride
        self.q_descale_block_stride = q_descale_block_stride
        self.k_descale_batch_stride = k_descale_batch_stride
        self.k_descale_head_stride = k_descale_head_stride
        self.k_descale_block_stride = k_descale_block_stride
        self.v_descale_batch_stride = v_descale_batch_stride
        self.v_descale_head_stride = v_descale_head_stride
        self.v_descale_block_stride = v_descale_block_stride
        # dropout
        self.is_dropout = is_dropout
        self.p_dropout = p_dropout
        self.rp_dropout = rp_dropout
        self.p_dropout_in_uint8_t = p_dropout_in_uint8_t
        self.philox_args = philox_args
        self.return_softmax = return_softmax
        # masking
        self.is_causal = is_causal
        self.is_local = is_local
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped
        self.is_paged = is_paged
        # alibi
        self.is_alibi = is_alibi
        self.alibi_slopes_ptr = alibi_slopes_ptr
        self.alibi_slopes_batch_stride = alibi_slopes_batch_stride
        # block table
        self.total_q = total_q
        self.page_table_ptr = page_table_ptr
        self.page_table_batch_stride = page_table_batch_stride
        self.block_size = block_size
        self.k_page_stride = k_page_stride
        self.precise_p = precise_p

    def args(self):
        return tuple(getattr(self, k) for k in self.__slots__)


def _get_varlen_fwd_config(
    args,
    head_size,
    use_varlen_split_d,
    is_paged,
    is_causal,
    window_size_left,
    window_size_right,
    is_alibi,
    is_softcap,
    is_dropout,
    total_q,
    batch_size,
    num_heads,
    max_seqlen_q,
    max_seqlen_k,
):
    total_rows = total_q * num_heads
    num_sms = torch_device_fn.get_device_properties(
        runtime.device.name
    ).multi_processor_count
    avg_rows_per_sm = total_rows / num_sms
    avg_rows_per_batch = total_q / batch_size
    avg_rows_per_cta = min(avg_rows_per_batch, avg_rows_per_sm)

    if avg_rows_per_cta > 64:
        varlen_fwd_config_str = "mha_block_128"
    elif avg_rows_per_cta > 32:
        varlen_fwd_config_str = "mha_block_64"
    elif avg_rows_per_cta > 16:
        varlen_fwd_config_str = "mha_block_32"
    else:
        varlen_fwd_config_str = "mha_block_16"
    if runtime.device.vendor_name == "mthreads":
        varlen_fwd_config_str = "mha_block_32"

    cfg = runtime.get_heuristic_config(varlen_fwd_config_str)
    cfg_params = {
        "BLOCK_M": cfg["BLOCK_M"](args),
        "BLOCK_N": cfg["BLOCK_N"](args),
        "BLOCK_K": max(32, triton.next_power_of_2(head_size)),
        "BLOCK_D": (
            64 if use_varlen_split_d else max(32, triton.next_power_of_2(head_size))
        ),
        "SPLIT_D": use_varlen_split_d,
        "num_warps": cfg["num_warps"](args),
        "num_stages": 1 if not is_paged else cfg["num_stages"](args),
    }

    cfg_params["BLOCK_N"] = max(32, cfg_params["BLOCK_N"])

    if head_size not in (64, 128) and not is_paged:
        cfg_params.update(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=1)

    if not is_paged:
        if (
            head_size in (64, 128)
            and not is_causal
            and window_size_left == -1
            and window_size_right == -1
            and max_seqlen_k >= 4096
        ):
            cfg_params.update(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=1)
        else:
            cfg_params.update(BLOCK_N=32, num_warps=4, num_stages=1)

    is_standard_attention = (
        not is_causal and window_size_left == -1 and window_size_right == -1
    ) or (is_causal and window_size_left == -1 and window_size_right == 0)
    is_hopper = (
        runtime.device.vendor_name == "nvidia" and get_device_capability()[0] == 9
    )
    if (
        is_hopper
        and not is_paged
        and is_standard_attention
        and not is_alibi
        and not is_softcap
        and not is_dropout
        and head_size in (64, 128)
    ):
        cfg_params.update(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128 if head_size == 64 and not is_causal else 64,
                "num_warps": 8,
                "num_stages": 1,
            }
        )

        use_small_q_config = (head_size == 64 and max_seqlen_q <= 64) or (
            head_size == 128 and max_seqlen_q <= 1024
        )
        if use_small_q_config:
            cfg_params.update(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "num_warps": 4,
                    "num_stages": 1,
                }
            )

        if head_size == 64 and is_causal:
            if max_seqlen_q <= 512:
                cfg_params.update(BLOCK_M=64, BLOCK_N=64, num_warps=4)
            else:
                cfg_params.update(BLOCK_N=128, num_warps=4, num_stages=3)
        elif head_size == 128 and not is_causal and max_seqlen_q > 1024:
            cfg_params["num_stages"] = 3

    use_long_d64_tile = (
        head_size == 64
        and not is_causal
        and max_seqlen_q >= 1024
        and max_seqlen_k >= 1024
    )
    cfg_params["USE_TN"] = (
        runtime.device.vendor_name == "metax"
        and not is_paged
        and (
            head_size in (64, 128, 192)
            or (
                head_size == 96
                and not is_causal
                and max_seqlen_q <= 1024
                and max_seqlen_k <= 1024
            )
            or (head_size == 256 and not is_causal)
        )
        and not (
            head_size == 64
            and max_seqlen_q >= 1024
            and total_q != batch_size * max_seqlen_q
            and not use_long_d64_tile
        )
        and is_standard_attention
        and not is_alibi
        and not is_softcap
        and not is_dropout
        and max_seqlen_q >= (4096 if head_size == 256 else 128)
        and max_seqlen_k >= (4096 if head_size == 256 else 512)
    )
    if cfg_params["USE_TN"]:
        cfg_params.update(
            BLOCK_M=32,
            BLOCK_N=64 if head_size == 256 else 128,
            BLOCK_D=128 if head_size in (96, 192) else head_size,
            SPLIT_D=head_size == 192,
            num_warps=4,
            num_stages=1,
        )
    cfg_params["CHUNKED_QK"] = cfg_params["USE_TN"] and head_size == 192
    if cfg_params["CHUNKED_QK"]:
        cfg_params["BLOCK_K"] = 64
    if cfg_params["USE_TN"]:
        if use_long_d64_tile:
            cfg_params.update(BLOCK_M=64, BLOCK_N=128, pipeline="basic", num_stages=1)
        elif (
            head_size == 128
            and is_causal
            and max_seqlen_q >= 1024
            and max_seqlen_k >= 1024
        ):
            cfg_params.update(
                BLOCK_M=64,
                BLOCK_N=64,
                pipeline="cpasync",
                scenario="noaddropt;disable_int8_opt",
                num_stages=2,
            )
    if (
        runtime.device.vendor_name == "metax"
        and not is_paged
        and is_standard_attention
        and not is_alibi
        and not is_softcap
        and not is_dropout
        and head_size in (96, 256)
    ):
        # Full-D PV avoids repeating QK for each output slice on these dimensions.
        cfg_params.update(
            BLOCK_M=32,
            BLOCK_N=64,
            BLOCK_K=triton.next_power_of_2(head_size),
            BLOCK_D=triton.next_power_of_2(head_size),
            SPLIT_D=False,
            USE_TN=True,
            CHUNKED_QK=False,
            num_warps=4,
            num_stages=1,
        )
    elif cfg_params["USE_TN"] and head_size == 192 and max_seqlen_q >= 1024:
        cfg_params.update(BLOCK_M=32, BLOCK_N=64, BLOCK_D=256, SPLIT_D=False)
    else:
        return cfg_params
    return cfg_params


@triton.jit
def _pack_int8_kv_heads(
    K,
    V,
    KO,
    VO,
    T,
    D: tl.constexpr,
    KR: tl.constexpr,
    KH: tl.constexpr,
    VR: tl.constexpr,
    VH: tl.constexpr,
    B: tl.constexpr,
):
    h = tl.program_id(1).to(tl.int64)
    x = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    n, d = x // D, x % D
    dst = h * T * D + x
    k = tl.load(K + n * KR + h * KH + d, n < T, 0)
    tl.store(KO + dst, k, n < T)
    v = tl.load(V + n * VR + h * VH + d, n < T, 0)
    tl.store(VO + dst, v, n < T)


def mha_varlan_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    leftpad_k,
    page_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    p_dropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen,
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    CHECK_DEVICE(q), CHECK_DEVICE(k), CHECK_DEVICE(v)
    q_device = q.device
    assert q.dtype == torch.int8, "W8A8 FlashAttention expects q to be int8"
    assert k.dtype == torch.int8, "W8A8 FlashAttention expects k to be int8"
    assert v.dtype == torch.int8, "W8A8 FlashAttention expects v to be int8"
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"

    assert cu_seqlens_q.dtype == torch.int32
    assert cu_seqlens_q.is_contiguous()

    assert cu_seqlens_k.dtype == torch.int32
    assert cu_seqlens_k.is_contiguous()

    is_paged = page_table is not None
    if not is_paged:
        page_table = torch.empty((0, 0), device=q_device, dtype=torch.int32)

    total_q, num_heads, head_size = q.size()
    num_heads_k = k.size(2) if is_paged else k.size(1)
    batch_size = cu_seqlens_q.numel() - 1
    assert batch_size > 0, "batch_size must be positive"
    block_size = k.size(1) if is_paged else 1
    num_pages = k.size(0) if is_paged else 0
    k_batch_size = num_pages
    page_table_batch_stride = page_table.stride(0)
    k_batch_stride = k.stride(0)
    v_batch_stride = v.stride(0)

    assert k.size() == v.size()
    assert cu_seqlens_q.size() == (batch_size + 1,)
    assert cu_seqlens_k.size() == (batch_size + 1,)

    # Check output shape
    if out is not None:
        assert out.stride(-1) == 1
        assert out.dtype in _HIGH_PRECISION_DTYPES
        assert out.size() == (total_q, num_heads, head_size)

    if seqused_k is not None:
        assert seqused_k.is_contiguous()
        assert seqused_k.size() == (batch_size,)

    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False

    if is_causal:
        window_size_right = 0

    # check disable swa
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    if window_size_right >= max_seqlen_q:
        window_size_right = -1

    is_local = window_size_left >= 0 or (window_size_right >= 0 and not is_causal)

    seqlenq_ngroups_swapped = (
        max_seqlen_q == 1
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
        and p_dropout == 0
    )
    q_groups = num_heads // num_heads_k
    if seqlenq_ngroups_swapped:
        logger.debug("Swapping query groups and sequence dimensions")
        q = (
            q.reshape((batch_size, num_heads_k, q_groups, head_size))
            .transpose(1, 2)
            .reshape(batch_size * q_groups, num_heads_k, head_size)
        )
        max_seqlen_q = q_groups
        num_heads = num_heads_k
        cu_seqlens_q = None
        q_batch_stride = q.stride(0) * max_seqlen_q
        k_batch_stride = k.stride(0)
        v_batch_stride = v.stride(0)
        # o_batch_stride = out.stride(0) * max_seqlen_q
    else:
        q_batch_stride = 0
        k_batch_stride = 0
        v_batch_stride = 0
        o_batch_stride = 0

    total_q = q.size(0)

    assert leftpad_k is None, "leftpad_k is not supported."
    assert (
        head_size <= 256
    ), "FlashAttention forward only supports head dimension at most 256"
    assert (
        head_size % 8 == 0
    ), "head_size must be a multiple of 8, this is ensured by padding!"
    assert (
        num_heads % num_heads_k == 0
    ), "Number of heads in key/value must divide number of heads in query"

    assert q.shape == (total_q, num_heads, head_size)
    if is_paged:
        assert k.shape == (num_pages, block_size, num_heads_k, head_size)
        assert v.shape == (num_pages, block_size, num_heads_k, head_size)
    assert k.stride() == v.stride()

    if softcap > 0.0:
        assert p_dropout == 0, "dropout is not supported if softcap is used."

    round_multiple = lambda x, m: (x + m - 1) // m * m
    head_size_rounded = round_multiple(head_size, 32) if head_size <= 192 else 256
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = round_multiple(max_seqlen_k, 32)

    M_LOG2E = 1.4426950408889634074
    if softcap > 0.0:
        is_softcap = True
        adjusted_scale_softmax = softcap
        adjusted_softcap = softmax_scale / softcap
        adjusted_scale_softmax_log2e = softcap * M_LOG2E
    else:
        is_softcap = False
        adjusted_softcap = 0.0
        adjusted_scale_softmax = softmax_scale
        adjusted_scale_softmax_log2e = softmax_scale * M_LOG2E

    # Set alibi params
    if alibi_slopes is not None:
        assert alibi_slopes.device == q_device
        assert alibi_slopes.dtype in (torch.float,)
        assert alibi_slopes.stride(-1) == 1
        assert alibi_slopes.shape == (num_heads,) or alibi_slopes.shape == (
            batch_size,
            num_heads,
        )
        alibi_slopes_batch_stride = (
            alibi_slopes.stride(0) if alibi_slopes.ndim == 2 else 0
        )
        is_alibi = True
    else:
        alibi_slopes_batch_stride = 0
        is_alibi = False

    # Prepare params to kernel
    with torch_device_fn.device(q_device):
        if out is not None:
            out_ = out
            if seqlenq_ngroups_swapped:
                out = torch.empty_like(q, dtype=torch.bfloat16)
        else:
            out_ = None
            out = torch.empty_like(q, dtype=torch.bfloat16)

        if seqlenq_ngroups_swapped:
            o_batch_stride = out.stride(0) * max_seqlen_q

        lse = torch.empty((num_heads, total_q), dtype=torch.float, device=q_device)

        if p_dropout > 0:
            is_dropout = True
            increment = batch_size * num_heads * 32
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            philox_args = torch.tensor(
                [philox_seed, philox_offset], dtype=torch.int64, device=q_device
            )
        else:
            is_dropout = False
            philox_args = torch.empty((2,), dtype=torch.int64, device=q_device)

        p_dropout = 1 - p_dropout
        p_dropout_in_uint8_t = math.floor(p_dropout * 255.0)
        rp_dropout = 1.0 / p_dropout

        if return_softmax:
            assert is_dropout, "Only supported with non-zero dropout."
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                device=q_device,
            )
        else:
            p = torch.empty((), device=q_device)

        if zero_tensors:
            out.zero_()
            lse.fill_(float("-inf"))

        q_nblocks = triton.cdiv(max_seqlen_q, 128)
        k_nblocks = triton.cdiv(max_seqlen_k, 128)
        q_descale = _normalize_dense_descale(
            q_descale, batch_size, num_heads, q_nblocks, q_device, "q_descale"
        )
        k_descale = _normalize_dense_descale(
            k_descale, batch_size, num_heads_k, k_nblocks, q_device, "k_descale"
        )
        v_descale = _normalize_dense_descale(
            v_descale, batch_size, num_heads_k, k_nblocks, q_device, "v_descale"
        )

        if (
            runtime.device.vendor_name == "metax"
            and head_size == 64
            and not is_paged
            and not is_alibi
            and not is_softcap
            and not is_dropout
            and not is_causal
            and window_size_left == -1
            and window_size_right == -1
            and max_seqlen_q >= 2048
            and max_seqlen_k >= 2048
            and total_q * num_heads >= 65536
            and cu_seqlens_q is not None
            and cu_seqlens_k is not None
            and seqused_k is None
            and not seqlenq_ngroups_swapped
        ):
            packed_v = torch.empty_strided(
                v.shape,
                (head_size, v.shape[0] * head_size, 1),
                dtype=v.dtype,
                device=v.device,
            )
            packed_k = torch.empty_strided(
                k.shape,
                (head_size, k.shape[0] * head_size, 1),
                dtype=k.dtype,
                device=k.device,
            )
            _pack_int8_kv_heads[(triton.cdiv(k.shape[0] * head_size, 2048), num_heads)](
                k,
                v,
                packed_k,
                packed_v,
                k.shape[0],
                head_size,
                k.stride(0),
                k.stride(1),
                v.stride(0),
                v.stride(1),
                B=2048,
                num_warps=4,
            )
            k, v = packed_k, packed_v

        params = fwd_params(
            q,  # q_ptr,
            k,  # k_ptr,
            v,  # v_ptr,
            out,  # o_ptr,
            p,  # p_ptr,
            lse,  # softmax_lse_ptr,
            q.stride(-3),  # q_row_stride,
            k.stride(-3),  # k_row_stride,
            v.stride(-3),  # v_row_stride,
            q.stride(-2),  # q_head_stride,
            k.stride(-2),  # k_head_stride,
            v.stride(-2),  # v_head_stride,
            out.stride(-3),  # o_row_stride,
            out.stride(-2),  # o_head_stride,
            q_batch_stride,  # q_batch_stride,
            k_batch_stride,  # k_batch_stride,
            v_batch_stride,  # v_batch_stride,
            o_batch_stride,  # o_batch_stride,
            cu_seqlens_q is not None,  # is_cu_seqlens_q,
            cu_seqlens_q,  # cu_seqlens_q_ptr,
            seqused_k is None,  # is_cu_seqlens_k,
            cu_seqlens_k,  # cu_seqlens_k_ptr,
            seqused_k is not None,  # is_seqused_k,
            seqused_k,  # seqused_k_ptr,
            # sizes
            batch_size,  # b,
            k_batch_size,  # bk,
            num_heads,  # h,
            num_heads_k,  # hk,
            num_heads // num_heads_k,  # h_hk_ratio,
            max_seqlen_q,  # seqlen_q,
            max_seqlen_k,  # seqlen_k,
            seqlen_q_rounded,  # seqlen_q_rounded,
            seqlen_k_rounded,  # seqlen_k_rounded,
            head_size,  # d,
            head_size_rounded,  # d_rounded,
            # scaling factors
            is_softcap,
            adjusted_softcap,  # softcap,
            adjusted_scale_softmax,  # scale_softmax,
            adjusted_scale_softmax_log2e,  # scale_softmax_log2,
            q_descale,  # q_descale_ptr,
            k_descale,  # k_descale_ptr,
            v_descale,  # v_descale_ptr,
            q_descale.stride(0),  # q_descale_batch_stride,
            q_descale.stride(1),  # q_descale_head_stride,
            q_descale.stride(2),  # q_descale_block_stride,
            k_descale.stride(0),  # k_descale_batch_stride,
            k_descale.stride(1),  # k_descale_head_stride,
            k_descale.stride(2),  # k_descale_block_stride,
            v_descale.stride(0),  # v_descale_batch_stride,
            v_descale.stride(1),  # v_descale_head_stride,
            v_descale.stride(2),  # v_descale_block_stride,
            # dropout
            is_dropout,
            p_dropout,
            rp_dropout,
            p_dropout_in_uint8_t,
            philox_args,
            return_softmax,
            # causal and swa
            is_causal,  # is_causal,
            is_local,  # is_local,
            window_size_left,  # window_size_left,
            window_size_right,  # window_size_right,
            seqlenq_ngroups_swapped,  # seqlenq_ngroups_swapped,
            is_paged,
            # alibi
            is_alibi,  #
            alibi_slopes,  # alibi_slopes_ptr,
            alibi_slopes_batch_stride,  # alibi_slopes_batch_stride,
            # block table params
            total_q,  # total_q,
            page_table,  # page_table_ptr,
            page_table_batch_stride,  # page_table_batch_stride,
            block_size,  # block_size,
            k.stride(0) if is_paged else 0,  # k_page_stride,
            max_seqlen_k < 512,  # precise_p,
        )

        if runtime.device.vendor_name == "iluvatar":
            params.k_ptr = k.view(k.shape[0], k.shape[1], -1)
            params.v_ptr = v.view(v.shape[0], v.shape[1], -1)
        logger.debug("kernel: flash_varlen_fwd")
        # Use D64 output slices for non-paged dimensions above 64. The
        # paged-cache loader continues to load the full D dimension.
        use_varlen_split_d = head_size > 64 and not is_paged
        args = tuple(getattr(params, k) for k in params.__slots__)

        cfg_params = _get_varlen_fwd_config(
            args,
            head_size,
            use_varlen_split_d,
            is_paged,
            is_causal,
            window_size_left,
            window_size_right,
            is_alibi,
            is_softcap,
            is_dropout,
            total_q,
            batch_size,
            num_heads,
            max_seqlen_q,
            max_seqlen_k,
        )
        num_d_splits = triton.cdiv(head_size, cfg_params["BLOCK_D"])
        if cfg_params["USE_TN"] and not cfg_params["SPLIT_D"]:
            grid = lambda meta: (
                triton.cdiv(max_seqlen_q, meta["BLOCK_M"]),
                batch_size,
                num_heads,
            )
            kernel = flash_varlen_fwd_tuned_kernel[grid]
            for name in ("BLOCK_M", "BLOCK_N", "num_warps", "num_stages"):
                cfg_params.pop(name)
        else:
            grid = (
                triton.cdiv(max_seqlen_q, cfg_params["BLOCK_M"]),
                batch_size,
                num_heads * num_d_splits,
            )
            kernel = flash_varlen_fwd_kernel[grid]

        logger.debug("Running flash_varlen_fwd_kernel with config: %s", cfg_params)
        kernel(*args, **cfg_params)

        if seqlenq_ngroups_swapped:
            out = out.reshape(
                batch_size, max_seqlen_q, num_heads_k, head_size
            ).transpose(1, 2)
            if out_ is not None:
                out_.view(batch_size, num_heads_k, max_seqlen_q, head_size).copy_(out)
                out = out_
            else:
                out = out.reshape(batch_size, num_heads_k * max_seqlen_q, head_size)
            lse = lse.reshape(num_heads_k, batch_size, max_seqlen_q)
            lse = lse.reshape(num_heads_k * max_seqlen_q, batch_size)

        unused = torch.empty((), dtype=torch.int64, device=q_device)
    return out, q, k, v, lse, philox_args, unused, p


def mha_fwd(
    q,
    k,
    v,
    out,
    alibi_slopes,
    p_dropout,
    softmax_scale,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    disable_splitkv=False,
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    CHECK_DEVICE(q), CHECK_DEVICE(k), CHECK_DEVICE(v)
    q_device = q.device
    assert q.dtype == torch.int8, "W8A8 FlashAttention expects q to be int8"
    assert k.dtype == torch.int8, "W8A8 FlashAttention expects k to be int8"
    assert v.dtype == torch.int8, "W8A8 FlashAttention expects v to be int8"
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    batch_size, seqlen_q, num_heads, head_size = q.size()
    _, seqlen_k, num_heads_k, _ = k.size()

    # Check output shape
    if out is not None:
        assert out.stride(-1) == 1
        # Output stays high precision instead of inheriting the INT8 V dtype.
        assert out.dtype in _HIGH_PRECISION_DTYPES
        assert out.size() == (batch_size, seqlen_q, num_heads, head_size)
        CHECK_DEVICE(out)

    assert (
        head_size % 8 == 0
    ), "head_size must be a multiple of 8, this is ensured by padding!"
    assert (
        num_heads % num_heads_k == 0
    ), "Number of heads in key/value must divide number of heads in query"
    if window_size_left >= seqlen_k:
        window_size_left = -1
    if window_size_right >= seqlen_q:
        window_size_right = -1
    if seqlen_q == 1 and alibi_slopes is None:
        is_causal = False
    if is_causal:
        window_size_right = 0

    is_causal = window_size_left < 0 and window_size_right == 0
    is_local = (window_size_left >= 0 or window_size_right >= 0) and not is_causal

    seqlenq_ngroups_swapped = (
        seqlen_q == 1
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
        and p_dropout == 0
    )
    q_groups = num_heads // num_heads_k

    if seqlenq_ngroups_swapped:
        logger.debug("q_kg swapped.")
        q = q.reshape(batch_size, num_heads_k, q_groups, head_size).transpose(1, 2)
        seqlen_q = q_groups
        num_heads = num_heads_k

    round_multiple = lambda x, m: (x + m - 1) // m * m
    head_size_rounded = round_multiple(head_size, 32)
    seqlen_q_rounded = round_multiple(seqlen_q, 128)
    seqlen_k_rounded = round_multiple(seqlen_k, 32)

    assert (
        head_size <= 256
    ), "FlashAttention forward only supports head dimension at most 256"
    assert head_size == head_size_rounded, "head_size must be rounded to 32"

    def splits_heuristic(num_tasks, num_sms, n_blocks):
        n_waves = triton.cdiv(num_tasks, num_sms)
        eff = (num_tasks / num_sms) / n_waves
        if eff > 0.8 or n_waves > 1:
            return 1

        min_blocks_per_split = 2
        best_splits = min(
            triton.cdiv(n_blocks, min_blocks_per_split),
            int(math.floor(1.0 / eff)),
            num_sms,
        )

        return best_splits

    with torch_device_fn.device(q_device):
        # Set softmax params
        lse = torch.empty(
            (batch_size, num_heads, seqlen_q), dtype=torch.float, device=q_device
        )

        if out is not None:
            if seqlenq_ngroups_swapped:
                out = out.reshape(
                    batch_size, num_heads_k, q_groups, head_size
                ).transpose(1, 2)
        else:
            out = torch.empty_like(q, dtype=torch.bfloat16)

        # Set dropout params
        if p_dropout > 0:
            is_dropout = True
            increment = batch_size * num_heads * 32
            philox_seed, philox_offset = philox_backend_seed_offset(increment)
            philox_args = torch.tensor(
                [philox_seed, philox_offset], dtype=torch.int64, device=q_device
            )
        else:
            is_dropout = False
            philox_args = torch.empty((2,), dtype=torch.int64, device=q_device)

        p_dropout = 1 - p_dropout
        p_dropout_in_uint8_t = math.floor(p_dropout * 255.0)
        rp_dropout = 1.0 / p_dropout

        if return_softmax:
            assert is_dropout, "Only supported with non-zero dropout."
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                device=q_device,
            )
        else:
            p = torch.empty((), device=q_device)

        q_nblocks = triton.cdiv(seqlen_q, 128)
        k_nblocks = triton.cdiv(seqlen_k, 128)
        # Per-block descale tensors are required by the INT8 kernel contract.
        q_descale = _normalize_dense_descale(
            q_descale, batch_size, num_heads, q_nblocks, q_device, "q_descale"
        )
        k_descale = _normalize_dense_descale(
            k_descale, batch_size, num_heads_k, k_nblocks, q_device, "k_descale"
        )
        v_descale = _normalize_dense_descale(
            v_descale, batch_size, num_heads_k, k_nblocks, q_device, "v_descale"
        )

        M_LOG2E = 1.4426950408889634074
        if softcap > 0.0:
            is_softcap = True
            adjusted_scale_softmax = softcap
            adjusted_softcap = softmax_scale / softcap
            adjusted_scale_softmax_log2e = softcap * M_LOG2E
        else:
            is_softcap = False
            adjusted_softcap = 0.0
            adjusted_scale_softmax = softmax_scale
            adjusted_scale_softmax_log2e = softmax_scale * M_LOG2E

        if alibi_slopes is not None:
            assert alibi_slopes.device == q_device
            assert alibi_slopes.dtype in (torch.float,)
            assert alibi_slopes.stride(-1) == 1
            assert alibi_slopes.shape == (num_heads,) or alibi_slopes.shape == (
                batch_size,
                num_heads,
            )
            alibi_slopes_batch_stride = (
                alibi_slopes.stride(0) if alibi_slopes.ndim == 2 else 0
            )
            is_alibi = True
        else:
            alibi_slopes_batch_stride = 0
            is_alibi = False

        # ONLY EVEN_K IS SUPPORTED
        assert head_size == head_size_rounded

        # Do kernel dispatching
        def dispatch(B, H, Q, K, D, params):
            num_sms = torch_device_fn.get_device_properties(
                "cuda"
            ).multi_processor_count
            use_split_d = D == 128
            # For short sequences, split-KV combine and temporary-tensor overhead
            # usually outweigh the benefit, so S <= 512 uses the dense path.
            disable_splitkv1 = disable_splitkv or seqlen_q <= 512

            # Try splitkv
            if (
                (not use_split_d)
                and not is_dropout
                and not is_local
                and not disable_splitkv1
            ):
                BM = block_m_splitkv_heuristic(D)
                n_tasks = B * H * triton.cdiv(seqlen_q, BM)
                BN = block_n_splitkv_heuristic(D)
                n_blocks = triton.cdiv(seqlen_k, BN)
                n_splits = splits_heuristic(n_tasks, num_sms, n_blocks)

                if n_splits > 1:
                    logger.debug("kernel: flash_fwd_splitkv")
                    lse_splits = torch.empty(
                        (n_splits, B, H, Q), dtype=torch.float, device=q_device
                    )
                    out_splits = torch.empty(
                        (n_splits, B, H, Q, D), dtype=torch.float, device=q_device
                    )
                    grid = lambda args: (
                        triton.cdiv(Q, args["BLOCK_M"]),
                        n_splits,
                        B * H,
                    )
                    splitkv_kernel = flash_fwd_splitkv_kernel[grid]
                    params.o_ptr = out_splits
                    params.softmax_lse_ptr = lse_splits
                    extra_args = {"blocks_per_split": triton.cdiv(n_blocks, n_splits)}
                    kernel = splitkv_kernel(*params.args(), **extra_args)

                    if D >= 128:
                        BLOCK_M = 4
                    elif D >= 64:
                        BLOCK_M = 8
                    else:
                        BLOCK_M = 16
                    BLOCK_K = triton.next_power_of_2(D)
                    grid = lambda args: (triton.cdiv(B * H * Q, BLOCK_M),)
                    combine_kernel = flash_fwd_splitkv_combine_kernel[grid]
                    combine_args = {
                        "out_ptr": out,
                        "lse_ptr": lse,
                        "head_size": head_size,
                        "out_split_stride": out_splits.stride(0),
                        "lse_split_stride": lse_splits.stride(0),
                        "out_b_stride": out.stride(0),
                        "out_s_stride": out.stride(-3),
                        "out_h_stride": out.stride(-1),
                        "out_splits_ptr": out_splits,
                        "lse_splits_ptr": lse_splits,
                        "n_splits": n_splits,
                        "BLOCK_M": BLOCK_M,
                        "BLOCK_K": BLOCK_K,
                        "q_total": B * H * Q,
                        "MAX_N_SPLITS": triton.next_power_of_2(n_splits),
                    }
                    combine_kernel(**combine_args)
                    return kernel

            # Last option: flash_fwd
            logger.debug("kernel: flash_fwd")
            grid = lambda args: (
                triton.cdiv(Q, args["BLOCK_M"]),
                H * B,
                2 if use_split_d else 1,
            )
            kernel = flash_fwd_kernel[grid]
            # For D128 split-D, QK still uses the full BLOCK_K with head_dim=128.
            # BLOCK_D=64 applies only to PV and output stores; D64 or non-split-D
            # retains the original BLOCK_K.
            extra_args = {
                "BLOCK_D": 64 if use_split_d else triton.next_power_of_2(D),
                "SPLIT_D": use_split_d,
            }
            kernel = kernel(*params.args(), **extra_args)
            return kernel

        if _debug:
            p = torch.empty(
                (batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded),
                dtype=torch.float32,
                device=q_device,
            )
            return_softmax = True

        params = fwd_params(
            q,  # q_ptr,
            k,  # k_ptr,
            v,  # v_ptr,
            out,  # o_ptr,
            p,  # p_ptr,
            lse,  # softmax_lse_ptr,
            q.stride(-3),  # q_row_stride,
            k.stride(-3),  # k_row_stride,
            v.stride(-3),  # v_row_stride,
            q.stride(-2),  # q_head_stride,
            k.stride(-2),  # k_head_stride,
            v.stride(-2),  # v_head_stride,
            out.stride(-3),  # o_row_stride,
            out.stride(-2),  # o_head_stride,
            q.stride(0),  # q_batch_stride,
            k.stride(0),  # k_batch_stride,
            v.stride(0),  # v_batch_stride,
            out.stride(0),  # o_batch_stride,
            False,  # is_cu_seqlens_q,
            None,  # cu_seqlens_q_ptr,
            False,  # is_cu_seqlens_k,
            None,  # cu_seqlens_k_ptr,
            False,  # is_seqused_k,
            None,  # seqused_k_ptr,
            # sizes
            batch_size,  # b,
            0,  # bk,
            num_heads,  # h,
            num_heads_k,  # hk,
            num_heads // num_heads_k,  # h_hk_ratio,
            seqlen_q,  # seqlen_q,
            seqlen_k,  # seqlen_k,
            seqlen_q_rounded,  # seqlen_q_rounded,
            seqlen_k_rounded,  # seqlen_k_rounded,
            head_size,  # d,
            head_size_rounded,  # d_rounded,
            # scaling factors
            is_softcap,
            adjusted_softcap,  # softcap,
            adjusted_scale_softmax,  # scale_softmax,
            adjusted_scale_softmax_log2e,  # scale_softmax_log2,
            # pass Q/K/V per-block descale metadata to every
            # compute kernel so QK and PV can both use INT8 tensor cores.
            q_descale,  # q_descale_ptr,
            k_descale,  # k_descale_ptr,
            v_descale,  # v_descale_ptr,
            q_descale.stride(0),  # q_descale_batch_stride,
            q_descale.stride(1),  # q_descale_head_stride,
            q_descale.stride(2),  # q_descale_block_stride,
            k_descale.stride(0),  # k_descale_batch_stride,
            k_descale.stride(1),  # k_descale_head_stride,
            k_descale.stride(2),  # k_descale_block_stride,
            v_descale.stride(0),  # v_descale_batch_stride,
            v_descale.stride(1),  # v_descale_head_stride,
            v_descale.stride(2),  # v_descale_block_stride,
            # dropout
            is_dropout,
            p_dropout,
            rp_dropout,
            p_dropout_in_uint8_t,
            philox_args,
            return_softmax,
            # causal and swa
            is_causal,  # is_causal,
            is_local,  # is_local,
            window_size_left,  # window_size_left,
            window_size_right,  # window_size_right,
            seqlenq_ngroups_swapped,  # seqlenq_ngroups_swapped,
            False,  # is_paged,
            # alibi
            is_alibi,  #
            alibi_slopes,  # alibi_slopes_ptr,
            alibi_slopes_batch_stride,  # alibi_slopes_batch_stride,
            # block table params
            0,  # total_q,
            None,  # page_table_ptr,
            0,  # page_table_batch_stride,
            0,  # block_size,
            0,  # k_page_stride,
            # The INT8 PV residual dot is host-gated to short KV;
            # long sequences average the rounding noise out and skip the
            # second dot.
            seqlen_k < 512,  # precise_p,
        )

        # Move TxD to last dims for correct stride in Triton tt.load
        if runtime.device.vendor_name == "iluvatar":
            params.q_ptr = q.transpose(1, 2)
            params.k_ptr = k.transpose(1, 2)
            params.v_ptr = v.transpose(1, 2)
        kernel = dispatch(batch_size, num_heads, seqlen_q, seqlen_k, head_size, params)

        if _debug:
            print(f"{kernel.name} shared memory:", kernel.metadata.shared)
            print(f"{kernel.name} num_warps:", kernel.metadata.num_warps)
            print(f"{kernel.name} num_stages:", kernel.metadata.num_stages)
            # print(kernel.asm['ttgir'])

        if seqlenq_ngroups_swapped:
            out = out.transpose(1, 2).reshape(
                (batch_size, 1, num_heads_k * seqlen_q, head_size)
            )
            q = q.transpose(1, 2).reshape(
                (batch_size, 1, num_heads_k * seqlen_q, head_size)
            )
            lse = lse.reshape((batch_size, num_heads_k * seqlen_q, 1))

        unused = torch.empty((), dtype=torch.int64, device=q_device)

    return out, q, k, v, lse, philox_args, unused, p


def flash_attn_varlen_func_w8a8_int8(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # Compatibility arguments from the shared FlashAttention API.
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
    fa_version: int = 2,
):
    """MetaX INT8 attention with FP32 softmax, for inference only.

    Paged KV and GQA use FP16 PV and support head dimensions 64 and 128.
    Descales index
    logical 128-token blocks, independently of physical cache pages. The
    output defaults to BF16 or uses a supplied FP16/BF16 buffer. Forward
    inference only; unsupported training and scheduling modes raise errors.
    """
    if dropout_p != 0.0:
        raise NotImplementedError("dropout is not supported by this inference path")
    if return_attn_probs:
        raise NotImplementedError("return_attn_probs is not supported")
    if q_v is not None:
        raise NotImplementedError("q_v is not supported")
    if scheduler_metadata is not None or s_aux is not None:
        raise NotImplementedError("scheduler arguments are not supported")
    if cp_world_size != 1 or cp_rank != 0 or cp_tot_seqused_k is not None:
        raise NotImplementedError("context parallel attention is not supported")
    if runtime.device.vendor_name != "metax":
        raise NotImplementedError(
            "This W8A8 INT8 attention implementation only supports MetaX"
        )
    if fa_version != 2:
        raise NotImplementedError("Only FA2 is implemented.")
    if num_splits != 0:
        raise NotImplementedError("Explicit num_splits is not implemented.")
    assert (
        cu_seqlens_k is not None or seqused_k is not None
    ), "cu_seqlens_k or seqused_k must be provided"
    assert (
        cu_seqlens_k is None or seqused_k is None
    ), "cu_seqlens_k and seqused_k cannot be provided at the same time"
    assert (
        block_table is None or seqused_k is not None
    ), "seqused_k must be provided if block_table is provided"
    if seqused_k is not None and block_table is None:
        raise NotImplementedError("seqused_k without a paged KV cache is not supported")

    if not isinstance(max_seqlen_q, int) or not isinstance(max_seqlen_k, int):
        raise TypeError("max_seqlen_q and max_seqlen_k must be Python integers")
    if max_seqlen_q < 0 or max_seqlen_k < 0:
        raise ValueError("max_seqlen_q and max_seqlen_k must be nonnegative")
    if q.ndim != 3:
        raise ValueError("q must have shape [total_q, heads, head_dim]")
    if max_seqlen_q == 0 and q.shape[0] != 0:
        raise ValueError("max_seqlen_q must be positive when q is nonempty")
    expected_kv_ndim = 4 if block_table is not None else 3
    if k.ndim != expected_kv_ndim or v.ndim != expected_kv_ndim:
        raise ValueError("k and v rank does not match the selected cache layout")
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.int8:
        raise TypeError("q, k, and v must have dtype torch.int8 for W8A8-INT8")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise NotImplementedError("q, k, and v must be contiguous in head_dim")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    head_size = q.shape[-1]
    if not 8 <= head_size <= 256 or head_size % 8 != 0:
        raise NotImplementedError("head_dim must be a multiple of 8 between 8 and 256")
    if k.shape[-1] != head_size:
        raise ValueError("q, k, and v must have the same head_dim")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_q.device != q.device:
        raise ValueError("cu_seqlens_q must be an int32 tensor on q.device")
    if cu_seqlens_k is not None and (
        cu_seqlens_k.dtype != torch.int32 or cu_seqlens_k.device != q.device
    ):
        raise ValueError("cu_seqlens_k must be an int32 tensor on q.device")
    if seqused_k is not None and (
        seqused_k.dtype != torch.int32 or seqused_k.device != q.device
    ):
        raise ValueError("seqused_k must be an int32 tensor on q.device")
    if block_table is not None and (
        block_table.dtype != torch.int32 or block_table.device != q.device
    ):
        raise ValueError("block_table must be an int32 tensor on q.device")
    if out is not None:
        if out.shape != q.shape or out.device != q.device:
            raise ValueError("out must match q shape and device")
        if out.dtype not in _HIGH_PRECISION_DTYPES:
            raise TypeError("out must have dtype torch.float16 or torch.bfloat16")
        if out.stride(-1) != 1:
            raise NotImplementedError("out must be contiguous in head_dim")
        if out.numel() and out.data_ptr() in (
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
        ):
            raise ValueError("out must not alias q, k, or v")

    num_heads_k = k.shape[2] if block_table is not None else k.shape[1]
    if num_heads_k <= 0 or q.shape[1] <= 0 or q.shape[1] % num_heads_k != 0:
        raise ValueError("The number of KV heads must divide the number of query heads")
    if q.shape[1] != num_heads_k and head_size not in (64, 128):
        raise NotImplementedError("GQA supports head dimensions 64 and 128")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])

    batch_size = cu_seqlens_q.numel() - 1
    assert batch_size > 0, "batch_size must be positive"
    if cu_seqlens_q.ndim != 1 or not cu_seqlens_q.is_contiguous():
        raise ValueError("cu_seqlens_q must be a contiguous 1D tensor")
    if cu_seqlens_k is not None and (
        cu_seqlens_k.ndim != 1
        or cu_seqlens_k.numel() != batch_size + 1
        or not cu_seqlens_k.is_contiguous()
    ):
        raise ValueError("cu_seqlens_k must be contiguous with shape [batch + 1]")
    if seqused_k is not None and (
        seqused_k.ndim != 1
        or seqused_k.numel() != batch_size
        or not seqused_k.is_contiguous()
    ):
        raise ValueError("seqused_k must be contiguous with shape [batch]")
    if block_table is not None and (
        block_table.ndim != 2
        or block_table.shape[0] != batch_size
        or block_table.stride(-1) != 1
    ):
        raise ValueError(
            "block_table must be contiguous in its last dimension with shape [batch, pages]"
        )
    if alibi_slopes is not None and (
        alibi_slopes.device != q.device
        or alibi_slopes.dtype != torch.float32
        or alibi_slopes.stride(-1) != 1
        or alibi_slopes.shape not in ((q.shape[1],), (batch_size, q.shape[1]))
    ):
        raise ValueError(
            "alibi_slopes must be FP32 [heads] or [batch, heads] on q.device"
        )
    if q.shape[0] == 0 or max_seqlen_k == 0:
        if out is None:
            out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
        lse = (
            torch.empty((q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device)
            if return_softmax_lse
            else None
        )
        if q.shape[0] != 0:
            from flaggems_vllm.runtime.backend._metax.ops.flash_attention.common import (
                fill_tensor,
            )

            fill_tensor(out, 0.0)
            if return_softmax_lse:
                fill_tensor(lse, float("inf"))
        return (out, lse) if return_softmax_lse else out
    if head_size in (64, 128) and (
        block_table is not None or q.shape[1] != num_heads_k
    ):
        from flaggems_vllm.runtime.backend._metax.fused.paged_attention import (
            launch_paged_int8_attention,
        )

        normalized_descales = []
        for descale, heads, max_length, name in (
            (q_descale, q.shape[1], max_seqlen_q, "q_descale"),
            (k_descale, num_heads_k, max_seqlen_k, "k_descale"),
            (v_descale, num_heads_k, max_seqlen_k, "v_descale"),
        ):
            # Padded maximum lengths need not allocate unused logical scale blocks.
            blocks = (
                descale.shape[2]
                if descale is not None and descale.ndim == 3
                else triton.cdiv(max_length, 128)
            )
            normalized_descales.append(
                _normalize_dense_descale(
                    descale, batch_size, heads, blocks, q.device, name
                )
            )
        return launch_paged_int8_attention(
            q,
            k,
            v,
            max_seqlen_q,
            cu_seqlens_q,
            max_seqlen_k,
            cu_seqlens_k,
            seqused_k,
            softmax_scale,
            causal,
            real_window_size,
            softcap,
            alibi_slopes,
            block_table,
            return_softmax_lse,
            out,
            *normalized_descales,
        )
    uniform_nonpaged = (
        head_size in (64, 128)
        and not (
            runtime.device.vendor_name == "metax"
            and max_seqlen_q >= 128
            and max_seqlen_k >= 512
            and softcap == 0.0
            and alibi_slopes is None
            and (
                (not causal and real_window_size == (-1, -1))
                or (causal and real_window_size in ((-1, -1), (-1, 0)))
            )
        )
        and block_table is None
        and cu_seqlens_k is not None
        and seqused_k is None
        and dropout_p == 0.0
        and not return_softmax_lse
        and max_seqlen_q % 128 == 0
        and max_seqlen_k % 128 == 0
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and (out is None or out.is_contiguous())
        and q.shape[0] == batch_size * max_seqlen_q
        and k.shape[0] == batch_size * max_seqlen_k
    )
    if uniform_nonpaged:
        q_dense = q.view(batch_size, max_seqlen_q, q.shape[1], q.shape[2])
        k_dense = k.view(batch_size, max_seqlen_k, k.shape[1], k.shape[2])
        v_dense = v.view(batch_size, max_seqlen_k, v.shape[1], v.shape[2])
        out_dense = (
            None
            if out is None
            else out.view(batch_size, max_seqlen_q, out.shape[1], out.shape[2])
        )
        dense_result = mha_fwd(
            q_dense,
            k_dense,
            v_dense,
            out_dense,
            alibi_slopes,
            dropout_p,
            softmax_scale,
            causal,
            real_window_size[0],
            real_window_size[1],
            softcap,
            False,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return out if out is not None else dense_result[0].view(q.shape)

    cu_seqlens_k_arg = (
        torch.empty_like(cu_seqlens_q) if cu_seqlens_k is None else cu_seqlens_k
    )
    result = mha_varlan_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k_arg,
        seqused_k,
        None,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        real_window_size[0],
        real_window_size[1],
        softcap,
        return_softmax_lse and dropout_p > 0,
        None,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    return (result[0], result[4]) if return_softmax_lse else result[0]
