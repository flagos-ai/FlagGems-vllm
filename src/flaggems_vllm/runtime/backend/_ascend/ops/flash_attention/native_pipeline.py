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

import triton
import triton.experimental.tle as tle
import triton.language as tl
from triton.experimental.tle.language.dsa import extract_slice

from flaggems_vllm.runtime.backend._ascend.ops.flash_attention.direct import (
    _softmax_rows_dispatch,
    _softmax_stack,
)


@triton.jit
def _native_compute_pv(
    batch,
    begin,
    kh,
    klen,
    native_prob_0,
    native_prob_1,
    native_prob_full,
    native_value_0,
    native_value_1,
    native_value_2,
    native_value_3,
    prob,
    pvwork,
    stack,
    BM: tl.constexpr,
    BS: tl.constexpr,
    CONTIG_K: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    PAGE: tl.constexpr,
    Q1_STREAM: tl.constexpr,
    RAW_V,
    SPLITS: tl.constexpr,
    TABLE,
    TS: tl.constexpr,
    V,
    VH: tl.constexpr,
    VP: tl.constexpr,
    VR: tl.constexpr,
):
    """Compute PV from the current probability slot; the caller owns synchronization."""
    if stack == BS:
        native_prob_full_ptr = tl.make_block_ptr(
            prob, (BM, BS), (BS, 1), (0, 0), (BM, BS), (1, 0)
        )
        tle.dsa.copy(native_prob_full_ptr, native_prob_full, [BM, BS])
        native_prob_tensor_full = tle.dsa.to_tensor(native_prob_full, writable=False)
        c2 = tl.zeros((BM, 256), tl.float32)
        native_vpair_end = (stack + 2 * PAGE - 1) // (2 * PAGE) * (2 * PAGE)
        if PAGE == 32:
            pv_pre_32_0_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 64) // PAGE
            ).to(tl.int64)
            pv_pre_32_0_0_base = (
                RAW_V + pv_pre_32_0_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_32_0_0_ptr = tl.make_block_ptr(
                pv_pre_32_0_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_32_0_0_ptr, native_value_2, [PAGE, 256])
            pv_pre_32_0_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 96) // PAGE
            ).to(tl.int64)
            pv_pre_32_0_1_base = (
                RAW_V + pv_pre_32_0_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_32_0_1_ptr = tl.make_block_ptr(
                pv_pre_32_0_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_32_0_1_ptr, native_value_3, [PAGE, 256])
            pv_use_32_0_0_p = extract_slice(
                native_prob_tensor_full, (0, 0), (BM, PAGE), (1, 1)
            )
            pv_use_32_0_0_v = tle.dsa.to_tensor(native_value_0, writable=False)
            c2 = tl.dot(pv_use_32_0_0_p, pv_use_32_0_0_v, c2)
            pv_use_32_0_1_p = extract_slice(
                native_prob_tensor_full, (0, 32), (BM, PAGE), (1, 1)
            )
            pv_use_32_0_1_v = tle.dsa.to_tensor(native_value_1, writable=False)
            c2 = tl.dot(pv_use_32_0_1_p, pv_use_32_0_1_v, c2)
            pv_pre_32_1_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 128) // PAGE
            ).to(tl.int64)
            pv_pre_32_1_0_base = (
                RAW_V + pv_pre_32_1_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_32_1_0_ptr = tl.make_block_ptr(
                pv_pre_32_1_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_32_1_0_ptr, native_value_0, [PAGE, 256])
            pv_pre_32_1_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 160) // PAGE
            ).to(tl.int64)
            pv_pre_32_1_1_base = (
                RAW_V + pv_pre_32_1_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_32_1_1_ptr = tl.make_block_ptr(
                pv_pre_32_1_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_32_1_1_ptr, native_value_1, [PAGE, 256])
            pv_use_32_1_0_p = extract_slice(
                native_prob_tensor_full, (0, 64), (BM, PAGE), (1, 1)
            )
            pv_use_32_1_0_v = tle.dsa.to_tensor(native_value_2, writable=False)
            c2 = tl.dot(pv_use_32_1_0_p, pv_use_32_1_0_v, c2)
            pv_use_32_1_1_p = extract_slice(
                native_prob_tensor_full, (0, 96), (BM, PAGE), (1, 1)
            )
            pv_use_32_1_1_v = tle.dsa.to_tensor(native_value_3, writable=False)
            c2 = tl.dot(pv_use_32_1_1_p, pv_use_32_1_1_v, c2)
            pv_pre_32_2_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 192) // PAGE
            ).to(tl.int64)
            pv_pre_32_2_0_base = (
                RAW_V + pv_pre_32_2_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_32_2_0_ptr = tl.make_block_ptr(
                pv_pre_32_2_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_32_2_0_ptr, native_value_2, [PAGE, 256])
            pv_pre_32_2_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 224) // PAGE
            ).to(tl.int64)
            pv_pre_32_2_1_base = (
                RAW_V + pv_pre_32_2_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_32_2_1_ptr = tl.make_block_ptr(
                pv_pre_32_2_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_32_2_1_ptr, native_value_3, [PAGE, 256])
            pv_use_32_2_0_p = extract_slice(
                native_prob_tensor_full, (0, 128), (BM, PAGE), (1, 1)
            )
            pv_use_32_2_0_v = tle.dsa.to_tensor(native_value_0, writable=False)
            c2 = tl.dot(pv_use_32_2_0_p, pv_use_32_2_0_v, c2)
            pv_use_32_2_1_p = extract_slice(
                native_prob_tensor_full, (0, 160), (BM, PAGE), (1, 1)
            )
            pv_use_32_2_1_v = tle.dsa.to_tensor(native_value_1, writable=False)
            c2 = tl.dot(pv_use_32_2_1_p, pv_use_32_2_1_v, c2)
            pv_pre_32_3_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 256) // PAGE
            ).to(tl.int64)
            pv_pre_32_3_0_base = (
                RAW_V + pv_pre_32_3_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_32_3_0_ptr = tl.make_block_ptr(
                pv_pre_32_3_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_32_3_0_ptr, native_value_0, [PAGE, 256])
            pv_pre_32_3_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 288) // PAGE
            ).to(tl.int64)
            pv_pre_32_3_1_base = (
                RAW_V + pv_pre_32_3_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_32_3_1_ptr = tl.make_block_ptr(
                pv_pre_32_3_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_32_3_1_ptr, native_value_1, [PAGE, 256])
            pv_use_32_3_0_p = extract_slice(
                native_prob_tensor_full, (0, 192), (BM, PAGE), (1, 1)
            )
            pv_use_32_3_0_v = tle.dsa.to_tensor(native_value_2, writable=False)
            c2 = tl.dot(pv_use_32_3_0_p, pv_use_32_3_0_v, c2)
            pv_use_32_3_1_p = extract_slice(
                native_prob_tensor_full, (0, 224), (BM, PAGE), (1, 1)
            )
            pv_use_32_3_1_v = tle.dsa.to_tensor(native_value_3, writable=False)
            c2 = tl.dot(pv_use_32_3_1_p, pv_use_32_3_1_v, c2)
            pv_pre_32_4_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 320) // PAGE
            ).to(tl.int64)
            pv_pre_32_4_0_base = (
                RAW_V + pv_pre_32_4_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_32_4_0_ptr = tl.make_block_ptr(
                pv_pre_32_4_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_32_4_0_ptr, native_value_2, [PAGE, 256])
            pv_pre_32_4_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 352) // PAGE
            ).to(tl.int64)
            pv_pre_32_4_1_base = (
                RAW_V + pv_pre_32_4_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_32_4_1_ptr = tl.make_block_ptr(
                pv_pre_32_4_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_32_4_1_ptr, native_value_3, [PAGE, 256])
            pv_use_32_4_0_p = extract_slice(
                native_prob_tensor_full, (0, 256), (BM, PAGE), (1, 1)
            )
            pv_use_32_4_0_v = tle.dsa.to_tensor(native_value_0, writable=False)
            c2 = tl.dot(pv_use_32_4_0_p, pv_use_32_4_0_v, c2)
            pv_use_32_4_1_p = extract_slice(
                native_prob_tensor_full, (0, 288), (BM, PAGE), (1, 1)
            )
            pv_use_32_4_1_v = tle.dsa.to_tensor(native_value_1, writable=False)
            c2 = tl.dot(pv_use_32_4_1_p, pv_use_32_4_1_v, c2)
            pv_pre_32_5_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 384) // PAGE
            ).to(tl.int64)
            pv_pre_32_5_0_base = (
                RAW_V + pv_pre_32_5_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_32_5_0_ptr = tl.make_block_ptr(
                pv_pre_32_5_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_32_5_0_ptr, native_value_0, [PAGE, 256])
            pv_pre_32_5_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 416) // PAGE
            ).to(tl.int64)
            pv_pre_32_5_1_base = (
                RAW_V + pv_pre_32_5_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_32_5_1_ptr = tl.make_block_ptr(
                pv_pre_32_5_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_32_5_1_ptr, native_value_1, [PAGE, 256])
            pv_use_32_5_0_p = extract_slice(
                native_prob_tensor_full, (0, 320), (BM, PAGE), (1, 1)
            )
            pv_use_32_5_0_v = tle.dsa.to_tensor(native_value_2, writable=False)
            c2 = tl.dot(pv_use_32_5_0_p, pv_use_32_5_0_v, c2)
            pv_use_32_5_1_p = extract_slice(
                native_prob_tensor_full, (0, 352), (BM, PAGE), (1, 1)
            )
            pv_use_32_5_1_v = tle.dsa.to_tensor(native_value_3, writable=False)
            c2 = tl.dot(pv_use_32_5_1_p, pv_use_32_5_1_v, c2)
            pv_pre_32_6_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 448) // PAGE
            ).to(tl.int64)
            pv_pre_32_6_0_base = (
                RAW_V + pv_pre_32_6_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_32_6_0_ptr = tl.make_block_ptr(
                pv_pre_32_6_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_32_6_0_ptr, native_value_2, [PAGE, 256])
            pv_pre_32_6_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 480) // PAGE
            ).to(tl.int64)
            pv_pre_32_6_1_base = (
                RAW_V + pv_pre_32_6_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_32_6_1_ptr = tl.make_block_ptr(
                pv_pre_32_6_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_32_6_1_ptr, native_value_3, [PAGE, 256])
            pv_use_32_6_0_p = extract_slice(
                native_prob_tensor_full, (0, 384), (BM, PAGE), (1, 1)
            )
            pv_use_32_6_0_v = tle.dsa.to_tensor(native_value_0, writable=False)
            c2 = tl.dot(pv_use_32_6_0_p, pv_use_32_6_0_v, c2)
            pv_use_32_6_1_p = extract_slice(
                native_prob_tensor_full, (0, 416), (BM, PAGE), (1, 1)
            )
            pv_use_32_6_1_v = tle.dsa.to_tensor(native_value_1, writable=False)
            c2 = tl.dot(pv_use_32_6_1_p, pv_use_32_6_1_v, c2)
            pv_use_32_7_0_p = extract_slice(
                native_prob_tensor_full, (0, 448), (BM, PAGE), (1, 1)
            )
            pv_use_32_7_0_v = tle.dsa.to_tensor(native_value_2, writable=False)
            c2 = tl.dot(pv_use_32_7_0_p, pv_use_32_7_0_v, c2)
            pv_use_32_7_1_p = extract_slice(
                native_prob_tensor_full, (0, 480), (BM, PAGE), (1, 1)
            )
            pv_use_32_7_1_v = tle.dsa.to_tensor(native_value_3, writable=False)
            c2 = tl.dot(pv_use_32_7_1_p, pv_use_32_7_1_v, c2)
        else:
            pv_pre_16_0_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 32) // PAGE
            ).to(tl.int64)
            pv_pre_16_0_0_base = (
                RAW_V + pv_pre_16_0_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_0_0_ptr = tl.make_block_ptr(
                pv_pre_16_0_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_0_0_ptr, native_value_2, [PAGE, 256])
            pv_pre_16_0_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 48) // PAGE
            ).to(tl.int64)
            pv_pre_16_0_1_base = (
                RAW_V + pv_pre_16_0_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_0_1_ptr = tl.make_block_ptr(
                pv_pre_16_0_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_0_1_ptr, native_value_3, [PAGE, 256])
            pv_use_16_0_0_p = extract_slice(
                native_prob_tensor_full, (0, 0), (BM, PAGE), (1, 1)
            )
            pv_use_16_0_0_v = tle.dsa.to_tensor(native_value_0, writable=False)
            c2 = tl.dot(pv_use_16_0_0_p, pv_use_16_0_0_v, c2)
            pv_use_16_0_1_p = extract_slice(
                native_prob_tensor_full, (0, 16), (BM, PAGE), (1, 1)
            )
            pv_use_16_0_1_v = tle.dsa.to_tensor(native_value_1, writable=False)
            c2 = tl.dot(pv_use_16_0_1_p, pv_use_16_0_1_v, c2)
            pv_pre_16_1_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 64) // PAGE
            ).to(tl.int64)
            pv_pre_16_1_0_base = (
                RAW_V + pv_pre_16_1_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_1_0_ptr = tl.make_block_ptr(
                pv_pre_16_1_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_1_0_ptr, native_value_0, [PAGE, 256])
            pv_pre_16_1_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 80) // PAGE
            ).to(tl.int64)
            pv_pre_16_1_1_base = (
                RAW_V + pv_pre_16_1_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_1_1_ptr = tl.make_block_ptr(
                pv_pre_16_1_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_1_1_ptr, native_value_1, [PAGE, 256])
            pv_use_16_1_0_p = extract_slice(
                native_prob_tensor_full, (0, 32), (BM, PAGE), (1, 1)
            )
            pv_use_16_1_0_v = tle.dsa.to_tensor(native_value_2, writable=False)
            c2 = tl.dot(pv_use_16_1_0_p, pv_use_16_1_0_v, c2)
            pv_use_16_1_1_p = extract_slice(
                native_prob_tensor_full, (0, 48), (BM, PAGE), (1, 1)
            )
            pv_use_16_1_1_v = tle.dsa.to_tensor(native_value_3, writable=False)
            c2 = tl.dot(pv_use_16_1_1_p, pv_use_16_1_1_v, c2)
            pv_pre_16_2_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 96) // PAGE
            ).to(tl.int64)
            pv_pre_16_2_0_base = (
                RAW_V + pv_pre_16_2_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_2_0_ptr = tl.make_block_ptr(
                pv_pre_16_2_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_2_0_ptr, native_value_2, [PAGE, 256])
            pv_pre_16_2_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 112) // PAGE
            ).to(tl.int64)
            pv_pre_16_2_1_base = (
                RAW_V + pv_pre_16_2_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_2_1_ptr = tl.make_block_ptr(
                pv_pre_16_2_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_2_1_ptr, native_value_3, [PAGE, 256])
            pv_use_16_2_0_p = extract_slice(
                native_prob_tensor_full, (0, 64), (BM, PAGE), (1, 1)
            )
            pv_use_16_2_0_v = tle.dsa.to_tensor(native_value_0, writable=False)
            c2 = tl.dot(pv_use_16_2_0_p, pv_use_16_2_0_v, c2)
            pv_use_16_2_1_p = extract_slice(
                native_prob_tensor_full, (0, 80), (BM, PAGE), (1, 1)
            )
            pv_use_16_2_1_v = tle.dsa.to_tensor(native_value_1, writable=False)
            c2 = tl.dot(pv_use_16_2_1_p, pv_use_16_2_1_v, c2)
            pv_pre_16_3_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 128) // PAGE
            ).to(tl.int64)
            pv_pre_16_3_0_base = (
                RAW_V + pv_pre_16_3_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_3_0_ptr = tl.make_block_ptr(
                pv_pre_16_3_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_3_0_ptr, native_value_0, [PAGE, 256])
            pv_pre_16_3_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 144) // PAGE
            ).to(tl.int64)
            pv_pre_16_3_1_base = (
                RAW_V + pv_pre_16_3_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_3_1_ptr = tl.make_block_ptr(
                pv_pre_16_3_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_3_1_ptr, native_value_1, [PAGE, 256])
            pv_use_16_3_0_p = extract_slice(
                native_prob_tensor_full, (0, 96), (BM, PAGE), (1, 1)
            )
            pv_use_16_3_0_v = tle.dsa.to_tensor(native_value_2, writable=False)
            c2 = tl.dot(pv_use_16_3_0_p, pv_use_16_3_0_v, c2)
            pv_use_16_3_1_p = extract_slice(
                native_prob_tensor_full, (0, 112), (BM, PAGE), (1, 1)
            )
            pv_use_16_3_1_v = tle.dsa.to_tensor(native_value_3, writable=False)
            c2 = tl.dot(pv_use_16_3_1_p, pv_use_16_3_1_v, c2)
            pv_pre_16_4_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 160) // PAGE
            ).to(tl.int64)
            pv_pre_16_4_0_base = (
                RAW_V + pv_pre_16_4_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_4_0_ptr = tl.make_block_ptr(
                pv_pre_16_4_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_4_0_ptr, native_value_2, [PAGE, 256])
            pv_pre_16_4_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 176) // PAGE
            ).to(tl.int64)
            pv_pre_16_4_1_base = (
                RAW_V + pv_pre_16_4_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_4_1_ptr = tl.make_block_ptr(
                pv_pre_16_4_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_4_1_ptr, native_value_3, [PAGE, 256])
            pv_use_16_4_0_p = extract_slice(
                native_prob_tensor_full, (0, 128), (BM, PAGE), (1, 1)
            )
            pv_use_16_4_0_v = tle.dsa.to_tensor(native_value_0, writable=False)
            c2 = tl.dot(pv_use_16_4_0_p, pv_use_16_4_0_v, c2)
            pv_use_16_4_1_p = extract_slice(
                native_prob_tensor_full, (0, 144), (BM, PAGE), (1, 1)
            )
            pv_use_16_4_1_v = tle.dsa.to_tensor(native_value_1, writable=False)
            c2 = tl.dot(pv_use_16_4_1_p, pv_use_16_4_1_v, c2)
            pv_pre_16_5_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 192) // PAGE
            ).to(tl.int64)
            pv_pre_16_5_0_base = (
                RAW_V + pv_pre_16_5_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_5_0_ptr = tl.make_block_ptr(
                pv_pre_16_5_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_5_0_ptr, native_value_0, [PAGE, 256])
            pv_pre_16_5_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 208) // PAGE
            ).to(tl.int64)
            pv_pre_16_5_1_base = (
                RAW_V + pv_pre_16_5_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_5_1_ptr = tl.make_block_ptr(
                pv_pre_16_5_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_5_1_ptr, native_value_1, [PAGE, 256])
            pv_use_16_5_0_p = extract_slice(
                native_prob_tensor_full, (0, 160), (BM, PAGE), (1, 1)
            )
            pv_use_16_5_0_v = tle.dsa.to_tensor(native_value_2, writable=False)
            c2 = tl.dot(pv_use_16_5_0_p, pv_use_16_5_0_v, c2)
            pv_use_16_5_1_p = extract_slice(
                native_prob_tensor_full, (0, 176), (BM, PAGE), (1, 1)
            )
            pv_use_16_5_1_v = tle.dsa.to_tensor(native_value_3, writable=False)
            c2 = tl.dot(pv_use_16_5_1_p, pv_use_16_5_1_v, c2)
            pv_pre_16_6_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 224) // PAGE
            ).to(tl.int64)
            pv_pre_16_6_0_base = (
                RAW_V + pv_pre_16_6_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_6_0_ptr = tl.make_block_ptr(
                pv_pre_16_6_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_6_0_ptr, native_value_2, [PAGE, 256])
            pv_pre_16_6_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 240) // PAGE
            ).to(tl.int64)
            pv_pre_16_6_1_base = (
                RAW_V + pv_pre_16_6_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_6_1_ptr = tl.make_block_ptr(
                pv_pre_16_6_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_6_1_ptr, native_value_3, [PAGE, 256])
            pv_use_16_6_0_p = extract_slice(
                native_prob_tensor_full, (0, 192), (BM, PAGE), (1, 1)
            )
            pv_use_16_6_0_v = tle.dsa.to_tensor(native_value_0, writable=False)
            c2 = tl.dot(pv_use_16_6_0_p, pv_use_16_6_0_v, c2)
            pv_use_16_6_1_p = extract_slice(
                native_prob_tensor_full, (0, 208), (BM, PAGE), (1, 1)
            )
            pv_use_16_6_1_v = tle.dsa.to_tensor(native_value_1, writable=False)
            c2 = tl.dot(pv_use_16_6_1_p, pv_use_16_6_1_v, c2)
            pv_pre_16_7_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 256) // PAGE
            ).to(tl.int64)
            pv_pre_16_7_0_base = (
                RAW_V + pv_pre_16_7_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_7_0_ptr = tl.make_block_ptr(
                pv_pre_16_7_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_7_0_ptr, native_value_0, [PAGE, 256])
            pv_pre_16_7_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 272) // PAGE
            ).to(tl.int64)
            pv_pre_16_7_1_base = (
                RAW_V + pv_pre_16_7_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_7_1_ptr = tl.make_block_ptr(
                pv_pre_16_7_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_7_1_ptr, native_value_1, [PAGE, 256])
            pv_use_16_7_0_p = extract_slice(
                native_prob_tensor_full, (0, 224), (BM, PAGE), (1, 1)
            )
            pv_use_16_7_0_v = tle.dsa.to_tensor(native_value_2, writable=False)
            c2 = tl.dot(pv_use_16_7_0_p, pv_use_16_7_0_v, c2)
            pv_use_16_7_1_p = extract_slice(
                native_prob_tensor_full, (0, 240), (BM, PAGE), (1, 1)
            )
            pv_use_16_7_1_v = tle.dsa.to_tensor(native_value_3, writable=False)
            c2 = tl.dot(pv_use_16_7_1_p, pv_use_16_7_1_v, c2)
            pv_pre_16_8_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 288) // PAGE
            ).to(tl.int64)
            pv_pre_16_8_0_base = (
                RAW_V + pv_pre_16_8_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_8_0_ptr = tl.make_block_ptr(
                pv_pre_16_8_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_8_0_ptr, native_value_2, [PAGE, 256])
            pv_pre_16_8_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 304) // PAGE
            ).to(tl.int64)
            pv_pre_16_8_1_base = (
                RAW_V + pv_pre_16_8_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_8_1_ptr = tl.make_block_ptr(
                pv_pre_16_8_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_8_1_ptr, native_value_3, [PAGE, 256])
            pv_use_16_8_0_p = extract_slice(
                native_prob_tensor_full, (0, 256), (BM, PAGE), (1, 1)
            )
            pv_use_16_8_0_v = tle.dsa.to_tensor(native_value_0, writable=False)
            c2 = tl.dot(pv_use_16_8_0_p, pv_use_16_8_0_v, c2)
            pv_use_16_8_1_p = extract_slice(
                native_prob_tensor_full, (0, 272), (BM, PAGE), (1, 1)
            )
            pv_use_16_8_1_v = tle.dsa.to_tensor(native_value_1, writable=False)
            c2 = tl.dot(pv_use_16_8_1_p, pv_use_16_8_1_v, c2)
            pv_pre_16_9_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 320) // PAGE
            ).to(tl.int64)
            pv_pre_16_9_0_base = (
                RAW_V + pv_pre_16_9_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_9_0_ptr = tl.make_block_ptr(
                pv_pre_16_9_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_9_0_ptr, native_value_0, [PAGE, 256])
            pv_pre_16_9_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 336) // PAGE
            ).to(tl.int64)
            pv_pre_16_9_1_base = (
                RAW_V + pv_pre_16_9_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_9_1_ptr = tl.make_block_ptr(
                pv_pre_16_9_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_9_1_ptr, native_value_1, [PAGE, 256])
            pv_use_16_9_0_p = extract_slice(
                native_prob_tensor_full, (0, 288), (BM, PAGE), (1, 1)
            )
            pv_use_16_9_0_v = tle.dsa.to_tensor(native_value_2, writable=False)
            c2 = tl.dot(pv_use_16_9_0_p, pv_use_16_9_0_v, c2)
            pv_use_16_9_1_p = extract_slice(
                native_prob_tensor_full, (0, 304), (BM, PAGE), (1, 1)
            )
            pv_use_16_9_1_v = tle.dsa.to_tensor(native_value_3, writable=False)
            c2 = tl.dot(pv_use_16_9_1_p, pv_use_16_9_1_v, c2)
            pv_pre_16_10_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 352) // PAGE
            ).to(tl.int64)
            pv_pre_16_10_0_base = (
                RAW_V + pv_pre_16_10_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_10_0_ptr = tl.make_block_ptr(
                pv_pre_16_10_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_10_0_ptr, native_value_2, [PAGE, 256])
            pv_pre_16_10_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 368) // PAGE
            ).to(tl.int64)
            pv_pre_16_10_1_base = (
                RAW_V + pv_pre_16_10_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_10_1_ptr = tl.make_block_ptr(
                pv_pre_16_10_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_10_1_ptr, native_value_3, [PAGE, 256])
            pv_use_16_10_0_p = extract_slice(
                native_prob_tensor_full, (0, 320), (BM, PAGE), (1, 1)
            )
            pv_use_16_10_0_v = tle.dsa.to_tensor(native_value_0, writable=False)
            c2 = tl.dot(pv_use_16_10_0_p, pv_use_16_10_0_v, c2)
            pv_use_16_10_1_p = extract_slice(
                native_prob_tensor_full, (0, 336), (BM, PAGE), (1, 1)
            )
            pv_use_16_10_1_v = tle.dsa.to_tensor(native_value_1, writable=False)
            c2 = tl.dot(pv_use_16_10_1_p, pv_use_16_10_1_v, c2)
            pv_pre_16_11_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 384) // PAGE
            ).to(tl.int64)
            pv_pre_16_11_0_base = (
                RAW_V + pv_pre_16_11_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_11_0_ptr = tl.make_block_ptr(
                pv_pre_16_11_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_11_0_ptr, native_value_0, [PAGE, 256])
            pv_pre_16_11_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 400) // PAGE
            ).to(tl.int64)
            pv_pre_16_11_1_base = (
                RAW_V + pv_pre_16_11_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_11_1_ptr = tl.make_block_ptr(
                pv_pre_16_11_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_11_1_ptr, native_value_1, [PAGE, 256])
            pv_use_16_11_0_p = extract_slice(
                native_prob_tensor_full, (0, 352), (BM, PAGE), (1, 1)
            )
            pv_use_16_11_0_v = tle.dsa.to_tensor(native_value_2, writable=False)
            c2 = tl.dot(pv_use_16_11_0_p, pv_use_16_11_0_v, c2)
            pv_use_16_11_1_p = extract_slice(
                native_prob_tensor_full, (0, 368), (BM, PAGE), (1, 1)
            )
            pv_use_16_11_1_v = tle.dsa.to_tensor(native_value_3, writable=False)
            c2 = tl.dot(pv_use_16_11_1_p, pv_use_16_11_1_v, c2)
            pv_pre_16_12_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 416) // PAGE
            ).to(tl.int64)
            pv_pre_16_12_0_base = (
                RAW_V + pv_pre_16_12_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_12_0_ptr = tl.make_block_ptr(
                pv_pre_16_12_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_12_0_ptr, native_value_2, [PAGE, 256])
            pv_pre_16_12_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 432) // PAGE
            ).to(tl.int64)
            pv_pre_16_12_1_base = (
                RAW_V + pv_pre_16_12_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_12_1_ptr = tl.make_block_ptr(
                pv_pre_16_12_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_12_1_ptr, native_value_3, [PAGE, 256])
            pv_use_16_12_0_p = extract_slice(
                native_prob_tensor_full, (0, 384), (BM, PAGE), (1, 1)
            )
            pv_use_16_12_0_v = tle.dsa.to_tensor(native_value_0, writable=False)
            c2 = tl.dot(pv_use_16_12_0_p, pv_use_16_12_0_v, c2)
            pv_use_16_12_1_p = extract_slice(
                native_prob_tensor_full, (0, 400), (BM, PAGE), (1, 1)
            )
            pv_use_16_12_1_v = tle.dsa.to_tensor(native_value_1, writable=False)
            c2 = tl.dot(pv_use_16_12_1_p, pv_use_16_12_1_v, c2)
            pv_pre_16_13_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 448) // PAGE
            ).to(tl.int64)
            pv_pre_16_13_0_base = (
                RAW_V + pv_pre_16_13_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_13_0_ptr = tl.make_block_ptr(
                pv_pre_16_13_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_13_0_ptr, native_value_0, [PAGE, 256])
            pv_pre_16_13_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 464) // PAGE
            ).to(tl.int64)
            pv_pre_16_13_1_base = (
                RAW_V + pv_pre_16_13_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_13_1_ptr = tl.make_block_ptr(
                pv_pre_16_13_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_13_1_ptr, native_value_1, [PAGE, 256])
            pv_use_16_13_0_p = extract_slice(
                native_prob_tensor_full, (0, 416), (BM, PAGE), (1, 1)
            )
            pv_use_16_13_0_v = tle.dsa.to_tensor(native_value_2, writable=False)
            c2 = tl.dot(pv_use_16_13_0_p, pv_use_16_13_0_v, c2)
            pv_use_16_13_1_p = extract_slice(
                native_prob_tensor_full, (0, 432), (BM, PAGE), (1, 1)
            )
            pv_use_16_13_1_v = tle.dsa.to_tensor(native_value_3, writable=False)
            c2 = tl.dot(pv_use_16_13_1_p, pv_use_16_13_1_v, c2)
            pv_pre_16_14_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 480) // PAGE
            ).to(tl.int64)
            pv_pre_16_14_0_base = (
                RAW_V + pv_pre_16_14_0_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_14_0_ptr = tl.make_block_ptr(
                pv_pre_16_14_0_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_14_0_ptr, native_value_2, [PAGE, 256])
            pv_pre_16_14_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 496) // PAGE
            ).to(tl.int64)
            pv_pre_16_14_1_base = (
                RAW_V + pv_pre_16_14_1_physical * VP + kh.to(tl.int64) * VH
            )
            pv_pre_16_14_1_ptr = tl.make_block_ptr(
                pv_pre_16_14_1_base,
                (PAGE, 256),
                (VR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(pv_pre_16_14_1_ptr, native_value_3, [PAGE, 256])
            pv_use_16_14_0_p = extract_slice(
                native_prob_tensor_full, (0, 448), (BM, PAGE), (1, 1)
            )
            pv_use_16_14_0_v = tle.dsa.to_tensor(native_value_0, writable=False)
            c2 = tl.dot(pv_use_16_14_0_p, pv_use_16_14_0_v, c2)
            pv_use_16_14_1_p = extract_slice(
                native_prob_tensor_full, (0, 464), (BM, PAGE), (1, 1)
            )
            pv_use_16_14_1_v = tle.dsa.to_tensor(native_value_1, writable=False)
            c2 = tl.dot(pv_use_16_14_1_p, pv_use_16_14_1_v, c2)
            pv_use_16_15_0_p = extract_slice(
                native_prob_tensor_full, (0, 480), (BM, PAGE), (1, 1)
            )
            pv_use_16_15_0_v = tle.dsa.to_tensor(native_value_2, writable=False)
            c2 = tl.dot(pv_use_16_15_0_p, pv_use_16_15_0_v, c2)
            pv_use_16_15_1_p = extract_slice(
                native_prob_tensor_full, (0, 496), (BM, PAGE), (1, 1)
            )
            pv_use_16_15_1_v = tle.dsa.to_tensor(native_value_3, writable=False)
            c2 = tl.dot(pv_use_16_15_1_p, pv_use_16_15_1_v, c2)
        tl.store(
            pvwork + tl.arange(0, BM)[:, None] * 256 + tl.arange(0, 256)[None, :],
            c2,
        )
    elif (
        H // HK == 8 and CONTIG_K == 2560 and (SPLITS == 1) and Q1_STREAM
    ) and stack <= PAGE:
        tail_prob_ptr = tl.make_block_ptr(
            prob, (BM, BS), (BS, 1), (0, 0), (BM, PAGE), (1, 0)
        )
        tle.dsa.copy(tail_prob_ptr, native_prob_0, [BM, PAGE])
        tail_weights = tle.dsa.to_tensor(native_prob_0, writable=False)
        tail_values = tle.dsa.to_tensor(native_value_0, writable=False)
        tail_pv = tl.dot(tail_weights, tail_values)
        tl.store(
            pvwork + tl.arange(0, BM)[:, None] * 256 + tl.arange(0, 256)[None, :],
            tail_pv,
        )
    else:
        c2 = tl.zeros((BM, 256), tl.float32)
        native_vpair_end = (stack + 2 * PAGE - 1) // (2 * PAGE) * (2 * PAGE)
        for native_cv in range(0, native_vpair_end, 2 * PAGE):
            if native_cv > 0:
                if begin + native_cv + PAGE <= klen:
                    native_v0_physical = tl.load(
                        TABLE + batch.to(tl.int64) * TS + (begin + native_cv) // PAGE
                    ).to(tl.int64)
                    native_v0_full_base = (
                        RAW_V + native_v0_physical * VP + kh.to(tl.int64) * VH
                    )
                    native_v0_full_ptr = tl.make_block_ptr(
                        native_v0_full_base,
                        (PAGE, 256),
                        (VR, 1),
                        (0, 0),
                        (PAGE, 256),
                        (1, 0),
                    )
                    tle.dsa.copy(native_v0_full_ptr, native_value_0, [PAGE, 256])
                else:
                    native_v0_tail_base = (
                        V
                        + (
                            (batch.to(tl.int64) * HK + kh) * CONTIG_K
                            + begin
                            + native_cv
                        )
                        * 256
                    )
                    native_v0_tail_ptr = tl.make_block_ptr(
                        native_v0_tail_base,
                        (PAGE, 256),
                        (256, 1),
                        (0, 0),
                        (PAGE, 256),
                        (1, 0),
                    )
                    tle.dsa.copy(native_v0_tail_ptr, native_value_0, [PAGE, 256])
                if begin + (native_cv + PAGE) + PAGE <= klen:
                    native_v1_physical = tl.load(
                        TABLE
                        + batch.to(tl.int64) * TS
                        + (begin + (native_cv + PAGE)) // PAGE
                    ).to(tl.int64)
                    native_v1_full_base = (
                        RAW_V + native_v1_physical * VP + kh.to(tl.int64) * VH
                    )
                    native_v1_full_ptr = tl.make_block_ptr(
                        native_v1_full_base,
                        (PAGE, 256),
                        (VR, 1),
                        (0, 0),
                        (PAGE, 256),
                        (1, 0),
                    )
                    tle.dsa.copy(native_v1_full_ptr, native_value_1, [PAGE, 256])
                else:
                    native_v1_tail_base = (
                        V
                        + (
                            (batch.to(tl.int64) * HK + kh) * CONTIG_K
                            + begin
                            + (native_cv + PAGE)
                        )
                        * 256
                    )
                    native_v1_tail_ptr = tl.make_block_ptr(
                        native_v1_tail_base,
                        (PAGE, 256),
                        (256, 1),
                        (0, 0),
                        (PAGE, 256),
                        (1, 0),
                    )
                    tle.dsa.copy(native_v1_tail_ptr, native_value_1, [PAGE, 256])
            native_p0_ptr = tl.make_block_ptr(
                prob + native_cv,
                (BM, BS),
                (BS, 1),
                (0, 0),
                (BM, PAGE),
                (1, 0),
            )
            tle.dsa.copy(native_p0_ptr, native_prob_0, [BM, PAGE])
            native_p1_ptr = tl.make_block_ptr(
                prob + (native_cv + PAGE),
                (BM, BS),
                (BS, 1),
                (0, 0),
                (BM, PAGE),
                (1, 0),
            )
            tle.dsa.copy(native_p1_ptr, native_prob_1, [BM, PAGE])
            native_v_tensor_0 = tle.dsa.to_tensor(native_value_0, writable=False)
            native_v_tensor_1 = tle.dsa.to_tensor(native_value_1, writable=False)
            native_p_tensor_0 = tle.dsa.to_tensor(native_prob_0, writable=False)
            native_p_tensor_1 = tle.dsa.to_tensor(native_prob_1, writable=False)
            c2 = tl.dot(native_p_tensor_0, native_v_tensor_0, c2)
            c2 = tl.dot(native_p_tensor_1, native_v_tensor_1, c2)
        tl.store(
            pvwork + tl.arange(0, BM)[:, None] * 256 + tl.arange(0, 256)[None, :],
            c2,
        )


@triton.jit
def _native_compute_qk(
    batch,
    begin,
    kh,
    klen,
    native_key_0,
    native_key_1,
    native_key_2,
    native_key_3,
    q1,
    scratch,
    stack,
    BM: tl.constexpr,
    BS: tl.constexpr,
    CONTIG_K: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    K,
    KH: tl.constexpr,
    KP: tl.constexpr,
    KR: tl.constexpr,
    PAGE: tl.constexpr,
    Q1_STREAM: tl.constexpr,
    RAW_K,
    SPLITS: tl.constexpr,
    TABLE,
    TS: tl.constexpr,
):
    """Compute QK from page pairs, preserving the load-ahead order."""
    if stack == BS:
        if PAGE == 32:
            qk_pre_32_0_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 0) // PAGE
            ).to(tl.int64)
            qk_pre_32_0_0_base = (
                RAW_K + qk_pre_32_0_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_0_0_ptr = tl.make_block_ptr(
                qk_pre_32_0_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_0_0_ptr, native_key_0, [PAGE, 256])
            qk_pre_32_0_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 32) // PAGE
            ).to(tl.int64)
            qk_pre_32_0_1_base = (
                RAW_K + qk_pre_32_0_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_0_1_ptr = tl.make_block_ptr(
                qk_pre_32_0_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_0_1_ptr, native_key_1, [PAGE, 256])
            qk_pre_32_1_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 64) // PAGE
            ).to(tl.int64)
            qk_pre_32_1_0_base = (
                RAW_K + qk_pre_32_1_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_1_0_ptr = tl.make_block_ptr(
                qk_pre_32_1_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_1_0_ptr, native_key_2, [PAGE, 256])
            qk_pre_32_1_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 96) // PAGE
            ).to(tl.int64)
            qk_pre_32_1_1_base = (
                RAW_K + qk_pre_32_1_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_1_1_ptr = tl.make_block_ptr(
                qk_pre_32_1_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_1_1_ptr, native_key_3, [PAGE, 256])
            qk_use_32_0_0_key = tle.dsa.to_tensor(native_key_0, writable=False)
            qk_use_32_0_0_score = tl.dot(q1, tl.trans(qk_use_32_0_0_key))
            qk_use_32_0_1_key = tle.dsa.to_tensor(native_key_1, writable=False)
            qk_use_32_0_1_score = tl.dot(q1, tl.trans(qk_use_32_0_1_key))
            qk_use_32_0_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 0 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_0_0_offset, qk_use_32_0_0_score)
            qk_use_32_0_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 32 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_0_1_offset, qk_use_32_0_1_score)
            qk_pre_32_2_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 128) // PAGE
            ).to(tl.int64)
            qk_pre_32_2_0_base = (
                RAW_K + qk_pre_32_2_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_2_0_ptr = tl.make_block_ptr(
                qk_pre_32_2_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_2_0_ptr, native_key_0, [PAGE, 256])
            qk_pre_32_2_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 160) // PAGE
            ).to(tl.int64)
            qk_pre_32_2_1_base = (
                RAW_K + qk_pre_32_2_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_2_1_ptr = tl.make_block_ptr(
                qk_pre_32_2_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_2_1_ptr, native_key_1, [PAGE, 256])
            qk_use_32_1_0_key = tle.dsa.to_tensor(native_key_2, writable=False)
            qk_use_32_1_0_score = tl.dot(q1, tl.trans(qk_use_32_1_0_key))
            qk_use_32_1_1_key = tle.dsa.to_tensor(native_key_3, writable=False)
            qk_use_32_1_1_score = tl.dot(q1, tl.trans(qk_use_32_1_1_key))
            qk_use_32_1_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 64 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_1_0_offset, qk_use_32_1_0_score)
            qk_use_32_1_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 96 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_1_1_offset, qk_use_32_1_1_score)
            qk_pre_32_3_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 192) // PAGE
            ).to(tl.int64)
            qk_pre_32_3_0_base = (
                RAW_K + qk_pre_32_3_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_3_0_ptr = tl.make_block_ptr(
                qk_pre_32_3_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_3_0_ptr, native_key_2, [PAGE, 256])
            qk_pre_32_3_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 224) // PAGE
            ).to(tl.int64)
            qk_pre_32_3_1_base = (
                RAW_K + qk_pre_32_3_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_3_1_ptr = tl.make_block_ptr(
                qk_pre_32_3_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_3_1_ptr, native_key_3, [PAGE, 256])
            qk_use_32_2_0_key = tle.dsa.to_tensor(native_key_0, writable=False)
            qk_use_32_2_0_score = tl.dot(q1, tl.trans(qk_use_32_2_0_key))
            qk_use_32_2_1_key = tle.dsa.to_tensor(native_key_1, writable=False)
            qk_use_32_2_1_score = tl.dot(q1, tl.trans(qk_use_32_2_1_key))
            qk_use_32_2_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 128 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_2_0_offset, qk_use_32_2_0_score)
            qk_use_32_2_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 160 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_2_1_offset, qk_use_32_2_1_score)
            qk_pre_32_4_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 256) // PAGE
            ).to(tl.int64)
            qk_pre_32_4_0_base = (
                RAW_K + qk_pre_32_4_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_4_0_ptr = tl.make_block_ptr(
                qk_pre_32_4_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_4_0_ptr, native_key_0, [PAGE, 256])
            qk_pre_32_4_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 288) // PAGE
            ).to(tl.int64)
            qk_pre_32_4_1_base = (
                RAW_K + qk_pre_32_4_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_4_1_ptr = tl.make_block_ptr(
                qk_pre_32_4_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_4_1_ptr, native_key_1, [PAGE, 256])
            qk_use_32_3_0_key = tle.dsa.to_tensor(native_key_2, writable=False)
            qk_use_32_3_0_score = tl.dot(q1, tl.trans(qk_use_32_3_0_key))
            qk_use_32_3_1_key = tle.dsa.to_tensor(native_key_3, writable=False)
            qk_use_32_3_1_score = tl.dot(q1, tl.trans(qk_use_32_3_1_key))
            qk_use_32_3_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 192 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_3_0_offset, qk_use_32_3_0_score)
            qk_use_32_3_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 224 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_3_1_offset, qk_use_32_3_1_score)
            qk_pre_32_5_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 320) // PAGE
            ).to(tl.int64)
            qk_pre_32_5_0_base = (
                RAW_K + qk_pre_32_5_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_5_0_ptr = tl.make_block_ptr(
                qk_pre_32_5_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_5_0_ptr, native_key_2, [PAGE, 256])
            qk_pre_32_5_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 352) // PAGE
            ).to(tl.int64)
            qk_pre_32_5_1_base = (
                RAW_K + qk_pre_32_5_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_5_1_ptr = tl.make_block_ptr(
                qk_pre_32_5_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_5_1_ptr, native_key_3, [PAGE, 256])
            qk_use_32_4_0_key = tle.dsa.to_tensor(native_key_0, writable=False)
            qk_use_32_4_0_score = tl.dot(q1, tl.trans(qk_use_32_4_0_key))
            qk_use_32_4_1_key = tle.dsa.to_tensor(native_key_1, writable=False)
            qk_use_32_4_1_score = tl.dot(q1, tl.trans(qk_use_32_4_1_key))
            qk_use_32_4_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 256 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_4_0_offset, qk_use_32_4_0_score)
            qk_use_32_4_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 288 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_4_1_offset, qk_use_32_4_1_score)
            qk_pre_32_6_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 384) // PAGE
            ).to(tl.int64)
            qk_pre_32_6_0_base = (
                RAW_K + qk_pre_32_6_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_6_0_ptr = tl.make_block_ptr(
                qk_pre_32_6_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_6_0_ptr, native_key_0, [PAGE, 256])
            qk_pre_32_6_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 416) // PAGE
            ).to(tl.int64)
            qk_pre_32_6_1_base = (
                RAW_K + qk_pre_32_6_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_6_1_ptr = tl.make_block_ptr(
                qk_pre_32_6_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_6_1_ptr, native_key_1, [PAGE, 256])
            qk_use_32_5_0_key = tle.dsa.to_tensor(native_key_2, writable=False)
            qk_use_32_5_0_score = tl.dot(q1, tl.trans(qk_use_32_5_0_key))
            qk_use_32_5_1_key = tle.dsa.to_tensor(native_key_3, writable=False)
            qk_use_32_5_1_score = tl.dot(q1, tl.trans(qk_use_32_5_1_key))
            qk_use_32_5_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 320 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_5_0_offset, qk_use_32_5_0_score)
            qk_use_32_5_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 352 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_5_1_offset, qk_use_32_5_1_score)
            qk_pre_32_7_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 448) // PAGE
            ).to(tl.int64)
            qk_pre_32_7_0_base = (
                RAW_K + qk_pre_32_7_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_7_0_ptr = tl.make_block_ptr(
                qk_pre_32_7_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_7_0_ptr, native_key_2, [PAGE, 256])
            qk_pre_32_7_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 480) // PAGE
            ).to(tl.int64)
            qk_pre_32_7_1_base = (
                RAW_K + qk_pre_32_7_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_32_7_1_ptr = tl.make_block_ptr(
                qk_pre_32_7_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_32_7_1_ptr, native_key_3, [PAGE, 256])
            qk_use_32_6_0_key = tle.dsa.to_tensor(native_key_0, writable=False)
            qk_use_32_6_0_score = tl.dot(q1, tl.trans(qk_use_32_6_0_key))
            qk_use_32_6_1_key = tle.dsa.to_tensor(native_key_1, writable=False)
            qk_use_32_6_1_score = tl.dot(q1, tl.trans(qk_use_32_6_1_key))
            qk_use_32_6_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 384 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_6_0_offset, qk_use_32_6_0_score)
            qk_use_32_6_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 416 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_6_1_offset, qk_use_32_6_1_score)
            qk_use_32_7_0_key = tle.dsa.to_tensor(native_key_2, writable=False)
            qk_use_32_7_0_score = tl.dot(q1, tl.trans(qk_use_32_7_0_key))
            qk_use_32_7_1_key = tle.dsa.to_tensor(native_key_3, writable=False)
            qk_use_32_7_1_score = tl.dot(q1, tl.trans(qk_use_32_7_1_key))
            qk_use_32_7_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 448 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_7_0_offset, qk_use_32_7_0_score)
            qk_use_32_7_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 480 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_32_7_1_offset, qk_use_32_7_1_score)
        else:
            qk_pre_16_0_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 0) // PAGE
            ).to(tl.int64)
            qk_pre_16_0_0_base = (
                RAW_K + qk_pre_16_0_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_0_0_ptr = tl.make_block_ptr(
                qk_pre_16_0_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_0_0_ptr, native_key_0, [PAGE, 256])
            qk_pre_16_0_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 16) // PAGE
            ).to(tl.int64)
            qk_pre_16_0_1_base = (
                RAW_K + qk_pre_16_0_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_0_1_ptr = tl.make_block_ptr(
                qk_pre_16_0_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_0_1_ptr, native_key_1, [PAGE, 256])
            qk_pre_16_1_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 32) // PAGE
            ).to(tl.int64)
            qk_pre_16_1_0_base = (
                RAW_K + qk_pre_16_1_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_1_0_ptr = tl.make_block_ptr(
                qk_pre_16_1_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_1_0_ptr, native_key_2, [PAGE, 256])
            qk_pre_16_1_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 48) // PAGE
            ).to(tl.int64)
            qk_pre_16_1_1_base = (
                RAW_K + qk_pre_16_1_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_1_1_ptr = tl.make_block_ptr(
                qk_pre_16_1_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_1_1_ptr, native_key_3, [PAGE, 256])
            qk_use_16_0_0_key = tle.dsa.to_tensor(native_key_0, writable=False)
            qk_use_16_0_0_score = tl.dot(q1, tl.trans(qk_use_16_0_0_key))
            qk_use_16_0_1_key = tle.dsa.to_tensor(native_key_1, writable=False)
            qk_use_16_0_1_score = tl.dot(q1, tl.trans(qk_use_16_0_1_key))
            qk_use_16_0_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 0 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_0_0_offset, qk_use_16_0_0_score)
            qk_use_16_0_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 16 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_0_1_offset, qk_use_16_0_1_score)
            qk_pre_16_2_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 64) // PAGE
            ).to(tl.int64)
            qk_pre_16_2_0_base = (
                RAW_K + qk_pre_16_2_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_2_0_ptr = tl.make_block_ptr(
                qk_pre_16_2_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_2_0_ptr, native_key_0, [PAGE, 256])
            qk_pre_16_2_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 80) // PAGE
            ).to(tl.int64)
            qk_pre_16_2_1_base = (
                RAW_K + qk_pre_16_2_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_2_1_ptr = tl.make_block_ptr(
                qk_pre_16_2_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_2_1_ptr, native_key_1, [PAGE, 256])
            qk_use_16_1_0_key = tle.dsa.to_tensor(native_key_2, writable=False)
            qk_use_16_1_0_score = tl.dot(q1, tl.trans(qk_use_16_1_0_key))
            qk_use_16_1_1_key = tle.dsa.to_tensor(native_key_3, writable=False)
            qk_use_16_1_1_score = tl.dot(q1, tl.trans(qk_use_16_1_1_key))
            qk_use_16_1_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 32 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_1_0_offset, qk_use_16_1_0_score)
            qk_use_16_1_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 48 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_1_1_offset, qk_use_16_1_1_score)
            qk_pre_16_3_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 96) // PAGE
            ).to(tl.int64)
            qk_pre_16_3_0_base = (
                RAW_K + qk_pre_16_3_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_3_0_ptr = tl.make_block_ptr(
                qk_pre_16_3_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_3_0_ptr, native_key_2, [PAGE, 256])
            qk_pre_16_3_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 112) // PAGE
            ).to(tl.int64)
            qk_pre_16_3_1_base = (
                RAW_K + qk_pre_16_3_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_3_1_ptr = tl.make_block_ptr(
                qk_pre_16_3_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_3_1_ptr, native_key_3, [PAGE, 256])
            qk_use_16_2_0_key = tle.dsa.to_tensor(native_key_0, writable=False)
            qk_use_16_2_0_score = tl.dot(q1, tl.trans(qk_use_16_2_0_key))
            qk_use_16_2_1_key = tle.dsa.to_tensor(native_key_1, writable=False)
            qk_use_16_2_1_score = tl.dot(q1, tl.trans(qk_use_16_2_1_key))
            qk_use_16_2_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 64 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_2_0_offset, qk_use_16_2_0_score)
            qk_use_16_2_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 80 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_2_1_offset, qk_use_16_2_1_score)
            qk_pre_16_4_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 128) // PAGE
            ).to(tl.int64)
            qk_pre_16_4_0_base = (
                RAW_K + qk_pre_16_4_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_4_0_ptr = tl.make_block_ptr(
                qk_pre_16_4_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_4_0_ptr, native_key_0, [PAGE, 256])
            qk_pre_16_4_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 144) // PAGE
            ).to(tl.int64)
            qk_pre_16_4_1_base = (
                RAW_K + qk_pre_16_4_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_4_1_ptr = tl.make_block_ptr(
                qk_pre_16_4_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_4_1_ptr, native_key_1, [PAGE, 256])
            qk_use_16_3_0_key = tle.dsa.to_tensor(native_key_2, writable=False)
            qk_use_16_3_0_score = tl.dot(q1, tl.trans(qk_use_16_3_0_key))
            qk_use_16_3_1_key = tle.dsa.to_tensor(native_key_3, writable=False)
            qk_use_16_3_1_score = tl.dot(q1, tl.trans(qk_use_16_3_1_key))
            qk_use_16_3_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 96 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_3_0_offset, qk_use_16_3_0_score)
            qk_use_16_3_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 112 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_3_1_offset, qk_use_16_3_1_score)
            qk_pre_16_5_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 160) // PAGE
            ).to(tl.int64)
            qk_pre_16_5_0_base = (
                RAW_K + qk_pre_16_5_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_5_0_ptr = tl.make_block_ptr(
                qk_pre_16_5_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_5_0_ptr, native_key_2, [PAGE, 256])
            qk_pre_16_5_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 176) // PAGE
            ).to(tl.int64)
            qk_pre_16_5_1_base = (
                RAW_K + qk_pre_16_5_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_5_1_ptr = tl.make_block_ptr(
                qk_pre_16_5_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_5_1_ptr, native_key_3, [PAGE, 256])
            qk_use_16_4_0_key = tle.dsa.to_tensor(native_key_0, writable=False)
            qk_use_16_4_0_score = tl.dot(q1, tl.trans(qk_use_16_4_0_key))
            qk_use_16_4_1_key = tle.dsa.to_tensor(native_key_1, writable=False)
            qk_use_16_4_1_score = tl.dot(q1, tl.trans(qk_use_16_4_1_key))
            qk_use_16_4_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 128 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_4_0_offset, qk_use_16_4_0_score)
            qk_use_16_4_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 144 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_4_1_offset, qk_use_16_4_1_score)
            qk_pre_16_6_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 192) // PAGE
            ).to(tl.int64)
            qk_pre_16_6_0_base = (
                RAW_K + qk_pre_16_6_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_6_0_ptr = tl.make_block_ptr(
                qk_pre_16_6_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_6_0_ptr, native_key_0, [PAGE, 256])
            qk_pre_16_6_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 208) // PAGE
            ).to(tl.int64)
            qk_pre_16_6_1_base = (
                RAW_K + qk_pre_16_6_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_6_1_ptr = tl.make_block_ptr(
                qk_pre_16_6_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_6_1_ptr, native_key_1, [PAGE, 256])
            qk_use_16_5_0_key = tle.dsa.to_tensor(native_key_2, writable=False)
            qk_use_16_5_0_score = tl.dot(q1, tl.trans(qk_use_16_5_0_key))
            qk_use_16_5_1_key = tle.dsa.to_tensor(native_key_3, writable=False)
            qk_use_16_5_1_score = tl.dot(q1, tl.trans(qk_use_16_5_1_key))
            qk_use_16_5_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 160 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_5_0_offset, qk_use_16_5_0_score)
            qk_use_16_5_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 176 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_5_1_offset, qk_use_16_5_1_score)
            qk_pre_16_7_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 224) // PAGE
            ).to(tl.int64)
            qk_pre_16_7_0_base = (
                RAW_K + qk_pre_16_7_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_7_0_ptr = tl.make_block_ptr(
                qk_pre_16_7_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_7_0_ptr, native_key_2, [PAGE, 256])
            qk_pre_16_7_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 240) // PAGE
            ).to(tl.int64)
            qk_pre_16_7_1_base = (
                RAW_K + qk_pre_16_7_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_7_1_ptr = tl.make_block_ptr(
                qk_pre_16_7_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_7_1_ptr, native_key_3, [PAGE, 256])
            qk_use_16_6_0_key = tle.dsa.to_tensor(native_key_0, writable=False)
            qk_use_16_6_0_score = tl.dot(q1, tl.trans(qk_use_16_6_0_key))
            qk_use_16_6_1_key = tle.dsa.to_tensor(native_key_1, writable=False)
            qk_use_16_6_1_score = tl.dot(q1, tl.trans(qk_use_16_6_1_key))
            qk_use_16_6_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 192 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_6_0_offset, qk_use_16_6_0_score)
            qk_use_16_6_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 208 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_6_1_offset, qk_use_16_6_1_score)
            qk_pre_16_8_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 256) // PAGE
            ).to(tl.int64)
            qk_pre_16_8_0_base = (
                RAW_K + qk_pre_16_8_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_8_0_ptr = tl.make_block_ptr(
                qk_pre_16_8_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_8_0_ptr, native_key_0, [PAGE, 256])
            qk_pre_16_8_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 272) // PAGE
            ).to(tl.int64)
            qk_pre_16_8_1_base = (
                RAW_K + qk_pre_16_8_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_8_1_ptr = tl.make_block_ptr(
                qk_pre_16_8_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_8_1_ptr, native_key_1, [PAGE, 256])
            qk_use_16_7_0_key = tle.dsa.to_tensor(native_key_2, writable=False)
            qk_use_16_7_0_score = tl.dot(q1, tl.trans(qk_use_16_7_0_key))
            qk_use_16_7_1_key = tle.dsa.to_tensor(native_key_3, writable=False)
            qk_use_16_7_1_score = tl.dot(q1, tl.trans(qk_use_16_7_1_key))
            qk_use_16_7_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 224 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_7_0_offset, qk_use_16_7_0_score)
            qk_use_16_7_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 240 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_7_1_offset, qk_use_16_7_1_score)
            qk_pre_16_9_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 288) // PAGE
            ).to(tl.int64)
            qk_pre_16_9_0_base = (
                RAW_K + qk_pre_16_9_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_9_0_ptr = tl.make_block_ptr(
                qk_pre_16_9_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_9_0_ptr, native_key_2, [PAGE, 256])
            qk_pre_16_9_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 304) // PAGE
            ).to(tl.int64)
            qk_pre_16_9_1_base = (
                RAW_K + qk_pre_16_9_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_9_1_ptr = tl.make_block_ptr(
                qk_pre_16_9_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_9_1_ptr, native_key_3, [PAGE, 256])
            qk_use_16_8_0_key = tle.dsa.to_tensor(native_key_0, writable=False)
            qk_use_16_8_0_score = tl.dot(q1, tl.trans(qk_use_16_8_0_key))
            qk_use_16_8_1_key = tle.dsa.to_tensor(native_key_1, writable=False)
            qk_use_16_8_1_score = tl.dot(q1, tl.trans(qk_use_16_8_1_key))
            qk_use_16_8_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 256 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_8_0_offset, qk_use_16_8_0_score)
            qk_use_16_8_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 272 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_8_1_offset, qk_use_16_8_1_score)
            qk_pre_16_10_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 320) // PAGE
            ).to(tl.int64)
            qk_pre_16_10_0_base = (
                RAW_K + qk_pre_16_10_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_10_0_ptr = tl.make_block_ptr(
                qk_pre_16_10_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_10_0_ptr, native_key_0, [PAGE, 256])
            qk_pre_16_10_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 336) // PAGE
            ).to(tl.int64)
            qk_pre_16_10_1_base = (
                RAW_K + qk_pre_16_10_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_10_1_ptr = tl.make_block_ptr(
                qk_pre_16_10_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_10_1_ptr, native_key_1, [PAGE, 256])
            qk_use_16_9_0_key = tle.dsa.to_tensor(native_key_2, writable=False)
            qk_use_16_9_0_score = tl.dot(q1, tl.trans(qk_use_16_9_0_key))
            qk_use_16_9_1_key = tle.dsa.to_tensor(native_key_3, writable=False)
            qk_use_16_9_1_score = tl.dot(q1, tl.trans(qk_use_16_9_1_key))
            qk_use_16_9_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 288 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_9_0_offset, qk_use_16_9_0_score)
            qk_use_16_9_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 304 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_9_1_offset, qk_use_16_9_1_score)
            qk_pre_16_11_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 352) // PAGE
            ).to(tl.int64)
            qk_pre_16_11_0_base = (
                RAW_K + qk_pre_16_11_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_11_0_ptr = tl.make_block_ptr(
                qk_pre_16_11_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_11_0_ptr, native_key_2, [PAGE, 256])
            qk_pre_16_11_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 368) // PAGE
            ).to(tl.int64)
            qk_pre_16_11_1_base = (
                RAW_K + qk_pre_16_11_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_11_1_ptr = tl.make_block_ptr(
                qk_pre_16_11_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_11_1_ptr, native_key_3, [PAGE, 256])
            qk_use_16_10_0_key = tle.dsa.to_tensor(native_key_0, writable=False)
            qk_use_16_10_0_score = tl.dot(q1, tl.trans(qk_use_16_10_0_key))
            qk_use_16_10_1_key = tle.dsa.to_tensor(native_key_1, writable=False)
            qk_use_16_10_1_score = tl.dot(q1, tl.trans(qk_use_16_10_1_key))
            qk_use_16_10_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 320 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_10_0_offset, qk_use_16_10_0_score)
            qk_use_16_10_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 336 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_10_1_offset, qk_use_16_10_1_score)
            qk_pre_16_12_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 384) // PAGE
            ).to(tl.int64)
            qk_pre_16_12_0_base = (
                RAW_K + qk_pre_16_12_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_12_0_ptr = tl.make_block_ptr(
                qk_pre_16_12_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_12_0_ptr, native_key_0, [PAGE, 256])
            qk_pre_16_12_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 400) // PAGE
            ).to(tl.int64)
            qk_pre_16_12_1_base = (
                RAW_K + qk_pre_16_12_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_12_1_ptr = tl.make_block_ptr(
                qk_pre_16_12_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_12_1_ptr, native_key_1, [PAGE, 256])
            qk_use_16_11_0_key = tle.dsa.to_tensor(native_key_2, writable=False)
            qk_use_16_11_0_score = tl.dot(q1, tl.trans(qk_use_16_11_0_key))
            qk_use_16_11_1_key = tle.dsa.to_tensor(native_key_3, writable=False)
            qk_use_16_11_1_score = tl.dot(q1, tl.trans(qk_use_16_11_1_key))
            qk_use_16_11_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 352 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_11_0_offset, qk_use_16_11_0_score)
            qk_use_16_11_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 368 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_11_1_offset, qk_use_16_11_1_score)
            qk_pre_16_13_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 416) // PAGE
            ).to(tl.int64)
            qk_pre_16_13_0_base = (
                RAW_K + qk_pre_16_13_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_13_0_ptr = tl.make_block_ptr(
                qk_pre_16_13_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_13_0_ptr, native_key_2, [PAGE, 256])
            qk_pre_16_13_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 432) // PAGE
            ).to(tl.int64)
            qk_pre_16_13_1_base = (
                RAW_K + qk_pre_16_13_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_13_1_ptr = tl.make_block_ptr(
                qk_pre_16_13_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_13_1_ptr, native_key_3, [PAGE, 256])
            qk_use_16_12_0_key = tle.dsa.to_tensor(native_key_0, writable=False)
            qk_use_16_12_0_score = tl.dot(q1, tl.trans(qk_use_16_12_0_key))
            qk_use_16_12_1_key = tle.dsa.to_tensor(native_key_1, writable=False)
            qk_use_16_12_1_score = tl.dot(q1, tl.trans(qk_use_16_12_1_key))
            qk_use_16_12_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 384 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_12_0_offset, qk_use_16_12_0_score)
            qk_use_16_12_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 400 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_12_1_offset, qk_use_16_12_1_score)
            qk_pre_16_14_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 448) // PAGE
            ).to(tl.int64)
            qk_pre_16_14_0_base = (
                RAW_K + qk_pre_16_14_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_14_0_ptr = tl.make_block_ptr(
                qk_pre_16_14_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_14_0_ptr, native_key_0, [PAGE, 256])
            qk_pre_16_14_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 464) // PAGE
            ).to(tl.int64)
            qk_pre_16_14_1_base = (
                RAW_K + qk_pre_16_14_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_14_1_ptr = tl.make_block_ptr(
                qk_pre_16_14_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_14_1_ptr, native_key_1, [PAGE, 256])
            qk_use_16_13_0_key = tle.dsa.to_tensor(native_key_2, writable=False)
            qk_use_16_13_0_score = tl.dot(q1, tl.trans(qk_use_16_13_0_key))
            qk_use_16_13_1_key = tle.dsa.to_tensor(native_key_3, writable=False)
            qk_use_16_13_1_score = tl.dot(q1, tl.trans(qk_use_16_13_1_key))
            qk_use_16_13_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 416 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_13_0_offset, qk_use_16_13_0_score)
            qk_use_16_13_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 432 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_13_1_offset, qk_use_16_13_1_score)
            qk_pre_16_15_0_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 480) // PAGE
            ).to(tl.int64)
            qk_pre_16_15_0_base = (
                RAW_K + qk_pre_16_15_0_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_15_0_ptr = tl.make_block_ptr(
                qk_pre_16_15_0_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_15_0_ptr, native_key_2, [PAGE, 256])
            qk_pre_16_15_1_physical = tl.load(
                TABLE + batch.to(tl.int64) * TS + (begin + 496) // PAGE
            ).to(tl.int64)
            qk_pre_16_15_1_base = (
                RAW_K + qk_pre_16_15_1_physical * KP + kh.to(tl.int64) * KH
            )
            qk_pre_16_15_1_ptr = tl.make_block_ptr(
                qk_pre_16_15_1_base,
                (PAGE, 256),
                (KR, 1),
                (0, 0),
                (PAGE, 256),
                (1, 0),
            )
            tle.dsa.copy(qk_pre_16_15_1_ptr, native_key_3, [PAGE, 256])
            qk_use_16_14_0_key = tle.dsa.to_tensor(native_key_0, writable=False)
            qk_use_16_14_0_score = tl.dot(q1, tl.trans(qk_use_16_14_0_key))
            qk_use_16_14_1_key = tle.dsa.to_tensor(native_key_1, writable=False)
            qk_use_16_14_1_score = tl.dot(q1, tl.trans(qk_use_16_14_1_key))
            qk_use_16_14_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 448 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_14_0_offset, qk_use_16_14_0_score)
            qk_use_16_14_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 464 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_14_1_offset, qk_use_16_14_1_score)
            qk_use_16_15_0_key = tle.dsa.to_tensor(native_key_2, writable=False)
            qk_use_16_15_0_score = tl.dot(q1, tl.trans(qk_use_16_15_0_key))
            qk_use_16_15_1_key = tle.dsa.to_tensor(native_key_3, writable=False)
            qk_use_16_15_1_score = tl.dot(q1, tl.trans(qk_use_16_15_1_key))
            qk_use_16_15_0_offset = (
                tl.arange(0, BM)[:, None] * BS + 480 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_15_0_offset, qk_use_16_15_0_score)
            qk_use_16_15_1_offset = (
                tl.arange(0, BM)[:, None] * BS + 496 + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + qk_use_16_15_1_offset, qk_use_16_15_1_score)
    elif (
        H // HK == 8 and CONTIG_K == 2560 and (SPLITS == 1) and Q1_STREAM
    ) and stack <= PAGE:
        tail_key_base = K + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin) * 256
        tail_key_ptr = tl.make_block_ptr(
            tail_key_base,
            (PAGE, 256),
            (256, 1),
            (0, 0),
            (PAGE, 256),
            (1, 0),
        )
        tle.dsa.copy(tail_key_ptr, native_key_0, [PAGE, 256])
        tail_key = tle.dsa.to_tensor(native_key_0, writable=False)
        tail_score = tl.dot(q1, tl.trans(tail_key))
        tail_offsets = tl.arange(0, BM)[:, None] * BS + tl.arange(0, PAGE)[None, :]
        tl.store(scratch + tail_offsets, tail_score)
    else:
        native_pair_end = (stack + 2 * PAGE - 1) // (2 * PAGE) * (2 * PAGE)
        for native_ck in range(0, native_pair_end, 2 * PAGE):
            if begin + native_ck + PAGE <= klen:
                native_k0_physical = tl.load(
                    TABLE + batch.to(tl.int64) * TS + (begin + native_ck) // PAGE
                ).to(tl.int64)
                native_k0_full_base = (
                    RAW_K + native_k0_physical * KP + kh.to(tl.int64) * KH
                )
                native_k0_full_ptr = tl.make_block_ptr(
                    native_k0_full_base,
                    (PAGE, 256),
                    (KR, 1),
                    (0, 0),
                    (PAGE, 256),
                    (1, 0),
                )
                tle.dsa.copy(native_k0_full_ptr, native_key_0, [PAGE, 256])
            else:
                native_k0_tail_base = (
                    K
                    + ((batch.to(tl.int64) * HK + kh) * CONTIG_K + begin + native_ck)
                    * 256
                )
                native_k0_tail_ptr = tl.make_block_ptr(
                    native_k0_tail_base,
                    (PAGE, 256),
                    (256, 1),
                    (0, 0),
                    (PAGE, 256),
                    (1, 0),
                )
                tle.dsa.copy(native_k0_tail_ptr, native_key_0, [PAGE, 256])
            if begin + (native_ck + PAGE) + PAGE <= klen:
                native_k1_physical = tl.load(
                    TABLE
                    + batch.to(tl.int64) * TS
                    + (begin + (native_ck + PAGE)) // PAGE
                ).to(tl.int64)
                native_k1_full_base = (
                    RAW_K + native_k1_physical * KP + kh.to(tl.int64) * KH
                )
                native_k1_full_ptr = tl.make_block_ptr(
                    native_k1_full_base,
                    (PAGE, 256),
                    (KR, 1),
                    (0, 0),
                    (PAGE, 256),
                    (1, 0),
                )
                tle.dsa.copy(native_k1_full_ptr, native_key_1, [PAGE, 256])
            else:
                native_k1_tail_base = (
                    K
                    + (
                        (batch.to(tl.int64) * HK + kh) * CONTIG_K
                        + begin
                        + (native_ck + PAGE)
                    )
                    * 256
                )
                native_k1_tail_ptr = tl.make_block_ptr(
                    native_k1_tail_base,
                    (PAGE, 256),
                    (256, 1),
                    (0, 0),
                    (PAGE, 256),
                    (1, 0),
                )
                tle.dsa.copy(native_k1_tail_ptr, native_key_1, [PAGE, 256])
            native_k_tensor_0 = tle.dsa.to_tensor(native_key_0, writable=False)
            native_k_tensor_1 = tle.dsa.to_tensor(native_key_1, writable=False)
            native_qk_0 = tl.dot(q1, tl.trans(native_k_tensor_0))
            native_qk_1 = tl.dot(q1, tl.trans(native_k_tensor_1))
            native_offsets = (
                tl.arange(0, BM)[:, None] * BS + native_ck + tl.arange(0, PAGE)[None, :]
            )
            tl.store(scratch + native_offsets, native_qk_0)
            tl.store(scratch + native_offsets + PAGE, native_qk_1)


@triton.jit
def _native_update_softmax(
    begin,
    group,
    head_rows,
    headstart,
    klen,
    prob,
    qlen,
    qs,
    qt,
    row_end_buf,
    scratch,
    slot,
    stack,
    state_buf,
    sub,
    x2_vf,
    BM: tl.constexpr,
    BS: tl.constexpr,
    CAUSAL: tl.constexpr,
    CONTIG_K: tl.constexpr,
    SCALE: tl.constexpr,
    SINGLE_HEAD_INDEX: tl.constexpr,
    SOFTMAX_VM_256: tl.constexpr,
    VM: tl.constexpr,
):
    """Select the softmax width for the active KV stack."""
    if stack <= 256:
        if CONTIG_K >= 2048 and BM == 128:
            _softmax_stack(
                scratch,
                prob,
                state_buf,
                begin,
                stack,
                slot,
                sub,
                qs,
                qt,
                head_rows,
                headstart,
                group,
                qlen,
                klen,
                x2_vf,
                BM,
                BS,
                256,
                CAUSAL,
                SCALE,
                SOFTMAX_VM_256,
                SINGLE_HEAD_INDEX,
                row_end_buf,
            )
        else:
            _softmax_rows_dispatch(
                scratch,
                prob,
                state_buf,
                begin,
                stack,
                slot,
                sub,
                qs,
                qt,
                head_rows,
                headstart,
                group,
                qlen,
                klen,
                x2_vf,
                BM,
                BS,
                256,
                CAUSAL,
                SCALE,
                SOFTMAX_VM_256,
                SINGLE_HEAD_INDEX,
                row_end_buf,
            )
    elif CONTIG_K >= 2048 and BM == 128:
        _softmax_stack(
            scratch,
            prob,
            state_buf,
            begin,
            stack,
            slot,
            sub,
            qs,
            qt,
            head_rows,
            headstart,
            group,
            qlen,
            klen,
            x2_vf,
            BM,
            BS,
            BS,
            CAUSAL,
            SCALE,
            VM,
            SINGLE_HEAD_INDEX,
            row_end_buf,
        )
    else:
        _softmax_rows_dispatch(
            scratch,
            prob,
            state_buf,
            begin,
            stack,
            slot,
            sub,
            qs,
            qt,
            head_rows,
            headstart,
            group,
            qlen,
            klen,
            x2_vf,
            BM,
            BS,
            BS,
            CAUSAL,
            SCALE,
            VM,
            SINGLE_HEAD_INDEX,
            row_end_buf,
        )


@triton.jit
def _native_update_accumulator(
    acc_buf,
    prevslot,
    pvwork,
    state_buf,
    worker,
    x4_vf,
    ACCWORK,
    ACC_UB_ROWS: tl.constexpr,
    BM: tl.constexpr,
    O_VM: tl.constexpr,
):
    """Accumulate PV using the kernel-selected UB rows and the existing GM remainder."""
    if ACC_UB_ROWS > 0:
        for x4_vs in range(0, ACC_UB_ROWS, O_VM):
            x4_av = tle.dsa.subview(state_buf, [2 + prevslot, x4_vs], [1, O_VM], [1, 1])
            x4_acc = tle.dsa.subview(acc_buf, [x4_vs, 0], [O_VM, 256], [1, 1])
            x4_alpha = tl.reshape(tle.dsa.to_tensor(x4_av, writable=False), (O_VM,))
            x4_old = tle.dsa.to_tensor(x4_acc, writable=False)
            x4_pv = tl.load(
                pvwork
                + (x4_vf + x4_vs + tl.arange(0, O_VM))[:, None] * 256
                + tl.arange(0, 256)[None, :],
                volatile=True,
            )
            tle.dsa.to_buffer(x4_old * x4_alpha[:, None] + x4_pv, bind_buffer=x4_acc)
    for x4_vs in range(ACC_UB_ROWS, BM // 2, O_VM):
        x4_av = tle.dsa.subview(state_buf, [2 + prevslot, x4_vs], [1, O_VM], [1, 1])
        x4_ab = (
            ACCWORK
            + worker * 128 * 256
            + (x4_vf + x4_vs + tl.arange(0, O_VM))[:, None] * 256
            + tl.arange(0, 256)[None, :]
        )
        x4_alpha = tl.reshape(tle.dsa.to_tensor(x4_av, writable=False), (O_VM,))
        x4_old = tl.load(x4_ab, volatile=True)
        x4_pv = tl.load(
            pvwork
            + (x4_vf + x4_vs + tl.arange(0, O_VM))[:, None] * 256
            + tl.arange(0, 256)[None, :],
            volatile=True,
        )
        tl.store(x4_ab, x4_old * x4_alpha[:, None] + x4_pv)
