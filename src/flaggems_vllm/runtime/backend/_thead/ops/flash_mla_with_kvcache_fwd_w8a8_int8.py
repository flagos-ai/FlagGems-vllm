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

import copy
import math
from dataclasses import fields
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.backends.ppu.compiler import HGGCOptions
from triton.language import core

from flaggems_vllm import runtime
from flaggems_vllm.runtime.backend._thead.fused.attention import (
    _flash_int8_merge_splits,
)
from flaggems_vllm.utils import libentry, libtuner


@core.extern
def native_mla_probability_reciprocal(probability_scale, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [probability_scale],
        {(core.float32,): ("__builtin_ppu_rcpf", core.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def quantize_mla_probabilities(normalized_probabilities):
    # Direct conversion preserves round-half-up without a float floor round trip.
    coarse, residual = tl.inline_asm_elementwise(
        """{
        .reg .s32 coarse, residual;
        .reg .f32 rounded, coarse_fp, error;
        ppu.add.f32 rounded, $2, 0f3F000000;
        ppu.cvt.rmi.s32.f32 coarse, rounded;
        ppu.cvt.rn.f32.s32 coarse_fp, coarse;
        ppu.sub.f32 error, $2, coarse_fp;
        ppu.fma.rn.f32 rounded, error, 0f42FE0000, 0f3F000000;
        ppu.cvt.rmi.s32.f32 residual, rounded;
        ppu.mov.b32 $0, coarse;
        ppu.mov.b32 $1, residual;
        }""",
        "=r,=r,f",
        [normalized_probabilities],
        dtype=(tl.int32, tl.int32),
        is_pure=True,
        pack=1,
    )
    return coarse.to(tl.int8), residual.to(tl.int8)


P14_PACKED_ASM = tl.constexpr(
    """{
.reg .s32 coarse<4>, residual<4>;
.reg .f32 rounded<4>, coarse_fp<4>, error<4>;
.reg .b32 packed0, packed1;
ppu.add.f32 rounded0, $2, 0f3F000000;
ppu.cvt.rmi.s32.f32 coarse0, rounded0;
ppu.cvt.rn.f32.s32 coarse_fp0, coarse0;
ppu.sub.f32 error0, $2, coarse_fp0;
ppu.fma.rn.f32 rounded0, error0, 0f42FE0000, 0f3F000000;
ppu.cvt.rmi.s32.f32 residual0, rounded0;
ppu.add.f32 rounded1, $3, 0f3F000000;
ppu.cvt.rmi.s32.f32 coarse1, rounded1;
ppu.cvt.rn.f32.s32 coarse_fp1, coarse1;
ppu.sub.f32 error1, $3, coarse_fp1;
ppu.fma.rn.f32 rounded1, error1, 0f42FE0000, 0f3F000000;
ppu.cvt.rmi.s32.f32 residual1, rounded1;
ppu.add.f32 rounded2, $4, 0f3F000000;
ppu.cvt.rmi.s32.f32 coarse2, rounded2;
ppu.cvt.rn.f32.s32 coarse_fp2, coarse2;
ppu.sub.f32 error2, $4, coarse_fp2;
ppu.fma.rn.f32 rounded2, error2, 0f42FE0000, 0f3F000000;
ppu.cvt.rmi.s32.f32 residual2, rounded2;
ppu.add.f32 rounded3, $5, 0f3F000000;
ppu.cvt.rmi.s32.f32 coarse3, rounded3;
ppu.cvt.rn.f32.s32 coarse_fp3, coarse3;
ppu.sub.f32 error3, $5, coarse_fp3;
ppu.fma.rn.f32 rounded3, error3, 0f42FE0000, 0f3F000000;
ppu.cvt.rmi.s32.f32 residual3, rounded3;
ppu.prmt.b32 packed0, coarse0, coarse1, 0x0040;
ppu.prmt.b32 packed1, coarse2, coarse3, 0x0040;
ppu.prmt.b32 $0, packed0, packed1, 0x5410;
ppu.prmt.b32 packed0, residual0, residual1, 0x0040;
ppu.prmt.b32 packed1, residual2, residual3, 0x0040;
ppu.prmt.b32 $1, packed0, packed1, 0x5410;
}"""
)

P14_NORMALIZED_PACKED_ASM = tl.constexpr(
    """{
.reg .s32 coarse<4>, residual<4>;
.reg .f32 rounded<4>, coarse_fp<4>, error<4>;
.reg .b32 packed0, packed1;
ppu.add.f32 rounded0, $2, 0f3F000000;
ppu.cvt.rmi.f32.f32 coarse_fp0, rounded0;
ppu.cvt.rzi.s32.f32 coarse0, coarse_fp0;
ppu.sub.f32 error0, $2, coarse_fp0;
ppu.fma.rn.f32 rounded0, error0, 0f42FE0000, 0f3F000000;
ppu.cvt.rmi.s32.f32 residual0, rounded0;
ppu.add.f32 rounded1, $3, 0f3F000000;
ppu.cvt.rmi.f32.f32 coarse_fp1, rounded1;
ppu.cvt.rzi.s32.f32 coarse1, coarse_fp1;
ppu.sub.f32 error1, $3, coarse_fp1;
ppu.fma.rn.f32 rounded1, error1, 0f42FE0000, 0f3F000000;
ppu.cvt.rmi.s32.f32 residual1, rounded1;
ppu.add.f32 rounded2, $4, 0f3F000000;
ppu.cvt.rmi.f32.f32 coarse_fp2, rounded2;
ppu.cvt.rzi.s32.f32 coarse2, coarse_fp2;
ppu.sub.f32 error2, $4, coarse_fp2;
ppu.fma.rn.f32 rounded2, error2, 0f42FE0000, 0f3F000000;
ppu.cvt.rmi.s32.f32 residual2, rounded2;
ppu.add.f32 rounded3, $5, 0f3F000000;
ppu.cvt.rmi.f32.f32 coarse_fp3, rounded3;
ppu.cvt.rzi.s32.f32 coarse3, coarse_fp3;
ppu.sub.f32 error3, $5, coarse_fp3;
ppu.fma.rn.f32 rounded3, error3, 0f42FE0000, 0f3F000000;
ppu.cvt.rmi.s32.f32 residual3, rounded3;
ppu.prmt.b32 packed0, coarse0, coarse1, 0x0040;
ppu.prmt.b32 packed1, coarse2, coarse3, 0x0040;
ppu.prmt.b32 $0, packed0, packed1, 0x5410;
ppu.prmt.b32 packed0, residual0, residual1, 0x0040;
ppu.prmt.b32 packed1, residual2, residual3, 0x0040;
ppu.prmt.b32 $1, packed0, packed1, 0x5410;
}"""
)


P15_NORMALIZED_PACKED_ASM = tl.constexpr(
    """{
.reg .s32 quantized<4>;
.reg .f32 rounded<4>;
.reg .b32 pair0, pair1;
ppu.fma.rn.f32 rounded0, $2, 0f43800000, 0f3F000000;
ppu.cvt.rmi.s32.f32 quantized0, rounded0;
ppu.fma.rn.f32 rounded1, $3, 0f43800000, 0f3F000000;
ppu.cvt.rmi.s32.f32 quantized1, rounded1;
ppu.fma.rn.f32 rounded2, $4, 0f43800000, 0f3F000000;
ppu.cvt.rmi.s32.f32 quantized2, rounded2;
ppu.fma.rn.f32 rounded3, $5, 0f43800000, 0f3F000000;
ppu.cvt.rmi.s32.f32 quantized3, rounded3;
ppu.prmt.b32 pair0, quantized0, quantized1, 0x5410;
ppu.prmt.b32 pair1, quantized2, quantized3, 0x5410;
ppu.prmt.b32 $1, pair0, pair1, 0x6420;
ppu.add.u32 pair0, pair0, 0x00800080;
ppu.add.u32 pair1, pair1, 0x00800080;
ppu.prmt.b32 $0, pair0, pair1, 0x7531;
}"""
)


@triton.jit
def quantize_mla_probabilities_packed(
    normalized_probabilities,
    NORMALIZED: tl.constexpr = False,
    RADIX: tl.constexpr = 127,
):
    tl.static_assert(RADIX == 127 or RADIX == 256)
    if RADIX == 256:
        tl.static_assert(NORMALIZED)
        # MLA normalization bounds finite inputs by 127 plus FP32 roundoff.
        assembly: tl.constexpr = P15_NORMALIZED_PACKED_ASM
    else:
        assembly: tl.constexpr = (
            P14_NORMALIZED_PACKED_ASM if NORMALIZED else P14_PACKED_ASM
        )
    return tl.inline_asm_elementwise(
        assembly,
        "=r,=r,f,f,f,f",
        [normalized_probabilities],
        dtype=(tl.int8, tl.int8),
        is_pure=True,
        pack=4,
    )


@triton.jit
def _split_mla_features(values, rows: tl.constexpr, half_width: tl.constexpr):
    return tl.split(tl.permute(tl.reshape(values, (rows, 2, half_width)), (0, 2, 1)))


@triton.jit
def _sum_and_max(sum_left, max_left, sum_right, max_right):
    return sum_left + sum_right, tl.maximum(max_left, max_right)


@triton.jit
def _sorted_mla_request(request_ids, request_lengths, batch, B: tl.constexpr):
    slots: tl.constexpr = request_ids.shape[0]
    max_priority: tl.constexpr = 0x7FFFFFFF // slots
    # Bound only scheduling priority; every request keeps its actual KV length.
    priority = tl.minimum(tl.maximum(request_lengths, 0), max_priority)
    keys = priority * slots + (B - 1 - request_ids)
    keys = tl.where(request_ids < B, keys, -1)
    sorted_keys = tl.sort(keys, descending=True)
    reverse_ids = sorted_keys & (slots - 1)
    selected = tl.sum(tl.where(request_ids == batch, reverse_ids, 0), 0)
    return B - 1 - selected


def dense_mla_configs() -> list[triton.Config]:
    configs = runtime.get_tuned_config("dense_int8_mla_kernel")
    defaults = {
        "ppu_fold_global_load_offsets": False,
        "ppu_vectorize_i8_shared_store": False,
        "ppu_hoist_readonly_shared_matrices": 0,
        "ppu_publish_shared_float_scales_early": False,
        "ppu_reuse_mla_kv": False,
        "ppu_async_mla_kv": False,
        "ppu_prefetch_mla_nope": False,
        "ppu_scalarize_mla_pv": False,
    }
    required = set(defaults)
    available = {field.name for field in fields(HGGCOptions)}
    if not required.issubset(available):
        return [config for config in configs if required.isdisjoint(config.kwargs)]
    if "ppu_scalarize_mla_p15" in available:
        defaults["ppu_scalarize_mla_p15"] = False
        defaults["PROBABILITY_RADIX"] = 127
    if "ppu_pipeline_mla_kv" in available:
        defaults["ppu_pipeline_mla_kv"] = False
    if "ppu_pipeline_mla_kv_pages" in available:
        defaults["ppu_pipeline_mla_kv_pages"] = False
    if "ppu_mla_h64_k128" in available:
        defaults["ppu_mla_h64_k128"] = False
    if "ppu_mla_h32_k128" in available:
        defaults["ppu_mla_h32_k128"] = False
    if "ppu_gather_mla_row_reductions" in available:
        defaults["ppu_gather_mla_row_reductions"] = False
    if "ppu_unique_mla_scale_writers" in available:
        defaults["ppu_unique_mla_scale_writers"] = False
    if "ppu_unique_mla_final_rows" in available:
        defaults["ppu_unique_mla_final_rows"] = False
    if "ppu_reuse_mla_split_kv" in available:
        defaults["ppu_reuse_mla_split_kv"] = False
    if "ppu_mla_kr_joint_copy" in available:
        defaults["ppu_mla_kr_joint_copy"] = False
    if "ppu_mla_logit_scale_fold" in available:
        defaults["ppu_mla_logit_scale_fold"] = False
    if "ppu_mla_low_probability_layout" in available:
        defaults["ppu_mla_low_probability_layout"] = False
    if "ppu_mla_h16_kv_pipeline" in available:
        defaults["ppu_mla_h16_kv_pipeline"] = False
    if "ppu_mla_h16_kr_copy" in available:
        defaults["ppu_mla_h16_kr_copy"] = False
    if "ppu_mla_row_subgroups" in available:
        defaults["ppu_mla_row_subgroups"] = False
    if "ppu_mla_full_tiles" in available:
        defaults["ppu_mla_full_tiles"] = False
    if "ppu_mla_uniform_request_sort" in available:
        defaults["SORT_SKIP_EQUAL"] = False
        defaults["ppu_mla_uniform_request_sort"] = False
    normalized = []
    for config in configs:
        if "ppu_mla_h32_k128" in config.kwargs and "ppu_mla_h32_k128" not in available:
            continue
        if "ppu_mla_h64_k128" in config.kwargs and "ppu_mla_h64_k128" not in available:
            continue
        if (
            "ppu_scalarize_mla_p15" in config.kwargs
            and "ppu_scalarize_mla_p15" not in available
        ):
            continue
        if (
            "ppu_pipeline_mla_kv" in config.kwargs
            and "ppu_pipeline_mla_kv" not in available
        ):
            continue
        if (
            "ppu_pipeline_mla_kv_pages" in config.kwargs
            and "ppu_pipeline_mla_kv_pages" not in available
        ):
            continue
        candidate = copy.copy(config)
        candidate.kwargs = {**defaults, **config.kwargs}
        normalized.append(candidate)
    if "ppu_gather_mla_row_reductions" in available:
        enabled = []
        for config in normalized:
            if config.kwargs["ppu_mla_h32_k128"]:
                candidate = copy.copy(config)
                candidate.kwargs = {
                    **config.kwargs,
                    "ppu_gather_mla_row_reductions": True,
                }
                enabled.append(candidate)
        normalized.extend(enabled)
    if "ppu_unique_mla_scale_writers" in available:
        enabled = []
        for config in normalized:
            if config.kwargs["ppu_mla_h32_k128"]:
                candidate = copy.copy(config)
                candidate.kwargs = {
                    **config.kwargs,
                    "ppu_unique_mla_scale_writers": True,
                }
                enabled.append(candidate)
        normalized.extend(enabled)
    if "ppu_unique_mla_final_rows" in available:
        enabled = []
        for config in normalized:
            if config.kwargs["ppu_mla_h32_k128"]:
                candidate = copy.copy(config)
                candidate.kwargs = {
                    **config.kwargs,
                    "ppu_unique_mla_final_rows": True,
                }
                enabled.append(candidate)
        normalized.extend(enabled)
    if "ppu_reuse_mla_split_kv" in available:
        for config in normalized:
            kwargs = config.kwargs
            if (
                config.num_warps == 8
                and config.num_stages == 1
                and kwargs["BH"] == 16
                and kwargs["BK"] == 128
                and kwargs["QK_SPLITS"] == 4
                and kwargs["FUSED_PV"]
                and not kwargs["PAIR_PV"]
                and not kwargs["COLUMN_SOFTMAX"]
                and kwargs["SORT_REQUESTS"]
                and kwargs["PROBABILITY_RADIX"] == 127
            ):
                candidate = copy.copy(config)
                candidate.kwargs = {
                    **kwargs,
                    "ppu_reuse_mla_split_kv": True,
                    "ppu_async_mla_kv": True,
                }
                normalized.append(candidate)
                break
    if "ppu_mla_kr_joint_copy" in available:
        enabled = []
        for config in normalized:
            kwargs = config.kwargs
            if (
                kwargs["ppu_mla_h32_k128"]
                and kwargs["SORT_REQUESTS"]
                and kwargs["ppu_gather_mla_row_reductions"]
                and kwargs["ppu_unique_mla_scale_writers"]
            ):
                candidate = copy.copy(config)
                candidate.kwargs = {**kwargs, "ppu_mla_kr_joint_copy": True}
                enabled.append(candidate)
        normalized.extend(enabled)
    if "ppu_mla_logit_scale_fold" in available:
        for config in normalized:
            kwargs = config.kwargs
            if (
                kwargs["ppu_mla_kr_joint_copy"]
                and not kwargs["ppu_unique_mla_final_rows"]
            ):
                candidate = copy.copy(config)
                candidate.kwargs = {**kwargs, "ppu_mla_logit_scale_fold": True}
                normalized.append(candidate)
                if "ppu_mla_low_probability_layout" in available:
                    repacked = copy.copy(candidate)
                    repacked.kwargs = {
                        **candidate.kwargs,
                        "ppu_mla_low_probability_layout": True,
                    }
                    normalized.append(repacked)
                break
    if "ppu_mla_h16_kv_pipeline" in available:
        enabled = []
        for config in normalized:
            if config.kwargs.get("ppu_reuse_mla_split_kv"):
                for reuse_pages in (False, True):
                    candidate = copy.copy(config)
                    candidate.kwargs = {
                        **config.kwargs,
                        "ppu_pipeline_mla_kv": True,
                        "ppu_pipeline_mla_kv_pages": reuse_pages,
                        "ppu_mla_h16_kv_pipeline": True,
                    }
                    enabled.append(candidate)
        normalized.extend(enabled)
    if "ppu_mla_h16_kr_copy" in available:
        enabled = []
        for config in normalized:
            if config.kwargs.get("ppu_mla_h16_kv_pipeline"):
                candidate = copy.copy(config)
                candidate.kwargs = {**config.kwargs, "ppu_mla_h16_kr_copy": True}
                enabled.append(candidate)
        normalized.extend(enabled)
    if "ppu_mla_uniform_request_sort" in available:
        enabled = []
        for config in normalized:
            kwargs = config.kwargs
            if kwargs["SORT_REQUESTS"] and (
                kwargs.get("ppu_reuse_mla_split_kv")
                or kwargs.get("ppu_mla_logit_scale_fold")
            ):
                candidate = copy.copy(config)
                candidate.kwargs = {
                    **kwargs,
                    "SORT_SKIP_EQUAL": True,
                    "ppu_mla_uniform_request_sort": True,
                }
                enabled.append(candidate)
        normalized.extend(enabled)
    if "ppu_mla_row_subgroups" in available:
        enabled = []
        for config in normalized:
            if config.kwargs.get("ppu_gather_mla_row_reductions"):
                candidate = copy.copy(config)
                candidate.kwargs = {**config.kwargs, "ppu_mla_row_subgroups": True}
                enabled.append(candidate)
        normalized.extend(enabled)
    if "ppu_mla_full_tiles" in available:
        enabled = []
        for config in normalized:
            if config.kwargs.get("ppu_mla_row_subgroups") and config.kwargs.get(
                "ppu_mla_kr_joint_copy"
            ):
                candidate = copy.copy(config)
                candidate.kwargs = {**config.kwargs, "ppu_mla_full_tiles": True}
                enabled.append(candidate)
        normalized.extend(enabled)
    return normalized


def _prune_mla_configs(configs, named_args, **kwargs):
    heads = named_args["H"]
    heads_per_query = named_args["HEADS_PER_QUERY"]
    single_query = heads == heads_per_query
    return [
        config
        for config in configs
        if (
            not config.kwargs.get("ppu_mla_low_probability_layout")
            or (single_query and heads == 32)
        )
        and (
            not config.kwargs.get("SORT_SKIP_EQUAL")
            or (
                heads in (heads_per_query, 2 * heads_per_query)
                and heads_per_query in (16, 32, 64, 128)
                and (
                    config.kwargs["BH"] == min(32, heads)
                    or config.kwargs.get("ppu_mla_h16_kv_pipeline")
                )
            )
        )
    ]


@libentry()
@libtuner(
    configs=dense_mla_configs(),
    prune_configs_by={"early_config_prune": _prune_mla_configs},
    key=["B", "H", "HEADS_PER_QUERY", "SPLITS", "TABLE_STRIDE", "CAUSAL"],
    warmup=30,
    rep=80,
    use_cuda_graph=False,
)
@triton.jit
def dense_int8_mla_kernel(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Table,
    Lengths,
    Partial,
    Stats,
    Offsets,
    Output,
    LSE,
    Q_STRIDES: tl.constexpr,
    QR_STRIDES: tl.constexpr,
    KV_STRIDES: tl.constexpr,
    KR_STRIDES: tl.constexpr,
    QS_STRIDES: tl.constexpr,
    KS_STRIDES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HEADS_PER_QUERY: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    FUSED_PV: tl.constexpr,
    COLUMN_SOFTMAX: tl.constexpr,
    QK_SPLITS: tl.constexpr,
    PAIR_PV: tl.constexpr,
    SORT_REQUESTS: tl.constexpr = False,
    PROBABILITY_RADIX: tl.constexpr = 127,
    SORT_SKIP_EQUAL: tl.constexpr = False,
):
    tl.static_assert(
        PROBABILITY_RADIX == 127
        or (
            PROBABILITY_RADIX == 256
            and ((BH == 32 and (BK == 128 or BK == 256)) or (BH == 64 and BK == 128))
        )
    )
    h32_k128: tl.constexpr = BH == 32 and BK == 128 and PROBABILITY_RADIX == 256
    head_tile, batch, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    sort_requests: tl.constexpr = SORT_REQUESTS and B * triton.cdiv(H, BH) > 64
    reuse_length: tl.constexpr = sort_requests and SORT_SKIP_EQUAL and BH == 16
    if sort_requests:
        # Process longer requests first while preserving every original output row.
        request_ids = tl.arange(0, triton.next_power_of_2(B))
        request_lengths = tl.load(
            Lengths + request_ids * LENGTH_STRIDE, request_ids < B, 0
        )
        if SORT_SKIP_EQUAL:
            first_length = tl.load(Lengths)
            different = tl.sum(
                ((request_ids < B) & (request_lengths != first_length)).to(tl.int32), 0
            )
            if different != 0:
                batch = _sorted_mla_request(request_ids, request_lengths, batch, B)
                if reuse_length:
                    length = tl.load(Lengths + batch * LENGTH_STRIDE)
            else:
                if reuse_length:
                    length = first_length
        else:
            batch = _sorted_mla_request(request_ids, request_lengths, batch, B)
    heads = head_tile * BH + tl.arange(0, BH)
    query_index = heads // HEADS_PER_QUERY
    query_head = heads % HEADS_PER_QUERY
    dims = tl.arange(0, 512)
    rope_dims = tl.arange(0, 64)
    tokens = tl.arange(0, BK)
    reduction_axis: tl.constexpr = 0 if COLUMN_SOFTMAX else 1
    late_rope: tl.constexpr = (
        (BH == 32 and BK == 256) or (BH == 64 and BK == 128) or h32_k128
    )
    q = tl.load(
        Q
        + batch * Q_STRIDES[0]
        + query_index[:, None] * Q_STRIDES[1]
        + query_head[:, None] * Q_STRIDES[2]
        + dims[None, :],
        heads[:, None] < H,
        0,
    )
    if QK_SPLITS == 4:
        q_left, q_right = _split_mla_features(q, BH, 256)
        q0, q1 = _split_mla_features(q_left, BH, 128)
        q2, q3 = _split_mla_features(q_right, BH, 128)
    qr = tl.load(
        QR
        + batch * QR_STRIDES[0]
        + query_index[:, None] * QR_STRIDES[1]
        + query_head[:, None] * QR_STRIDES[2]
        + rope_dims[None, :],
        heads[:, None] < H,
        0,
    )
    qs = tl.load(
        QS
        + batch * QS_STRIDES[0]
        + query_index * QS_STRIDES[1]
        + query_head * QS_STRIDES[2],
        heads < H,
        0,
    )
    if not reuse_length:
        length = tl.load(Lengths + batch * LENGTH_STRIDE)
    query_limit = (
        tl.maximum(length - H // HEADS_PER_QUERY + query_index + 1, 0)
        if CAUSAL
        else tl.full((BH,), 0, tl.int32) + length
    )
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    denominator = tl.zeros((BH,), tl.float32)
    accumulator = tl.zeros((BH, 512), tl.float32)
    valid_pages = tl.cdiv(length, 64)
    first_page = valid_pages * split // SPLITS
    end_page = valid_pages * (split + 1) // SPLITS
    for page_idx in range(first_page, end_page, BK // 64):
        page_offsets = page_idx + tokens // 64
        if BK == 64:
            page = tl.load(Table + batch * TABLE_STRIDE + page_idx).to(tl.int64)[None]
        else:
            page = tl.load(
                Table + batch * TABLE_STRIDE + page_offsets,
                page_offsets < end_page,
                0,
            ).to(tl.int64)
        valid_tokens = (page_idx * 64 + tokens < length) & (page_offsets < end_page)
        kv = tl.load(
            KV
            + page[:, None] * KV_STRIDES[0]
            + (tokens % 64)[:, None] * KV_STRIDES[1]
            + dims[None, :],
            valid_tokens[:, None],
            0,
            cache_modifier=".cg",
        )
        if not late_rope:
            kr = tl.load(
                KR
                + page[:, None] * KR_STRIDES[0]
                + (tokens % 64)[:, None] * KR_STRIDES[1]
                + rope_dims[None, :],
                valid_tokens[:, None],
                0,
            )
            ks = tl.load(
                KS + page * KS_STRIDES[0] + (tokens % 64) * KS_STRIDES[1],
                valid_tokens,
                0,
            )
        if QK_SPLITS == 4:
            kv_left, kv_right = _split_mla_features(kv, BK, 256)
            kv0, kv1 = _split_mla_features(kv_left, BK, 128)
            kv2, kv3 = _split_mla_features(kv_right, BK, 128)
            # INT32 partial sums are exact: 512 * 128 * 128 fits in INT32.
            content_scores = (
                tl.dot(q0, tl.trans(kv0), out_dtype=tl.int32)
                + tl.dot(q1, tl.trans(kv1), out_dtype=tl.int32)
                + tl.dot(q2, tl.trans(kv2), out_dtype=tl.int32)
                + tl.dot(q3, tl.trans(kv3), out_dtype=tl.int32)
            ).to(tl.float32)
        else:
            content_scores = tl.dot(q, tl.trans(kv), out_dtype=tl.int32).to(tl.float32)
        if late_rope:
            # Delay KR and KS to avoid spills across the content dot.
            kr = tl.load(
                KR
                + page[:, None] * KR_STRIDES[0]
                + (tokens % 64)[:, None] * KR_STRIDES[1]
                + rope_dims[None, :],
                valid_tokens[:, None],
                0,
            )
            ks = tl.load(
                KS + page * KS_STRIDES[0] + (tokens % 64) * KS_STRIDES[1],
                valid_tokens,
                0,
            )
        rope_scores = tl.dot(qr, tl.trans(kr), out_dtype=tl.float32)
        # Separating the dot chains avoids replicated QK warps on older compilers.
        qk = content_scores + rope_scores
        qk = tl.trans(qk) if COLUMN_SOFTMAX else qk
        scores = (
            qk
            * tl.expand_dims(qs, reduction_axis)
            * tl.expand_dims(ks, 1 - reduction_axis)
            * (SCALE * 1.4426950408889634)
        )
        scores = tl.where(
            (
                page_idx * 64 + tl.expand_dims(tokens, 1 - reduction_axis)
                < tl.expand_dims(query_limit, reduction_axis)
            )
            & tl.expand_dims(page_offsets < end_page, 1 - reduction_axis),
            scores,
            -float("inf"),
        )
        new_maximum = tl.maximum(maximum, tl.max(scores, reduction_axis))
        safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
        probabilities = tl.exp2(scores - tl.expand_dims(safe_maximum, reduction_axis))
        rescale = tl.exp2(maximum - safe_maximum)
        if BH == 16 and BK == 128:
            # Retain the original reduction schedule for this tile geometry.
            denominator = denominator * rescale + tl.sum(probabilities, reduction_axis)
            scaled_probabilities = probabilities * tl.expand_dims(
                ks, 1 - reduction_axis
            )
            probability_scale = tl.maximum(
                tl.max(scaled_probabilities, reduction_axis) / 127.0, 1.0e-30
            )
        else:
            # Fold KV scales into P and share synchronization for both statistics.
            scaled_probabilities = probabilities * tl.expand_dims(
                ks, 1 - reduction_axis
            )
            probability_sum, scaled_probability_max = tl.reduce(
                (probabilities, scaled_probabilities), reduction_axis, _sum_and_max
            )
            denominator = denominator * rescale + probability_sum
            probability_scale = tl.maximum(scaled_probability_max / 127.0, 1.0e-30)
        if (BH == 32 and BK == 256) or (BH == 64 and BK == 128) or h32_k128:
            # The original MLA scale bounds make both full-div scaling branches false.
            normalized_probabilities = scaled_probabilities * tl.expand_dims(
                native_mla_probability_reciprocal(probability_scale), reduction_axis
            )
            quantized_probabilities, quantized_residual = (
                quantize_mla_probabilities_packed(
                    normalized_probabilities, NORMALIZED=True, RADIX=PROBABILITY_RADIX
                )
            )
        else:
            if BH == 16 and BK == 128:
                normalized_probabilities = scaled_probabilities * tl.expand_dims(
                    native_mla_probability_reciprocal(probability_scale),
                    reduction_axis,
                )
            else:
                normalized_probabilities = scaled_probabilities / tl.expand_dims(
                    probability_scale, reduction_axis
                )
            quantized_probabilities, quantized_residual = quantize_mla_probabilities(
                normalized_probabilities
            )
        quantized_probabilities = (
            tl.trans(quantized_probabilities)
            if COLUMN_SOFTMAX
            else quantized_probabilities
        )
        quantized_residual = (
            tl.trans(quantized_residual) if COLUMN_SOFTMAX else quantized_residual
        )
        if PAIR_PV and FUSED_PV:
            paired_probabilities = tl.reshape(
                tl.permute(
                    tl.join(quantized_probabilities, quantized_residual), (2, 0, 1)
                ),
                (2 * BH, BK),
            )
            # Keep JoinOp from requesting 64-bit packed dot operands.
            paired_probabilities = tl.inline_asm_elementwise(
                "ppu.mov.b32 $0, $1;",
                "=r,r",
                [paired_probabilities.to(tl.int32)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            ).to(tl.int8)
            paired_products = tl.dot(paired_probabilities, kv, out_dtype=tl.int32)
            high, low = tl.split(
                tl.permute(tl.reshape(paired_products, (2, BH, 512)), (1, 2, 0))
            )
            contribution = (high * PROBABILITY_RADIX + low).to(tl.float32)
            contribution_scale = probability_scale * (1.0 / PROBABILITY_RADIX)
        else:
            if FUSED_PV:
                # Both supported radices keep the 256-token sum within INT32.
                high = tl.dot(quantized_probabilities, kv, out_dtype=tl.int32)
                contribution = tl.dot(
                    quantized_residual, kv, high * PROBABILITY_RADIX, out_dtype=tl.int32
                ).to(tl.float32)
                contribution_scale = probability_scale * (1.0 / PROBABILITY_RADIX)
            else:
                contribution = tl.dot(
                    quantized_probabilities, kv, out_dtype=tl.int32
                ).to(tl.float32)
                correction = tl.dot(quantized_residual, kv, out_dtype=tl.int32).to(
                    tl.float32
                )
                contribution += correction * (1.0 / PROBABILITY_RADIX)
                contribution_scale = probability_scale
        accumulator = (
            accumulator * rescale[:, None] + contribution * contribution_scale[:, None]
        )
        maximum = new_maximum
    if SPLITS == 1:
        normalized = accumulator / tl.where(denominator > 0, denominator, 1.0)[:, None]
        tl.store(
            Output + (batch * H + heads[:, None]) * 512 + dims[None, :],
            normalized,
            heads[:, None] < H,
        )
        logsum = tl.where(
            denominator > 0,
            maximum * 0.6931471805599453 + tl.log(denominator),
            float("inf"),
        )
        tl.store(LSE + heads * B + batch, logsum, heads < H)
    else:
        tl.store(
            Partial + ((split * B + batch) * H + heads[:, None]) * 512 + dims[None, :],
            accumulator,
            heads[:, None] < H,
        )
        tl.store(Stats + (split * 2 * H + heads) * B + batch, maximum, heads < H)
        tl.store(
            Stats + ((split * 2 + 1) * H + heads) * B + batch, denominator, heads < H
        )
        if head_tile == 0 and split == 0:
            tl.store(Offsets + batch, batch)
            if batch == 0:
                tl.store(Offsets + B, B)


class FlashMLAInt8PreparedHandle:
    """Reuse workspace for one/two-token decode; calls must be serialized on one stream."""

    def __init__(
        self,
        tensors: tuple[torch.Tensor, ...],
        splits: int,
        scale: float,
        causal: bool = False,
    ):
        self.tensors = tensors
        query, rope = tensors[:2]
        self.batch, self.queries, self.heads, _ = query.shape
        self.total_heads = self.queries * self.heads
        self.causal = causal
        self.splits = splits
        self.scale = scale
        self.output = torch.empty(
            (self.batch, self.queries, self.heads, 512),
            dtype=rope.dtype,
            device=query.device,
        )
        self.lse = torch.empty_strided(
            (self.batch, self.heads, self.queries),
            (1, self.batch, self.batch * self.heads),
            dtype=torch.float32,
            device=query.device,
        )
        workspace_splits = splits if splits > 1 else 0
        self.partial = torch.empty(
            (workspace_splits, self.batch, self.total_heads, 512),
            dtype=torch.float32,
            device=query.device,
        )
        self.stats = torch.empty(
            (workspace_splits, 2, self.total_heads, self.batch),
            dtype=torch.float32,
            device=query.device,
        )
        self.offsets = torch.empty(
            (self.batch + 1 if splits > 1 else 0,),
            dtype=torch.int32,
            device=query.device,
        )

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch == 0 or self.heads == 0:
            return self.output, self.lse
        query, rope, kv, kv_rope, qs, ks, table, lengths = self.tensors
        dense_int8_mla_kernel[
            lambda meta: (
                triton.cdiv(self.total_heads, meta["BH"]),
                self.batch,
                self.splits,
            )
        ](
            *self.tensors,
            self.partial,
            self.stats,
            self.offsets,
            self.output,
            self.lse,
            query.stride(),
            rope.stride(),
            kv.stride(),
            kv_rope.stride(),
            qs.stride(),
            ks.stride(),
            table.stride(0),
            lengths.stride(0),
            self.batch,
            self.total_heads,
            self.heads,
            self.splits,
            self.scale,
            self.causal,
        )
        if self.splits > 1:
            _flash_int8_merge_splits[(self.batch, self.total_heads)](
                self.partial,
                self.stats,
                self.output,
                self.lse,
                self.offsets,
                self.batch,
                self.total_heads,
                512,
                self.batch,
                self.splits,
                self.total_heads * 512,
                512,
                True,
                num_warps=4,
            )
        return self.output, self.lse


def prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_lora: torch.Tensor,
    k_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int = 512,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    num_splits: Optional[int] = None,
) -> tuple[FlashMLAInt8PreparedHandle, tuple[torch.Tensor, torch.Tensor]]:
    """RoPE inputs are divided by the content scale, matching PR851.

    Active page IDs and lengths must be valid; active scales finite and positive.
    Unused cache slots may contain NaNs.
    Queries are the final one or two cache tokens; causal masks preserve their order.
    """
    if q_nope.ndim != 4 or q_nope.shape[1] not in (1, 2) or q_nope.shape[-1] != 512:
        raise NotImplementedError("requires Q [batch, 1 or 2, heads, 512]")
    batch, queries, heads, _ = q_nope.shape
    if k_lora.ndim != 3 or k_lora.shape[1:] != (64, 512) or head_dim_v != 512:
        raise NotImplementedError("requires KV [pages, 64, 512] and head_dim_v=512")
    expected_shapes = (
        (batch, queries, heads, 64),
        (k_lora.shape[0], 64, 64),
        (batch, queries, heads, 1),
        (k_lora.shape[0], 64, 1),
        (batch,),
    )
    for tensor, shape in zip(
        (q_rope, k_rope, q_scale, k_scale, cache_seqlens), expected_shapes
    ):
        if tensor.shape != shape:
            raise ValueError(f"expected shape {shape}, got {tuple(tensor.shape)}")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.shape[1] == 0
    ):
        raise ValueError("block_table must have shape [batch, positive page capacity]")
    if q_nope.dtype != torch.int8 or k_lora.dtype != torch.int8:
        raise TypeError("Q and KV content must be INT8")
    if (
        q_rope.dtype not in (torch.float16, torch.bfloat16)
        or k_rope.dtype != q_rope.dtype
    ):
        raise TypeError("RoPE tensors must have the same FP16 or BF16 dtype")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("scales must be FP32")
    if block_table.dtype != torch.int32 or cache_seqlens.dtype != torch.int32:
        raise TypeError("block_table and cache_seqlens must be INT32")
    tensors = (
        q_nope,
        q_rope,
        k_lora,
        k_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
    )
    for tensor in tensors:
        if tensor.device != q_nope.device or (
            tensor.numel() != 0 and tensor.stride(-1) != 1
        ):
            raise ValueError(
                "all tensors must share a device and contiguous last dimension"
            )
    scale = 576**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    splits = num_splits
    if splits is None:
        splits = min(
            16,
            triton.next_power_of_2(
                max(1, 64 // max(1, batch * triton.cdiv(queries * heads, 16)))
            ),
        )
        # Low-head decode benefits from shorter KV loops even at large batch size.
        minimum_splits = (
            2 if queries * heads <= 16 and block_table.shape[1] >= 64 else 1
        )
        splits = max(minimum_splits, splits)
        splits = min(splits, max(1, triton.next_power_of_2(block_table.shape[1])))
    if splits not in (1, 2, 4, 8, 16):
        raise ValueError("num_splits must be one of 1, 2, 4, 8, 16")
    handle = FlashMLAInt8PreparedHandle(tensors, splits, scale, causal)
    return handle, handle()


def flash_mla_with_kvcache_fwd_w8a8_int8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_lora: torch.Tensor,
    k_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int = 512,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    num_splits: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, outputs = prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
        q_nope,
        q_rope,
        k_lora,
        k_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
        softmax_scale=softmax_scale,
        causal=causal,
        num_splits=num_splits,
    )
    return outputs
