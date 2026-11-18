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

"""NVIDIA 3.6 launch cache; retains compiler handles and metadata only."""

import triton

from flaggems_vllm.ops.ihc.ihc_pre import (
    _ihc_pre_stage1,
    _ihc_pre_stage1_decode,
    _ihc_pre_stage2,
    _ihc_pre_stage2_decode,
)

_DECODE_LAUNCHES = {}
_PREFILL_LAUNCHES = {}


def _launch_decode(
    x,
    weight,
    partial,
    scale,
    base,
    output,
    post,
    n,
    hc,
    d,
    hc_pad,
    block_k,
    splits,
    partial_stride,
    block_d,
    magnitude,
    norm_eps,
    hc_eps,
    layout_tag,
):
    k = hc * d
    first_constants = (k, 2 * hc, 2 * hc_pad, splits, block_k, partial_stride)
    second_constants = (
        d,
        k,
        hc,
        hc_pad,
        2 * hc,
        splits,
        partial_stride,
        triton.next_power_of_2(splits),
        block_d,
        magnitude,
        norm_eps,
        hc_eps,
    )
    first_grid = (n, splits, 1)
    second_grid = (n, triton.cdiv(d, block_d), 1)
    key = (
        x.device.index,
        x.dtype,
        n,
        hc,
        d,
        layout_tag,
        first_constants,
        second_constants[:-3],
        magnitude.hex(),
        norm_eps.hex(),
        hc_eps.hex(),
        first_grid,
        second_grid,
        1,
        4,
        False,
        x.data_ptr() % 16 == 0,
        weight.data_ptr() % 16 == 0,
        partial.data_ptr() % 16 == 0,
        scale.data_ptr() % 16 == 0,
        base.data_ptr() % 16 == 0,
        output.data_ptr() % 16 == 0,
        post.data_ptr() % 16 == 0,
    )
    cacheable = magnitude == magnitude and norm_eps == norm_eps and hc_eps == hc_eps
    cached = _DECODE_LAUNCHES.get(key) if cacheable else None
    if cached is not None:
        first, second, first_constants, second_constants = cached
        first(x, weight, partial, *first_constants)
        second(x, partial, scale, base, output, post, *second_constants)
        return
    first = _ihc_pre_stage1_decode[first_grid](
        x, weight, partial, *first_constants, num_warps=1, enable_fp_fusion=False
    )
    second = _ihc_pre_stage2_decode[second_grid](
        x,
        partial,
        scale,
        base,
        output,
        post,
        *second_constants,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if cacheable:
        _DECODE_LAUNCHES[key] = (
            first[first_grid],
            second[second_grid],
            first_constants,
            second_constants,
        )


def _launch_prefill(
    x,
    weight,
    sumsq,
    mix,
    scale,
    base,
    output,
    post,
    n,
    hc,
    d,
    hc_pad,
    grid_m,
    splits,
    block_m,
    block_k,
    block_d,
    num_warps,
    pipeline_stages,
    magnitude,
    norm_eps,
    hc_eps,
    layout_tag,
):
    k = hc * d
    first_constants = (
        k,
        k // splits,
        2 * hc,
        block_m,
        block_k,
        splits,
        pipeline_stages,
    )
    second_constants = (
        d,
        k,
        hc,
        hc_pad,
        2 * hc,
        splits,
        triton.next_power_of_2(splits),
        block_d,
        magnitude,
        norm_eps,
        hc_eps,
    )
    first_grid = (grid_m, splits, 1)
    second_grid = (n, triton.cdiv(d, block_d), 1)
    key = (
        x.device.index,
        x.dtype,
        n,
        hc,
        d,
        layout_tag,
        first_constants,
        second_constants[:-3],
        magnitude.hex(),
        norm_eps.hex(),
        hc_eps.hex(),
        first_grid,
        second_grid,
        num_warps,
        4,
        False,
        x.data_ptr() % 16 == 0,
        weight.data_ptr() % 16 == 0,
        sumsq.data_ptr() % 16 == 0,
        mix.data_ptr() % 16 == 0,
        scale.data_ptr() % 16 == 0,
        base.data_ptr() % 16 == 0,
        output.data_ptr() % 16 == 0,
        post.data_ptr() % 16 == 0,
    )
    cacheable = magnitude == magnitude and norm_eps == norm_eps and hc_eps == hc_eps
    cached = _PREFILL_LAUNCHES.get(key) if cacheable else None
    if cached is not None:
        first, second, first_constants, second_constants = cached
        first(x, weight, sumsq, mix, n, *first_constants)
        second(x, mix, sumsq, scale, base, output, post, n, *second_constants)
        return
    first = _ihc_pre_stage1[first_grid](
        x,
        weight,
        sumsq,
        mix,
        n,
        *first_constants,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    second = _ihc_pre_stage2[second_grid](
        x,
        mix,
        sumsq,
        scale,
        base,
        output,
        post,
        n,
        *second_constants,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if cacheable:
        _PREFILL_LAUNCHES[key] = (
            first[first_grid],
            second[second_grid],
            first_constants,
            second_constants,
        )
