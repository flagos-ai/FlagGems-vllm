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

"""Hopper warp-per-token routing with FlagTree TLE explicit layouts."""

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.gemma4_fused_routing_kernel_triton import (
    _finish_routes,
    _kernel_args,
    _prepare_routes,
    _prune_token_tiles,
    _select_routes,
)
from flaggems_vllm.ops.gemma4_fused_routing_kernel_triton import (
    gemma4_fused_routing_kernel_triton as _generic_routing,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.triton_version_utils import has_triton_tle_attrs

_HAS_LAYOUT = has_triton_tle_attrs(
    ("gpu.set_layout", "gpu.BlockEncoding", "gpu.SlicedEncoding"), 3, 6, 0
)
if _HAS_LAYOUT:
    import triton.experimental.tle.language as tle
    from triton._C.libtriton import ir

    _HAS_LAYOUT = hasattr(ir.builder, "ensure_ttg_layout_attrs")
else:
    tle = None


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gemma4_routing_select"),
    key=["T", "E", "K", "HAS_PADDING", "CONTIGUOUS"],
    prune_configs_by={"early_config_prune": _prune_token_tiles},
    use_cuda_graph=True,
    rep=20,
)
@triton.jit
def _gemma4_hopper_kernel(
    G,
    S,
    W,
    I,
    P,
    T: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    S0: tl.constexpr,
    P0: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    BLOCK_T: tl.constexpr,
    num_warps: tl.constexpr,
):
    # Each warp owns whole tokens: integer redux never needs a CTA-wide barrier.
    encoding: tl.constexpr = tle.gpu.BlockEncoding(
        [1, 1], [1, 32], [num_warps, 1], [1, 0]
    )
    rows = tle.gpu.set_layout(
        tl.arange(0, BLOCK_T), tle.gpu.SlicedEncoding(1, encoding)
    )
    cols = tle.gpu.set_layout(tl.arange(0, B), tle.gpu.SlicedEncoding(0, encoding))
    slots = tle.gpu.set_layout(
        tl.arange(0, triton.next_power_of_2(K)), tle.gpu.SlicedEncoding(0, encoding)
    )
    rows = tl.program_id(0) * BLOCK_T + rows
    x = tl.load(
        G + rows[:, None] * G0 + cols[None, :] * G1,
        (rows[:, None] < T) & (cols[None, :] < E),
        other=-float("inf"),
    ).to(tl.float32)
    exps = tle.gpu.set_layout(
        tl.full((BLOCK_T, triton.next_power_of_2(K)), 0.0, tl.float32), encoding
    )
    ids = tle.gpu.set_layout(
        tl.full((BLOCK_T, triton.next_power_of_2(K)), 0, tl.int32), encoding
    )
    exps, ids = _select_routes(x, cols, slots, exps, ids, E, K)
    _finish_routes(S, W, I, P, rows, slots, exps, ids, T, K, S0, P0, HAS_PADDING)


def gemma4_fused_routing_kernel_triton(
    gating_output, topk, per_expert_scale, num_warps=1, *, is_padding=None
):
    if not _HAS_LAYOUT or topk > 8:
        return _generic_routing(
            gating_output, topk, per_expert_scale, num_warps, is_padding=is_padding
        )
    t, e, w, i = _prepare_routes(
        gating_output, topk, per_expert_scale, num_warps, is_padding
    )
    if not t:
        return w, i
    args = _kernel_args(gating_output, topk, per_expert_scale, w, i, is_padding)
    grid = lambda meta: (triton.cdiv(t, meta["BLOCK_T"]),)
    with torch_device_fn.device(gating_output.device):
        _gemma4_hopper_kernel[grid](*args)
    return w, i
