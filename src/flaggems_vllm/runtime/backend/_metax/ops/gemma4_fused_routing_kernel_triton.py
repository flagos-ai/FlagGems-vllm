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
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.gemma4_fused_routing_kernel_triton import (
    _HAS_PARTIAL_TOPK,
    _finish_routes,
    _kernel_args,
    _key_value,
    _order_key,
    _prepare_routes,
)
from flaggems_vllm.ops.gemma4_fused_routing_kernel_triton import (
    gemma4_fused_routing_kernel_triton as _generic_routing,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner


def _prune_token_tiles(configs, named_args, **kwargs):
    # Keep one program per token while the grid is small.
    limit = 1 if named_args["T"] <= 128 else named_args["T"]
    return [config for config in configs if config.kwargs["BLOCK_T"] <= limit]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gemma4_routing_metax"),
    key=["T", "E", "K", "HAS_PADDING", "CONTIGUOUS"],
    prune_configs_by={"early_config_prune": _prune_token_tiles},
    use_cuda_graph=True,
    rep=20,
)
@triton.jit
def _gemma4_metax_kernel(
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
):
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, B)
    logits = tl.load(
        G + rows[:, None] * G0 + cols[None, :] * G1,
        (rows[:, None] < T) & (cols[None, :] < E),
        other=-float("inf"),
    ).to(tl.float32)
    keys = tl.where(cols[None, :] < E, _order_key(logits), -2147483648)
    # The low word breaks ties in favor of smaller expert IDs.
    packed = (keys.to(tl.int64) << 32) | (0xFFFFFFFF - cols[None, :].to(tl.int64))
    width: tl.constexpr = triton.next_power_of_2(K)
    selected = tl.topk(packed, width)
    ids = (0xFFFFFFFF - (selected & 0xFFFFFFFF)).to(tl.int32)
    scores = _key_value((selected >> 32).to(tl.int32))
    slots = tl.arange(0, width)
    valid = slots[None, :] < K
    peak = tl.max(tl.where(valid, scores, -float("inf")), 1)
    exps = tl.where(valid, tl.exp(scores - peak[:, None]), 0.0)
    _finish_routes(S, W, I, P, rows, slots, exps, ids, T, K, S0, P0, HAS_PADDING)


def gemma4_fused_routing_kernel_triton(
    gating_output, topk, per_expert_scale, num_warps=1, *, is_padding=None
):
    if (
        not _HAS_PARTIAL_TOPK
        or not isinstance(topk, int)
        or not 1 <= topk <= 8
        or gating_output.ndim != 2
        or gating_output.shape[1] > 128
    ):
        return _generic_routing(
            gating_output, topk, per_expert_scale, num_warps, is_padding=is_padding
        )

    t, _, weights, ids = _prepare_routes(
        gating_output, topk, per_expert_scale, num_warps, is_padding
    )
    if t:
        args = _kernel_args(
            gating_output, topk, per_expert_scale, weights, ids, is_padding
        )
        grid = lambda meta: (triton.cdiv(t, meta["BLOCK_T"]),)
        with torch_device_fn.device(gating_output.device):
            _gemma4_metax_kernel[grid](*args)
    return weights, ids
