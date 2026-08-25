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

import pytest
import torch

import flaggems_vllm

from . import base


def torch_grouped_topk(
    scores: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
):
    """
    Adapted from vLLM: vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py.
    Ignore the non-stablility of torch.topk in benchmark test.
    """
    scores = scores.float()
    if scoring_func == 1:
        scores = scores.sigmoid()

    num_token = scores.size(0)
    original_scores = scores
    scores = scores + bias.unsqueeze(0)
    group_scores = (
        scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    )

    use_sorted = True
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]

    if bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(
            tmp_scores, k=topk, dim=-1, sorted=use_sorted
        )

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def aiter_biased_grouped_topk(
    scores: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
):
    num_tokens = scores.size(0)
    topk_weights = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device=scores.device
    )
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=scores.device)
    moe_fused_gate(
        scores.float(),
        bias.float(),
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        topk=topk,
        num_fused_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
    )
    return topk_weights, topk_ids


def mthreads_grouped_topk(
    scores: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
):
    num_fused_shared_experts = 0
    apply_routed_scaling_factor_on_output = routed_scaling_factor != 1.0
    topk_weights, topk_ids = moe_fused_gate(
        scores.float(),
        bias.float(),
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor if routed_scaling_factor is not None else 1.0,
        renormalize,
        apply_routed_scaling_factor_on_output,
    )
    return topk_weights, topk_ids


vendor_name = flaggems_vllm.vendor_name
USE_AITER = False
USE_MATE = False

try:
    if vendor_name == "hygon":
        from aiter import moe_fused_gate  # noqa: F401

        ref_grouped_topk = aiter_biased_grouped_topk
        USE_AITER = True
    elif vendor_name == "mthreads":
        from mate import moe_fused_gate  # noqa: F401

        ref_grouped_topk = mthreads_grouped_topk
        USE_MATE = True
    else:
        import vllm._custom_ops  # noqa: F401

        if hasattr(torch.ops._moe_C, "grouped_topk"):
            ref_grouped_topk = torch.ops._moe_C.grouped_topk
        else:
            ref_grouped_topk = torch_grouped_topk
except (ImportError, AttributeError):
    ref_grouped_topk = torch_grouped_topk


class GroupedTopKBenchmark(base.Benchmark):
    def __init__(
        self,
        op_name,
        torch_op,
        dtypes,
        renormalize=True,
        routed_scaling_factor=1.0,
        scoring_func=0,
    ):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func

    def set_shapes(self, shape_file_path=None):
        grouped_topk_configs = [
            # Deepseek-3.2
            (num_tokens, num_experts, n_group, topk_group, topk)
            for num_tokens in [1, 8, 32, 64, 128, 256, 496, 512, 16384]
            for num_experts in [256]
            for n_group in [8]
            for topk_group in [4]
            for topk in [8]
        ]
        self.shapes = grouped_topk_configs

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from self.grouped_topk_input_fn(config, dtype, self.device)

    def grouped_topk_input_fn(self, config, dtype, device):
        num_tokens, num_experts, n_group, topk_group, topk = config

        scores = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
        bias = torch.randn(num_experts, device=device, dtype=torch.float32)

        yield (
            scores,
            n_group,
            topk_group,
            topk,
            self.renormalize,
            self.routed_scaling_factor,
            bias,
            self.scoring_func,
        )


@pytest.mark.grouped_topk
@pytest.mark.skipif(
    USE_AITER, reason="scoring_func == 0 is not supported by moe_fused_gate in aiter"
)
@pytest.mark.skipif(
    USE_MATE, reason="scoring_func == 0 is not supported by moe_fused_gate in mate"
)
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="#2891: Not working")
@pytest.mark.skipif(vendor_name == "iluvatar", reason="#2891: Not working")
@pytest.mark.skipif(flaggems_vllm.vendor_name == "cambricon", reason="#2891: TypeError")
def test_grouped_topk_no_renorm():
    bench = GroupedTopKBenchmark(
        op_name="grouped_topk",
        torch_op=ref_grouped_topk,
        dtypes=[torch.bfloat16],
        renormalize=False,
        scoring_func=0,
    )

    bench.set_gems(flaggems_vllm.grouped_topk)
    bench.run()


@pytest.mark.grouped_topk
@pytest.mark.skipif(
    USE_AITER, reason="scoring_func == 0 is not supported by moe_fused_gate in aiter"
)
@pytest.mark.skipif(
    USE_MATE, reason="scoring_func == 0 is not supported by moe_fused_gate in mate"
)
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="#2891: Not working ")
@pytest.mark.skipif(vendor_name == "iluvatar", reason="#2891: Not working")
@pytest.mark.skipif(flaggems_vllm.vendor_name == "cambricon", reason="#2891: TypeError")
def test_grouped_topk_score_0():
    bench = GroupedTopKBenchmark(
        op_name="grouped_topk",
        torch_op=ref_grouped_topk,
        dtypes=[torch.bfloat16],
        renormalize=True,
        scoring_func=0,
    )

    bench.set_gems(flaggems_vllm.grouped_topk)
    bench.run()


@pytest.mark.grouped_topk
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="#2891: Not working")
@pytest.mark.skipif(vendor_name == "iluvatar", reason="#2891: Not working")
@pytest.mark.skipif(flaggems_vllm.vendor_name == "cambricon", reason="#2891: TypeError")
def test_grouped_topk_score_1():
    bench = GroupedTopKBenchmark(
        op_name="grouped_topk",
        torch_op=ref_grouped_topk,
        dtypes=[torch.bfloat16],
        renormalize=True,
        scoring_func=1,
    )

    bench.set_gems(flaggems_vllm.grouped_topk)
    bench.run()
