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

import math
import random
import time

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)

device = flaggems_vllm.device
vendor_name = flaggems_vllm.vendor_name

if cfg.QUICK_MODE:
    N_TOKEN_LIST = [8]
    N_TOKEN_LIST_DEEPSEEK_V3_2 = [16384]
    N_TOKEN_LIST_GLM_5_2 = [64]
    N_EXPERT_LIST = [16]
    N_GROUP_LIST = [4]
    TOPK_LIST = [2]
    RENORMALIZE_LIST = [True]
    SCORING_FUNC_LIST = [0]
    DTYPE_LIST = [torch.float32]
else:
    N_TOKEN_LIST = [1, 3, 8]
    N_TOKEN_LIST_DEEPSEEK_V3_2 = [1, 8, 32, 64, 128, 256, 496, 512, 16384]
    N_TOKEN_LIST_GLM_5_2 = [1, 8, 64, 193, 449, 16384]
    N_EXPERT_LIST = [16]
    N_EXPERT_LIST = [8, 16]
    N_GROUP_LIST = [2, 4]
    TOPK_LIST = [1, 2]
    RENORMALIZE_LIST = [True, False]
    SCORING_FUNC_LIST = [0, 1]
    DTYPE_LIST = [torch.float32]


MAX_IDX = 0xFFFF
SIGN_MASK_INT32 = torch.tensor(0x80000000, dtype=torch.uint32, device=device).view(
    torch.int32
)
SIGN_MASK_INT64 = torch.tensor(0x80000000, dtype=torch.int64, device=device)


def _pack_val_idx_fp32(val: torch.Tensor, idx: torch.Tensor):
    bits = val.view(torch.int32)
    sign = bits & SIGN_MASK_INT32
    key = torch.where(sign != 0, ~bits, bits).to(torch.int64)
    key = torch.where(sign != 0, key, key | SIGN_MASK_INT64)
    high = key << 16
    low = (0xFFFF & (MAX_IDX - idx)).to(torch.int64)
    return high | low


def _unpack_val_idx_fp32(pair: torch.Tensor):
    key = pair >> 16
    sign = key & SIGN_MASK_INT64
    bits = torch.where(sign != 0, key ^ SIGN_MASK_INT64, key).to(torch.int32)
    bits = torch.where(sign != 0, bits, ~bits)
    val = bits.view(torch.float32)
    idx = (MAX_IDX - (pair & 0xFFFF)).to(torch.int64)
    return val, idx


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
    Adapted from vLLM: vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py
    Wrap torch.topk with packing and unpacking logic to fix stability issues.
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
    tmp_group_ids = torch.arange(
        0, num_expert_group, dtype=torch.int32, device=scores.device
    )
    tmp_group_ids = tmp_group_ids[None, :].expand(num_token, -1)
    group_pairs = _pack_val_idx_fp32(group_scores, tmp_group_ids)
    if vendor_name == "mthreads":
        # muDNN(v3105): ERROR# INVALID_PARAMETER in TopK::Run, Reason: Unsupported in data type: INT64
        top_group_pairs = torch.topk(
            group_pairs.cpu(), k=topk_group, dim=-1, sorted=use_sorted
        )[0].to(scores.device)
    else:
        top_group_pairs = torch.topk(
            group_pairs, k=topk_group, dim=-1, sorted=use_sorted
        )[0]
    _top_group_scores, group_idx = _unpack_val_idx_fp32(
        top_group_pairs
    )  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]

    tmp_ids = torch.arange(0, scores.size(1), dtype=torch.int32, device=scores.device)
    tmp_ids = tmp_ids[None, :].expand(num_token, -1)
    pairs = _pack_val_idx_fp32(tmp_scores, tmp_ids)
    if vendor_name == "mthreads":
        # muDNN(v3105): ERROR# INVALID_PARAMETER in TopK::Run, Reason: Unsupported in data type: INT64
        top_pairs = torch.topk(pairs.cpu(), k=topk, dim=-1, sorted=use_sorted)[0].to(
            scores.device
        )
    else:
        top_pairs = torch.topk(pairs, k=topk, dim=-1, sorted=use_sorted)[0]
    if bias is not None:
        _, topk_ids = _unpack_val_idx_fp32(top_pairs)
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = _unpack_val_idx_fp32(top_pairs)

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
    if renormalize and scoring_func == 1:
        num_tokens = scores.size(0)
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device=scores.device
        )
        topk_ids = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=scores.device
        )
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
    else:
        return torch_grouped_topk(
            scores,
            num_expert_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            bias,
            scoring_func,
        )


def is_power_of_two(n):
    return n > 0 and math.log2(n).is_integer()


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
    """
    Adapted from vllm-musa: vllm_musa/model_executor/layers/fused_moe/router/grouped_topk_router.py
    """
    if (
        scoring_func == 1
        and scores.shape[1] % 32 == 0
        and (
            scores.shape[1] // num_expert_group <= 32
            or (num_expert_group == 1 and scores.shape[1] in {160, 256, 384})
        )
        and (bias is not None and is_power_of_two(bias.shape[0]))
    ):
        # Conditions added relative to vllm-musa:
        # 1. moe_fused_gate in mate always perform a sigmoid operation
        # 2. moe_fused_gate in mate requires (num_expert % WARP_SIZE) == 0
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
    else:
        return torch_grouped_topk(
            scores,
            num_expert_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            bias,
            scoring_func,
        )


def ascend_moe_gating_top_k(
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
    Adapted from vllm-ascend: ./vllm_ascend/ops/fused_moe/experts_selector.py
    """
    if scoring_func == 1 and renormalize:
        norm_type = 1  # 0: softmax; 1: sigmoid
        renorm = int(renormalize)
        topk_weights, topk_ids, _out = torch.ops._C_ascend.moe_gating_top_k(
            scores.float(),
            k=topk,
            k_group=topk_group,
            group_count=num_expert_group,
            group_select_mode=1,
            renorm=renorm,
            norm_type=norm_type,
            out_flag=False,
            routed_scaling_factor=routed_scaling_factor,
            eps=1e-20,
            bias_opt=bias.float(),
        )
        return topk_weights, topk_ids.to(torch.int32)
    else:
        return torch_grouped_topk(
            scores,
            num_expert_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            bias,
            scoring_func,
        )


try:
    if vendor_name == "hygon":
        from aiter import moe_fused_gate  # noqa: F401

        ref_grouped_topk = aiter_biased_grouped_topk
    elif vendor_name == "mthreads":
        from mate import moe_fused_gate  # noqa: F401

        ref_grouped_topk = mthreads_grouped_topk
    elif vendor_name == "ascend":
        from vllm_ascend.utils import enable_custom_op

        enable_custom_op()
        if hasattr(torch.ops._C_ascend, "moe_gating_top_k"):
            ref_grouped_topk = ascend_moe_gating_top_k
        else:
            ref_grouped_topk = torch_grouped_topk
    else:
        import vllm._custom_ops  # noqa: F401

        if hasattr(torch.ops._moe_C, "grouped_topk"):
            ref_grouped_topk = torch.ops._moe_C.grouped_topk
        else:
            ref_grouped_topk = torch_grouped_topk
except (ImportError, AttributeError):
    ref_grouped_topk = torch_grouped_topk


def get_tolerance(dtype, scoring_func, renormalize):
    if dtype == torch.bfloat16:
        return 5e-3, 1e-3

    if dtype == torch.float16:
        if scoring_func == 1:
            return 1e-3, 1e-4
        else:
            return 5e-3, 1e-3

    if renormalize:
        return 5e-4, 1e-4
    return 1e-5, 1e-5


@pytest.mark.grouped_topk
@pytest.mark.parametrize("n_token", N_TOKEN_LIST_DEEPSEEK_V3_2)
@pytest.mark.parametrize("renormalize", RENORMALIZE_LIST)
@pytest.mark.parametrize("scoring_func", SCORING_FUNC_LIST)
def test_grouped_topk_deepseek_v3_2(
    n_token,
    renormalize,
    scoring_func,
):
    """Test grouped_topk accuracy with configs from DeepSeek-v3.2"""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    n_expert = 256
    n_group = 8
    topk = 8
    topk_group = 4
    routed_scaling_factor = 1.0
    scores_dtype = torch.bfloat16
    bias_dtype = torch.float32

    scores = torch.randn(
        (n_token, n_expert), dtype=scores_dtype, device=flaggems_vllm.device
    )
    bias = torch.randn((n_expert,), dtype=bias_dtype, device=flaggems_vllm.device)

    ref_topk_weights, ref_topk_ids = ref_grouped_topk(
        scores.clone(),
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )
    ref_topk_weights = utils.to_reference(ref_topk_weights)
    ref_topk_ids = utils.to_reference(ref_topk_ids)

    res_topk_weights, res_topk_ids = flaggems_vllm.grouped_topk(
        scores.clone(),
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )

    utils.gems_assert_equal(res_topk_ids, ref_topk_ids)

    atol, rtol = get_tolerance(ref_topk_weights.dtype, scoring_func, renormalize)
    res_topk_weights = utils.to_reference(res_topk_weights)
    torch.testing.assert_close(res_topk_weights, ref_topk_weights, atol=atol, rtol=rtol)


@pytest.mark.grouped_topk
@pytest.mark.parametrize("n_token", N_TOKEN_LIST_GLM_5_2)
@pytest.mark.parametrize("renormalize", RENORMALIZE_LIST)
@pytest.mark.parametrize("scoring_func", SCORING_FUNC_LIST)
def test_grouped_topk_glm_5_2(
    n_token,
    renormalize,
    scoring_func,
):
    """Test grouped_topk accuracy with configs from GLM-5.2"""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    n_expert = 256
    n_group = 1
    topk = 8
    topk_group = 1
    routed_scaling_factor = 1.0
    scores_dtype = torch.bfloat16
    bias_dtype = torch.float32

    scores = torch.randn(
        (n_token, n_expert), dtype=scores_dtype, device=flaggems_vllm.device
    )
    bias = torch.randn((n_expert,), dtype=bias_dtype, device=flaggems_vllm.device)

    ref_topk_weights, ref_topk_ids = ref_grouped_topk(
        scores.clone(),
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )
    ref_topk_weights = utils.to_reference(ref_topk_weights)
    ref_topk_ids = utils.to_reference(ref_topk_ids)

    res_topk_weights, res_topk_ids = flaggems_vllm.grouped_topk(
        scores.clone(),
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )

    utils.gems_assert_equal(res_topk_ids, ref_topk_ids)

    atol, rtol = get_tolerance(ref_topk_weights.dtype, scoring_func, renormalize)
    res_topk_weights = utils.to_reference(res_topk_weights)
    torch.testing.assert_close(res_topk_weights, ref_topk_weights, atol=atol, rtol=rtol)


@pytest.mark.grouped_topk
@pytest.mark.parametrize("n_token", N_TOKEN_LIST)
@pytest.mark.parametrize("n_expert", N_EXPERT_LIST)
@pytest.mark.parametrize("n_group", N_GROUP_LIST)
@pytest.mark.parametrize("topk", TOPK_LIST)
@pytest.mark.parametrize("renormalize", RENORMALIZE_LIST)
@pytest.mark.parametrize("scoring_func", SCORING_FUNC_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_grouped_topk(
    n_token,
    n_expert,
    n_group,
    topk,
    renormalize,
    scoring_func,
    dtype,
):
    """Test grouped_topk accuracy against vLLM CUDA implementation"""

    if n_expert % n_group != 0:
        return

    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    topk_group = topk
    routed_scaling_factor = 1.0

    scores = torch.randn((n_token, n_expert), dtype=dtype, device=flaggems_vllm.device)
    bias = torch.randn((n_expert,), dtype=dtype, device=flaggems_vllm.device)

    ref_topk_weights, ref_topk_ids = ref_grouped_topk(
        scores.clone(),
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )
    ref_topk_weights = utils.to_reference(ref_topk_weights)
    ref_topk_ids = utils.to_reference(ref_topk_ids)

    res_topk_weights, res_topk_ids = flaggems_vllm.grouped_topk(
        scores.clone(),
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )

    utils.gems_assert_equal(res_topk_ids, ref_topk_ids)

    atol, rtol = get_tolerance(dtype, scoring_func, renormalize)
    res_topk_weights = utils.to_reference(res_topk_weights)
    torch.testing.assert_close(res_topk_weights, ref_topk_weights, atol=atol, rtol=rtol)


@pytest.mark.grouped_topk
@pytest.mark.parametrize("n_token", [32, 64])
@pytest.mark.parametrize("n_expert", [64])
@pytest.mark.parametrize("n_group", [8])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("topk_group", [2])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_grouped_topk_large_scale(
    n_token,
    n_expert,
    n_group,
    topk,
    topk_group,
    renormalize,
    scoring_func,
    dtype,
):
    """Test grouped_topk with larger scale configurations"""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    routed_scaling_factor = 1.0

    scores = torch.randn((n_token, n_expert), dtype=dtype, device=flaggems_vllm.device)
    bias = torch.randn((n_expert,), dtype=dtype, device=flaggems_vllm.device)

    ref_topk_weights, ref_topk_ids = ref_grouped_topk(
        scores.clone(),
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )
    ref_topk_weights = utils.to_reference(ref_topk_weights)
    ref_topk_ids = utils.to_reference(ref_topk_ids)

    res_topk_weights, res_topk_ids = flaggems_vllm.grouped_topk(
        scores.clone(),
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )

    utils.gems_assert_equal(res_topk_ids, ref_topk_ids)

    atol, rtol = get_tolerance(dtype, scoring_func, renormalize)
    res_topk_weights = utils.to_reference(res_topk_weights)
    torch.testing.assert_close(res_topk_weights, ref_topk_weights, atol=atol, rtol=rtol)


@pytest.mark.grouped_topk
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize("renormalize", [True, False])
def test_grouped_topk_scaling_factor(routed_scaling_factor, renormalize):
    """Test grouped_topk with different scaling factors"""

    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    dtype = torch.float32
    scores = torch.randn((8, 16), dtype=dtype, device=flaggems_vllm.device)
    bias = torch.randn((16,), dtype=dtype, device=flaggems_vllm.device)

    ref_weights, ref_ids = ref_grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, routed_scaling_factor, bias, 0
    )
    ref_weights = utils.to_reference(ref_weights)
    ref_ids = utils.to_reference(ref_ids)

    res_weights, res_ids = flaggems_vllm.grouped_topk(
        scores.clone(),
        4,
        2,
        2,
        renormalize,
        routed_scaling_factor,
        bias,
        0,
    )

    utils.gems_assert_equal(res_ids, ref_ids)

    atol, rtol = get_tolerance(dtype, 0, renormalize)
    res_weights = utils.to_reference(res_weights)
    torch.testing.assert_close(res_weights, ref_weights, atol=atol, rtol=rtol)


@pytest.mark.grouped_topk
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", [0, 1])
def test_grouped_topk_single_token(renormalize, scoring_func):
    """Test grouped_topk with single token"""

    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    dtype = torch.float32
    scores = torch.randn((1, 16), dtype=dtype, device=flaggems_vllm.device)
    bias = torch.randn((16,), dtype=dtype, device=flaggems_vllm.device)

    ref_weights, ref_ids = ref_grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, 1.0, bias, scoring_func
    )
    ref_weights = utils.to_reference(ref_weights)
    ref_ids = utils.to_reference(ref_ids)

    res_weights, res_ids = flaggems_vllm.grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, 1.0, bias, scoring_func
    )

    utils.gems_assert_equal(res_ids, ref_ids)

    atol, rtol = get_tolerance(dtype, scoring_func, renormalize)
    res_weights = utils.to_reference(res_weights)
    torch.testing.assert_close(res_weights, ref_weights, atol=atol, rtol=rtol)


@pytest.mark.grouped_topk
@pytest.mark.parametrize("renormalize", [True, False])
def test_grouped_topk_sigmoid(renormalize):
    """Test grouped_topk with sigmoid scoring function"""
    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    dtype = torch.float32
    scores = torch.randn((8, 16), dtype=dtype, device=flaggems_vllm.device)
    bias = torch.randn((16,), dtype=dtype, device=flaggems_vllm.device)

    ref_weights, ref_ids = ref_grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, 1.0, bias, 1
    )
    ref_weights = utils.to_reference(ref_weights)
    ref_ids = utils.to_reference(ref_ids)

    res_weights, res_ids = flaggems_vllm.grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, 1.0, bias, 1
    )

    utils.gems_assert_equal(res_ids, ref_ids)

    atol, rtol = get_tolerance(dtype, 1, renormalize)
    res_weights = utils.to_reference(res_weights)
    torch.testing.assert_close(res_weights, ref_weights, atol=atol, rtol=rtol)


def _torch_padding_routing_reference(
    scores, groups, keep, topk, renormalize, scale, bias, scoring_func
):
    """Independent eager oracle, matching each existing route's score precision."""
    experts = scores.size(1)
    per_group = experts // groups
    fast = (groups == keep == 1 and experts <= 256 and topk <= 8) or (
        groups > 1
        and experts <= 256
        and per_group <= 32
        and per_group * keep <= 128
        and topk <= 8
        and keep <= 4
    )
    if fast:
        u = scores.float()
        unbiased = u.sigmoid() if scoring_func == 1 else u
        ranked = unbiased + bias.float()
    else:
        # Legacy generic path rounds each pointwise step to the input dtype,
        # including the final addition of the routing bias.
        unbiased = 0.5 * torch.tanh(0.5 * scores) + 0.5 if scoring_func == 1 else scores
        ranked = unbiased + bias.to(scores.dtype)
    group_score = ranked.float().view(-1, groups, per_group).topk(2, dim=-1)[0].sum(-1)
    if not fast:
        group_score = group_score.to(scores.dtype).float()
    group_ids = group_score.argsort(descending=True, stable=True)[:, :keep]
    enabled = torch.zeros_like(group_score, dtype=torch.bool)
    enabled.scatter_(1, group_ids, True)
    selected = enabled.unsqueeze(-1).expand(-1, -1, per_group).reshape_as(ranked)
    ranked = ranked.float().masked_fill(~selected, float("-inf"))
    ids = ranked.argsort(descending=True, stable=True)[:, :topk]
    weights = unbiased.float().gather(1, ids)
    if renormalize:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return weights * scale, ids.to(torch.int32)


@pytest.mark.grouped_topk
@pytest.mark.skipif(
    vendor_name != "nvidia", reason="padding is a CUDA routing contract"
)
@pytest.mark.parametrize(
    "experts,groups,keep,topk",
    [(512, 1, 1, 16), (256, 1, 1, 8), (256, 8, 4, 8), (384, 3, 2, 4)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["mixed", "all", "none"])
def test_grouped_topk_padding(experts, groups, keep, topk, dtype, mode):
    rows = 9
    # A deterministic, margin-heavy fixture avoids numerical ties in the valid rows.
    score = torch.linspace(-2.0, 2.0, experts, device=device)
    scores = torch.stack([score.roll(i * 17) for i in range(rows)]).to(dtype)
    bias = torch.linspace(-0.30, 0.30, experts, device=device, dtype=dtype)
    args = (scores, groups, keep, topk, True, 3.998, bias, 1)
    valid_reference = _torch_padding_routing_reference(*args)
    if mode == "mixed":
        mask = torch.arange(rows, device=device) % 3 == 1
    else:
        mask = torch.full((rows,), mode == "all", device=device, dtype=torch.bool)
    w, ids = flaggems_vllm.grouped_topk(*args, is_padding=mask)
    assert (w.shape, ids.shape) == ((rows, topk), (rows, topk))
    assert (w.dtype, ids.dtype) == (torch.float32, torch.int32)
    assert w.device == ids.device == scores.device
    torch.testing.assert_close(
        w[~mask], valid_reference[0][~mask], atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(ids[~mask], valid_reference[1][~mask], atol=0, rtol=0)
    assert torch.all(w[mask] == 0)
    assert torch.all(ids[mask] == -1)


@pytest.mark.grouped_topk
@pytest.mark.skipif(
    vendor_name != "nvidia", reason="padding is a CUDA routing contract"
)
def test_grouped_topk_padding_graph_is_dynamic():
    rows, experts, topk = 33, 512, 16
    scores = torch.randn(rows, experts, device=device, dtype=torch.float32)
    bias = torch.rand(experts, device=device, dtype=scores.dtype)
    mask = torch.zeros(rows, device=device, dtype=torch.bool)
    args = (scores, 1, 1, topk, True, 3.998, bias, 1)
    baseline = _torch_padding_routing_reference(*args)
    for _ in range(2):
        flaggems_vllm.grouped_topk(*args, is_padding=mask)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w, ids = flaggems_vllm.grouped_topk(*args, is_padding=mask)
    for a in (1, 2):
        mask.copy_(torch.arange(rows, device=device) % 3 == a)
        graph.replay()
        torch.testing.assert_close(w[~mask], baseline[0][~mask], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(ids[~mask], baseline[1][~mask], atol=0, rtol=0)
        assert torch.all(w[mask] == 0)
        assert torch.all(ids[mask] == -1)


@pytest.mark.grouped_topk
@pytest.mark.skipif(
    vendor_name != "nvidia", reason="padding is a CUDA routing contract"
)
def test_grouped_topk_padding_metadata_and_empty():
    scores = torch.empty((0, 16), device=device)
    bias = torch.zeros(16, device=device)
    mask = torch.empty(0, device=device, dtype=torch.bool)
    w, ids = flaggems_vllm.grouped_topk(scores, 1, 1, 4, False, 1.0, bias, 0, mask)
    assert w.shape == ids.shape == (0, 4)
    scores = torch.empty((3, 16), device=device)
    for bad in (
        torch.empty(2, device=device, dtype=torch.bool),
        torch.empty(3, device=device),
        torch.empty(6, device=device, dtype=torch.bool)[::2],
        torch.empty(3, device="cpu", dtype=torch.bool),
    ):
        with pytest.raises(ValueError, match="is_padding"):
            flaggems_vllm.grouped_topk(scores, 1, 1, 4, False, 1.0, bias, 0, bad)


@pytest.mark.grouped_topk
@pytest.mark.skipif(
    vendor_name != "nvidia", reason="padding is a CUDA routing contract"
)
def test_grouped_topk_padding_with_strided_scores():
    rows, experts = 7, 256
    backing = torch.randn(rows, experts + 8, device=device)
    scores = backing[:, :experts]
    bias = torch.randn(experts, device=device)
    mask = (torch.arange(rows, device=device) % 2) == 0
    args = (8, 4, 8, False, 1.5, bias, 0)
    ref = _torch_padding_routing_reference(scores, *args)
    w, ids = flaggems_vllm.grouped_topk(scores, *args, mask)
    torch.testing.assert_close(w[~mask], ref[0][~mask], rtol=0, atol=0)
    torch.testing.assert_close(ids[~mask], ref[1][~mask], rtol=0, atol=0)
    assert torch.all(w[mask] == 0)
    assert torch.all(ids[mask] == -1)


@pytest.mark.grouped_topk
@pytest.mark.parametrize("groups,keep", [(1, 1), (8, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("separate_scores", [False, True])
def test_grouped_topk_fast_path_fp32_order(groups, keep, dtype, separate_scores):
    # Tiny FP32 bias differences must survive routing even with BF16 inputs.
    scores = torch.ones((3, 256), device=device, dtype=dtype)
    bias = torch.arange(256, device=device, dtype=torch.float32) * (
        2**-20 if separate_scores else 0.0
    )
    weights, ids = flaggems_vllm.grouped_topk(
        scores, groups, keep, 8, True, 2.5, bias, 0
    )
    expected = (
        torch.arange(255, 247, -1, device=device)
        if separate_scores
        else torch.arange(8, device=device)
    )
    torch.testing.assert_close(ids, expected.expand(3, -1).int(), rtol=0, atol=0)
    torch.testing.assert_close(weights, torch.full_like(weights, 2.5 / 8))


@pytest.mark.grouped_topk
@pytest.mark.parametrize("experts,topk", [(8, 3), (17, 5), (63, 7), (129, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_grouped_topk_single_group_strided_ties(experts, topk, dtype):
    backing = torch.ones((5, experts + 11), device=device, dtype=dtype)
    scores = backing[:, :experts]
    bias = torch.zeros(experts, device=device, dtype=torch.float32)
    weights, ids = flaggems_vllm.grouped_topk(scores, 1, 1, topk, True, 1.0, bias, 0)
    expected = torch.arange(topk, device=device, dtype=torch.int32).expand(5, -1)
    torch.testing.assert_close(ids, expected, rtol=0, atol=0)
    torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / topk))


@pytest.mark.grouped_topk
@pytest.mark.parametrize("groups,keep", [(1, 1), (8, 4)])
def test_grouped_topk_fast_path_infinity_and_signed_zero(groups, keep):
    scores = torch.ones((2, 256), device=device)
    scores[:, [1, 253]] = float("inf")
    bias = torch.zeros(256, device=device)
    weights, ids = flaggems_vllm.grouped_topk(
        scores, groups, keep, 8, False, 1.0, bias, 0
    )
    expected = torch.tensor([1, 253, 0, 2, 3, 4, 5, 6], device=device).int()
    torch.testing.assert_close(ids, expected.expand(2, -1), rtol=0, atol=0)
    assert torch.isposinf(weights[:, :2]).all()
    torch.testing.assert_close(weights[:, 2:], torch.ones_like(weights[:, 2:]))

    scores.zero_()
    scores[:, ::2] = -0.0
    weights, ids = flaggems_vllm.grouped_topk(
        scores, groups, keep, 8, False, 1.0, bias, 0
    )
    expected = torch.arange(8, device=device).int().expand(2, -1)
    torch.testing.assert_close(ids, expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(weights), torch.signbit(scores[:, :8]))


@pytest.mark.grouped_topk
@pytest.mark.parametrize(
    "experts,groups,keep,topk", [(384, 3, 2, 4), (512, 8, 4, 8), (1024, 8, 4, 16)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("renormalize", [False, True])
def test_grouped_topk_generic_sigmoid_precision(
    experts, groups, keep, topk, dtype, renormalize
):
    torch.manual_seed(0)
    scores = torch.randn(9, experts, device=device, dtype=dtype)
    bias = torch.randn(experts, device=device, dtype=dtype)
    args = (scores, groups, keep, topk, renormalize, 2.5, bias, 1)
    expected = _torch_padding_routing_reference(*args)
    actual = flaggems_vllm.grouped_topk(*args)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-6, atol=2e-6)


@pytest.mark.grouped_topk
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA register selection path")
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("experts,topk", [(17, 3), (129, 5), (256, 8)])
@pytest.mark.parametrize("pattern", ["random", "refine", "boundary"])
def test_grouped_topk_register_candidate_refinement(
    compact, dtype, experts, topk, pattern
):
    from flaggems_vllm.ops.grouped_topk import _grouped_topk_register_select_kernel

    torch.manual_seed(2026)
    rows = 3
    # Exercise both expert and bias strides, including the exact-refinement load.
    scores = torch.randn(rows, 2 * experts, device=device, dtype=dtype)[:, ::2]
    bias = torch.randn(2 * experts, device=device, dtype=torch.float32)[::2]
    if pattern != "random":
        scores.fill_(1.0)
        if pattern == "refine":
            # Only K candidates share the bucket: compact selection must sort
            # their full FP32 keys, reversing the coarse index tie-break.
            bias.fill_(-1.0)
            bias[:topk] = 1.0 + torch.arange(topk, device=device) * 2**-20
        else:
            # The cutoff bucket contains more than K candidates: retry with
            # full-width keys or a true winner would be discarded.
            bias.copy_(torch.arange(experts, device=device) * 2**-20)
    reference = _torch_padding_routing_reference(scores, 1, 1, topk, True, 2.5, bias, 1)
    weights = torch.empty(rows, topk, device=device, dtype=torch.float32)
    indices = torch.empty(rows, topk, device=device, dtype=torch.int32)
    _grouped_topk_register_select_kernel.fn[(rows,)](
        scores,
        bias,
        weights,
        indices,
        None,
        rows,
        experts,
        topk,
        scores.stride(0),
        scores.stride(1),
        bias.stride(0),
        2.5,
        SCORING_FUNC=1,
        RENORMALIZE=True,
        HAS_PADDING=False,
        COMPACT=compact,
        num_warps=1,
        num_stages=1,
    )
    torch.testing.assert_close(indices, reference[1], rtol=0, atol=0)
    torch.testing.assert_close(weights, reference[0], rtol=2e-6, atol=2e-6)


@pytest.mark.grouped_topk
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA fused routing path")
@pytest.mark.parametrize(
    "experts,groups,keep,topk",
    [(256, 1, 1, 8), (256, 8, 4, 8), (384, 3, 2, 4), (512, 8, 4, 8), (1024, 8, 4, 16)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scoring_func", [0, 1])
@pytest.mark.parametrize("layout", ["row_gap", "expert_step", "transpose"])
def test_grouped_topk_fused_layouts(
    experts, groups, keep, topk, dtype, scoring_func, layout
):
    torch.manual_seed(2026)
    rows = 9
    if layout == "row_gap":
        scores = torch.randn(rows, experts + 11, device=device, dtype=dtype)[
            :, :experts
        ]
    elif layout == "expert_step":
        scores = torch.randn(rows, 2 * experts, device=device, dtype=dtype)[:, ::2]
    else:
        scores = torch.randn(experts, rows, device=device, dtype=dtype).T
    bias = torch.randn(2 * experts, device=device, dtype=torch.float32)[::2]
    args = (groups, keep, topk, True, 2.5, bias, scoring_func)
    expected = _torch_padding_routing_reference(scores, *args)
    actual = flaggems_vllm.grouped_topk(scores, *args)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-6, atol=2e-6)


@pytest.mark.grouped_topk
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA fused routing path")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "pattern", ["ties", "zeros", "all_neginf", "infinity", "one_finite", "nan"]
)
def test_grouped_topk_generic_nonfinite(dtype, pattern):
    scores = torch.ones((3, 384), device=device, dtype=dtype)
    bias = torch.zeros(384, device=device, dtype=dtype)
    expected_ids = torch.arange(4, device=device).expand(3, -1).int()
    expected_weights = torch.ones((3, 4), device=device)
    if pattern == "zeros":
        scores.zero_()
        scores[:, ::2] = -0.0
        expected_weights.zero_()
    elif pattern == "all_neginf":
        scores.fill_(-float("inf"))
        expected_weights.fill_(0.25)
    elif pattern == "infinity":
        scores[:, 1] = float("inf")
        expected_ids = expected_ids + 128
    elif pattern == "one_finite":
        scores.fill_(-float("inf"))
        scores[:, 0] = 1.0
        expected_weights.fill_(0.25)
    elif pattern == "nan":
        scores[:, 0] = float("nan")
        expected_ids = expected_ids + 1
    weights, ids = flaggems_vllm.grouped_topk(scores, 3, 2, 4, False, 1.0, bias, 0)
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
