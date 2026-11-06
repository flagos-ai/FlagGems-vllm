# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Call the installed vLLM-Ascend 0.24.0rc1 BF16/FP16 MoE implementation."""

from functools import lru_cache

import torch
from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    build_fused_experts_input,
    build_mlp_compute_input,
    build_token_dispatch_input,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import enable_custom_op


@lru_cache(maxsize=16)
def dispatcher(experts, topk):
    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend custom operators could not be loaded")
    return TokenDispatcherWithAllGather(
        num_experts=experts, num_local_experts=experts, top_k=topk
    )


def native_moe(hidden_states, w1, w2, probabilities, expert_ids):
    routing = dispatcher(w1.shape[0], expert_ids.shape[1])
    inputs = build_fused_experts_input(
        hidden_states=hidden_states,
        topk_weights=probabilities.to(hidden_states.dtype),
        topk_ids=expert_ids,
        w1=w1,
        w2=w2,
        quant_type=QuantType.NONE,
        dynamic_eplb=False,
        need_trans=False,
    )
    routed = routing.token_dispatch(
        build_token_dispatch_input(fused_experts_input=inputs)
    )
    expert_outputs, _ = unified_apply_mlp(
        mlp_compute_input=build_mlp_compute_input(
            fused_experts_input=inputs,
            token_dispatch_output=routed,
            use_fusion_ops=False,
        )
    )
    return routing.token_combine(expert_outputs, routed.combine_metadata)


native_bf16 = native_moe


def int8_input_iter(shapes, dtype):
    import flaggems_vllm
    from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT8B128
    from tests.accuracy_utils import gems_assert_close
    from tests.test_fused_marlin_moe import (
        _make_inputs_w8a16_int8,
        _reference_w8a16_grouped,
        compute_max_diff,
    )

    geometry = None
    for num_tokens, experts, hidden, intermediate, topk in shapes:
        if geometry != (experts, hidden, intermediate):
            _, w1, w2, ref1, ref2, _, _, s1, s2 = _make_inputs_w8a16_int8(
                1, experts, hidden, intermediate, topk, dtype, flaggems_vllm.device
            )
            baseline1 = ref1.to(torch.bfloat16).transpose(1, 2).contiguous()
            baseline2 = ref2.to(torch.bfloat16).transpose(1, 2).contiguous()
            fp16_references = (
                (ref1.transpose(1, 2).contiguous(), ref2.transpose(1, 2).contiguous())
                if dtype == torch.float16
                else None
            )
            geometry = (experts, hidden, intermediate)
        else:
            pass
        torch.manual_seed(num_tokens)
        hidden_states = torch.randn(
            (num_tokens, hidden), device=flaggems_vllm.device, dtype=dtype
        )
        probabilities, ids = torch.randn(
            (num_tokens, experts), device=flaggems_vllm.device
        ).topk(topk)
        probabilities = torch.softmax(probabilities, dim=-1)
        ids = ids.to(torch.int32)
        expected = _reference_w8a16_grouped(
            hidden_states, ref1, ref2, probabilities, ids
        )
        baseline_hidden = hidden_states.to(torch.bfloat16)
        baseline_expected = (
            expected
            if dtype == torch.bfloat16
            else _reference_w8a16_grouped(
                baseline_hidden,
                baseline1.transpose(1, 2),
                baseline2.transpose(1, 2),
                probabilities,
                ids,
            )
        )
        outputs = (
            native_moe(baseline_hidden, baseline1, baseline2, probabilities, ids),
            flaggems_vllm.fused_marlin_moe(
                hidden_states,
                w1,
                w2,
                None,
                None,
                s1,
                s2,
                probabilities,
                ids,
                QUANT_TYPE_UINT8B128,
            ),
        )
        for output, reference in zip(outputs, (baseline_expected, expected)):
            error = compute_max_diff(output.float(), reference)
            gems_assert_close(error, torch.zeros_like(error), torch.float32, atol=0.04)
        yield hidden_states, w1, w2, s1, s2, baseline1, baseline2, (
            baseline_hidden if dtype == torch.float16 else None
        ), fp16_references, probabilities, ids
