# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""PyTorch implementation of the vLLM gate projection and routing contract."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GateProfile:
    hidden: int
    experts: int
    topk: int
    scoring: str
    scale: float
    logits_dtype: torch.dtype = torch.float32
    groups: int = 1
    top_groups: int = 1


GATE_PROFILES = {
    "dsv4-flash": GateProfile(4096, 256, 6, "sqrtsoftplus", 1.5),
    "dsv41-flash": GateProfile(5120, 384, 6, "sqrtsoftplus", 1.5),
    "dsv32": GateProfile(7168, 256, 8, "sigmoid", 2.5, torch.bfloat16, 8, 4),
    "qwen-35b": GateProfile(2048, 256, 8, "softmax", 1.0, torch.bfloat16),
    "qwen-397b": GateProfile(4096, 512, 10, "softmax", 1.0, torch.bfloat16),
    "kimi-k25": GateProfile(7168, 384, 8, "sigmoid", 2.827, torch.bfloat16),
}


def generate_gate_inputs(profile, tokens, device):
    rng = torch.Generator(device="cpu").manual_seed(319 + tokens)
    x = torch.randn((tokens, profile.hidden), generator=rng).to(
        device=device, dtype=torch.bfloat16
    )
    weight = (
        torch.randn((profile.experts, profile.hidden), generator=rng)
        * profile.hidden**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    bias = None
    if profile.scoring != "softmax":
        bias = (torch.randn(profile.experts, generator=rng) * 0.03).to(device)
    args = (x, weight, profile.topk)
    kwargs = {
        "scoring_func": profile.scoring,
        "routed_scaling_factor": profile.scale,
        "bias": bias,
        "router_logits_dtype": profile.logits_dtype,
        "num_expert_group": profile.groups,
        "topk_group": profile.top_groups,
    }
    return args, kwargs


def router_scores(logits, scoring_func):
    logits = logits.float()
    if scoring_func == "sqrtsoftplus":
        return F.softplus(logits).sqrt()
    if scoring_func == "sigmoid":
        return logits.sigmoid()
    if scoring_func == "softmax":
        return logits.softmax(dim=-1)
    return logits


def torch_megagate(
    x,
    weight,
    num_topk,
    *,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.0,
    bias=None,
    image_bias=None,
    image_token_mask=None,
    fix_routing_mask=None,
    unmapped_topk_idx=None,
    num_expert_group=1,
    topk_group=1,
    renormalize=True,
    router_logits_dtype=torch.float32,
):
    if router_logits_dtype == torch.float32:
        logits = torch.mm(x, weight.T, out_dtype=torch.float32)
    else:
        logits = torch.mm(x, weight.T)
    scores = router_scores(logits, scoring_func)
    if unmapped_topk_idx is not None and fix_routing_mask is None:
        ids = unmapped_topk_idx
    else:
        ranking = scores if bias is None else scores + bias
        if image_bias is not None:
            ranking = torch.where(
                image_token_mask[:, None], scores + image_bias, ranking
            )
        if num_expert_group > topk_group:
            grouped = ranking.unflatten(-1, (num_expert_group, -1))
            if bias is None:
                group_scores = grouped.amax(-1)
            else:
                group_scores = grouped.topk(2, dim=-1).values.sum(-1)
            groups = group_scores.topk(topk_group, dim=-1).indices
            keep = torch.zeros_like(group_scores, dtype=torch.bool)
            keep.scatter_(1, groups, True)
            ranking = grouped.masked_fill(~keep.unsqueeze(-1), -torch.inf).flatten(1)
        ids = ranking.topk(num_topk, dim=-1).indices
        if unmapped_topk_idx is not None:
            ids = torch.where(fix_routing_mask[:, None], unmapped_topk_idx, ids)
    values = scores.gather(1, ids)
    if renormalize:
        values = values / values.sum(1, keepdim=True)
    if routed_scaling_factor != 1.0:
        values = values * routed_scaling_factor
    return values, ids
