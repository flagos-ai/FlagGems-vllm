# SPDX-License-Identifier: Apache-2.0
"""Direct vLLM baseline contract. Each invocation overwrites the value input.

Collectors must restore v outside warmup/timing. The task runner implements
that protocol explicitly. The legacy repository collector ignores IS_INPLACE
and must not collect this module until it implements state restoration.
"""

import torch

OP_NAME = "chunk_kda_with_fused_gate"
DTYPES = (torch.float16, torch.bfloat16)
IS_INPLACE = True


def native():
    try:
        from vllm.model_executor.layers.fla.ops.kda import chunk_kda_with_fused_gate
    except (ImportError, AttributeError):
        return None
    return chunk_kda_with_fused_gate


def grid():
    return [
        {"lengths": [64], "H": 8, "K": 64, "V": 64},
        {"lengths": [512], "H": 8, "K": 128, "V": 128},
        {"lengths": [2048], "H": 16, "K": 128, "V": 128},
        {"lengths": [8192], "H": 96, "K": 128, "V": 128},
        {"lengths": [1300, 547, 2048, 963, 271, 3063], "H": 96, "K": 128, "V": 128},
        {"lengths": [1024] * 8, "H": 96, "K": 128, "V": 128},
    ]


def build_inputs(binding, dtype, device):
    import itertools

    lengths, h, dk, dv = (binding[k] for k in ("lengths", "H", "K", "V"))
    t = sum(lengths)
    torch.manual_seed(20261029)
    q = (torch.randn(1, t, h, dk, device=device) * 0.2).to(dtype)
    k = torch.randn_like(q) * 0.2
    v = (torch.randn(1, t, h, dv, device=device) * 0.2).to(dtype)
    g = (torch.randn(1, t, h, dk, device=device) * 0.5).to(dtype)
    beta = torch.sigmoid(torch.randn(1, t, h, device=device))
    a = torch.full((h,), -2.0, device=device)
    bias = torch.randn(h, dk, device=device) * 0.1
    initial = torch.randn(len(lengths), h, dv, dk, device=device) * 0.1
    cu = (
        torch.tensor(
            [0, *itertools.accumulate(lengths)], dtype=torch.long, device=device
        )
        if len(lengths) > 1
        else None
    )
    return (q, k, v, g, beta, a, bias), dict(
        scale=dk**-0.5,
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
    )


def key_shape(binding):
    return (
        1,
        sum(binding["lengths"]),
        binding["H"],
        binding["K"],
        binding["V"],
        tuple(binding["lengths"]),
    )


def config(binding, dtype):
    t, h, dk, dv = sum(binding["lengths"]), binding["H"], binding["K"], binding["V"]
    return dict(
        op=OP_NAME,
        variant="softplus_gate_prefill64",
        dtype=str(dtype),
        qk_shape=[1, t, h, dk],
        value_shape=[1, t, h, dv],
        gate_shape=[1, t, h, dk],
        beta_shape=[1, t, h],
        beta_dtype="float32",
        parameter_dtype="float32",
        gate_input_dtype=str(dtype),
        state_shape=[len(binding["lengths"]), h, dv, dk],
        state_dtype="float32",
        lengths=binding["lengths"],
        layout="contiguous_seq_first",
        qk_stride=[t * h * dk, h * dk, dk, 1],
        value_stride=[t * h * dv, h * dv, dv, 1],
        scale=dk**-0.5,
        norm=True,
        gate="negative_exp_A_times_softplus",
        beta_already_activated=True,
        chunk_size=64,
        output_final_state=True,
        mutation="contiguous v overwritten; output aliases v",
    )
