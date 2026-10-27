# SPDX-License-Identifier: Apache-2.0
"""Native-only workload definition; never imported by production kernels."""
from importlib import import_module

import torch

OP_NAME = "chunk_kda_scaled_dot_kkt_fwd"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False


def native():
    return getattr(import_module("vllm.model_executor.layers.fla.ops.kda"), OP_NAME)


def grid():
    return [
        dict(B=1, T=1, H=1, K=64, BT=64, lengths=None),
        dict(B=1, T=65, H=4, K=128, BT=64, lengths=None),
        dict(B=1, T=512, H=8, K=128, BT=64, lengths=None),
        dict(B=1, T=2048, H=16, K=128, BT=64, lengths=None),
        dict(B=1, T=513, H=4, K=128, BT=64, lengths=[17, 0, 64, 432]),
        dict(B=1, T=1024, H=8, K=128, BT=64, lengths=[128] * 8),
    ]


def build_inputs(binding, dtype, device):
    b, t, h, k, bt = [binding[n] for n in ("B", "T", "H", "K", "BT")]
    gen = torch.Generator().manual_seed(2702)
    q, key = [
        (torch.randn((b, t, h, k), generator=gen) * 0.2).to(dtype) for _ in range(2)
    ]
    gate = torch.empty((b, t, h, k), dtype=torch.float32)
    lengths = binding["lengths"]
    spans = (
        [(0, sum(lengths[:i]), sum(lengths[: i + 1])) for i in range(len(lengths))]
        if lengths is not None
        else [(ib, 0, t) for ib in range(b)]
    )
    for ib, start, end in spans:
        for begin in range(start, end, bt):
            n = min(bt, end - begin)
            gate[ib, begin : begin + n] = (
                -torch.rand((n, h, k), generator=gen) * 0.1
            ).cumsum(0)
    beta = torch.rand((b, t, h), generator=gen)
    kw = dict(scale=k**-0.5, chunk_size=bt, output_dtype=torch.float32)
    if lengths is not None:
        cu = [0]
        for n in lengths:
            cu.append(cu[-1] + n)
        kw["cu_seqlens"] = torch.tensor(cu, device=device, dtype=torch.int64)
        kw["chunk_indices"] = torch.tensor(
            [(i, c) for i, n in enumerate(lengths) for c in range((n + bt - 1) // bt)],
            device=device,
            dtype=torch.int64,
        ).reshape(-1, 2)
    return tuple(x.to(device).contiguous() for x in (q, key, gate, beta)), kw


def key_shape(binding):
    return repr(tuple(binding[n] for n in ("B", "T", "H", "K", "BT", "lengths")))


def config(binding, dtype):
    b, t, h, k, bt = [binding[n] for n in ("B", "T", "H", "K", "BT")]
    return dict(
        binding=binding,
        qk_dtype=str(dtype),
        gate_dtype="torch.float32",
        beta_dtype="torch.float32",
        output_dtype="torch.float32",
        qk_shape=[b, t, h, k],
        beta_shape=[b, t, h],
        output_shape=[b, t, h, bt],
        layout="contiguous seq-first",
        gate_domain="chunk cumulative log2",
        scale=k**-0.5,
        indices="explicit" if binding["lengths"] else None,
        compute_precision=(
            "FP32 accumulation; operand precision varies by implementation; "
            "see run manifest"
        ),
    )
