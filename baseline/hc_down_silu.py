# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Installed Qwen4 native and the shared BF16 down-projection workload."""

import torch

OP_NAME = "hc_down_silu"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ()
NATIVE_API = "vllm.models.qwen4_exp.nvidia.ops.cute_dsl.hc_down_silu.hc_down_silu"
NATIVE_UNAVAILABLE_REASON = None
NATIVE_VERSION = None
FIELDS = ("m", "k", "n", "rank", "hc", "layout")


def native():
    global NATIVE_UNAVAILABLE_REASON, NATIVE_VERSION
    NATIVE_VERSION = None
    try:
        import vllm
        from vllm.models.qwen4_exp.nvidia.ops.cute_dsl.hc_down_silu import hc_down_silu
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    if not callable(hc_down_silu):
        NATIVE_UNAVAILABLE_REASON = "installed vLLM hc_down_silu is not callable"
        return None
    NATIVE_VERSION = getattr(vllm, "__version__", None)
    NATIVE_UNAVAILABLE_REASON = None
    return hc_down_silu


def grid():
    # Upstream model tests use K=10240/rank=320/hc=4 and physical N=336.
    # Keep a valid native binding first for the standalone native collector.
    shapes = [(m, 10240, 320, 4) for m in (1, 2, 4, 5, 16, 32, 48, 49, 512, 0)]
    shapes += [
        (1, 2040, 29, 3),
        (4, 2048, 29, 3),
        (5, 2048, 29, 3),
        (16, 2056, 29, 3),
        (5, 2047, 17, 5),
        (32, 2051, 17, 5),
    ]
    bindings = [
        dict(m=m, k=k, n=r + h + pad, rank=r, hc=h, layout=layout)
        for m, k, r, h in shapes
        for pad in (0, 12)
        for layout in ("dense", "row_padded")
    ]
    # One shared 4 GiB backing validates both input address products at 2^31
    # elements. Only three elements are initialized or read, and outputs are tiny.
    bindings.append(dict(m=3, k=1, n=3, rank=1, hc=2, layout="wide_alias"))
    return bindings


def key_shape(binding):
    return [binding[key] for key in FIELDS]


def build_inputs(binding, dtype, device):
    if dtype != torch.bfloat16:
        raise TypeError("hc_down_silu workload requires BF16")
    m, k, n, rank, hc, layout = key_shape(binding)
    gen = torch.Generator().manual_seed(20261113)
    if layout == "wide_alias":
        if (m, k, n, rank, hc) != (3, 1, 3, 1, 2):
            raise ValueError("wide_alias requires the shared large-stride binding")
        values = torch.tensor([[1.0], [-0.5], [2.0]], dtype=dtype)
        shared = torch.empty_strided((3, 1), (1 << 30, 1), dtype=dtype, device=device)
        shared.copy_(values)
        return (shared, shared, rank, hc), {}
    pad = 13 if layout == "row_padded" else 0

    def tensor(rows):
        host = (torch.randn((rows, k), generator=gen) * 0.1).to(dtype)
        out = torch.empty_strided((rows, k), (k + pad, 1), dtype=dtype, device=device)
        out.copy_(host)
        return out

    return (tensor(m), tensor(n), rank, hc), {}


def unsupported_reason(binding, dtype):
    m, k, _, _, _, layout = key_shape(binding)
    if m < 1 or m > 48:
        return "outside native fused model domain 1<=M<=48 (no Linear substitute)"
    if k % 8:
        return "native model gate requires K divisible by 8"
    if layout != "dense":
        return "row-strided inputs are a candidate extension to native dense layout"
    return None


def config(binding, dtype):
    m, k, n, rank, hc, layout = key_shape(binding)
    stride = (
        (1 << 30)
        if layout == "wide_alias"
        else k + (13 if layout == "row_padded" else 0)
    )

    def output(width, offset):
        return dict(
            shape=[m, width],
            dtype=str(dtype),
            stride=[n, 1],
            storage_offset=offset,
            storage="packed_output",
        )

    return dict(
        dims=dict(binding),
        inputs=dict(
            x=dict(shape=[m, k], dtype=str(dtype), stride=[stride, 1]),
            weight=dict(shape=[n, k], dtype=str(dtype), stride=[stride, 1]),
            rank=dict(scalar=rank),
            hc=dict(scalar=hc),
        ),
        outputs=dict(activation=output(rank, 0), injection_logits=output(hc, rank)),
        alias="two disjoint views of fresh shared [M,N] storage; no input alias",
        input_alias=(
            "x and weight share the same view" if layout == "wide_alias" else "none"
        ),
        computed_columns=rank + hc,
        pad_columns="uncomputed and unread",
        mutation="none",
        accumulation="FP32 -> BF16 -> FP32 rank SiLU -> BF16",
        native_api=NATIVE_API,
        native_version=NATIVE_VERSION,
        baseline_kind="installed-vllm",
        graph=False,
        scope=(
            "native-model"
            if unsupported_reason(binding, dtype) is None
            else "candidate-extension"
        ),
    )
