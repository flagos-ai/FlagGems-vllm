# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Installed vLLM hc_combine baseline and shared, immutable workload matrix."""

import torch

OP_NAME = "hc_combine"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
IS_INPLACE = False
MUTATED_INPUTS = ()
NATIVE_API = "vllm.models.qwen4_exp.nvidia.ops.hc.hc_combine"
NATIVE_UNAVAILABLE_REASON = None
NATIVE_VERSION = None


def native():
    global NATIVE_UNAVAILABLE_REASON, NATIVE_VERSION
    NATIVE_VERSION = None
    try:
        import vllm
        from vllm.models.qwen4_exp.nvidia.ops.hc import hc_combine
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    if not callable(hc_combine):
        NATIVE_UNAVAILABLE_REASON = "Installed vLLM hc_combine is not callable"
        return None
    NATIVE_VERSION = getattr(vllm, "__version__", None)
    NATIVE_UNAVAILABLE_REASON = None
    return hc_combine


def grid():
    # N, HC, D, layout; both branches have exactly the same workload denominator.
    shapes = [
        (1, 1, 1),
        (1, 3, 127),
        (1, 4, 4096),
        (7, 3, 513),
        (16, 4, 4096),
        (127, 4, 1024),
        (1024, 4, 4096),
        (4096, 4, 1024),
    ]
    return [
        dict(n=n, hc=hc, d=d, layout=layout, injection=injection)
        for n, hc, d in shapes
        for layout in ("dense", "packed")
        for injection in ("none", "same", "fp32")
    ]


def key_shape(binding):
    return [binding[k] for k in ("n", "hc", "d", "layout", "injection")]


def build_inputs(binding, dtype, device):
    n, hc, d = (binding[k] for k in ("n", "hc", "d"))
    packed = binding["layout"] == "packed"
    gen = torch.Generator().manual_seed(20261106)

    def tensor(width, dt, pad):
        # Construct on CPU so every backend sees identical input bits and strides.
        host = torch.randn((n, width + pad), generator=gen, dtype=dt)[:, :width]
        out = torch.empty_strided(host.shape, host.stride(), dtype=dt, device=device)
        out.copy_(host)
        return out

    residual = tensor(hc * d, dtype, 17 if packed else 0)
    block = tensor(d, dtype, 11 if packed else 0)
    variant = binding["injection"]
    injection = (
        None
        if variant == "none"
        else tensor(
            hc, torch.float32 if variant == "fp32" else dtype, 256 if packed else 0
        )
    )
    return (residual, block, injection, hc), {}


def config(binding, dtype):
    n, hc, d = (binding[k] for k in ("n", "hc", "d"))
    packed = binding["layout"] == "packed"
    variant = binding["injection"]

    def tensor(width, dt, pad=0):
        return dict(shape=[n, width], dtype=str(dt), stride=[width + pad, 1])

    return dict(
        inputs=dict(
            residual=tensor(hc * d, dtype, 17 if packed else 0),
            block_output=tensor(d, dtype, 11 if packed else 0),
            injection_logits=(
                None
                if variant == "none"
                else tensor(
                    hc,
                    torch.float32 if variant == "fp32" else dtype,
                    256 if packed else 0,
                )
            ),
            hc_count=dict(scalar=hc),
        ),
        outputs=dict(output=tensor(hc * d, dtype)),
        dims=dict(binding),
        native_version=NATIVE_VERSION,
        native_api=NATIVE_API,
        baseline_kind="installed-vllm",
        mutation="none",
        alias="fresh output, no input alias",
        graph=False,
    )
