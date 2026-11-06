# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Pinned Qwen4 hc_combine native and shared, immutable workload matrix."""

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import torch

OP_NAME = "hc_combine"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
IS_INPLACE = False
MUTATED_INPUTS = ()
NATIVE_API = "vllm.models.qwen4_exp.nvidia.ops.hc.hc_combine"
NATIVE_REF = "a17e79abde45715e79a00deb99ad34be49162231"
NATIVE_URL = f"https://github.com/vllm-project/vllm/blob/{NATIVE_REF}/vllm/models/qwen4_exp/nvidia/ops/hc.py"
NATIVE_SHA256 = "8f9b47aa1620d2809f1887a2ce333f5c3bb40f2b05abdaa42d072c19b7176f5a"
NATIVE_UNAVAILABLE_REASON = None


def native():
    global NATIVE_UNAVAILABLE_REASON
    path = os.environ.get("HC_COMBINE_NATIVE_SOURCE")
    if not path:
        NATIVE_UNAVAILABLE_REASON = (
            "HC_COMBINE_NATIVE_SOURCE must point to the pinned native hc.py"
        )
        return None
    source = Path(path)
    if not source.is_file():
        NATIVE_UNAVAILABLE_REASON = f"Pinned source missing: {source}"
        return None
    if hashlib.sha256(source.read_bytes()).hexdigest() != NATIVE_SHA256:
        raise ValueError("hc_combine native source does not match the frozen revision")
    spec = importlib.util.spec_from_file_location("hc_combine_native_pinned", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except (ImportError, AttributeError) as exc:
        sys.modules.pop(spec.name, None)
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    op = getattr(module, "hc_combine", None)
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = "Pinned native hc_combine is not callable"
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return op


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
        native_ref=NATIVE_REF,
        native_api=NATIVE_API,
        baseline_kind="explicit-pinned-vllm-source; not a vendor-C baseline",
        mutation="none",
        alias="fresh output, no input alias",
        graph=False,
    )
