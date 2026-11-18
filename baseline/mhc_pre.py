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

"""Pinned mHC pre workloads and exact native resolution."""

import importlib

import torch

OP_NAME = "mhc_pre"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ()
NATIVE_UNAVAILABLE_REASON = None

HC_MULT = 4
MHC_PRE_SHAPES = [
    (512, 1280),
    (512, 2560),
    (512, 4096),
    (1024, 1280),
    (1024, 2560),
    (1024, 4096),
    (2048, 1280),
    (2048, 2560),
    (2048, 4096),
    (8192, 1280),
    (8192, 2560),
    (8192, 4096),
]


def native():
    global NATIVE_UNAVAILABLE_REASON
    errors = []
    for module_name, attr in (
        ("vllm.model_executor.kernels.mhc", "mhc_pre_tilelang"),
        ("vllm.model_executor.layers.mhc", "mhc_pre"),
    ):
        try:
            op = getattr(importlib.import_module(module_name), attr)
        except Exception as exc:
            errors.append(f"{module_name}.{attr}: {type(exc).__name__}: {exc}")
            continue
        NATIVE_UNAVAILABLE_REASON = None
        return op
    NATIVE_UNAVAILABLE_REASON = "; ".join(errors)
    return None


def grid():
    """Twelve pinned (tokens, hidden) workloads at the model's Sinkhorn count."""
    return [{"shape": list(shape)} for shape in MHC_PRE_SHAPES]


def build_inputs(binding, dtype, device):
    tokens, hidden = binding["shape"]
    # sinkhorn_repeat: the model's Sinkhorn iteration count.
    repeat = 10
    hc_mult3 = HC_MULT * 2 + HC_MULT * HC_MULT
    torch.manual_seed(42)
    per_head = 1 + torch.arange(HC_MULT, device=device).mul(0.01).view(1, -1, 1)
    residual = (
        torch.randn((tokens, HC_MULT, hidden), dtype=torch.float, device=device)
        .mul(per_head)
        .bfloat16()
    )
    fn = (
        torch.randn((hc_mult3, HC_MULT, hidden), dtype=torch.float, device=device)
        * 1e-4
        * per_head
    ).flatten(1, 2)
    hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1
    return (
        residual,
        fn,
        hc_scale,
        hc_base,
        # rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value
        1e-6,
        1e-6,
        1e-6,
        1.0,
        repeat,
    ), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    tokens, hidden = binding["shape"]
    hc_mult3 = HC_MULT * 2 + HC_MULT * HC_MULT
    return {
        "inputs": {
            "residual": {
                "shape": [tokens, HC_MULT, hidden],
                "dtype": str(dtype),
            },
            "fn": {"shape": [hc_mult3, HC_MULT * hidden], "dtype": str(torch.float32)},
            "hc_scale": {"shape": [3], "dtype": str(torch.float32)},
            "hc_base": {"shape": [hc_mult3], "dtype": str(torch.float32)},
        },
        "outputs": {
            "post_mix": {
                "shape": [tokens, HC_MULT, 1],
                "dtype": str(torch.float32),
                "aliases_inputs": False,
            },
            "comb_mix": {
                "shape": [tokens, HC_MULT, HC_MULT],
                "dtype": str(torch.float32),
                "aliases_inputs": False,
            },
            "layer_input": {
                "shape": [tokens, hidden],
                "dtype": str(dtype),
                "aliases_inputs": False,
            },
        },
        "dims": {
            "tokens": tokens,
            "hidden": hidden,
            "hc_mult": HC_MULT,
            "sinkhorn_repeat": 10,
        },
    }
