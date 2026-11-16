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

"""Pinned mHC post workloads and exact native resolution."""

import importlib

import torch

OP_NAME = "mhc_post"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ()
NATIVE_UNAVAILABLE_REASON = None

HC_MULT = 4
MHC_POST_SHAPES = [(4096, 1280), (4096, 2560), (4096, 7168)]


def native():
    global NATIVE_UNAVAILABLE_REASON
    errors = []
    for module_name, attr in (
        ("vllm.model_executor.kernels.mhc", "mhc_post_tilelang"),
        ("vllm.model_executor.layers.mhc", "mhc_post"),
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
    """Three pinned (tokens, hidden) workloads."""
    return [{"shape": list(shape)} for shape in MHC_POST_SHAPES]


def build_inputs(binding, dtype, device):
    tokens, hidden = binding["shape"]
    # The mix tensors are float32 by contract; only x/residual take the dtype.
    x = torch.randn((tokens, hidden), dtype=dtype, device=device)
    residual = torch.randn((tokens, HC_MULT, hidden), dtype=dtype, device=device)
    post_layer_mix = torch.randn(
        (tokens, HC_MULT, 1), dtype=torch.float32, device=device
    )
    comb_res_mix = torch.randn(
        (tokens, HC_MULT, HC_MULT), dtype=torch.float32, device=device
    )
    return (x, residual, post_layer_mix, comb_res_mix), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    tokens, hidden = binding["shape"]
    return {
        "inputs": {
            "x": {"shape": [tokens, hidden], "dtype": str(dtype)},
            "residual": {"shape": [tokens, HC_MULT, hidden], "dtype": str(dtype)},
            "post_layer_mix": {
                "shape": [tokens, HC_MULT, 1],
                "dtype": str(torch.float32),
            },
            "comb_res_mix": {
                "shape": [tokens, HC_MULT, HC_MULT],
                "dtype": str(torch.float32),
            },
        },
        "outputs": {
            "out": {
                "shape": [tokens, hidden],
                "dtype": str(dtype),
                "aliases_inputs": False,
            }
        },
        "dims": {"tokens": tokens, "hidden": hidden, "hc_mult": HC_MULT},
    }
