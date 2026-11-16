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

"""Pinned mHC shifted post/pre workloads and exact native resolution.

The workload is vLLM's DSV4.1 production kernel
(``vllm.models.deepseek_v41.nvidia.ops.mhc.mhc_shifted_post_pre``, TileLang /
DeepGEMM).  The shifted boundary only exists in a vLLM tree that ships DSV4.1,
so ``native()`` probes that module and returns ``None`` with a reason when the
installed vLLM does not provide it.  Do not substitute the FlagGems kernel
here: a self-baseline makes every SpeedUp 1.0 and hides real regressions.

Timing is eager (no CUDA graph): the operator dispatches into several sub-ops
with host-side autotune caches, and a cold capture invalidates the stream.
"""

import importlib

import torch

OP_NAME = "mhc_shifted_post_pre"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ()
NATIVE_UNAVAILABLE_REASON = None

_HC = 4
_MIX = _HC * (_HC + 2)
_SEED = 0

# DSV4/DSV4.1 boundary shapes.  Keep in sync with benchmark/core_shapes.yaml
# and benchmark/test_mhc_shifted_post_pre.py.
SHAPES = [
    (1, 1280),
    (16, 1280),
    (512, 1280),
    (512, 7168),
    (4096, 7168),
    (128, 5120),
    (2048, 5120),
]

# Release-specific locations of the production kernel.  Add a new entry when
# vLLM moves the module; never fall back to a candidate implementation.
_NATIVE_MODULES = (("vllm.models.deepseek_v41.nvidia.ops.mhc", "mhc_shifted_post_pre"),)


def native():
    """Resolve the vLLM production kernel, or return None with a reason."""
    global NATIVE_UNAVAILABLE_REASON

    errors = []
    for module_name, attr in _NATIVE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - resolution must not abort collection
            errors.append(f"{module_name}.{attr}: {type(exc).__name__}: {exc}")
            continue
        op = getattr(module, attr, None)
        if callable(op):
            NATIVE_UNAVAILABLE_REASON = None
            return op
        errors.append(f"{module_name}.{attr}: not callable")

    NATIVE_UNAVAILABLE_REASON = "; ".join(errors)
    return None


def grid():
    """Seven pinned DSV4/DSV4.1 (tokens, hidden) boundary workloads."""
    return [{"shape": list(shape)} for shape in SHAPES]


def build_inputs(binding, dtype, device):
    tokens, hidden = binding["shape"]
    if dtype is not torch.bfloat16:
        raise NotImplementedError("mhc_shifted_post_pre only supports bfloat16")

    # Local generator: build_inputs must be repeatable without perturbing the
    # process-wide RNG (the collector calls it in the main and NCU processes).
    gen = torch.Generator(device=device).manual_seed(_SEED)
    x = torch.randn(tokens, hidden, dtype=dtype, device=device, generator=gen)
    residual = torch.randn(
        tokens, _HC, hidden, dtype=dtype, device=device, generator=gen
    )
    post_layer_mix = (
        torch.rand(tokens, _HC, 1, dtype=torch.float32, device=device, generator=gen)
        * 0.3
    )
    comb_res_mix = (
        torch.randn(tokens, _HC, _HC, dtype=torch.float32, device=device, generator=gen)
        * 0.2
    )
    fn = (
        torch.randn(
            _MIX, _HC * hidden, dtype=torch.float32, device=device, generator=gen
        )
        * 0.02
    )
    hc_scale = torch.tensor([0.1, 0.3, 0.2], dtype=torch.float32, device=device)
    hc_base = torch.randn(_MIX, dtype=torch.float32, device=device, generator=gen) * 0.1
    pre_mix = (
        torch.rand(tokens, _HC, dtype=torch.float32, device=device, generator=gen) + 0.5
    )
    norm_weight = (
        torch.randn(hidden, dtype=dtype, device=device, generator=gen) * 0.3 + 1
    )
    args = (
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        1e-6,  # rms_eps
        1e-6,  # hc_pre_eps
        1e-6,  # hc_sinkhorn_eps
        1.0,  # hc_post_mult_value
        5,  # sinkhorn_repeat
    )
    kwargs = {
        "pre_mix": pre_mix,
        "norm_weight": norm_weight,
        "norm_eps": 1e-6,
        "capture_aux": True,
    }
    return args, kwargs


def key_shape(binding):
    """Primary key = the first tensor's shape (x is (tokens, hidden))."""
    return list(binding["shape"])


def config(binding, dtype):
    tokens, hidden = binding["shape"]
    dt = str(dtype)
    f32 = str(torch.float32)
    return {
        "inputs": {
            "x": {"shape": [tokens, hidden], "dtype": dt},
            "residual": {"shape": [tokens, _HC, hidden], "dtype": dt},
            "post_layer_mix": {"shape": [tokens, _HC, 1], "dtype": f32},
            "comb_res_mix": {"shape": [tokens, _HC, _HC], "dtype": f32},
            "fn": {"shape": [_MIX, _HC * hidden], "dtype": f32},
            "hc_scale": {"shape": [3], "dtype": f32},
            "hc_base": {"shape": [_MIX], "dtype": f32},
            "pre_mix": {"shape": [tokens, _HC], "dtype": f32},
            "norm_weight": {"shape": [hidden], "dtype": dt},
            "rms_eps": {"scalar": 1e-6},
            "hc_pre_eps": {"scalar": 1e-6},
            "hc_sinkhorn_eps": {"scalar": 1e-6},
            "hc_post_mult_value": {"scalar": 1.0},
            "sinkhorn_repeat": {"scalar": 5},
            "norm_eps": {"scalar": 1e-6},
            "capture_aux": {"scalar": True},
        },
        "outputs": {
            "residual_cur": {
                "shape": [tokens, _HC, hidden],
                "dtype": dt,
                "aliases_inputs": False,
            },
            "post_mix": {
                "shape": [tokens, _HC, 1],
                "dtype": f32,
                "aliases_inputs": False,
            },
            "comb_mix": {
                "shape": [tokens, _HC, _HC],
                "dtype": f32,
                "aliases_inputs": False,
            },
            "layer_input": {
                "shape": [tokens, hidden],
                "dtype": dt,
                "aliases_inputs": False,
            },
            "next_pre": {
                "shape": [tokens, _HC],
                "dtype": f32,
                "aliases_inputs": False,
            },
            "aux": {
                "shape": [tokens, hidden],
                "dtype": dt,
                "aliases_inputs": False,
            },
        },
        "dims": {
            "tokens": tokens,
            "hidden": hidden,
            "hc_mult": _HC,
            "sinkhorn_repeat": 5,
        },
        "input_seed": _SEED,
    }
