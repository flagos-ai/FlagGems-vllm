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

"""Shared shape contract; CUDA baseline resolves only the upstream callable."""
import importlib
import warnings

import torch

OP_NAME = "hc_collapse"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ()
# vLLM tests/kernels/mhc/test_mhc_kernels.py
# test_hc_collapse_preserves_weighted_residual_sum.
SHAPES = [
    [0, 4, 5120],
    [1, 4, 5120],
    [33, 4, 5137],
    [128, 4, 5120],
    [1024, 4, 5120],
    [2, 1, 513],
    [7, 8, 5137],
    [65537, 4, 65],
]


def native():
    try:
        return getattr(
            importlib.import_module("vllm.model_executor.kernels.mhc.triton"),
            "hc_collapse_triton",
        )
    except (ImportError, AttributeError) as exc:
        warnings.warn(f"CUDA hc_collapse baseline unavailable: {exc}")
        return None


VLLM_ASCEND_REPLAY_SOURCE = (
    "https://github.com/vllm-project/vllm-ascend/blob/"
    "70a336fd1697e710f476747e40465a1a48baf079/"
    "vllm_ascend/models/deepseek_v41/model.py#L857-L859"
)


def vllm_ascend_replay(x, pre_mix):
    return (pre_mix.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)


def vendor_native(vendor, *, required=False):
    if vendor == "nvidia":
        return native()
    if vendor != "ascend":
        return None
    try:
        mod = importlib.import_module("vllm_ascend.models.deepseek_v41.model")
        methods = [
            v.__dict__["hc_collapse"].__func__
            for v in vars(mod).values()
            if isinstance(v, type)
            and v.__module__ == mod.__name__
            and isinstance(vars(v).get("hc_collapse"), staticmethod)
        ]
        methods = list(dict.fromkeys(methods))
        if len(methods) != 1:
            raise RuntimeError(f"expected one vendor hc_collapse, found {len(methods)}")
        return methods[0]
    except Exception as exc:
        message = (
            f"vLLM-Ascend hc_collapse baseline unavailable: {exc}. "
            "Use an environment where vllm_ascend.models.deepseek_v41.model "
            "imports and exposes the hc_collapse static method."
        )
        if required:
            raise RuntimeError(message) from exc
        warnings.warn(message)
        return None


def all_cases():
    return [{"shape": list(s), "strided": b} for s in SHAPES for b in (False, True)]


def grid():
    return [{"shape": list(s), "strided": False} for s in SHAPES if s[0] > 0]


def build_inputs(binding, dtype, device):
    t, c, h = binding["shape"]
    step = 2 if binding.get("strided", False) else 1
    x = torch.randn(t, c, h * step, dtype=dtype, device=device)[..., ::step]
    p = torch.rand(t, c, dtype=torch.float32, device=device)
    if step == 2:
        p = p.t().contiguous().t()
    if t and c:
        p[0].zero_()
        p[0, -1] = 1
    if t > 1 and c == 4:
        p[1].fill_(1)
        x[1, 1] = -x[1, 0]
        x[1, 3] = -x[1, 2]
    return (x, p), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    t, c, h = binding["shape"]
    step = 2 if binding.get("strided", False) else 1
    return {
        "inputs": {
            "x": {
                "shape": [t, c, h],
                "dtype": str(dtype),
                "stride": [c * h * step, h * step, step],
            },
            "pre_mix": {
                "shape": [t, c],
                "dtype": "torch.float32",
                "stride": [1, t] if step == 2 else [c, 1],
            },
        },
        "outputs": {"out": {"shape": [t, h], "dtype": str(dtype)}},
        "dims": {"shape": [t, c, h], "strided": step == 2},
    }
