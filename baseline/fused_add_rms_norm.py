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

"""fused_add_rms_norm baseline（方案 B）。

native：vllm._custom_ops.fused_add_rms_norm(input, residual, weight, eps)
    input/residual 原地写回，返回 None。

输入构造复刻 FlagGems-vllm/benchmark/test_fused_add_rms_norm.py 的 _input_fn：
    input    = randn(shape)
    residual = randn(shape)
    weight   = randn(shape[-1])
    eps      = 1e-5
注意 native 签名不含 benchmark 里的 layer_shape 参数（那是 FlagGems 顶层包装层
的入参）；native 直接吃 (input, residual, weight, eps)。

shape 来源：benchmark 的 FusedAddRmsNormBenchmark 继承 GenericBenchmarkExcluse1D，
op_name "fused_add_rms_norm" 不在 core_shapes.yaml，按 MRO 命中类名键
「GenericBenchmarkExcluse1D」（core_shapes.yaml#L234）：
    [64,64] [1024,1024] [4096,4096] [64,512,512] [1024,1024,1024]
（含 2D/3D；native 支持任意前导维，weight 对齐 shape[-1]）。
"""

import importlib

import torch

OP_NAME = "fused_add_rms_norm"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = True

# core_shapes.yaml「GenericBenchmarkExcluse1D」键（test_fused_add_rms_norm.py 的
# FusedAddRmsNormBenchmark 走该键）。
_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]

_EPS = 1.0e-5


def native():
    """解析 vllm._custom_ops.fused_add_rms_norm；解析不到返回 None。"""
    try:
        mod = importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(mod, "fused_add_rms_norm", None)
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = list(binding["shape"])
    N = shape[-1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    residual = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    args = (inp, residual, weight, _EPS)
    return args, {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    """真实输入输出 shape 描述（写入 JSON 的 config 字段）。

    input/residual 原地写回，既是输入也是输出。
    """
    shape = list(binding["shape"])
    N = shape[-1]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": shape, "dtype": dt, "note": "原地写回"},
            "residual": {"shape": shape, "dtype": dt, "note": "原地写回"},
            "weight": {"shape": [N], "dtype": dt},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "input": {"shape": shape, "dtype": dt},
            "residual": {"shape": shape, "dtype": dt},
        },
        "dims": {"N": N, "shape": shape},
    }
