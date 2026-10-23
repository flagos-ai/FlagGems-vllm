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

"""gelu_and_mul baseline（方案 B）。

native：torch.ops._C.gelu_and_mul(out, input) -> ()。out 原地写回。
    上层入口 GeluAndMul（vllm/model_executor/layers/activation.py#L419，approximate="none"
    时 self.op = torch.ops._C.gelu_and_mul；forward_cuda 里 d = x.shape[-1] // 2,
    out = empty(..., d)，调 self.op(out, x)）。native 形态：input 形状 (..., 2d)，
    前半走 GELU、后半相乘，输出 out 形状 (..., d)。需先 import vllm._custom_ops 注册
    torch.ops._C。

输入构造复刻 FlagGems-vllm/benchmark/test_gelu_and_mul.py 的 test_gelu_and_mul：
    用 GenericBenchmark + utils.binary_input_fn 产两张 (..., d) 张量喂给参考实现
    torch.mul(gelu(x), y)，等价于把它们 cat 成 (..., 2d) 喂给 native。这里直接按
    native 形态构造 input = randn(..., 2d)。

shape：benchmark 用 GenericBenchmark（无 yaml 专属键），回落 core_shapes.yaml 的
「Benchmark」默认。取其中 2D/3D 项（1D 的 1073741824 对该算子无意义，剔除），
每项最后一维为每半的宽度 d。
"""

import importlib

import torch

OP_NAME = "gelu_and_mul"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True  # out 原地写回

# core_shapes.yaml「Benchmark」默认里的 2D/3D 项；每项最后一维为半宽 d。
_SHAPES = [
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]


def native():
    """解析 torch.ops._C.gelu_and_mul；解析不到返回 None。

    先 import vllm._custom_ops 以注册 torch.ops._C 命名空间。
    """
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(getattr(torch.ops, "_C", None), "gelu_and_mul", None)
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = list(binding["shape"])
    d = shape[-1]
    in_shape = shape[:-1] + [2 * d]
    out_shape = shape[:-1] + [d]
    inp = torch.randn(in_shape, dtype=dtype, device=device)
    out = torch.empty(out_shape, dtype=dtype, device=device)
    return (out, inp), {}


def key_shape(binding):
    shape = list(binding["shape"])
    d = shape[-1]
    return shape[:-1] + [2 * d]


def config(binding, dtype):
    shape = list(binding["shape"])
    d = shape[-1]
    in_shape = shape[:-1] + [2 * d]
    out_shape = shape[:-1] + [d]
    dt = str(dtype)
    return {
        "inputs": {
            "out": {"shape": out_shape, "dtype": dt, "note": "原地写回"},
            "input": {"shape": in_shape, "dtype": dt,
                      "note": "前半 gate(GELU) / 后半 up"},
        },
        "outputs": {
            "out": {"shape": out_shape, "dtype": dt},
        },
        "dims": {"d": d, "in_last": 2 * d},
    }
