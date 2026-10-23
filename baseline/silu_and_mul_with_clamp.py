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

"""silu_and_mul_with_clamp baseline（方案 B）。

native：torch.ops._C.silu_and_mul_with_clamp(result, input, limit, alpha=1.0,
        beta=0.0) -> ()。result 原地写回。
    绑定签名见 csrc/libtorch_stable/torch_bindings.cpp#L592；
    上层 CustomOp 入口 SiluAndMulWithClamp（vllm/model_executor/layers/activation.py
    #L204，forward_cuda 里 d=x.shape[-1]//2, out=empty(...,d)）。
    需先 import vllm._custom_ops 触发 torch.ops._C 命名空间注册。

注意 native 是“单张量 concat”形态：input 形状 (..., 2d)，前半是 gate、后半是 up，
输出 result 形状 (..., d)。FlagGems-vllm/benchmark/test_silu_and_mul_with_clamp.py
用 binary_input_fn 产两张 (..., d) 张量喂给参考实现，等价于把它们 cat 成 (..., 2d)
喂给 native。这里直接按 native 形态构造 input=randn(..., 2d)。limit=7.0（benchmark
里的常量）。

shape：benchmark 用 GenericBenchmark（无 yaml 专属键，回落到 core_shapes.yaml 的
Benchmark 默认）。取其中 2D/3D 项，最后一维为每半的宽度 d。
"""

import importlib

import torch

OP_NAME = "silu_and_mul_with_clamp"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True  # result 原地写回

_LIMIT = 7.0
_ALPHA = 1.0
_BETA = 0.0

# core_shapes.yaml「Benchmark」默认里的 2D/3D 项；每项最后一维为半宽 d。
_SHAPES = [
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]


def native():
    """解析 torch.ops._C.silu_and_mul_with_clamp；解析不到返回 None。

    先 import vllm._custom_ops 以注册 torch.ops._C 命名空间。
    """
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(getattr(torch.ops, "_C", None), "silu_and_mul_with_clamp", None)
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = list(binding["shape"])
    d = shape[-1]
    in_shape = shape[:-1] + [2 * d]
    out_shape = shape[:-1] + [d]
    inp = torch.randn(in_shape, dtype=dtype, device=device)
    result = torch.empty(out_shape, dtype=dtype, device=device)
    return (result, inp, _LIMIT, _ALPHA, _BETA), {}


def key_shape(binding):
    # key = 第一个输入张量 result 的形状（宽度 d）。benchmark 侧匹配按
    # _first_tensor_shape 取第一个张量 shape 直查，故此处不再 ×2，保持两端一致。
    return list(binding["shape"])


def config(binding, dtype):
    shape = list(binding["shape"])
    d = shape[-1]
    in_shape = shape[:-1] + [2 * d]
    out_shape = shape[:-1] + [d]
    dt = str(dtype)
    return {
        "inputs": {
            "result": {"shape": out_shape, "dtype": dt, "note": "原地写回"},
            "input": {"shape": in_shape, "dtype": dt,
                      "note": "前半 gate / 后半 up"},
            "limit": {"scalar": _LIMIT},
            "alpha": {"scalar": _ALPHA},
            "beta": {"scalar": _BETA},
        },
        "outputs": {
            "result": {"shape": out_shape, "dtype": dt},
        },
        "dims": {"d": d, "in_last": 2 * d},
    }
