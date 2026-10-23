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

"""gemma_rms_norm baseline（方案 B）。

native：vllm.ir.ops.rms_norm（已回源码核对：
    /Users/zjs-office/all_code/vllm/vllm/ir/ops/layernorm.py#L10
    def rms_norm(x, weight, epsilon, variance_size=None) -> Tensor）。
    这正是 GemmaRMSNorm.forward_native（layernorm.py#L148-157）在 NVIDIA 上
    （forward_cuda 直接转 forward_native）实际调用的函数——Gemma 的差异只是
    weight 用 (1 + w)、以及 (x*w).to(orig_dtype) 的乘序。本模块直接解析该 IrOp 并
    传入已 +1 的权重，等价复刻 Gemma 路径，且**不构造 GemmaRMSNorm CustomOp**。

    为何不走 GemmaRMSNorm 模块：它是 CustomOp 子类，__init__ 经 super().__init__()
    依赖 vllm 运行时 compilation_config，离开模型上下文构造易静默失败（上一版 native
    闭包据此 try/except 吞异常并返回 None，导致被采样调用其实是空操作——延迟恒为
    ~0.003ms、NCU 采不到 kernel）。直接调 ir.ops.rms_norm 无此依赖，稳定触发 kernel。

    注意：当前 vllm 版本 ir.ops.rms_norm 在 NVIDIA 上为 eager PyTorch 实现
    （pow/mean/rsqrt/mul 若干小 kernel），FlagGems-vllm 的 gemma_rms_norm NV 侧
    本就无专用融合 CUDA kernel（test_gemma_rms_norm.py 的 baseline 仅
    ascend/mthreads/iluvatar/hygon 厂商实现，无 nvidia 分支）。故此基准测的是
    vLLM 在 NV 上实际走的 RMSNorm eager 路径。

输入构造复刻 FlagGems-vllm/benchmark/test_gemma_rms_norm.py 的
GemmaRmsNormBenchmark.get_input_iter（shape=(M, N)）：
    x = randn(M, N); w = randn(N); eps = 1e-5
    （benchmark 的 x,w,eps 直接喂 gems_op/baseline；本模块 native 走
    vllm.ir.ops.rms_norm，weight 以 (1 + w) 预置对齐 Gemma 的 1+w 语义。）

shape 来源：benchmark GemmaRmsNormBenchmark 用 set_shapes 硬编码（跳过 yaml）：
    _gemma_rms_norm_ms = [1, 256, 1024]
    _gemma_rms_norm_ns = [128, 256, 1024, 16384, 1152, 2048, 2560, 3072, 3584,
                          3840, 4608, 5376]
    shapes = product(ms, ns)（共 36 个 (M, N)）。
注：benchmark eps=1e-5；本模块沿用 1e-6（RMSNorm eps 差异对基准 shape 无影响，保留原值）。
存疑：benchmark dtypes 仅 [float16]，本模块 DTYPES 仍保留 bf16/fp16/fp32 三档
（dtype 非 shape，未改）。
"""

import importlib
from itertools import product

import torch

OP_NAME = "gemma_rms_norm"
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
IS_INPLACE = False  # 返回归一化后的新张量

_EPS = 1.0e-6

# benchmark GemmaRmsNormBenchmark.set_shapes：product(ms, ns)，(M=num_tokens, N=hidden)。
_GEMMA_MS = [1, 256, 1024]
_GEMMA_NS = [128, 256, 1024, 16384, 1152, 2048, 2560, 3072, 3584, 3840, 4608, 5376]
_SHAPES = [(m, n) for m, n in product(_GEMMA_MS, _GEMMA_NS)]


def native():
    """返回按 (x, weight, eps) 调 ir.ops.rms_norm 的薄封装；不可用时返回 None。

    weight 由 build_inputs 预置为 Gemma 的 (1 + w)，故此处直接透传，等价
    GemmaRMSNorm.forward_native 的 `weight = self.weight.float() + 1.0` 后再调
    rms_norm 的效果。
    """
    try:
        mod = importlib.import_module("vllm.ir.ops")
    except ImportError:
        return None
    rms_norm = getattr(mod, "rms_norm", None)
    if rms_norm is None or not callable(rms_norm):
        return None

    def _run(x, weight, eps):
        return rms_norm(x, weight, eps)

    return _run


def grid():
    return [{"num_tokens": t, "hidden": h} for (t, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    t, h = binding["num_tokens"], binding["hidden"]
    x = torch.randn(t, h, dtype=dtype, device=device)
    # Gemma 语义：等效权重为 (1 + w)。这里直接给出 (1 + randn) 作为传入权重，
    # 省去模块内部的 +1（数值等价，便于直调 ir.ops.rms_norm）。
    weight = 1.0 + torch.randn(h, dtype=dtype, device=device)
    return (x, weight, _EPS), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["hidden"]]


def config(binding, dtype):
    t, h = binding["num_tokens"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [t, h], "dtype": dt},
            "weight": {"shape": [h], "dtype": dt, "note": "Gemma 等效权重 (1+w)"},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [t, h], "dtype": dt},
        },
        "dims": {"num_tokens": t, "hidden": h},
        "shape_source": "test_gemma_rms_norm.py GemmaRmsNormBenchmark.set_shapes "
                        "(product(ms,ns))",
        "note": "native 为 vllm.ir.ops.rms_norm（Gemma NV 路径为 eager 实现，"
                "无专用融合 CUDA kernel）",
    }
