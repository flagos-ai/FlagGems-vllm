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

"""per_token_group_quant_fp8 baseline（方案 B）。

native：vllm.model_executor.layers.quantization.utils.fp8_utils
            .per_token_group_quant_fp8(x, group_size, use_ue8m0=...)
    这是族C 的 Python wrapper：在 CUDA 上内部转调原生
    torch.ops._C.per_token_group_fp8_quant（见 vllm fp8_utils.py），因此 NCU
    采到的仍是 NV 原生 kernel。返回 (x_q, x_s)（量化张量 + scale），非原地。

输入构造复刻 FlagGems-vllm/benchmark/test_per_token_group_quant_fp8.py 的
_input_fn：
    shape = (num_tokens, d, group_size)      要求 d % group_size == 0
    x     = torch.rand(num_tokens, d)        注意是 rand 不是 randn
    dtype = torch.bfloat16                   benchmark 只跑 bf16
    use_ue8m0 = scale_ue8m0 ∈ {False, True}  benchmark 两个 scale 分支都跑

网格采用 benchmark 的 CORE_SHAPES（来自 core_shapes.yaml 的模型实测档），
与 use_ue8m0 开关做笛卡尔积。use_ue8m0 影响 scale 计算的 kernel 分支，故两个
都采、并编进主键避免 False/True 撞键。
"""

import importlib
import inspect
import itertools

import torch

OP_NAME = "per_token_group_quant_fp8"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_MODULE = "vllm.model_executor.layers.quantization.utils.fp8_utils"
_SYMBOL = "per_token_group_quant_fp8"

# benchmark CORE_SHAPES：(num_tokens, d, group_size)，均满足 d % group_size == 0。
_CORE_SHAPES = [
    (7, 512, 512),
    (7, 4096, 256),
    (83, 512, 64),
    (2048, 4096, 256),
    (2048, 13824, 512),
]

# scale 分支开关：标准 scale 与 ue8m0（DeepGemm E8M0）各采一遍。
_UE8M0 = [False, True]


def _resolve():
    """import fp8_utils 并取出 per_token_group_quant_fp8；取不到返回 None。"""
    try:
        mod = importlib.import_module(_MODULE)
    except ImportError:
        return None
    op = getattr(mod, _SYMBOL, None)
    return op if callable(op) else None


def _supports_ue8m0(op):
    """wrapper 是否接受 use_ue8m0 关键字（照搬 benchmark 的能力探测）。

    老版本 vLLM 无此参数——那种情况下只能采 use_ue8m0=False 分支，采到 True
    绑定时不传该 kwarg（等价于走安装环境的默认）。"""
    try:
        params = inspect.signature(op).parameters
    except (TypeError, ValueError):
        return False
    return "use_ue8m0" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def native():
    """解析 fp8_utils.per_token_group_quant_fp8；解析不到返回 None。"""
    return _resolve()


def grid():
    """CORE_SHAPES × use_ue8m0 开关，展开成绑定列表。"""
    return [
        {"num_tokens": t, "d": d, "group_size": g, "ue8m0": ue8m0}
        for (t, d, g), ue8m0 in itertools.product(_CORE_SHAPES, _UE8M0)
    ]


def build_inputs(binding, dtype, device):
    """构造 (args, kwargs)：x=rand(num_tokens, d)，位置传 group_size，
    use_ue8m0 走 kwargs（安装的 vLLM 不支持该关键字时省略）。"""
    T, d, g = binding["num_tokens"], binding["d"], binding["group_size"]
    x = torch.rand(T, d, dtype=dtype, device=device)
    args = (x, g)
    kwargs = {}
    op = _resolve()
    if op is not None and _supports_ue8m0(op):
        kwargs["use_ue8m0"] = binding["ue8m0"]
    return args, kwargs


def key_shape(binding):
    """主键：shape 三元组 + ue8m0 开关，避免 False/True 两个绑定撞键。"""
    T, d, g = binding["num_tokens"], binding["d"], binding["group_size"]
    return f"t{T}_d{d}_g{g}_ue8m0-{int(binding['ue8m0'])}"


def config(binding, dtype):
    """真实输入输出 shape 描述（写入 JSON 的 config 字段）。

    输出 x_q 与 x 同形、dtype=float8_e4m3fn；scale x_s 为 [num_tokens,
    d//group_size] 的 float32。"""
    T, d, g = binding["num_tokens"], binding["d"], binding["group_size"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [T, d], "dtype": dt, "fill": "rand"},
            "group_size": {"scalar": g},
            "use_ue8m0": {"scalar": binding["ue8m0"]},
        },
        "outputs": {
            "x_q": {"shape": [T, d], "dtype": "torch.float8_e4m3fn"},
            "x_s": {"shape": [T, d // g], "dtype": "torch.float32"},
        },
        "dims": {"num_tokens": T, "d": d, "group_size": g,
                 "num_groups": d // g},
    }
