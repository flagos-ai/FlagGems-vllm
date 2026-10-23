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

"""mhc_post baseline（方案 A：直调官方 torch.ops.vllm.mhc_post_tilelang）。

native：官方 tilelang 自定义算子 torch.ops.vllm.mhc_post_tilelang
    （vllm/model_executor/kernels/mhc/tilelang.py#L410，经 direct_register_custom_op
    注册为 op_name="mhc_post_tilelang"）。签名：
    mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix) -> out。
    这正是 MHCPostOp.forward_cuda（mhc.py#L206）在 CUDA 平台实际 dispatch 的目标，
    因此直调它可绕开 CustomOp 封装、精确对齐 vllm 的 CUDA tilelang kernel。
    native() 先 import kernels.mhc（触发算子注册），再返回
    torch.ops.vllm.mhc_post_tilelang（callable）。tilelang 不可用
    （layers.mhc.HAS_TILELANG_MHC=False，或算子未注册）时返回 None 优雅跳过。

输入构造复刻 FlagGems-vllm/benchmark/test_mhc.py 的 MHCPostBenchmark.get_input_iter：
    hc_mult = 4
    x              = randn(N, H)            bfloat16
    residual       = randn(N, hc_mult, H)   bfloat16
    post_layer_mix = randn(N, hc_mult, 1)   float32
    comb_res_mix   = randn(N, hc_mult, hc_mult) float32
    注意 mix 张量恒 float32，仅 x/residual 用 dtype(bf16)。

shape：直接采用 benchmark 的 (N, H) 三档网格。
"""

import importlib

import torch

OP_NAME = "mhc_post"
DTYPES = [torch.bfloat16]  # benchmark 仅 bf16；x/residual 用之，mix 恒 float32
IS_INPLACE = False  # 返回 out 张量

_HC_MULT = 4

# benchmark 的 (N, H) 网格。
_SHAPES = [
    (4096, 1280),
    (4096, 2560),
    (4096, 7168),
]


def native():
    """返回 torch.ops.vllm.mhc_post_tilelang（callable）；不可用时返回 None。

    先 import kernels.mhc（direct_register_custom_op 在此把 mhc_post_tilelang 注册到
    torch.ops.vllm），并校验 layers.mhc.HAS_TILELANG_MHC——为假说明当前平台无
    tilelang CUDA kernel，优雅跳过。
    """
    try:
        importlib.import_module("vllm.model_executor.kernels.mhc")
    except ImportError:
        return None
    try:
        layers = importlib.import_module("vllm.model_executor.layers.mhc")
    except ImportError:
        layers = None
    if layers is not None and not getattr(layers, "HAS_TILELANG_MHC", False):
        return None  # tilelang CUDA kernel 不可用，优雅跳过
    op = getattr(torch.ops.vllm, "mhc_post_tilelang", None)
    return op if callable(op) else None


def grid():
    return [{"N": n, "H": h} for (n, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    n, h = binding["N"], binding["H"]
    hc_mult = _HC_MULT

    x = torch.randn(n, h, dtype=dtype, device=device)
    residual = torch.randn(n, hc_mult, h, dtype=dtype, device=device)
    post_layer_mix = torch.randn(n, hc_mult, 1, dtype=torch.float32, device=device)
    comb_res_mix = torch.randn(
        n, hc_mult, hc_mult, dtype=torch.float32, device=device
    )

    args = (x, residual, post_layer_mix, comb_res_mix)
    return args, {}


def key_shape(binding):
    return [binding["N"], _HC_MULT, binding["H"]]


def config(binding, dtype):
    n, h = binding["N"], binding["H"]
    hc_mult = _HC_MULT
    dt = str(dtype)
    f32 = str(torch.float32)
    return {
        "inputs": {
            "x": {"shape": [n, h], "dtype": dt},
            "residual": {"shape": [n, hc_mult, h], "dtype": dt},
            "post_layer_mix": {"shape": [n, hc_mult, 1], "dtype": f32},
            "comb_res_mix": {"shape": [n, hc_mult, hc_mult], "dtype": f32},
        },
        "outputs": {
            "out": {"shape": [n, hc_mult, h], "dtype": dt},
        },
        "dims": {"N": n, "H": h, "hc_mult": hc_mult},
    }
