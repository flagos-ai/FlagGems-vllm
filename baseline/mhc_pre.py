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

"""mhc_pre baseline（方案 A：直调官方 torch.ops.vllm.mhc_pre_tilelang）。

native：官方 tilelang 自定义算子 torch.ops.vllm.mhc_pre_tilelang
    （vllm/model_executor/kernels/mhc/tilelang.py#L90，经 direct_register_custom_op
    注册为 op_name="mhc_pre_tilelang"）。签名：
    mhc_pre_tilelang(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
    hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=1,
    norm_weight=None, norm_eps=1e-6) -> (post_mix, comb_mix, layer_input)。
    这正是 MHCPreOp.forward_cuda（mhc.py#L47）在 CUDA 平台实际 dispatch 的目标，
    因此直调它可绕开 CustomOp 封装、精确对齐 vllm 的 CUDA tilelang kernel。
    native() 先 import kernels.mhc（触发算子注册），再返回 torch.ops.vllm.mhc_pre_tilelang
    （callable），op(*args) 即触发 tilelang CUDA kernel。tilelang 不可用
    （layers.mhc.HAS_TILELANG_MHC=False，或算子未注册）时返回 None 优雅跳过。

输入构造复刻 FlagGems-vllm/benchmark/test_mhc.py 的 MHCPreBenchmark.get_input_iter：
    hc_mult = 4, sinkhorn_repeat = 10, hc_mult3 = hc_mult*2 + hc_mult*hc_mult = 24
    residual = randn(N, hc_mult, hidden_size).bfloat16()   # 归约主张量
    fn       = randn(hc_mult3, hc_mult, hidden_size).flatten(1,2)  # -> (24, hc_mult*hidden) float32
    hc_scale = randn(3) float32
    hc_base  = randn(hc_mult3) float32
    标量：rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
          hc_post_mult_value=1.0, sinkhorn_repeat=10
    （n_splits/norm_weight/norm_eps 走默认值，benchmark 未传。）
    注意 fn/hc_scale/hc_base 在源码里保持 float32，仅 residual 用 dtype(bf16)。

shape：直接采用 benchmark 的 (N, hidden_size) 12 档网格。
"""

import importlib

import torch

OP_NAME = "mhc_pre"
DTYPES = [torch.bfloat16]  # benchmark 仅 bf16；residual 用之，其余张量恒 float32
IS_INPLACE = False  # 返回 (post_mix, comb_mix, layer_input) 三元组

_HC_MULT = 4
_SINKHORN_REPEAT = 10
_RMS_EPS = 1.0e-6
_HC_PRE_EPS = 1.0e-6
_HC_SINKHORN_EPS = 1.0e-6
_HC_POST_MULT_VALUE = 1.0

# benchmark 的 (N, hidden_size) 网格。
_SHAPES = [
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
    """返回 torch.ops.vllm.mhc_pre_tilelang（callable）；不可用时返回 None。

    先 import kernels.mhc（direct_register_custom_op 在此把 mhc_pre_tilelang 注册到
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
    op = getattr(torch.ops.vllm, "mhc_pre_tilelang", None)
    return op if callable(op) else None


def grid():
    return [{"N": n, "hidden_size": h} for (n, h) in _SHAPES]


def build_inputs(binding, dtype, device):
    n, hidden = binding["N"], binding["hidden_size"]
    hc_mult = _HC_MULT
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult  # = 24

    residual = torch.randn(n, hc_mult, hidden, dtype=dtype, device=device)
    # fn 源码 shape (hc_mult3, hc_mult, hidden) 再 flatten(1,2) -> (hc_mult3, hc_mult*hidden)
    fn = torch.randn(hc_mult3, hc_mult * hidden, dtype=torch.float32, device=device)
    hc_scale = torch.randn(3, dtype=torch.float32, device=device)
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device=device)

    args = (
        residual,
        fn,
        hc_scale,
        hc_base,
        _RMS_EPS,
        _HC_PRE_EPS,
        _HC_SINKHORN_EPS,
        _HC_POST_MULT_VALUE,
        _SINKHORN_REPEAT,
    )
    return args, {}


def key_shape(binding):
    return [binding["N"], _HC_MULT, binding["hidden_size"]]


def config(binding, dtype):
    n, hidden = binding["N"], binding["hidden_size"]
    hc_mult = _HC_MULT
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    dt = str(dtype)
    f32 = str(torch.float32)
    return {
        "inputs": {
            "residual": {"shape": [n, hc_mult, hidden], "dtype": dt},
            "fn": {"shape": [hc_mult3, hc_mult * hidden], "dtype": f32},
            "hc_scale": {"shape": [3], "dtype": f32},
            "hc_base": {"shape": [hc_mult3], "dtype": f32},
            "rms_eps": {"scalar": _RMS_EPS},
            "hc_pre_eps": {"scalar": _HC_PRE_EPS},
            "hc_sinkhorn_eps": {"scalar": _HC_SINKHORN_EPS},
            "hc_post_mult_value": {"scalar": _HC_POST_MULT_VALUE},
            "sinkhorn_repeat": {"scalar": _SINKHORN_REPEAT},
        },
        "outputs": {
            "post_mix": {"note": "sinkhorn 组合权重"},
            "comb_mix": {"note": "组合权重"},
            "layer_input": {"shape": [n, hidden], "dtype": dt},
        },
        "dims": {"N": n, "hidden_size": hidden, "hc_mult": hc_mult},
    }
