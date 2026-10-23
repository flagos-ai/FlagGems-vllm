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

"""fused_recurrent_gated_delta_rule_fwd baseline（方案 B）——FLA 门控 delta rule 循环前向。

native：vllm.third_party.flash_linear_attention.ops.fused_recurrent_gated_delta_rule
    （由 ops/__init__ 导出；定义见 .../ops/fused_recurrent.py#L523，已回源核对）。签名：
        fused_recurrent_gated_delta_rule(q, k, v, g, beta, scale, initial_state=None,
            inplace_final_state=True, cu_seqlens=None, ssm_state_indices=None,
            num_accepted_tokens=None, use_qk_l2norm_in_kernel=False)
    返回 (core_attn_out, final_state)。ops_interface_1.csv 的 example 用了简化签名
    （output_final_state 实为 inplace_final_state），这里以源码真实签名为准、按位置传参。

输入构造复刻 FlagGems-vllm/benchmark/test_FLA/test_fused_recurrent_gated_delta_rule_perf.py
    FusedRecurrentGatedDeltaRuleBenchmark._build_inputs（qkv_contiguous=False）：
    B=1, H=16, HV=32, K=128, V=128, tp_size=4：
      key_dim=H*K=2048, value_dim=HV*V=4096, mixed_qkv_dim=(2*key_dim+value_dim)//tp=2048
      mixed_qkv = randn(B*T, 2048) → split[512,512,1024] → view:
        q = (1, T, 4, 128), k = (1, T, 4, 128), v = (1, T, 8, 128)   # HV_local=8
      g    = logsigmoid(randn(1, T, 8))
      beta = rand(1, T, 8).sigmoid()
      cu_seqlens = arange(T+1) long
      initial_state = zeros(1024, 8, 128, 128)
      ssm_state_indices = zeros(T) long
      scale = 0.08838834764831845
      调用 (q,k,v,g,beta,scale,initial_state,True,cu_seqlens,ssm_state_indices,None,True)

shape 来源：benchmark 的 self.shapes = core_shapes.yaml「FusedRecurrentGatedDeltaRuleBenchmark」
基础档 {1,64,184,256,512} 与 set_more_shapes() 的 T 列表取并集（T 即序列长度）。
"""

import importlib

import torch

OP_NAME = "fused_recurrent_gated_delta_rule_fwd"
DTYPES = [torch.bfloat16]  # benchmark DEFAULT_DTYPES
IS_INPLACE = False  # 返回新张量；final_state inplace 到传入的 initial_state

# 固定维度（benchmark 常量）
_B = 1
_H = 16
_HV = 32
_K = 128
_V = 128
_TP = 4
_SCALE = 0.08838834764831845

# benchmark self.shapes：core_shapes 基础档 ∪ set_more_shapes()（去重升序）。
_BASE_T = [1, 64, 184, 256, 512]
_MORE_T = [
    1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 72, 80, 88, 96, 104, 112, 120, 128,
    136, 144, 152, 160, 168, 176, 192, 200, 208, 216, 224, 232, 240, 248,
    272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496,
]
_SEQ_LENS = sorted(set(_BASE_T) | set(_MORE_T))

_KEY_DIM = _H * _K          # 2048
_VALUE_DIM = _HV * _V       # 4096
_MIXED_QKV_DIM = (2 * _KEY_DIM + _VALUE_DIM) // _TP  # 2048
_HV_LOCAL = (_VALUE_DIM // _TP) // _V               # 8
_H_LOCAL = (_KEY_DIM // _TP) // _K                  # 4


def native():
    """解析 FLA fused_recurrent_gated_delta_rule；解析不到返回 None。"""
    for module in (
        "vllm.third_party.flash_linear_attention.ops",
        "vllm.model_executor.layers.fla.ops",
    ):
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        op = getattr(mod, "fused_recurrent_gated_delta_rule", None)
        if callable(op):
            return op
    return None


def grid():
    return [{"T": t} for t in _SEQ_LENS]


def build_inputs(binding, dtype, device):
    T = binding["T"]
    total_tokens = _B * T
    mixed_qkv = torch.randn(total_tokens, _MIXED_QKV_DIM, dtype=dtype,
                            device=device)
    # split[key_dim//tp, key_dim//tp, value_dim//tp] along dim=-1（切片等价
    # torch.split，兼容离线冒烟 stub）
    kq = _KEY_DIM // _TP
    vq = _VALUE_DIM // _TP
    q = mixed_qkv[:, :kq]
    k = mixed_qkv[:, kq:2 * kq]
    v = mixed_qkv[:, 2 * kq:2 * kq + vq]
    q = q.view(1, q.shape[0], -1, _K)
    k = k.view(1, k.shape[0], -1, _K)
    v = v.view(1, v.shape[0], -1, _V)

    # torch.logsigmoid 等价 torch.nn.functional.logsigmoid（真 torch 两者皆有）
    g = torch.logsigmoid(
        torch.randn(_B, T, _HV_LOCAL, dtype=dtype, device=device))
    beta = torch.rand(_B, T, _HV_LOCAL, dtype=dtype, device=device).sigmoid()
    cu_seqlens = torch.arange(T + 1, device=device, dtype=torch.int64)
    initial_state = torch.zeros(1024, _HV_LOCAL, _K, _V, dtype=dtype,
                                device=device)
    ssm_state_indices = torch.zeros(T, device=device, dtype=torch.int64)

    args = (q, k, v, g, beta, _SCALE, initial_state, True, cu_seqlens,
            ssm_state_indices, None, True)
    return args, {}


def key_shape(binding):
    T = binding["T"]
    return f"B{_B}_T{T}_H{_H_LOCAL}_HV{_HV_LOCAL}_K{_K}_V{_V}"


def config(binding, dtype):
    T = binding["T"]
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [1, T, _H_LOCAL, _K], "dtype": dt},
            "k": {"shape": [1, T, _H_LOCAL, _K], "dtype": dt},
            "v": {"shape": [1, T, _HV_LOCAL, _V], "dtype": dt},
            "g": {"shape": [_B, T, _HV_LOCAL], "dtype": dt},
            "beta": {"shape": [_B, T, _HV_LOCAL], "dtype": dt},
            "scale": {"scalar": _SCALE},
            "initial_state": {"shape": [1024, _HV_LOCAL, _K, _V], "dtype": dt},
            "cu_seqlens": {"shape": [T + 1], "dtype": "torch.int64"},
            "ssm_state_indices": {"shape": [T], "dtype": "torch.int64"},
        },
        "outputs": {
            "core_attn_out": {"shape": [1, T, _HV_LOCAL, _V], "dtype": dt},
            "final_state": {"note": "inplace_final_state=True，写回 initial_state"},
        },
        "dims": {
            "B": _B, "T": T, "H_local": _H_LOCAL, "HV_local": _HV_LOCAL,
            "K": _K, "V": _V, "tp_size": _TP,
        },
    }
