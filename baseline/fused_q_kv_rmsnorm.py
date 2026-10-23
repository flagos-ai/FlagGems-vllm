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

"""fused_q_kv_rmsnorm baseline（方案 B）。

native：DeepseekV4 融合 Q/KV RMSNorm 的 Triton wrapper。
    - 本地 vllm 实际符号在
      vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py#L57
      def fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)
          -> tuple[Tensor, Tensor]
      （亦经 common/ops/__init__.py 再导出为
      vllm.models.deepseek_v4.common.ops.fused_q_kv_rmsnorm）。
      内部启动 _fused_q_kv_rmsnorm_kernel[(num_tokens, 2)]，非原地（新分配
      qr_out/kv_out 两个输出），全程 fp32 归约后单次 cast 落回。
    - 另有旧布局/另一发行的回退路径
      vllm.v1.attention.ops.deepseek_v4_ops / vllm.models.common.ops.fused_qk_rmsnorm。
    native() 依次尝试上述路径，均取不到返回 None（采集器优雅跳过）。

输入构造复刻
FlagGems-vllm/benchmark/test_deepseek_v4_attention_fused_q_kv_rmsnorm.py 的
get_input_iter（dtype=bf16）：
    shape = (tokens, qdim, kvdim)
    qr        = randn(tokens, qdim)
    kv        = randn(tokens, kvdim)
    q_weight  = randn(qdim,)
    kv_weight = randn(kvdim,)
    eps       = 1e-6
shape 取该 benchmark set_shapes 里的 7 组档位。
"""

import importlib

import torch

OP_NAME = "fused_q_kv_rmsnorm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False  # 新分配 qr_out/kv_out，返回 (qr_out, kv_out)

_EPS = 1e-6

# (module path, symbol) — 依次尝试。本地 vllm 实际落点为
# vllm.models.deepseek_v4.common.ops.fused_qk_rmsnorm.fused_q_kv_rmsnorm（已回源核对：
# /Users/zjs-office/all_code/vllm/vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py#L57，
# 亦经 common/ops/__init__.py 再导出）；其余为旧布局/另一发行的回退路径。
_CANDIDATES = [
    ("vllm.models.deepseek_v4.common.ops.fused_qk_rmsnorm", "fused_q_kv_rmsnorm"),
    ("vllm.models.deepseek_v4.common.ops", "fused_q_kv_rmsnorm"),
    ("vllm.v1.attention.ops.deepseek_v4_ops", "fused_q_kv_rmsnorm"),
    ("vllm.models.common.ops.fused_qk_rmsnorm", "fused_q_kv_rmsnorm"),
]

# benchmark set_shapes：(tokens, qdim, kvdim)。64*576 == 36864。
_SHAPES = [
    (1, 1536, 512),
    (32, 1536, 512),
    (128, 1536, 512),
    (512, 1536, 512),
    (2048, 1536, 512),
    (32, 64 * 576, 576),
    (128, 64 * 576, 576),
]


def native():
    """依次尝试公开路径与实际源码路径，取出 fused_q_kv_rmsnorm。"""
    for module, symbol in _CANDIDATES:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        op = getattr(mod, symbol, None)
        if callable(op):
            return op
    return None


def grid():
    return [{"tokens": t, "qdim": q, "kvdim": kv} for (t, q, kv) in _SHAPES]


def build_inputs(binding, dtype, device):
    t, q, kv = binding["tokens"], binding["qdim"], binding["kvdim"]
    qr = torch.randn((t, q), dtype=dtype, device=device)
    kv_t = torch.randn((t, kv), dtype=dtype, device=device)
    q_weight = torch.randn((q,), dtype=dtype, device=device)
    kv_weight = torch.randn((kv,), dtype=dtype, device=device)
    args = (qr, kv_t, q_weight, kv_weight, _EPS)
    return args, {}


def key_shape(binding):
    return [binding["tokens"], binding["qdim"], binding["kvdim"]]


def config(binding, dtype):
    t, q, kv = binding["tokens"], binding["qdim"], binding["kvdim"]
    dt = str(dtype)
    return {
        "inputs": {
            "qr": {"shape": [t, q], "dtype": dt, "fill": "randn"},
            "kv": {"shape": [t, kv], "dtype": dt, "fill": "randn"},
            "q_weight": {"shape": [q], "dtype": dt, "fill": "randn"},
            "kv_weight": {"shape": [kv], "dtype": dt, "fill": "randn"},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "qr_out": {"shape": [t, q], "dtype": dt},
            "kv_out": {"shape": [t, kv], "dtype": dt},
        },
        "dims": {"tokens": t, "qdim": q, "kvdim": kv},
    }
