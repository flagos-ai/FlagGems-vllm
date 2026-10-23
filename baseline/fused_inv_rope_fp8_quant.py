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

"""fused_inv_rope_fp8_quant baseline（方案 B）。

native：DeepseekV4 融合逆 RoPE + 分块 FP8 量化的 wrapper。
    - benchmark 记载的公开入口是
      vllm.v1.attention.ops.deepseek_v4_ops.fused_inv_rope_fp8_quant；
      本地 vllm 检出（76ba32160a）里 **不存在** 该模块。
    - 实际符号在
      vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py#L151
      def fused_inv_rope_fp8_quant(o, positions, cos_sin_cache, n_groups,
          heads_per_group, nope_dim=448, rope_dim=64, quant_group_size=128,
          tma_aligned_scales=False) -> tuple[Tensor, Tensor]
      （亦经 common/ops/__init__.py 再导出）。内部经
      torch.ops.vllm.fused_inv_rope_fp8_quant_kernel 启动 Triton kernel，
      非原地（返回 o_fp8 / o_scale 两个新张量）。
    native() 先试公开路径，再回落 __init__ 再导出，最后回落实际模块；
    都取不到返回 None。

输入构造复刻 FlagGems-vllm/benchmark/test_fused_inv_rope_fp8_quant.py 的
_input_fn（dtype=bf16，需 native float8_e4m3fn）：
    常量 HEAD_DIM=512, NOPE_DIM=448, ROPE_DIM=64, QUANT_GROUP_SIZE=128
    shape = (num_tokens, num_heads, n_groups, tma_aligned_scales)
    heads_per_group = num_heads // n_groups
    max_pos = max(4096, num_tokens*2)
    o          = randn(num_tokens, num_heads, HEAD_DIM)           bf16
    positions  = randint(0, max_pos, (num_tokens,))              int64
    cos_sin_cache = _make_cos_sin_cache(max_pos, ROPE_DIM)        fp32
    传参顺序：(o, positions, cos_sin_cache, n_groups, heads_per_group,
              NOPE_DIM, ROPE_DIM, QUANT_GROUP_SIZE, tma_aligned_scales)
    （benchmark torch_op 走位置参数；本模块 nope/rope/quant/tma 用 kwargs，
      与源码默认名一致，等价。）
shape 取该 benchmark 的 DEFAULT_SHAPES 两组档位。
"""

import importlib

import torch

OP_NAME = "fused_inv_rope_fp8_quant"
DTYPES = [torch.bfloat16]
IS_INPLACE = False  # 返回 (o_fp8, o_scale) 两个新张量

# benchmark 常量。
_HEAD_DIM = 512
_NOPE_DIM = 448
_ROPE_DIM = 64
_QUANT_GROUP_SIZE = 128

_CANDIDATES = [
    ("vllm.v1.attention.ops.deepseek_v4_ops", "fused_inv_rope_fp8_quant"),
    ("vllm.models.deepseek_v4.common.ops", "fused_inv_rope_fp8_quant"),
    ("vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant",
     "fused_inv_rope_fp8_quant"),
]

# benchmark DEFAULT_SHAPES：(num_tokens, num_heads, n_groups, tma_aligned_scales)
_SHAPES = [
    (1, 8, 1, True),
    (16, 64, 8, True),
]


def _make_cos_sin_cache(max_pos, rope_dim, device):
    """复刻 benchmark _make_cos_sin_cache：[max_pos, rope_dim] fp32，cos||sin。"""
    half = rope_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def native():
    """依次尝试公开路径 / 包再导出 / 实际模块。"""
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
    return [
        {
            "num_tokens": nt,
            "num_heads": nh,
            "n_groups": ng,
            "tma_aligned_scales": tma,
        }
        for (nt, nh, ng, tma) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    nt = binding["num_tokens"]
    nh = binding["num_heads"]
    ng = binding["n_groups"]
    tma = binding["tma_aligned_scales"]
    heads_per_group = nh // ng
    max_pos = max(4096, nt * 2)
    o = torch.randn(nt, nh, _HEAD_DIM, dtype=dtype, device=device)
    # torch.int64 == torch.long；用 int64 兼容离线冒烟 stub。
    positions = torch.randint(0, max_pos, (nt,), dtype=torch.int64, device=device)
    try:
        cos_sin_cache = _make_cos_sin_cache(max_pos, _ROPE_DIM,
                                            torch.device(device))
    except (AttributeError, TypeError):
        # 离线冒烟 stub 缺 outer/cos/cat 等；回落到 shape 正确的占位张量，
        # 真卡上走上面的真实构造（cos||sin，fp32）。
        cos_sin_cache = torch.empty((max_pos, _ROPE_DIM), dtype=torch.float32,
                                    device=device)
    args = (o, positions, cos_sin_cache, ng, heads_per_group)
    kwargs = {
        "nope_dim": _NOPE_DIM,
        "rope_dim": _ROPE_DIM,
        "quant_group_size": _QUANT_GROUP_SIZE,
        "tma_aligned_scales": tma,
    }
    return args, kwargs


def key_shape(binding):
    return [
        binding["num_tokens"],
        binding["num_heads"],
        binding["n_groups"],
        int(binding["tma_aligned_scales"]),
    ]


def config(binding, dtype):
    nt = binding["num_tokens"]
    nh = binding["num_heads"]
    ng = binding["n_groups"]
    tma = binding["tma_aligned_scales"]
    heads_per_group = nh // ng
    max_pos = max(4096, nt * 2)
    d = heads_per_group * _HEAD_DIM
    dt = str(dtype)
    return {
        "inputs": {
            "o": {"shape": [nt, nh, _HEAD_DIM], "dtype": dt, "fill": "randn"},
            "positions": {"shape": [nt], "dtype": "torch.int64"},
            "cos_sin_cache": {"shape": [max_pos, _ROPE_DIM],
                              "dtype": "torch.float32"},
            "n_groups": {"scalar": ng},
            "heads_per_group": {"scalar": heads_per_group},
            "nope_dim": {"scalar": _NOPE_DIM},
            "rope_dim": {"scalar": _ROPE_DIM},
            "quant_group_size": {"scalar": _QUANT_GROUP_SIZE},
            "tma_aligned_scales": {"scalar": tma},
        },
        "outputs": {
            "o_fp8": {"shape": [nt, ng, d], "dtype": "torch.float8_e4m3fn",
                      "note": "源码返回 transpose(0,1) 视图"},
            "o_scale": {"dtype": "torch.int32" if tma else "torch.float32",
                        "note": "预变换 scale，形状随 tma_aligned 分支"},
        },
        "dims": {
            "num_tokens": nt,
            "num_heads": nh,
            "n_groups": ng,
            "heads_per_group": heads_per_group,
            "head_dim": _HEAD_DIM,
        },
    }
