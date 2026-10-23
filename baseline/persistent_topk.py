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

"""persistent_topk baseline（方案 B）。

native：torch.ops._C.persistent_topk(logits, lengths, output, workspace, k,
        max_seq_len) -> ()。output 原地写 top-k 索引。
    绑定签名见 csrc/libtorch_stable/torch_bindings.cpp#L569。
    需先 import vllm._custom_ops 触发 torch.ops._C 命名空间注册。

输入构造复刻 FlagGems-vllm/benchmark/test_persistent_topk.py 的
PersistentTopKBenchmark.get_input_iter（K=512, STRIDE=262144）：
  同构档（主 shapes，num_rows/seq_len/max_seq_len）：
    logits = full((num_rows, STRIDE), -inf, fp32); logits[:, :seq_len]=randn(...)
    lengths = full((num_rows,), seq_len, int32)
    indices = empty((num_rows, K), int32)                 输出
    workspace = empty(1024*1024, uint8)                   HAS_VLLM 路径为 1MB
    native 调用：persistent_topk(logits, lengths, indices, workspace, K, max_seq_len)
  异构档（hetero_shapes，num_rows/max_len）：
    lengths = linspace(1, min(max_len, STRIDE), num_rows) int32（各行 seq_len 不等）
    logits = full((num_rows, STRIDE), -inf, fp32); 逐行 logits[i,:sl]=randn(sl)
    native 调用：persistent_topk(logits, lengths, indices, workspace, K, max_len)

shape 与 benchmark set_shapes 完全一致（NV 路径 BASELINE_BROKEN_MAX_SEQ=0，不过滤；
_METAX_EXTRA_SHAPES 仅 MetaX 追加，NV 不含）：
  主 shapes 42 档 = 7 个 decode 档 + (1,8192,8192) + (1,32773,32773) 下界
    + num_rows∈1..32 的 (nr,262144,1048576) 32 档 + (496,...) + (512,...)；
  hetero_shapes 2 档 = (496,1048576)、(496,1)。
注意：大档单个 logits 张量达数百 MB（512*262144*4B≈537MB），真卡全量采集显存压力大，
按「与 benchmark 一致」要求保留全部档位，实跑若 OOM 可用 --ops 单独跑或临时裁剪。
"""

import importlib

import torch

OP_NAME = "persistent_topk"
DTYPES = [torch.float32]
IS_INPLACE = True  # indices 原地写

_K = 512
_STRIDE = 262144
# benchmark 侧 workspace = (1 if HAS_VLLM else 2)*1MB。签名匹配按 HAS_VLLM=False
# 的 2MB 口径声明（见 config 注释）；build_inputs 的实采仍用 1MB，互不影响。
_WORKSPACE_SIG = 2 * 1024 * 1024  # 2097152

# benchmark set_shapes 的同构主 shapes：(num_rows, seq_len, max_seq_len)。
_SHAPES = [
    # Decode path
    (1, 4102, 4102),
    (4, 4102, 4102),
    (10, 1055, 1055),
    (12, 4105, 4105),
    (20, 4105, 4105),
    (28, 4109, 4109),
    (1, 8192, 8192),
    # Large path lower bound
    (1, 32773, 32773),
    # Large path: num_rows=1..32, seq_len=262144, max_seq_len=1048576
    *[(nr, 262144, 1048576) for nr in range(1, 33)],
    (496, 262144, 1048576),
    (512, 262144, 1048576),
]

# benchmark hetero_shapes：(num_rows, max_len)。各行 seq_len 用 linspace 从 1 到
# min(max_len, STRIDE) 铺开（异构 batch）。
_HETERO_SHAPES = [
    (496, 1048576),
    (496, 1),
]


def native():
    """解析 torch.ops._C.persistent_topk；解析不到返回 None。

    先 import vllm._custom_ops 以注册 torch.ops._C 命名空间。
    """
    try:
        importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(getattr(torch.ops, "_C", None), "persistent_topk", None)
    return op if callable(op) else None


def grid():
    homo = [{"kind": "homo", "num_rows": r, "seq_len": s, "max_seq_len": m}
            for (r, s, m) in _SHAPES]
    hetero = [{"kind": "hetero", "num_rows": r, "max_len": ml}
              for (r, ml) in _HETERO_SHAPES]
    return homo + hetero


def build_inputs(binding, dtype, device):
    r = binding["num_rows"]
    indices = torch.empty((r, _K), dtype=torch.int32, device=device)
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)

    if binding["kind"] == "hetero":
        # 各行 seq_len 用 linspace(1, min(max_len, STRIDE), num_rows) 铺开。
        ml = binding["max_len"]
        hi = min(ml, _STRIDE)
        lengths = torch.linspace(1, hi, r, device=device).to(torch.int32)
        logits = torch.full((r, _STRIDE), float("-inf"),
                            dtype=torch.float32, device=device)
        for i in range(r):
            sl = int(lengths[i].item())
            logits[i, :sl] = torch.randn(sl, device=device)
        return (logits, lengths, indices, workspace, _K, ml), {}

    s, m = binding["seq_len"], binding["max_seq_len"]
    logits = torch.full((r, _STRIDE), float("-inf"),
                        dtype=torch.float32, device=device)
    logits[:, :s] = torch.randn(r, s, device=device)
    lengths = torch.full((r,), s, dtype=torch.int32, device=device)
    return (logits, lengths, indices, workspace, _K, m), {}


# 已知限制：benchmark 的张量签名匹配只看 shape + 标量，看不到张量内容。
# homo 档 [496, 262144, 1048576] 与 hetero 档 nr496_maxlen1048576 在 num_rows=496、
# max_seq_len=1048576 下，所有张量 shape 与标量完全相同（唯一区别是 lengths 的
# 内容：homo 全等 seq_len vs hetero linspace 逐行不等），故这两个档彼此无法区分，
# 均会匹配未命中（benchmark 的 >8192 过滤不改变这一点，两档都保留）。其余所有档
# 可由 max_seq_len 标量唯一区分。此歧义源于 benchmark 的 shape 设计，baseline 侧
# 的 config 无法消除——接受这两个档 miss。
def key_shape(binding):
    if binding["kind"] == "hetero":
        return f"hetero_nr{binding['num_rows']}_maxlen{binding['max_len']}"
    return [binding["num_rows"], binding["seq_len"], binding["max_seq_len"]]


def config(binding, dtype):
    r = binding["num_rows"]
    if binding["kind"] == "hetero":
        ml = binding["max_len"]
        return {
            # 签名匹配口径对齐 benchmark get_input_iter 的 record（yield 的 6 元组）：
            #   logits, lengths, indices, workspace, max_seq_len(标量), seq_lens
            # 故此处：补 seq_lens 张量、workspace 用 2MB（HAS_VLLM=False）、去掉 k
            # 标量（benchmark record 未含 k，k 写死在 gems/baseline 调用里）。
            # build_inputs 的实采参数不受影响。
            "inputs": {
                "logits": {"shape": [r, _STRIDE], "dtype": "torch.float32",
                           "note": "各行 seq_len=linspace(1,min(max_len,STRIDE),nr)，"
                                   "其余 -inf"},
                "lengths": {"shape": [r], "dtype": "torch.int32",
                            "note": "异构：逐行不等"},
                "output": {"shape": [r, _K], "dtype": "torch.int32",
                           "note": "原地写回（top-k 索引）"},
                "workspace": {"shape": [_WORKSPACE_SIG], "dtype": "torch.uint8"},
                "max_seq_len": {"scalar": ml},
                "seq_lens": {"shape": [r], "dtype": "torch.int32",
                             "note": "benchmark 第 6 参，逐行 seq_len"},
            },
            "outputs": {
                "output": {"shape": [r, _K], "dtype": "torch.int32"},
            },
            "dims": {"kind": "hetero", "num_rows": r, "max_len": ml,
                     "K": _K, "STRIDE": _STRIDE},
        }

    s, m = binding["seq_len"], binding["max_seq_len"]
    return {
        # 签名匹配口径对齐 benchmark record（见 hetero 分支注释）：补 seq_lens、
        # workspace 用 2MB、去掉 k 标量。
        "inputs": {
            "logits": {"shape": [r, _STRIDE], "dtype": "torch.float32",
                       "note": f"前 seq_len={s} 有效，其余 -inf"},
            "lengths": {"shape": [r], "dtype": "torch.int32"},
            "output": {"shape": [r, _K], "dtype": "torch.int32",
                       "note": "原地写回（top-k 索引）"},
            "workspace": {"shape": [_WORKSPACE_SIG], "dtype": "torch.uint8"},
            "max_seq_len": {"scalar": m},
            "seq_lens": {"shape": [r], "dtype": "torch.int32",
                         "note": "benchmark 第 6 参，逐行 seq_len"},
        },
        "outputs": {
            "output": {"shape": [r, _K], "dtype": "torch.int32"},
        },
        "dims": {"kind": "homo", "num_rows": r, "seq_len": s, "max_seq_len": m,
                 "K": _K, "STRIDE": _STRIDE},
    }
