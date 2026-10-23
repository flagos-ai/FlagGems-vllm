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

"""unpack_seq_triton baseline（方案 B）。

native：vllm.v1.attention.ops.common.unpack_seq_triton(packed_tensor, lengths,
        block_t=64, block_d=64) -> unpacked[N, ...]，N=sum(lengths)。
    pack_seq_triton 的逆操作，把 [B, Lmax, D] 还原成 [N, D]。作为 torch_op 直接调用。
    src: vllm/v1/attention/ops/common.py#L177

输入构造复刻 FlagGems-vllm/benchmark/test_unpack_seq.py 的 _unpack_input_fn：
    Lmax = max(lengths_list)
    packed = randn(B, Lmax, D); lengths = tensor(lengths_list, int32)
    shape 配置为 (N, D, B, lengths_list)（N 仅用于对齐 pack 侧，unpack 输入是 packed）。
"""

import importlib

import torch

OP_NAME = "unpack_seq_triton"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False

# benchmark UNPACK_BENCH_SHAPES（test_unpack_seq.py#L72）：(N, D, B, lengths_list)。
_SHAPES = [
    (512, 64, 5, [64, 128, 64, 128, 128]),
    (4096, 128, 4, [1024, 1024, 1024, 1024]),
    (8192, 256, 5, [1024, 2048, 1024, 2048, 2048]),
    (2048, 512, 4, [512, 512, 512, 512]),
    (16384, 64, 8, [2048] * 8),
    (1024, 1024, 4, [256] * 4),
    (2048, 2048, 512, [4] * 512),
    (4094, 1024, 1024, [4] * 1024),
    (8192, 1024, 1024, [8] * 1024),
]


def native():
    """解析 vllm.v1.attention.ops.common.unpack_seq_triton；解析不到返回 None。"""
    try:
        mod = importlib.import_module("vllm.v1.attention.ops.common")
    except ImportError:
        return None
    op = getattr(mod, "unpack_seq_triton", None)
    return op if callable(op) else None


def grid():
    return [{"N": n, "D": d, "B": b, "lengths": list(ls)}
            for (n, d, b, ls) in _SHAPES]


def build_inputs(binding, dtype, device):
    d, b = binding["D"], binding["B"]
    lengths_list = binding["lengths"]
    lmax = max(lengths_list)
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)
    packed = torch.randn(b, lmax, d, dtype=dtype, device=device)
    return (packed, lengths), {}


def key_shape(binding):
    lmax = max(binding["lengths"])
    return f"B{binding['B']}_Lmax{lmax}_D{binding['D']}"


def config(binding, dtype):
    n, d, b = binding["N"], binding["D"], binding["B"]
    lengths = binding["lengths"]
    lmax = max(lengths)
    dt = str(dtype)
    return {
        "inputs": {
            "packed_tensor": {"shape": [b, lmax, d], "dtype": dt},
            "lengths": {"shape": [b], "dtype": "torch.int32",
                        "values": lengths},
        },
        "outputs": {
            "unpacked": {"shape": [sum(lengths), d], "dtype": dt},
        },
        "dims": {"N": n, "D": d, "B": b, "Lmax": lmax},
    }
