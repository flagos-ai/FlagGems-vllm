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

"""pack_seq_triton baseline（方案 B）。

native：vllm.v1.attention.ops.common.pack_seq_triton(x, lengths, pad_value=-inf,
        block_t=64, block_d=64) -> packed[B, Lmax, ...]。
    把 [N, D] 里按 lengths 分段的序列打包成 [B, Lmax, D]。作为 torch_op 直接调用。
    src: vllm/v1/attention/ops/common.py#L63

输入构造复刻 FlagGems-vllm/benchmark/test_pack_seq.py 的 _pack_input_fn：
    x = randn(N, D); lengths = tensor(lengths_list, int32)
    shape 配置为 (N, D, B, lengths_list)。
"""

import importlib

import torch

OP_NAME = "pack_seq_triton"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False

# benchmark PACK_BENCH_SHAPES（test_pack_seq.py#L71）：(N, D, B, lengths_list)。
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
    """解析 vllm.v1.attention.ops.common.pack_seq_triton；解析不到返回 None。"""
    try:
        mod = importlib.import_module("vllm.v1.attention.ops.common")
    except ImportError:
        return None
    op = getattr(mod, "pack_seq_triton", None)
    return op if callable(op) else None


def grid():
    return [{"N": n, "D": d, "B": b, "lengths": list(ls)}
            for (n, d, b, ls) in _SHAPES]


def build_inputs(binding, dtype, device):
    n, d = binding["N"], binding["D"]
    lengths = torch.tensor(binding["lengths"], dtype=torch.int32, device=device)
    x = torch.randn(n, d, dtype=dtype, device=device)
    return (x, lengths), {}


def key_shape(binding):
    return f"N{binding['N']}_D{binding['D']}_B{binding['B']}"


def config(binding, dtype):
    n, d, b = binding["N"], binding["D"], binding["B"]
    lengths = binding["lengths"]
    lmax = max(lengths)
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [n, d], "dtype": dt},
            "lengths": {"shape": [b], "dtype": "torch.int32",
                        "values": lengths},
        },
        "outputs": {
            "packed": {"shape": [b, lmax, d], "dtype": dt},
        },
        "dims": {"N": n, "D": d, "B": b, "Lmax": lmax},
    }
