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

"""w8a8_block_fp8_matmul baseline（方案 B）——FP8 分块缩放 GEMM。

native：vllm.model_executor.layers.quantization.utils.fp8_utils.w8a8_triton_block_scaled_mm
    （src: fp8_utils.py#L865）。签名（已回源核对）：
        w8a8_triton_block_scaled_mm(A, B, As, Bs, block_size, output_dtype=torch.float16)
    A(activation, fp8)、B(weight, fp8)、As(A 的 per-token-group 缩放)、
    Bs(B 的 per-block 缩放)、block_size=[128,128]，输出 output_dtype 张量。
    OP_NAME 保留 CSV 里的 w8a8_block_fp8_matmul，native 符号为 w8a8_triton_block_scaled_mm。

输入构造复刻 FlagGems-vllm/benchmark/test_blas_perf_parallel.py 的
    W8A8BlockFP8MatmulBenchmark.get_input_iter：
    block_n, block_k = 128, 128
    num_k_groups = cdiv(k, block_k)，num_n_groups = cdiv(n, block_n)
    A  = rand_fp8_tensor((m, k))  fp8_e4m3fn contiguous
    B  = rand_fp8_tensor((n, k))  fp8_e4m3fn contiguous
    As = (0.01*rand(m, num_k_groups)+0.005) fp32
    Bs = (0.01*rand(num_n_groups, num_k_groups)+0.005) fp32
    调用 op(A, B, As, Bs, [128,128], torch.float16)
shape 来源：test_blas_perf_parallel.py W8A8_BLOCK_FP8_MNK_SHAPES（(M,N,K) 七档）。
"""

import importlib

import torch

OP_NAME = "w8a8_block_fp8_matmul"
DTYPES = [torch.float8_e4m3fn]  # A/B 为 fp8；输出 float16
IS_INPLACE = False

_BLOCK_SIZE = [128, 128]

# (M, N, K)（test_blas_perf_parallel.py W8A8_BLOCK_FP8_MNK_SHAPES）
_SHAPES = [
    (64, 128, 128),
    (128, 256, 512),
    (1, 4096, 7168),
    (16, 4096, 7168),
    (64, 4096, 7168),
    (83, 7748, 3884),
    (84, 7168, 3884),
]


def _cdiv(a, b):
    return -(-a // b)


def _rand_fp8(shape, device, dtype):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return (
        torch.randn(shape, device=device, dtype=torch.float32)
        .clamp(min=finfo.min, max=finfo.max)
        .to(torch.float8_e4m3fn)
    )


def native():
    """解析 fp8_utils.w8a8_triton_block_scaled_mm；解析不到返回 None。"""
    try:
        mod = importlib.import_module(
            "vllm.model_executor.layers.quantization.utils.fp8_utils"
        )
    except ImportError:
        return None
    op = getattr(mod, "w8a8_triton_block_scaled_mm", None)
    return op if callable(op) else None


def grid():
    return [{"m": m, "n": n, "k": k} for (m, n, k) in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n, k = binding["m"], binding["n"], binding["k"]
    block_n, block_k = _BLOCK_SIZE
    num_k_groups = _cdiv(k, block_k)
    num_n_groups = _cdiv(n, block_n)

    A = _rand_fp8((m, k), device, dtype).contiguous()
    B = _rand_fp8((n, k), device, dtype).contiguous()
    As = (
        0.01 * torch.rand((m, num_k_groups), dtype=torch.float32, device=device)
        + 0.005
    ).contiguous()
    Bs = (
        0.01
        * torch.rand((num_n_groups, num_k_groups), dtype=torch.float32, device=device)
        + 0.005
    ).contiguous()
    return (A, B, As, Bs, list(_BLOCK_SIZE), torch.float16), {}


def key_shape(binding):
    return [binding["m"], binding["n"], binding["k"]]


def config(binding, dtype):
    m, n, k = binding["m"], binding["n"], binding["k"]
    block_n, block_k = _BLOCK_SIZE
    return {
        "quant": {"scheme": "W8A8 block fp8", "block_size": _BLOCK_SIZE},
        "inputs": {
            "A": {"shape": [m, k], "dtype": "torch.float8_e4m3fn"},
            "B": {"shape": [n, k], "dtype": "torch.float8_e4m3fn"},
            "As": {"shape": [m, _cdiv(k, block_k)], "dtype": "torch.float32"},
            "Bs": {"shape": [_cdiv(n, block_n), _cdiv(k, block_k)],
                   "dtype": "torch.float32"},
        },
        "outputs": {"out": {"shape": [m, n], "dtype": "torch.float16"}},
        "dims": {"M": m, "N": n, "K": k,
                 "block_n": block_n, "block_k": block_k},
    }
