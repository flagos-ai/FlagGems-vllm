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

import importlib

import torch

OP_NAME = "marlin_gemm"
DTYPES = [torch.float16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = None

GROUP_SIZE = 128

# Identical to the benchmark shape grid (M, N, K).
MARLIN_GEMM_MS = [1, 16, 64, 128]
MARLIN_GEMM_NS = [512, 1024, 2048]
MARLIN_GEMM_KS = [512, 1024, 2048]
MARLIN_GEMM_SHAPES = [
    (m, n, k) for m in MARLIN_GEMM_MS for n in MARLIN_GEMM_NS for k in MARLIN_GEMM_KS
]


def native():
    """Resolve the vLLM marlin_gemm CUDA baseline; no torch fallback."""
    global NATIVE_UNAVAILABLE_REASON
    NATIVE_UNAVAILABLE_REASON = None
    try:
        mod = importlib.import_module("vllm._custom_ops")
    except (ImportError, OSError) as exc:
        NATIVE_UNAVAILABLE_REASON = str(exc)
        return None
    op = getattr(mod, "marlin_gemm", None)
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = "vllm._custom_ops has no marlin_gemm"
        return None
    return op


def _uint4b8_scalar_type():
    from vllm.scalar_type import scalar_types

    return scalar_types.uint4b8


def _make_workspace(device):
    # Same size/semantics as benchmark's make_workspace; torch-only fallback.
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new,
        )

        return marlin_make_workspace_new(torch.device(device), 4)
    except (ImportError, OSError):
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        return torch.zeros(sms * 4, dtype=torch.int32, device=device)


def _quantize_and_pack(n, k, dtype, device):
    """Symmetric uint4b8 W4A16 quantization + Marlin repack (torch, baseline-only).

    Returns (b_q_weight, b_scales): int32 (K//16, N*2) Marlin 16x64-unit weights
    and marlin-permuted (K//gs, N) scales; mirrors the benchmark input logic.
    """
    scale_perm = torch.tensor(
        [i + 8 * j for i in range(8) for j in range(8)], device=device
    )
    w = torch.randn((n, k), dtype=dtype, device=device) * 0.1
    grouped = w.reshape(n, k // GROUP_SIZE, GROUP_SIZE).float()
    scales_nk = grouped.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7.0
    q = (
        (torch.round(grouped / scales_nk).clamp(-7, 7) + 8)
        .to(torch.int32)
        .reshape(n, k)
    )
    scales = (
        scales_nk.squeeze(-1)
        .t()
        .reshape(-1, 64)[:, scale_perm]
        .reshape(-1, n)
        .to(dtype)
        .contiguous()
    )

    words = torch.zeros((k // 16) * (n // 64) * 128, dtype=torch.int32, device=device)
    kk = torch.arange(16, device=device)
    n_loc = torch.arange(64, device=device)
    waddr = 4 * (((n_loc % 8) * 4).unsqueeze(0) + ((kk % 8) // 2).unsqueeze(1)) + (
        n_loc // 16
    ).unsqueeze(0)
    j4 = (
        4
        * (
            (4 * (kk % 2) + 2 * (kk // 8)).unsqueeze(1)
            + ((n_loc // 8) % 2).unsqueeze(0)
        )
    ).int()
    waddr = waddr.reshape(-1).long()
    for kb in range(k // 16):
        for pid_n in range(n // 64):
            base = (kb * (n // 64) + pid_n) * 128
            tile = q[pid_n * 64 : (pid_n + 1) * 64, kb * 16 : (kb + 1) * 16].T
            words.index_put_((base + waddr,), (tile << j4).reshape(-1), accumulate=True)
    return words.view(k // 16, n * 2), scales


def grid():
    return [{"shape": [m, n, k]} for m, n, k in MARLIN_GEMM_SHAPES]


def build_inputs(binding, dtype, device):
    m, n, k = binding["shape"]
    b_q_weight, b_scales = _quantize_and_pack(n, k, dtype, device)
    a = torch.randn((m, k), dtype=dtype, device=device)
    workspace = _make_workspace(device)
    kwargs = {
        "a": a,
        "c": None,
        "b_q_weight": b_q_weight,
        "b_bias": None,
        "b_scales": b_scales,
        "a_scales": None,
        "global_scale": None,
        "b_zeros": None,
        "g_idx": None,
        "perm": None,
        "workspace": workspace,
        "b_q_type": _uint4b8_scalar_type(),
        "size_m": m,
        "size_n": n,
        "size_k": k,
    }
    return (), kwargs


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    m, n, k = binding["shape"]
    dt = str(dtype)
    return {
        "inputs": {
            "a": {"shape": [m, k], "dtype": dt},
            "b_q_weight": {
                "shape": [k // 16, n * 2],
                "dtype": "torch.int32",
                "alias": "uint4b8 codes in Marlin 16x64-unit int32 layout",
            },
            "b_scales": {
                "shape": [k // GROUP_SIZE, n],
                "dtype": dt,
                "alias": "vLLM get_scale_perm applied to (K//gs, N)",
            },
            "b_q_type": {"scalar": "uint4b8"},
            "group_size": {"scalar": GROUP_SIZE},
        },
        "outputs": {
            "y": {"shape": [m, n], "dtype": dt, "alias": "fresh; c=None"},
        },
        "dims": {"m": m, "n": n, "k": k},
        "compute_dtype": dt,
        "math": "y = a @ dequant(b_q_weight, b_scales)^T",
    }
