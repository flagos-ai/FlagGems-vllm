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

"""Optimized fused MoE kernel for Ascend 910B.

This module provides an Ascend-optimized implementation of the fused MoE
(Mixture of Experts) operator, designed for vLLM inference workloads.
The kernel performs expert routing, GEMM operations, and activation fusion
with optimizations tailored for Ascend NPU architecture.

[KernelGen] This kernel was generated and optimized using automated kernel
generation and tuning infrastructure.
"""


import functools
import logging
import os
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import yaml

try:
    # High-performance CANN primitives (vectorized sort/gather/scatter for
    # Ascend). Used by the sort-based align scatter.
    import triton.language.extra.cann.extension as al_ext

    _HAS_CANN_EXT = tl.constexpr(True)
except ImportError:
    al_ext = None
    _HAS_CANN_EXT = tl.constexpr(False)

# Using relative imports will cause the module to be not found.
from flaggems_vllm.runtime.backend._ascend.fused.moe_sum import moe_sum
from flaggems_vllm.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# OCP MX quantization helpers (requires amd-quark)

OCP_MX_BLOCK_SIZE = 32


@functools.lru_cache(maxsize=1)
def get_embedded_moe_configs():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "utils", "configs", "fused_moe_config.yaml"
    )
    if not os.path.exists(config_path):
        return {}, {}
    with open(config_path, "r") as f:
        # JSON keys are strings, values are dicts where keys are M and values are configs
        data = yaml.safe_load(f)

        fallback = data.get("_FALLBACK", {})

        # We need to convert the innermost keys (which are stringified integers for M) back to integers.
        # Ensure we map the lists back to config dicts.
        keys_order = [
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_N",
            "BLOCK_SIZE_K",
            "GROUP_SIZE_M",
            "num_warps",
            "num_stages",
        ]
        parsed_data = {}
        for dev, configs in data.items():
            if dev == "_FALLBACK":
                continue
            parsed_data[dev] = {}
            for k, m_dict in configs.items():
                parsed_dict = {}
                for m, v in m_dict.items():
                    if isinstance(v, list):
                        parsed_dict[int(m)] = dict(zip(keys_order, v))
                    else:
                        parsed_dict[int(m)] = v
                parsed_data[dev][k] = parsed_dict

        return parsed_data, fallback


def dequant_mxfp4(
    x: torch.Tensor,
    scale: torch.Tensor,
    float_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize MXFP4 tensor via quark.torch.kernel.mx.dq_mxfp4."""
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError("amd-quark is required for MX-FP4") from err

    return mx.dq_mxfp4(x, scale, float_dtype)


def dequant_mxfp6(
    x: torch.Tensor,
    scale: torch.Tensor,
    float_dtype: torch.dtype,
    quant_dtype: str,
) -> torch.Tensor:
    """Dequantize MXFP6 tensor via quark hw_emulation."""
    try:
        from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
            dequantize_fp4_fp6_per_group,
        )
        from quark.torch.utils.pack import create_pack_method
    except ImportError as err:
        raise ImportError("amd-quark is required for MX-FP6") from err

    pack_method = create_pack_method(None, dtype=quant_dtype)
    unpacked_x = pack_method.unpack(x, reorder=False)

    scale = 2 ** (scale.view(torch.uint8).to(torch.int16) - 127).to(float_dtype)

    return dequantize_fp4_fp6_per_group(
        unpacked_x,
        scale,
        axis=-1,
        group_size=OCP_MX_BLOCK_SIZE,
        quant_dtype=quant_dtype,
    ).to(float_dtype)


# Activation quantization helpers


@functools.lru_cache(maxsize=1)
def _get_device_name() -> str:
    """Return the normalised CUDA device name (spaces replaced by underscores).

    Matches the naming convention used by vLLM for its per-device config files.
    H800 falls back to H100_80GB_HBM3 (same SM 9.0 architecture).
    """
    name = torch.npu.get_device_name().replace(" ", "_")
    # Normalise the H200 product family to a single key, following vLLM.
    if "H200" in name.split("_"):
        name = "NVIDIA_H200"
    # H800 has the same SM 9.0 as H100; use H100 configs as fallback.
    embedded_configs, fallback_mapping = get_embedded_moe_configs()
    if name in embedded_configs:
        return name
    # Fallback mapping for devices whose tuning profiles are equivalent.
    fallback = fallback_mapping.get(name)
    if fallback and fallback in embedded_configs:
        logger.info(
            "GEMS_ASCEND Device %s not in config table, falling back to %s",
            name,
            fallback,
        )
        return fallback
    return name


def get_moe_configs(
    E: int,
    N: int,
    dtype: str | None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> dict[int, Any] | None:
    """
    Return optimized configurations for the fused MoE kernel.

    Looks up pre-tuned configs from the embedded table (ported from vLLM)
    for the current GPU device. Returns None if no matching config is found.
    """
    device_name = _get_device_name()
    embedded_configs, _ = get_embedded_moe_configs()
    device_table = embedded_configs.get(device_name)
    if device_table is None:
        logger.warning(
            "GEMS_ASCEND No embedded MoE configs for device %s. Will use default config.",
            device_name,
        )
        return None

    _block_n = block_n if block_n else 0
    _block_k = block_k if block_k else 0
    key = f"{E},{N},{dtype},{_block_n},{_block_k}"
    configs = device_table.get(key)
    if configs is not None:
        logger.info(
            "GEMS_ASCEND Using embedded MoE config for device=%s, key=%s",
            device_name,
            key,
        )
        return configs
    logger.warning(
        "GEMS_ASCEND No embedded MoE config for device=%s, key=%s. Will use default config.",
        device_name,
        key,
    )
    return None


def try_get_optimal_moe_config(
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    top_k: int,
    dtype: str | None,
    M: int,
    block_shape: list[int] | None = None,
) -> dict[str, int]:
    # First try to load optimal config from the embedded table
    E, _, N = w2_shape
    if dtype == "int4_w4a16":
        N = N * 2
    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0
    configs = get_moe_configs(E, N, dtype, block_n, block_k)

    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        config = get_default_config(M, E, N, w1_shape[2], top_k, dtype, block_shape)
    return config


def _get_config_quant_dtype(
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    ocp_mx_scheme: str | None,
) -> None | torch.dtype | str:
    """Map quantization flags to the corresponding dtype."""
    if use_fp8_w8a8:
        return torch.float8_e4m3fn
    elif use_int8_w8a8:
        return torch.int8
    elif ocp_mx_scheme == "w_mxfp4_a_mxfp4":
        return "mxfp4"
    elif ocp_mx_scheme in {"w_mxfp4_a_mxfp6_e3m2", "w_mxfp6_e3m2_a_mxfp6_e3m2"}:
        return "mxfp6_e3m2"
    elif ocp_mx_scheme in {"w_mxfp4_a_mxfp6_e2m3", "w_mxfp6_e2m3_a_mxfp6_e2m3"}:
        return "mxfp6_e2m3"
    elif ocp_mx_scheme in {"w_mxfp4", "w_mxfp6_e3m2", "w_mxfp6_e2m3"}:
        return torch.bfloat16
    elif ocp_mx_scheme in {"w_mxfp4_a_fp8", "w_mxfp6_e3m2_a_fp8", "w_mxfp6_e2m3_a_fp8"}:
        return torch.float8_e4m3fn

    return None


def get_moe_wna16_block_config(
    config: dict[str, int],
    use_moe_wna16_cuda: bool,
    num_valid_tokens: int,
    size_k: int,
    size_n: int,
    num_experts: int,
    group_size: int,
    real_top_k: int,
    block_size_m: int,
):
    if "BLOCK_SIZE_N" in config and "BLOCK_SIZE_K" in config:
        return {}
    if not use_moe_wna16_cuda:
        if num_valid_tokens // real_top_k == 1:
            return {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}
        else:
            return {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}
    else:
        block_size_n = 128
        block_size_k = 128
        if block_size_k <= group_size:
            block_size_k = group_size

        num_n_blocks = size_k // block_size_k
        num_k_blocks = size_n // block_size_k
        num_m_blocks = (
            num_valid_tokens + block_size_m - 1
        ) / block_size_m + num_experts
        if num_valid_tokens // real_top_k <= block_size_m:
            num_m_blocks = min(num_m_blocks, num_valid_tokens)
        num_blocks = num_m_blocks * num_n_blocks * num_k_blocks

        if size_k % 256 == 0 and num_blocks >= 256 and block_size_k < 256:
            block_size_k = 256
            num_blocks = num_blocks // (256 // block_size_k)

        if (
            num_m_blocks <= 16
            and size_k % (block_size_k * 2) == 0
            and size_k % (block_size_k * 2) == 0
            and block_size_k <= 512
            and num_blocks >= 512
        ):
            block_size_k = block_size_k * 2
            num_blocks = num_blocks // 2

        if num_blocks > 1024:
            block_size_n = 256
            num_n_blocks = num_n_blocks // 2
            num_blocks = num_blocks // 2

        if size_n <= 1024 and num_blocks >= 1024:
            block_size_n = 1024

        block_size_k = _ensure_block_size_k_divisible(size_k, block_size_k, group_size)

        return {"BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k}


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    block_shape: list[int] | None = None,
) -> dict[str, int]:
    """Default Triton config for fused MoE kernel.

    Heuristic selection aligned with vLLM v0.17.0 defaults, tuned on H20/H100.
    Key insight: for high-expert-count MoE (e.g. DeepSeek-V3 E=256), each
    expert sees very few tokens, so small BLOCK_SIZE_M (16) is critical.
    """
    if dtype == "fp8_w8a8" and block_shape is not None:
        config = {
            "BLOCK_SIZE_M": 16 if M <= 64 else 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 1 if M <= 16 else 32,
            "num_warps": 4,
            "num_stages": 3,
        }
    else:
        # tokens_per_expert drives block_m. Float division: integer division
        # truncates to 0 on decode-scale inputs (M < E), silently
        # misclassifying the shape into the wrong ladder branch.
        tokens_per_expert = M / max(E, 1)

        # BM=16 for decode-scale routing (< ~1 token per expert): the GEMM is
        # B-streaming bound there, and smaller BM directly cuts padded-row
        # waste (every non-empty expert costs BM rows of compute regardless
        # of its token count). Measured tl.dot cube caps on Ascend910B4-1
        # (perf_gemm_micro.py, bf16): BM=16 ~16, BM=64 ~41, BM=128 ~96-105;
        # BM=256 fails to compile. BN=256 + 8 warps > BN=128 + 4 warps.
        if tokens_per_expert < 1:
            block_m = 16
        elif tokens_per_expert <= 16:
            block_m = 64
        else:
            block_m = 128

        if N >= 2048:
            block_n = 256
            num_warps = 8
            # BK=256 is the current best on HBM-bound large shapes (512B
            # contiguous bursts; gemm1 16.3 -> 14.8 ms on DSv3 TP8). Only
            # safe with BM=64; BM=128 caps at BK=128 (measured on Mixtral
            # 512-token: gemm1 4.3 -> 2.9 ms, and it just fits L0C).
            block_k = 256 if block_m == 64 else 128
        else:
            # BK=64 in this branch overflows UB on small-N shapes
            # (measured: needs 198KB > 192KB UB on (8,4,64,128,2)).
            block_n = 128
            block_k = 32
            num_warps = 4

        # BK>=64 is safe again on large-N shapes now that B tiles are loaded
        # in the natural direction (the old 'vsel' compiler failure was tied
        # to the transposed-B pattern). fp8 blockwise keeps its own block_k
        # above.

        if tokens_per_expert > 128:
            group_m = 16
        elif tokens_per_expert > 32:
            group_m = 8
        else:
            group_m = 1

        num_stages = 2

        config = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    return config


def _get_config_dtype_str(
    dtype: Optional[torch.dtype] = None,
    use_fp8_w8a8: bool = False,
    use_fp8_w8a16: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
) -> str | None:
    """Return dtype string for kernel config lookup."""
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_fp8_w8a16:
        return "fp8_w8a16"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif ocp_mx_scheme is not None:
        return None
    elif dtype == torch.float:
        return "float32"
    return None


# MoE activation enum


class MoEActivation(Enum):
    """Activation functions for MoE layers."""

    # Gated: gate * activation(up), input [..., 2*d] -> output [..., d]
    SILU = "silu"
    GELU = "gelu"
    RELU2 = "relu2"
    SWIGLUOAI = "swigluoai"
    SWIGLUSTEP = "swiglustep"

    # Non-gated: input [..., d] -> output [..., d]
    SILU_NO_MUL = "silu_no_mul"
    GELU_NO_MUL = "gelu_no_mul"
    RELU2_NO_MUL = "relu2_no_mul"

    @property
    def is_gated(self) -> bool:
        return not self.value.endswith("_no_mul")

    def without_mul(self) -> "MoEActivation":
        """Return the non-gated variant."""
        _without_mul: dict[MoEActivation, MoEActivation] = {
            MoEActivation.SILU: MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU: MoEActivation.GELU_NO_MUL,
            MoEActivation.RELU2: MoEActivation.RELU2_NO_MUL,
        }
        return _without_mul.get(self, self)

    @classmethod
    def from_str(cls, s: str) -> "MoEActivation":
        for member in cls:
            if member.value == s:
                return member
        valid = [m.value for m in cls]
        raise ValueError(f"Unknown MoE activation: {s!r}. Valid activations: {valid}")

    @staticmethod
    def adjust_N_for_activation(N: int, activation: "MoEActivation") -> int:
        """Return N for non-gated, N // 2 for gated activations."""
        return N if not activation.is_gated else N // 2


def apply_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor:
    """Apply MoE activation (pure PyTorch / FlagGems Triton)."""
    assert input.dim() == 2, "Input must be 2D"
    assert output.dim() == 2, "Output must be 2D"
    if activation.is_gated:
        assert output.size(-1) * 2 == input.size(
            -1
        ), f"{activation.value} expects 2x ratio: {output.size(-1) * 2} vs {input.size(-1)}"
    else:
        assert output.size(-1) == input.size(
            -1
        ), f"{activation.value} expects equal sizes: {output.size(-1)} vs {input.size(-1)}"

    if activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI):
        N = output.size(-1)
        x, y = input[:, :N], input[:, N:]
        _silu_and_mul_kernel(x, y, out0=output)
    elif activation == MoEActivation.GELU:
        N = output.size(-1)
        gate, up = input[:, :N], input[:, N:]
        output.copy_(F.gelu(gate) * up)
    elif activation == MoEActivation.SWIGLUSTEP:
        N = output.size(-1)
        gate, up = input[:, :N], input[:, N:]
        output.copy_(torch.sigmoid(gate) * up)
    elif activation == MoEActivation.RELU2:
        N = output.size(-1)
        gate, up = input[:, :N], input[:, N:]
        output.copy_(F.relu(gate).square() * up)

    elif activation == MoEActivation.SILU_NO_MUL:
        output.copy_(F.silu(input))
    elif activation == MoEActivation.GELU_NO_MUL:
        output.copy_(F.gelu(input))
    elif activation == MoEActivation.RELU2_NO_MUL:
        F.relu(input, inplace=True)
        torch.square(input, out=output)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    return output


def _fp8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    per_act_token: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 E4M3 quantization: per-tensor, per-token, or block-wise."""
    fp8_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(fp8_dtype)
    fp8_max = finfo.max
    fp8_min = finfo.min
    eps = 1e-10

    if block_shape is not None:
        assert not per_act_token
        assert len(block_shape) == 2
        block_k = block_shape[1]
        assert A.size(-1) % block_k == 0
        orig_shape = A.shape
        A_flat = A.reshape(-1, A.size(-1))
        M, K = A_flat.shape
        A_groups = A_flat.reshape(M * (K // block_k), block_k)
        amax = (
            A_groups.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        )
        scale = amax / fp8_max
        A_q = (A_groups.float() / scale).clamp(fp8_min, fp8_max).to(fp8_dtype)
        A_q = A_q.reshape(orig_shape)
        scale = scale.reshape(M, K // block_k)
        return A_q, scale

    elif per_act_token:
        A_flat = A.reshape(-1, A.size(-1))
        amax = A_flat.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        scale = amax / fp8_max
        min_scale = torch.tensor(
            1.0 / (fp8_max * 512.0), dtype=torch.float32, device=A.device
        )
        scale = scale.clamp(min=min_scale)
        A_q = (A_flat.float() / scale).clamp(fp8_min, fp8_max).to(fp8_dtype)
        A_q = A_q.reshape(A.shape)
        scale = scale.reshape(A.shape[:-1] + (1,))
        return A_q, scale

    else:
        if A_scale is not None:
            scale = (
                A_scale.float().view(1, 1) if A_scale.numel() == 1 else A_scale.float()
            )
            A_q = (A.float() / scale).clamp(fp8_min, fp8_max).to(fp8_dtype)
            return A_q, A_scale
        else:
            amax = A.abs().amax().clamp(min=eps).to(torch.float32)
            scale = amax / fp8_max
            iscale = 1.0 / scale
            A_q = (A.float() * iscale).clamp(fp8_min, fp8_max).to(fp8_dtype)
            return A_q, scale.view(1)


def _int8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    per_act_token: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """INT8 quantization: per-tensor, per-token, or block-wise."""
    iinfo = torch.iinfo(torch.int8)
    int8_max = iinfo.max
    int8_min = iinfo.min
    eps = 1e-10

    if block_shape is not None:
        assert not per_act_token
        assert len(block_shape) == 2
        block_k = block_shape[1]
        assert A.size(-1) % block_k == 0
        orig_shape = A.shape
        A_flat = A.reshape(-1, A.size(-1))
        M, K = A_flat.shape
        A_groups = A_flat.reshape(M * (K // block_k), block_k)
        amax = (
            A_groups.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        )
        scale = amax / int8_max
        A_q = (
            (A_groups.float() / scale).round().clamp(int8_min, int8_max).to(torch.int8)
        )
        A_q = A_q.reshape(orig_shape)
        scale = scale.reshape(M, K // block_k)
        return A_q, scale

    elif per_act_token:
        A_flat = A.reshape(-1, A.size(-1))
        amax = A_flat.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        scale = amax / int8_max
        A_q = (A_flat.float() / scale).round().clamp(int8_min, int8_max).to(torch.int8)
        A_q = A_q.reshape(A.shape)
        scale = scale.reshape(A.shape[:-1] + (1,))
        return A_q, scale

    else:
        assert A_scale is not None, "int8 per-tensor requires A_scale"
        scale = A_scale.float().view(1, 1) if A_scale.numel() == 1 else A_scale.float()
        A_q = (A.float() / scale).round().clamp(int8_min, int8_max).to(torch.int8)
        return A_q, A_scale


def moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    quant_dtype: None | torch.dtype | str,
    per_act_token_quant: bool,
    block_shape: Optional[list[int]] = None,
    ocp_mx_scheme: str | None = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Quantize MoE input activations before GEMM."""
    if ocp_mx_scheme is not None:
        if ocp_mx_scheme in {"w_mxfp4", "w_mxfp4_a_mxfp4"}:
            pass
        elif ocp_mx_scheme.endswith("a_fp8"):
            qA, qA_scale = _fp8_quantize(A, A_scale, per_act_token=False)
            A = (qA.float() * qA_scale.float()).to(A.dtype)
            return A, None

    if quant_dtype is None:
        return A, A_scale
    elif quant_dtype == torch.float8_e4m3fn:
        return _fp8_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == torch.int8:
        return _int8_quantize(A, A_scale, per_act_token_quant, block_shape)
    else:
        return A, A_scale


def _ensure_block_size_k_divisible(
    size_k: int, block_size_k: int, group_size: int
) -> int:
    """Find largest block_size_k that divides size_k and is divisible by group_size."""
    if size_k % block_size_k == 0 and block_size_k % group_size == 0:
        return block_size_k

    max_search = min(block_size_k, size_k)
    start = (max_search // group_size) * group_size
    for candidate in range(start, group_size - 1, -group_size):
        if size_k % candidate == 0:
            return candidate

    if size_k % group_size == 0:
        return group_size

    return size_k


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _silu_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_silu = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return x_silu * y


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    block_k_diviable: tl.constexpr,
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
):
    """Fused MoE kernel for GPTQ/AWQ (WNA16) quantized weights."""
    # Map pid to C block (grouped ordering for L2 reuse)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for first blocks of A and B
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    # Cast to int64 to prevent overflow in stride*offset products
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    if use_int4_w4a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # Accumulate C block in fp32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[None, :] * stride_bsn
            + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
        )
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + (offs_bn[None, :] // 2) * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = (b_zp >> b_zp_shifter) & 0xF
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + offs_bn[None, :] * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # Write back output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Fused MoE kernel: token × expert GEMM with quantization support."""
    # Map pid to C block (grouped ordering for L2 reuse)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for first blocks of A and B
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,  # first element = pid_m
            num_valid_tokens,  # remaining elements = constant
        )
    offs_token = offs_token.to(tl.int64)  # prevent int32 overflow

    token_mask = offs_token < num_valid_tokens

    offs_token = tl.where(token_mask, offs_token, 0)

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # Expert not in current EP rank, write zeros
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    if EVEN_N:
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    else:
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    # Load B tiles in the natural (N, K) direction and transpose in-register:
    # on triton-ascend, tl.dot only lowers to the cube unit when the B tile
    # is loaded along its contiguous axis -- the transposed-tile pattern
    # scalarizes the whole GEMM (~0.1 TFLOPS vs ~100 TFLOPS, measured with
    # perf_gemm_micro.py on Ascend910B4-1).
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:  # block-wise
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        elif per_channel_quant:  # channel-wise
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        else:  # tensor-wise
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)
    if HAS_BIAS:
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
    # Accumulate C block in fp32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    k_total = K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Pre-compute remaining K for this iteration
        k_offset = k * BLOCK_SIZE_K
        k_remaining = k_total - k_offset
        # Use other=0.0 for proper tail block handling - compatible with all batch sizes
        if EVEN_K:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        if use_int8_w8a16:
            accumulator = tl.dot(a, tl.trans(b).to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += (
                    tl.dot(a, tl.trans(b)) * a_scale[:, None] * b_scale[None, :]
                )
            else:
                if use_fp8_w8a8:
                    accumulator = tl.dot(a, tl.trans(b), acc=accumulator)
                else:
                    accumulator += tl.dot(a, tl.trans(b))
        else:
            accumulator += tl.dot(a, tl.trans(b))
        # Update pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Dequantization
    if use_int8_w8a16:
        accumulator = accumulator * b_scale
    elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
        accumulator = accumulator * a_scale * b_scale

    if HAS_BIAS:
        accumulator += bias[None, :]

    # Router weight multiplication (must be in fp32)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    # Write back output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    if EVEN_N:
        c_mask = token_mask[:, None]
    else:
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_wna16_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: list[int] | None,
):
    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None or B_zp.ndim == 3
    assert block_shape is not None and block_shape[0] == 0

    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique,
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )
    config = config.copy()
    config.update(
        get_moe_wna16_block_config(
            config=config,
            use_moe_wna16_cuda=False,
            num_valid_tokens=num_tokens,
            size_k=A.size(1),
            size_n=B.size(1),
            num_experts=B.size(1),
            group_size=block_shape[1],
            real_top_k=top_k,
            block_size_m=config["BLOCK_SIZE_M"],
        )
    )

    fused_moe_kernel_gptq_awq[grid](
        A,
        B,
        C,
        B_scale,
        B_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        A.size(1),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        B_zp.stride(0) if B_zp is not None else 0,
        B_zp.stride(2) if B_zp is not None else 0,
        B_zp.stride(1) if B_zp is not None else 0,
        block_k_diviable=A.size(1) % config["BLOCK_SIZE_K"] == 0,
        group_size=block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        has_zp=B_zp is not None,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        **config,
    )


def invoke_fused_moe_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    block_shape: Optional[list[int]] = None,
    B_bias: torch.Tensor | None = None,
) -> None:
    """Launch the fused_moe_kernel Triton kernel."""
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
        assert block_shape is None or triton.cdiv(
            B.size(-2), block_shape[0]
        ) == B_scale.size(-2)
        assert block_shape is None or triton.cdiv(
            B.size(-1), block_shape[1]
        ) == B_scale.size(-1)
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    M = A.size(0)
    num_tokens = M * top_k
    if sorted_token_ids is not None:
        EM = sorted_token_ids.size(0)
        if A.size(0) < config["BLOCK_SIZE_M"]:
            EM = min(
                sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"]
            )
    else:
        EM = num_tokens * config["BLOCK_SIZE_M"]
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )
    HAS_BIAS = B_bias is not None

    config = config.copy()
    # The legacy kernel carries more live buffers than the lean fast-path
    # GEMM (scale handling, both mask patterns, ...). The fast-path tile
    # sizes (BN=256, BK=128) overflow UB here -- and note the w8a16/w4a16
    # callers dequantize weights and CLEAR their quant flags before
    # dispatching, so this guard must be unconditional. Clamp to the
    # empirically safe envelope, then shrink stages to fit UB.
    config["BLOCK_SIZE_N"] = min(config["BLOCK_SIZE_N"], 128)
    config["BLOCK_SIZE_K"] = min(config["BLOCK_SIZE_K"], 32)
    config["num_warps"] = min(config["num_warps"], 4)
    UB_LIMIT = 196608  # 192KB
    ub_per_stage = (
        config["BLOCK_SIZE_M"] * config["BLOCK_SIZE_K"] * 2
        + config["BLOCK_SIZE_K"] * config["BLOCK_SIZE_N"] * 2
        + config["BLOCK_SIZE_M"] * config["BLOCK_SIZE_N"] * 4
    )
    while (
        config["num_stages"] > 1
        and ub_per_stage * config["num_stages"] > UB_LIMIT * 0.7
    ):
        config["num_stages"] -= 1
    config["SPLIT_K"] = 1
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    if block_shape is not None:
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0], block_shape[1]))

    # When K/N divide evenly by the block sizes, drop the tail masks and the
    # `% N` address wraparound from the hot loop: on triton-ascend they block
    # vectorized/DMA lowering of the loads and stores (scalar fallback).
    EVEN_K = (B.size(2) % BLOCK_SIZE_K) == 0
    EVEN_N = (B.size(1) % config["BLOCK_SIZE_N"]) == 0

    fused_moe_kernel[grid](
        A,
        B,
        C,
        B_bias,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),  # N
        B.size(2),  # K
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_bias.stride(0) if B_bias is not None else 0,
        B_bias.stride(1) if B_bias is not None else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        per_channel_quant=per_channel_quant,
        naive_block_assignment=(sorted_token_ids is None),
        HAS_BIAS=HAS_BIAS,
        EVEN_K=EVEN_K,
        EVEN_N=EVEN_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        **config,
    )


def dispatch_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[list[int]] = None,
    B_bias: Optional[torch.Tensor] = None,
) -> None:
    """Dispatch to the appropriate fused MoE kernel based on quantization flags."""
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    # M = A.size(0)
    # num_tokens = M * top_k

    if False:
        # TODO: Other precision-specific implementations
        # use_fp8_w8a8,
        # use_int8_w8a8,
        # use_int8_w8a16,
        # use_int4_w4a16,
        pass
    if (use_int8_w8a16 or use_int4_w4a16) and (
        block_shape is not None and block_shape[1] > 0
    ):
        assert B_bias is None
        invoke_fused_moe_wna16_triton_kernel(
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_int8_w8a16,
            use_int4_w4a16,
            block_shape,
        )
    else:
        invoke_fused_moe_triton_kernel(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape,
            B_bias,
        )


# ---------------------------------------------------------------------------
# Ascend-local moe_align_block_size replacement.
#
# The shared 4-stage Triton implementation in
# flaggems_vllm.ops.moe_align_block_size miscompiles on
# triton-ascend: its stage4 computes the scatter rank with atomic_add followed
# by a dependent load of the same address, which yields wrong ranks -- most
# tokens are never scattered (silently dropped), and out-of-bounds ranks
# corrupt neighbouring tensors or kill the device with "DDR address of the
# MTE instruction is out of range". The three kernels below use no atomics at
# all: each expert owns a private, non-overlapping output region, so every
# store is race-free and deterministic.
# ---------------------------------------------------------------------------


@triton.jit
def _moe_align_count_kernel(
    topk_ids_ptr,
    counts_ptr,
    numel,
    BLOCK_TOKENS: tl.constexpr,
):
    """One program per expert: count tokens routed to this expert."""
    expert = tl.program_id(0)
    offs = tl.arange(0, BLOCK_TOKENS)
    cnt = 0
    for start in range(0, numel, BLOCK_TOKENS):
        token_offs = start + offs
        ids = tl.load(topk_ids_ptr + token_offs, mask=token_offs < numel, other=-1)
        cnt += tl.sum((ids == expert).to(tl.int32), axis=0)
    tl.store(counts_ptr + expert, cnt)


@triton.jit
def _moe_align_meta_kernel(
    counts_ptr,
    expert_offsets_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    expert_map_ptr,
    num_experts,
    block_size,
    numel,
    numel_sorted_token_ids,
    numel_expert_ids,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_FILL: tl.constexpr,
    VECTORIZE_META: tl.constexpr = False,
    BLOCK_E: tl.constexpr = 256,
    BLOCK_B: tl.constexpr = 32,
):
    """Single program: buffer init, aligned expert offsets and expert_ids.

    Fills sorted_token_ids with the padding marker (numel), zeroes
    expert_ids, then assigns each expert its aligned region and stamps the
    per-block expert ids. Runs before the scatter kernel on the same stream.
    """
    offs_fill = tl.arange(0, BLOCK_FILL)
    for start in range(0, numel_sorted_token_ids, BLOCK_FILL):
        fill_offs = start + offs_fill
        tl.store(
            sorted_token_ids_ptr + fill_offs,
            numel,
            mask=fill_offs < numel_sorted_token_ids,
        )
    for start in range(0, numel_expert_ids, BLOCK_FILL):
        fill_offs = start + offs_fill
        tl.store(expert_ids_ptr + fill_offs, 0, mask=fill_offs < numel_expert_ids)

    base = 0
    tl.store(expert_offsets_ptr, 0)
    if VECTORIZE_META:
        # Vectorized path: all experts at once (no per-expert serial loop --
        # that loop costs ~1ms at E=256). counts -> aligned -> inclusive
        # cumsum in registers, offsets stored in one shot.
        offs_e = tl.arange(0, BLOCK_E)
        emask = offs_e < num_experts
        cnts = tl.load(counts_ptr + offs_e, mask=emask, other=0)
        aligned = tl.cdiv(cnts, block_size) * block_size
        cum = tl.cumsum(aligned, axis=0)
        starts = cum - aligned
        tl.store(expert_offsets_ptr + 1 + offs_e, cum, mask=emask)
        total = tl.sum(aligned, axis=0)
        tl.store(num_tokens_post_pad_ptr, total)

        # expert_ids stamping: for each output block b, its expert is the
        # last e with starts[e] <= b*block_size. Chunked 2D compare+sum
        # (expert_map applied at store time).
        offs_b = tl.arange(0, BLOCK_B)
        for start in range(0, numel_expert_ids, BLOCK_B):
            blk = start + offs_b
            bmask = blk < numel_expert_ids
            blk_start = blk * block_size
            # count of experts whose region starts at or before this block
            exp_idx = (
                tl.sum((starts[None, :] <= blk_start[:, None]).to(tl.int32), axis=1) - 1
            )
            in_region = bmask & (blk_start < total)
            if HAS_EXPERT_MAP:
                exp_idx = tl.load(expert_map_ptr + exp_idx, mask=in_region, other=0)
            tl.store(expert_ids_ptr + blk, exp_idx, mask=in_region)
    else:
        for e in range(num_experts):
            cnt = tl.load(counts_ptr + e)
            aligned = tl.cdiv(cnt, block_size) * block_size
            for i in range(base, base + aligned, block_size):
                ei = e
                if HAS_EXPERT_MAP:
                    ei = tl.load(expert_map_ptr + e)
                tl.store(expert_ids_ptr + i // block_size, ei)
            base += aligned
            tl.store(expert_offsets_ptr + 1 + e, base)
        tl.store(num_tokens_post_pad_ptr, base)


@triton.jit
def _moe_align_scatter_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_offsets_ptr,
    numel,
    BLOCK_TOKENS: tl.constexpr,
):
    """One program per expert: scatter my tokens into my private region.

    Regions are disjoint by construction, so no atomics are needed; the
    running rank is a plain per-program scalar carried across chunks.
    """
    expert = tl.program_id(0)
    base = tl.load(expert_offsets_ptr + expert)
    offs = tl.arange(0, BLOCK_TOKENS)
    rank = 0
    for start in range(0, numel, BLOCK_TOKENS):
        token_offs = start + offs
        ids = tl.load(topk_ids_ptr + token_offs, mask=token_offs < numel, other=-1)
        match = ids == expert
        pos = tl.cumsum(match.to(tl.int32), axis=0) - 1 + rank
        tl.store(
            sorted_token_ids_ptr + base + pos,
            token_offs.to(tl.int32),
            mask=match,
        )
        rank += tl.sum(match.to(tl.int32), axis=0)


def _moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Ascend-local drop-in replacement for moe_align_block_size.

    Same output contract: tokens grouped by expert with each region padded
    up to a multiple of block_size, padding slots hold numel, trailing
    expert_ids blocks hold 0, expert_map (EP) is applied to expert_ids.
    """
    numel = topk_ids.numel()
    device = topk_ids.device
    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    counts = torch.empty((num_experts,), dtype=torch.int32, device=device)
    expert_offsets = torch.empty((num_experts + 1,), dtype=torch.int32, device=device)

    BLOCK_TOKENS = 2048
    BLOCK_FILL = 4096
    _moe_align_count_kernel[(num_experts,)](
        topk_ids, counts, numel, BLOCK_TOKENS=BLOCK_TOKENS
    )
    _moe_align_meta_kernel[(1,)](
        counts,
        expert_offsets,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map,
        num_experts,
        block_size,
        numel,
        max_num_tokens_padded,
        max_num_m_blocks,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_FILL=BLOCK_FILL,
        VECTORIZE_META=True,
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK_B=32,
    )
    _moe_align_scatter_kernel[(num_experts,)](
        topk_ids,
        sorted_token_ids,
        expert_offsets,
        numel,
        BLOCK_TOKENS=BLOCK_TOKENS,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_pad


# ---------------------------------------------------------------------------
# Fast path for the unquantized fused MoE on Ascend.
#
# Measured on Ascend910B4-1 (perf_gemm_micro.py): tl.dot only lowers to the
# cube unit when BOTH operands' tiles are loaded in their contiguous
# direction AND addresses are affine -- gathering A rows through
# sorted_token_ids (the classic vLLM pattern) drops the kernel to scalar
# fallback (~0.1 TFLOPS vs ~105 TFLOPS). So the fast path materializes A in
# sorted order with a cheap gather kernel first (like the vendor's
# npu_moe_init_routing), runs a lean all-affine GEMM, and finally
# unsorts+reduces with inv_perm instead of moe_sum.
# ---------------------------------------------------------------------------


@triton.jit
def _moe_side_tables_kernel(
    sorted_token_ids_ptr,
    w_ptr,
    w_sorted_ptr,
    inv_perm_ptr,
    EM,
    num_valid,
    BLOCK: tl.constexpr,
):
    """Build w_sorted and inv_perm with vector masked stores only.

    The sorted layout is PADDED: valid entries sit at expert-region starts,
    padding slots hold a marker >= num_valid. Iterate over the full padded
    range [0, EM) and only touch rows whose flat id is a real token:

    w_sorted[pos] = topk_weights_flat[sorted_token_ids[pos]]   (valid rows)
    inv_perm[sorted_token_ids[pos]] = pos                      (valid rows)

    NOTE(ascend): these used to be scalar stores inside an `if nb == 0`
    branch of the gather kernel; triton-ascend silently dropped them, so
    they live here in the plain vector-store form proven in the align
    kernels.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < EM
    flat = tl.load(sorted_token_ids_ptr + offs, mask=in_range, other=-1)
    valid = in_range & (flat >= 0) & (flat < num_valid)
    flat_safe = tl.where(valid, flat, 0)
    w = tl.load(w_ptr + flat_safe, mask=valid, other=0.0)
    tl.store(w_sorted_ptr + offs, w, mask=valid)
    tl.store(inv_perm_ptr + flat_safe, offs.to(tl.int32), mask=valid)


@triton.jit
def _moe_gather_a_kernel(
    a_ptr,
    w_ptr,
    a_sorted_ptr,
    sorted_token_ids_ptr,
    num_valid,
    stride_am,
    stride_ak,
    K,
    top_k: tl.constexpr,
    APPLY_WEIGHT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Gather rows of A into routing-sorted (PADDED) order.

    Rows cover the full padded range [0, EM); padding rows (flat id >=
    num_valid) are left unwritten and are masked out in the GEMM.
    a_sorted[pos] = a[sorted_token_ids[pos] // top_k]  (optionally * weight)
    """
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(K, BLOCK_N)
    row = pid // num_blocks_n
    nb = pid % num_blocks_n

    flat_id = tl.load(sorted_token_ids_ptr + row)
    valid = flat_id < num_valid
    src_row = tl.where(valid, flat_id // top_k, 0).to(tl.int64)

    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < K
    # Load only for valid rows; padding rows get zeros so the GEMM's K loop
    # can load A completely unmasked (a row mask in the hot loop roughly
    # halves cube throughput on triton-ascend).
    val = tl.load(
        a_ptr + src_row * stride_am + offs_n * stride_ak,
        mask=n_mask & valid,
        other=0.0,
    )
    if APPLY_WEIGHT:
        w = tl.load(w_ptr + tl.where(valid, flat_id, 0), mask=valid, other=0.0)
        val = (val.to(tl.float32) * w.to(tl.float32)).to(val.dtype)
    tl.store(a_sorted_ptr + row.to(tl.int64) * K + offs_n, val, mask=n_mask)


@triton.jit
def _moe_silu_mul_kernel(
    x_ptr,
    out_ptr,
    D,
    BLOCK_N: tl.constexpr,
):
    """out[row] = silu(x[row, :D]) * x[row, D:2D] for x of shape (R, 2D).

    Fast-path replacement for the generic pointwise_dynamic silu: one wide
    vector load per operand, fp32 math.
    """
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(D, BLOCK_N)
    row = pid // num_blocks_n
    nb = pid % num_blocks_n
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < D
    base = x_ptr + row.to(tl.int64) * (2 * D)
    gate = tl.load(base + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    up = tl.load(base + D + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    res = gate * tl.sigmoid(gate) * up
    tl.store(
        out_ptr + row.to(tl.int64) * D + offs_n,
        res.to(out_ptr.dtype.element_ty),
        mask=n_mask,
    )


# Persistent grouped GEMM: cap the grid at the AIC core count (20 on 910B4)
# and let each program walk multiple (m, n) tiles with a stride (empty padded
# blocks never launch; exactly total_tiles programs when there are fewer
# tiles than cores).
_MOE_PERSISTENT_PROGS = 20

# 2x unroll of the K loop in the persistent GEMM (non-fused path only).
# Measured ~+3% on the repro kernel.
_MOE_UNROLL_K = tl.constexpr(True)


@triton.jit
def _fused_moe_gemm_persistent_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    w_sorted_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    FUSE_SILU: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    compute_type: tl.constexpr,
    IDX64: tl.constexpr,
    UNROLL_K: tl.constexpr,
):
    """Persistent grouped GEMM: the grid is capped at
    the AIC core count (20 on 910B4) and each program walks the (m, n) tiles
    with a num_programs stride. Tiles are visited in the same M-major group
    order as the classic kernel, so consecutive tiles inside a program share
    the A strip (L2 serves the reuse) and empty padded blocks never launch.
    """
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    total_m_blocks = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    if FUSE_SILU:
        N_OUT = N // 2
    else:
        N_OUT = N
    num_pid_n = tl.cdiv(N_OUT, BLOCK_SIZE_N)
    total_tiles = total_m_blocks * num_pid_n
    num_progs = tl.num_programs(0)
    pid0 = tl.program_id(0)

    for tile in range(pid0, total_tiles, num_progs):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(total_m_blocks - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile % num_pid_in_group) % group_size_m)
        pid_n = (tile % num_pid_in_group) // group_size_m

        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        sorted_ids = tl.load(sorted_token_ids_ptr + offs_m)
        token_mask = sorted_ids < num_valid_tokens
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        if IDX64:
            offs_am = offs_m.to(tl.int64)
        else:
            offs_am = offs_m
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        if EVEN_N:
            offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        else:
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_OUT
        b_base = b_ptr + off_experts * stride_be
        b_ptrs = b_base + offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk
        if FUSE_SILU:
            b_ptrs_up = b_ptrs + N_OUT * stride_bn

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        if FUSE_SILU:
            accumulator_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        k_total = K
        if FUSE_SILU:
            # fused path untouched (UB pressure already high with two B tiles)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                k_remaining = k_total - k * BLOCK_SIZE_K
                if EVEN_K:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                    b_up = tl.load(b_ptrs_up)
                else:
                    a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
                    b = tl.load(b_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
                    b_up = tl.load(
                        b_ptrs_up, mask=offs_k[None, :] < k_remaining, other=0.0
                    )
                accumulator = tl.dot(a, tl.trans(b), acc=accumulator)
                accumulator_up = tl.dot(a, tl.trans(b_up), acc=accumulator_up)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk
                b_ptrs_up += BLOCK_SIZE_K * stride_bk
        elif UNROLL_K:
            # 2x unrolled K loop: two A/B tiles per iteration. Uses ceil
            # pairs so the odd remainder is covered by the last pair's
            # masks (kr2 <= 0 -> all-false -> zero-tile dot). No runtime `if`
            # and no loop-tail loads -- those broke triton-ascend's
            # TritonToLinalgIncubated pass on some shapes.
            num_iters = tl.cdiv(K, BLOCK_SIZE_K)
            num_pairs = (num_iters + 1) // 2
            a_p = a_ptrs
            b_p = b_ptrs
            for _p in range(0, num_pairs):
                kr1 = k_total - _p * (2 * BLOCK_SIZE_K)
                kr2 = kr1 - BLOCK_SIZE_K
                a1 = tl.load(a_p, mask=offs_k[None, :] < kr1, other=0.0)
                b1 = tl.load(b_p, mask=offs_k[None, :] < kr1, other=0.0)
                a2 = tl.load(
                    a_p + BLOCK_SIZE_K * stride_ak,
                    mask=offs_k[None, :] < kr2,
                    other=0.0,
                )
                b2 = tl.load(
                    b_p + BLOCK_SIZE_K * stride_bk,
                    mask=offs_k[None, :] < kr2,
                    other=0.0,
                )
                accumulator = tl.dot(a1, tl.trans(b1), acc=accumulator)
                accumulator = tl.dot(a2, tl.trans(b2), acc=accumulator)
                a_p += 2 * BLOCK_SIZE_K * stride_ak
                b_p += 2 * BLOCK_SIZE_K * stride_bk
        else:
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                k_remaining = k_total - k * BLOCK_SIZE_K
                if EVEN_K:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                else:
                    a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
                    b = tl.load(b_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
                accumulator = tl.dot(a, tl.trans(b), acc=accumulator)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

        if FUSE_SILU:
            accumulator = accumulator * tl.sigmoid(accumulator) * accumulator_up

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(w_sorted_ptr + offs_m, mask=token_mask, other=0)
            accumulator *= moe_weight.to(tl.float32)[:, None]

        accumulator = accumulator.to(compute_type)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        if EVEN_N:
            c_mask = token_mask[:, None]
        else:
            c_mask = token_mask[:, None] & (offs_cn[None, :] < N_OUT)
        tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _moe_unsort_sum_kernel(
    c_ptr,
    inv_perm_ptr,
    w_sorted_ptr,
    out_ptr,
    K,
    top_k: tl.constexpr,
    APPLY_WEIGHT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """out[m] = sum_j w[pos_j] * c_sorted[pos_j], pos_j = inv_perm[m*top_k+j]

    (fp32 accumulation; w_sorted multiply only when APPLY_WEIGHT).
    """
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(K, BLOCK_N)
    token = pid // num_blocks_n
    nb = pid % num_blocks_n
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < K
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for j in tl.static_range(top_k):
        pos = tl.load(inv_perm_ptr + token * top_k + j).to(tl.int64)
        v = tl.load(c_ptr + pos * K + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        if APPLY_WEIGHT:
            w = tl.load(w_sorted_ptr + pos).to(tl.float32)
            v = v * w
        acc += v
    tl.store(
        out_ptr + token.to(tl.int64) * K + offs_n,
        acc.to(out_ptr.dtype.element_ty),
        mask=n_mask,
    )


# Configs whose fused-silu GEMM1 failed to compile (UB pressure); falls back
# to plain GEMM1 + _moe_silu_mul_kernel for them.
_FUSE_SILU_FAILED_CONFIGS = set()

# triton-ascend's UB accounting varies across CANN versions (a config that
# compiles on 9.0 can overflow on 9.1), so aggressive tile configs are
# probe-compiled once per shape and shrunk on failure instead of trusting
# the heuristics blindly.
_MOE_GEMM_RESOLVED_CONFIGS = {}


def _shrink_gemm_config(cfg):
    """One step smaller; returns None when nothing more to shrink."""
    cfg = dict(cfg)
    if cfg["BLOCK_SIZE_K"] > 128:
        cfg["BLOCK_SIZE_K"] = 128
    elif cfg["BLOCK_SIZE_M"] > 64:
        cfg["BLOCK_SIZE_M"] = 64
    elif cfg["BLOCK_SIZE_N"] > 128:
        cfg["BLOCK_SIZE_N"] = 128
    elif cfg["BLOCK_SIZE_K"] > 64:
        cfg["BLOCK_SIZE_K"] = 64
    elif cfg["num_warps"] > 4:
        cfg["num_warps"] = 4
    elif cfg["num_stages"] > 1:
        cfg["num_stages"] -= 1
    else:
        return None
    return cfg


def _probe_gemm_kernel(cfg, n, k, n_out, fuse_silu, dtype, compute_type, device):
    """Compile-only launch (npp=0 -> immediate return) to catch UB overflow
    for this exact (config, shape, dtype) combination."""
    BM = cfg["BLOCK_SIZE_M"]
    a = torch.zeros(BM, k, device=device, dtype=dtype)
    b = torch.zeros(1, n, k, device=device, dtype=dtype)
    c = torch.zeros(BM, n_out, device=device, dtype=dtype)
    ids = torch.zeros(BM, dtype=torch.int32, device=device)
    npp = torch.zeros(1, dtype=torch.int32, device=device)
    _fused_moe_gemm_persistent_kernel[(1,)](
        a,
        b,
        c,
        None,
        ids,
        ids,
        npp,
        n,
        k,
        BM,
        0,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        MUL_ROUTED_WEIGHT=False,
        EVEN_K=(k % cfg["BLOCK_SIZE_K"]) == 0,
        EVEN_N=(n_out % cfg["BLOCK_SIZE_N"]) == 0,
        FUSE_SILU=fuse_silu,
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        compute_type=compute_type,
        IDX64=False,
        UNROLL_K=_MOE_UNROLL_K,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def _resolve_gemm_config(cfg, gemm_shapes, dtype, compute_type, device):
    """Probe-compile the candidate config on all gemm shapes; shrink until it
    compiles. Result is cached per (config, shapes, dtype)."""
    key = (tuple(sorted(cfg.items())), tuple(gemm_shapes), str(dtype))
    if key in _MOE_GEMM_RESOLVED_CONFIGS:
        return _MOE_GEMM_RESOLVED_CONFIGS[key]
    while cfg is not None:
        try:
            for n, k, n_out in gemm_shapes:
                _probe_gemm_kernel(cfg, n, k, n_out, False, dtype, compute_type, device)
            break
        except Exception:
            cfg = _shrink_gemm_config(cfg)
    if cfg is None:
        raise RuntimeError("no compilable gemm config found for fused_moe")
    _MOE_GEMM_RESOLVED_CONFIGS[key] = cfg
    return cfg


# Below this many routed pairs, a single-program fused align (count + meta +
# scatter, one launch) beats the 3-kernel pipeline + separate side-tables
# kernel. NOTE the fused loop iterates experts serially in one program, so
# it only wins when E is small (E=256 measured 23ms vs 1ms for 3-kernel).
_MOE_ALIGN_SMALL_MAX_TOKENS = 256
_MOE_ALIGN_SMALL_MAX_EXPERTS = 64

# Below this many routed pairs, even the fused sort kernel is overkill: use
# one block per (token, expert) pair -- a single trivial kernel builds the
# padded sorted layout directly.
_MOE_ALIGN_NAIVE_MAX_TOKENS = 64

# num_valid <= this: E-parallel align (one program per expert; more
# parallelism wins for few routed pairs). Above it: chunked counting sort.
# 2048 keeps ds_m256 on the expert path (bench_ab measured it slightly
# better than the chunked path there).
_MOE_ALIGN_EXPERT_MAX_TOKENS = 2048


@triton.jit
def _moe_align_naive_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    w_ptr,
    w_sorted_ptr,
    inv_perm_ptr,
    numel,
    block_size,
    numel_sorted_token_ids,
    BLOCK_FILL: tl.constexpr,
):
    """One block per (token, expert) pair: expert_ids[i] = flat topk id,
    sorted_token_ids[i*block_size] = i, padding elsewhere. Tiny-input
    fast path; skips the whole count/meta/scatter pipeline. Side tables are
    trivial here: inv_perm[i] = i*block_size, w_sorted[i*block_size] = w[i]."""
    pid = tl.program_id(0)
    offs_fill = tl.arange(0, BLOCK_FILL)

    # grid-stride fill: keep the per-program vector small (BLOCK_FILL=4096
    # overflows UB via multi-buffering; 512 is safe)
    nprog = tl.num_programs(0)
    for start in range(pid * BLOCK_FILL, numel_sorted_token_ids, nprog * BLOCK_FILL):
        f = start + offs_fill
        tl.store(sorted_token_ids_ptr + f, numel, mask=f < numel_sorted_token_ids)

    offs = pid * BLOCK_FILL + offs_fill
    in_e = offs < numel
    ids = tl.load(topk_ids_ptr + offs, mask=in_e, other=0)
    tl.store(expert_ids_ptr + offs, ids, mask=in_e)
    tl.store(sorted_token_ids_ptr + offs * block_size, offs.to(tl.int32), mask=in_e)
    wv = tl.load(w_ptr + offs, mask=in_e, other=0.0)
    tl.store(w_sorted_ptr + offs * block_size, wv, mask=in_e)
    tl.store(inv_perm_ptr + offs, (offs * block_size).to(tl.int32), mask=in_e)
    if pid == 0:
        tl.store(num_tokens_post_pad_ptr, numel * block_size)


def _moe_align_naive(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """One-block-per-pair align for tiny inputs (decode path); also builds
    the w_sorted / inv_perm side tables in the same launch."""
    numel = topk_ids.numel()
    device = topk_ids.device
    max_num_tokens_padded = numel * block_size
    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=device
    )
    expert_ids = torch.empty((numel,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    w_sorted = torch.empty(
        (max_num_tokens_padded,), dtype=topk_weights.dtype, device=device
    )
    inv_perm = torch.empty((numel,), dtype=torch.int32, device=device)

    grid = (triton.cdiv(max(max_num_tokens_padded, numel), 512),)
    _moe_align_naive_kernel[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        topk_weights,
        w_sorted,
        inv_perm,
        numel,
        block_size,
        max_num_tokens_padded,
        BLOCK_FILL=512,
    )
    return (sorted_token_ids, expert_ids, num_tokens_post_pad, w_sorted, inv_perm)


@triton.jit
def _moe_align_small_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    w_ptr,
    w_sorted_ptr,
    inv_perm_ptr,
    num_experts,
    block_size,
    numel,
    numel_sorted_token_ids,
    numel_expert_ids,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_FILL: tl.constexpr,
):
    """Fused moe_align_block_size for small inputs (one launch, one program).

    Also emits the w_sorted / inv_perm side tables directly from register
    values (no readback of freshly-stored data -- that is unreliable on
    triton-ascend): inv_perm[flat] = pos and w_sorted[pos] = w[flat] are
    written with the same match mask/pos that place sorted_token_ids.

    Reuses only store patterns proven on triton-ascend (vector masked stores,
    scalar stores in per-expert loops).
    """
    offs_t = tl.arange(0, BLOCK_TOKENS)
    tmask = offs_t < numel
    ids = tl.load(topk_ids_ptr + offs_t, mask=tmask, other=-1)

    offs_fill = tl.arange(0, BLOCK_FILL)
    for start in range(0, numel_sorted_token_ids, BLOCK_FILL):
        f = start + offs_fill
        tl.store(sorted_token_ids_ptr + f, numel, mask=f < numel_sorted_token_ids)
    for start in range(0, numel_expert_ids, BLOCK_FILL):
        f = start + offs_fill
        tl.store(expert_ids_ptr + f, 0, mask=f < numel_expert_ids)

    base = 0
    for e in range(num_experts):
        match = (ids == e) & tmask
        cnt = tl.sum(match.to(tl.int32), axis=0)
        aligned = tl.cdiv(cnt, block_size) * block_size
        for i in range(base, base + aligned, block_size):
            tl.store(expert_ids_ptr + i // block_size, e)
        pos = tl.cumsum(match.to(tl.int32), axis=0) - 1 + base
        tl.store(sorted_token_ids_ptr + pos, offs_t.to(tl.int32), mask=match)
        # Side tables from registers: no readback of sorted_token_ids needed.
        wv = tl.load(w_ptr + offs_t, mask=match, other=0.0)
        tl.store(w_sorted_ptr + pos, wv, mask=match)
        tl.store(inv_perm_ptr + offs_t, pos.to(tl.int32), mask=match)
        base += aligned
    tl.store(num_tokens_post_pad_ptr, base)


def _moe_align_small(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """Single-launch align for small inputs (decode path); also builds the
    w_sorted / inv_perm side tables in the same launch."""
    numel = topk_ids.numel()
    device = topk_ids.device
    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    w_sorted = torch.empty(
        (max_num_tokens_padded,), dtype=topk_weights.dtype, device=device
    )
    inv_perm = torch.empty((numel,), dtype=torch.int32, device=device)

    _moe_align_small_kernel[(1,)](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        topk_weights,
        w_sorted,
        inv_perm,
        num_experts,
        block_size,
        numel,
        max_num_tokens_padded,
        max_num_m_blocks,
        BLOCK_TOKENS=triton.next_power_of_2(numel),
        BLOCK_FILL=4096,
    )
    return (sorted_token_ids, expert_ids, num_tokens_post_pad, w_sorted, inv_perm)


@triton.jit
def _moe_align_pcount_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    counts_ptr,
    numel,
    numel_sorted_token_ids,
    BLOCK_V: tl.constexpr,
    BLOCK_FILL: tl.constexpr,
):
    """counts[e] = number of routed pairs targeting expert e (one program per
    expert; num_valid is small so a single vector pass suffices). Also
    grid-stride fills sorted_token_ids with the marker: the gather kernel
    reads the full allocated range and would fault on uninitialized slots."""
    e = tl.program_id(0)
    offs = tl.arange(0, BLOCK_V)
    ids = tl.load(topk_ids_ptr + offs, mask=offs < numel, other=-1)
    cnt = tl.sum((ids == e).to(tl.int32), axis=0)
    tl.store(counts_ptr + e, cnt)

    nprog = tl.num_programs(0)
    offs_fill = tl.arange(0, BLOCK_FILL)
    for start in range(e * BLOCK_FILL, numel_sorted_token_ids, nprog * BLOCK_FILL):
        f = start + offs_fill
        tl.store(sorted_token_ids_ptr + f, numel, mask=f < numel_sorted_token_ids)


@triton.jit
def _moe_align_pmeta_kernel(
    counts_ptr,
    row_start_ptr,
    block_start_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    num_experts,
    block_size,
    max_num_m_blocks,
    BLOCK_E: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Single program, expert-count-sized vectors only (no per-token work):
    exclusive cumsums for row/block starts + the expert_ids block table."""
    offs_e = tl.arange(0, BLOCK_E)
    emask = offs_e < num_experts
    counts = tl.load(counts_ptr + offs_e, mask=emask, other=0)
    aligned = (tl.cdiv(counts, block_size) * block_size).to(tl.int32)
    row_start = tl.cumsum(aligned, axis=0) - aligned
    nblocks = tl.cdiv(counts, block_size)
    block_start = tl.cumsum(nblocks, axis=0) - nblocks
    tl.store(row_start_ptr + offs_e, row_start, mask=emask)
    tl.store(block_start_ptr + offs_e, block_start, mask=emask)
    total = tl.sum(aligned, axis=0)
    tl.store(num_tokens_post_pad_ptr, total)

    # expert_ids[b] = the expert owning block b: max e with block_start[e] <= b
    total_blocks = tl.sum(nblocks, axis=0)
    # use the register vector (no GM readback of freshly-stored data -- that
    # is unreliable on triton-ascend)
    bs = tl.where(emask, block_start, 2147483647)
    for b0 in range(0, max_num_m_blocks, BLOCK_B):
        offs_b = b0 + tl.arange(0, BLOCK_B)
        bmask = offs_b < total_blocks
        ge = offs_b[:, None] >= bs[None, :]
        expert = tl.sum(ge.to(tl.int32), axis=1) - 1
        tl.store(expert_ids_ptr + offs_b, expert, mask=bmask)


@triton.jit
def _moe_align_pscatter_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    row_start_ptr,
    counts_ptr,
    numel,
    BLOCK_M: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Scatter routed pairs into the padded sorted layout (one program per
    expert). Padding slots of each expert region are filled with the numel
    marker. NOTE: w_sorted/inv_perm stay in _moe_side_tables_kernel --
    mixing pos-addressed and offs-addressed stores in this kernel miscompiles
    on triton-ascend (measured: inv_perm half garbage while sorted_token_ids
    is fine)."""
    e = tl.program_id(0)
    offs = tl.arange(0, BLOCK_V)
    vmask = offs < numel
    ids = tl.load(topk_ids_ptr + offs, mask=vmask, other=-1)
    match = (ids == e) & vmask
    base = tl.load(row_start_ptr + e)
    pos = tl.cumsum(match.to(tl.int32), axis=0) - 1 + base
    tl.store(sorted_token_ids_ptr + pos, offs.to(tl.int32), mask=match)
    # fill this expert's padding slots with the marker
    cnt = tl.load(counts_ptr + e)
    aligned = tl.cdiv(cnt, BLOCK_M) * BLOCK_M
    offs_pad = tl.arange(0, BLOCK_M)
    tl.store(
        sorted_token_ids_ptr + base + cnt + offs_pad,
        numel,
        mask=(cnt + offs_pad) < aligned,
    )


def _moe_align_parallel_expert(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """E-parallel padded align for big-E small-token inputs (DSv3 decode).
    Side tables (w_sorted/inv_perm) are built by _moe_side_tables_kernel at
    the call site (the proven pattern; see the pscatter note above)."""
    numel = topk_ids.numel()
    device = topk_ids.device
    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    counts = torch.empty((num_experts,), dtype=torch.int32, device=device)
    row_start = torch.empty((num_experts,), dtype=torch.int32, device=device)
    block_start = torch.empty((num_experts,), dtype=torch.int32, device=device)

    block_v = triton.next_power_of_2(numel)
    _moe_align_pcount_kernel[(num_experts,)](
        topk_ids,
        sorted_token_ids,
        counts,
        numel,
        max_num_tokens_padded,
        BLOCK_V=block_v,
        BLOCK_FILL=512,
    )
    _moe_align_pmeta_kernel[(1,)](
        counts,
        row_start,
        block_start,
        expert_ids,
        num_tokens_post_pad,
        num_experts,
        block_size,
        max_num_m_blocks,
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK_B=64,
    )
    _moe_align_pscatter_kernel[(num_experts,)](
        topk_ids,
        sorted_token_ids,
        row_start,
        counts,
        numel,
        BLOCK_M=block_size,
        BLOCK_V=block_v,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_pad


@triton.jit
def _moe_align_chunkcount_kernel(
    topk_ids_ptr,
    chunk_counts_ptr,
    numel,
    num_experts,
    BLOCK_C: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Pass 1 of the chunked counting sort: one program per id-chunk,
    chunk_counts[c][e] = pairs of chunk c targeting expert e."""
    c = tl.program_id(0)
    offs = c * BLOCK_C + tl.arange(0, BLOCK_C)
    ids = tl.load(topk_ids_ptr + offs, mask=offs < numel, other=-1)
    for e0 in range(0, num_experts, BLOCK_E):
        offs_e = e0 + tl.arange(0, BLOCK_E)
        cnt = tl.sum((ids[:, None] == offs_e[None, :]).to(tl.int32), axis=0)
        # mask: offs_e beyond num_experts would clobber neighbouring chunks'
        # counts (and run past the buffer entirely for the last chunk)
        tl.store(
            chunk_counts_ptr + c * num_experts + offs_e, cnt, mask=offs_e < num_experts
        )


@triton.jit
def _moe_align_chunkmeta_kernel(
    chunk_counts_ptr,
    sorted_token_ids_ptr,
    counts_ptr,
    row_start_ptr,
    chunk_offs_ptr,
    block_start_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    num_experts,
    block_size,
    num_chunks,
    numel,
    numel_sorted_token_ids,
    max_num_m_blocks,
    BLOCK_E: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_FILL: tl.constexpr,
):
    """Pass 2 (single program): marker pre-fill, totals + padded row starts,
    block table, and chunk_offs[c][e] = padded region start of e plus the
    exclusive cumsum of counts over previous chunks."""
    # 1) pre-fill sorted_token_ids with the marker: the gather kernel reads
    # the full allocated range and would fault on uninitialized slots.
    offs_fill = tl.arange(0, BLOCK_FILL)
    for start in range(0, numel_sorted_token_ids, BLOCK_FILL):
        f = start + offs_fill
        tl.store(sorted_token_ids_ptr + f, numel, mask=f < numel_sorted_token_ids)

    # 2) totals + exclusive cumsums (register vectors only -- GM readback of
    # freshly-stored data is unreliable on triton-ascend)
    offs_e = tl.arange(0, BLOCK_E)
    emask = offs_e < num_experts
    counts = tl.zeros((BLOCK_E,), dtype=tl.int32)
    for c in range(0, num_chunks):
        counts += tl.load(
            chunk_counts_ptr + c * num_experts + offs_e, mask=emask, other=0
        )
    tl.store(counts_ptr + offs_e, counts, mask=emask)

    aligned = (tl.cdiv(counts, block_size) * block_size).to(tl.int32)
    row_start = tl.cumsum(aligned, axis=0) - aligned
    nblocks = tl.cdiv(counts, block_size)
    block_start = tl.cumsum(nblocks, axis=0) - nblocks
    tl.store(row_start_ptr + offs_e, row_start, mask=emask)
    tl.store(block_start_ptr + offs_e, block_start, mask=emask)
    total = tl.sum(aligned, axis=0)
    tl.store(num_tokens_post_pad_ptr, total)

    # 3) chunk_offs[c][e]: padded start of e + counts of chunks < c
    running = tl.zeros((BLOCK_E,), dtype=tl.int32)
    for c in range(0, num_chunks):
        tl.store(
            chunk_offs_ptr + c * num_experts + offs_e, row_start + running, mask=emask
        )
        running += tl.load(
            chunk_counts_ptr + c * num_experts + offs_e, mask=emask, other=0
        )

    # 4) expert_ids[b] = the expert owning block b
    total_blocks = tl.sum(nblocks, axis=0)
    bs = tl.where(emask, block_start, 2147483647)
    for b0 in range(0, max_num_m_blocks, BLOCK_B):
        offs_b = b0 + tl.arange(0, BLOCK_B)
        bmask = offs_b < total_blocks
        ge = offs_b[:, None] >= bs[None, :]
        expert = tl.sum(ge.to(tl.int32), axis=1) - 1
        tl.store(expert_ids_ptr + offs_b, expert, mask=bmask)


@triton.jit
def _moe_keyed_add(x, y):
    # associative_scan combine for packed (key << 12 | count): sums counts
    # while keys match, resets on a key change (flagtree routing pattern).
    # 0x7FFFF000 keeps bits 12-30 (our keys stay < 2^28); 0xFFFFF000 would
    # overflow int32.
    key_mask: tl.constexpr = 0x7FFFF000
    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


@triton.jit
def _moe_align_chunkscatter_kernel(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    chunk_offs_ptr,
    numel,
    num_experts,
    BLOCK_C: tl.constexpr,
):
    """Pass 3: one program per id-chunk. Sort the chunk's fp32 keys
    (expert * 4096 + local_index), derive local ranks via a keyed run-length
    scan, then store at chunk_offs[c][e] + rank. All 1D ops; the sort is
    small (BLOCK_C elements), unlike the reverted single-program full sort."""
    c = tl.program_id(0)
    offs = c * BLOCK_C + tl.arange(0, BLOCK_C)
    valid = offs < numel
    ids = tl.load(topk_ids_ptr + offs, mask=valid, other=0)
    kv = ids.to(tl.float32) * 4096.0 + tl.arange(0, BLOCK_C).to(tl.float32)
    kv = tl.where(valid, kv, 1e30)
    if _HAS_CANN_EXT:
        kv = al_ext.sort(kv, dim=0)
    else:
        kv = tl.sort(kv, 0)
    kv_i = tl.where(valid, kv, 0.0).to(tl.int32)
    s_e = kv_i >> 12
    s_local = kv_i & 4095
    # local rank of this pair inside its expert group within the chunk
    x = (kv_i & 0x7FFFF000) | 1
    run = tl.associative_scan(x, 0, _moe_keyed_add)
    local_rank = (run - 1) & 4095
    base = tl.load(chunk_offs_ptr + c * num_experts + s_e, mask=valid, other=0)
    tl.store(
        sorted_token_ids_ptr + base + local_rank,
        (c * BLOCK_C + s_local).to(tl.int32),
        mask=valid,
    )


def _moe_align_parallel(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """E-parallel padded align for big-E small-token inputs (DSv3 decode).
    Side tables (w_sorted/inv_perm) are built by _moe_side_tables_kernel at
    the call site (the proven pattern; see the pscatter note above).

    NOTE: a tl.sort-based scatter was tried and REVERTED (2026-09-18): a
    single-program sort of ~2K fp32 keys is slower than the E-parallel cumsum
    scans (DSv3 256-tok e2e 21.4ms -> 29.4ms). The extension sort pays off
    for small keys-per-block cases (topk), not here."""
    numel = topk_ids.numel()
    device = topk_ids.device
    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    counts = torch.empty((num_experts,), dtype=torch.int32, device=device)
    row_start = torch.empty((num_experts,), dtype=torch.int32, device=device)
    block_start = torch.empty((num_experts,), dtype=torch.int32, device=device)

    BLOCK_C = 256
    num_chunks = triton.cdiv(numel, BLOCK_C)
    chunk_counts = torch.empty(
        (num_chunks * num_experts,), dtype=torch.int32, device=device
    )
    chunk_offs = torch.empty(
        (num_chunks * num_experts,), dtype=torch.int32, device=device
    )

    _moe_align_chunkcount_kernel[(num_chunks,)](
        topk_ids,
        chunk_counts,
        numel,
        num_experts,
        BLOCK_C=BLOCK_C,
        BLOCK_E=32,
    )
    _moe_align_chunkmeta_kernel[(1,)](
        chunk_counts,
        sorted_token_ids,
        counts,
        row_start,
        chunk_offs,
        block_start,
        expert_ids,
        num_tokens_post_pad,
        num_experts,
        block_size,
        num_chunks,
        numel,
        max_num_tokens_padded,
        max_num_m_blocks,
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK_B=64,
        BLOCK_FILL=512,
    )
    _moe_align_chunkscatter_kernel[(num_chunks,)](
        topk_ids,
        sorted_token_ids,
        chunk_offs,
        numel,
        num_experts,
        BLOCK_C=BLOCK_C,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_pad


def _fused_experts_fast(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool,
    activation_enum: "MoEActivation",
    apply_router_weight_on_input: bool,
    get_config_func,
    compute_type,
) -> torch.Tensor:
    """Unquantized fast path: gather-sort -> GEMM1 -> act -> GEMM2 -> unsort+sum."""
    num_tokens = hidden_states.size(0)
    E, N, K1 = w1.size()
    K2 = w2.size(1)
    top_k_num = topk_ids.size(1)
    device = hidden_states.device
    dtype = hidden_states.dtype

    out_hidden_states = hidden_states if inplace else torch.empty_like(hidden_states)

    CHUNK_SIZE: int = 16 * 1024
    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        tokens_in_chunk = end_chunk_idx - begin_chunk_idx
        if tokens_in_chunk == 0:
            break

        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        config = get_config_func(tokens_in_chunk)

        num_valid = tokens_in_chunk * top_k_num

        # Resolve a compilable tile config BEFORE align: the align block size
        # must equal the GEMM's BLOCK_SIZE_M, and triton-ascend's UB
        # accounting varies across CANN versions, so probe-compile and shrink.
        act_dim = MoEActivation.adjust_N_for_activation(N, activation_enum)
        config = _resolve_gemm_config(
            config,
            [(N, K1, N), (K2, act_dim, K2)],
            dtype,
            compute_type,
            device,
        )

        # gather hidden rows into routing-sorted (padded) order; all
        # intermediate buffers are sized EM (the padded sorted length)
        _use_expert_align = tokens_in_chunk * top_k_num <= _MOE_ALIGN_EXPERT_MAX_TOKENS
        if (
            tokens_in_chunk * top_k_num <= _MOE_ALIGN_NAIVE_MAX_TOKENS
            and E > _MOE_ALIGN_SMALL_MAX_EXPERTS
        ):
            # tiny input: one block per (token, expert) pair, one launch
            # (side tables are fused into the align kernel)
            (
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                w_sorted,
                inv_perm,
            ) = _moe_align_naive(
                curr_topk_ids,
                curr_topk_weights,
                config["BLOCK_SIZE_M"],
                E,
            )
            EM = sorted_token_ids.numel()
        elif (
            tokens_in_chunk * top_k_num <= _MOE_ALIGN_SMALL_MAX_TOKENS
            and E <= _MOE_ALIGN_SMALL_MAX_EXPERTS
        ):
            # decode-scale, moderate E: one fused align launch
            (
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                w_sorted,
                inv_perm,
            ) = _moe_align_small(
                curr_topk_ids,
                curr_topk_weights,
                config["BLOCK_SIZE_M"],
                E,
            )
            EM = sorted_token_ids.numel()
        elif _use_expert_align:
            # small/mid: one program per expert (more parallelism wins for
            # few routed pairs)
            (
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
            ) = _moe_align_parallel_expert(
                curr_topk_ids,
                config["BLOCK_SIZE_M"],
                E,
            )
            EM = sorted_token_ids.numel()
            w_sorted = torch.empty(EM, device=device, dtype=topk_weights.dtype)
            inv_perm = torch.empty(num_valid, device=device, dtype=torch.int32)
            _moe_side_tables_kernel[(triton.cdiv(EM, 4096),)](
                sorted_token_ids,
                curr_topk_weights,
                w_sorted,
                inv_perm,
                EM,
                num_valid,
                BLOCK=4096,
            )
        else:
            # large num_valid: chunked counting sort (no upper limit; also
            # covers the big-prefill shapes that used to hit the slow general
            # path)
            (
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
            ) = _moe_align_parallel(
                curr_topk_ids,
                config["BLOCK_SIZE_M"],
                E,
            )
            EM = sorted_token_ids.numel()
            w_sorted = torch.empty(EM, device=device, dtype=topk_weights.dtype)
            inv_perm = torch.empty(num_valid, device=device, dtype=torch.int32)
            _moe_side_tables_kernel[(triton.cdiv(EM, 4096),)](
                sorted_token_ids,
                curr_topk_weights,
                w_sorted,
                inv_perm,
                EM,
                num_valid,
                BLOCK=4096,
            )

        a1_sorted = torch.empty(EM, K1, device=device, dtype=dtype)
        BLOCK_N = 4096
        grid = (EM * triton.cdiv(K1, BLOCK_N),)
        _moe_gather_a_kernel[grid](
            curr_hidden_states,
            curr_topk_weights,
            a1_sorted,
            sorted_token_ids,
            num_valid,
            curr_hidden_states.stride(0),
            curr_hidden_states.stride(1),
            K1,
            top_k=top_k_num,
            APPLY_WEIGHT=apply_router_weight_on_input,
            BLOCK_N=BLOCK_N,
        )

        def _gemm(a, b, c, w, mul_weight, fuse_silu=False, cfg=None):
            cfg = cfg or config
            n, k = b.size(1), b.size(2)
            n_out = n // 2 if fuse_silu else n
            # int64 row addressing only when the buffer can exceed int32
            idx64 = EM * max(a.stride(0), c.stride(0)) > 2**31 - 1
            # persistent: cap the grid at the AIC core count; each program
            # walks the tiles with a stride (empty padded blocks never
            # launch). Launch exactly total_tiles programs when there are
            # fewer tiles than cores.
            n_progs = min(
                _MOE_PERSISTENT_PROGS,
                triton.cdiv(EM, cfg["BLOCK_SIZE_M"])
                * triton.cdiv(n_out, cfg["BLOCK_SIZE_N"]),
            )
            _fused_moe_gemm_persistent_kernel[(n_progs,)](
                a,
                b,
                c,
                w,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                n,
                k,
                EM,
                num_valid,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                b.stride(2),
                c.stride(0),
                c.stride(1),
                MUL_ROUTED_WEIGHT=mul_weight,
                EVEN_K=(k % cfg["BLOCK_SIZE_K"]) == 0,
                EVEN_N=(n_out % cfg["BLOCK_SIZE_N"]) == 0,
                FUSE_SILU=fuse_silu,
                BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
                BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
                BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
                GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
                compute_type=compute_type,
                IDX64=idx64,
                UNROLL_K=_MOE_UNROLL_K,
                num_warps=cfg["num_warps"],
                num_stages=cfg["num_stages"],
            )

        # GEMM1 (+fused silu): (EM, K1) x w1 -> (EM, act_dim)
        if activation_enum == MoEActivation.SILU:
            cache2 = torch.empty(EM, act_dim, device=device, dtype=dtype)
            # Fusing silu doubles the accumulators; they must fit the 128KB
            # L0C: BM*BN*8 bytes for two fp32 accumulators. BM=128 x BN=256
            # overflows (measured cc overflow), BM=64 x BN=256 just fits.
            fuse_ok = config["BLOCK_SIZE_M"] * config["BLOCK_SIZE_N"] <= 16384
            fuse_key = (N, K1, str(dtype), tuple(sorted(config.items())))
            if fuse_ok and fuse_key not in _FUSE_SILU_FAILED_CONFIGS:
                try:
                    # The fused kernel runs two B tiles + two accumulators;
                    # BK beyond 128 overflows the 512KB cbuf at BN=256.
                    fused_config = {
                        **config,
                        "BLOCK_SIZE_K": min(config["BLOCK_SIZE_K"], 128),
                    }
                    _gemm(
                        a1_sorted,
                        w1,
                        cache2,
                        None,
                        False,
                        fuse_silu=True,
                        cfg=fused_config,
                    )
                except Exception:
                    # fused gemm1 doubles UB pressure (two accumulators + two
                    # B tiles); some shape/config combos fail to compile.
                    # Remember and fall back permanently for this key.
                    _FUSE_SILU_FAILED_CONFIGS.add(fuse_key)
            if not fuse_ok or fuse_key in _FUSE_SILU_FAILED_CONFIGS:
                cache1 = torch.empty(EM, N, device=device, dtype=dtype)
                _gemm(a1_sorted, w1, cache1, None, False)
                grid = (EM * triton.cdiv(act_dim, 2048),)
                _moe_silu_mul_kernel[grid](cache1, cache2, act_dim, BLOCK_N=2048)
        else:
            cache1 = torch.empty(EM, N, device=device, dtype=dtype)
            _gemm(a1_sorted, w1, cache1, None, False)
            cache2 = torch.empty(EM, act_dim, device=device, dtype=dtype)
            apply_moe_activation(activation_enum, cache2, cache1)

        # GEMM2: (EM, act_dim) x w2 -> (EM, K2)
        cache3 = torch.empty(EM, K2, device=device, dtype=dtype)
        _gemm(cache2, w2, cache3, w_sorted, not apply_router_weight_on_input)

        # unsort + sum over topk
        out_chunk = out_hidden_states[begin_chunk_idx:end_chunk_idx]
        BLOCK_OUT = 1024
        grid = (tokens_in_chunk * triton.cdiv(K2, BLOCK_OUT),)
        _moe_unsort_sum_kernel[grid](
            cache3,
            inv_perm,
            w_sorted,
            out_chunk,
            K2,
            top_k=top_k_num,
            # GEMM2 already multiplied the router weight in (tl.dot
            # epilogue), so the unsort+sum reduction never applies it.
            APPLY_WEIGHT=False,
            BLOCK_N=BLOCK_OUT,
        )
    return out_hidden_states


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("GEMS_ASCEND FUSED_MOE")
    if hasattr(activation, "value"):
        activation = activation.value
    assert (
        activation == "silu"
    ), f"Only 'silu' activation is supported, got {activation}"

    activation_enum = MoEActivation.from_str(activation)

    # Check constraints
    if use_int4_w4a16:
        # INT4 stored unpacked in INT8 containers (full K dim)
        assert hidden_states.size(1) == w1.size(
            2
        ), f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"
    elif ocp_mx_scheme is not None:
        if ocp_mx_scheme.startswith("w_mxfp4"):
            assert hidden_states.size(1) == w1.size(2) * 2, "hidden size mismatch"
        elif ocp_mx_scheme.startswith("w_mxfp6"):
            assert (
                hidden_states.size(1) == (w1.size(2) * 4) // 3
            ), "hidden size mismatch"
        else:
            raise NotImplementedError(f"Unsupported ocp_mx_scheme={ocp_mx_scheme}")
    else:
        assert hidden_states.size(1) == w1.size(
            2
        ), f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)

    CHUNK_SIZE: int = 16 * 1024
    M = min(num_tokens, CHUNK_SIZE)

    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=ocp_mx_scheme,
        dtype=hidden_states.dtype,
    )

    quant_dtype = _get_config_quant_dtype(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        ocp_mx_scheme=ocp_mx_scheme,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=block_shape,
    )

    config = get_config_func(M)

    # cache1 and cache3 share memory (non-overlapping lifetime)
    cache13 = torch.empty(
        M * top_k_num * max(N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)

    # cache2 needs separate memory (concurrent with cache1)
    activation_out_dim = MoEActivation.adjust_N_for_activation(N, activation_enum)
    intermediate_cache2 = torch.empty(
        (M * top_k_num, activation_out_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    # Fast path: unquantized, no EP map -> gather-sort + all-affine GEMMs.
    # (gathering A rows through sorted_token_ids inside the GEMM kernel
    # prevents triton-ascend from lowering tl.dot to the cube unit, so the
    # legacy kernel path below is only kept for quantized/EP cases.)
    no_quant = not (
        use_fp8_w8a8
        or use_int8_w8a8
        or use_int8_w8a16
        or use_int4_w4a16
        or ocp_mx_scheme is not None
        or block_shape is not None
    )
    if no_quant and expert_map is None and w1_bias is None and w2_bias is None:
        return _fused_experts_fast(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace,
            activation_enum,
            apply_router_weight_on_input,
            get_config_func,
            compute_type,
        )

    out_hidden_states = hidden_states if inplace else torch.empty_like(hidden_states)

    if ocp_mx_scheme is not None:
        # Dequantize OCP MX weights (TODO: skip on platforms with native MX)
        if ocp_mx_scheme.startswith("w_mxfp4"):
            w1 = dequant_mxfp4(w1, w1_scale, hidden_states.dtype)
            w1_scale = None
            w2 = dequant_mxfp4(w2, w2_scale, hidden_states.dtype)
            w2_scale = None
        elif ocp_mx_scheme.startswith("w_mxfp6_e3m2"):
            w1 = dequant_mxfp6(
                w1, w1_scale, quant_dtype="fp6_e3m2", float_dtype=hidden_states.dtype
            )
            w1_scale = None
            w2 = dequant_mxfp6(
                w2, w2_scale, quant_dtype="fp6_e3m2", float_dtype=hidden_states.dtype
            )
            w2_scale = None
        elif ocp_mx_scheme.startswith("w_mxfp6_e2m3"):
            w1 = dequant_mxfp6(
                w1, w1_scale, quant_dtype="fp6_e2m3", float_dtype=hidden_states.dtype
            )
            w1_scale = None
            w2 = dequant_mxfp6(
                w2, w2_scale, quant_dtype="fp6_e2m3", float_dtype=hidden_states.dtype
            )
            w2_scale = None
        else:
            raise NotImplementedError(f"Unsupported ocp_mx_scheme={ocp_mx_scheme}")

    # Dequant INT8/INT4 weights (Triton can't do mixed-dtype dot)
    if use_int8_w8a16 or use_int4_w4a16:
        w1 = w1.to(hidden_states.dtype) * w1_scale.unsqueeze(-1).to(hidden_states.dtype)
        w1_scale = None
        w2 = w2.to(hidden_states.dtype) * w2_scale.unsqueeze(-1).to(hidden_states.dtype)
        w2_scale = None
        use_int8_w8a16 = False
        use_int4_w4a16 = False

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.size()

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust cache size for last chunk
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[
                : tokens_in_chunk * topk_ids.size(1)
            ]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
            A=curr_hidden_states,
            A_scale=a1_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
            ocp_mx_scheme=ocp_mx_scheme,
        )

        SPARSITY_FACTOR = 4
        # For small tokens (< 32), always use naive assignment to skip alignment overhead
        use_naive_small = tokens_in_chunk < 32
        naive_block_assignment = use_naive_small or (
            expert_map is None
            and tokens_in_chunk * top_k_num * SPARSITY_FACTOR <= global_num_experts
            and not (
                (use_int8_w8a16 or use_int4_w4a16)
                and block_shape is not None
                and block_shape[1] > 0
            )
        )
        # FIXME(ascend): the naive path makes fused_moe_kernel emit select-based
        # offs_token/masks that crash triton-ascend's DiscreteMaskAccessConversion
        # pass (MLIRCompilationError: PassManager::run failed). Force the
        # moe_align_block_size path until the compiler issue is resolved.
        naive_block_assignment = False

        if not naive_block_assignment:
            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                _moe_align_block_size(
                    curr_topk_ids,
                    config["BLOCK_SIZE_M"],
                    global_num_experts,
                    expert_map,
                )
            )
        else:
            max_num_tokens_padded = topk_ids.numel() * config["BLOCK_SIZE_M"]
            expert_ids = curr_topk_ids.view(-1)
            num_tokens_post_padded = torch.empty(
                (1), dtype=torch.int32, device=topk_ids.device
            )
            num_tokens_post_padded.fill_(max_num_tokens_padded)
            sorted_token_ids = None

        dispatch_fused_moe_kernel(
            qcurr_hidden_states,
            w1,
            intermediate_cache1,
            a1q_scale,
            w1_scale,
            w1_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            top_k_num,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w1_bias,
        )

        apply_moe_activation(
            activation_enum, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=a2_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
            ocp_mx_scheme=ocp_mx_scheme,
        )

        if expert_map is not None:
            intermediate_cache3.zero_()

        dispatch_fused_moe_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w2_bias,
        )

        moe_sum(
            intermediate_cache3.view(*intermediate_cache3.size()),
            out_hidden_states[begin_chunk_idx:end_chunk_idx],
        )

    return out_hidden_states


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> None:
    """
    In-place fused MoE: writes output directly into ``hidden_states``.

    Same semantics as ``fused_experts_impl(..., inplace=True)``.
    Returns None (the result is stored in ``hidden_states``).
    """
    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=True,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Out-of-place fused MoE: allocates and returns a new output tensor.

    Same semantics as ``fused_experts_impl(..., inplace=False)``.
    """
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )
