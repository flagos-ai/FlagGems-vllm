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

# The YaRN helpers and DeepSeek cache construction are adapted from DeepSeek-R1:
# https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/modeling_deepseek.py

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from flaggems_vllm.ops.rotary_embedding import apply_rotary_pos_emb

logger = logging.getLogger(__name__)

__all__ = [
    "gems_rope_forward",
    "GemsDeepseekYarnRoPE",
    "GemsRope",
]


def gems_rope_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
    inplace: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings with the FlagGems-vLLM operator."""
    tensors = {"key": key, "cos": cos, "sin": sin}
    if position_ids is not None:
        tensors["position_ids"] = position_ids
    for name, tensor in tensors.items():
        if tensor.device != query.device:
            raise ValueError(
                f"query and {name} must be on the same device, "
                f"got {query.device} and {tensor.device}"
            )

    if position_ids is not None and position_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("position_ids must have dtype torch.int32 or torch.int64")
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, cos, sin)
    ):
        raise NotImplementedError("gems_rope_forward is an inference-only operation")

    if query.ndim < 2 or key.ndim < 2:
        raise ValueError("query and key must each have at least two dimensions")
    if query.shape[:-2] != key.shape[:-2] or query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "query and key must have matching token dimensions and head dimensions, "
            f"got {tuple(query.shape)} and {tuple(key.shape)}"
        )
    if cos.ndim != 2 or sin.ndim != 2 or cos.shape != sin.shape:
        raise ValueError(
            "cos and sin must be two-dimensional tensors with identical shapes, "
            f"got {tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    if cos.shape[-1] * 2 != query.shape[-1]:
        raise ValueError(
            "the last cos/sin dimension must be half the query/key head dimension, "
            f"got {cos.shape[-1]} and {query.shape[-1]}"
        )

    logger.debug("GEMS CUSTOM ROPE FORWARD")
    return apply_rotary_pos_emb(
        query,
        key,
        cos,
        sin,
        position_ids,
        rotary_interleaved,
        inplace,
    )


class GemsRope(nn.Module):
    """Vanilla rotary position embedding with a precomputed cos/sin cache.

    ``rotary_interleaved=True`` rotates adjacent pairs (GPT-J layout), while
    ``False`` pairs the first and second halves (GPT-NeoX layout).
    """

    def __init__(
        self,
        rotary_dim,
        max_position_embeddings,
        base,
        rotary_interleaved,
        dtype,
        device,
    ):
        super().__init__()
        if rotary_dim <= 0 or rotary_dim % 2 != 0:
            raise ValueError(
                f"rotary_dim must be a positive even integer, got {rotary_dim}"
            )
        if max_position_embeddings <= 0:
            raise ValueError(
                "max_position_embeddings must be positive, "
                f"got {max_position_embeddings}"
            )
        if base <= 0:
            raise ValueError(f"base must be positive, got {base}")

        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rotary_interleaved = rotary_interleaved
        self.dtype = dtype
        self.device = device
        self._set_cos_sin_cache()

    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute the inverse frequencies with shape ``[rotary_dim / 2]``."""
        return 1.0 / (
            self.base
            ** (
                torch.arange(
                    0, self.rotary_dim, 2, dtype=torch.float32, device=self.device
                )
                / self.rotary_dim
            )
        )

    def _set_cos_sin_cache(self) -> None:
        # Build this immutable module buffer once, outside the operator forward
        # path. Runtime rotary computation remains in the Triton kernel.
        inv_freq = self._compute_inv_freq()
        positions = torch.arange(
            self.max_position_embeddings,
            device=self.device,
            dtype=torch.float32,
        )
        freqs = torch.outer(positions, inv_freq)

        self.register_buffer("cos_cached", freqs.cos().to(self.dtype), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().to(self.dtype), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: Optional[torch.IntTensor] = None,
        inplace: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "cos_cached") or not hasattr(self, "sin_cached"):
            self._set_cos_sin_cache()

        return gems_rope_forward(
            query,
            key,
            self.cos_cached,
            self.sin_cached,
            position_ids,
            self.rotary_interleaved,
            inplace,
        )


def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    """Find the dimension corresponding to a requested number of rotations."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    """Find and clamp the YaRN correction range."""
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    return torch.clamp(linear_func, 0, 1)


class GemsDeepseekYarnRoPE(GemsRope):
    """YaRN rotary position embedding used by DeepSeek models."""

    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        rotary_interleaved: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ):
        if scaling_factor <= 0:
            raise ValueError(f"scaling_factor must be positive, got {scaling_factor}")
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            rotary_interleaved=rotary_interleaved,
            dtype=dtype,
            device=device,
        )

    def _compute_inv_freq(self) -> torch.Tensor:
        freq_extra = 1.0 / (
            self.base
            ** (
                torch.arange(
                    0, self.rotary_dim, 2, dtype=torch.float32, device=self.device
                )
                / self.rotary_dim
            )
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (
                torch.arange(
                    0, self.rotary_dim, 2, dtype=torch.float32, device=self.device
                )
                / self.rotary_dim
            )
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, self.rotary_dim // 2).to(
            device=self.device, dtype=torch.float32
        )
        return freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

    def _set_cos_sin_cache(self) -> None:
        # YaRN cache construction is one-time module setup, not a numerical
        # fallback for the runtime rotary operator.
        inv_freq = self._compute_inv_freq()
        positions = torch.arange(
            self.max_position_embeddings,
            device=self.device,
            dtype=torch.float32,
        )
        freqs = torch.outer(positions, inv_freq)
        mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        self.register_buffer(
            "cos_cached", (freqs.cos() * mscale).to(self.dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (freqs.sin() * mscale).to(self.dtype), persistent=False
        )
