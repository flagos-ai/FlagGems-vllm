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

import contextlib
import threading
from typing import Any

import flaggems_vllm.ops.fused_moe as generic_fused_moe

from .moe_sum import moe_sum as mthreads_moe_sum

_PATCH_LOCK = threading.RLock()
_GENERIC_GET_DEFAULT_CONFIG = generic_fused_moe.get_default_config
_PLAIN_HALF_CONFIG_DTYPES = ("fp16", "bf16")

# Target model shape families: (w1.shape, w2.shape, topk).
#   Qwen3-235B-A22B TP shards   : E=256, topk=8
#   Qwen3-Max style             : E=512, topk=10
#   DeepSeek-V3 TP8             : E=256, topk=8, hidden=7168
_TARGET_SHAPES = (
    # Qwen3 family
    ((256, 1024, 2048), (256, 2048, 512), 8),
    ((256, 256, 2048), (256, 2048, 128), 8),
    ((512, 2048, 4096), (512, 4096, 1024), 10),
    # DeepSeek-V3 TP8
    ((256, 4096, 7168), (256, 7168, 2048), 8),
)

# MTT S5000 config sweep results (2026-08-04, 2026-08-05): (E, topk) -> (M threshold, config).
# 2026-08-05: small-M (M<=16) configs added — BM16/BN64/BK32 single-stage wins for
# sparse token-expert mapping (small-M tile padding dominates otherwise).
# 16/128/64 single-stage tiles win for high-expert-count MoE (E=256/512);
# E=8 (Mixtral-like) prefers BM=32 for small M and BM=64 beyond that.
_MTHREADS_TUNED_CONFIGS = {
    (8, 2): (
        (
            1,
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 1,
                "SWAP_AB": True,
            },
        ),
        (
            16,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 1,
            },
        ),
        (
            128,
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 1,
            },
        ),
        (
            float("inf"),
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 8,
                "num_stages": 1,
            },
        ),
    ),
    (256, 8): (
        (
            1,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 1,
                "SWAP_AB": True,
            },
        ),
        (
            16,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 1,
            },
        ),
        (
            128,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 1,
            },
        ),
        # M>128: GEMM2 (N>512, main table) must keep BLOCK_SIZE_M aligned with
        # the GEMM1 Qwen3.6 table (BM64) — moe_align_block_size uses
        # base_config["BLOCK_SIZE_M"], so GEMM2 BM < GEMM1 BM makes the GEMM2
        # grid (cdiv(EM, BM2)) outrun expert_ids (padded by BM1) -> OOB IMA.
        (
            float("inf"),
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
                "num_warps": 8,
                "num_stages": 1,
            },
        ),
    ),
    (512, 10): (
        (
            float("inf"),
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 1,
            },
        ),
    ),
}

# FP8 W8A8 (per-tensor) tuned configs, MTT S5000 (2026-08-06 sweep):
# fp8 prefers BN=64/BK=64 (vs BK=32 for plain half); small-M only — larger M
# keeps the generic (MUSA-safe) config.
_MTHREADS_TUNED_CONFIGS_FP8 = {
    (8, 2): (
        (
            16,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 1,
            },
        ),
        (
            64,
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 2,
                "num_warps": 4,
                "num_stages": 1,
            },
        ),
    ),
    (256, 8): (
        (
            16,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 1,
            },
        ),
        (
            128,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 1,
            },
        ),
        # M>128 (Qwen3.6/DeepSeek fp8 large-M): BM64/BN128/BK64 — 2026-08-07 scan,
        # M=16384 I=128 9.2->7.4ms (-20%), I=512 30.1->18.5ms (-39%). GEMM1/GEMM2
        # share this entry: BLOCK_SIZE_M must stay aligned with align BS(64).
        (
            float("inf"),
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "num_warps": 8,
                "num_stages": 1,
            },
        ),
    ),
}

# FP8 W8A8 blockwise (128x128 blocks) tuned configs, MTT S5000 (2026-08-06 sweep).
# BN/BK are fixed by the block shape; E=256 prefers BM16/NW8, E=8 prefers BM8/NW4.
# Qwen3.6-35B-A3B tuned configs (E=256, H=2048, I=128/512, topk=8), MTT S5000
# (2026-08-06 sweep): the small hidden/intermediate dims prefer BK=32 beyond
# small M (vs BK=64 for DeepSeek's I=2048); selected by N(=I) <= 512.
_MTHREADS_TUNED_CONFIGS_QWEN36 = {
    (256, 8): (
        (
            1,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 1,
                "SWAP_AB": True,
            },
        ),
        (
            16,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 1,
            },
        ),
        (
            128,
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 1,
            },
        ),
        (
            float("inf"),
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
                "num_warps": 8,
                "num_stages": 1,
            },
        ),
    ),
}

_MTHREADS_TUNED_CONFIGS_FP8_BLOCKWISE = {
    (8, 2): (
        (
            float("inf"),
            {
                "BLOCK_SIZE_M": 8,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 2,
                "num_warps": 4,
                "num_stages": 1,
            },
        ),
    ),
    (256, 8): (
        (
            float("inf"),
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 8,
                "num_stages": 1,
            },
        ),
    ),
    (512, 10): (
        (
            float("inf"),
            {
                "BLOCK_SIZE_M": 8,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 1,
            },
        ),
    ),
}


def _is_qwen_plain_half_call(args, kwargs) -> bool:
    try:
        hidden_states = args[0] if len(args) > 0 else kwargs["hidden_states"]
        w1 = args[1] if len(args) > 1 else kwargs["w1"]
        w2 = args[2] if len(args) > 2 else kwargs["w2"]
        topk_ids = args[4] if len(args) > 4 else kwargs["topk_ids"]
    except (KeyError, IndexError):
        return False

    if w1.dtype not in ("torch.float16", "torch.bfloat16") or w2.dtype not in (
        "torch.float16",
        "torch.bfloat16",
    ):
        # Quantized weights (int8 w8a8) never use the plain-half tables.
        return False

    if str(hidden_states.dtype) not in ("torch.float16", "torch.bfloat16"):
        return False
    if topk_ids.ndim != 2:
        return False

    return (tuple(w1.shape), tuple(w2.shape), topk_ids.size(1)) in _TARGET_SHAPES


def _mthreads_get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    block_shape: list[int] | None = None,
    gemm_stage: str = "gemm1",
    enable_gemm_fast_path: bool = False,
) -> dict[str, Any]:
    """MUSA-tuned default config for target MoE shapes.

    Tuned on MTT S5000 (FlagTree 3.6 mthreads): for plain-half dtypes, the
    generic (NVIDIA-derived) heuristics are far off (2-3x slower); a config
    sweep across all benchmark/target shape families showed that small tiles
    with a single pipeline stage win on MUSA. Keyed by (E, topk) and an
    M-threshold; falls back to the generic (MUSA-safe) heuristic otherwise.
    """
    if dtype in _PLAIN_HALF_CONFIG_DTYPES and block_shape is None:
        # Qwen3.6 table only keys (256, 8); on miss fall back to the main
        # table (Mixtral/DeepSeek etc.) instead of the generic heuristic,
        # which can emit configs that crash on MUSA (e.g. BM64/BN128/BK128/NS2).
        tables = (
            (_MTHREADS_TUNED_CONFIGS_QWEN36, _MTHREADS_TUNED_CONFIGS)
            if N <= 512
            else (_MTHREADS_TUNED_CONFIGS,)
        )
        for table in tables:
            for m_max, cfg in table.get((E, topk), ()):
                if M <= m_max:
                    return dict(cfg)
    if dtype == "fp8_w8a8" and block_shape is None:
        for m_max, cfg in _MTHREADS_TUNED_CONFIGS_FP8.get((E, topk), ()):
            if M <= m_max:
                return dict(cfg)
    if dtype == "fp8_w8a8" and block_shape == [128, 128]:
        for m_max, cfg in _MTHREADS_TUNED_CONFIGS_FP8_BLOCKWISE.get((E, topk), ()):
            if M <= m_max:
                return dict(cfg)
    # Non-target shapes: fall back to a conservative MUSA-safe config instead
    # of the generic (NVIDIA-derived) heuristic, which can emit configs that
    # crash on MUSA (e.g. BM64/BN128/BK128/NS2 — see notes 1.8).
    return {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 1,
    }


@contextlib.contextmanager
def _mthreads_moe_config_patch(
    use_mthreads_moe_sum: bool,
):
    with _PATCH_LOCK:
        original_moe_sum = generic_fused_moe.moe_sum
        original_get_default_config = generic_fused_moe.get_default_config
        generic_fused_moe.get_default_config = _mthreads_get_default_config
        if use_mthreads_moe_sum:
            generic_fused_moe.moe_sum = mthreads_moe_sum
        try:
            yield
        finally:
            generic_fused_moe.moe_sum = original_moe_sum
            generic_fused_moe.get_default_config = original_get_default_config


def fused_experts_impl(*args, **kwargs):
    is_qwen_half = _is_qwen_plain_half_call(args, kwargs)
    with _mthreads_moe_config_patch(is_qwen_half):
        return generic_fused_moe.fused_experts_impl(*args, **kwargs)


def inplace_fused_experts(*args, **kwargs):
    is_qwen_half = _is_qwen_plain_half_call(args, kwargs)
    with _mthreads_moe_config_patch(is_qwen_half):
        return generic_fused_moe.inplace_fused_experts(*args, **kwargs)


def outplace_fused_experts(*args, **kwargs):
    is_qwen_half = _is_qwen_plain_half_call(args, kwargs)
    with _mthreads_moe_config_patch(is_qwen_half):
        return generic_fused_moe.outplace_fused_experts(*args, **kwargs)
