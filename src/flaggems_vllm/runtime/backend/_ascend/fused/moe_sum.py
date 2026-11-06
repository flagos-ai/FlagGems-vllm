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

import logging

import torch
import triton
import triton.language as tl
from triton.backends.ascend.testing import do_bench_npu

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.triton_version_utils import HAS_TLE

logger = logging.getLogger(__name__)

MAX_CORE_NUM = 65535

if HAS_TLE:
    import triton.experimental.tle.language as tle
else:
    tle = None


def _do_bench_npu_for_autotune(kernel_call, quantiles=None):
    v = do_bench_npu(kernel_call, warmup=5, active=30, clear_l2_cache=True)
    return [v, v, v]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("moe_sum_pair"),
    do_bench=_do_bench_npu_for_autotune,
    key=["hidden_size", "topk", "ELEM_SIZE"],
)
@triton.jit
def _ascend_moe_sum_pair_kernel(
    input_ptr,
    output_ptr,
    router_weights_ptr,
    num_tokens,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    APPLY_ROUTER_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
):
    """Per-token kernel for small batches (num_tokens < 128).

    Each program handles one token's hidden-dim block and loads the topk
    rows two at a time. For small token counts the multi-token tiles would
    underfill the cores, so per-token processing wins. Uses a 1D capped
    grid with a grid-stride loop.
    """
    input_stride_token: tl.constexpr = topk * hidden_size
    input_stride_topk: tl.constexpr = hidden_size
    output_stride_token: tl.constexpr = hidden_size

    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_h_blocks: tl.constexpr = (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    total = num_tokens * num_h_blocks

    for flat in range(pid, total, num_pids):
        token_idx = flat // num_h_blocks
        hidden_block_idx = flat % num_h_blocks
        hidden_offsets = hidden_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        hidden_offsets = tl.max_contiguous(
            tl.multiple_of(hidden_offsets, BLOCK_SIZE), BLOCK_SIZE
        )

        hidden_mask = hidden_offsets < hidden_size
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        input_base = input_ptr + token_idx * input_stride_token

        for k in tl.static_range(0, topk, 2):
            x0 = tl.load(
                input_base + k * input_stride_topk + hidden_offsets,
                mask=hidden_mask,
                other=0.0,
            )
            if APPLY_ROUTER_WEIGHT:
                w0 = tl.load(router_weights_ptr + token_idx * topk + k)
                x0 = x0.to(tl.float32) * w0.to(tl.float32)
            if k + 1 < topk:
                x1 = tl.load(
                    input_base + (k + 1) * input_stride_topk + hidden_offsets,
                    mask=hidden_mask,
                    other=0.0,
                )
                if APPLY_ROUTER_WEIGHT:
                    w1 = tl.load(router_weights_ptr + token_idx * topk + (k + 1))
                    x1 = x1.to(tl.float32) * w1.to(tl.float32)
                acc += x0.to(tl.float32) + x1.to(tl.float32)
            else:
                acc += x0.to(tl.float32)

        output_ptrs = output_ptr + token_idx * output_stride_token + hidden_offsets
        tl.store(
            output_ptrs,
            acc.to(output_ptr.dtype.element_ty),
            mask=hidden_mask,
        )


if HAS_TLE:

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("moe_sum_mt_2d"),
        do_bench=_do_bench_npu_for_autotune,
        key=["hidden_size", "topk", "token_bucket", "ELEM_SIZE"],
    )
    @triton.jit
    def _ascend_moe_sum_mt_2d_kernel(
        input_ptr,
        output_ptr,
        router_weights_ptr,
        num_tokens,
        hidden_size: tl.constexpr,
        token_bucket,
        topk: tl.constexpr,
        APPLY_ROUTER_WEIGHT: tl.constexpr,
        TOKENS: tl.constexpr,
        BLOCK_H: tl.constexpr,
        ELEM_SIZE: tl.constexpr,
        W_ELEM_SIZE: tl.constexpr,
    ):
        """Multi-token tile kernel with a 2D loop-free grid (large batches).

        Each program handles exactly one (TOKENS, BLOCK_H) tile. When the
        weight block is 32B-aligned and topk is a power of two, router
        weights are preloaded into UB as one DMA block (TLE dsa.copy) and
        column k is extracted with a register-only one-hot reduction;
        otherwise weights fall back to per-k tl.load. Requires 16B-aligned
        hidden rows (vec_ok, enforced by the caller).
        """
        input_stride_token: tl.constexpr = topk * hidden_size
        input_stride_topk: tl.constexpr = hidden_size
        output_stride_token: tl.constexpr = hidden_size

        pid_hidden = tl.program_id(0)
        pid_token = tl.program_id(1)

        token_offsets = pid_token * TOKENS + tl.arange(0, TOKENS)
        hidden_offsets = pid_hidden * BLOCK_H + tl.arange(0, BLOCK_H)
        hidden_offsets = tl.max_contiguous(
            tl.multiple_of(hidden_offsets, BLOCK_H), BLOCK_H
        )

        token_mask = token_offsets < num_tokens
        EVEN_H: tl.constexpr = hidden_size % BLOCK_H == 0
        if EVEN_H:
            mask = tl.broadcast_to(token_mask[:, None], (TOKENS, BLOCK_H))
        else:
            mask = token_mask[:, None] & (hidden_offsets[None, :] < hidden_size)

        acc = tl.zeros((TOKENS, BLOCK_H), dtype=tl.float32)
        input_base = (
            input_ptr
            + token_offsets[:, None] * input_stride_token
            + hidden_offsets[None, :]
        )

        # TLE one-hot weight-block preload gate: the weight block must be
        # 32B-aligned (note: W_ELEM_SIZE of the weights, not the input's)
        # and topk must be a power of two (tl.arange/alloc require pow2
        # dims). When False the constexpr branch is pruned at compile time
        # and weights fall back to per-k tl.load.
        W_ONEHOT_OK: tl.constexpr = (TOKENS * topk * W_ELEM_SIZE) % 32 == 0 and (
            topk & (topk - 1)
        ) == 0
        if APPLY_ROUTER_WEIGHT and W_ONEHOT_OK:
            valid_tokens = tl.minimum(num_tokens - pid_token * TOKENS, TOKENS)
            w_ub = tle.dsa.alloc(
                [TOKENS, topk],
                dtype=router_weights_ptr.dtype.element_ty,
                mem_addr_space=tle.dsa.ascend.UB,
            )
            w_ptrs = (
                router_weights_ptr
                + token_offsets[:, None] * topk
                + tl.arange(0, topk)[None, :]
            )
            tle.dsa.copy(w_ptrs, w_ub, [valid_tokens, topk])
            w_mat = tle.dsa.to_tensor(w_ub)
            k_offs = tl.arange(0, topk)

        for k in tl.static_range(topk):
            x = tl.load(input_base + k * input_stride_topk, mask=mask, other=0.0)
            if APPLY_ROUTER_WEIGHT:
                if W_ONEHOT_OK:
                    w_col = tl.sum(
                        tl.where(k_offs[None, :] == k, w_mat, 0.0).to(tl.float32),
                        axis=1,
                    )
                    x = x.to(tl.float32) * w_col[:, None]
                else:
                    w = tl.load(
                        router_weights_ptr + token_offsets[:, None] * topk + k,
                        mask=token_mask[:, None],
                        other=0.0,
                    )
                    x = x.to(tl.float32) * w.to(tl.float32)
            acc += x.to(tl.float32)

        output_ptrs = (
            output_ptr
            + token_offsets[:, None] * output_stride_token
            + hidden_offsets[None, :]
        )
        tl.store(output_ptrs, acc.to(output_ptr.dtype.element_ty), mask=mask)

    @libentry()
    @libtuner(
        configs=runtime.get_tuned_config("moe_sum_mt"),
        do_bench=_do_bench_npu_for_autotune,
        key=["hidden_size", "topk", "token_bucket", "ELEM_SIZE"],
    )
    @triton.jit
    def _ascend_moe_sum_mt_kernel(
        input_ptr,
        output_ptr,
        router_weights_ptr,
        num_tokens,
        hidden_size: tl.constexpr,
        token_bucket,
        topk: tl.constexpr,
        APPLY_ROUTER_WEIGHT: tl.constexpr,
        TOKENS: tl.constexpr,
        BLOCK_H: tl.constexpr,
        ELEM_SIZE: tl.constexpr,
        W_ELEM_SIZE: tl.constexpr,
    ):
        """Grid-stride-loop variant of the multi-token tile kernel.

        Semantically identical to _ascend_moe_sum_mt_2d_kernel (including
        the TLE one-hot weight preload); used when the flattened 2D grid
        would exceed MAX_CORE_NUM, i.e. very large token counts.
        """
        input_stride_token: tl.constexpr = topk * hidden_size
        input_stride_topk: tl.constexpr = hidden_size
        output_stride_token: tl.constexpr = hidden_size

        pid = tl.program_id(0)
        num_pids = tl.num_programs(0)
        num_h_blocks: tl.constexpr = (hidden_size + BLOCK_H - 1) // BLOCK_H
        num_t_blocks = (num_tokens + TOKENS - 1) // TOKENS
        total = num_t_blocks * num_h_blocks

        W_ONEHOT_OK: tl.constexpr = (TOKENS * topk * W_ELEM_SIZE) % 32 == 0 and (
            topk & (topk - 1)
        ) == 0
        if APPLY_ROUTER_WEIGHT and W_ONEHOT_OK:
            w_ub = tle.dsa.alloc(
                [TOKENS, topk],
                dtype=router_weights_ptr.dtype.element_ty,
                mem_addr_space=tle.dsa.ascend.UB,
            )
            k_offs = tl.arange(0, topk)

        for flat in range(pid, total, num_pids):
            token_block_idx = flat // num_h_blocks
            hidden_block_idx = flat % num_h_blocks

            token_offsets = token_block_idx * TOKENS + tl.arange(0, TOKENS)
            hidden_offsets = hidden_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
            hidden_offsets = tl.max_contiguous(
                tl.multiple_of(hidden_offsets, BLOCK_H), BLOCK_H
            )

            token_mask = token_offsets < num_tokens
            EVEN_H: tl.constexpr = hidden_size % BLOCK_H == 0
            if EVEN_H:
                mask = tl.broadcast_to(token_mask[:, None], (TOKENS, BLOCK_H))
            else:
                mask = token_mask[:, None] & (hidden_offsets[None, :] < hidden_size)

            acc = tl.zeros((TOKENS, BLOCK_H), dtype=tl.float32)
            input_base = (
                input_ptr
                + token_offsets[:, None] * input_stride_token
                + hidden_offsets[None, :]
            )

            if APPLY_ROUTER_WEIGHT and W_ONEHOT_OK:
                valid_tokens = tl.minimum(num_tokens - token_block_idx * TOKENS, TOKENS)
                w_ptrs = (
                    router_weights_ptr
                    + token_offsets[:, None] * topk
                    + tl.arange(0, topk)[None, :]
                )
                tle.dsa.copy(w_ptrs, w_ub, [valid_tokens, topk])
                w_mat = tle.dsa.to_tensor(w_ub)

            for k in tl.static_range(topk):
                x = tl.load(input_base + k * input_stride_topk, mask=mask, other=0.0)
                if APPLY_ROUTER_WEIGHT:
                    if W_ONEHOT_OK:
                        w_col = tl.sum(
                            tl.where(k_offs[None, :] == k, w_mat, 0.0).to(tl.float32),
                            axis=1,
                        )
                        x = x.to(tl.float32) * w_col[:, None]
                    else:
                        w = tl.load(
                            router_weights_ptr + token_offsets[:, None] * topk + k,
                            mask=token_mask[:, None],
                            other=0.0,
                        )
                        x = x.to(tl.float32) * w.to(tl.float32)
                acc += x.to(tl.float32)

            output_ptrs = (
                output_ptr
                + token_offsets[:, None] * output_stride_token
                + hidden_offsets[None, :]
            )
            tl.store(output_ptrs, acc.to(output_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("moe_sum_topk2_fused"),
    do_bench=_do_bench_npu_for_autotune,
    key=["hidden_size", "ELEM_SIZE"],
)
@triton.jit
def _ascend_moe_sum_topk2_fused_kernel(
    input_ptr,
    output_ptr,
    router_weights_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    APPLY_ROUTER_WEIGHT: tl.constexpr,
    TOKENS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
):
    """topk=2 special-case kernel (large batches).

    Uses a flattened (tokens*2, hidden) view so a token's two rows sit
    adjacent: each program loads a (2*TOKENS, BLOCK_H) tile of consecutive
    rows (one fully contiguous memory block when BLOCK_H == hidden_size),
    reshapes to (TOKENS, 2, BLOCK_H) and reduces with weights in registers.
    This fixes the DMA fragmentation of small topk. No 16B-alignment
    requirement on hidden (masked tail).
    """
    pid_hidden = tl.program_id(0)
    pid_token = tl.program_id(1)

    row_offsets = pid_token * (2 * TOKENS) + tl.arange(0, 2 * TOKENS)
    row_mask = row_offsets < 2 * num_tokens
    hidden_offsets = pid_hidden * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_offsets = tl.max_contiguous(tl.multiple_of(hidden_offsets, BLOCK_H), BLOCK_H)

    mask = row_mask[:, None] & (hidden_offsets[None, :] < hidden_size)
    input_ptrs = (
        input_ptr + row_offsets[:, None] * hidden_size + hidden_offsets[None, :]
    )
    x = tl.load(input_ptrs, mask=mask, other=0.0)

    x = tl.reshape(x, (TOKENS, 2, BLOCK_H)).to(tl.float32)
    if APPLY_ROUTER_WEIGHT:
        w_flat = tl.load(
            router_weights_ptr + pid_token * (2 * TOKENS) + tl.arange(0, 2 * TOKENS),
            mask=row_mask,
            other=0.0,
        )
        w_mat = tl.reshape(w_flat, (TOKENS, 2)).to(tl.float32)
        x = x * w_mat[:, :, None]

    acc = tl.sum(x, axis=1)

    token_offsets = pid_token * TOKENS + tl.arange(0, TOKENS)
    token_mask = token_offsets < num_tokens
    output_ptrs = (
        output_ptr + token_offsets[:, None] * hidden_size + hidden_offsets[None, :]
    )
    tl.store(
        output_ptrs,
        acc.to(output_ptr.dtype.element_ty),
        mask=token_mask[:, None] & (hidden_offsets[None, :] < hidden_size),
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("moe_sum_general"),
    do_bench=_do_bench_npu_for_autotune,
    key=["hidden_size", "topk", "ELEM_SIZE"],
)
@triton.jit
def _ascend_moe_sum_general_kernel(
    input_ptr,
    output_ptr,
    router_weights_ptr,
    router_weights_stride_token,
    router_weights_stride_topk,
    num_tokens,
    topk,
    hidden_size: tl.constexpr,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    APPLY_ROUTER_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
):
    """Fully stride-aware fallback kernel.

    input / output / router_weights are all addressed by runtime strides,
    so any layout works: non-contiguous tensors, topk > 16, and hidden
    sizes the fast paths reject (not 16B-aligned). Uses a 1D capped grid
    with a grid-stride loop.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_h_blocks: tl.constexpr = (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    total = num_tokens * num_h_blocks

    for flat in range(pid, total, num_pids):
        token_idx = flat // num_h_blocks
        hidden_block_idx = flat % num_h_blocks
        hidden_offsets = hidden_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        hidden_mask = hidden_offsets < hidden_size
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        input_base = (
            input_ptr
            + token_idx * input_stride_token
            + hidden_offsets * input_stride_hidden
        )

        for k in range(topk):
            x = tl.load(
                input_base + k * input_stride_topk,
                mask=hidden_mask,
                other=0.0,
            )
            if APPLY_ROUTER_WEIGHT:
                w = tl.load(
                    router_weights_ptr
                    + token_idx * router_weights_stride_token
                    + k * router_weights_stride_topk
                )
                x = x.to(tl.float32) * w.to(tl.float32)
            acc += x.to(tl.float32)

        output_ptrs = (
            output_ptr
            + token_idx * output_stride_token
            + hidden_offsets * output_stride_hidden
        )
        tl.store(
            output_ptrs,
            acc.to(output_ptr.dtype.element_ty),
            mask=hidden_mask,
        )


def _token_bucket(num_tokens: int) -> int:
    if num_tokens < 1024:
        return 0
    if num_tokens < 8192:
        return 1
    return 2


def _check_moe_sum_inputs(input: torch.Tensor, output: torch.Tensor):
    assert input.dim() == 3, (
        f"moe_sum: input must be 3D [num_tokens, topk, hidden], "
        f"got {input.dim()}D shape {tuple(input.shape)}"
    )
    assert output.dim() == 2, (
        f"moe_sum: output must be 2D [num_tokens, hidden], "
        f"got {output.dim()}D shape {tuple(output.shape)}"
    )
    num_tokens, topk, hidden_size = input.shape
    assert topk >= 1, f"moe_sum: topk must be >= 1, got {topk}"
    assert output.shape == (num_tokens, hidden_size), (
        f"moe_sum: output shape {tuple(output.shape)} mismatch, "
        f"expected ({num_tokens}, {hidden_size}) from input shape"
    )
    assert (
        input.dtype == output.dtype
    ), f"moe_sum: dtype mismatch, input {input.dtype} vs output {output.dtype}"
    assert input.dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"moe_sum: unsupported dtype {input.dtype}, "
        f"expected float16 / bfloat16 / float32"
    )
    assert (
        input.device == output.device
    ), f"moe_sum: device mismatch, input {input.device} vs output {output.device}"


def _min_vec_elems(dtype: torch.dtype, vec_bytes: int = 16) -> int:
    """Min elements for a 16B vectorized load:
    fp16/bf16 -> 8 elems, fp32 -> 4 elems.
    """
    return max(1, vec_bytes // torch.tensor([], dtype=dtype).element_size())


def _capped_grid(total: int):
    """Clamp a flattened 1D grid to MAX_CORE_NUM (Ascend coreDim limit)."""
    return (min(total, MAX_CORE_NUM),)


def _autotune_configs(kernel):
    """Return the autotune config list of a kernel."""
    obj = kernel
    while obj is not None and not hasattr(obj, "configs"):
        obj = getattr(obj, "fn", None)
    if obj is None:
        raise AttributeError(
            f"cannot locate autotune configs on {type(kernel).__name__}"
        )
    return obj.configs


def _grid_fits_2d(kernel, num_tokens: int, hidden_size: int) -> bool:
    """Check whether every autotune config's flattened 2D grid stays <= MAX_CORE_NUM.

    Triton-Ascend flattens the 2D grid into a 1D coreDim (max 65535). The 2D
    loop-free fast path is safe only if no config exceeds the limit; otherwise
    the caller falls back to the 1D loop variant.
    (TOKENS, BLOCK_H) are read from the kernel's own autotune configs.
    """
    for cfg in _autotune_configs(kernel):
        flat = triton.cdiv(hidden_size, cfg.kwargs["BLOCK_H"]) * triton.cdiv(
            num_tokens, cfg.kwargs["TOKENS"]
        )
        if flat > MAX_CORE_NUM:
            return False
    return True


def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
    router_weights: torch.Tensor | None = None,
):
    logger.debug("GEMS_ASCEND MOE SUM")
    _check_moe_sum_inputs(input, output)
    num_tokens, topk, hidden_size = input.shape

    if router_weights is not None:
        assert router_weights.shape == (num_tokens, topk), (
            f"moe_sum: router_weights shape {tuple(router_weights.shape)} mismatch, "
            f"expected ({num_tokens}, {topk}) from input shape"
        )
        router_weights_strides = router_weights.stride()
    else:
        router_weights_strides = (0, 0)

    elem_size = input.element_size()
    w_elem_size = (
        router_weights.element_size() if router_weights is not None else elem_size
    )
    input_stride = input.stride()
    output_stride = output.stride()

    # Fast tile loads use 16B vector instructions, so hidden must be a
    # multiple of the vector width (fp16/bf16: 8 elems, fp32: 4 elems);
    # otherwise rows become misaligned and the mt path is skipped.
    vec_ok = hidden_size % _min_vec_elems(input.dtype, vec_bytes=16) == 0
    weights_contiguous = router_weights is None or router_weights.is_contiguous()
    # Fast paths (fused/mt/pair) derive input/output strides from shapes
    # inside the kernel and require contiguous layouts; non-contiguous
    # tensors fall back to the fully stride-aware general kernel.
    contiguous = input.is_contiguous() and output.is_contiguous()

    if (
        contiguous
        and weights_contiguous
        and topk == 2
        and num_tokens >= 128
        and _grid_fits_2d(_ascend_moe_sum_topk2_fused_kernel, num_tokens, hidden_size)
    ):
        grid = lambda meta: (
            triton.cdiv(hidden_size, meta["BLOCK_H"]),
            triton.cdiv(num_tokens, meta["TOKENS"]),
        )
        _ascend_moe_sum_topk2_fused_kernel[grid](
            input,
            output,
            input if router_weights is None else router_weights,
            num_tokens,
            hidden_size,
            APPLY_ROUTER_WEIGHT=router_weights is not None,
            ELEM_SIZE=elem_size,
        )
    elif (
        HAS_TLE
        and contiguous
        and weights_contiguous
        and topk <= 16
        and num_tokens >= 128
        and vec_ok
    ):
        if _grid_fits_2d(_ascend_moe_sum_mt_2d_kernel, num_tokens, hidden_size):
            grid = lambda meta: (
                triton.cdiv(hidden_size, meta["BLOCK_H"]),
                triton.cdiv(num_tokens, meta["TOKENS"]),
            )
            _ascend_moe_sum_mt_2d_kernel[grid](
                input,
                output,
                input if router_weights is None else router_weights,
                num_tokens,
                hidden_size,
                _token_bucket(num_tokens),
                topk,
                APPLY_ROUTER_WEIGHT=router_weights is not None,
                ELEM_SIZE=elem_size,
                W_ELEM_SIZE=w_elem_size,
            )
        else:
            grid = lambda meta: _capped_grid(
                triton.cdiv(hidden_size, meta["BLOCK_H"])
                * triton.cdiv(num_tokens, meta["TOKENS"])
            )
            _ascend_moe_sum_mt_kernel[grid](
                input,
                output,
                input if router_weights is None else router_weights,
                num_tokens,
                hidden_size,
                _token_bucket(num_tokens),
                topk,
                APPLY_ROUTER_WEIGHT=router_weights is not None,
                ELEM_SIZE=elem_size,
                W_ELEM_SIZE=w_elem_size,
            )
    elif contiguous and weights_contiguous and topk <= 16 and num_tokens < 128:
        grid = lambda meta: _capped_grid(
            num_tokens * triton.cdiv(hidden_size, meta["BLOCK_SIZE"])
        )
        _ascend_moe_sum_pair_kernel[grid](
            input,
            output,
            input if router_weights is None else router_weights,
            num_tokens,
            topk,
            hidden_size,
            APPLY_ROUTER_WEIGHT=router_weights is not None,
            ELEM_SIZE=elem_size,
        )
    else:
        grid = lambda meta: _capped_grid(
            num_tokens * triton.cdiv(hidden_size, meta["BLOCK_SIZE"])
        )
        _ascend_moe_sum_general_kernel[grid](
            input,
            output,
            input if router_weights is None else router_weights,
            router_weights_strides[0],
            router_weights_strides[1],
            num_tokens,
            topk,
            hidden_size,
            input_stride[0],
            input_stride[1],
            input_stride[2],
            output_stride[0],
            output_stride[1],
            APPLY_ROUTER_WEIGHT=router_weights is not None,
            ELEM_SIZE=elem_size,
        )
