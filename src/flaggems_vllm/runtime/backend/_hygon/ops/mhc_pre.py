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

"""Hygon (gfx93x DCU) parts of ``ops/mhc/mhc_pre.py``.

Nothing here is re-exported from ``_hygon/ops/__init__.py`` on purpose: the
``mhc_pre`` entry point is registered once from the shared module, which pulls
the split-pass structure in lazily through ``_mhc_pre_setup``.  Adding the
module to the package ``__init__`` would register a second, hygon-only spec.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.mhc.mhc_pre import (
    _MHC_PRE_MIX_PAD,
    _mhc_epilogue_tile,
    _mhc_zero_,
)

# The split kernels index the partial mixes with a literal 32-wide stride; keep
# that in sync with the main module instead of trusting it silently.
_MIX_PAD = int(_MHC_PRE_MIX_PAD.value)
assert _MIX_PAD == 32, f"split kernels assume a 32-wide mix pad, got {_MIX_PAD}"

KERNELS = (
    "mhc_pre_mm_fuse_kernel",
    "mhc_pre_mm_fuse_tiled_kernel",
    "mhc_pre_generic_kernel",
    "mhc_pre_packed_reader_kernel",
    "mhc_pre_sinkhorn_fuse_kernel",
    "mhc_pre_split_pass_kernel",
)


def _autotune_key_args(inner):
    """``((name, positional index), ...)`` for the autotuner's key arguments.

    ``inner`` is a ``triton.runtime.autotuner.Autotuner``: ``keys`` holds the
    names of the arguments the tuner distinguishes, ``fn.arg_names`` maps them
    onto the call signature.
    """

    arg_names = list(getattr(getattr(inner, "fn", None), "arg_names", ()) or ())
    key_args = []
    for name in getattr(inner, "keys", ()) or ():
        index = arg_names.index(name) if name in arg_names else None
        key_args.append((name, index))
    return tuple(key_args)


def _autotune_key_value(name, index, args, kwargs):
    """Value of one autotune key argument, hashable, ``None`` when unusable."""

    if index is not None and index < len(args):
        value = args[index]
    else:
        value = kwargs.get(name)
    try:
        hash(value)
    except TypeError:
        return (name, None)
    return (name, value)


class FastLaunchKernel:
    """``kernel[grid](...args)`` with the autotuner wrapper removed."""

    __slots__ = ("_inner", "_cache", "_key_args")

    def __init__(self, inner):
        self._inner = inner
        self._cache = {}
        self._key_args = _autotune_key_args(inner)

    def __getattr__(self, name):

        return getattr(self._inner, name)

    @staticmethod
    def _proxy_key(grid, args, kwargs, key_args):
        if callable(grid):
            return None
        return (
            grid,
            tuple(getattr(a, "dtype", a) for a in args),
            tuple((k, getattr(v, "dtype", v)) for k, v in kwargs.items()),
            # The autotuner tunes per key value (hidden_size, num_tokens_bucket,
            # ...); the grid and the dtypes alone let one shape reuse another
            # shape's configuration.
            tuple(
                _autotune_key_value(name, index, args, kwargs) for name, index in key_args
            ),
        )

    def __getitem__(self, grid):
        inner = self._inner
        cache = self._cache
        key_args = self._key_args
        proxy_key = self._proxy_key

        def launcher(*args, **kwargs):
            key = proxy_key(grid, args, kwargs, key_args)
            cfg = None if key is None else cache.get(key)
            if cfg is None:
                inner[grid](*args, **kwargs)
                best = getattr(inner, "best_config", None)
                if (
                    key is not None
                    and best is not None
                    and getattr(best, "pre_hook", None) is None
                ):
                    if len(cache) >= 256:
                        cache.clear()
                    cache[key] = dict(
                        num_warps=best.num_warps,
                        num_stages=best.num_stages,
                        **best.kwargs,
                    )
                return None
            return inner.fn[grid](*args, **kwargs, **cfg)

        return launcher


def _wrap(namespace, names):
    wrapped = 0
    for name in names:
        kernel = namespace.get(name)
        if (
            kernel is not None
            and getattr(kernel, "configs", None)
            and not isinstance(kernel, FastLaunchKernel)
        ):
            namespace[name] = FastLaunchKernel(kernel)
            wrapped += 1
    return wrapped


def install_fastlaunch(namespace, names):
    """Launch the hot kernels through the JITFunction, skipping the autotuner,
    which rebuilds several dicts per call.
    """

    return _wrap(namespace, names) + _wrap(globals(), names)


@triton.jit
def _sp_epilogue(
    scratch_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    layer_input_ptr,
    rms_inv,
    offs_t,
    tmask,
    res_row,
    res_stride_i,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    HC_MULT3: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """mixes, Sinkhorn and the weighted sum, from the spilled projection."""
    _mhc_epilogue_tile(
        scratch_ptr,
        rms_inv,
        hc_scale_ptr,
        hc_base_ptr,
        residual_ptr,
        post_mix_ptr,
        comb_mix_ptr,
        layer_input_ptr,
        offs_t,
        tmask,
        res_row,
        res_stride_i,
        1,
        li_stride_n,
        li_stride_h,
        hidden_size,
        MIX_STRIDE=32,
        MIX_MASKED=False,
        H_STRIDE_ONE=True,
        H_EXACT=True,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        BLOCK_H=BLOCK_H,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 128}, num_warps=8, num_stages=3),
    ],
    key=["hidden_size"],
)
@triton.jit
def mhc_pre_split_pass_kernel(
    BT_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    layer_input_ptr,
    part_ptr,  # (N_SPLITS, num_tokens, 32) float32
    part_sq_ptr,  # (N_SPLITS, num_tokens) float32
    scratch_ptr,  # (num_tokens, 32) float32 spill for the epilogue
    cnt_ptr,  # (grid_m,) int32, zeroed before the first launch
    num_tokens,
    num_tokens_bucket,
    res_stride_n,
    res_stride_i,
    li_stride_n,
    li_stride_h,
    hidden_size,
    hc_hidden_size,
    K_LEN: tl.constexpr,
    N_SPLITS: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
    HC_MULT3: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """The mixes, Sinkhorn and weighted sum with K cut across N_SPLITS programs:
    each split stores its projection and sqrsum partial, the last one to arrive
    reduces them and runs the epilogue.
    """
    pid_t = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_t = pid_t * TOKEN_TILE + tl.arange(0, TOKEN_TILE)
    tmask = offs_t < num_tokens
    res_row = offs_t * res_stride_n

    offs_n = tl.arange(0, 32)
    K_SPLIT: tl.constexpr = K_LEN // N_SPLITS
    k_base = pid_k * K_SPLIT
    acc = tl.zeros((TOKEN_TILE, 32), dtype=tl.float32)
    sq2 = tl.zeros((TOKEN_TILE, BLOCK_K), dtype=tl.float32)
    for k0 in range(0, K_SPLIT, BLOCK_K):
        offs_k = k_base + k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            residual_ptr + res_row[:, None] + offs_k[None, :],
            mask=tmask[:, None],
            other=0.0,
        )
        b = tl.load(BT_ptr + offs_k[:, None] * 32 + offs_n[None, :])
        acc += tl.dot(a, b)
        af = a.to(tl.float32)
        sq2 += af * af
    sq = tl.sum(sq2, axis=1)

    part_base = pid_k * num_tokens
    tl.store(
        part_ptr + part_base * 32 + offs_t[:, None] * 32 + offs_n[None, :],
        acc,
        mask=tmask[:, None],
    )
    tl.store(part_sq_ptr + part_base + offs_t, sq, mask=tmask)

    old = tl.atomic_add(cnt_ptr + pid_t, 1, sem="acq_rel", scope="gpu")
    if old == N_SPLITS - 1:
        acc_tot = tl.zeros((TOKEN_TILE, 32), dtype=tl.float32)
        sq_tot = tl.zeros((TOKEN_TILE,), dtype=tl.float32)
        for i in tl.static_range(N_SPLITS):
            ib = i * num_tokens
            acc_tot += tl.load(
                part_ptr + ib * 32 + offs_t[:, None] * 32 + offs_n[None, :],
                mask=tmask[:, None],
                other=0.0,
            )
            sq_tot += tl.load(part_sq_ptr + ib + offs_t, mask=tmask, other=0.0)

        rms_inv = tl.rsqrt(sq_tot / hc_hidden_size + rms_eps)
        acc_tot = acc_tot.to(tl.bfloat16).to(tl.float32)
        tl.store(
            scratch_ptr + offs_t[:, None] * 32 + offs_n[None, :],
            acc_tot,
            mask=tmask[:, None],
        )
        tl.debug_barrier()
        _sp_epilogue(
            scratch_ptr,
            hc_scale_ptr,
            hc_base_ptr,
            residual_ptr,
            post_mix_ptr,
            comb_mix_ptr,
            layer_input_ptr,
            rms_inv,
            offs_t,
            tmask,
            res_row,
            res_stride_i,
            li_stride_n,
            li_stride_h,
            hidden_size,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            HC_MULT3,
            BLOCK_H,
        )
        tl.store(cnt_ptr + pid_t, 0)


_SPLIT_BUFFERS: dict[tuple[int, int, int, int], tuple] = {}


def split_buffers(num_tokens, hidden_size, n_splits, token_tile, device):
    """Per-shape scratch for the split kernel: the two partial buffers, the
    epilogue spill and the arrival counters. Kept alive across calls because a
    fresh allocation here costs more than the split itself saves.
    """
    key = (num_tokens, hidden_size, n_splits, token_tile)
    bufs = _SPLIT_BUFFERS.get(key)
    if bufs is not None and bufs[0].device == device:
        return bufs
    if len(_SPLIT_BUFFERS) >= 32:
        _SPLIT_BUFFERS.clear()
    bufs = (
        torch.empty(n_splits, num_tokens, 32, dtype=torch.float32, device=device),
        torch.empty(n_splits, num_tokens, dtype=torch.float32, device=device),
        # token_tile rows of slack so the epilogue can read a full tile
        # unmasked; the padding is zeroed once and never written again.
        _mhc_zero_(
            torch.empty(num_tokens + token_tile, 32, dtype=torch.float32, device=device)
        ),
        _mhc_zero_(
            torch.empty(
                (triton.cdiv(num_tokens, token_tile),),
                dtype=torch.int32,
                device=device,
            )
        ),
    )
    _SPLIT_BUFFERS[key] = bufs
    return bufs
