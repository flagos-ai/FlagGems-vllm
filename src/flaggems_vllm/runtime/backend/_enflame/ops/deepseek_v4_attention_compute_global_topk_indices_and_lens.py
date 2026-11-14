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

"""compute_global_topk_indices_and_lens (enflame/GCU local copy).

Generic kernel is kept verbatim as the fallback path. On top of it this copy
adds a specialized fast kernel dispatched when all of the following hold:

* ``topk`` is a power of two  -> one tile column block covers the whole row
  (``BLOCK == topk``), so the per-column runtime mask ``offs < topk`` and the
  ``for start in range(0, topk, BLOCK)`` loop disappear;
* ``block_size`` is a power of two -> integer div/mod becomes shift/mask;
* ``topk_indices`` is row-contiguous (``stride == (topk, 1)``) and
  ``global_indices`` (allocated here) matches -> the row base is a plain
  ``token * topk`` multiply shared by load and store.

Anything else (non-pow2 sizes, strided inputs) falls back to the generic
kernel, which is identical to the FlagGems-vllm generic implementation except
for one defensive guard in its config prune (see ``_prune_configs``): upstream
raises AutotunerError for topk < 4 because every config gets pruned; the guard
falls back to the smallest-BLOCK config instead, turning that crash into a
correct masked-loop execution.

GCU perf notes (Libra/GCU400, do_bench kernel-mode median, job
j20261108102035_9fa8db9d): the generic form is latency-bound on the
data-dependent ``block_table`` gather plus a 2D runtime-mask tax on the big
load/store; raising tokens-per-program (TPP) with num_warps=4 restores
memory-level parallelism for the gather. Production-path harness (top-level
API, fresh-tuned), per core shape (num_tokens x topk), baseline -> this copy:
5x4 9.13->6.74us, 128x32 12.86->8.00us, 512x64 19.60->11.19us,
4096x128 75.8->48.5us (x2 req variants), 8192x128 148.3->89.8us.

Autotune exemption note: this local copy uses a curated in-file config space
instead of ``runtime.get_tuned_config`` because the enflame backend yaml is
outside the editable scope of this copy (single-file drop). The flagtune
expansion hooks are deliberately omitted for the fast kernel so the curated
space stays authoritative; the fallback kernel keeps its original tuner.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner


def _prune_configs(configs, named_args, **kwargs):
    topk = named_args["topk"]
    num_tokens = named_args["num_tokens"]
    pruned = []
    for cfg in configs:
        BLOCK = cfg.kwargs.get("BLOCK", 1024)
        TPP = cfg.kwargs.get("TPP", 1)
        if BLOCK * TPP > 1024:
            continue
        if BLOCK > topk * 4:
            continue
        if num_tokens <= 64 and TPP >= 8:
            continue
        if num_tokens <= 32 and TPP >= 4:
            continue
        if num_tokens <= 16 and TPP >= 2:
            continue
        if TPP > triton.cdiv(num_tokens, 2):
            continue
        pruned.append(cfg)
    if not pruned:
        # Defensive (enflame-local deviation): upstream prunes every config for
        # topk < 4 (min BLOCK=16 > topk*4) and raises AutotunerError. The kernel
        # is correct for BLOCK > topk (masked loop), so fall back to the
        # smallest-BLOCK config instead of crashing. Only reachable on inputs
        # that crashed upstream; previously-working selections are unaffected.
        pruned = [min(configs, key=lambda cfg: cfg.kwargs.get("BLOCK", 1024))]
    return pruned


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("compute_global_topk_indices_and_lens"),
    key=["topk", "num_tokens"],
    strategy=["log", "log"],
    prune_configs_by={"early_config_prune": _prune_configs},
    reset_to_zero=["lens_ptr"],
    flagtune_op_name="compute_global_topk_indices_and_lens",
    flagtune_expand_op_name="compute_global_topk_indices_and_lens",
)
@triton.jit
def _compute_global_topk_indices_and_lens_kernel(
    global_indices_ptr,
    global_stride,
    lens_ptr,
    local_indices_ptr,
    local_stride,
    topk,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    is_valid_token_ptr,
    num_tokens,
    BLOCK: tl.constexpr,
    TPP: tl.constexpr,
):
    pid = tl.program_id(0)
    token_start = pid * TPP
    token_offs = token_start + tl.arange(0, TPP)
    token_mask = token_offs < num_tokens

    is_valid = tl.load(is_valid_token_ptr + token_offs, mask=token_mask, other=0)
    req_idx = tl.load(token_to_req_indices_ptr + token_offs, mask=token_mask, other=0)

    local_base = token_offs[:, None] * local_stride
    global_base = token_offs[:, None] * global_stride
    block_table_base = req_idx[:, None] * block_table_stride

    counts = tl.zeros((TPP,), dtype=tl.int32)

    for start in range(0, topk, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        topk_mask = offs < topk
        mask_2d = token_mask[:, None] & topk_mask[None, :]

        local_idx = tl.load(
            local_indices_ptr + local_base + offs[None, :],
            mask=mask_2d,
            other=-1,
        )
        valid = local_idx >= 0

        block_idx = local_idx // block_size
        block_off = local_idx - block_idx * block_size

        block_no = tl.load(
            block_table_ptr + block_table_base + block_idx,
            mask=mask_2d & valid,
            other=0,
        )
        slot = block_no * block_size + block_off
        slot = tl.where(valid, slot, -1)

        tl.store(
            global_indices_ptr + global_base + offs[None, :],
            slot,
            mask=mask_2d,
        )
        counts += tl.sum(valid.to(tl.int32), axis=1)

    lens = tl.where(is_valid != 0, counts, 0)
    tl.store(lens_ptr + token_offs, lens, mask=token_mask)


def _is_pow2(value):
    return value > 0 and (value & (value - 1)) == 0


def _fast_prune_configs(configs, named_args, **kwargs):
    num_tokens = named_args["num_tokens"]
    topk = named_args["topk"]
    work = num_tokens * topk
    if work <= 32768:
        # Pinned arms for small work: at this scale the kernel is 6-11us and
        # the autotuner's own do_bench closure (zero_ launch + event gaps)
        # carries O(10us) pollution that misranks configs. A single surviving
        # config also skips the benchmark pass entirely in LibTuner.run.
        # Calibrated on the core shapes (5x4 / 128x32 / 512x64), Libra/GCU400.
        pinned_tpp = 4 if num_tokens <= 128 else 16
        return [
            cfg
            for cfg in configs
            if cfg.kwargs.get("TPP") == pinned_tpp and cfg.num_warps == 4
        ]
    pruned = []
    for cfg in configs:
        TPP = cfg.kwargs.get("TPP", 8)
        if num_tokens >= 2048 and TPP <= 8:
            # Large token counts: TPP<=8 measured 1.5-8x slower (fewer
            # gather lanes in flight per program); keep the bench lean.
            continue
        if TPP > 2 and TPP > num_tokens:
            continue
        pruned.append(cfg)
    return pruned


# Curated space for the fast kernel (see module docstring). num_warps=4 is the
# sweet spot for the gather-bound large shapes; the nw=2 arms cover the small
# token counts where a [TPP, topk] tile has few lanes. num_stages is flat
# within noise (2/3/4 all within 0.6us at TPP=64), so a single value is kept.
_FAST_CONFIGS = [
    triton.Config({"TPP": 2}, num_warps=4, num_stages=2),
    triton.Config({"TPP": 4}, num_warps=2, num_stages=2),
    triton.Config({"TPP": 4}, num_warps=4, num_stages=2),
    triton.Config({"TPP": 8}, num_warps=2, num_stages=2),
    triton.Config({"TPP": 8}, num_warps=4, num_stages=2),
    triton.Config({"TPP": 16}, num_warps=4, num_stages=2),
    triton.Config({"TPP": 32}, num_warps=4, num_stages=2),
    triton.Config({"TPP": 64}, num_warps=4, num_stages=2),
]


@libentry()
@libtuner(
    configs=_FAST_CONFIGS,
    key=["topk", "num_tokens"],
    strategy=["log", "log"],
    prune_configs_by={"early_config_prune": _fast_prune_configs},
)
@triton.jit
def _compute_global_topk_indices_and_lens_fast_kernel(
    global_indices_ptr,
    lens_ptr,
    local_indices_ptr,
    topk,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    is_valid_token_ptr,
    num_tokens,
    LOG2_BS: tl.constexpr,
    BLOCK: tl.constexpr,
    TPP: tl.constexpr,
):
    # Dispatch preconditions (checked on host): BLOCK == topk (pow2), row
    # stride of both index tensors == topk, block_size == 1 << LOG2_BS.
    pid = tl.program_id(0)
    token_offs = pid * TPP + tl.arange(0, TPP)
    token_mask = token_offs < num_tokens

    is_valid = tl.load(is_valid_token_ptr + token_offs, mask=token_mask, other=0)
    req_idx = tl.load(token_to_req_indices_ptr + token_offs, mask=token_mask, other=0)

    local_base = token_offs[:, None] * topk
    block_table_base = req_idx[:, None] * block_table_stride

    offs = tl.arange(0, BLOCK)
    # No column mask: BLOCK == topk is a host-dispatch precondition, so the
    # [TPP, BLOCK] tile covers each row exactly; only the tail token program
    # needs the row mask (measured ~3% cost, kept unconditionally to decouple
    # the kernel from the autotuned TPP choice).
    mask_2d = token_mask[:, None]

    local_idx = tl.load(
        local_indices_ptr + local_base + offs[None, :],
        mask=mask_2d,
        other=-1,
    )
    valid = local_idx >= 0

    # pow2 block_size: floor-div/mod via arithmetic shift + mask. For
    # local_idx = -1 (invalid) the shifted values are garbage but the gather
    # is masked by `valid` and the slot is overridden below.
    block_idx = local_idx >> LOG2_BS
    block_off = local_idx - (block_idx << LOG2_BS)

    block_no = tl.load(
        block_table_ptr + block_table_base + block_idx,
        mask=mask_2d & valid,
        other=0,
    )
    slot = (block_no << LOG2_BS) + block_off
    slot = tl.where(valid, slot, -1)

    tl.store(
        global_indices_ptr + local_base + offs[None, :],
        slot,
        mask=mask_2d,
    )
    counts = tl.sum(valid.to(tl.int32), axis=1)

    lens = tl.where(is_valid != 0, counts, 0)
    tl.store(lens_ptr + token_offs, lens, mask=token_mask)


def compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert topk_indices.ndim == 2
    if is_valid_token is None:
        is_valid_token = torch.ones(
            (topk_indices.shape[0],), device=topk_indices.device, dtype=torch.int32
        )
    num_tokens, topk = topk_indices.shape
    global_indices = torch.empty_like(topk_indices, dtype=torch.int32)
    lens = torch.empty((num_tokens,), device=topk_indices.device, dtype=torch.int32)
    with torch_device_fn.device(topk_indices.device):
        grid = lambda meta: (triton.cdiv(num_tokens, meta["TPP"]),)
        if (
            _is_pow2(topk)
            and _is_pow2(block_size)
            and topk_indices.stride(1) == 1
            and topk_indices.stride(0) == topk
        ):
            _compute_global_topk_indices_and_lens_fast_kernel[grid](
                global_indices,
                lens,
                topk_indices,
                topk,
                token_to_req_indices,
                block_table,
                block_table.stride(0),
                is_valid_token,
                num_tokens,
                LOG2_BS=block_size.bit_length() - 1,
                BLOCK=topk,
            )
        else:
            _compute_global_topk_indices_and_lens_kernel[grid](
                global_indices,
                global_indices.stride(0),
                lens,
                topk_indices,
                topk_indices.stride(0),
                topk,
                token_to_req_indices,
                block_table,
                block_table.stride(0),
                block_size,
                is_valid_token,
                num_tokens,
            )
    return global_indices, lens


__all__ = [
    "compute_global_topk_indices_and_lens",
]
