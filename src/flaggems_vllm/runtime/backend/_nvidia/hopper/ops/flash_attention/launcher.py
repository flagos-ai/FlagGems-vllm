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

"""Input preparation, dispatch, and kernel execution for Hopper FA3."""

import logging

import torch

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._nvidia.hopper.ops.flash_attention.common import (
    _fa3_device_properties,
    _reset_scheduler_counter,
    _tma_strides_are_aligned,
    ensure_tma_allocator,
    is_fa3_supported,
    normalize_window,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.ops.flash_attention.config import (
    PAGED_GATHER_AUTO,
    _ceil_div,
    _next_power_of_2,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.ops.flash_attention.decode import (
    _zero_decode_split_counters,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.ops.flash_attention.direct import (
    launch_direct,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.ops.flash_attention.persistent import (
    launch_persistent,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.ops.flash_attention.scheduling import (
    MIN_EXPLICIT_SPLIT_K,
    select_decode_schedule,
    select_paged_schedule,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.ops.flash_attention.split_combine import (
    combine_persistent_split_kv,
)
from flaggems_vllm.utils.shape_utils import can_use_int32_index

logger = logging.getLogger(__name__)

# Split-KV's dynamic tile scheduler needs one graph-stable counter.  Keying the
# tiny workspace by a caller-stable tensor address (out when supplied, otherwise
# q) keeps it private to TLE, reusable across eager calls and CUDA Graph capture,
# and independent of any vLLM metadata tensor.
_split_scheduler_counters: dict[tuple, torch.Tensor] = {}


def _get_split_scheduler_counter(owner: torch.Tensor, num_splits: int):
    key = (
        owner.device.type,
        owner.device.index,
        owner.data_ptr(),
        num_splits,
    )
    counter = _split_scheduler_counters.get(key)
    if counter is None:
        # Initialize once on the current stream. The combine kernel resets
        # the counter for the next invocation or CUDA Graph replay.
        counter = torch.empty((1,), dtype=torch.int32, device=owner.device)
        _reset_scheduler_counter[(1,)](counter, num_warps=1)
        _split_scheduler_counters[key] = counter
    return counter


# Counter slots are zeroed before capture and are private to an output/stream.
# A new graph may allocate different Q/O addresses than its eager warmup; using
# a primed pool avoids capturing a reset launch for every attention layer.
_decode_split_counters: dict[tuple, torch.Tensor] = {}
_decode_counter_pools: dict[tuple, list] = {}
_DECODE_COUNTER_POOL_SIZE = 65536


def _get_decode_split_counter(output: torch.Tensor, num_splits: int, size: int):
    # Keep pooled pointers 16-byte aligned so new outputs do not create
    # alignment-only JIT/cache variants.
    size = max(4, _next_power_of_2(size))
    device_key = (output.device.type, output.device.index)
    stream = torch_device_fn.current_stream(output.device)
    capturing = torch_device_fn.is_current_stream_capturing()
    pools = _decode_counter_pools.setdefault(device_key, [])
    pool = pools[-1] if pools else None
    # Leave ample zeroed slots for graph outputs whose addresses are not known
    # during warmup. Retain old chunks because live graphs can still use them.
    if not capturing and (
        pool is None
        or _DECODE_COUNTER_POOL_SIZE - pool["next"] < _DECODE_COUNTER_POOL_SIZE // 2
    ):
        storage = torch.empty(
            (_DECODE_COUNTER_POOL_SIZE,), dtype=torch.int32, device=output.device
        )
        _zero_decode_split_counters[(_ceil_div(_DECODE_COUNTER_POOL_SIZE, 256),)](
            storage, _DECODE_COUNTER_POOL_SIZE, num_warps=1
        )
        ready_event = torch_device_fn.Event()
        ready_event.record(stream)
        # Publish only initialized slots: graph replay may use any stream, and
        # capture cannot query/wait on pending work outside that graph. This is
        # a one-time setup cost per pool, outside capture and the replay path.
        ready_event.synchronize()
        pool = {"storage": storage, "next": 0}
        pools.append(pool)
    key = (*device_key, output.data_ptr(), stream.cuda_stream, num_splits, size)
    counter = _decode_split_counters.get(key)
    if counter is None:
        if pool is not None and pool["next"] + size <= _DECODE_COUNTER_POOL_SIZE:
            start = pool["next"]
            pool["next"] += size
            counter = pool["storage"][start : start + size]
        else:
            # An unprimed or unusually large capture gets a private reset.
            # Never reset a shared pool inside a graph: another graph may be
            # using other slots when that reset is replayed.
            counter = torch.empty((size,), dtype=torch.int32, device=output.device)
            _zero_decode_split_counters[(_ceil_div(size, 256),)](
                counter, size, num_warps=1
            )
        _decode_split_counters[key] = counter
    return counter


def launch_attention(
    *,
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k,
    seqused_k,
    q_v,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    return_attn_probs,
    return_softmax_lse,
    block_table,
    out,
    num_splits,
    s_aux,
    cp_world_size,
    cp_rank,
    cp_tot_seqused_k,
):
    """Prepare this call, select a cached route, and execute its kernels."""

    if not is_fa3_supported(q.device):
        raise RuntimeError("fa_version=3 requires Hopper SM90 and FlagTree TLE >= 3.6")
    if (
        dropout_p != 0
        or q_v is not None
        or return_attn_probs
        or cp_world_size != 1
        or cp_rank != 0
        or cp_tot_seqused_k is not None
    ):
        raise NotImplementedError(
            "TLE FA3 does not support dropout, q_v, attention probabilities, "
            "or context parallelism"
        )

    total_q, num_heads, head_dim = q.shape
    num_heads_k = k.size(-2)
    is_paged = block_table is not None
    batch_size = cu_seqlens_q.numel() - 1
    requested_window = (-1, -1) if window_size is None else window_size

    # The kernels share Q/K/V dtype and K/V addressing; no implicit copies.
    assert q.dtype in (torch.float16, torch.bfloat16) and q.dtype == k.dtype == v.dtype
    assert q.device == k.device == v.device
    assert (
        k.shape == v.shape
        and k.ndim == (4 if is_paged else 3)
        and k.size(-1) == head_dim
    )
    assert 32 <= head_dim <= 256 and head_dim % 8 == 0
    assert num_heads > 0 and num_heads_k > 0 and num_heads % num_heads_k == 0
    assert all(t.stride(-1) == 1 for t in (q, k, v)) and k.stride() == v.stride()
    assert max_seqlen_q > 0 and max_seqlen_k > 0 and 0 <= num_splits <= 32

    # Prefix/length tensors are read as contiguous int32 vectors by every path.
    assert (seqused_k is not None) == is_paged and (cu_seqlens_k is None) == is_paged
    lengths_k = seqused_k if is_paged else cu_seqlens_k
    assert batch_size > 0 and cu_seqlens_q.ndim == 1 and lengths_k is not None
    assert lengths_k.shape == (batch_size if is_paged else batch_size + 1,)
    assert all(
        t.dtype == torch.int32 and t.device == q.device and t.is_contiguous()
        for t in (cu_seqlens_q, lengths_k)
    )
    if is_paged:
        block_size = k.size(1)
        assert block_size >= 16 and block_size & (block_size - 1) == 0
        if k.stride(0) % k.stride(1) != 0:
            # Paged K/V tiles are addressed in whole token rows (TMA descriptors
            # and gather indices), which requires a row-aligned page stride.
            raise NotImplementedError(
                "TLE FA3 requires the paged KV page stride to be a multiple of "
                "the token stride; use fa_version=2 for this cache layout"
            )
        assert (
            block_table.ndim == 2
            and block_table.size(0) == batch_size
            and block_table.dtype == torch.int32
            and block_table.device == q.device
            and block_table.stride(-1) == 1
        )
    if alibi_slopes is not None:
        assert (
            alibi_slopes.shape in ((num_heads,), (batch_size, num_heads))
            and alibi_slopes.dtype == torch.float32
            and alibi_slopes.device == q.device
            and alibi_slopes.stride(-1) == 1
        )
    if s_aux is not None:
        assert (
            s_aux.shape == (num_heads,)
            and num_heads <= 64
            and s_aux.dtype == torch.bfloat16
            and s_aux.device == q.device
            and s_aux.is_contiguous()
        )
    if out is not None:
        assert (
            out.shape == q.shape
            and out.dtype == q.dtype
            and out.device == q.device
            and out.stride(-1) == 1
        )

    # The seqused-K specialization never reads cu_seqlens_k.  Reuse the always
    # valid Q prefix tensor instead of allocating a device-side ABI placeholder.
    cu_seqlens_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    is_causal, is_local, window_left, window_right = normalize_window(
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        head_dim=q.size(-1),
        is_paged=is_paged,
        causal=causal,
        window_size_left=requested_window[0],
        window_size_right=requested_window[1],
    )

    block_size = k.size(1) if is_paged else 1
    num_pages = k.size(0) if is_paged else 0
    # Dense kernels never read the page table. Reuse the int32 prefix tensor
    # as its ABI placeholder without allocating another tensor.
    page_table = cu_seqlens_q if block_table is None else block_table

    scale = head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
    softcap = float(softcap)
    if softcap > 0.0:
        is_softcap = True
        adjusted_softcap = scale / softcap
        adjusted_scale_softmax = softcap
    else:
        is_softcap = False
        adjusted_softcap = 0.0
        adjusted_scale_softmax = scale

    device_properties = _fa3_device_properties(q.device)
    qo_tma_aligned = _tma_strides_are_aligned(q) and (
        out is None or _tma_strides_are_aligned(out)
    )
    # K and V already have matching dtypes and strides.
    kv_tma_aligned = _tma_strides_are_aligned(k)

    # Registration mutates global Triton runtime state, so do it only after all
    # argument validation and canonical tensor preparation have succeeded.
    ensure_tma_allocator()

    num_sms = device_properties.multi_processor_count
    is_alibi = alibi_slopes is not None
    is_s_aux = s_aux is not None
    # Classify this request directly from the validated scalar metadata.
    gqa_ratio = num_heads // num_heads_k
    plain_scores = not (is_local or is_alibi or is_softcap)
    if batch_size == 1:
        population = "single"
    elif batch_size > 31 * 32:
        population = "oversized"
    elif total_q == batch_size * max_seqlen_q:
        population = "uniform"
    else:
        population = (
            "ragged" if 4 * total_q <= 3 * batch_size * max_seqlen_q else "mixed"
        )

    if max_seqlen_q == 1:
        workload = "token_query"
    elif is_paged and max_seqlen_q <= 8:
        workload = "speculative_query"
    elif max_seqlen_q >= 1024 and total_q >= 128 * batch_size:
        workload = "prefill"
    elif is_paged:
        serving_prefill = (population == "ragged" and max_seqlen_q >= 256) or (
            population == "single" and total_q == max_seqlen_q and max_seqlen_q >= 512
        )
        workload = "serving_prefill" if serving_prefill else "short_query"
    else:
        workload = (
            "short_query"
            if max_seqlen_q * gqa_ratio <= 128 or total_q <= 64 * batch_size
            else "prefill"
        )

    if workload == "token_query" and batch_size >= 32 and max_seqlen_k >= 4096:
        input_profile = "throughput_decode"
    elif (
        population == "single"
        and total_q == max_seqlen_q
        and max_seqlen_q >= 8192
        and max_seqlen_k >= 8192
    ):
        input_profile = "large_single_prefill"
    elif total_q * max_seqlen_k * num_heads * head_dim > 1 << 26:
        input_profile = "heavy"
    else:
        input_profile = "general"
    paged_prefill_candidate = is_paged and workload == "prefill"
    paged_d256_prefill = (
        is_paged
        and head_dim == 256
        and (block_size, gqa_ratio) in {(16, 4), (32, 8)}
        and workload in {"prefill", "serving_prefill"}
        and population != "oversized"
        and input_profile in {"heavy", "large_single_prefill"}
        and is_causal
        and plain_scores
    )
    # Non-power-of-two packing is safe for direct decode, but not TMA Q/O tiles.
    pack_gqa = (
        1 < gqa_ratio <= 16
        and (gqa_ratio & (gqa_ratio - 1) == 0 or workload == "token_query")
        and (
            head_dim <= 128
            or (workload == "token_query" and head_dim in (192, 256))
            or paged_d256_prefill
        )
    )
    gqa_decode = pack_gqa and workload == "token_query"
    if gqa_decode and not is_local:
        # The left bound is unused outside local attention. Normalization may
        # otherwise specialize this value on every new causal decode length.
        window_left = 0
    direct_num_splits = 1
    explicit_pow2_decode = (
        gqa_decode and num_splits > 1 and gqa_ratio & (gqa_ratio - 1) == 0
    )
    if gqa_decode:
        decode_pack_gqa, direct_num_splits = select_decode_schedule(
            batch_size,
            num_heads_k,
            max_seqlen_k,
            num_sms,
            num_splits,
            plain_scores and not is_s_aux,
        )
        if gqa_ratio & (gqa_ratio - 1) != 0:
            pack_gqa = decode_pack_gqa
    pack_factor = gqa_ratio if pack_gqa else 1
    if is_paged:
        (
            kernel,
            paged_kv_non_tma,
            pack_factor,
            ragged_scheduler,
            paged_gather_mode,
            persistent_num_splits,
            explicit_split_k_chunk,
        ) = select_paged_schedule(
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            total_q,
            num_heads_k,
            head_dim,
            gqa_ratio,
            block_size,
            num_sms,
            q.dtype == torch.float16,
            is_causal,
            is_local or is_alibi,
            plain_scores,
            is_s_aux,
            qo_tma_aligned,
            kv_tma_aligned,
            population,
            workload,
            input_profile,
            pack_factor,
            paged_d256_prefill,
            num_splits,
        )
        pack_gqa = pack_factor > 1
    else:
        kernel = "long" if workload == "prefill" else "direct"
        # Short queries stay on direct: a heavy batch of them leaves most
        # rows of each 128-row persistent tile empty.
        if (
            gqa_ratio == 1
            and plain_scores
            and not is_causal
            and head_dim in (64, 128)
            and population in {"single", "uniform"}
            and max_seqlen_q >= 64
        ):
            kernel = (
                "long"
                if input_profile in {"heavy", "large_single_prefill"}
                else "direct"
            )
        if not (qo_tma_aligned and kv_tma_aligned):
            kernel = "direct"
        ragged_scheduler = population == "ragged"
        paged_kv_non_tma = False
        paged_gather_mode = PAGED_GATHER_AUTO
        persistent_num_splits = 0
        explicit_split_k_chunk = MIN_EXPLICIT_SPLIT_K

    # Compact-page decode can use direct splitting when it fills a wave.
    # A lower explicit cap favors the persistent pipeline for long KV.
    if explicit_pow2_decode and (
        batch_size * num_heads_k * 4 >= num_sms
        or (
            kernel != "direct"
            and (
                block_size != 16
                or batch_size * num_heads_k * direct_num_splits * 10 < num_sms * 9
            )
        )
    ):
        direct_num_splits = 1
    fused_split_kv = explicit_pow2_decode and direct_num_splits > 1

    if direct_num_splits > 1:
        # Add KV parallelism to the decode cases selected above.
        kernel = "direct"
        paged_kv_non_tma = is_paged
        paged_gather_mode = PAGED_GATHER_AUTO
        ragged_scheduler = population == "ragged"
        persistent_num_splits = 0
        explicit_split_k_chunk = MIN_EXPLICIT_SPLIT_K

    heads_in_l2 = "l2_auto" if paged_d256_prefill else int(is_causal or is_local)
    tile_rows = (total_q if ragged_scheduler else max_seqlen_q) * pack_factor
    tile_heads = num_heads_k if pack_gqa else num_heads
    tile_groups = tile_heads if ragged_scheduler else batch_size * tile_heads
    # D256 prefill also admits BM64; other workloads use the BM128 threshold.
    work_items = (
        _ceil_div(tile_rows, 64 if paged_d256_prefill else 128)
        + (batch_size - 1 if ragged_scheduler else 0)
    ) * tile_groups
    dynamic_scheduler = (is_causal or is_local) and work_items > num_sms
    split_kv = persistent_num_splits > 1
    if split_kv:
        dynamic_scheduler = population != "single"
    max_splits = (
        direct_num_splits if kernel == "direct" else max(1, persistent_num_splits)
    )

    q_device = q.device
    alibi_slopes_batch_stride = (
        alibi_slopes.stride(0)
        if alibi_slopes is not None and alibi_slopes.ndim == 2
        else 0
    )
    page_table_batch_stride = page_table.stride(0)
    # Keep Q as the counter owner when out was omitted, before allocating out.
    scheduler_owner = q if out is None else out

    # Exact launch geometry and workspace remain local to this call.
    effective_max_q = max_seqlen_q * pack_factor
    effective_num_heads = num_heads_k if pack_gqa else num_heads

    k_batch_stride = 0
    if is_paged:
        # Reuse the otherwise-unused K batch-stride lane for the physical page
        # stride, supporting both NHD and HND cache allocations.
        k_batch_stride = k.stride(0)
    # Widen dense K/V row offsets only when the cache exceeds the int32 span;
    # 64-bit index math costs a few percent on direct kernels.
    index_int64 = not is_paged and not (
        can_use_int32_index(k) and can_use_int32_index(v)
    )

    with torch_device_fn.device(q_device):
        if out is None:
            out = torch.empty_like(q)

        if return_softmax_lse:
            lse = torch.empty(
                (num_heads, total_q), dtype=torch.float32, device=q_device
            )
            lse_ptr = lse
        else:
            lse = None
            # Every kernel reads this pointer behind STORE_LSE.  A valid existing
            # tensor avoids allocating an output that the public API will discard.
            lse_ptr = q

        # Keep one compact FA3 ABI shared by the direct and persistent kernels.
        # K/V matching strides are validated during input preparation, so the
        # paged V descriptor can reuse k_batch_stride without a duplicate ABI
        # lane. Dropout/softmax-output placeholders from the generic FA2 ABI are
        # intentionally absent because FA3 rejects those features before launch.
        args = (
            q,
            k,
            v,
            out,
            lse_ptr,
            q.stride(-3),
            k.stride(-3),
            v.stride(-3),
            q.stride(-2),
            k.stride(-2),
            v.stride(-2),
            out.stride(-3),
            out.stride(-2),
            k_batch_stride,
            cu_seqlens_q,
            seqused_k is not None,
            cu_seqlens_k,
            seqused_k,
            batch_size,
            num_pages,
            num_heads,
            num_heads_k,
            num_heads // num_heads_k,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            is_softcap,
            adjusted_softcap,
            adjusted_scale_softmax,
            (adjusted_scale_softmax * 1.4426950408889634),
            is_causal,
            is_local,
            window_left,
            window_right,
            is_paged,
            is_alibi,
            alibi_slopes,
            alibi_slopes_batch_stride,
            is_s_aux,
            s_aux,
            total_q,
            page_table,
            page_table_batch_stride,
            block_size,
        )

        partial_out = partial_lse = scheduler_counter = None
        if max_splits > 1:
            partial_out = torch.empty(
                (max_splits, num_heads, total_q, head_dim),
                dtype=torch.float32,
                device=q_device,
            )
            partial_lse = torch.empty(
                (max_splits, num_heads, total_q),
                dtype=torch.float32,
                device=q_device,
            )
            if fused_split_kv:
                scheduler_counter = _get_decode_split_counter(
                    out, max_splits, batch_size * num_heads_k
                )
            elif kernel != "direct":
                scheduler_counter = _get_split_scheduler_counter(
                    scheduler_owner, max_splits
                )

        logger.debug(
            "kernel: flash_varlen_fwd_v3_tle route=%s max_splits=%s",
            kernel,
            max_splits,
        )
        if kernel == "direct":
            launch_direct(
                args,
                max_seqlen_k=max_seqlen_k,
                batch_size=batch_size,
                effective_max_q=effective_max_q,
                effective_num_heads=effective_num_heads,
                num_sms=num_sms,
                pack_factor=pack_factor,
                is_paged=is_paged,
                paged_prefill=paged_prefill_candidate,
                pack_gqa=pack_gqa,
                paged_gather_mode=paged_gather_mode,
                paged_kv_non_tma=paged_kv_non_tma,
                store_lse=return_softmax_lse,
                total_q=total_q,
                ragged_scheduler=ragged_scheduler,
                heads_in_l2=heads_in_l2,
                partial_out=partial_out,
                partial_lse=partial_lse,
                max_splits=max_splits,
                gqa_decode=gqa_decode,
                fused_split_kv=fused_split_kv,
                split_counter=scheduler_counter,
                index_int64=index_int64,
            )
        else:
            paged_prefill_profile = paged_d256_prefill and not split_kv
            launch_persistent(
                args,
                output=out,
                total_q=total_q,
                head_size=head_dim,
                num_sms=num_sms,
                max_seqlen_k=max_seqlen_k,
                batch_size=batch_size,
                num_heads=num_heads,
                num_heads_k=num_heads_k,
                effective_max_q=effective_max_q,
                effective_num_heads=effective_num_heads,
                pack_factor=pack_factor,
                pack_gqa=pack_gqa,
                paged_gather_mode=paged_gather_mode,
                paged_kv_non_tma=paged_kv_non_tma,
                # Logical KV extents are derived from the selected profile.
                exact_paged_kv_tiles=paged_prefill_profile and paged_kv_non_tma,
                ragged_scheduler=ragged_scheduler,
                heads_in_l2=heads_in_l2,
                dynamic_scheduler=dynamic_scheduler,
                store_lse=return_softmax_lse,
                paged_prefill_profile=paged_prefill_profile,
                partial_out=partial_out,
                partial_lse=partial_lse,
                max_splits=max_splits,
                explicit_split_k_chunk=explicit_split_k_chunk,
                scheduler_counter=scheduler_counter,
            )
        if max_splits > 1 and not fused_split_kv:
            combine_persistent_split_kv(
                out,
                lse_ptr,
                partial_out,
                partial_lse,
                scheduler_counter,
                seqused_k,
                cu_seqlens_q,
                batch_size=batch_size,
                num_heads=num_heads,
                num_heads_k=num_heads_k,
                block_size=block_size,
                max_seqlen_q=max_seqlen_q,
                head_dim=head_dim,
                total_q=total_q,
                max_splits=max_splits,
                explicit_split_k_chunk=(
                    128 if kernel == "direct" else explicit_split_k_chunk
                ),
                store_lse=return_softmax_lse,
                fixed_splits=kernel == "direct",
            )

    return (out, lse) if return_softmax_lse else out


__all__ = ["launch_attention"]
