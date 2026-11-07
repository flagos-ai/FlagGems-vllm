# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Pack and restore batch-sharded sampling outputs around vLLM collectives."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import has_triton_tle_attrs, libentry, libtuner

if TYPE_CHECKING:
    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.sample.batch_shard import BatchShardMetadata
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

_HAS_LAYOUT = runtime.device.vendor_name == "nvidia" and has_triton_tle_attrs(
    ("gpu.set_layout", "gpu.BlockEncoding", "gpu.SlicedEncoding")
)
if _HAS_LAYOUT:
    import triton.experimental.tle.language as tle


def _prune_request_tiles(configs, named_args, **kwargs):
    # Bound live elements per thread. Large multi-request tiles otherwise
    # consume too many registers and expose cold-memory latency on H100.
    width = named_args.get("BLOCK_SIZE", kwargs.get("BLOCK_SIZE"))
    # Wide rows need enough warps to overlap memory requests. A single warp
    # won hot-cache tuning at width 128 but regressed in cold-cache profiling.
    min_warps = min(4, max(1, width // 32))
    return [
        config
        for config in configs
        if config.num_warps >= min_warps
        and config.num_warps * 32 <= max(32, config.kwargs["BLOCK_ROWS"] * width)
        and (
            config.kwargs["BLOCK_ROWS"] == 1
            or config.kwargs["BLOCK_ROWS"] * width <= 8 * 32 * config.num_warps
        )
    ]


def _prune_sampler_tiles(configs, named_args, **kwargs):
    # Keep a one-request launch for widths beyond the grouped tile budget.
    # The regular candidate set stays unchanged for narrower rows.
    regular = [
        config
        for config in configs
        if config.kwargs["BLOCK_ROWS"] != 1 or config.num_warps == 1
    ]
    return _prune_request_tiles(regular, named_args, **kwargs) or _prune_request_tiles(
        configs, named_args, **kwargs
    )


@triton.jit
def _request_tile(
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WARPS: tl.constexpr,
    USE_LAYOUT: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_SIZE)
    if USE_LAYOUT:
        lane_cols: tl.constexpr = min(BLOCK_SIZE, 32)
        warp_cols: tl.constexpr = min(WARPS, max(1, BLOCK_SIZE // 32))
        layout: tl.constexpr = tle.gpu.BlockEncoding(
            [1, 1],
            [32 // lane_cols, lane_cols],
            [WARPS // warp_cols, warp_cols],
            [1, 0],
        )
        rows = tle.gpu.set_layout(rows, tle.gpu.SlicedEncoding(1, layout))
        cols = tle.gpu.set_layout(cols, tle.gpu.SlicedEncoding(0, layout))
    return rows, cols


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("pack_sampler_output"),
    prune_configs_by={"early_config_prune": _prune_sampler_tiles},
    key=["NUM_REQS", "max_num_logits_per_req"],
    warmup=5,
    rep=30,
    use_cuda_graph=runtime.device.name == "cuda",
)
@triton.jit(
    do_not_specialize=["max_num_logits_per_req", "num_src_cols", "packed_stride"]
)
def _pack_sampler_output_kernel(
    packed_ptr,
    packed_stride,
    sampled_token_ids_ptr,
    sampled_token_ids_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    num_nans_ptr,
    local_cu_num_logits_ptr,
    max_num_logits_per_req,
    num_src_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_REQS,
    BLOCK_ROWS: tl.constexpr,
    WARPS: tl.constexpr,
    USE_LAYOUT: tl.constexpr,
):
    req, cols = _request_tile(BLOCK_ROWS, BLOCK_SIZE, WARPS, USE_LAYOUT)
    req = tl.program_id(0) * BLOCK_ROWS + req
    valid = req < NUM_REQS
    row_ptr = packed_ptr + req * packed_stride
    token_ids = tl.load(
        sampled_token_ids_ptr + req[:, None] * sampled_token_ids_stride + cols[None, :],
        mask=valid[:, None] & (cols[None, :] < num_src_cols),
        other=0,
    )
    tl.store(
        row_ptr[:, None] + cols[None, :],
        token_ids,
        mask=valid[:, None] & (cols[None, :] < max_num_logits_per_req),
    )
    num_sampled = tl.load(num_sampled_ptr + req, mask=valid, other=0)
    num_rejected = tl.load(num_rejected_ptr + req, mask=valid, other=0)
    tl.store(row_ptr + max_num_logits_per_req, num_sampled.to(tl.int64), mask=valid)
    tl.store(
        row_ptr + max_num_logits_per_req + 1, num_rejected.to(tl.int64), mask=valid
    )
    if num_nans_ptr is not None:
        start = tl.load(local_cu_num_logits_ptr + req, mask=valid, other=0)
        end = tl.load(local_cu_num_logits_ptr + req + 1, mask=valid, other=0)
        nans = tl.load(
            num_nans_ptr + start[:, None] + cols[None, :],
            mask=valid[:, None] & (cols[None, :] < (end - start)[:, None]),
            other=0,
        )
        tl.store(
            row_ptr + max_num_logits_per_req + 2,
            tl.sum(nans.to(tl.int64), axis=1),
            mask=valid,
        )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("unpack_gathered_output"),
    key=[],
    use_cuda_graph=runtime.device.name == "cuda",
)
@triton.jit(do_not_specialize=["max_num_logits_per_req", "gathered_stride"])
def _unpack_gathered_output_kernel(
    gathered_ptr,
    gathered_stride,
    gathered_src_indices_ptr,
    sampled_token_ids_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    num_nans_ptr,
    max_num_logits_per_req,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    src = tl.load(gathered_src_indices_ptr + req_idx)
    row_ptr = gathered_ptr + src * gathered_stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < max_num_logits_per_req

    token_ids = tl.load(row_ptr + cols, mask=mask)
    tl.store(
        sampled_token_ids_ptr + req_idx * max_num_logits_per_req + cols,
        token_ids,
        mask=mask,
    )
    num_sampled = tl.load(row_ptr + max_num_logits_per_req)
    tl.store(num_sampled_ptr + req_idx, num_sampled.to(tl.int32))
    num_rejected = tl.load(row_ptr + max_num_logits_per_req + 1)
    tl.store(num_rejected_ptr + req_idx, num_rejected.to(tl.int32))
    if num_nans_ptr is not None:
        num_nans = tl.load(row_ptr + max_num_logits_per_req + 2)
        tl.store(num_nans_ptr + req_idx, num_nans.to(tl.int32))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("pack_logprobs"),
    prune_configs_by={"early_config_prune": _prune_request_tiles},
    key=["NUM_REQS", "max_num_logits_per_req", "num_logprob_cols"],
    warmup=5,
    rep=30,
    use_cuda_graph=runtime.device.name == "cuda",
)
@triton.jit(
    do_not_specialize=[
        "max_num_logits_per_req",
        "num_logprob_cols",
        "num_src_cols",
        "send_ids_stride",
        "send_logprobs_stride",
        "src_ids_stride",
        "src_logprobs_stride",
    ]
)
def _pack_logprobs_kernel(
    send_ids_ptr,
    send_ids_stride,
    send_logprobs_ptr,
    send_logprobs_stride,
    src_ids_ptr,
    src_ids_stride,
    src_logprobs_ptr,
    src_logprobs_stride,
    selected_token_ranks_ptr,
    local_cu_num_logits_ptr,
    max_num_logits_per_req,
    num_logprob_cols,
    num_src_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_REQS,
    BLOCK_ROWS: tl.constexpr,
    WARPS: tl.constexpr,
    USE_LAYOUT: tl.constexpr,
):
    if BLOCK_ROWS == 1:
        req_idx = tl.program_id(0)
        pos = tl.program_id(1)
        start = tl.load(local_cu_num_logits_ptr + req_idx)
        if pos < tl.load(local_cu_num_logits_ptr + req_idx + 1) - start:
            src = start + pos
            dst = req_idx * max_num_logits_per_req + pos
            _, cols = _request_tile(1, BLOCK_SIZE, WARPS, USE_LAYOUT)
            # A rank whose sub-batch has no draft tokens runs the regular sampler
            # while others run the rejection sampler, and their column counts can
            # differ (e.g. logprob_token_ids widening); the two masks pad or
            # truncate to the agreed width.
            src_mask = cols < num_src_cols
            dst_mask = cols < num_logprob_cols
            ids = tl.load(
                src_ids_ptr + src * src_ids_stride + cols, mask=src_mask, other=0
            )
            logprobs = tl.load(
                src_logprobs_ptr + src * src_logprobs_stride + cols,
                mask=src_mask,
                other=float("-inf"),
            )
            ids_row = send_ids_ptr + dst * send_ids_stride
            tl.store(ids_row + cols, ids.to(tl.int64), mask=dst_mask)
            rank = tl.load(selected_token_ranks_ptr + src)
            tl.store(ids_row + num_logprob_cols, rank.to(tl.int64))
            tl.store(
                send_logprobs_ptr + dst * send_logprobs_stride + cols,
                logprobs.to(tl.float32),
                mask=dst_mask,
            )
    else:
        req, cols = _request_tile(BLOCK_ROWS, BLOCK_SIZE, WARPS, USE_LAYOUT)
        req = tl.program_id(0) * BLOCK_ROWS + req
        pos = tl.program_id(1)
        start = tl.load(local_cu_num_logits_ptr + req, mask=req < NUM_REQS, other=0)
        end = tl.load(local_cu_num_logits_ptr + req + 1, mask=req < NUM_REQS, other=0)
        valid = (req < NUM_REQS) & (pos < end - start)
        src = start + pos
        dst = req * max_num_logits_per_req + pos
        src_mask = valid[:, None] & (cols[None, :] < num_src_cols)
        dst_mask = valid[:, None] & (cols[None, :] < num_logprob_cols)
        ids = tl.load(
            src_ids_ptr + src[:, None] * src_ids_stride + cols[None, :],
            mask=src_mask,
            other=0,
        )
        logprobs = tl.load(
            src_logprobs_ptr + src[:, None] * src_logprobs_stride + cols[None, :],
            mask=src_mask,
            other=float("-inf"),
        )
        ids_row = send_ids_ptr + dst * send_ids_stride
        tl.store(ids_row[:, None] + cols[None, :], ids.to(tl.int64), mask=dst_mask)
        rank = tl.load(selected_token_ranks_ptr + src, mask=valid, other=0)
        tl.store(ids_row + num_logprob_cols, rank.to(tl.int64), mask=valid)
        tl.store(
            send_logprobs_ptr + dst[:, None] * send_logprobs_stride + cols[None, :],
            logprobs.to(tl.float32),
            mask=dst_mask,
        )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("unpack_logprobs"),
    prune_configs_by={"early_config_prune": _prune_request_tiles},
    key=["NUM_REQS", "max_num_logits_per_req", "num_logprob_cols"],
    warmup=5,
    rep=30,
    use_cuda_graph=runtime.device.name == "cuda",
)
@triton.jit(
    do_not_specialize=[
        "max_num_logits_per_req",
        "num_logprob_cols",
        "gathered_ids_stride",
        "gathered_logprobs_stride",
    ]
)
def _unpack_logprobs_kernel(
    gathered_ids_ptr,
    gathered_ids_stride,
    gathered_logprobs_ptr,
    gathered_logprobs_stride,
    gathered_src_indices_ptr,
    cu_num_logits_ptr,
    logprob_token_ids_ptr,
    logprobs_ptr,
    selected_token_ranks_ptr,
    max_num_logits_per_req,
    num_logprob_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_REQS,
    BLOCK_ROWS: tl.constexpr,
    WARPS: tl.constexpr,
    USE_LAYOUT: tl.constexpr,
):
    if BLOCK_ROWS == 1:
        req_idx = tl.program_id(0)
        pos = tl.program_id(1)
        start = tl.load(cu_num_logits_ptr + req_idx)
        if pos < tl.load(cu_num_logits_ptr + req_idx + 1) - start:
            src = (
                tl.load(gathered_src_indices_ptr + req_idx) * max_num_logits_per_req
                + pos
            )
            dst = start + pos
            _, cols = _request_tile(1, BLOCK_SIZE, WARPS, USE_LAYOUT)
            mask = cols < num_logprob_cols
            src_ids_row = gathered_ids_ptr + src * gathered_ids_stride
            ids = tl.load(src_ids_row + cols, mask=mask)
            tl.store(
                logprob_token_ids_ptr + dst * num_logprob_cols + cols, ids, mask=mask
            )
            rank = tl.load(src_ids_row + num_logprob_cols)
            tl.store(selected_token_ranks_ptr + dst, rank)
            logprobs = tl.load(
                gathered_logprobs_ptr + src * gathered_logprobs_stride + cols, mask=mask
            )
            tl.store(logprobs_ptr + dst * num_logprob_cols + cols, logprobs, mask=mask)
    else:
        req, cols = _request_tile(BLOCK_ROWS, BLOCK_SIZE, WARPS, USE_LAYOUT)
        req = tl.program_id(0) * BLOCK_ROWS + req
        pos = tl.program_id(1)
        start = tl.load(cu_num_logits_ptr + req, mask=req < NUM_REQS, other=0)
        end = tl.load(cu_num_logits_ptr + req + 1, mask=req < NUM_REQS, other=0)
        valid = (req < NUM_REQS) & (pos < end - start)
        idx = tl.load(gathered_src_indices_ptr + req, mask=req < NUM_REQS, other=0)
        src = idx * max_num_logits_per_req + pos
        dst = start + pos
        mask = valid[:, None] & (cols[None, :] < num_logprob_cols)
        src_ids_row = gathered_ids_ptr + src * gathered_ids_stride
        ids = tl.load(src_ids_row[:, None] + cols[None, :], mask=mask, other=0)
        tl.store(
            logprob_token_ids_ptr + dst[:, None] * num_logprob_cols + cols[None, :],
            ids,
            mask=mask,
        )
        rank = tl.load(src_ids_row + num_logprob_cols, mask=valid, other=0)
        tl.store(selected_token_ranks_ptr + dst, rank, mask=valid)
        logprobs = tl.load(
            gathered_logprobs_ptr
            + src[:, None] * gathered_logprobs_stride
            + cols[None, :],
            mask=mask,
            other=0,
        )
        tl.store(
            logprobs_ptr + dst[:, None] * num_logprob_cols + cols[None, :],
            logprobs,
            mask=mask,
        )


def _gather_logprobs_tensors(
    local_output: SamplerOutput | None,
    metadata: BatchShardMetadata,
    global_batch: InputBatch,
    local_batch: InputBatch,
    logprobs_dims: tuple[int, int],
    device: torch.device,
) -> LogprobsTensors:
    from vllm.distributed.communication_op import tensor_model_parallel_all_gather
    from vllm.v1.outputs import LogprobsTensors

    num_logprobs, max_token_ids = logprobs_dims
    num_logprob_cols = 1 + max(num_logprobs, max_token_ids)
    max_num_logits_per_req = metadata.max_num_logits_per_req
    num_send_rows = metadata.max_num_reqs_per_rank * max_num_logits_per_req
    block_size = triton.next_power_of_2(num_logprob_cols)

    # Padded slots are never read back: gathered_src_indices only addresses
    # rows that an owner rank wrote.
    send_ids_ranks = torch.empty(
        num_send_rows, num_logprob_cols + 1, dtype=torch.int64, device=device
    )
    send_logprobs = torch.empty(
        num_send_rows, num_logprob_cols, dtype=torch.float32, device=device
    )
    if local_output is not None and local_output.logprobs_tensors is not None:
        lp = local_output.logprobs_tensors
        _pack_logprobs_kernel[
            lambda meta: (
                triton.cdiv(metadata.num_local_reqs, meta["BLOCK_ROWS"]),
                max_num_logits_per_req,
            )
        ](
            send_ids_ranks,
            send_ids_ranks.stride(0),
            send_logprobs,
            send_logprobs.stride(0),
            lp.logprob_token_ids,
            lp.logprob_token_ids.stride(0),
            lp.logprobs,
            lp.logprobs.stride(0),
            lp.selected_token_ranks,
            local_batch.cu_num_logits,
            max_num_logits_per_req,
            num_logprob_cols,
            lp.logprob_token_ids.shape[1],
            BLOCK_SIZE=block_size,
            NUM_REQS=metadata.num_local_reqs,
            USE_LAYOUT=_HAS_LAYOUT,
        )

    gathered_ids_ranks = tensor_model_parallel_all_gather(send_ids_ranks, dim=0)
    gathered_logprobs = tensor_model_parallel_all_gather(send_logprobs, dim=0)

    num_reqs = global_batch.num_reqs
    num_logits = int(global_batch.cu_num_logits_np[-1])
    logprob_token_ids = torch.empty(
        num_logits, num_logprob_cols, dtype=torch.int64, device=device
    )
    logprobs = torch.empty(
        num_logits, num_logprob_cols, dtype=torch.float32, device=device
    )
    selected_token_ranks = torch.empty(num_logits, dtype=torch.int64, device=device)
    _unpack_logprobs_kernel[
        lambda meta: (
            triton.cdiv(num_reqs, meta["BLOCK_ROWS"]),
            max_num_logits_per_req,
        )
    ](
        gathered_ids_ranks,
        gathered_ids_ranks.stride(0),
        gathered_logprobs,
        gathered_logprobs.stride(0),
        metadata.gathered_src_indices,
        global_batch.cu_num_logits,
        logprob_token_ids,
        logprobs,
        selected_token_ranks,
        max_num_logits_per_req,
        num_logprob_cols,
        BLOCK_SIZE=block_size,
        NUM_REQS=num_reqs,
        USE_LAYOUT=_HAS_LAYOUT,
    )
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=selected_token_ranks,
        cu_num_generated_tokens=(
            global_batch.cu_num_logits_np.tolist() if num_logits != num_reqs else None
        ),
    )


def gather_sampler_output(
    local_output: SamplerOutput | None,
    metadata: BatchShardMetadata,
    device: torch.device,
    global_batch: InputBatch,
    local_batch: InputBatch,
    gather_num_nans: bool = False,
    logprobs_dims: tuple[int, int] | None = None,
) -> SamplerOutput:
    from vllm.distributed.communication_op import tensor_model_parallel_all_gather
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    if device.type == "cpu":
        raise NotImplementedError("gather_sampler_output requires an accelerator")

    max_num_logits_per_req = metadata.max_num_logits_per_req
    num_packed_cols = max_num_logits_per_req + 2 + (1 if gather_num_nans else 0)
    block_size = triton.next_power_of_2(max_num_logits_per_req)

    # Pack the sampler output tensors (excluding logprobs) into a single
    # tensor for the all-gather.
    packed = torch.empty(
        metadata.max_num_reqs_per_rank,
        num_packed_cols,
        dtype=torch.int64,
        device=device,
    )
    if local_output is not None:
        assert not gather_num_nans or local_output.num_nans is not None
        num_src_cols = min(
            local_output.sampled_token_ids.shape[1], max_num_logits_per_req
        )
        _pack_sampler_output_kernel[
            lambda meta: (triton.cdiv(metadata.num_local_reqs, meta["BLOCK_ROWS"]),)
        ](
            packed,
            packed.stride(0),
            local_output.sampled_token_ids,
            local_output.sampled_token_ids.stride(0),
            local_output.num_sampled,
            local_output.num_rejected,
            local_output.num_nans if gather_num_nans else None,
            local_batch.cu_num_logits if gather_num_nans else None,
            max_num_logits_per_req,
            num_src_cols,
            BLOCK_SIZE=block_size,
            NUM_REQS=metadata.num_local_reqs,
            USE_LAYOUT=_HAS_LAYOUT,
        )

    # All-gather the sampler output tensors (excluding logprobs).
    gathered = tensor_model_parallel_all_gather(packed, dim=0)
    # All-gather the logprobs tensors.
    logprobs_tensors = None
    if logprobs_dims is not None:
        logprobs_tensors = _gather_logprobs_tensors(
            local_output, metadata, global_batch, local_batch, logprobs_dims, device
        )

    # Unpack the gathered tensor into a sampler output object.
    num_reqs = metadata.gathered_src_indices.shape[0]
    sampled_token_ids = torch.empty(
        num_reqs, max_num_logits_per_req, dtype=torch.int64, device=device
    )
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=device)
    num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=device)
    num_nans = (
        torch.empty(num_reqs, dtype=torch.int32, device=device)
        if gather_num_nans
        else None
    )
    _unpack_gathered_output_kernel[(num_reqs,)](
        gathered,
        gathered.stride(0),
        metadata.gathered_src_indices,
        sampled_token_ids,
        num_sampled,
        num_rejected,
        num_nans,
        max_num_logits_per_req,
        BLOCK_SIZE=block_size,
    )
    return SamplerOutput(
        sampled_token_ids=sampled_token_ids,
        logprobs_tensors=logprobs_tensors,
        num_nans=num_nans,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )
