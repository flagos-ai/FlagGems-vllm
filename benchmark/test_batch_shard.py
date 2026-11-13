# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Preallocated pack/unpack kernel benchmarks; collective time is excluded."""

from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

from benchmark import base
from flaggems_vllm import device as DEVICE
from flaggems_vllm import runtime
from flaggems_vllm.ops import batch_shard as impl

pytestmark = [
    pytest.mark.gather_sampler_output,
    pytest.mark.skipif(
        DEVICE == "cpu" or not runtime.torch_device_fn.is_available(),
        reason=f"requires an available {DEVICE} accelerator",
    ),
]

# Unchanged reference kernels from vLLM commit
# 947dd62c0700851a79b0a8f32cb823e25a003b31,
# vllm/v1/worker/gpu/sample/batch_shard.py.


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
):
    req_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_ptr = packed_ptr + req_idx * packed_stride

    token_ids = tl.load(
        sampled_token_ids_ptr + req_idx * sampled_token_ids_stride + cols,
        mask=cols < num_src_cols,
        other=0,
    )
    tl.store(row_ptr + cols, token_ids, mask=cols < max_num_logits_per_req)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    tl.store(row_ptr + max_num_logits_per_req, num_sampled.to(tl.int64))
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    tl.store(row_ptr + max_num_logits_per_req + 1, num_rejected.to(tl.int64))
    if num_nans_ptr is not None:
        # num_nans is per logits row; reduce it per request. The replicated
        # samplers report per-row counts that downstream zips per request,
        # so a per-request sum is the well-defined equivalent.
        start = tl.load(local_cu_num_logits_ptr + req_idx)
        num_req_logits = tl.load(local_cu_num_logits_ptr + req_idx + 1) - start
        nans = tl.load(num_nans_ptr + start + cols, mask=cols < num_req_logits, other=0)
        tl.store(
            row_ptr + max_num_logits_per_req + 2, tl.sum(nans.to(tl.int64), axis=0)
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
):
    req_idx = tl.program_id(0)
    pos = tl.program_id(1)
    start = tl.load(local_cu_num_logits_ptr + req_idx)
    if pos < tl.load(local_cu_num_logits_ptr + req_idx + 1) - start:
        src = start + pos
        dst = req_idx * max_num_logits_per_req + pos
        cols = tl.arange(0, BLOCK_SIZE)
        # A rank whose sub-batch has no draft tokens runs the regular sampler
        # while others run the rejection sampler, and their column counts can
        # differ (e.g. logprob_token_ids widening); the two masks pad or
        # truncate to the agreed width.
        src_mask = cols < num_src_cols
        dst_mask = cols < num_logprob_cols
        ids = tl.load(src_ids_ptr + src * src_ids_stride + cols, mask=src_mask, other=0)
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
):
    req_idx = tl.program_id(0)
    pos = tl.program_id(1)
    start = tl.load(cu_num_logits_ptr + req_idx)
    if pos < tl.load(cu_num_logits_ptr + req_idx + 1) - start:
        src = tl.load(gathered_src_indices_ptr + req_idx) * max_num_logits_per_req + pos
        dst = start + pos
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < num_logprob_cols
        src_ids_row = gathered_ids_ptr + src * gathered_ids_stride
        ids = tl.load(src_ids_row + cols, mask=mask)
        tl.store(logprob_token_ids_ptr + dst * num_logprob_cols + cols, ids, mask=mask)
        rank = tl.load(src_ids_row + num_logprob_cols)
        tl.store(selected_token_ranks_ptr + dst, rank)
        logprobs = tl.load(
            gathered_logprobs_ptr + src * gathered_logprobs_stride + cols, mask=mask
        )
        tl.store(logprobs_ptr + dst * num_logprob_cols + cols, logprobs, mask=mask)


REFERENCE_KERNELS = {
    name: globals()[name]
    for name in [
        "_pack_logprobs_kernel",
        "_pack_sampler_output_kernel",
        "_unpack_gathered_output_kernel",
        "_unpack_logprobs_kernel",
    ]
}


SAMPLER_SHAPES = [
    (n, k, 1, "dense") for n in (1, 8, 32, 128, 512, 2048) for k in (1, 4, 8, 16)
]
LOGPROB_SHAPES = [
    (n, k, c, "dense")
    for n in (1, 32, 128, 512)
    for k in (1, 8)
    for c in (1, 6, 21, 65)
]
EXTRA_SHAPES = [
    (31, 3, 7, "ragged"),
    (129, 7, 21, "pad"),
    (513, 15, 65, "truncate"),
    (128, 8, 21, "strided"),
    (32, 4, 257, "dense"),
    (512, 4, 1025, "dense"),
]
KERNEL_NAMES = (
    "_pack_sampler_output_kernel",
    "_unpack_gathered_output_kernel",
    "_pack_logprobs_kernel",
    "_unpack_logprobs_kernel",
)
SAMPLER_EXTRA_SHAPES = EXTRA_SHAPES[:4] + [
    (32, 33, 1, "dense"),
    (512, 65, 1, "dense"),
]


def make_case(
    name,
    n,
    k,
    c=6,
    scenario="dense",
    *,
    device=None,
    dtype=torch.float32,
    id_dtype=torch.int64,
    index_dtype=None,
    with_nans=True,
    source_cols=None,
    seed=19,
):
    device = DEVICE if device is None else device
    metadata_dtype = torch.int32 if index_dtype is None else index_dtype
    mapping_dtype = torch.int64 if index_dtype is None else index_dtype
    g = torch.Generator().manual_seed(seed)
    outputs, expected, views = [], [], []

    def random(shape, dt):
        if dt.is_floating_point:
            return torch.randn(shape, generator=g).to(dt).to(device)
        return torch.randint(0, 32000, shape, generator=g, dtype=dt).to(device)

    def buffer(shape, dt, padded=False, output=False):
        stride = (
            shape[1] + 3
            if len(shape) == 2 and padded
            else (shape[-1] if len(shape) == 2 else 1)
        )
        size = (shape[0] * stride if len(shape) == 2 else shape[0]) + 17
        storage = torch.full((size,), -123, dtype=dt, device=device)
        strides = (stride, 1) if len(shape) == 2 else (1,)
        offset = 3 if scenario == "strided" else 0
        value = storage.as_strided(shape, strides, offset)
        if output:
            ref = storage.cpu().clone()
            outputs.append(storage)
            expected.append(ref)
            views.append(ref.as_strided(shape, strides, offset))
        else:
            value.copy_(random(shape, dt))
        return value

    strided = scenario == "strided"
    counts = torch.full((n,), k, dtype=metadata_dtype)
    if scenario == "ragged":
        counts = torch.arange(n, dtype=metadata_dtype) % (k + 1)
    cu_cpu = torch.cat(
        [torch.zeros(1, dtype=metadata_dtype), counts.cumsum(0).to(metadata_dtype)]
    )
    cu = cu_cpu.to(device)
    total = int(cu_cpu[-1])
    src_width = (
        source_cols
        if source_cols is not None
        else (
            max(0, c - 3)
            if scenario == "pad"
            else c + 5 if scenario == "truncate" else c
        )
    )
    if name == "_pack_sampler_output_kernel":
        src_width = min(
            k,
            (
                source_cols
                if source_cols is not None
                else max(0, k - 2) if scenario == "pad" else k
            ),
        )
        packed = buffer((n, k + 2 + int(with_nans)), torch.int64, strided, True)
        tokens = buffer((n, src_width), id_dtype, strided)
        sampled = buffer((n,), metadata_dtype)
        rejected = buffer((n,), metadata_dtype)
        nans = buffer((total,), metadata_dtype) if with_nans else None
        dst = views[0]
        dst[:, :k] = 0
        dst[:, :src_width] = tokens.cpu()
        dst[:, k] = sampled.cpu().long()
        dst[:, k + 1] = rejected.cpu().long()
        if with_nans:
            nans_cpu = nans.cpu()
            for i in range(n):
                dst[i, k + 2] = nans_cpu[cu_cpu[i] : cu_cpu[i + 1]].long().sum()
        args = (
            packed,
            packed.stride(0),
            tokens,
            tokens.stride(0),
            sampled,
            rejected,
            nans,
            cu if with_nans else None,
            k,
            src_width,
        )
        block = triton.next_power_of_2(k)
    elif name == "_unpack_gathered_output_kernel":
        gathered = buffer((max(1, 2 * n), k + 2 + int(with_nans)), torch.int64, strided)
        idx = (
            torch.randperm(max(1, 2 * n), generator=g)[:n].to(mapping_dtype).to(device)
        )
        tokens = buffer((n, k), torch.int64, output=True)
        sampled = buffer((n,), torch.int32, output=True)
        rejected = buffer((n,), torch.int32, output=True)
        nans = buffer((n,), torch.int32, output=True) if with_nans else None
        selected = gathered.cpu()[idx.cpu().long()]
        views[0].copy_(selected[:, :k])
        views[1].copy_(selected[:, k].int())
        views[2].copy_(selected[:, k + 1].int())
        if with_nans:
            views[3].copy_(selected[:, k + 2].int())
        args = (gathered, gathered.stride(0), idx, tokens, sampled, rejected, nans, k)
        block = triton.next_power_of_2(k)
    elif name == "_pack_logprobs_kernel":
        ids = buffer((n * k, c + 1), torch.int64, strided, True)
        probs = buffer((n * k, c), torch.float32, strided, True)
        src_ids = buffer((total, src_width), id_dtype, strided)
        src_probs = buffer((total, src_width), dtype, strided)
        rank = buffer((total,), id_dtype)
        src_ids_cpu, src_probs_cpu, rank_cpu = (
            src_ids.cpu(),
            src_probs.cpu(),
            rank.cpu(),
        )
        for i in range(n):
            begin, end = int(cu_cpu[i]), int(cu_cpu[i + 1])
            count = end - begin
            dst = slice(i * k, i * k + count)
            views[0][dst, :c] = 0
            views[1][dst] = -float("inf")
            width = min(c, src_width)
            views[0][dst, :width] = src_ids_cpu[begin:end, :width].long()
            views[0][dst, c] = rank_cpu[begin:end].long()
            views[1][dst, :width] = src_probs_cpu[begin:end, :width].float()
        args = (
            ids,
            ids.stride(0),
            probs,
            probs.stride(0),
            src_ids,
            src_ids.stride(0),
            src_probs,
            src_probs.stride(0),
            rank,
            cu,
            k,
            c,
            src_width,
        )
        block = triton.next_power_of_2(c)
    else:
        gathered_ids = buffer((max(1, 2 * n) * k, c + 1), torch.int64, strided)
        gathered_probs = buffer((max(1, 2 * n) * k, c), torch.float32, strided)
        idx = (
            torch.randperm(max(1, 2 * n), generator=g)[:n].to(mapping_dtype).to(device)
        )
        ids = buffer((total, c), torch.int64, output=True)
        probs = buffer((total, c), torch.float32, output=True)
        ranks = buffer((total,), torch.int64, output=True)
        ids_cpu, probs_cpu, idx_cpu = (
            gathered_ids.cpu(),
            gathered_probs.cpu(),
            idx.cpu(),
        )
        for i in range(n):
            begin, end = int(cu_cpu[i]), int(cu_cpu[i + 1])
            start = int(idx_cpu[i]) * k
            count = end - begin
            views[0][begin:end] = ids_cpu[start : start + count, :c]
            views[1][begin:end] = probs_cpu[start : start + count]
            views[2][begin:end] = ids_cpu[start : start + count, c]
        args = (
            gathered_ids,
            gathered_ids.stride(0),
            gathered_probs,
            gathered_probs.stride(0),
            idx,
            cu,
            ids,
            probs,
            ranks,
            k,
            c,
        )
        block = triton.next_power_of_2(c)
    return SimpleNamespace(
        name=name,
        n=n,
        k=k,
        c=c,
        scenario=scenario,
        args=args,
        block=block,
        outputs=outputs,
        expected=expected,
    )


def run_reference(case):
    grid = (case.n, case.k) if "logprobs" in case.name else (case.n,)
    REFERENCE_KERNELS[case.name][grid](*case.args, BLOCK_SIZE=case.block)


def run_candidate(case, config=None, use_layout=None):
    kernel = getattr(impl, case.name)
    block = (
        case.block
        if "logprobs" in case.name
        else triton.next_power_of_2(case.k + 2 + int(case.args[6] is not None))
    )
    layout = impl._HAS_LAYOUT if use_layout is None else use_layout

    def grid(meta):
        dims = (triton.cdiv(case.n, meta["BLOCK_ROWS"]),)
        return dims + ((case.k,) if "logprobs" in case.name else ())

    extra = (
        {"PACK_RANK": case.c < case.block <= 32}
        if case.name == "_pack_logprobs_kernel"
        else {}
    )
    if config is None:
        kernel[grid](
            *case.args, BLOCK_SIZE=block, NUM_REQS=case.n, USE_LAYOUT=layout, **extra
        )
    else:
        kernel.jit_function[grid](
            *case.args,
            BLOCK_SIZE=block,
            NUM_REQS=case.n,
            USE_LAYOUT=layout,
            **extra,
            **config.kwargs,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )


def check_case(case):
    for output, ref in zip(case.outputs, case.expected):
        torch.testing.assert_close(output.cpu(), ref, rtol=0, atol=0, equal_nan=True)


class BatchShardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = (
            LOGPROB_SHAPES + EXTRA_SHAPES
            if "logprobs" in self.kernel_name
            else SAMPLER_SHAPES + SAMPLER_EXTRA_SHAPES
        )

    def record_shapes(self, *args, **kwargs):
        return {
            "kernel": self.kernel_name,
            "scenario": self._case.scenario,
            "requests": self._case.n,
            "max_logits_per_request": self._case.k,
            "logprob_cols": self._case.c,
            "with_nans": self._with_nans,
            "input_dtypes": [
                str(arg.dtype)
                for arg in self._case.args
                if isinstance(arg, torch.Tensor)
            ],
            "inputs": super().record_shapes(*self._case.args),
        }

    def get_input_iter(self, dtype):
        flags = (False, True) if "logprobs" not in self.kernel_name else (False,)
        for with_nans in flags:
            yield from self._iter_cases(dtype, with_nans)

    def _iter_cases(self, dtype, with_nans):
        for n, k, c, scenario in self.shapes:
            self._with_nans = with_nans
            self._case = make_case(
                self.kernel_name,
                n,
                k,
                c,
                scenario,
                device=self.device,
                dtype=dtype if dtype.is_floating_point else torch.float32,
                id_dtype=dtype if not dtype.is_floating_point else torch.int64,
                with_nans=with_nans,
            )
            run_candidate(self._case)
            check_case(self._case)
            run_reference(self._case)
            check_case(self._case)
            yield ({"case": self._case},)


@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
def test_gather_sampler_output_perf(kernel_name):
    dtypes = {
        "_pack_sampler_output_kernel": [torch.int32, torch.int64],
        "_unpack_gathered_output_kernel": [torch.int64],
        "_pack_logprobs_kernel": [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ],
        "_unpack_logprobs_kernel": [torch.float32],
    }
    bench = BatchShardBenchmark(
        op_name=kernel_name,
        torch_op=run_reference,
        dtypes=dtypes[kernel_name],
    )
    bench.kernel_name = kernel_name
    bench.set_gems(run_candidate)
    bench.run()
