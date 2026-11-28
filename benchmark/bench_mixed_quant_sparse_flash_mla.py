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

"""Benchmark A5 mixed_quant_sparse_flash_mla against aclnnMixedQuantSparseFlashMla.

    ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=src:$PYTHONPATH python \
        benchmark/bench_mixed_quant_sparse_flash_mla.py

Append to ``PYTHONPATH`` rather than replacing it: the vendor op lives in the
CANN tree that the environment script puts there, and overwriting the variable
drops it, which shows up here as ``vendor=unavailable``.

The plan is built and the output buffer allocated before measurement, so
neither Triton's compilation nor the staging-buffer allocation is counted.

``--timing kernel`` (the default) reports device kernel time through
``triton.backends.ascend.testing.do_bench_npu``, which is what the repository's
pytest benchmarks report in their default ``--mode kernel``: mspti collects
each kernel's execution interval, and a call that launches several kernels is
reported as their sum.  Host launch cost is excluded, and so are the gaps
between kernels.  This is the number to optimize against, because the per-launch
cost is fixed by the Triton runtime and disappears under graph capture.

``--timing wall`` reports end-to-end latency instead, in two modes, because
they disagree:

  * ``sync``       one call between two syncs, trimmed mean.  Latency with an
                   idle host: host dispatch plus device time, serialized.
  * ``pipelined``  ten calls between two NPU events, median.  Steady-state
                   throughput, which is ``max(host dispatch, device)``.

This op launches two or three Triton kernels per call through Python, the
vendor launches one aclnn kernel, so wall-clock charges us for launches that
kernel time and graph replay do not.

Both sides are given the same sparse workload: the same K random token indices
per row, over the same cache contents.  ``--vendor-form dense`` instead calls
the vendor with ``seqused_ori_kv = K`` and no indices, which makes it walk the
first K tokens in order; that is the easier access pattern, and on this
hardware it is worth only about 5 us, so the default sparse form is the one to
read.

What the two still do not share is the cache bytes, which cannot be matched:
ours is 544B/token (fp8 data then 16 bf16 group scales of 32 channels), the
vendor's ``quant_mode=1`` is 608B/token (bf16 rope, fp8 nope, 7 bf16 group
scales of 64 channels).  The vendor therefore reads 12% more per token, and
the two cannot be compared by value, only by time.
"""

import argparse
import importlib
import math
import statistics
import time
from collections import defaultdict

import torch
import torch_npu  # noqa: F401
import triton

from flaggems_vllm.runtime.backend._ascend.a5.ops.mixed_quant_sparse_flash_mla import (
    MixedQuantSparseFlashMlaPlan,
)

D = 512
N1 = 64
D_ROPE = 64
D_NOPE = D - D_ROPE
FP8_MAX = 448.0

# ours: group size 32 over the whole row
N_GROUPS = 16
GROUP_SIZE = 32
KV_ROW_BYTES = 544

# vendor quant_mode=1: group size 64 over the nope segment only
VENDOR_N_GROUPS = 7
VENDOR_GROUP_SIZE = 64
VENDOR_KV_ROW_BYTES = 608

_T1_VALUES = [1, 2, 4, 8, 16, 32]


def _kernel_us(fn, warmup, iterations):
    """Device kernel time per call, summed over the kernels a call launches."""
    return (
        triton.backends.ascend.testing.do_bench_npu(
            fn, warmup=warmup, active=max(iterations, 10)
        )
        * 1e3
    )


def _kernel_breakdown(fn, iterations):
    """Mean us per kernel name, from mspti, for one call of ``fn``."""
    try:
        from mspti import KernelMonitor
    except ImportError:
        return {}

    records = defaultdict(list)
    fn()
    torch.npu.synchronize()
    monitor = KernelMonitor()
    monitor.start(lambda data: records[data.name].append((data.end - data.start) / 1e3))
    try:
        for _ in range(iterations):
            fn()
            torch.npu.synchronize()
    finally:
        torch.npu.synchronize()
        monitor.stop()
    return {name: sum(t) / len(t) for name, t in records.items()}


def _pipelined_us(fn, warmup, iterations):
    """Median per-call time with ten calls in flight between two events."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iterations):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(10):
            fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 100)
    return statistics.median(samples)


def _sync_us(fn, warmup, iterations):
    """Per-call latency, one call between two syncs, mean of the middle 60%."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(max(iterations, 5) * 2):
        torch.npu.synchronize()
        begin = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - begin) * 1e6)
    samples.sort()
    lo = len(samples) // 5
    trimmed = samples[lo : len(samples) - lo]
    return sum(trimmed) / len(trimmed)


def _phys_rows(perm, block_size, count):
    rows = (
        perm.long()[:, None] * block_size + torch.arange(block_size)[None, :]
    ).reshape(-1)
    return rows[:count]


def _pack_ours(kv_f32, block_num, block_size, perm):
    """544B/token rows: fp8_e4m3 data then 16 bf16 group scales."""
    s = kv_f32.shape[0]
    grp = kv_f32.reshape(s, N_GROUPS, GROUP_SIZE)
    amax = grp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (amax / FP8_MAX).to(torch.bfloat16)
    fp8 = (grp / scale.float()).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

    cache = torch.zeros(block_num, block_size, 1, KV_ROW_BYTES, dtype=torch.uint8)
    flat = cache.view(-1, KV_ROW_BYTES)
    rows = _phys_rows(perm, block_size, s)
    flat[rows, :D] = fp8.reshape(s, D).view(torch.uint8)
    flat[rows, D : D + N_GROUPS * 2] = (
        scale.reshape(s, N_GROUPS).contiguous().view(torch.uint8)
    )
    return cache


def _pack_vendor(kv_f32, block_num, block_size, perm):
    """608B/token rows: bf16 rope, fp8_e4m3 nope, 7 bf16 group scales, pad."""
    s = kv_f32.shape[0]
    rope = kv_f32[:, D_NOPE:].to(torch.bfloat16)
    grp = kv_f32[:, :D_NOPE].reshape(s, VENDOR_N_GROUPS, VENDOR_GROUP_SIZE)
    amax = grp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (amax / FP8_MAX).to(torch.bfloat16)
    fp8 = (grp / scale.float()).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

    cache = torch.zeros(
        block_num, block_size, 1, VENDOR_KV_ROW_BYTES, dtype=torch.uint8
    )
    flat = cache.view(-1, VENDOR_KV_ROW_BYTES)
    rows = _phys_rows(perm, block_size, s)
    flat[rows, 0:128] = rope.contiguous().view(torch.uint8)
    flat[rows, 128:576] = fp8.reshape(s, D_NOPE).view(torch.uint8)
    flat[rows, 576:590] = (
        scale.reshape(s, VENDOR_N_GROUPS).contiguous().view(torch.uint8)
    )
    return cache


def _vendor_callable(
    q, vendor_kv, block_table, sinks, softmax_scale, indices, seqlen_kv, form
):
    """Bind the vendor op, or return None when it is unavailable.

    ``form="sparse"`` passes the same indices we gather; the vendor wants them
    as ``[T1, num_heads_kv, K]`` with an ``[T1, num_heads_kv]`` length, not the
    2-D and 1-D shapes this op takes.  ``form="dense"`` drops the indices and
    shortens ``seqused_ori_kv`` to K, which walks the first K tokens instead.
    """
    try:
        importlib.import_module("cann_ops_transformer")
    except ImportError:
        return None

    op = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla
    metadata_op = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata

    t1, k = indices.shape
    sparse = form == "sparse"
    used = seqlen_kv if sparse else k
    seqused = torch.full((t1,), used, device=q.device, dtype=torch.int32)
    cu_seqlens_q = torch.arange(t1 + 1, device=q.device, dtype=torch.int32)
    topk = torch.full((t1, 1), k, device=q.device, dtype=torch.int32)

    metadata = metadata_op(
        N1,
        1,
        D,
        1,
        cu_seqlens_q=cu_seqlens_q,
        seqused_ori_kv=seqused,
        ori_topk_length=topk if sparse else None,
        max_seqlen_q=1,
        max_seqlen_ori_kv=used,
        max_seqlen_cmp_kv=0,
        ori_topk=k if sparse else 0,
        cmp_topk=0,
        rope_head_dim=D_ROPE,
        cmp_ratio=1,
        ori_mask_mode=0,
        cmp_mask_mode=0,
        ori_win_left=-1,
        ori_win_right=-1,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=False,
    )

    kwargs = dict(
        ori_kv=vendor_kv,
        ori_block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_ori_kv=seqused,
        sinks=sinks,
        metadata=metadata,
        rope_head_dim=D_ROPE,
        softmax_scale=softmax_scale,
        cmp_ratio=1,
        ori_mask_mode=0,
        cmp_mask_mode=0,
        ori_win_left=-1,
        ori_win_right=-1,
        layout_q="TND",
        layout_kv="PA_BBND",
        topk_value_mode=1,
        return_softmax_lse=False,
    )
    if sparse:
        kwargs["ori_sparse_indices"] = indices.reshape(t1, 1, k)
        kwargs["ori_topk_length"] = topk

    def run():
        op(q, 1, **kwargs)

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--t1", type=int, nargs="+", default=_T1_VALUES)
    parser.add_argument("--k", type=int, default=2048, help="sparse tokens per query")
    parser.add_argument("--seqlen-kv", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--path", default="auto", help="force a dispatch path instead of the default"
    )
    parser.add_argument(
        "--vendor-form",
        default="sparse",
        choices=("sparse", "dense"),
        help="give the vendor the same indices (default) or a dense prefix",
    )
    parser.add_argument(
        "--timing",
        default="kernel",
        choices=("kernel", "wall"),
        help="device kernel time (default) or end-to-end wall clock",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="also print per-kernel mean time for our side",
    )
    args = parser.parse_args()

    if args.timing == "kernel":
        modes = (("kernel", _kernel_us),)
    else:
        modes = (("sync", _sync_us), ("pipelined", _pipelined_us))

    k, block_size = args.k, args.block_size
    softmax_scale = 1.0 / math.sqrt(D)

    torch.manual_seed(0)
    n_blk = args.seqlen_kv // block_size
    block_num = n_blk + 16
    perm = torch.randperm(block_num)[:n_blk].to(torch.int32)
    kv_f32 = torch.randn(args.seqlen_kv, D) * 0.5
    ours_kv = _pack_ours(kv_f32, block_num, block_size, perm).npu()
    vendor_kv = _pack_vendor(kv_f32, block_num, block_size, perm).npu()

    speedups = {name: [] for name, _ in modes}
    print(
        f"{'mode':<10}{'T1':>4}{'K':>7}  {'path':<22}"
        f"{'ours(us)':>10}{'vendor(us)':>12}{'speedup':>10}",
        flush=True,
    )
    print(
        "speedup = vendor / ours: >1x means this op is faster, <1x slower.",
        flush=True,
    )
    for t1 in args.t1:
        q = torch.randn(t1, N1, D, device="npu", dtype=torch.bfloat16)
        sinks = torch.zeros(N1, device="npu", dtype=torch.float32)
        block_table = perm.unsqueeze(0).expand(t1, -1).contiguous().npu()
        indices = torch.randint(
            0, args.seqlen_kv, (t1, k), device="npu", dtype=torch.int32
        )
        topk = torch.full((t1,), k, device="npu", dtype=torch.int32)

        plan = MixedQuantSparseFlashMlaPlan(
            q,
            ori_kv=ours_kv,
            ori_sparse_indices=indices,
            ori_block_table=block_table,
            ori_topk_length=topk,
            sinks=sinks,
            softmax_scale=softmax_scale,
            path=args.path,
        )
        out = torch.empty(t1, N1, D, device="npu", dtype=torch.bfloat16)

        def ours():
            plan(
                q,
                ori_kv=ours_kv,
                ori_sparse_indices=indices,
                ori_block_table=block_table,
                ori_topk_length=topk,
                sinks=sinks,
                out=out,
            )

        vendor = _vendor_callable(
            q,
            vendor_kv,
            block_table,
            sinks,
            softmax_scale,
            indices,
            args.seqlen_kv,
            args.vendor_form,
        )

        for name, timer in modes:
            ours_us = timer(ours, args.warmup, args.iterations)
            head = f"{name:<10}{t1:>4}{k:>7}  {plan.path:<22}{ours_us:>10.2f}"
            if vendor is None:
                print(f"{head}{'unavailable':>12}{'':>10}", flush=True)
                continue
            vendor_us = timer(vendor, args.warmup, args.iterations)
            speedups[name].append(vendor_us / ours_us)
            print(
                f"{head}{vendor_us:>12.2f}{vendor_us / ours_us:>9.3f}x",
                flush=True,
            )

        if args.breakdown:
            per_kernel = _kernel_breakdown(ours, max(args.iterations, 10))
            for name, mean_us in sorted(per_kernel.items(), key=lambda kv: -kv[1]):
                print(f"{'':10}{'':11}  {name[:22]:<22}{mean_us:>10.2f}", flush=True)

    for name, _ in modes:
        values = speedups[name]
        if len(values) > 1:
            geomean = math.exp(sum(math.log(s) for s in values) / len(values))
            print(f"{name} geomean speedup={geomean:.3f}x", flush=True)


if __name__ == "__main__":
    main()
