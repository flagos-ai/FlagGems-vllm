# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Preallocated pack/unpack kernel benchmarks; collective time is excluded."""

from types import SimpleNamespace

import pytest
import torch
import triton

from benchmark import base, consts
from benchmark.conftest import TEST_RESULTS, update_result
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

# Reference kernels provided by vLLM 0.32.
try:
    from vllm.v1.worker.gpu.sample.batch_shard import (
        _pack_logprobs_kernel,
        _pack_sampler_output_kernel,
        _unpack_gathered_output_kernel,
        _unpack_logprobs_kernel,
    )
except Exception as exc:
    # Native extension initialization can fail even when vLLM is installed.
    REFERENCE_KERNELS = {}
    VLLM_REFERENCE_UNAVAILABLE_REASON = (
        "vLLM batch_shard reference unavailable: " f"{type(exc).__name__}: {exc}"
    )
else:
    REFERENCE_KERNELS = {
        "_pack_sampler_output_kernel": _pack_sampler_output_kernel,
        "_unpack_gathered_output_kernel": _unpack_gathered_output_kernel,
        "_pack_logprobs_kernel": _pack_logprobs_kernel,
        "_unpack_logprobs_kernel": _unpack_logprobs_kernel,
    }
    VLLM_REFERENCE_UNAVAILABLE_REASON = None


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


# Core covers small/medium/large batches, wide rows, and each edge scenario.
CORE_SAMPLER_SHAPES = [
    (1, 1, 1, "dense"),
    (32, 4, 1, "dense"),
    (128, 8, 1, "dense"),
    (2048, 16, 1, "dense"),
] + SAMPLER_EXTRA_SHAPES
CORE_LOGPROB_SHAPES = [
    (1, 1, 1, "dense"),
    (32, 8, 6, "dense"),
    (128, 8, 21, "dense"),
    (512, 8, 65, "dense"),
] + EXTRA_SHAPES


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
    validate=True,
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
        if output and validate:
            ref = storage.cpu().clone()
            outputs.append(storage)
            expected.append(ref)
            views.append(ref.as_strided(shape, strides, offset))
        elif not output:
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
        if validate:
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
        if validate:
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
        if validate:
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
        if validate:
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
    if not REFERENCE_KERNELS:
        pytest.skip(VLLM_REFERENCE_UNAVAILABLE_REASON)
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
    def run(self):
        previous = len(TEST_RESULTS.get(self.op_name, {}).get("details", []))
        try:
            super().run()
        finally:
            self._record_operator_results(previous)

    def _record_operator_results(self, previous):
        # Keep kernel names for baseline lookup; publish one record per dtype
        # under the operator mark so optest does not overwrite sibling kernels.
        for record in TEST_RESULTS.get(self.op_name, {}).get("details", [])[previous:]:
            if self.torch_op is None:
                record["native_baseline_skip_reason"] = (
                    VLLM_REFERENCE_UNAVAILABLE_REASON
                )
            details = TEST_RESULTS.get("gather_sampler_output", {}).get("details", [])
            merged = next(
                (
                    item
                    for item in details
                    if all(
                        item[key] == record[key] for key in ("dtype", "mode", "level")
                    )
                ),
                None,
            )
            if merged is None:
                update_result(
                    "gather_sampler_output",
                    {
                        **record,
                        "op_name": "gather_sampler_output",
                        "result": list(record["result"]),
                    },
                )
            else:
                merged["result"].extend(record["result"])

    def set_shapes(self, shape_file_path=None):
        if base.Config.bench_level == consts.BenchLevel.CORE:
            self.shapes = (
                CORE_LOGPROB_SHAPES
                if "logprobs" in self.kernel_name
                else CORE_SAMPLER_SHAPES
            )
            return
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
        # Check one dense configuration plus each edge scenario per dtype/NaN mode.
        # The remaining rows measure performance without building CPU oracles.
        checked_scenarios = set()
        for n, k, c, scenario in self.shapes:
            validate = scenario not in checked_scenarios
            checked_scenarios.add(scenario)
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
                validate=validate,
            )
            if validate:
                run_candidate(self._case)
                check_case(self._case)
                if REFERENCE_KERNELS:
                    run_reference(self._case)
                    check_case(self._case)
            yield ({"case": self._case},)


@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
def test_gather_sampler_output_perf(kernel_name):
    if not REFERENCE_KERNELS:
        print(VLLM_REFERENCE_UNAVAILABLE_REASON)
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
        torch_op=run_reference if REFERENCE_KERNELS else None,
        dtypes=dtypes[kernel_name],
    )
    bench.kernel_name = kernel_name
    bench.set_gems(run_candidate)
    bench.run()
