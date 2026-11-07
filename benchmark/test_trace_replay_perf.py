# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Benchmark trace replay against the upstream vLLM Triton implementation."""

import contextlib
import io

import pytest
import torch
import triton
import triton.language as tl

from flaggems_vllm import apply_trace_tokens as flaggems_apply_trace_tokens

try:
    from vllm.v1.worker.gpu.sample.trace_replay import (
        apply_trace_tokens as vllm_reference,
    )
except (ImportError, OSError):
    vllm_reference = None

from benchmark import base
from benchmark.conftest import Config

# vllm implementation.
@triton.jit
def _upstream_trace_replay_kernel(
    sampled_ptr,
    idx_mapping_ptr,
    trace_token_ids_ptr,
    trace_token_ids_stride,
    trace_len_ptr,
    total_len_ptr,
    prompt_len_ptr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_state_idx < 0:
        return
    trace_len = tl.load(trace_len_ptr + req_state_idx)
    if trace_len <= 0:
        return
    step = tl.load(total_len_ptr + req_state_idx) - tl.load(
        prompt_len_ptr + req_state_idx
    )
    if step < 0 or step >= trace_len:
        return
    token_id = tl.load(
        trace_token_ids_ptr + req_state_idx * trace_token_ids_stride + step
    )
    tl.store(sampled_ptr + batch_idx, token_id.to(tl.int64))


def _upstream_triton_reference(
    sampled, idx_mapping, trace_token_ids, trace_len, total_len, prompt_len
):
    """Run the upstream vLLM one-program-per-request Triton algorithm."""
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return
    _upstream_trace_replay_kernel[(num_reqs,)](
        sampled,
        idx_mapping,
        trace_token_ids,
        trace_token_ids.stride(0),
        trace_len,
        total_len,
        prompt_len,
    )


REFERENCE_FN = vllm_reference or _upstream_triton_reference
REFERENCE_NAME = "vLLM" if vllm_reference is not None else "vLLM Triton"


_kernel_mark = pytest.MarkDecorator(
    pytest.Mark("_trace_replay_kernel", (), {}, _ispytest=True),
    _ispytest=True,
)
pytestmark = [
    pytest.mark.apply_trace_tokens,
    _kernel_mark,
]


class TraceReplayBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [
        (n, 2048) for n in (1, 2, 3, 5, 32, 33, 127, 128, 129, 256, 257, 4096)
    ]
    DEFAULT_SHAPE_DESC = "num_reqs, max_model_len"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._latency_ms = {}

    def get_latency(self, op, *args, **kwargs):
        latency_ms = super().get_latency(op, *args, **kwargs)
        shape = (args[0].shape[0], args[2].shape[1])
        measured = self._latency_ms.setdefault(shape, {})
        if op is self.torch_op:
            measured["reference"] = latency_ms
        elif op is self.gems_op:
            measured["flaggems"] = latency_ms
        return latency_ms

    def run(self):
        if Config.query:
            return super().run()
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                super().run()
        except BaseException:
            print(output.getvalue(), end="")
            raise
        self.print_summary()

    def print_summary(self):
        if not self._latency_ms:
            return
        mode_label = {
            "kernel": "GPU event",
            "cudagraph": "GPU Graph",
        }.get(Config.mode.value, Config.mode.value)
        print(f"\nTrace replay latency ({mode_label}, microseconds)")
        print(
            f"Requests  Trace slots  {REFERENCE_NAME + ' ref':>16}  "
            f"{'FlagGems-vllm':>14}  Speedup"
        )
        print("-" * 69)
        for num_reqs, max_model_len in self.shapes:
            measured = self._latency_ms.get((num_reqs, max_model_len), {})
            if "flaggems" not in measured or "reference" not in measured:
                continue
            baseline_us = measured["reference"] * 1000
            candidate_us = measured["flaggems"] * 1000
            print(
                f"{num_reqs:>8}  {max_model_len:>11}  "
                f"{baseline_us:>16.3f}  {candidate_us:>14.3f}  "
                f"{baseline_us / candidate_us:>7.3f}x"
            )

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for num_reqs, max_model_len in self.shapes:
            sampled = torch.full((num_reqs,), -7, dtype=dtype, device=self.device)
            mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
            traces = torch.arange(
                num_reqs * max_model_len, dtype=torch.int32, device=self.device
            ).reshape(num_reqs, max_model_len)
            lengths = torch.full((num_reqs,), 3, dtype=torch.int32, device=self.device)
            prompt = torch.full((num_reqs,), 5, dtype=torch.int32, device=self.device)
            total = prompt + 1
            if num_reqs >= 3:
                # Exercise replayed, skipped and untraced requests.
                mapping[0] = 1
                mapping[1] = -1
                mapping[2] = 0
                lengths[0] = 0

            arguments = (mapping, traces, lengths, total, prompt)
            candidate_sampled = sampled.clone()
            assert flaggems_apply_trace_tokens(candidate_sampled, *arguments) is None
            reference_sampled = sampled.clone()
            assert REFERENCE_FN(reference_sampled, *arguments) is None
            torch.testing.assert_close(
                candidate_sampled, reference_sampled, rtol=0, atol=0
            )
            yield (sampled, *arguments)


def test_trace_replay_perf():
    bench = TraceReplayBenchmark(
        op_name="apply_trace_tokens",
        torch_op=REFERENCE_FN,
        dtypes=[torch.int64],
        is_inplace=True,
    )
    bench.set_gems(flaggems_apply_trace_tokens)
    bench.run()
