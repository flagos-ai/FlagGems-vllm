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

import math

import pytest
import torch
import triton

import flaggems_vllm as gems
from flaggems_vllm.runtime import torch_device_fn

from . import base, consts
from .conftest import Config

pytestmark = [
    pytest.mark.recurrent_gated_delta_rule,
    pytest.mark.skipif(
        gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _inputs(shape, dtype, device):
    pool, hv = shape
    q = torch.randn((1, 8, 128), device=device, dtype=dtype)
    k = torch.nn.functional.normalize(
        torch.randn((1, 8, 128), device=device), dim=-1
    ).to(dtype)
    v = torch.randn((1, hv, 128), device=device, dtype=dtype)
    state = torch.randn((pool, hv + 2, 128, 128), device=device, dtype=torch.float32)[
        :, :hv
    ]
    beta = torch.rand((1, hv), device=device, dtype=dtype)
    g = -torch.rand((1, hv), device=device, dtype=torch.float32)
    cu = torch.tensor([0, 1], device=device, dtype=torch.int32)
    idx = torch.tensor([1], device=device, dtype=torch.int32)
    return (q, k, v, state), dict(
        beta=beta, g=g, actual_seq_lengths=cu, ssm_state_indices=idx, scale=128**-0.5
    )


def _reference():
    from fla_npu.ops.ascendc import recurrent_gated_delta_rule

    return recurrent_gated_delta_rule


def _input_fn(shape, dtype, device):
    args, kwargs = _inputs(shape, dtype, device)
    yield (*args, kwargs)


class A5Benchmark(base.GenericBenchmark):
    def get_latency(self, op, *args, **kwargs):
        # Each invocation receives an independent copy of the source state.
        # Its layout is preserved. Setup is complete before device profiling.
        if Config.mode != consts.BenchMode.KERNEL:
            raise NotImplementedError("state-isolated measurement requires kernel mode")
        initial = args[3]
        # The profiler performs compilation/initial warmup before its active steps.
        warmup, active = 5, 15
        prepared = []
        for _ in range(warmup + active + 12):
            values = list(args)
            state = torch.empty_strided(
                initial.shape,
                initial.stride(),
                dtype=initial.dtype,
                device=initial.device,
            )
            state.copy_(initial)
            values[3] = state
            call_kwargs = kwargs
            if "out" in kwargs:
                call_kwargs = dict(kwargs, out=torch.empty_like(kwargs["out"]))
            prepared.append((values, call_kwargs))
        torch_device_fn.synchronize()
        calls = iter(prepared)

        def invoke():
            values, call_kwargs = next(calls)
            return op(*values, **call_kwargs)

        latency = triton.backends.ascend.testing.do_bench_npu(
            invoke, warmup=warmup, active=active
        )
        if not math.isfinite(latency) or latency <= 0:
            raise RuntimeError("device profiler did not return a finite positive time")
        return latency

    def set_more_shapes(self):
        return []


def test_recurrent_gated_delta_rule():
    b = A5Benchmark(
        op_name="recurrent_gated_delta_rule",
        torch_op=_reference(),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
    )
    b.set_gems(gems.recurrent_gated_delta_rule)
    b.run()
