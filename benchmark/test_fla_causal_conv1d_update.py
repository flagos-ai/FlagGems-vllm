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
    pytest.mark.fla_causal_conv1d_update,
    pytest.mark.skipif(
        gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _inputs(shape, dtype, device):
    t, d = shape
    x = torch.randn(t, d + 16, device=device, dtype=dtype)[:, :d]
    weight = torch.randn(4, d, device=device, dtype=dtype) * 0.1
    state = torch.randn(t + 2, 4, d, device=device, dtype=dtype)[:, :3, :]
    indices = torch.arange(1, t + 1, dtype=torch.int32, device=device)
    cu = torch.arange(t + 1, dtype=torch.int32, device=device)

    out = torch.empty_like(x)
    return (x, state, weight), dict(
        activation="silu",
        conv_state_indices=indices,
        query_start_loc=cu,
        null_block_id=0,
        out=out,
    )


def _reference():
    from fla_npu.ops.ascendc import causal_conv1d_update

    return causal_conv1d_update


def _input_fn(shape, dtype, device):
    args, kwargs = _inputs(shape, dtype, device)
    yield (*args, kwargs)


class A5Benchmark(base.GenericBenchmark):
    def get_latency(self, op, *args, **kwargs):
        # Each invocation receives an independent copy of the source state.
        # Its layout is preserved. Setup is complete before device profiling.
        if Config.mode != consts.BenchMode.KERNEL:
            raise NotImplementedError("state-isolated measurement requires kernel mode")
        initial = args[1]
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
            values[1] = state
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


def test_fla_causal_conv1d_update():
    b = A5Benchmark(
        op_name="fla_causal_conv1d_update",
        torch_op=_reference(),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
    )
    b.set_gems(gems.fla_causal_conv1d_update)
    b.run()
