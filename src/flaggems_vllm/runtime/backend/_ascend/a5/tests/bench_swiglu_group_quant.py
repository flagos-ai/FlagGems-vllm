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

"""Benchmark A5 swiglu_group_quant against aclnnSwigluGroupQuant.

    ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=src python \
        src/flaggems_vllm/runtime/backend/_ascend/a5/tests/bench_swiglu_group_quant.py

The stdout latency is the median NPU Event time after warmup. The
operator is compiled and its raw launcher bound before measurement, so neither
Triton's dispatch nor its compilation is included. The Python vendor wrapper's
host costs are included, as this measures end-to-end dispatch, not kernel-only
profiler time; see swiglu_group_quant.py's performance notes for that
kernel-only distinction.
"""

import argparse
import statistics

import cann_ops_nn  # noqa: F401
import torch
import torch_npu  # noqa: F401

from flaggems_vllm.runtime.backend._ascend.a5.ops.swiglu_group_quant import (
    SwigluGroupQuantPlan,
)

_SHAPES = [
    (1, 4096),
    (16, 4096),
    (64, 4096),
    (256, 4096),
    (1024, 4096),
    (4096, 4096),
    (64, 512),
    (256, 1024),
    (4096, 1024),
    (256, 2560),
    (4096, 2560),
    (256, 3584),
    (4096, 8192),
    (4096, 14336),
]


def _time_us(fn, warmup, iterations):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--mode", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    for T, D in _SHAPES:
        torch.manual_seed(0)
        x = torch.randn(T, D, device="npu", dtype=torch.bfloat16)
        plan = SwigluGroupQuantPlan(x, quant_mode=args.mode)
        y = torch.empty((T, D // 2), dtype=torch.float8_e4m3fn, device="npu")
        scale = plan._new_scale(x.device)

        def ours():
            plan(x, out=y, scale=scale)

        def vendor():
            torch.ops.cann_ops_nn.swiglu_group_quant(
                x,
                None,
                None,
                None,
                dst_type=292,
                quant_mode=args.mode,
                block_size=0,
                round_scale=args.mode == 1,
                clamp_limit=-1.0,
                dst_type_max=15.0,
                output_origin=False,
            )

        ours_us = _time_us(ours, args.warmup, args.iterations)
        vendor_us = _time_us(vendor, args.warmup, args.iterations)
        print(
            f"T={T:5d} D={D:5d} mode={args.mode} tile={plan.rows}x{plan.cols} "
            f"ours={ours_us:8.2f}us vendor={vendor_us:8.2f}us "
            f"speedup={vendor_us / ours_us:.3f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
