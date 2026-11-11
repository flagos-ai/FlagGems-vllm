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

"""Eager benchmark for the public single-kernel Hopper MegaMHC operator."""

from __future__ import annotations

import importlib

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

from . import base
from .conftest import Config
from .consts import BenchLevel, BenchMode

CORE_TOKENS = (1, 24, 64, 128, 512, 1028, 2048, 3076, 16384)
COMPREHENSIVE_TOKENS = (
    1,
    2,
    4,
    8,
    16,
    *range(24, 257, 8),
    *range(272, 513, 16),
    1028,
    2048,
    3076,
    3805,
    4100,
    12991,
    13005,
    13021,
    16384,
)
_NATIVE = None


def _native_operator():
    if flaggems_vllm.vendor_name != "nvidia":
        return None
    try:
        module = importlib.import_module("vllm.model_executor.kernels.mhc")
        return module.mhc_fused_post_pre_tilelang
    except (AttributeError, ImportError):
        return None


def _candidate(x, residual, post, comb, fn, scale, base_value, norm, context):
    return flaggems_vllm.mega_mhc(
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        base_value,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        1,
        1,
        norm,
        1e-6,
        context=context,
    )


def _baseline(x, residual, post, comb, fn, scale, base_value, norm, context):
    del context
    if _NATIVE is None:
        raise RuntimeError("vLLM MegaMHC baseline is unavailable")
    return _NATIVE(
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        base_value,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        1,
        1,
        norm,
        1e-6,
    )


class MegaMHCBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_DESC = "T"

    def set_shapes(self, shape_file_path=None):
        del shape_file_path
        tokens = (
            COMPREHENSIVE_TOKENS
            if Config.bench_level is BenchLevel.COMPREHENSIVE
            else CORE_TOKENS
        )
        self.shapes = [(tokens_value,) for tokens_value in tokens]
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, dtype):
        assert dtype is torch.bfloat16
        for (tokens,) in self.shapes:
            generator = torch.Generator(device=self.device).manual_seed(
                20261111 + tokens
            )

            def randn(shape, tensor_dtype):
                return torch.randn(
                    shape,
                    dtype=tensor_dtype,
                    device=self.device,
                    generator=generator,
                )

            comb = torch.rand((tokens, 4, 4), device=self.device, generator=generator)
            for _ in range(3):
                comb /= comb.sum(-1, keepdim=True)
                comb /= comb.sum(-2, keepdim=True)
            values = (
                randn((tokens, 4096), torch.bfloat16),
                randn((tokens, 4, 4096), torch.bfloat16),
                randn((tokens, 4, 1), torch.float32).sigmoid(),
                comb,
                randn((24, 16384), torch.float32).mul_(1e-2),
                randn((3,), torch.float32).mul_(0.1),
                randn((24,), torch.float32).mul_(0.1),
                randn((4096,), torch.float32).mul_(0.1).add_(1).bfloat16(),
            )
            context = flaggems_vllm.create_mega_mhc_context(*values)
            yield (*values, {"context": context})


@pytest.mark.megamhc
def test_megamhc():
    global _NATIVE

    if Config.mode is not BenchMode.KERNEL:
        raise ValueError("MegaMHC benchmark requires --mode kernel")
    if flaggems_vllm.vendor_name != "nvidia":
        pytest.skip("MegaMHC is currently implemented for NVIDIA Hopper")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires SM90")
    if not has_triton_tle(3, 6, 0):
        pytest.skip("requires FlagTree TLE >= 3.6")
    _NATIVE = _native_operator()
    if _NATIVE is None:
        pytest.skip("MegaMHC benchmark requires the vLLM production baseline")
    benchmark = MegaMHCBenchmark(
        op_name="megamhc",
        torch_op=_baseline,
        gems_op=_candidate,
        dtypes=[torch.bfloat16],
    )
    benchmark.run()
