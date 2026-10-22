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

"""Shared NVIDIA helpers for per-token group FP8 quantization schedules."""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=None)
def _is_h20(device_index):
    return torch.cuda.get_device_name(device_index) in ("NVIDIA H20", "NVIDIA H20-3e")


@triton.jit
def _clamp_like_upstream(value):
    # A symmetric constant clip can map NaNs to the upper bound. Upstream's
    # runtime-bound max/min instead maps them to the lower bound; retain that
    # sequence explicitly, including for Inf/Inf during normalization.
    return tl.inline_asm_elementwise(
        "max.f32 $0, $1, 0fC3E00000; min.f32 $0, $0, 0f43E00000;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
