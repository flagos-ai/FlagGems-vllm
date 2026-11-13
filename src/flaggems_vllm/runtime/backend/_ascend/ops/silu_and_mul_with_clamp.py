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

import os
from contextlib import contextmanager
from threading import RLock

import torch

from flaggems_vllm.ops.silu_and_mul_with_clamp import (
    SiluAndMulWithClamp as _GenericSiluAndMulWithClamp,
)
from flaggems_vllm.ops.silu_and_mul_with_clamp import (
    silu_and_mul_with_clamp as _generic_silu_and_mul_with_clamp,
)
from flaggems_vllm.ops.silu_and_mul_with_clamp import (
    silu_and_mul_with_clamp_out as _generic_silu_and_mul_with_clamp_out,
)

_LAUNCH_LOCK = RLock()


@contextmanager
def _blockified():
    # This compiler option is process-wide; only SiLU scopes share this lock.
    with _LAUNCH_LOCK:
        previous = os.environ.get("TRITON_ALL_BLOCKS_PARALLEL")
        os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("TRITON_ALL_BLOCKS_PARALLEL", None)
            else:
                os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = previous


class _AscendSiluAndMulWithClamp(_GenericSiluAndMulWithClamp):
    @staticmethod
    def forward(ctx, x, y, limit):
        with _blockified():
            return _GenericSiluAndMulWithClamp.forward(ctx, x, y, limit)

    @staticmethod
    def backward(ctx, dgrad):
        with _blockified():
            return _GenericSiluAndMulWithClamp.backward(ctx, dgrad)


def silu_and_mul_with_clamp(x, y, limit):
    if torch.is_grad_enabled() and (x.requires_grad or y.requires_grad):
        return _AscendSiluAndMulWithClamp.apply(x, y, limit)
    with _blockified():
        return _generic_silu_and_mul_with_clamp(x, y, limit)


def silu_and_mul_with_clamp_out(x, y, out, limit):
    with _blockified():
        return _generic_silu_and_mul_with_clamp_out(x, y, out, limit)
