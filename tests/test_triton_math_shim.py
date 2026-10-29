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

import pytest
import torch
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.utils.triton_lang_helper import _fallback_pow, _fallback_tanh

from . import accuracy_utils as utils


@triton.jit
def _fallback_math_kernel(X, Power, Tanh, N: tl.constexpr, Exponent: tl.constexpr):
    offsets = tl.arange(0, triton.next_power_of_2(N))
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    tl.store(Power + offsets, _fallback_pow(x, Exponent), offsets < N)
    tl.store(Tanh + offsets, _fallback_tanh(x), offsets < N)


@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("exponent", [2, 3])
def test_core_math_fallbacks(dtype, exponent):
    values = torch.cat(
        [
            torch.linspace(-6, 6, 257),
            torch.tensor([-float("inf"), float("inf"), float("nan"), -0.0, 0.0]),
        ]
    ).to(dtype)
    x = values.to(flaggems_vllm.device)
    power = torch.empty_like(x)
    tanh = torch.empty_like(x)

    _fallback_math_kernel[(1,)](x, power, tanh, x.numel(), exponent)

    reference = values.double()
    utils.gems_assert_close(power.cpu(), reference.pow(exponent), dtype, equal_nan=True)
    utils.gems_assert_close(tanh.cpu(), reference.tanh(), dtype, equal_nan=True)
