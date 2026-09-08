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
import random
from itertools import product

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils
from . import conftest as cfg

os.environ["FLAGTREE_AABS"] = "0"

# hidden_size of the gemma-3 series models
_gemma_rmsnorm_ns = [1152, 3840, 5376]
# Batch size
_gemma_rmsnorm_ms = [1, 32, 128]
_gemma_rmsnorm_shapes = list(product(_gemma_rmsnorm_ms, _gemma_rmsnorm_ns))
if cfg.QUICK_MODE:
    _gemma_rmsnorm_shapes = random.sample(_gemma_rmsnorm_shapes, 8)


@pytest.mark.gemma_rmsnorm
@pytest.mark.parametrize("shape", _gemma_rmsnorm_shapes)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gemma_rmsnorm(shape, dtype):
    N = shape[-1]
    x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.runtime.device.name)
    w = torch.randn((N), dtype=dtype, device=flaggems_vllm.runtime.device.name)
    eps = 1e-5

    def _torch_gemma_rmsnorm(x, w, eps):
        x = x.to(dtype=torch.float32)
        w = w.to(dtype=torch.float32)
        rrms = 1 / ((x**2).mean(dim=-1, keepdim=True) + eps).sqrt()
        return (1 + w) * x * rrms

    ref_out = _torch_gemma_rmsnorm(x, w, eps)

    with flaggems_vllm.use_gems():
        res_out = flaggems_vllm.gemma_rmsnorm(x, w, eps)

    utils.gems_assert_close(res_out, ref_out, dtype)
