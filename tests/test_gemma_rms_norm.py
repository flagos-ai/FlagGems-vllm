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

os.environ["FLAGTREE_AABS"] = "0"

from itertools import product  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

import flaggems_vllm  # noqa: E402

from . import accuracy_utils as utils  # noqa: E402

_gemma_rms_norm_ns = [1152, 5376, 16384]
_gemma_rms_norm_ms = [1, 32, 128, 256]
_gemma_rms_norm_shapes = list(product(_gemma_rms_norm_ms, _gemma_rms_norm_ns))


@pytest.mark.gemma_rms_norm
@pytest.mark.parametrize("shape", _gemma_rms_norm_shapes)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_gemma_rms_norm(shape, dtype):
    N = shape[-1]
    x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    w = torch.randn((N), dtype=dtype, device=flaggems_vllm.device)
    eps = 1e-5

    def _torch_gemma_rms_norm(x, w, eps):
        x = x.to(dtype=torch.float32)
        w = w.to(dtype=torch.float32)
        rrms = 1 / ((x**2).mean(dim=-1, keepdim=True) + eps).sqrt()
        return (1 + w) * x * rrms

    ref_out = _torch_gemma_rms_norm(x, w, eps)

    with flaggems_vllm.use_gems():
        res_out = flaggems_vllm.gemma_rms_norm(x, w, eps)

    utils.gems_assert_close(res_out, ref_out, dtype)
