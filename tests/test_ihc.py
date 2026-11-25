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

"""Accuracy tests for iHC (independent Hyper-Connections) operators.

Tests iHC pre against an independent CPU oracle and post against its reference.
"""

from itertools import product

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.ihc import ihc_post, ihc_post_ref, ihc_pre

from .test_triton_ihc_pre import _ihc_pre_cpu_threads, ihc_pre_reference  # noqa: F401

DEVICE = flaggems_vllm.device

# vLLM HY V4 shapes: hidden in {4096, 6144}, hc_mult == 4.
_HIDDEN = [4096, 6144]
_HC_MULT = [4]
_TOKENS = [0, 1, 7, 64, 256, 4096]
_DTYPES = [torch.bfloat16, torch.float16]


def _make_pre_inputs(n, hc, h, dtype, device=DEVICE):
    torch.manual_seed(42)
    x = torch.randn((n, hc, h), dtype=dtype, device=device)
    weight = torch.randn((2 * hc, hc * h), dtype=torch.float32, device=device)
    scale = torch.randn((2,), dtype=torch.float32, device=device) * 0.01
    base = torch.randn((2 * hc,), dtype=torch.float32, device=device)
    return x, weight, scale, base


@pytest.mark.ihc_pre
@pytest.mark.usefixtures("_ihc_pre_cpu_threads")
@pytest.mark.parametrize("n", _TOKENS)
@pytest.mark.parametrize("h", _HIDDEN)
@pytest.mark.parametrize("hc", _HC_MULT)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_ihc_pre_vs_ref(n, h, hc, dtype):
    x, weight, scale, base = _make_pre_inputs(n, hc, h, dtype)
    y, post = ihc_pre(x, weight, scale, base, 2.0, 1e-6, 1e-5)
    y_ref, post_ref = ihc_pre_reference(x, weight, scale, base, 2.0, 1e-6, 1e-5)

    assert y.shape == (n, h)
    assert post.shape == (n, hc)
    assert y.dtype == dtype and post.dtype == torch.float32
    torch.testing.assert_close(y.cpu(), y_ref, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(post.cpu(), post_ref, rtol=1e-5, atol=2e-5)


@pytest.mark.ihc_post
@pytest.mark.parametrize("n", _TOKENS)
@pytest.mark.parametrize("h", _HIDDEN)
@pytest.mark.parametrize("hc", _HC_MULT)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_ihc_post_vs_ref(n, h, hc, dtype):
    torch.manual_seed(42)
    x = torch.randn((n, h), dtype=dtype, device=DEVICE)
    residual = torch.randn((n, hc, h), dtype=dtype, device=DEVICE)
    post = torch.randn((n, hc), dtype=torch.float32, device=DEVICE)

    out = ihc_post(x, residual, post)
    out_ref = ihc_post_ref(x, residual, post)

    assert out.shape == (n, hc, h)
    assert out.dtype == dtype
    # post is a pure gated residual-add: should match the reference exactly.
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)


@pytest.mark.ihc
@pytest.mark.parametrize("n,h,hc", list(product([64, 4096], [4096, 6144], [4])))
def test_ihc_pre_post_chain(n, h, hc):
    """pre produces the post gates that post consumes -- test them together."""
    dtype = torch.bfloat16
    x, weight, scale, base = _make_pre_inputs(n, hc, h, dtype)
    _, post = ihc_pre(x, weight, scale, base, 2.0, 1e-6, 1e-5)

    block_out = torch.randn((n, h), dtype=dtype, device=DEVICE)
    residual = torch.randn((n, hc, h), dtype=dtype, device=DEVICE)
    out = ihc_post(block_out, residual, post)
    out_ref = ihc_post_ref(block_out, residual, post)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)


# Preserve every former native-comparison shape using the independent CPU oracle.
_VLLM_TOKENS = [1, 64, 1024, 8192]


@pytest.mark.ihc_pre
@pytest.mark.usefixtures("_ihc_pre_cpu_threads")
@pytest.mark.parametrize("n", _VLLM_TOKENS)
@pytest.mark.parametrize("h", _HIDDEN)
@pytest.mark.parametrize("hc", _HC_MULT)
def test_ihc_pre_model_shapes_vs_cpu_ref(n, h, hc):
    """Keep the legacy wrapper's model-size BF16 coverage independent of vLLM."""
    dtype = torch.bfloat16
    x, weight, scale, base = _make_pre_inputs(n, hc, h, dtype)
    y, post = ihc_pre(x, weight, scale, base, 2.0, 1e-6, 1e-5)
    y_ref, post_ref = ihc_pre_reference(x, weight, scale, base, 2.0, 1e-6, 1e-5)
    torch.testing.assert_close(y.cpu(), y_ref, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(post.cpu(), post_ref, rtol=1e-5, atol=2e-5)


@pytest.mark.ihc_post
@pytest.mark.parametrize("n", _VLLM_TOKENS)
@pytest.mark.parametrize("h", _HIDDEN)
@pytest.mark.parametrize("hc", _HC_MULT)
def test_ihc_post_model_shapes_vs_cpu_ref(n, h, hc):
    """Keep the legacy wrapper's model-size BF16 coverage independent of vLLM."""
    dtype = torch.bfloat16
    torch.manual_seed(42)
    x = torch.randn((n, h), dtype=dtype, device=DEVICE)
    residual = torch.randn((n, hc, h), dtype=dtype, device=DEVICE)
    post = torch.randn((n, hc), dtype=torch.float32, device=DEVICE)
    out = ihc_post(x, residual, post)

    # Limit CPU oracle intermediates at the largest model sizes.
    for start in range(0, n, 32):
        part = slice(start, start + 32)
        expected = ihc_post_ref(x[part].cpu(), residual[part].cpu(), post[part].cpu())
        torch.testing.assert_close(out[part].cpu(), expected, rtol=0, atol=0)
