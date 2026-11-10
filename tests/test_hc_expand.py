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

import flaggems_vllm
from baseline.hc_expand import (
    DTYPES,
    FUNCTIONAL_SHAPES,
    INPUT_LAYOUTS,
    build_inputs,
    torch_reference,
)
from vllm.utils.torch_utils import set_random_seed

TORCH_REFERENCE = torch_reference()

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.hc_expand,
    pytest.mark.skipif(
        TORCH_REFERENCE is None,
        reason="vLLM PyTorch hc_expand unavailable",
    ),
]


def _assert_bits_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if actual.numel():
        torch.testing.assert_close(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )


@pytest.mark.hc_expand
@pytest.mark.parametrize(
    "num_tokens,hidden_size,hc_mult",
    FUNCTIONAL_SHAPES,
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", INPUT_LAYOUTS)
def test_hc_expand_preserves_bits_in_eager(
    num_tokens, hidden_size, hc_mult, dtype, layout
):
    """Expansion preserves every input bit, including NaNs and signed zero."""
    set_random_seed(42)
    binding = {
        "num_tokens": num_tokens,
        "hidden_size": hidden_size,
        "hc_mult": hc_mult,
    }
    args, _ = build_inputs(
        binding,
        dtype,
        flaggems_vllm.device,
        layout=layout,
        special_values=False,
    )
    x, hc_mult = args
    if num_tokens:
        x[0, 0] = float("nan")
    if num_tokens > 1:
        x[1, -1] = -0.0
    x_before = x.clone()

    def check(actual):
        expected = TORCH_REFERENCE(x, hc_mult)
        assert actual.shape == expected.shape and actual.dtype == x.dtype
        assert actual.is_contiguous()
        _assert_bits_equal(actual, expected)

    result = flaggems_vllm.hc_expand(x, hc_mult)
    check(result)
    assert result.device == x.device
    if result.numel():
        assert result.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        assert result.stride() == (hc_mult * hidden_size, hidden_size, 1)
    _assert_bits_equal(x, x_before)
