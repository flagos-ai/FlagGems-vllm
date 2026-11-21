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
from triton import language as tl

import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

pytestmark = [
    pytest.mark.libentry,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@triton.jit
def _scale_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, x * 2.0, mask=mask)


def _launch(entry, **options):
    x = torch.randn(256, device=flaggems_vllm.device)
    y = torch.empty_like(x)
    entry[(1,)](x, y, 256, BLOCK=256, **options)
    torch.testing.assert_close(y, x * 2.0)
    return x, y


def _entries(entry):
    return entry.kernel_cache[torch_device_fn.current_device()]


def test_compile_options_get_separate_entries():
    """A launch must run a kernel built with the options it asked for."""
    entry = libentry()(_scale_kernel)

    for num_stages in (2, 5):
        _launch(entry, num_stages=num_stages)

    built = sorted(
        kernel.metadata.num_stages for kernel, *_ in _entries(entry).values()
    )
    assert built == [2, 5]


def test_repeated_launch_reuses_its_entry():
    entry = libentry()(_scale_kernel)

    for _ in range(3):
        _launch(entry, num_warps=4, num_stages=2)

    assert len(_entries(entry)) == 1


def test_entry_is_independent_of_option_order():
    entry = libentry()(_scale_kernel)

    _launch(entry, num_warps=4, num_stages=2)
    _launch(entry, num_stages=2, num_warps=4)

    assert len(_entries(entry)) == 1


def test_grid_does_not_split_entries():
    """`grid` steers the launch, not the build."""
    entry = libentry()(_scale_kernel)
    x = torch.randn(512, device=flaggems_vllm.device)
    y = torch.empty_like(x)

    for grid in ((1,), (2,)):
        entry[grid](x, y, 256, BLOCK=256, num_warps=4)

    assert len(_entries(entry)) == 1


def test_key_rebuilds_an_option_free_entry():
    """Callers outside this module index `kernel_cache` with `key`."""
    entry = libentry()(_scale_kernel)

    x, y = _launch(entry)

    assert entry.key([x, y, 256], [], [256]) in _entries(entry)


def test_key_with_options_rebuilds_an_entry():
    entry = libentry()(_scale_kernel)
    options = {"num_warps": 4}

    x, y = _launch(entry, **options)

    entry_key = entry.key([x, y, 256], [], [256])
    assert entry_key not in _entries(entry)
    assert entry_key + entry._launch_options_key(options) in _entries(entry)
