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

"""MThreads backend options for the mHC pre kernels.

The kernels become proxies rather than ``JITFunction``/``Autotuner`` instances,
the same trade hygon's ``FastLaunchKernel`` makes.
"""

import os

# Kernels mhc_pre launches; device functions stay bare for the code generator.
KERNELS = (
    "mhc_pre_proj_gemm_kernel",
    "mhc_pre_packed_reader_kernel",
    "mhc_pre_sinkhorn_fuse_kernel",
    "mhc_pre_mm_fuse_kernel",
    "mhc_pre_mm_fuse_tiled_kernel",
    "mhc_pre_generic_kernel",
    "_mhc_cast_kernel",
    "_mhc_copy_nd_kernel",
    "_mhc_fill_kernel",
    "_mhc_pack_kernel",
)


class _LLCOptKernel:
    """``kernel[grid](...args)`` with the LLC option added to the launch."""

    __slots__ = ("_kernel", "_opts")

    def __init__(self, kernel, opts):
        self._kernel = kernel
        self._opts = opts

    def __getitem__(self, grid):
        run = self._kernel[grid]
        return lambda *args, **kwargs: run(*args, **{**self._opts, **kwargs})

    def __getattr__(self, name):
        # Via object.__getattribute__: a half-built copy has no slot to read yet.
        return getattr(object.__getattribute__(self, "_kernel"), name)


def install_llc_opt(namespace) -> int:
    """Compile this backend's mHC pre kernels with FlagTree's LLC optimization.

    An explicit ``TRITON_MUSA_ENABLE_LLC_OPT=0`` still opts out.
    """
    if os.environ.get("TRITON_MUSA_ENABLE_LLC_OPT") == "0":
        return 0
    opts = {"enable_llc_opt": True}
    for name in KERNELS:
        namespace[name] = _LLCOptKernel(namespace[name], opts)
    return len(KERNELS)
