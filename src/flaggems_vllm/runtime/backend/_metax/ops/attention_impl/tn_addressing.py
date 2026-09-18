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

"""Select safe address compilation for the MetaX async-TN kernels."""


def tn_compile_scenario(*tensors):
    """Enable the 4 GiB address optimization only for bounded byte spans."""
    for tensor in tensors:
        if tensor is None or tensor.numel() == 0:
            continue
        # numel() misses holes in noncontiguous caches. The origin is the
        # pointer passed to the kernel, so storage_offset is not added again.
        last_element = sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        if (last_element + 1) * tensor.element_size() > (1 << 32):
            # cpasync otherwise enables metaxgpu-aggressive-4g-addr-opt,
            # which truncates large K/V byte offsets even with int64 source.
            return "storeCoalesce;noaddropt"
    return "storeCoalesce"
