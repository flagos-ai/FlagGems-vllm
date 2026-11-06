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

import torch


def torch_w8a8_block_matmul(A, B, As, Bs, block_size, output_dtype=torch.bfloat16):
    """Torch-only reference for 2-D block-scaled matmul, on the input device.

    Each K group has its own activation/weight scales. Accumulate scaled
    group dot products in FP32 before converting the final output.
    This is a test/benchmark baseline, not a production fallback.
    """
    m, k = A.shape
    n = B.shape[0]
    block_n, block_k = block_size
    a = A.float()
    b = B.float()
    a_scale = As.float()
    b_scale = Bs.float()
    result = torch.zeros((m, n), dtype=torch.float32, device=A.device)
    for group, start in enumerate(range(0, k, block_k)):
        end = min(start + block_k, k)
        partial = a[:, start:end] @ b[:, start:end].T
        partial.mul_(a_scale[:, group, None])
        weight_scale = b_scale[:, group].repeat_interleave(block_n)[:n]
        partial.mul_(weight_scale[None, :])
        result.add_(partial)
    return result.to(output_dtype)
