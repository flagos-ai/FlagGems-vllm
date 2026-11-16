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

"""MetaX iHC tile with explicit FP32 rounding barriers.

FlagTree 0.7.0+metax3.6 contracts plain mul/add despite enable_fp_fusion=False.
The libdevice intrinsics preserve the multiplication rounding boundary.
"""

import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("triton_ihc_post"),
    key=["NUM_TOKENS", "HC_MULT", "HIDDEN_SIZE", "DTYPE_TAG", "LAYOUT_TAG"],
    warmup=5,
    rep=10,
    use_cuda_graph=False,
)
@triton.jit
def _ihc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    DTYPE_TAG: tl.constexpr,
    LAYOUT_TAG: tl.constexpr,
    TOKEN_MAJOR: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    hidden_blocks = tl.cdiv(HIDDEN_SIZE, BLOCK_D)
    if TOKEN_MAJOR:
        token_idx = pid % NUM_TOKENS
        hidden_block_idx = pid // NUM_TOKENS
    else:
        hidden_block_idx = pid % hidden_blocks
        token_idx = pid // hidden_blocks
    EVEN_D: tl.constexpr = HIDDEN_SIZE % BLOCK_D == 0
    EVEN_HC: tl.constexpr = HC_MULT == HC_POW2

    hc_idx = tl.arange(0, HC_POW2)[:, None]
    hidden_offsets = (hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D))[None, :]
    # post[n, i] broadcast over the hidden axis: [HC_POW2, 1] * [1, BLOCK_D].
    gate_ptr = post_ptr + token_idx * HC_MULT + hc_idx
    if EVEN_HC:
        gate = tl.load(gate_ptr)
    else:
        gate = tl.load(gate_ptr, mask=hc_idx < HC_MULT, other=0.0)

    offsets = token_idx * HC_MULT * HIDDEN_SIZE + hc_idx * HIDDEN_SIZE + hidden_offsets
    if EVEN_D and EVEN_HC:
        x = tl.load(x_ptr + token_idx * HIDDEN_SIZE + hidden_offsets).to(tl.float32)
        residual = tl.load(residual_ptr + offsets).to(tl.float32)
        output = libdevice.add_rn(libdevice.mul_rn(gate, x), residual)
        tl.store(output_ptr + offsets, output.to(output_ptr.dtype.element_ty))
    else:
        hidden_mask = hidden_offsets < HIDDEN_SIZE
        mask = hidden_mask if EVEN_HC else (hc_idx < HC_MULT) & hidden_mask
        x = tl.load(
            x_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        output = libdevice.add_rn(libdevice.mul_rn(gate, x), residual)
        tl.store(
            output_ptr + offsets,
            output.to(output_ptr.dtype.element_ty),
            mask=mask,
        )
