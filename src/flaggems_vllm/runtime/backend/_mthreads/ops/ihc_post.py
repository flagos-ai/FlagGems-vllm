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

"""MThreads iHC tile with exact FP16 writeback."""

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


@triton.jit
def _fp16_rne(value):
    # Vectorized FP32 -> FP16 narrowing on mp_31 can misround halfway values.
    bits = value.to(tl.uint32, bitcast=True)
    magnitude = bits & 0x7FFFFFFF
    sign = (bits >> 16) & 0x8000

    # Normal half: remove 13 fraction bits, resolving ties with the retained LSB.
    rounded = (magnitude + 0xFFF + ((magnitude >> 13) & 1)) >> 13
    normal = tl.minimum(rounded - 0x1C000, 0x7C00)

    # Subnormal half: shift the implicit leading bit into the 10-bit fraction.
    exponent = (magnitude >> 23).to(tl.int32)
    shift = tl.maximum(14, tl.minimum(24, 126 - exponent)).to(tl.uint32)
    significand = (magnitude & 0x7FFFFF) | 0x800000
    bias = (1 << (shift - 1)) - 1 + ((significand >> shift) & 1)
    subnormal = tl.where(exponent < 102, 0, (significand + bias) >> shift)

    half = tl.where(magnitude < 0x38800000, subnormal, normal)
    half = tl.where(magnitude > 0x7F800000, 0x7E00, half)
    return (sign | half).to(tl.uint16).to(tl.float16, bitcast=True)


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
        token = pid % NUM_TOKENS
        hidden_block = pid // NUM_TOKENS
    else:
        hidden_block = pid % hidden_blocks
        token = pid // hidden_blocks
    channel = tl.arange(0, HC_POW2)[:, None]
    hidden = (hidden_block * BLOCK_D + tl.arange(0, BLOCK_D))[None, :]
    offsets = (token * HC_MULT + channel) * HIDDEN_SIZE + hidden

    channel_valid = channel < HC_MULT
    hidden_valid = hidden < HIDDEN_SIZE
    mask = channel_valid & hidden_valid
    gate = tl.load(post_ptr + token * HC_MULT + channel, mask=channel_valid, other=0)
    x = tl.load(x_ptr + token * HIDDEN_SIZE + hidden, mask=hidden_valid, other=0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0)
    result = gate * x.to(tl.float32) + residual.to(tl.float32)
    tl.store(output_ptr + offsets, _fp16_rne(result), mask=mask)
