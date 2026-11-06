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

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_RANK = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gate_sigmoid_mul"),
    key=["N", "WIDTH", "LAYOUT_KEY"],
    warmup=5,
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _gate_sigmoid_mul_kernel(
    A,
    G,
    O,
    N,
    WIDTH: tl.constexpr,
    SHAPE: tl.constexpr,
    A_STRIDES: tl.constexpr,
    G_STRIDES: tl.constexpr,
    FLAT: tl.constexpr,
    LAYOUT_KEY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = index < N
    if FLAT:
        a_offset = index
        g_offset = index
    else:
        remaining = index
        a_offset = tl.full((BLOCK_SIZE,), 0, tl.int64)
        g_offset = tl.full((BLOCK_SIZE,), 0, tl.int64)
        for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
            coordinate = remaining % SHAPE[dim]
            remaining = remaining // SHAPE[dim]
            a_offset += coordinate * A_STRIDES[dim]
            g_offset += coordinate * G_STRIDES[dim]
    a = tl.load(A + a_offset, mask=mask, other=0).to(tl.float32)
    g = tl.load(G + g_offset, mask=mask, other=0).to(tl.float32)
    # The pinned function slice compiles with an FP32 sigmoid intermediate.
    sigmoid = 1.0 / (1.0 + tl.exp(-g))
    value32 = a * sigmoid
    value = value32.to(O.dtype.element_ty)
    if O.dtype.element_ty == tl.float16:
        # HCU's mixed FMA lowering adds +0 while converting the product to FP16,
        # which loses negative zero. Restore only the sign of rounded zeros.
        bits = value.to(tl.uint16, bitcast=True)
        sign = ((value32.to(tl.uint32, bitcast=True) >> 16) & 0x8000).to(tl.uint16)
        bits = tl.where((bits & 0x7FFF) == 0, sign, bits)
        value = bits.to(tl.float16, bitcast=True)
    tl.store(O + index, value, mask=mask)


def gate_sigmoid_mul(attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply Kimi-K3's output gate in one launch (forward inference only).

    Supports broadcast strided FP16/BF16/FP32 inputs of rank at most eight.
    The result uses PyTorch's dtype promotion and independent contiguous storage.
    """
    logger.debug("GEMS_HYGON GATE_SIGMOID_MUL")
    if not isinstance(attn_out, torch.Tensor) or not isinstance(gate, torch.Tensor):
        raise TypeError("attn_out and gate must be tensors")
    if attn_out.device != gate.device:
        raise ValueError("attn_out and gate must be on the same device")
    if attn_out.device.type in ("cpu", "meta"):
        raise NotImplementedError("gate_sigmoid_mul requires an accelerator device")
    if attn_out.layout != torch.strided or gate.layout != torch.strided:
        raise NotImplementedError("gate_sigmoid_mul requires strided tensors")
    if attn_out.dtype not in _SUPPORTED_DTYPES or gate.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError("gate_sigmoid_mul supports FP16, BF16 and FP32")
    rank = max(attn_out.ndim, gate.ndim)
    if rank > _MAX_RANK:
        raise NotImplementedError("gate_sigmoid_mul supports rank at most eight")
    a_shape = (1,) * (rank - attn_out.ndim) + tuple(attn_out.shape)
    g_shape = (1,) * (rank - gate.ndim) + tuple(gate.shape)
    a_stride = (0,) * (rank - attn_out.ndim) + attn_out.stride()
    g_stride = (0,) * (rank - gate.ndim) + gate.stride()
    shape, a_strides, g_strides = [], [], []
    n = 1
    for a_size, g_size, a_step, g_step in zip(a_shape, g_shape, a_stride, g_stride):
        if a_size != g_size and a_size != 1 and g_size != 1:
            raise ValueError("attn_out and gate shapes are not broadcastable")
        size = g_size if a_size == 1 else a_size
        shape.append(size)
        a_strides.append(0 if a_size == 1 else a_step)
        g_strides.append(0 if g_size == 1 else g_step)
        n *= size
    # result_type is a metadata query, including zero-dimensional promotion.
    output = torch.empty(
        tuple(shape), device=attn_out.device, dtype=torch.result_type(attn_out, gate)
    )
    if n == 0:
        return output
    flat = (
        attn_out.shape == gate.shape
        and attn_out.is_contiguous()
        and gate.is_contiguous()
    )
    width = shape[-1] if shape else 1
    layout_key = f"{shape}:{a_strides}:{g_strides}:{flat}"
    with runtime.torch_device_fn.device(attn_out.device):
        _gate_sigmoid_mul_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)](
            attn_out,
            gate,
            output,
            n,
            width,
            tuple(shape),
            tuple(a_strides),
            tuple(g_strides),
            flat,
            layout_key,
        )
    return output
