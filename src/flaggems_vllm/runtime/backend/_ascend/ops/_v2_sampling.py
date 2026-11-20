# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import torch


def validate_logits(logits):
    if logits.device.type != "npu":
        raise NotImplementedError("V2 sampling transforms require an Ascend NPU")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("logits must be FP16, BF16, or FP32")
    if logits.dim() != 2 or logits.stride(1) != 1:
        raise ValueError("logits must be a matrix with contiguous columns")
    rows, columns = logits.shape
    if rows > 1 and logits.stride(0) < columns:
        raise ValueError("logits rows must not overlap")
    return rows, columns


def validate_vector(vector, device, dtypes, name, *, size=None):
    if vector.device != device or vector.dtype not in dtypes:
        raise TypeError(f"{name} has an unsupported device or dtype")
    if vector.dim() != 1 or vector.stride(0) != 1:
        raise ValueError(f"{name} must be a contiguous vector")
    if size is not None and vector.numel() != size:
        raise ValueError(f"{name} must have length {size}")
