# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public patch-affine contract; runtime selects the numerical backend."""


def fused_mm_input_norm(inputs, outputs, weight, bias):
    if inputs.ndim != 3 or outputs.shape != inputs.shape:
        raise ValueError("normalization input/output must share shape [N, C, L]")
    raise NotImplementedError("patch affine normalization requires the NVIDIA backend")


__all__ = ["fused_mm_input_norm"]
