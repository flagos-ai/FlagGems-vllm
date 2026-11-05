# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA clamped-SiLU contracts selected through the normal backend registry."""


def supports_silu_clamp_int8_quant(input, activation_output, clamp_limit):
    return False


def silu_clamp_int8_quant(input, activation_output, clamp_limit):
    raise NotImplementedError(
        "clamped SiLU/INT8 requires the registered NVIDIA backend"
    )


def silu_clamp(input, activation_output, clamp_limit):
    raise NotImplementedError("clamped SiLU requires the registered NVIDIA backend")


__all__ = ["silu_clamp", "silu_clamp_int8_quant", "supports_silu_clamp_int8_quant"]


def supports_int8_quantize_per_token(x):
    return False


def int8_quantize_per_token(x):
    raise NotImplementedError("per-token INT8 requires the registered NVIDIA backend")


__all__ += ["int8_quantize_per_token", "supports_int8_quantize_per_token"]
