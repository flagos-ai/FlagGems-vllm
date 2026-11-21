# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compatibility for the tested MetaX Torch 2.8 / Triton 3.6 benchmark stack."""

import torch
import triton

if torch.__version__.split(".")[:2] == ["2", "8"] and triton.__version__.split(".")[
    :2
] == ["3", "6"]:
    import triton.compiler.compiler as compiler

    if "triton_key" not in vars(compiler):
        from triton.runtime.cache import triton_key

        # Inductor 2.8 imports the cache fingerprint from the old module path.
        compiler.triton_key = triton_key
