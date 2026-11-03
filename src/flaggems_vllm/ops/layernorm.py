# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Sequence, Union

import torch


def layernorm(
    input: torch.Tensor,
    normalized_shape: Union[int, Sequence[int]],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm inference entry point; currently implemented for MetaX."""
    from flaggems_vllm import runtime

    if runtime.device.vendor_name != "metax":
        raise NotImplementedError("layernorm is currently available on MetaX only")
    from flaggems_vllm.runtime.backend._metax.ops.layernorm import layernorm as impl

    return impl(input, normalized_shape, weight, bias, eps)
