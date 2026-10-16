"""Utility functions extracted from vLLM framework dependencies."""

import torch


class _CurrentPlatform:
    """Small platform adapter used by the MSA Triton kernels."""

    def is_arch_support_pdl(self) -> bool:
        if not torch.cuda.is_available():
            return False
        # HIP/ROCm does not accept CUDA's launch_pdl runtime argument.
        if torch.version.hip is not None:
            return False
        try:
            capability = torch.cuda.get_device_capability()
        except RuntimeError:
            return False
        # PDL is enabled only for CUDA devices with the required capability.
        return capability >= (9, 0)


current_platform = _CurrentPlatform()


def round_up(n: int, d: int) -> int:
    """Round n up to the nearest multiple of d."""
    return (n + d - 1) // d * d
