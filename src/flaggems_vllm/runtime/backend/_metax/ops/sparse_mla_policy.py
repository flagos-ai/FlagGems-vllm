# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
# Launch proxy adapted from vllm_fl.patches.triton_kernel.
"""MetaX attention policy and local bindings; library tuners stay unchanged."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class KernelLaunchMetaProxy:
    """Override selected launch parameters while preserving the kernel API."""

    def __init__(self, kernel: Any, launch_overrides: Mapping[str, Any]) -> None:
        if not launch_overrides:
            raise ValueError("At least one kernel launch override is required.")
        self._kernel = kernel
        # Keep a plain dict on this hot path. The public property returns a copy.
        self._launch_overrides = dict(launch_overrides)

    @property
    def launch_overrides(self) -> dict[str, Any]:
        return self._launch_overrides.copy()

    def __getitem__(self, grid: Any) -> Any:
        launch = self._kernel[grid]

        def launch_with_overrides(*args: Any, **kwargs: Any) -> Any:
            kwargs.update(self._launch_overrides)
            return launch(*args, **kwargs)

        return launch_with_overrides

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kernel, name)


@dataclass(frozen=True)
class KernelPolicy:
    decode_split_topk: bool = True
    decode_splits: int = 4
    decode_block_k: int = 16
    decode_block_h: int = 16
    decode_warps: int = 4
    decode_stages: int = 1
    decode_max_batch: int = 64
    decode_qkfold: bool = True
    prefill_half_d: bool = True
    prefill_blockdiag: bool = True
    prefill_staged: bool = True


DEFAULT_POLICY = KernelPolicy()


def _jit_function(kernel):
    """Bind known metadata after unwrapping autotuners, without mutating them.

    Passing duplicate metadata through an existing autotuner can conflict with
    its own Config. This handle is owned by the DS path; generic callers retain
    the original library kernel and tuner.
    """
    import triton

    seen = set()
    while not isinstance(kernel, triton.runtime.JITFunction):
        if id(kernel) in seen:
            raise TypeError("Cyclic or unsupported Triton wrapper")
        seen.add(id(kernel))
        kernel = getattr(kernel, "fn", None)
        if kernel is None:
            raise TypeError("Expected a JITFunction or a wrapper exposing fn")
    return kernel


def local_launch(kernel, overrides):
    return KernelLaunchMetaProxy(_jit_function(kernel), overrides)
