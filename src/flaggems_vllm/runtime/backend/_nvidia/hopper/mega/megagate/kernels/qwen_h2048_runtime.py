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

"""Qwen-H2048 runner for lowerings with retained global scratch."""

from math import prod
from typing import Any, Sequence

from .runtime import normalize_grid


class ExactGridUnboundPreallocatedScratchCompiledRunner:
    """Functional exact-grid launch with caller-selected args and scratch.

    This is the unbound counterpart of
    :class:`ExactGridPreallocatedScratchCompiledRunner`.  It retains only the
    compiled specialization and low-level launcher; a functional operator
    supplies the current tensor arguments, per-stream global scratch, and
    current stream handle for every invocation.  Scratch size and cooperative
    launch semantics are validated once at construction and again for the
    caller-provided storage before launch.
    """

    __slots__ = (
        "_compiled",
        "_expected_argument_count",
        "_expected_grid",
        "_launcher",
        "_normalized_grid",
        "_required_global_scratch_alignment",
        "_required_global_scratch_bytes",
    )

    def __init__(
        self,
        *,
        compiled: Any,
        grid: Sequence[int],
        expected_argument_count: int,
    ) -> None:
        if type(expected_argument_count) is not int or expected_argument_count <= 0:
            raise ValueError("expected_argument_count must be a positive integer")
        expected_grid = tuple(grid)
        normalized_grid = normalize_grid(expected_grid)
        metadata = compiled.metadata
        global_scratch_size = int(metadata.global_scratch_size)
        if global_scratch_size <= 0:
            raise ValueError("unbound preallocated runner requires global scratch")
        if int(metadata.profile_scratch_size) != 0:
            raise ValueError("unbound preallocated runner rejects profile scratch")
        if bool(metadata.launch_cooperative_grid):
            raise ValueError("unbound preallocated runner rejects cooperative kernels")
        launcher = compiled.run
        if not callable(getattr(launcher, "launch", None)):
            raise TypeError("compiled CUDA launcher has no low-level launch")

        self._compiled = compiled
        self._expected_argument_count = expected_argument_count
        self._expected_grid = expected_grid
        self._normalized_grid = normalized_grid
        self._launcher = launcher
        self._required_global_scratch_alignment = max(
            1, int(getattr(metadata, "global_scratch_align", 1))
        )
        self._required_global_scratch_bytes = (
            prod(normalized_grid)
            * int(getattr(metadata, "num_ctas", 1))
            * global_scratch_size
        )

    def _launch_unbound(
        self,
        *launch_args: Any,
        global_scratch: Any,
        stream: int | None = None,
    ) -> None:
        if len(launch_args) != self._expected_argument_count:
            raise ValueError(
                "unbound cached runner argument count mismatch: expected "
                f"{self._expected_argument_count}, got {len(launch_args)}"
            )
        storage_bytes = int(global_scratch.numel()) * int(global_scratch.element_size())
        if storage_bytes < self._required_global_scratch_bytes:
            raise ValueError(
                "preallocated global scratch is too small: required="
                f"{self._required_global_scratch_bytes}, got={storage_bytes}"
            )
        device = getattr(global_scratch, "device", None)
        if device is None or getattr(device, "type", None) != "cuda":
            raise ValueError("preallocated global scratch must be a CUDA tensor")
        is_contiguous = getattr(global_scratch, "is_contiguous", None)
        if not callable(is_contiguous) or not bool(is_contiguous()):
            raise ValueError("preallocated global scratch must be contiguous")
        data_ptr = getattr(global_scratch, "data_ptr", None)
        if not callable(data_ptr):
            raise TypeError("preallocated global scratch has no data_ptr()")
        address = int(data_ptr())
        if address % self._required_global_scratch_alignment:
            raise ValueError(
                "preallocated global scratch alignment mismatch: required="
                f"{self._required_global_scratch_alignment}, address={address}"
            )

        # Delayed imports keep CPU-only source and static tests independent of
        # an installed Triton runtime.
        from triton import knobs
        from triton.runtime.driver import driver

        compiled = self._compiled
        grid = self._normalized_grid
        if stream is None:
            device = driver.active.get_current_device()
            stream = driver.active.get_current_stream(device)
        launch_metadata = compiled.launch_metadata(grid, stream, *launch_args)
        launcher = self._launcher
        launcher.launch(
            grid[0],
            grid[1],
            grid[2],
            stream,
            compiled.function,
            launcher.launch_cooperative_grid,
            launcher.launch_pdl,
            global_scratch,
            None,
            compiled.packed_metadata,
            launch_metadata,
            knobs.runtime.launch_enter_hook,
            knobs.runtime.launch_exit_hook,
            *launch_args,
        )

    def __getitem__(self, grid: Sequence[int]) -> Any:
        observed = tuple(grid)
        if observed != self._expected_grid:
            raise ValueError(
                "cached runner is exact-grid only: expected "
                f"{self._expected_grid}, got {observed}"
            )
        return self._launch_unbound

    @property
    def compiled(self) -> Any:
        return self._compiled

    @property
    def normalized_grid(self) -> tuple[int, int, int]:
        return self._normalized_grid

    @property
    def expected_argument_count(self) -> int:
        return self._expected_argument_count

    @property
    def required_global_scratch_bytes(self) -> int:
        return self._required_global_scratch_bytes

    @property
    def required_global_scratch_alignment(self) -> int:
        return self._required_global_scratch_alignment


__all__ = ["ExactGridUnboundPreallocatedScratchCompiledRunner"]
