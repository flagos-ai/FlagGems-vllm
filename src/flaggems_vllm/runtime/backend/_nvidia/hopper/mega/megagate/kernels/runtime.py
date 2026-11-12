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

"""Host adapter for an exact-grid compiled Triton kernel runner."""

from __future__ import annotations

from typing import Any, Sequence


def normalize_grid(grid: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(grid)
    if not 1 <= len(values) <= 3:
        raise ValueError("cached launch grid must have one to three axes")
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("cached launch grid axes must be positive integers")
    return values + (1,) * (3 - len(values))


class ExactGridUnboundCompiledRunner:
    """Retain one loaded exact-grid runner without binding tensor arguments.

    The compiled specialization and launch grid are fixed by ``prepare``.
    Runtime descriptors, pointers, outputs, and the current stream remain
    invocation-owned and are supplied to :meth:`_launch_unbound`.
    """

    __slots__ = (
        "_compiled",
        "_expected_argument_count",
        "_expected_grid",
        "_normalized_grid",
        "_runner",
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
        runner = compiled[normalized_grid]
        if not callable(runner):
            raise TypeError("CompiledKernel did not provide a callable runner")

        self._compiled = compiled
        self._expected_argument_count = expected_argument_count
        self._expected_grid = expected_grid
        self._normalized_grid = normalized_grid
        self._runner = runner

    def _launch_unbound(self, *launch_args: Any, stream: int | None = None) -> Any:
        if len(launch_args) != self._expected_argument_count:
            raise ValueError(
                "unbound cached runner argument count mismatch: expected "
                f"{self._expected_argument_count}, got {len(launch_args)}"
            )
        if stream is None:
            return self._runner(*launch_args)
        return self._runner(*launch_args, stream=stream)

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


__all__ = ["ExactGridUnboundCompiledRunner", "normalize_grid"]
