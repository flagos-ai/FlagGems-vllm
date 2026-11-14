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

"""Offline collection and fixed-config timing helpers for LibTuner."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import triton

if TYPE_CHECKING:
    from flaggems_vllm.utils.libentry import LibEntry, LibTuner

DEFAULT_BENCHMARK_RETRIES = 10


try:
    from triton.flagtune.runtime.benchmark_protocol import (
        BenchmarkMode,
        BenchmarkProtocol,
        resolve_benchmarker,
    )
except ModuleNotFoundError as exc:
    if exc.name not in (
        "triton.flagtune",
        "triton.flagtune.runtime.benchmark_protocol",
    ):
        raise
    from .benchmark_protocol import (
        BenchmarkMode,
        BenchmarkProtocol,
        resolve_benchmarker,
    )


class LibTunerRunMode(str, Enum):
    """Control one scoped :class:`LibTuner.run` config-selection pass."""

    NORMAL = "normal"
    FORCE_POLICY = "force_policy"
    EXHAUSTIVE_COLLECTION = "exhaustive_collection"


@contextmanager
def use_run_mode(self, mode: Union[LibTunerRunMode, str]) -> Iterator[None]:
    """Temporarily bypass best-config caching for policy or exhaustive collection."""
    selected = LibTunerRunMode(mode)
    previous = getattr(self, "_run_mode", LibTunerRunMode.NORMAL)
    self._run_mode = selected
    try:
        yield
    finally:
        self._run_mode = previous


@contextmanager
def use_benchmark_protocol(
    self,
    benchmark_mode: Union[BenchmarkMode, str],
    warmup: int,
    rep: int,
    benchmark_retries: int = DEFAULT_BENCHMARK_RETRIES,
) -> Iterator[BenchmarkProtocol]:
    """Scope offline timing and its cache identity, restoring tuner state on exit."""
    selected_mode = BenchmarkMode(benchmark_mode)
    resolved = resolve_benchmarker(
        selected_mode,
        warmup_ms=warmup,
        measurement_ms=rep,
        n_retries=benchmark_retries,
    )
    previous_protocol = self._benchmark_protocol
    previous_public_protocol = getattr(self, "benchmark_protocol", None)
    previous_do_bench = self.do_bench
    previous_retries = getattr(self, "_benchmark_retries", DEFAULT_BENCHMARK_RETRIES)
    previous_table_name = self.config_table_name
    previous_cache = self.cache
    self._benchmark_protocol = resolved.protocol.cache_key()
    self.benchmark_protocol = resolved.protocol
    self._benchmark_retries = benchmark_retries
    self.do_bench = resolved.benchmark
    self.config_table_name = self._make_config_table_name()
    self.cache = self._flagtune_cache[self.config_table_name]
    try:
        yield resolved.protocol
    finally:
        self._benchmark_protocol = previous_protocol
        self.benchmark_protocol = previous_public_protocol
        self._benchmark_retries = previous_retries
        self.do_bench = previous_do_bench
        self.config_table_name = previous_table_name
        self.cache = previous_cache


def benchmark_config(
    self,
    config: triton.Config,
    *,
    warmup: int,
    rep: int,
    benchmark_mode: Optional[Union[BenchmarkMode, str]] = None,
    benchmark_retries: Optional[int] = None,
    quantiles: Tuple[float, ...] = (0.5, 0.2, 0.8),
    args: Optional[Tuple[Any, ...]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """Measure a fixed config without reading or writing tuning caches.

    Explicit args/meta override the most recent tuner invocation. Quantiles are
    returned in the requested order; native event/replay timing is preserved."""
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if rep <= 0:
        raise ValueError("rep must be positive")
    if not quantiles or any(value < 0 or value > 1 for value in quantiles):
        raise ValueError("quantiles must contain values between 0 and 1")
    benchmark_quantiles = tuple(quantiles)
    benchmark_args = (
        tuple(args) if args is not None else getattr(self, "_last_benchmark_args", None)
    )
    benchmark_meta = (
        dict(meta) if meta is not None else getattr(self, "_last_benchmark_meta", None)
    )
    if benchmark_args is None or benchmark_meta is None:
        raise RuntimeError(
            "benchmark_config requires explicit args/meta or a prior LibTuner.run"
        )

    original_nargs = getattr(self, "nargs", None)
    current_protocol = getattr(self, "benchmark_protocol", None)
    selected_mode = (
        benchmark_mode
        if benchmark_mode is not None
        else (
            current_protocol.requested_mode
            if current_protocol is not None
            else (
                BenchmarkMode.REPLAY
                if self._benchmark_protocol[0] == "triton_do_bench_cudagraph"
                else BenchmarkMode.EVENT
            )
        )
    )
    selected_retries = (
        benchmark_retries
        if benchmark_retries is not None
        else getattr(self, "_benchmark_retries", DEFAULT_BENCHMARK_RETRIES)
    )
    self.nargs = dict(zip(self.arg_names, benchmark_args))
    if hasattr(self, "seen_tuned_metas"):
        self.seen_tuned_metas = {}
    try:
        with use_benchmark_protocol(
            self,
            selected_mode,
            warmup,
            rep,
            selected_retries,
        ):
            protocol_benchmark = self.do_bench

            def benchmark_with_requested_quantiles(kernel_call, quantiles):
                # Triton's _bench always requests its canonical
                # (p50, p20, p80) order. This fixed-config API promises the
                # caller's explicit order, so override only that argument
                # while retaining the resolved event/replay implementation.
                return protocol_benchmark(
                    kernel_call,
                    quantiles=benchmark_quantiles,
                )

            self.do_bench = benchmark_with_requested_quantiles
            bench_ret = self._bench(
                *benchmark_args,
                config=config,
                **benchmark_meta,
            )
            # Normalize scalar float to 3-element tuple for consistency
            if isinstance(bench_ret, (int, float)):
                bench_ret = (bench_ret, bench_ret, bench_ret)
            return list(bench_ret)
    finally:
        self.nargs = original_nargs


def find_flagtune_benchmark_target(
    public_operator: Callable[..., Any], op_id: str, variant: str
) -> Tuple["LibEntry", LibTuner]:
    """Find a unique tuner by model identity in the public operator's own module."""
    from flaggems_vllm.utils.libentry import LibEntry, LibTuner

    module = importlib.import_module(public_operator.__module__)
    candidates: Dict[int, Tuple[LibEntry, LibTuner]] = {}
    for outer in vars(module).values():
        if not isinstance(outer, LibEntry):
            continue
        current = outer.fn
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if (
                isinstance(current, LibTuner)
                and current._flagtune_op_id == op_id
                and current._flagtune_variant == variant
            ):
                candidates[id(current)] = (outer, current)
            current = getattr(current, "fn", None)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one LibTuner for op_id={op_id!r}, "
            f"variant={variant!r}; found {len(candidates)}"
        )
    return next(iter(candidates.values()))


def clear_libentry_dispatch_cache(entry: "LibEntry") -> None:
    """Clear wrapper dispatch caches only; retain compiled kernels and SQLite data."""
    for cache in entry.kernel_cache:
        cache.clear()
    entry._cpu_cache.clear()
