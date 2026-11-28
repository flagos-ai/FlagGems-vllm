# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact-shape tuning of complete MUSA pipelines during eager warmup."""

import hashlib
import inspect
import json
import logging
import math
import os
import statistics
from functools import lru_cache
from pathlib import Path
from threading import RLock

import torch
import triton
from triton.compiler.errors import CompilationError

from flaggems_vllm.runtime import get_tuned_config
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.execution import (
    compile_only,
    current_config,
    is_compiling,
    use_config,
)
from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.policy import (
    candidates,
    compiler_environment,
    tensor_identity,
)
from flaggems_vllm.utils.libentry import libcache

logger = logging.getLogger(__name__)
_MEMORY = {}
_WARMED = set()
_LOCK = RLock()
TUNING_RECORDS = {}
_TABLE = "mthreads_attention_pipeline"
RESOURCE_ERRORS = (
    CompilationError,
    triton.OutOfResources,
    ValueError,
    RuntimeError,
    AssertionError,
)


@lru_cache(None)
def cache_table():
    # Config.all_kwargs() has vendor/version-specific fields. libcache's SQL
    # schema is per table, so a compiler fingerprint in the row key alone
    # cannot isolate incompatible Config schemas (MUSA Triton 3.2 vs 3.6).
    config = triton.Config({}, num_warps=4, num_stages=1)
    fields = sorted(
        (name, type(value).__name__)
        for name, value in config.all_kwargs().items()
        if isinstance(value, (int, float, str))
    )
    schema = json.dumps((triton.__version__, fields), separators=(",", ":"))
    return _TABLE + "_" + hashlib.sha256(schema.encode()).hexdigest()[:16]


@lru_cache(None)
def fingerprint(device):
    from triton._C import libtriton
    from triton.compiler.compiler import make_backend

    backend = make_backend(triton.runtime.driver.active.get_current_target())
    try:
        compiler_path = Path(inspect.getfile(type(backend)))
    except (TypeError, OSError):
        # MUSA Triton 3.2 exposes a native `mtgpu.MUSABackend` class without
        # a Python module file. Its canonical hash identifies SDK and arch.
        compiler_path = None

    root = Path(__file__).resolve().parent
    files = sorted(root.glob("*.py")) + sorted(root.glob("*.bc"))
    files += [root.parents[1] / "tune_configs.yaml"]
    if compiler_path is not None:
        files.append(compiler_path)
    digest = hashlib.sha256()
    digest.update(type(backend).__module__.encode())
    digest.update(type(backend).__qualname__.encode())
    digest.update(str(backend.hash()).encode())
    for path in files:
        digest.update(path.name.encode())
        digest.update(
            path.read_bytes().replace(b"\r\n", b"\n")
            if path.suffix != ".bc"
            else path.read_bytes()
        )
    # Compiler Python files alone do not identify a rebuilt native backend.
    with Path(libtriton.__file__).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    properties = torch.musa.get_device_properties(device)
    hardware = tuple(
        str(getattr(properties, name, "unknown"))
        for name in (
            "name",
            "major",
            "minor",
            "multi_processor_count",
            "total_memory",
        )
    )
    return (triton.__version__, torch.__version__, hardware, digest.hexdigest())


def exact_key(tensors, scalars, device):
    metadata = tuple(tensor_identity(tensor) for tensor in tensors)
    payload = (
        fingerprint(device),
        metadata,
        scalars,
        compiler_environment(os.environ),
    )
    return (json.dumps(payload, separators=(",", ":"), allow_nan=False),)


def graph_bench(fn, rep=5, quantiles=None):
    """MUSA adapter: whole-pipeline Graph replay, without CUDA monkeypatches."""
    device = torch.musa
    device.synchronize()
    stream = device.Stream()
    stream.wait_stream(device.current_stream())
    with device.stream(stream):
        for _ in range(3):
            fn()
        start, end = device.Event(enable_timing=True), device.Event(enable_timing=True)
        start.record()
        for _ in range(3):
            fn()
        end.record()
        stream.synchronize()
        estimate = start.elapsed_time(end) / 3
        count = min(256, max(1, math.ceil(rep / max(estimate, 1e-4))))
        graph = device.MUSAGraph()
        with device.graph(graph, stream=stream):
            for _ in range(count):
                fn()
        samples = []
        for _ in range(5):
            start.record()
            graph.replay()
            end.record()
            stream.synchronize()
            samples.append(start.elapsed_time(end) / count)
    ordered = sorted(samples)
    if quantiles is not None:
        return [ordered[round(q * (len(ordered) - 1))] for q in quantiles]
    return statistics.median(samples)


def compile_candidate(fn, candidate):
    with use_config(candidate), compile_only() as kernels:
        fn()
    if not kernels:
        raise RuntimeError("No MThreads kernels compiled for attention candidate")
    resources = [
        dict(
            name=k.name, registers=k.n_regs, spills=k.n_spills, shared=k.metadata.shared
        )
        for k in kernels
    ]
    if any(k["spills"] >= 40 for k in resources):
        raise ValueError(f"Attention candidate exceeds spill limit: {resources}")
    return resources


def measure_candidate(fn, candidate, rep=5):
    with use_config(candidate):
        latency = graph_bench(fn, rep=rep)
    if not math.isfinite(latency) or latency <= 0:
        raise RuntimeError(f"Invalid tuning latency: {latency}")
    return latency


def confirm_selection(fn, reference, finalists):
    """Compare several finalists while resisting positive timing interference."""
    contenders = tuple(dict.fromkeys((reference, *finalists)))
    if len(contenders) == 1:
        return reference, {}
    samples = {candidate: [] for candidate in contenders}
    for index in range(5):
        offset = index % len(contenders)
        order = contenders[offset:] + contenders[:offset]
        for candidate in order:
            samples[candidate].append(measure_candidate(fn, candidate, rep=20))
    # Contention can inflate an entire Graph timing sample. The lower quartile
    # of five observations still requires two fast observations and rejects a
    # single spuriously low sample. Preserve the seed when differences are <1%.
    scores = {candidate: sorted(values)[1] for candidate, values in samples.items()}
    winner = reference
    for candidate in contenders[1:]:
        if scores[candidate] < scores[winner] * 0.99:
            winner = candidate
    return winner, samples


def run(fn, key, problem, default):
    override = current_config()
    if override is not None or is_compiling():
        with use_config(override or default):
            return fn()
    if os.environ.get("MTHREADS_FA_AUTOTUNE", "1") == "0":
        with use_config(default):
            result = fn()
        _WARMED.add(key)
        return result
    if torch.musa.is_current_stream_capturing():
        selected = _MEMORY.get(key)
        if selected is None:
            if key not in _WARMED:
                raise RuntimeError(
                    "Warm up this exact attention shape before MUSA Graph capture"
                )
            selected = default
        with use_config(selected):
            return fn()
    with _LOCK:
        selected = _MEMORY.get(key)
        if selected is None:
            pool = candidates(
                problem, default, get_tuned_config("flash_attn_varlen_mthreads")
            )
            table = cache_table()
            cache = libcache[table]
            saved = cache.get(key)
            if saved is not None:
                selected = next((c for c in pool if c.values() == saved.kwargs), None)
            if selected is None:
                records = []
                best_ms = float("inf")
                reference = None
                measured = []
                for candidate in pool:
                    record = dict(config=candidate.values())
                    try:
                        record["resources"] = compile_candidate(fn, candidate)
                        latency = measure_candidate(fn, candidate)
                        if reference is None:
                            reference = candidate
                        record.update(status="measured", latency_ms=latency)
                        measured.append((latency, candidate))
                        config = triton.Config(
                            candidate.values(), num_warps=4, num_stages=1
                        )
                        libcache[(table, key)][config] = (latency, latency, latency)
                        if latency < best_ms * 0.99:
                            best_ms, selected = latency, candidate
                    except RESOURCE_ERRORS as error:
                        record.update(status="rejected", error=str(error))
                        logger.debug(
                            "Rejected attention candidate %s: %s", candidate, error
                        )
                    records.append(record)
                TUNING_RECORDS[key] = records
                if selected is None:
                    raise RuntimeError(f"No legal attention configuration: {records}")
                finalists = [
                    candidate
                    for _, candidate in sorted(measured, key=lambda item: item[0])[:3]
                ]
                selected, confirmations = confirm_selection(fn, reference, finalists)
                for candidate, samples in confirmations.items():
                    for record in records:
                        if record["config"] == candidate.values():
                            record["confirmation_ms"] = samples
                            record["confirmation_score_ms"] = sorted(samples)[1]
                    low, _, _, high = statistics.quantiles(
                        samples, n=5, method="inclusive"
                    )
                    config = triton.Config(
                        candidate.values(), num_warps=4, num_stages=1
                    )
                    libcache[(table, key)][config] = (
                        statistics.median(samples),
                        low,
                        high,
                    )
                cache[key] = triton.Config(selected.values(), num_warps=4, num_stages=1)
            _MEMORY[key] = selected
    with use_config(selected):
        result = fn()
    _WARMED.add(key)
    return result
