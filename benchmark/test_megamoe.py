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

"""Benchmark the multi-rank Hopper MegaMoE kernel on real FP8 data.

MegaMoE requires one process and one GPU per rank, so this benchmark launches
its MPI entry point as a subprocess.  Collective latency is the slowest rank's
CUDA-event latency.
"""

import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch

from .conftest import Config, emit_record_logger, update_result
from .consts import (
    DEFAULT_ITER_TIME,
    DEFAULT_WARMUP_TIME,
    BenchmarkMetrics,
    BenchmarkResult,
)

_MEGAMOE_MODULE = "flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamoe"

try:
    from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamoe import (
        MEGAMOE_KERNEL_PATH,
    )
except ModuleNotFoundError as exc:
    # A non-NVIDIA package may omit this module or one of its parent packages.
    # Do not turn a missing dependency or another package-import bug into a
    # skipped benchmark.
    if exc.name is None or not (
        exc.name == _MEGAMOE_MODULE or _MEGAMOE_MODULE.startswith(f"{exc.name}.")
    ):
        raise
    MEGAMOE_KERNEL_PATH = None

OP_NAME = "megamoe"


# The defaults are the previously validated UserHopper-aligned launch shape.
# Keep the benchmark geometry tied to the same variables consumed by kernel.py
# so a different real-data manifest does not require editing this file.
def _shape_env(name, default):
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


NUM_RANKS = _shape_env("MEGAMOE_NP", 8)
HIDDEN = _shape_env("W_K", 4096)
INTERMEDIATE = _shape_env("W_INTER", 1536)
NUM_EXPERTS = _shape_env("W_NEXP", 128)
TOPK = _shape_env("W_TOPK", 8)
STAGES = _shape_env("W_STAGES", 4)

DEFAULT_TOKENS = (512, 1024, 2048)
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 30
DEFAULT_TIMEOUT_S = 1800

# "[rank 0/8] BENCH h=4096 ih=1536 E=128 k=8 tokens=512 recv=4096 experts=16 |
#     351.2 us   123.4 TFLOPS  ws=True ... data=Qwen/Qwen3-235B-A22B-FP8@...:layer0"
BENCH_LINE = re.compile(
    r"\[rank (?P<rank>\d+)/(?P<ranks>\d+)\] BENCH "
    r"h=(?P<hidden>\d+) ih=(?P<intermediate>\d+) E=(?P<experts>\d+) "
    r"k=(?P<topk>\d+) tokens=(?P<tokens>\d+) recv=(?P<recv>\d+) "
    r"experts=(?P<experts_per_rank>\d+) \|\s*(?P<us>[0-9.]+) us\s+"
    r"(?P<tflops>[0-9.]+) TFLOPS"
)


def _iteration_count(name, configured, configured_default, fallback):
    """Resolve fixed iteration counts while honoring pytest benchmark options."""
    raw = os.environ.get(name, "").strip()
    if raw:
        return int(raw)
    if configured != configured_default:
        return configured
    return fallback


def _env_int(name, default):
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _resolve_mpirun():
    override = os.environ.get("MEGAMOE_MPIRUN", "").strip()
    if override:
        return shutil.which(override)
    return shutil.which("mpirun")


def _interpreter():
    """Python that runs the entry point and, through it, the MPI workers.

    The entry point imports the FlagTree/TLE Triton at module scope, so it has
    to run under the TLE interpreter.  That is usually not the interpreter
    running pytest, which only needs torch: TLE_PYTHON_OVERRIDE names it, and
    the entry point already uses the same variable for its workers.
    """
    override = os.environ.get("TLE_PYTHON_OVERRIDE", "").strip()
    return override or sys.executable


def _token_sweep():
    raw = os.environ.get("MEGAMOE_BENCH_TOKENS", "").strip()
    if not raw:
        return list(DEFAULT_TOKENS)
    return sorted({int(token) for token in raw.replace(",", " ").split()})


def _skip_reason():
    if MEGAMOE_KERNEL_PATH is None or not MEGAMOE_KERNEL_PATH.is_file():
        return "the Hopper MegaMoE entry point is not available in this build"
    if not torch.cuda.is_available():
        return "requires cuda"
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        return f"requires SM90, got SM{major}0"
    visible = torch.cuda.device_count()
    if visible < NUM_RANKS:
        return f"requires {NUM_RANKS} visible GPUs, got {visible}"
    if _resolve_mpirun() is None:
        return "an OpenMPI launcher is required; set MEGAMOE_MPIRUN"
    if not os.environ.get("MEGAMOE_SHARED_DATA_DIR", "").strip():
        return (
            "set MEGAMOE_SHARED_DATA_DIR to the immutable Qwen3 FP8 dataset; "
            "the synthetic path is a regression mode, not a benchmark"
        )
    return None


def _child_env(data_dir, tokens, warmup, iters, timeout_s, launcher):
    env = os.environ.copy()
    env.update(
        {
            "MEGAMOE_NP": str(NUM_RANKS),
            "MEGAMOE_MPIRUN": launcher,
            "MEGAMOE_SHARED_DATA_DIR": data_dir,
            "W_NTOK": str(tokens),
            "W_TOPK": str(TOPK),
            "W_NEXP": str(NUM_EXPERTS),
            "W_K": str(HIDDEN),
            "W_INTER": str(INTERMEDIATE),
            "W_DROP": "0",
            "W_STAGES": str(STAGES),
            "W_BENCH": "1",
            "W_WARMUP": str(warmup),
            "W_ITERS": str(iters),
            "W_BENCH_REDUCE": "mean",
            "W_GPU_START_BARRIER": "1",
            "W_TIMEOUT": str(timeout_s),
        }
    )
    # Each MPI worker recomputes its own cache directory from its rank; an
    # inherited one would put every rank on a single Triton cache lock.
    env.pop("TRITON_CACHE_DIR", None)
    return env


def _terminate_process_group(proc):
    """Stop the benchmark session and collect its final output."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    try:
        return proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return proc.communicate()


def _failure_output(summary, stdout, stderr):
    return (
        f"{summary}\n--- stdout tail ---\n{(stdout or '')[-3000:]}\n"
        f"--- stderr tail ---\n{(stderr or '')[-2000:]}"
    )


def _run_one(data_dir, tokens, warmup, iters, timeout_s, launcher):
    """Run one token count end to end and return its per-rank BENCH rows."""
    env = _child_env(data_dir, tokens, warmup, iters, timeout_s, launcher)
    proc = subprocess.Popen(
        [_interpreter(), str(MEGAMOE_KERNEL_PATH)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # mpirun and the ranks are grandchildren, so the timeout kill has to
        # reach the whole group.  Killing the entry point alone would leave
        # eight workers holding a GPU, a CUDA context and a symmetric heap,
        # and the next token count in the sweep would run against them.
        start_new_session=True,
    )
    try:
        # The entry point applies W_TIMEOUT to mpirun itself; leave it room to
        # report that timeout rather than being killed mid-report.
        stdout, stderr = proc.communicate(timeout=timeout_s + 120)
    except subprocess.TimeoutExpired:
        stdout, stderr = _terminate_process_group(proc)
        return [], _failure_output(
            f"timed out after {timeout_s + 120}s (process group killed)",
            stdout,
            stderr,
        )

    if proc.returncode != 0:
        return [], _failure_output(
            f"MegaMoE entry point exited with status {proc.returncode}",
            stdout,
            stderr,
        )

    rows = {}
    for line in stdout.splitlines():
        match = BENCH_LINE.search(line)
        if match is None:
            continue

        expected = {
            "ranks": NUM_RANKS,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "experts": NUM_EXPERTS,
            "topk": TOPK,
            "tokens": tokens,
            "experts_per_rank": NUM_EXPERTS // NUM_RANKS,
        }
        mismatches = {
            field: (int(match[field]), value)
            for field, value in expected.items()
            if int(match[field]) != value
        }
        if mismatches:
            return [], _failure_output(
                f"unexpected BENCH metadata (reported, expected): {mismatches}",
                stdout,
                stderr,
            )

        rank = int(match["rank"])
        if rank in rows:
            return [], _failure_output(
                f"duplicate BENCH row for rank {rank}", stdout, stderr
            )
        rows[rank] = {
            "rank": rank,
            "recv": int(match["recv"]),
            "latency_ms": float(match["us"]) / 1e3,
        }

    expected_ranks = set(range(NUM_RANKS))
    if set(rows) != expected_ranks:
        return [], _failure_output(
            f"expected BENCH ranks {sorted(expected_ranks)}, parsed {sorted(rows)}",
            stdout,
            stderr,
        )
    return [rows[rank] for rank in sorted(rows)], None


def _bench_level():
    level = getattr(Config, "bench_level", None)
    return level.value if level is not None else "core"


@pytest.mark.megamoe
def test_megamoe():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)

    launcher = _resolve_mpirun()
    data_dir = os.path.abspath(
        os.path.expanduser(os.environ["MEGAMOE_SHARED_DATA_DIR"].strip())
    )
    warmup = _iteration_count(
        "MEGAMOE_BENCH_WARMUP",
        Config.warm_up,
        DEFAULT_WARMUP_TIME,
        DEFAULT_WARMUP,
    )
    iters = _iteration_count(
        "MEGAMOE_BENCH_ITERS",
        Config.repetition,
        DEFAULT_ITER_TIME,
        DEFAULT_ITERS,
    )
    timeout_s = _env_int("MEGAMOE_BENCH_TIMEOUT", DEFAULT_TIMEOUT_S)

    metrics = []
    failures = []
    for tokens in _token_sweep():
        shape_detail = (tokens, HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOPK, NUM_RANKS)
        rows, error = _run_one(data_dir, tokens, warmup, iters, timeout_s, launcher)
        if error:
            failures.append(f"tokens/rank={tokens}: {error}")
            metric = BenchmarkMetrics(shape_detail=shape_detail, error_msg=error)
            metrics.append(metric)
            continue
        # A collective is as fast as its slowest rank, and every rank does its
        # own share of the work, so the recorded throughput is the whole step's
        # FLOPs over that rank's latency -- not a single rank's local figure.
        slowest = max(row["latency_ms"] for row in rows)
        step_flops = sum(6.0 * row["recv"] * HIDDEN * INTERMEDIATE for row in rows)
        metrics.append(
            BenchmarkMetrics(
                shape_detail=shape_detail,
                latency=slowest,
                tflops=step_flops / (slowest * 1e-3) / 1e12,
            )
        )

    result = BenchmarkResult(
        op_name=OP_NAME,
        dtype=str(torch.float8_e4m3fn),
        # CUDA events bracket the single persistent launch on every rank.
        mode="kernel",
        level=_bench_level(),
        result=metrics,
    )
    print(result)
    update_result(OP_NAME, asdict(result))
    emit_record_logger(result.to_json())

    if failures:
        pytest.fail("\n\n".join(failures))
