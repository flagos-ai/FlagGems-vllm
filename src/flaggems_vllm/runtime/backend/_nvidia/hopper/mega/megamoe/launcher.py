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

"""MPI launcher for the Hopper MegaMoE candidate."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

MEGAMOE_KERNEL_PATH = Path(__file__).with_name("kernel.py")

_BENCH_LINE = re.compile(
    r"\[rank (?P<rank>\d+)/(?P<ranks>\d+)\] BENCH "
    r"h=(?P<hidden>\d+) ih=(?P<intermediate>\d+) "
    r"E=(?P<num_experts>\d+) k=(?P<topk>\d+) "
    r"tokens=(?P<tokens>\d+).*?\|\s*"
    r"(?P<latency_us>[0-9.]+) us"
)


@dataclass(frozen=True)
class MegaMoEConfig:
    """Configuration shared by the launcher and every MPI worker."""

    worker: bool = False
    num_ranks: int = 2
    tokens: int = 128
    hidden_size: int = 256
    intermediate_size: int = 128
    num_experts: int = 16
    topk: int = 4
    stages: int = 2
    num_sms: int = 0
    max_recv: int = 512
    drop_rate: float = 0.1
    benchmark: bool = False
    warmup: int = 5
    iterations: int = 20
    reduce: str = "median"
    gpu_start_barrier: bool = False
    data_dir: Path | None = None
    verify_data_sha256: bool = False
    inject_fault: str | None = None
    mpirun: str | None = None
    timeout: int = 600
    cuda_home: Path | None = None
    nvshmem_home: Path | None = None


@dataclass(frozen=True)
class MegaMoEResult:
    """Output captured from one complete multi-rank MegaMoE run."""

    rank_latencies_ms: tuple[float, ...]
    stdout: str
    stderr: str
    command: tuple[str, ...]

    @property
    def slowest_rank_latency_ms(self) -> float:
        if not self.rank_latencies_ms:
            raise RuntimeError("MegaMoE did not produce benchmark latency")
        return max(self.rank_latencies_ms)


class MegaMoELaunchError(RuntimeError):
    """Failure raised while starting or monitoring the MPI workers."""

    def __init__(
        self,
        message: str,
        *,
        stdout: str = "",
        stderr: str = "",
        returncode: int | None = None,
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.timed_out = timed_out


def _config_parser() -> argparse.ArgumentParser:
    defaults = MegaMoEConfig()
    parser = argparse.ArgumentParser(
        description="Run the Hopper MegaMoE standalone launcher."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--num-ranks", type=int, default=defaults.num_ranks)
    parser.add_argument("--tokens", type=int, default=defaults.tokens)
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument(
        "--intermediate-size", type=int, default=defaults.intermediate_size
    )
    parser.add_argument("--num-experts", type=int, default=defaults.num_experts)
    parser.add_argument("--topk", type=int, default=defaults.topk)
    parser.add_argument("--stages", type=int, default=defaults.stages)
    parser.add_argument("--num-sms", type=int, default=defaults.num_sms)
    parser.add_argument("--max-recv", type=int, default=defaults.max_recv)
    parser.add_argument("--drop-rate", type=float)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=defaults.warmup)
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--reduce", choices=("mean", "median"), default=defaults.reduce)
    parser.add_argument("--gpu-start-barrier", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--verify-data-sha256", action="store_true")
    parser.add_argument(
        "--inject-fault",
        choices=("queue", "recv", "meta", "scatter", "combine"),
    )
    parser.add_argument("--mpirun")
    parser.add_argument("--timeout", type=int, default=defaults.timeout)
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--nvshmem-home", type=Path)
    return parser


def validate_megamoe_config(config: MegaMoEConfig) -> MegaMoEConfig:
    if config.num_ranks <= 0:
        raise ValueError("--num-ranks must be positive")
    if config.tokens <= 0:
        raise ValueError("--tokens must be positive")
    if config.hidden_size <= 0 or config.hidden_size % 128:
        raise ValueError("--hidden-size must be a positive multiple of 128")
    if config.intermediate_size <= 0 or config.intermediate_size % 64:
        raise ValueError("--intermediate-size must be a positive multiple of 64")
    if config.num_experts <= 0 or config.num_experts % config.num_ranks:
        raise ValueError("--num-experts must be positive and divisible by --num-ranks")
    if not 0 < config.topk <= config.num_experts:
        raise ValueError("--topk must be in [1, num_experts]")
    if config.stages <= 0:
        raise ValueError("--stages must be positive")
    if config.num_sms < 0:
        raise ValueError("--num-sms cannot be negative")
    if config.max_recv <= 0:
        raise ValueError("--max-recv must be positive")
    if not 0.0 <= config.drop_rate < 1.0:
        raise ValueError("--drop-rate must be in [0, 1)")
    if config.warmup < 0 or config.iterations <= 0:
        raise ValueError(
            "--warmup cannot be negative and --iterations must be positive"
        )
    if config.reduce not in ("mean", "median"):
        raise ValueError("--reduce must be 'mean' or 'median'")
    if config.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if config.data_dir is not None and config.drop_rate != 0.0:
        raise ValueError("--data-dir requires --drop-rate=0")
    return config


def parse_megamoe_config(argv: list[str] | None = None) -> MegaMoEConfig:
    args = _config_parser().parse_args(argv)
    if args.drop_rate is None:
        args.drop_rate = 0.0 if args.data_dir is not None else 0.1
    return validate_megamoe_config(MegaMoEConfig(**vars(args)))


def resolve_mpirun(explicit: str | None = None) -> str | None:
    """Return an executable MPI launcher without modifying the environment."""

    return shutil.which(explicit or "mpirun")


def _worker_cli_args(config: MegaMoEConfig) -> list[str]:
    args = [
        "--worker",
        "--num-ranks",
        str(config.num_ranks),
        "--tokens",
        str(config.tokens),
        "--hidden-size",
        str(config.hidden_size),
        "--intermediate-size",
        str(config.intermediate_size),
        "--num-experts",
        str(config.num_experts),
        "--topk",
        str(config.topk),
        "--stages",
        str(config.stages),
        "--num-sms",
        str(config.num_sms),
        "--max-recv",
        str(config.max_recv),
        "--drop-rate",
        str(config.drop_rate),
        "--warmup",
        str(config.warmup),
        "--iterations",
        str(config.iterations),
        "--reduce",
        config.reduce,
    ]
    if config.benchmark:
        args.append("--benchmark")
    if config.gpu_start_barrier:
        args.append("--gpu-start-barrier")
    if config.data_dir is not None:
        args.extend(("--data-dir", str(config.data_dir)))
    if config.verify_data_sha256:
        args.append("--verify-data-sha256")
    if config.inject_fault is not None:
        args.extend(("--inject-fault", config.inject_fault))
    if config.cuda_home is not None:
        args.extend(("--cuda-home", str(config.cuda_home)))
    if config.nvshmem_home is not None:
        args.extend(("--nvshmem-home", str(config.nvshmem_home)))
    return args


def _resolve_runtime_paths(config: MegaMoEConfig) -> MegaMoEConfig:
    if __package__:
        from .runtime import _find_cuda_home, _find_nvshmem_home
    else:
        from runtime import _find_cuda_home, _find_nvshmem_home

    return replace(
        config,
        cuda_home=_find_cuda_home(config.cuda_home),
        nvshmem_home=_find_nvshmem_home(config.nvshmem_home),
    )


def _parse_benchmark_output(stdout: str, config: MegaMoEConfig) -> tuple[float, ...]:
    expected_shape = {
        "hidden": config.hidden_size,
        "intermediate": config.intermediate_size,
        "num_experts": config.num_experts,
        "topk": config.topk,
        "tokens": config.tokens,
    }
    latencies: dict[int, float] = {}
    for match in _BENCH_LINE.finditer(stdout):
        rank = int(match["rank"])
        ranks = int(match["ranks"])
        if ranks != config.num_ranks:
            raise RuntimeError(
                f"MegaMoE rank {rank} reported world size {ranks}, "
                f"expected {config.num_ranks}"
            )
        for field, expected in expected_shape.items():
            actual = int(match[field])
            if actual != expected:
                raise RuntimeError(
                    f"MegaMoE rank {rank} reported {field}={actual}, "
                    f"expected {expected}"
                )
        if rank in latencies:
            raise RuntimeError(f"MegaMoE rank {rank} reported benchmark latency twice")
        latencies[rank] = float(match["latency_us"]) / 1e3

    expected_ranks = set(range(config.num_ranks))
    if set(latencies) != expected_ranks:
        raise RuntimeError(
            "MegaMoE did not report latency for every rank: "
            f"expected {sorted(expected_ranks)}, got {sorted(latencies)}"
        )
    return tuple(latencies[rank] for rank in range(config.num_ranks))


def _terminate_process_group(
    proc: subprocess.Popen[str],
) -> tuple[str, str]:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        stdout, stderr = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return proc.communicate()
    # ``mpirun`` may exit before a rank in its process group.  Always issue a
    # final best-effort kill so a detached rank cannot retain GPU/NVSHMEM state.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return stdout, stderr


def _diagnostic_tail(stdout: str, stderr: str, limit: int = 1000) -> str:
    sections = []
    if stdout.strip():
        sections.append(f"stdout: {stdout.strip()[-limit:]}")
    if stderr.strip():
        sections.append(f"stderr: {stderr.strip()[-limit:]}")
    return "\n".join(sections)


def launch_megamoe(
    config: MegaMoEConfig,
    *,
    kernel_path: Path = MEGAMOE_KERNEL_PATH,
) -> MegaMoEResult:
    """Launch all ranks and return their structured benchmark output."""

    config = validate_megamoe_config(config)
    if config.worker:
        raise ValueError("launch_megamoe requires worker=False")
    config = _resolve_runtime_paths(config)

    mpirun = resolve_mpirun(config.mpirun)
    if mpirun is None:
        requested = config.mpirun or "mpirun on PATH"
        raise MegaMoELaunchError(f"MegaMoE MPI launcher not found: {requested}")

    command = (
        mpirun,
        "--allow-run-as-root",
        "-np",
        str(config.num_ranks),
        sys.executable,
        str(Path(kernel_path).resolve()),
        *_worker_cli_args(config),
    )
    env = os.environ.copy()
    env.pop("TRITON_CACHE_DIR", None)
    env.update(
        {
            "NVSHMEM_BOOTSTRAP": "MPI",
            "CUDA_HOME": str(config.cuda_home),
        }
    )

    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=env,
        )
    except OSError as error:
        raise MegaMoELaunchError(f"Unable to start MegaMoE: {error}") from error

    try:
        stdout, stderr = proc.communicate(timeout=config.timeout)
    except subprocess.TimeoutExpired:
        stdout, stderr = _terminate_process_group(proc)
        detail = _diagnostic_tail(stdout, stderr)
        suffix = f"\n{detail}" if detail else ""
        raise MegaMoELaunchError(
            f"MegaMoE timed out after {config.timeout}s{suffix}",
            stdout=stdout,
            stderr=stderr,
            returncode=proc.returncode,
            timed_out=True,
        ) from None
    except BaseException:
        _terminate_process_group(proc)
        raise

    if proc.returncode != 0:
        _terminate_process_group(proc)
        detail = _diagnostic_tail(stdout, stderr)
        suffix = f"\n{detail}" if detail else ""
        raise MegaMoELaunchError(
            f"MegaMoE exited with status {proc.returncode}{suffix}",
            stdout=stdout,
            stderr=stderr,
            returncode=proc.returncode,
        )

    try:
        rank_latencies = (
            _parse_benchmark_output(stdout, config) if config.benchmark else ()
        )
    except RuntimeError as error:
        raise MegaMoELaunchError(
            str(error),
            stdout=stdout,
            stderr=stderr,
            returncode=proc.returncode,
        ) from error
    return MegaMoEResult(
        rank_latencies_ms=rank_latencies,
        stdout=stdout,
        stderr=stderr,
        command=command,
    )


def launch_megamoe_cli(config: MegaMoEConfig) -> int:
    """Run the launcher while preserving ``kernel.py``'s CLI behavior."""

    try:
        result = launch_megamoe(config)
    except MegaMoELaunchError as error:
        if error.timed_out:
            print("TIMEOUT\nstdout:", error.stdout[-2500:])
            print("stderr:", error.stderr[-2500:])
            return 1
        print(error.stdout, end="")
        if error.stderr:
            print("STDERR:", error.stderr[-4000:])
        else:
            print(str(error), file=sys.stderr)
        return error.returncode or 1

    print(result.stdout, end="")
    return 0


__all__ = [
    "MEGAMOE_KERNEL_PATH",
    "MegaMoEConfig",
    "MegaMoELaunchError",
    "MegaMoEResult",
    "launch_megamoe",
    "launch_megamoe_cli",
    "parse_megamoe_config",
    "resolve_mpirun",
    "validate_megamoe_config",
]
