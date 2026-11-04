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

"""Standalone Hygon MegaMoE correctness and event-timing harness.

Run this file with ``torchrun``. The harness owns the process group, DUSHMEM
bootstrap/finalization, symmetric allocation, peer tables, scratch tensors,
and one-time weight prepacking. Only ``megamoe_int8`` is inside the event
timing interval.

Example::

    DUSHMEM_REMOTE_TRANSPORT=none PYTHONPATH=src \
      timeout --signal=TERM --kill-after=10s 600s \
      python -m torch.distributed.run --nnodes=1 --nproc-per-node=8 \
      --master-addr=127.0.0.1 --master-port=29500 \
      benchmark/test_megamoe.py \
      --world-size 8 --tokens 512 \
      --data-dir /workspace/data/qwen36_int8/canonical \
      --warmup 20 --iterations 100 --output /tmp/megamoe-np8-t512.json

The canonical Qwen3.6 bundle is the same ``INT8 A1 + FP32 scale + TopK``
contract used by the migrated kernel. Correctness uses a deliberately small
CPU integer reference before the performance loop; it is never timed.

Add ``--compare-vllm-eager --repeats 5`` to measure the operator and the
included vLLM 0.18.1 native-EP eager baseline in the same ``torchrun``. That
paired mode excludes qA1/router from both paths, keeps dynamic qA2 inside both
timed regions, alternates trial order, and reports ``baseline / candidate``.
Graph comparison is intentionally unsupported because the caller-owned epoch
is advanced on the host for every MegaMoE collective launch.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

if "LOCAL_RANK" in os.environ:
    _cache_root = os.environ.get(
        "FLAGGEMS_MEGAMOE_CACHE_ROOT",
        "/tmp/flaggems-vllm-megamoe-cache",
    )
    os.environ.setdefault(
        "TRITON_CACHE_DIR",
        os.path.join(_cache_root, f"rank-{os.environ['LOCAL_RANK']}"),
    )

import pytest
import torch
import torch.distributed as dist

TIMING_PROTOCOL = "hygon-single-launch-max-rank-event-v1"
DEFAULT_DATA_DIR = Path("/workspace/data/qwen36_int8/canonical")
HOST_SOURCE = Path(__file__).with_name("megamoe_hygon_host.hip")
DEFAULT_HOST_LIBRARY = Path("/tmp/flaggems-vllm-megamoe-hygon-host.so")
SUPPORTED_WORLD_SIZES = (2, 4, 8)
MODEL_HIDDEN = 2048
MODEL_INTERMEDIATE = 512
MODEL_EXPERTS = 256
MODEL_TOP_K = 8
_ALIGNMENT = 256
_NUM_LOCAL_PHASES = 9
_NUM_RANK_PHASES = 2


def _align(value: int, alignment: int = _ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path):
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _compile_host_library(output: Path, *, force: bool = False) -> Path:
    if (
        not force
        and output.is_file()
        and output.stat().st_mtime_ns >= HOST_SOURCE.stat().st_mtime_ns
    ):
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "/opt/dtk/bin/hipcc",
        "-shared",
        "-fPIC",
        "-fgpu-rdc",
        "--offload-arch=gfx936",
        "-O3",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
        "-mcode-object-version=4",
        "-I/opt/dtk/include",
        "-I/opt/dtk/include/dushmem",
        "-L/opt/dtk/lib/dushmem",
        "-Wl,-rpath,/opt/dtk/lib/dushmem",
        str(HOST_SOURCE),
        "-o",
        str(output),
        "-ldushmem_host",
        "-ldushmem_device",
    ]
    build = subprocess.run(command, capture_output=True, text=True)
    if build.returncode:
        raise RuntimeError(
            "failed to build the benchmark DUSHMEM shim:\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{build.stdout}\nstderr:\n{build.stderr}"
        )
    return output


class _DushmemRuntime:
    """Benchmark-owned DUSHMEM process state and one symmetric allocation."""

    def __init__(
        self,
        library: ctypes.CDLL,
        rank: int,
        world_size: int,
        local_rank: int,
        group=None,
    ):
        self.library = library
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.group = group
        self.allocation = ctypes.c_void_p()
        self.initialized = False
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        lib = self.library
        lib.flaggems_megamoe_unique_id_nbytes.argtypes = []
        lib.flaggems_megamoe_unique_id_nbytes.restype = ctypes.c_size_t
        lib.flaggems_megamoe_get_unique_id.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.flaggems_megamoe_get_unique_id.restype = ctypes.c_int
        lib.flaggems_megamoe_init.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.flaggems_megamoe_init.restype = ctypes.c_int
        lib.flaggems_megamoe_rank_info.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.flaggems_megamoe_rank_info.restype = ctypes.c_int
        lib.flaggems_megamoe_is_single_node.argtypes = []
        lib.flaggems_megamoe_is_single_node.restype = ctypes.c_int
        lib.flaggems_megamoe_symmetric_alloc.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.flaggems_megamoe_symmetric_alloc.restype = ctypes.c_int
        lib.flaggems_megamoe_peer_pointer.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.flaggems_megamoe_peer_pointer.restype = ctypes.c_void_p
        lib.flaggems_megamoe_symmetric_free.argtypes = [ctypes.c_void_p]
        lib.flaggems_megamoe_symmetric_free.restype = ctypes.c_int
        lib.flaggems_megamoe_finalize.argtypes = []
        lib.flaggems_megamoe_finalize.restype = ctypes.c_int

    def initialize(self) -> None:
        if os.environ.get("DUSHMEM_REMOTE_TRANSPORT") != "none":
            raise RuntimeError(
                "single-node benchmark requires DUSHMEM_REMOTE_TRANSPORT=none"
            )
        nbytes = int(self.library.flaggems_megamoe_unique_id_nbytes())
        if not 0 < nbytes <= 4096:
            raise RuntimeError(f"unexpected DUSHMEM unique-id size: {nbytes}")
        payload: bytes | None = None
        if self.rank == 0:
            buffer = ctypes.create_string_buffer(nbytes)
            status = self.library.flaggems_megamoe_get_unique_id(buffer, nbytes)
            if status:
                raise RuntimeError(f"dushmemx_get_uniqueid failed: {status}")
            payload = bytes(buffer.raw)
        objects = [payload]
        dist.broadcast_object_list(objects, src=0, group=self.group)
        payload = objects[0]
        if not isinstance(payload, bytes) or len(payload) != nbytes:
            raise RuntimeError("broadcast DUSHMEM unique ID is invalid")
        buffer = ctypes.create_string_buffer(payload, nbytes)
        status = self.library.flaggems_megamoe_init(
            self.rank,
            self.world_size,
            self.local_rank,
            buffer,
            nbytes,
        )
        if status:
            raise RuntimeError(f"DUSHMEM initialization failed: {status}")

        rank = ctypes.c_int()
        world = ctypes.c_int()
        local = ctypes.c_int()
        status = self.library.flaggems_megamoe_rank_info(
            ctypes.byref(rank), ctypes.byref(world), ctypes.byref(local)
        )
        observed = (rank.value, world.value, local.value)
        expected = (self.rank, self.world_size, self.local_rank)
        if status or observed != expected:
            raise RuntimeError(
                f"Torch/DUSHMEM identity mismatch: expected={expected}, observed={observed}"
            )
        if self.library.flaggems_megamoe_is_single_node() != 1:
            raise RuntimeError("all MegaMoE ranks must be in one DUSHMEM node team")
        self.initialized = True

    def allocate(self, nbytes: int) -> int:
        status = self.library.flaggems_megamoe_symmetric_alloc(
            nbytes, ctypes.byref(self.allocation)
        )
        if status or not self.allocation.value:
            raise RuntimeError(f"DUSHMEM symmetric allocation failed: {status}")
        return int(self.allocation.value)

    def peer_bases(self) -> list[int]:
        bases = []
        for peer in range(self.world_size):
            pointer = self.library.flaggems_megamoe_peer_pointer(self.allocation, peer)
            if not pointer:
                raise RuntimeError(f"rank {peer} is not directly peer-addressable")
            bases.append(int(pointer))
        return bases

    def close(self) -> None:
        if self.allocation.value:
            status = self.library.flaggems_megamoe_symmetric_free(self.allocation)
            if status:
                raise RuntimeError(f"DUSHMEM symmetric free failed: {status}")
            self.allocation = ctypes.c_void_p()
        if self.initialized:
            status = self.library.flaggems_megamoe_finalize()
            if status:
                raise RuntimeError(f"DUSHMEM finalize failed: {status}")
            self.initialized = False


def _tensor_from_pointer(pointer: int, shape: tuple[int, ...], dtype, device):
    elements = math.prod(shape)
    storage = torch._C._construct_storage_from_data_pointer(
        pointer, device, elements * dtype.itemsize
    )
    return torch.empty(0, dtype=dtype, device=device).set_(storage).view(shape)


def _load_canonical(
    data_dir: Path, rank: int, world_size: int, tokens: int, verify: bool
):
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shape = manifest["shape"]
    expected = {
        "hidden_size": MODEL_HIDDEN,
        "moe_intermediate_size": MODEL_INTERMEDIATE,
        "num_experts": MODEL_EXPERTS,
        "num_experts_per_tok": MODEL_TOP_K,
    }
    mismatches = {
        key: (shape.get(key), value)
        for key, value in expected.items()
        if int(shape.get(key, -1)) != value
    }
    if mismatches:
        raise ValueError(f"canonical model shape mismatch: {mismatches}")
    capacity = int(shape["tokens_per_rank_capacity"])
    if not 0 < tokens <= capacity:
        raise ValueError(f"tokens must be in [1,{capacity}], got {tokens}")

    files = manifest["files"]
    shared_path = data_dir / files["shared"]["name"]
    experts_path = data_dir / files["experts"]["name"]
    if verify:
        for key, path in (("shared", shared_path), ("experts", experts_path)):
            expected_hash = files[key].get("sha256")
            if expected_hash and _sha256(path) != expected_hash:
                raise ValueError(f"SHA256 mismatch for {path}")
    shared = _torch_load(shared_path)
    experts = _torch_load(experts_path)
    experts_per_rank = MODEL_EXPERTS // world_size
    begin = rank * experts_per_rank
    end = begin + experts_per_rank
    local = {name: tensor[begin:end] for name, tensor in experts.items()}
    return manifest, shared, local


def _make_weights(local: dict[str, torch.Tensor], device):
    from flaggems_vllm.runtime.backend._hygon.mega.megamoe.api import MegaMoEWeights

    w13 = local["l1_weight_int8"]
    w13_scale = local["l1_weight_scale"].reshape(w13.shape[0], 2 * MODEL_INTERMEDIATE)
    return MegaMoEWeights(
        w1_gate_t=w13[:, :MODEL_INTERMEDIATE].transpose(1, 2).contiguous().to(device),
        w1_up_t=w13[:, MODEL_INTERMEDIATE:].transpose(1, 2).contiguous().to(device),
        w2_t=local["l2_weight_int8"].transpose(1, 2).contiguous().to(device),
        w1_gate_scale=w13_scale[:, :MODEL_INTERMEDIATE].contiguous().to(device),
        w1_up_scale=w13_scale[:, MODEL_INTERMEDIATE:].contiguous().to(device),
        w2_scale=local["l2_weight_scale"]
        .reshape(w13.shape[0], MODEL_HIDDEN)
        .contiguous()
        .to(device),
    )


def _symmetric_workspace(runtime: _DushmemRuntime, tokens: int, device):
    definitions = (
        ("q_local", (tokens, MODEL_HIDDEN), torch.int8, "q_table"),
        ("qscale_local", (tokens,), torch.float32, "qscale_table"),
        ("ids_local", (tokens, MODEL_TOP_K), torch.int32, "ids_table"),
        (
            "route_weights_local",
            (tokens, MODEL_TOP_K),
            torch.float32,
            "route_weights_table",
        ),
        (
            "partials_local",
            (tokens, MODEL_TOP_K, MODEL_HIDDEN),
            torch.bfloat16,
            "partials_table",
        ),
        (
            "flags_local",
            (_NUM_RANK_PHASES, runtime.world_size),
            torch.int32,
            "flag_table",
        ),
    )
    offsets = {}
    cursor = 0
    for local_name, shape, dtype, table_name in definitions:
        cursor = _align(cursor)
        offsets[local_name] = (cursor, shape, dtype, table_name)
        cursor += math.prod(shape) * dtype.itemsize
    base = runtime.allocate(_align(cursor))
    peer_bases = runtime.peer_bases()
    values = {}
    for local_name, (offset, shape, dtype, table_name) in offsets.items():
        values[local_name] = _tensor_from_pointer(base + offset, shape, dtype, device)
        values[table_name] = torch.tensor(
            [peer_base + offset for peer_base in peer_bases],
            dtype=torch.int64,
            device=device,
        )
    return values


def _make_context(runtime: _DushmemRuntime, tokens: int, device, ncta: int, spins: int):
    from flaggems_vllm.runtime.backend._hygon.mega.megamoe.api import (
        MegaMoEContext,
        default_tile_config,
    )

    tile = default_tile_config(tokens)
    symmetric = _symmetric_workspace(runtime, tokens, device)
    experts_per_rank = MODEL_EXPERTS // runtime.world_size
    total_tokens = runtime.world_size * tokens
    total_routes = total_tokens * MODEL_TOP_K
    ni = MODEL_INTERMEDIATE // tile.block_n
    empty = lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device)
    zeros = lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=device)
    properties = torch.cuda.get_device_properties(device)
    architecture = getattr(properties, "gcnArchName", "gfx936")
    return MegaMoEContext(
        rank=runtime.rank,
        world_size=runtime.world_size,
        tokens_per_rank=tokens,
        hidden_size=MODEL_HIDDEN,
        intermediate_size=MODEL_INTERMEDIATE,
        num_experts=MODEL_EXPERTS,
        top_k=MODEL_TOP_K,
        architecture=architecture,
        ncta=ncta,
        sync_max_spins=spins,
        tile=tile,
        **symmetric,
        expert_counts=zeros((experts_per_rank,), torch.int32),
        expert_offsets=zeros((experts_per_rank + 1,), torch.int32),
        expert_write=zeros((experts_per_rank,), torch.int32),
        expert_block_offsets=zeros((experts_per_rank + 1,), torch.int32),
        route_queue=empty((total_routes,), torch.int32),
        gathered_q=empty((total_tokens, MODEL_HIDDEN), torch.int8),
        block_to_expert=empty((total_routes,), torch.int32),
        gathered_qscale=empty((total_tokens,), torch.float32),
        queue_route_weight=empty((total_routes,), torch.float32),
        activation_tmp=empty((total_routes, MODEL_INTERMEDIATE), torch.bfloat16),
        activation_amax=empty((total_routes, ni), torch.float32),
        q2=empty((total_routes, MODEL_INTERMEDIATE), torch.int8),
        q2_scale=empty((total_routes,), torch.float32),
        owner_accum=empty((1,), torch.float32),
        phase_done=zeros((_NUM_LOCAL_PHASES, ncta), torch.int32),
        phase_released=zeros((_NUM_LOCAL_PHASES,), torch.int32),
        rank_released=zeros((_NUM_RANK_PHASES,), torch.int32),
        sync_status=zeros((1,), torch.int32),
        output=empty((tokens, MODEL_HIDDEN), torch.bfloat16),
    )


@torch.no_grad()
def _cpu_reference(
    shared: dict[str, torch.Tensor],
    local: dict[str, torch.Tensor],
    rank: int,
    world_size: int,
    tokens: int,
):
    """Compute a small exact integer reference and sum owner contributions."""

    experts_per_rank = MODEL_EXPERTS // world_size
    expert_begin = rank * experts_per_rank
    expert_end = expert_begin + experts_per_rank
    result = torch.zeros(
        (world_size, tokens, MODEL_HIDDEN), dtype=torch.float32, device="cpu"
    )
    w13 = local["l1_weight_int8"]
    s13 = local["l1_weight_scale"].reshape(experts_per_rank, 2 * MODEL_INTERMEDIATE)
    w2 = local["l2_weight_int8"]
    s2 = local["l2_weight_scale"].reshape(experts_per_rank, MODEL_HIDDEN)

    for source_rank in range(world_size):
        for token in range(tokens):
            q1 = shared["input_int8"][source_rank, token].to(torch.int32)
            q1_scale = shared["input_scales"][source_rank, token].float()
            for slot in range(MODEL_TOP_K):
                expert = int(shared["topk_idx"][source_rank, token, slot])
                if not expert_begin <= expert < expert_end:
                    continue
                local_expert = expert - expert_begin
                gu = torch.mv(w13[local_expert].to(torch.int32), q1).float()
                gu = (gu * q1_scale * s13[local_expert]).bfloat16()
                gate = gu[:MODEL_INTERMEDIATE].float()
                up = gu[MODEL_INTERMEDIATE:].float()
                activation = (
                    torch.nn.functional.silu(gate).bfloat16().float() * up
                ).bfloat16()
                absolute_max = activation.float().abs().max()
                scale = absolute_max / 127.0 if absolute_max > 0 else torch.tensor(1.0)
                q2 = (
                    torch.round(activation.float() / scale)
                    .clamp(-127, 127)
                    .to(torch.int32)
                )
                partial = torch.mv(w2[local_expert].to(torch.int32), q2).float()
                route_weight = shared["topk_weights"][source_rank, token, slot]
                partial = (partial * scale * s2[local_expert] * route_weight).bfloat16()
                result[source_rank, token].add_(partial.float())
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result[rank].bfloat16()


def _correctness(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    delta = actual - expected
    expected_norm = float(torch.linalg.vector_norm(expected))
    actual_norm = float(torch.linalg.vector_norm(actual))
    relative_l2 = float(torch.linalg.vector_norm(delta)) / max(expected_norm, 1.0e-12)
    cosine = float(torch.sum(actual * expected)) / max(
        actual_norm * expected_norm, 1.0e-12
    )
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "relative_l2": relative_l2,
        "cosine_similarity": cosine,
        "max_absolute_error": float(delta.abs().max()),
        "passed": bool(torch.isfinite(actual).all())
        and relative_l2 <= 0.005
        and cosine >= 0.999,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world-size", type=int, choices=SUPPORTED_WORLD_SIZES, required=True
    )
    parser.add_argument("--tokens", type=int, required=True, help="tokens per rank")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="paired vLLM comparison repeats",
    )
    parser.add_argument(
        "--compare-vllm-eager",
        action="store_true",
        help=(
            "measure this PR and the prequantized vLLM native-EP eager "
            "baseline in one torchrun"
        ),
    )
    parser.add_argument("--correctness-tokens", type=int, default=1)
    parser.add_argument("--ncta", type=int, choices=range(1, 81), default=80)
    parser.add_argument("--sync-max-spins", type=int, default=100_000_000)
    parser.add_argument("--host-library", type=Path, default=DEFAULT_HOST_LIBRARY)
    parser.add_argument("--rebuild-host-library", action="store_true")
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    if args.warmup < 1 or args.iterations < 1:
        parser.error("--warmup and --iterations must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not 0 < args.correctness_tokens <= args.tokens:
        parser.error("--correctness-tokens must be in [1,--tokens]")
    if args.sync_max_spins <= 0:
        parser.error("--sync-max-spins must be positive")
    return args


def _run_candidate(args) -> None:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            f"run with torchrun; missing environment variables: {missing}"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.world_size:
        raise ValueError(
            f"torchrun WORLD_SIZE={world_size} does not match --world-size={args.world_size}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    runtime = None
    try:
        if rank == 0:
            _compile_host_library(args.host_library, force=args.rebuild_host_library)
        dist.barrier()
        runtime = _DushmemRuntime(
            ctypes.CDLL(str(args.host_library)), rank, world_size, local_rank
        )
        runtime.initialize()

        manifest, shared_cpu, local_cpu = _load_canonical(
            args.data_dir,
            rank,
            world_size,
            args.tokens,
            args.verify_sha256,
        )
        weights = _make_weights(local_cpu, device)
        context = _make_context(
            runtime, args.tokens, device, args.ncta, args.sync_max_spins
        )
        input_int8 = (
            shared_cpu["input_int8"][rank, : args.tokens].contiguous().to(device)
        )
        input_scales = (
            shared_cpu["input_scales"][rank, : args.tokens].contiguous().to(device)
        )
        topk_ids = shared_cpu["topk_idx"][rank, : args.tokens].contiguous().to(device)
        topk_weights = (
            shared_cpu["topk_weights"][rank, : args.tokens].contiguous().to(device)
        )

        from flaggems_vllm.runtime.backend._hygon.mega.megamoe.api import megamoe_int8

        expected = _cpu_reference(
            shared_cpu,
            local_cpu,
            rank,
            world_size,
            args.correctness_tokens,
        )
        dist.barrier()
        output = megamoe_int8(
            input_int8,
            input_scales,
            topk_ids,
            topk_weights,
            weights,
            context,
        )
        context.raise_if_sync_failed()
        correctness = _correctness(
            output[: args.correctness_tokens], expected.to(device)
        )
        gathered_correctness = [None] * world_size
        dist.all_gather_object(gathered_correctness, correctness)
        if not all(item["passed"] for item in gathered_correctness):
            raise AssertionError(f"MegaMoE correctness failed: {gathered_correctness}")

        for _ in range(args.warmup):
            megamoe_int8(
                input_int8,
                input_scales,
                topk_ids,
                topk_weights,
                weights,
                context,
            )
        context.raise_if_sync_failed()

        latencies_us = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(args.iterations):
            dist.barrier()
            start.record()
            megamoe_int8(
                input_int8,
                input_scales,
                topk_ids,
                topk_weights,
                weights,
                context,
            )
            end.record()
            end.synchronize()
            rank_latency = torch.tensor(
                [start.elapsed_time(end) * 1.0e3], dtype=torch.float64
            )
            dist.all_reduce(rank_latency, op=dist.ReduceOp.MAX)
            latencies_us.append(float(rank_latency.item()))
        context.raise_if_sync_failed()

        payload = {
            "operator": "megamoe_int8",
            "backend": "hygon",
            "timing_protocol": TIMING_PROTOCOL,
            "timing_scope": "one megamoe_int8 launch; rank barriers outside events",
            "world_size": world_size,
            "tokens_per_rank": args.tokens,
            "model": {
                "hidden_size": MODEL_HIDDEN,
                "intermediate_size": MODEL_INTERMEDIATE,
                "num_experts": MODEL_EXPERTS,
                "top_k": MODEL_TOP_K,
            },
            "tile": vars(context.tile),
            "ncta": context.ncta,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "latency_us": {
                "mean": statistics.fmean(latencies_us),
                "median": statistics.median(latencies_us),
                "minimum": min(latencies_us),
                "maximum": max(latencies_us),
                "samples": latencies_us,
            },
            "correctness": {
                "tokens_per_rank": args.correctness_tokens,
                "per_rank": gathered_correctness,
                "all_ranks_passed": True,
            },
            "canonical_data": {
                "root": str(args.data_dir.resolve()),
                "format": manifest.get("format"),
                "format_version": manifest.get("format_version"),
                "sha256_verified": args.verify_sha256,
            },
        }
        if rank == 0:
            rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                temporary.write_text(rendered, encoding="utf-8")
                temporary.replace(args.output)
            print(rendered, end="")
    finally:
        if runtime is not None:
            torch.cuda.synchronize(device)
            dist.barrier()
            runtime.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _gather_objects(value, *, group, world_size: int):
    gathered = [None] * world_size
    dist.all_gather_object(gathered, value, group=group)
    return gathered


def _measure_paired_repeat(
    function: Callable[[], torch.Tensor],
    *,
    device,
    group,
    world_size: int,
    warmup: int,
    iterations: int,
    repeat_index: int,
    order_index: int,
):
    output = None
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    local_samples_us = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        elapsed_us = float(start.elapsed_time(end)) * 1.0e3
        if not math.isfinite(elapsed_us) or elapsed_us <= 0.0:
            raise RuntimeError(f"invalid HCU event latency: {elapsed_us!r} us")
        local_samples_us.append(elapsed_us)
        dist.barrier(group=group)

    if output is None:
        raise AssertionError("paired measurement produced no output")
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    rank_samples = _gather_objects(
        local_samples_us,
        group=group,
        world_size=world_size,
    )
    rank_means = [statistics.fmean(samples) for samples in rank_samples]
    return output, {
        "repeat_index": repeat_index,
        "measurement_order_index": order_index,
        "rank_samples_us": rank_samples,
        "rank_mean_us": rank_means,
        "repeat_latency_us": max(rank_means),
        "rank_latency_aggregation": "maximum rank arithmetic mean",
    }


def _summarize_paired(reports, *, warmup: int, iterations: int):
    latencies = [float(report["repeat_latency_us"]) for report in reports]
    primary = statistics.fmean(latencies)
    return {
        "warmup_per_repeat": warmup,
        "iterations_per_repeat": iterations,
        "repeats": len(reports),
        "repeat_latency_us": latencies,
        "primary_latency_us": primary,
        "primary_latency_ms": primary / 1.0e3,
        "timer": "HCU events around complete eager forward",
        "rank_latency_aggregation": "maximum rank arithmetic mean",
        "repeat_details": reports,
    }


def _run_paired(args) -> None:
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    try:
        from benchmark.megamoe_hygon_vllm_baseline import (
            HcuNativeEpInt8Baseline,
            NativeEpContract,
        )
    except ModuleNotFoundError as error:
        if error.name != "benchmark":
            raise
        from megamoe_hygon_vllm_baseline import (  # type: ignore[no-redef]
            HcuNativeEpInt8Baseline,
            NativeEpContract,
        )
    from flaggems_vllm.runtime.backend._hygon.mega.megamoe.api import megamoe_int8

    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            f"run paired comparison with torchrun; missing variables: {missing}"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.world_size:
        raise ValueError(
            f"torchrun WORLD_SIZE={world_size} does not match "
            f"--world-size={args.world_size}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Hygon CUDA-compatible device runtime is unavailable")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_grad_enabled(False)
    parallel = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=world_size,
        data_parallel_size_local=world_size,
        data_parallel_rank=rank,
        data_parallel_rank_local=local_rank,
        enable_expert_parallel=True,
        is_moe_model=True,
        all2all_backend="allgather_reducescatter",
        distributed_executor_backend="external_launcher",
    )
    vllm_config = VllmConfig(parallel_config=parallel)

    distributed_initialized = False
    control_group = None
    runtime = None
    with set_current_vllm_config(vllm_config):
        try:
            init_workspace_manager(device, num_ubatches=1)
            init_distributed_environment(
                world_size,
                rank,
                "env://",
                local_rank,
                "nccl",
            )
            distributed_initialized = True
            initialize_model_parallel(1, 1, 1, 1, backend="nccl")
            control_group = dist.new_group(
                ranks=list(range(world_size)),
                backend="gloo",
            )

            if rank == 0:
                _compile_host_library(
                    args.host_library,
                    force=args.rebuild_host_library,
                )
            dist.barrier(group=control_group)
            runtime = _DushmemRuntime(
                ctypes.CDLL(str(args.host_library)),
                rank,
                world_size,
                local_rank,
                group=control_group,
            )
            runtime.initialize()

            manifest, shared_cpu, local_cpu = _load_canonical(
                args.data_dir,
                rank,
                world_size,
                args.tokens,
                args.verify_sha256,
            )
            candidate_weights = _make_weights(local_cpu, device)
            candidate_context = _make_context(
                runtime,
                args.tokens,
                device,
                args.ncta,
                args.sync_max_spins,
            )
            input_int8 = (
                shared_cpu["input_int8"][rank, : args.tokens].contiguous().to(device)
            )
            input_scales = (
                shared_cpu["input_scales"][rank, : args.tokens].contiguous().to(device)
            )
            topk_ids = (
                shared_cpu["topk_idx"][rank, : args.tokens].contiguous().to(device)
            )
            topk_weights = (
                shared_cpu["topk_weights"][rank, : args.tokens].contiguous().to(device)
            )
            baseline = HcuNativeEpInt8Baseline(
                contract=NativeEpContract(
                    world_size=world_size,
                    rank=rank,
                    tokens_per_rank=args.tokens,
                    hidden_size=MODEL_HIDDEN,
                    intermediate_size=MODEL_INTERMEDIATE,
                    num_experts=MODEL_EXPERTS,
                    top_k=MODEL_TOP_K,
                ),
                w13=local_cpu["l1_weight_int8"].contiguous().to(device),
                w2=local_cpu["l2_weight_int8"].contiguous().to(device),
                w13_scale=local_cpu["l1_weight_scale"].contiguous().to(device),
                w2_scale=local_cpu["l2_weight_scale"].contiguous().to(device),
            )
            input_audit = baseline.audit_input_values(
                input_int8,
                input_scales,
                topk_weights,
                topk_ids,
            )

            candidate_reports = []
            baseline_reports = []
            trial_order = []
            latest_candidate = None
            latest_baseline = None
            with set_forward_context(
                attn_metadata=None,
                vllm_config=vllm_config,
                num_tokens=args.tokens,
            ):
                dp_metadata = get_forward_context().dp_metadata
                if dp_metadata is None:
                    raise RuntimeError("vLLM did not create native-EP DP metadata")
                with dp_metadata.sp_local_sizes(1):
                    latest_candidate = megamoe_int8(
                        input_int8,
                        input_scales,
                        topk_ids,
                        topk_weights,
                        candidate_weights,
                        candidate_context,
                    ).clone()
                    candidate_context.raise_if_sync_failed()
                    dist.barrier(group=control_group)
                    latest_baseline = baseline(
                        input_int8,
                        input_scales,
                        topk_weights,
                        topk_ids,
                    ).clone()
                    torch.cuda.synchronize(device)
                    dist.barrier(group=control_group)

                    initial_local = _correctness(
                        latest_candidate,
                        latest_baseline,
                    )
                    initial_by_rank = _gather_objects(
                        initial_local,
                        group=control_group,
                        world_size=world_size,
                    )
                    if not all(item["passed"] for item in initial_by_rank):
                        raise RuntimeError(
                            "candidate/vLLM correctness failed before timing: "
                            f"{initial_by_rank}"
                        )

                    functions = {
                        "candidate": lambda: megamoe_int8(
                            input_int8,
                            input_scales,
                            topk_ids,
                            topk_weights,
                            candidate_weights,
                            candidate_context,
                        ),
                        "vllm_native_ep_eager": lambda: baseline(
                            input_int8,
                            input_scales,
                            topk_weights,
                            topk_ids,
                        ),
                    }
                    first_order = (
                        ["candidate", "vllm_native_ep_eager"]
                        if hashlib.sha256(
                            f"np{world_size}-t{args.tokens}".encode()
                        ).digest()[0]
                        % 2
                        == 0
                        else ["vllm_native_ep_eager", "candidate"]
                    )
                    for repeat_index in range(args.repeats):
                        order = (
                            list(first_order)
                            if repeat_index % 2 == 0
                            else list(reversed(first_order))
                        )
                        trial_order.append(order)
                        for order_index, name in enumerate(order):
                            output, report = _measure_paired_repeat(
                                functions[name],
                                device=device,
                                group=control_group,
                                world_size=world_size,
                                warmup=args.warmup,
                                iterations=args.iterations,
                                repeat_index=repeat_index,
                                order_index=order_index,
                            )
                            if name == "candidate":
                                candidate_context.raise_if_sync_failed()
                                latest_candidate = output
                                candidate_reports.append(report)
                            else:
                                latest_baseline = output
                                baseline_reports.append(report)

            final_local = _correctness(latest_candidate, latest_baseline)
            final_by_rank = _gather_objects(
                final_local,
                group=control_group,
                world_size=world_size,
            )
            if not all(item["passed"] for item in final_by_rank):
                raise RuntimeError(
                    "candidate/vLLM correctness failed after timing: "
                    f"{final_by_rank}"
                )

            candidate_performance = _summarize_paired(
                candidate_reports,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            baseline_performance = _summarize_paired(
                baseline_reports,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            speedup = (
                baseline_performance["primary_latency_us"]
                / candidate_performance["primary_latency_us"]
            )
            payload = {
                "schema_version": 1,
                "operator": "megamoe_int8",
                "backend": "hygon",
                "timing_protocol": "hygon-paired-native-ep-event-v1",
                "comparison": ("FlagGems-vllm MegaMoE vs vLLM 0.18.1 native EP eager"),
                "shape": {
                    "world_size": world_size,
                    "tokens_per_rank": args.tokens,
                    "global_tokens": args.tokens * world_size,
                    "hidden_size": MODEL_HIDDEN,
                    "intermediate_size": MODEL_INTERMEDIATE,
                    "num_experts": MODEL_EXPERTS,
                    "experts_per_rank": MODEL_EXPERTS // world_size,
                    "top_k": MODEL_TOP_K,
                    "token_semantics": "per_rank",
                },
                "paired_protocol": {
                    "same_torchrun": True,
                    "same_prequantized_tensors": True,
                    "trial_order": trial_order,
                    "timer": "HCU events around each complete eager forward",
                    "barriers_inside_event_interval": False,
                    "first_activation_quantization_in_timed_region": False,
                    "router_in_timed_region": False,
                    "candidate_graph_capture_supported": False,
                },
                "input_audit_rank0": input_audit,
                "correctness": {
                    "relative_l2_max": 0.005,
                    "cosine_min": 0.999,
                    "pre_measurement_by_rank": initial_by_rank,
                    "post_measurement_by_rank": final_by_rank,
                    "passed_all_ranks": True,
                },
                "candidate": {
                    "performance": candidate_performance,
                    "primary_latency_us": candidate_performance["primary_latency_us"],
                    "tile": vars(candidate_context.tile),
                    "ncta": candidate_context.ncta,
                    "timed_boundary": (
                        "one persistent launch including DUSHMEM communication, "
                        "both INT8 expert GEMMs, qA2, return, and TopK combine"
                    ),
                },
                "baseline": {
                    "performance": baseline_performance,
                    "primary_latency_us": baseline_performance["primary_latency_us"],
                    "metadata": baseline.metadata(include_source_hashes=True),
                    "timed_boundary": (
                        "native all-gather dispatch, local W8A8 experts with "
                        "dynamic qA2, and reduce-scatter finalize"
                    ),
                },
                "speedup_baseline_over_candidate": speedup,
                "candidate_faster_than_eager_baseline": speedup > 1.0,
                "canonical_data": {
                    "root": str(args.data_dir.resolve()),
                    "format": manifest.get("format"),
                    "format_version": manifest.get("format_version"),
                    "sha256_verified": args.verify_sha256,
                },
                "source_sha256": {
                    "benchmark/test_megamoe.py": _sha256(Path(__file__)),
                    "benchmark/megamoe_hygon_vllm_baseline.py": _sha256(
                        Path(__file__).with_name("megamoe_hygon_vllm_baseline.py")
                    ),
                    "benchmark/megamoe_hygon_host.hip": _sha256(HOST_SOURCE),
                    "src/flaggems_vllm/runtime/backend/_hygon/mega/"
                    "megamoe/api.py": _sha256(
                        Path(__file__).resolve().parents[1]
                        / "src/flaggems_vllm/runtime/backend/_hygon/mega/"
                        "megamoe/api.py"
                    ),
                    "src/flaggems_vllm/runtime/backend/_hygon/mega/"
                    "megamoe/kernel.py": _sha256(
                        Path(__file__).resolve().parents[1]
                        / "src/flaggems_vllm/runtime/backend/_hygon/mega/"
                        "megamoe/kernel.py"
                    ),
                },
            }
            if rank == 0:
                rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
                if args.output is not None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                    temporary.write_text(rendered, encoding="utf-8")
                    temporary.replace(args.output)
                print(rendered, end="")

            torch.cuda.synchronize(device)
            dist.barrier(group=control_group)
            runtime.close()
            runtime = None
            dist.barrier(group=control_group)
            dist.destroy_process_group(control_group)
            control_group = None
        finally:
            if distributed_initialized:
                try:
                    destroy_model_parallel()
                finally:
                    destroy_distributed_environment()
            reset_workspace_manager()


def _main(argv=None) -> None:
    args = _parse_args(argv)
    if args.compare_vllm_eager:
        _run_paired(args)
    else:
        _run_candidate(args)


@pytest.mark.megamoe
@pytest.mark.megamoe_int8
def test_megamoe_benchmark_cli_contract():
    args = _parse_args(
        [
            "--world-size",
            "8",
            "--tokens",
            "512",
            "--correctness-tokens",
            "1",
        ]
    )
    assert args.world_size == 8
    assert args.tokens == 512
    assert args.repeats == 5
    assert not args.compare_vllm_eager
    assert TIMING_PROTOCOL.endswith("event-v1")


if __name__ == "__main__":
    _main(sys.argv[1:])
