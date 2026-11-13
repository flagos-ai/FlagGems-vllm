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

"""Prepare caller-owned MUSA MegaMoE accuracy fixtures without import effects."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

# Match the public unique-ID bootstrap ABI in MT-SHMEM 3.x headers.
UNIQUE_ID_BYTES = 128
# Match nvshmemx_init_args_v1 and nvshmemx_init_attr_v1 on the 64-bit host ABI.
INIT_ARGS_BYTES = 128
INIT_ATTR_BYTES = 144


class MtshmemRuntime:
    """Own explicit device setup, unique-ID bootstrap and symmetric allocations.

    No library is loaded until open(). The supplied shim is the existing host
    shim; its MT-SHMEM dependency also exposes the unique-ID bootstrap API.
    Direct subprocess ranks use that API instead of MPI initialization.
    The ctypes ABI mirrors nvshmem_types.h and nvshmem_uniqueid.h in MT-SHMEM
    3.x; reject other ABI sizes rather than silently guessing a layout.
    All ranks allocate/free in identical order and close collectively.
    """

    def __init__(
        self,
        shim: str,
        rank: int,
        world: int,
        device: int,
        rendezvous: str,
        runtime_library: str = "libmusart.so",
    ) -> None:
        self.shim = shim
        self.rank = rank
        self.world = world
        self.device_index = device
        self.rendezvous = Path(rendezvous)
        self.runtime_library = runtime_library
        self.allocations = []
        self.views = []
        self.opened = False

    def open(self) -> "MtshmemRuntime":
        """Load libraries and initialize the selected rank explicitly."""
        if self.opened:
            raise RuntimeError("MT-SHMEM runtime is already open")
        import torch
        import torch_musa  # noqa: F401

        self.torch = torch
        self.runtime = ctypes.CDLL(self.runtime_library, mode=ctypes.RTLD_GLOBAL)
        self.runtime.musaSetDevice.argtypes = [ctypes.c_int]
        self.runtime.musaSetDevice.restype = ctypes.c_int
        self._check(self.runtime.musaSetDevice(self.device_index), "musaSetDevice")
        torch.musa.set_device(self.device_index)
        self.device = torch.device("musa", self.device_index)
        self.lib = ctypes.CDLL(self.shim, mode=ctypes.RTLD_GLOBAL)
        self._configure_api()
        uid = (ctypes.c_byte * UNIQUE_ID_BYTES)()
        ctypes.c_int.from_buffer(uid).value = (1 << 16) + UNIQUE_ID_BYTES
        if self.rank == 0:
            self._check(self.lib.nvshmemx_get_uniqueid(uid), "get_uniqueid")
            temporary = self.rendezvous.with_suffix(".pending")
            temporary.write_bytes(bytes(uid))
            temporary.replace(self.rendezvous)
        else:
            deadline = time.monotonic() + 60
            while not self.rendezvous.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Rank zero did not publish the MT-SHMEM ID")
                time.sleep(0.02)
            data = self.rendezvous.read_bytes()
            if len(data) != UNIQUE_ID_BYTES:
                raise RuntimeError("Invalid MT-SHMEM unique-ID byte count")
            ctypes.memmove(uid, data, UNIQUE_ID_BYTES)
        attr = self._attribute_type()()
        attr.version = (1 << 16) + INIT_ATTR_BYTES
        attr.args.version = (1 << 16) + INIT_ARGS_BYTES
        attr.args.uid_args.version = (1 << 16) + 24
        self._check(
            self.lib.nvshmemx_set_attr_uniqueid_args(
                self.rank, self.world, uid, ctypes.byref(attr)
            ),
            "set_attr_uniqueid_args",
        )
        provided = ctypes.c_int()
        version = self._version_type()(3, 0, 0)
        # This is the C++ entry used by nvshmemx_init_attr in the installed header.
        init = getattr(
            self.lib,
            "_Z20nvshmemi_init_threadiPijP21nvshmemx_init_attr_v118nvshmemi_version_t",
        )
        init.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
            ctypes.POINTER(type(attr)),
            type(version),
        ]
        init.restype = ctypes.c_int
        self._check(
            init(2, ctypes.byref(provided), 1 << 3, ctypes.byref(attr), version),
            "nvshmemi_init_thread",
        )
        self.opened = True
        actual = (self.lib.probe_nvshmem_my_pe(), self.lib.probe_nvshmem_n_pes())
        if actual != (self.rank, self.world):
            raise RuntimeError(f"MT-SHMEM rank mismatch: {actual}")
        return self

    def close(self) -> None:
        """Synchronize, free allocations collectively and finalize explicitly."""
        if not self.opened:
            return
        self.torch.musa.synchronize()
        self.barrier()
        self.views.clear()
        for address in reversed(self.allocations):
            self.lib.probe_nvshmem_free(address)
        self.allocations.clear()
        self.lib.probe_nvshmem_finalize()
        self.opened = False

    def barrier(self) -> None:
        """Perform a host all-rank rendezvous after caller stream synchronization."""
        if not self.opened:
            raise RuntimeError("MT-SHMEM runtime is not open")
        self.lib.probe_nvshmem_barrier_all()

    def allocate(self, shape: Tuple[int, ...], dtype: Any) -> Tuple[Any, Any]:
        """Return a borrowed symmetric tensor and its rank-ordered peer table."""
        if not self.opened:
            raise RuntimeError("MT-SHMEM runtime is not open")
        torch = self.torch
        elements = 1
        for dimension in shape:
            elements *= dimension
        nbytes = elements * torch.empty((), dtype=dtype).element_size()
        address = int(self.lib.probe_nvshmem_malloc(nbytes) or 0)
        if not address:
            raise MemoryError(f"Symmetric allocation failed: {nbytes} bytes")
        self.allocations.append(address)
        storage = torch._C._construct_storage_from_data_pointer(
            address, self.device, nbytes
        )
        tensor = torch.empty(0, dtype=dtype, device=self.device)
        tensor.set_(storage, 0, shape)
        self.views.append(tensor)
        addresses = [
            int(
                address
                if rank == self.rank
                else self.lib.probe_nvshmem_ptr(address, rank) or 0
            )
            for rank in range(self.world)
        ]
        if not all(addresses):
            raise RuntimeError(f"Missing symmetric peer mapping: {addresses}")
        table = torch.tensor(addresses, dtype=torch.int64, device=self.device)
        return tensor, table

    def _configure_api(self) -> None:
        for name in ["barrier_all", "finalize"]:
            fn = getattr(self.lib, "probe_nvshmem_" + name)
            fn.argtypes = []
            fn.restype = None
        for name in ["my_pe", "n_pes"]:
            fn = getattr(self.lib, "probe_nvshmem_" + name)
            fn.argtypes = []
            fn.restype = ctypes.c_int
        self.lib.probe_nvshmem_malloc.argtypes = [ctypes.c_size_t]
        self.lib.probe_nvshmem_malloc.restype = ctypes.c_void_p
        self.lib.probe_nvshmem_free.argtypes = [ctypes.c_void_p]
        self.lib.probe_nvshmem_free.restype = None
        self.lib.probe_nvshmem_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.probe_nvshmem_ptr.restype = ctypes.c_void_p
        self.lib.nvshmemx_get_uniqueid.argtypes = [ctypes.c_void_p]
        self.lib.nvshmemx_get_uniqueid.restype = ctypes.c_int
        self.lib.nvshmemx_set_attr_uniqueid_args.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.nvshmemx_set_attr_uniqueid_args.restype = ctypes.c_int

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status:
            raise RuntimeError(f"{operation} failed: {status}")

    @staticmethod
    def _version_type() -> Any:
        class Version(ctypes.Structure):
            _fields_ = [
                ("major", ctypes.c_int),
                ("minor", ctypes.c_int),
                ("patch", ctypes.c_int),
            ]

        return Version

    @staticmethod
    def _attribute_type() -> Any:
        class UniqueArgs(ctypes.Structure):
            _fields_ = [
                ("version", ctypes.c_int),
                ("id", ctypes.c_void_p),
                ("rank", ctypes.c_int),
                ("world", ctypes.c_int),
            ]

        class InitArgs(ctypes.Structure):
            _fields_ = [
                ("version", ctypes.c_int),
                ("uid_args", UniqueArgs),
                ("content", ctypes.c_char * 96),
            ]

        class Attribute(ctypes.Structure):
            _fields_ = [
                ("version", ctypes.c_int),
                ("mpi_comm", ctypes.c_void_p),
                ("args", InitArgs),
            ]

        if (
            ctypes.sizeof(UniqueArgs),
            ctypes.sizeof(InitArgs),
            ctypes.sizeof(Attribute),
        ) != (24, INIT_ARGS_BYTES, INIT_ATTR_BYTES):
            raise RuntimeError("Unsupported MT-SHMEM unique-ID ABI")
        return Attribute


def load_case(
    shared_path: str, weights_path: str, rank: int, world: int, tokens: int
) -> Dict[str, Any]:
    """Load external Qwen layer-zero data and prepare the FP64 CPU oracle."""
    import torch

    shared = torch.load(shared_path, mmap=True, map_location="cpu", weights_only=True)
    if shared["topk_idx"].shape[0] != world:
        raise ValueError("Frozen input world size differs from requested ranks")
    if not 0 < tokens <= shared["topk_idx"].shape[1]:
        raise ValueError("Requested tokens exceed frozen input capacity")
    x = shared["input_fp8_bits"][rank, :tokens].contiguous().view(torch.float8_e4m3fn)
    scales = shared["input_scales"][rank, :tokens].contiguous()
    ids = shared["topk_idx"][rank, :tokens].int().contiguous()
    weights = shared["topk_weights"][rank, :tokens].contiguous()
    all_ids = shared["topk_idx"][:, :tokens].contiguous()
    active = sorted(set(ids.flatten().tolist()) - {-1})
    oracle_weights = _read_experts(weights_path, active)
    reference, parts = fp64_oracle(x, scales, ids, weights, active, oracle_weights)
    local = _read_experts(
        weights_path, range(rank * (256 // world), (rank + 1) * (256 // world))
    )
    return dict(
        x=x,
        scales=scales,
        ids=ids,
        weights=weights,
        all_ids=all_ids,
        local_weights=local,
        reference=reference,
        reference_parts=parts,
    )


def fp64_oracle(
    x: Any, scales: Any, ids: Any, route_weights: Any, active: Any, expert_weights: Any
) -> Tuple[Any, Any]:
    """Match SRC FP64 gate/up, FP8 block64 activation and BF16 partial rounding."""
    import torch

    w1, s1, w2, s2 = expert_weights
    tokens, hidden = x.shape
    top = ids.shape[1]
    intermediate = w2.shape[-1]
    xref = x.double() * scales.double().repeat_interleave(128, -1)
    parts = torch.zeros(tokens, top, hidden, dtype=torch.bfloat16)
    for offset, expert in enumerate(active):
        where = (ids == expert).nonzero()
        tok, slot = where[:, 0], where[:, 1]
        value = xref[tok] @ _dequantize(w1[offset], s1[offset]).T
        gate, up = value.chunk(2, -1)
        sw = torch.nn.functional.silu(gate) * up
        magnitude = (
            sw.reshape(-1, intermediate // 64, 64).abs().amax(-1)
            * route_weights[tok, slot].double()[:, None]
            / 448
        )
        scale = torch.where(
            magnitude > 0,
            torch.exp2(torch.ceil(torch.log2(magnitude))),
            torch.ones_like(magnitude),
        )
        quant = (
            sw
            * route_weights[tok, slot].double()[:, None]
            / scale.repeat_interleave(64, -1)
        ).to(torch.float8_e4m3fn)
        activation = quant.double() * scale.repeat_interleave(64, -1)
        parts[tok, slot] = (
            activation @ _dequantize(w2[offset], s2[offset]).T
        ).bfloat16()
    reference = torch.zeros(tokens, hidden)
    for slot in range(top):
        reference += parts[:, slot].float()
    return reference.bfloat16(), parts


def make_context(runtime: MtshmemRuntime, case: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the caller-owned context from the source driver's capacity plan.

    SRC benchmark.py uses CAP=T*TOP*R, POOL=ceil(CAP/128)+E, EPW=16,
    staged/quant rows=POOL*256, scales rounded into 64-element descriptor rows,
    and gu256 gate/up interleave. No im2col transform is used: inputs are
    contiguous token-by-hidden matrices, H=2048/I=512/E*R=256/TOP=8 and
    descriptor tiles A32x128/B1 256x128/B2 256x128/Q64x64 are fixed.

    The actual gathered frozen routes prove max_recv<=T*TOP, each destination
    expert total<=CAP, padded expert sum<=128*POOL and 32-row metadata task
    count<=8*POOL. EPW=16 and CTAS=60 preserve the source residency premise.
    The API's explicit max scratch bounds also reserve NP2/4 remote caches.
    Borrowed symmetric input tensors share storage with their pointer tables.
    """
    import torch

    from flaggems_vllm.runtime.backend._mthreads.mega.megamoe import MegaMoEContext

    t, h = case["x"].shape
    world = runtime.world
    top, intermediate, experts = 8, 512, 256 // world
    cap = t * top * world
    pool = (cap + 127) // 128 + experts
    all_ids = case["all_ids"]
    max_recv = 0
    for source in range(world):
        counts = torch.bincount(all_ids[source].flatten().long(), minlength=256)
        max_recv = max(max_recv, int(counts.max()))
    for destination in range(world):
        counts = torch.bincount(all_ids.flatten().long(), minlength=256)[
            destination * experts : (destination + 1) * experts
        ]
        assert int(counts.max()) <= cap
        assert int((((counts + 127) // 128) * 128).sum()) <= pool * 128
        assert int(((counts + 31) // 32).sum()) <= pool * 8
    tensors = {}
    tables = {}
    for name, shape, dtype in [
        ("input", (t, h), torch.float8_e4m3fn),
        ("input_scale", (t, h // 128), torch.float32),
        ("route_weight", (t, top), torch.float32),
        ("queue", (experts, cap), torch.int32),
        ("count", (experts,), torch.int32),
        ("partial", (t, top, h), torch.bfloat16),
        ("signal", (3,), torch.int32),
    ]:
        tensors[name], tables[name] = runtime.allocate(shape, dtype)
    tensors["input"].copy_(case["x"])
    tensors["input_scale"].copy_(case["scales"])
    tensors["route_weight"].copy_(case["weights"])
    tensors["count"].zero_()
    tensors["signal"].zero_()
    tensors["partial"].fill_(float("nan"))
    w1, s1, w2, s2 = [x.to(runtime.device) for x in case["local_weights"]]
    packed = (
        w1.reshape(experts, 2, intermediate // 128, 128, h)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .reshape(experts * 2 * intermediate, h)
    )
    p = pool * 128
    rows = max(2 * p, cap + world * t)
    quant_scale_size = max(
        2 * p * (intermediate // 64),
        p * (intermediate // 64) + 32 * pool + world * t * top,
    )
    quant_scale_size = ((quant_scale_size + 63) // 64) * 64
    context = MegaMoEContext(
        **{name + "_ptrs": table for name, table in tables.items()},
        grid=torch.zeros(10, dtype=torch.int32, device=runtime.device),
        input_workspace=torch.empty(
            rows, h, dtype=torch.float8_e4m3fn, device=runtime.device
        ),
        input_scale_workspace=torch.empty(
            rows * (h // 128), dtype=torch.float32, device=runtime.device
        ),
        route_weight_workspace=torch.empty(
            p, dtype=torch.float32, device=runtime.device
        ),
        quant_workspace=torch.empty(
            2 * p, intermediate, dtype=torch.float8_e4m3fn, device=runtime.device
        ),
        quant_scale_workspace=torch.empty(
            quant_scale_size, dtype=torch.float32, device=runtime.device
        ),
        ready=torch.empty(pool, dtype=torch.int32, device=runtime.device),
        quant_ready=torch.empty(pool, dtype=torch.int32, device=runtime.device),
        barrier=runtime.barrier,
        E=experts,
        CAP=cap,
        MY=runtime.rank,
        R=world,
        POOL=pool,
        EPW=16,
        max_recv=max_recv,
    )
    inputs = (
        tensors["input"],
        tensors["input_scale"],
        case["ids"].to(runtime.device),
        tensors["route_weight"],
        packed,
        s1,
        w2.reshape(experts * h, intermediate),
        s2,
    )
    output = torch.empty(t, h, dtype=torch.bfloat16, device=runtime.device)
    torch.musa.synchronize()
    runtime.barrier()
    return dict(context=context, inputs=inputs, output=output, symmetric=tensors)


def assert_accuracy(
    case: Dict[str, Any], fixture: Dict[str, Any], rank: int, world: int
) -> Dict[str, Any]:
    """Enforce source route, partial, ordered-combine and finite assertions.

    Every assertion runs for NP1 as well as NP>=2. For NP1, partial ownership
    is local; NP>=2 additionally exercises remote partial writes. Combining
    TOP route partials is meaningful even with one rank and is never skipped.
    """
    import torch

    out = fixture["output"].cpu().double()
    expected = case["reference"].double()
    partials = fixture["symmetric"]["partial"].cpu()
    partial_reference = case["reference_parts"].double()
    relative = _relative_l2(out, expected)
    partial_relative = _relative_l2(partials.double(), partial_reference)
    assert torch.isfinite(out).all() and torch.isfinite(partials).all()
    assert relative <= 0.003 and partial_relative <= 0.003
    combined = torch.zeros_like(case["reference"], dtype=torch.float32)
    for slot in range(case["ids"].shape[1]):
        combined += partials[:, slot].float()
    exact = torch.equal(
        out.bfloat16().view(torch.int16), combined.bfloat16().view(torch.int16)
    )
    assert exact
    count = fixture["symmetric"]["count"].cpu().long()
    queue = fixture["symmetric"]["queue"].cpu()
    all_ids = case["all_ids"]
    experts = 256 // world
    expected_counts = torch.bincount(all_ids.flatten().long(), minlength=256)[
        rank * experts : (rank + 1) * experts
    ]
    assert torch.equal(count, expected_counts)
    t, top = case["ids"].shape
    for local_expert in range(experts):
        routes = queue[local_expert, : int(count[local_expert])].long()
        assert routes.numel() == torch.unique(routes).numel()
        source = routes // (t * top)
        local = routes % (t * top)
        assert bool(((source >= 0) & (source < world)).all())
        assert bool(
            (
                all_ids.reshape(world, t * top)[source, local]
                == rank * experts + local_expert
            ).all()
        )
    valid = case["ids"] >= 0
    written = torch.isfinite(partials).all(-1)
    missing = int((valid & ~written).sum())
    assert missing == 0
    return dict(
        status="PASS",
        rank=rank,
        world=world,
        tokens=t,
        relative_l2=relative,
        partial_relative_l2=partial_relative,
        token_order_exact=exact,
        combine_bitwise_exact=exact,
        route_counts_exact=True,
        queue_unique_and_expert_exact=True,
        cross_rank_partial_pass=True,
        missing_route_slots=missing,
        duplicate_queue_slots=0,
        finite=True,
        output_nan=int(torch.isnan(out).sum()),
        output_inf=int(torch.isinf(out).sum()),
        partial_nan=int(torch.isnan(partials).sum()),
        partial_inf=int(torch.isinf(partials).sum()),
    )


def run_accuracy_worker(
    shim: str,
    shared_path: str,
    weights_path: str,
    rank: int,
    world: int,
    device: int,
    rendezvous: str,
    iterations: int = 25,
    tokens: int = 1,
    runtime_library: str = "libmusart.so",
) -> Dict[str, Any]:
    """Run one rank through the public API, with compile warmup and real checks."""
    import torch

    from flaggems_vllm.runtime.backend._mthreads.mega.megamoe import megamoe_fp8

    if iterations < 1:
        raise ValueError("At least one checked iteration is required")
    torch.set_num_threads(2)
    runtime = MtshmemRuntime(
        shim, rank, world, device, rendezvous, runtime_library
    ).open()
    try:
        case = load_case(shared_path, weights_path, rank, world, tokens)
        fixture = make_context(runtime, case)
        result = {}
        # The first API call performs compile-only 1/2/16 warmups and one launch.
        for iteration in range(iterations + 1):
            megamoe_fp8(*fixture["inputs"], fixture["context"], fixture["output"])
            torch.musa.synchronize()
            runtime.barrier()
            result = assert_accuracy(case, fixture, rank, world)
        result["warmup_launches"] = 1
        result["checked_iterations"] = iterations
        return result
    finally:
        runtime.close()


def _read_experts(path: str, experts: Any) -> Tuple[Any, Any, Any, Any]:
    import torch
    from safetensors import safe_open

    first, first_scale, second, second_scale = [], [], [], []
    with safe_open(path, framework="pt", device="cpu") as source:
        for expert in experts:
            prefix = f"model.language_model.layers.0.mlp.experts.{expert}."
            first.append(
                torch.cat(
                    [
                        source.get_tensor(prefix + "gate_proj.weight"),
                        source.get_tensor(prefix + "up_proj.weight"),
                    ]
                )
            )
            first_scale.append(
                torch.cat(
                    [
                        source.get_tensor(prefix + "gate_proj.weight_scale_inv"),
                        source.get_tensor(prefix + "up_proj.weight_scale_inv"),
                    ]
                ).float()
            )
            second.append(source.get_tensor(prefix + "down_proj.weight"))
            second_scale.append(
                source.get_tensor(prefix + "down_proj.weight_scale_inv").float()
            )
    return tuple(torch.stack(v) for v in [first, first_scale, second, second_scale])


def _dequantize(weight: Any, scale: Any) -> Any:
    return weight.double() * scale.double().repeat_interleave(
        128, -2
    ).repeat_interleave(128, -1)


def _relative_l2(actual: Any, expected: Any) -> float:
    import torch

    denominator = torch.linalg.vector_norm(expected)
    if not denominator:
        return float(torch.linalg.vector_norm(actual - expected))
    return float(torch.linalg.vector_norm(actual - expected) / denominator)


_SRC_ROOT = Path(__file__).resolve().parents[2]
_RESULT_PREFIX = "MEGAMOE_RESULT "
_LATENCY_PREFIX = "MEGAMOE_LATENCY_US "


def _worker_environment(cache_dir: str) -> dict:
    """Return a child environment that can import this checkout."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{_SRC_ROOT}{os.pathsep}{existing}" if existing else str(_SRC_ROOT)
    )
    env["TRITON_CACHE_DIR"] = cache_dir
    env["NVSHMEM_BOOTSTRAP"] = "UID"
    env["NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME"] = "lo"
    return env


def _rank_command(role, rank, world, device, rendezvous, iterations, tokens):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        f"--{role}-worker",
        "--rank",
        str(rank),
        "--world",
        str(world),
        "--device",
        str(device),
        "--rendezvous",
        str(rendezvous),
        "--iterations",
        str(iterations),
        "--tokens",
        str(tokens),
    ]


def run_ranked(role, devices, iterations=25, tokens=1, timeout=1800.0):
    """Run one worker per device and return the parsed per-rank records."""
    devices = [str(device) for device in devices]
    world = len(devices)
    if world < 1:
        raise ValueError("At least one device is required")
    records = []
    with tempfile.TemporaryDirectory(prefix=f"megamoe-{role}-") as root:
        root = Path(root)
        rendezvous = root / "unique-id"
        spawned = []
        try:
            for rank, device in enumerate(devices):
                command = _rank_command(
                    role, rank, world, device, rendezvous, iterations, tokens
                )
                env = _worker_environment(str(root / "cache" / str(rank)))
                out = (root / f"rank{rank}.out").open("w")
                err = (root / f"rank{rank}.err").open("w")
                process = subprocess.Popen(command, env=env, stdout=out, stderr=err)
                spawned.append((rank, process, out, err))
            deadline = time.monotonic() + timeout
            pending = list(spawned)
            while pending:
                for item in pending[:]:
                    rank, process, out, err = item
                    if process.poll() is None:
                        continue
                    out.flush()
                    err.flush()
                    if process.returncode:
                        raise RuntimeError(
                            f"rank {rank} exit={process.returncode}: "
                            + (root / f"rank{rank}.err").read_text()[-2000:]
                        )
                    pending.remove(item)
                if pending:
                    if time.monotonic() > deadline:
                        raise TimeoutError("MegaMoE rank workers timed out")
                    time.sleep(0.05)
            for rank in range(world):
                text = (root / f"rank{rank}.out").read_text()
                errors = (root / f"rank{rank}.err").read_text()
                if errors.strip():
                    print(f"MEGAMOE_STDERR rank={rank}\n{errors}", flush=True)
                for line in text.splitlines():
                    if line.startswith(_RESULT_PREFIX):
                        record = json.loads(line[len(_RESULT_PREFIX) :])
                    elif line.startswith(_LATENCY_PREFIX):
                        record = {
                            "rank": rank,
                            "latency_us": float(line[len(_LATENCY_PREFIX) :]),
                        }
                    else:
                        continue
                    records.append(record)
        finally:
            for rank, process, out, err in spawned:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                out.close()
                err.close()
    if len(records) != world:
        raise RuntimeError(f"expected {world} rank records, got {len(records)}")
    return records


def run_accuracy(devices, iterations=25, tokens=1, timeout=1800.0):
    """Validate every rank against the FP64 oracle and return the records."""
    records = run_ranked("accuracy", devices, iterations, tokens, timeout)
    for record in records:
        if record.get("status") != "PASS":
            raise AssertionError(f"rank {record.get('rank')} not PASS: {record}")
        if record["relative_l2"] > 0.003:
            raise AssertionError(f"relative_l2 too large: {record['relative_l2']}")
    return records


def run_timing(devices, iterations=20, warmup=5, tokens=1, timeout=3600.0):
    """Return the slowest-rank mean latency in milliseconds."""
    records = run_ranked("benchmark", devices, iterations, tokens, timeout)
    return max(float(record["latency_us"]) for record in records) / 1e3


def run_benchmark_worker(
    shim,
    shared_path,
    weights_path,
    rank,
    world,
    device,
    rendezvous,
    iterations=20,
    tokens=1,
    warmup=5,
    runtime_library="libmusart.so",
):
    """Time one rank for ``iterations`` calls and print the mean latency."""
    import torch

    from flaggems_vllm.runtime.backend._mthreads.mega.megamoe import megamoe_fp8

    torch.set_num_threads(2)
    runtime = MtshmemRuntime(
        shim, rank, world, device, rendezvous, runtime_library
    ).open()
    try:
        case = load_case(shared_path, weights_path, rank, world, tokens)
        fixture = make_context(runtime, case)
        for _ in range(warmup):
            megamoe_fp8(*fixture["inputs"], fixture["context"], fixture["output"])
        torch.musa.synchronize()
        runtime.barrier()
        samples = []
        for _ in range(iterations):
            start = torch.musa.Event(enable_timing=True)
            end = torch.musa.Event(enable_timing=True)
            runtime.barrier()
            start.record()
            megamoe_fp8(*fixture["inputs"], fixture["context"], fixture["output"])
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        mean_ms = sum(samples) / len(samples)
        return {
            "rank": rank,
            "world": world,
            "samples": len(samples),
            "latency_us": mean_ms * 1e3,
        }
    finally:
        runtime.close()


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy-worker", action="store_true")
    parser.add_argument("--benchmark-worker", action="store_true")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world", type=int)
    parser.add_argument("--device", type=int)
    parser.add_argument("--rendezvous")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--tokens", type=int, default=1)
    args = parser.parse_args(argv)
    common = dict(
        shim=os.environ["MEGAMOE_TEST_SHIM"],
        shared_path=os.environ["MEGAMOE_TEST_SHARED"],
        weights_path=os.environ["MEGAMOE_TEST_WEIGHTS"],
        rank=args.rank,
        world=args.world,
        device=args.device,
        rendezvous=args.rendezvous,
        iterations=args.iterations,
        tokens=args.tokens,
        runtime_library=os.environ.get("MEGAMOE_TEST_MUSART", "libmusart.so"),
    )
    if args.accuracy_worker:
        print(_RESULT_PREFIX + json.dumps(run_accuracy_worker(**common)), flush=True)
        return
    if args.benchmark_worker:
        result = run_benchmark_worker(**common)
        print(f"{_LATENCY_PREFIX}{result['latency_us']:.6f}", flush=True)
        return
    parser.error("Use a worker mode")


if __name__ == "__main__":
    _main()
