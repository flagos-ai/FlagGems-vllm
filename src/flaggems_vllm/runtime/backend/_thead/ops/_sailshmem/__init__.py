# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Native sailSHMEM workspace and JIT-module registration for single-node PPU TP."""

import ctypes
import functools
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_process_group_store
from triton.runtime.cache import get_cache_manager

_RUNTIME = None


def _command(args, directory, **kwargs):
    result = subprocess.run(
        [str(arg) for arg in args],
        cwd=directory,
        stderr=subprocess.PIPE,
        **kwargs,
    )
    if result.returncode:
        raise RuntimeError(
            f"{Path(args[0]).name} failed while preparing sailSHMEM: "
            f"{result.stderr.decode(errors='replace')}"
        )


@functools.lru_cache(maxsize=1)
def _build_libraries():
    sdk = Path(os.getenv("PPU_SDK", "/usr/local/PPU_SDK"))
    home = Path(os.getenv("SAILSHMEM_HOME", "/usr"))
    lib = home / "lib" / "x86_64-linux-gnu"
    if not (lib / "libnvshmem_device.a").exists():
        lib = home / "lib"
    archive = lib / "libnvshmem_device.a"
    native_host = lib / "libnvshmem_host.so"
    if not (archive.is_file() and native_host.is_file()):
        raise NotImplementedError(
            "PPU MegaPreAttention requires the sailSHMEM SDK and device archive"
        )

    source = Path(__file__).parent
    host_source = source / "bootstrap.cpp"
    device_source = source / "p2p.cu"
    digest = hashlib.sha256()
    for path in (
        host_source,
        device_source,
        archive,
        native_host,
        sdk / "bin" / "clang++",
        home / "include" / "device" / "nvshmem_defines.h",
    ):
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    cache = get_cache_manager(digest.hexdigest())
    host_name, device_name = "sailshmem-bootstrap.so", "sailshmem-ppu10.bc"
    host_path, device_path = cache.get_file(host_name), cache.get_file(device_name)
    if host_path is not None and device_path is not None:
        return native_host, host_path, device_path

    with tempfile.TemporaryDirectory(dir=cache.cache_dir) as temporary:
        work = Path(temporary)
        include = ["-I", sdk / "CUDA_SDK" / "include", "-I", home / "include"]
        host = work / host_name
        device = work / device_name
        _command(
            [
                "g++",
                "-O2",
                "-std=c++17",
                "-fPIC",
                "-shared",
                *include,
                host_source,
                native_host,
                "-o",
                host,
            ],
            work,
        )
        init_object = work / "init_device.cu.o"
        with init_object.open("wb") as output:
            _command(["ar", "p", archive, init_object.name], work, stdout=output)
        _command([sdk / "bin" / "hgobjdump", "--extract-bc=0", init_object], work)
        # The ZW810E uses PPU 1.0. Keep the SDK's device state ABI local to
        # the library that's installed on the running host.
        (init_bc,) = work.glob("*_PPU10")
        p2p_bc = work / "p2p.bc"
        _command(
            [
                sdk / "bin" / "clang++",
                "-x",
                "hggc",
                "--hggc-device-only",
                "--cuda-gpu-arch=ppu001",
                "-std=c++17",
                "-O2",
                "-fgpu-rdc",
                "-emit-llvm",
                "-c",
                *include,
                "-Xclang",
                "-fno-cuda-host-device-constexpr",
                "-D___HGGC_CG_HELPER_H___",
                "-D__CUDACC_RDC__",
                device_source,
                "-o",
                p2p_bc,
            ],
            work,
        )
        _command([sdk / "bin" / "llvm-link", p2p_bc, init_bc, "-o", device], work)
        host_path = cache.put(host.read_bytes(), host_name, binary=True)
        device_path = cache.put(device.read_bytes(), device_name, binary=True)
    return native_host, host_path, device_path


def _bind(library, name, result, *args):
    function = getattr(library, name)
    function.restype = result
    function.argtypes = args
    return function


def _check(result, operation):
    if result != 0:
        raise RuntimeError(f"sailSHMEM {operation} failed with status {result}")


class _Runtime:
    def __init__(self, group, device):
        if torch.cuda.get_device_capability(device) != (8, 0):
            raise NotImplementedError("PPU MegaPreAttention requires PPU-ZW810E")
        native_file, bootstrap_file, self.bitcode = _build_libraries()
        native = ctypes.CDLL(str(native_file), mode=ctypes.RTLD_GLOBAL)
        bootstrap = ctypes.CDLL(bootstrap_file)
        pointer, integer, size = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        uid_size = _bind(bootstrap, "mega_uid_size", size)
        uid_create = _bind(bootstrap, "mega_make_uid", integer, pointer)
        init = _bind(bootstrap, "mega_init", integer, pointer, integer, integer)
        self.register = _bind(native, "nvshmemx_hgmodule_init", integer, pointer)
        self.unregister = _bind(native, "nvshmemx_hgmodule_finalize", integer, pointer)
        self.alloc = _bind(native, "nvshmem_calloc", pointer, size, size)
        self.free = _bind(native, "nvshmem_free", None, pointer)
        self.peer = _bind(native, "nvshmem_ptr", pointer, pointer, integer)
        self.close = _bind(native, "nvshmemx_hostlib_finalize", None)
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.workspaces = {}
        self.modules = {}

        store = dist.PrefixStore(
            "flaggems_vllm/sailshmem", _get_process_group_store(group)
        )
        generation = store.add(f"generation/{self.rank}", 1)
        self.store = dist.PrefixStore(str(generation), store)
        self.workspace_attempts = 0

        uid = (ctypes.c_ubyte * uid_size())()
        if self.rank == 0:
            status = uid_create(uid)
            self.store.set(
                "uid", status.to_bytes(4, "little", signed=True) + bytes(uid)
            )
        payload = self.store.get("uid")
        _check(int.from_bytes(payload[:4], "little", signed=True), "unique ID creation")
        uid[:] = payload[4:]
        _check(init(uid, self.rank, self.world), "initialization")
        native_rank = _bind(native, "nvshmem_my_pe", integer)()
        native_world = _bind(native, "nvshmem_n_pes", integer)()
        if (native_rank, native_world) != (self.rank, self.world):
            self.close()
            raise RuntimeError("sailSHMEM is bound to a different process group")

    def workspace(self, key, size):
        if key not in self.workspaces:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm up MegaPreAttention before graph capture")
            pointer = self.alloc(1, size)
            peers_mapped = pointer is not None and all(
                self.peer(pointer, peer) for peer in range(self.world)
            )
            prefix = f"workspace/{self.workspace_attempts}"
            self.workspace_attempts += 1
            peer_status = self.exchange_status(prefix, peers_mapped)
            if any(status != b"1" for status in peer_status):
                if pointer is not None:
                    self.free(pointer)
                raise NotImplementedError(
                    "PPU MegaPreAttention requires all TP peers to be P2P accessible"
                )
            tickets = torch.zeros((), device=self.device, dtype=torch.uint64)
            self.workspaces[key] = pointer, tickets
        return self.workspaces[key]

    def exchange_status(self, prefix, success=True):
        self.store.set(f"{prefix}/{self.rank}", b"1" if success else b"0")
        return self.store.multi_get([f"{prefix}/{rank}" for rank in range(self.world)])

    def register_kernel(self, kernel):
        if kernel.module not in self.modules:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm up the MegaPreAttention layout before capture")
            kernel._init_handles()
            _check(self.register(kernel.module), "device module registration")
            self.modules[kernel.module] = kernel

    def release(self):
        with torch.cuda.device(self.device):
            torch.cuda.synchronize()
            self.exchange_status("release")
            for module in self.modules:
                _check(self.unregister(module), "device module cleanup")
            for pointer, _ in self.workspaces.values():
                self.free(pointer)
            self.close()


def runtime_for(group, device):
    global _RUNTIME
    if _RUNTIME is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up MegaPreAttention before graph capture")
        _RUNTIME = _Runtime(group, device)
    elif _RUNTIME.group is not group or _RUNTIME.device != device:
        raise NotImplementedError(
            "PPU MegaPreAttention needs one active sailSHMEM group per process"
        )
    return _RUNTIME


def clear_workspace():
    global _RUNTIME
    if _RUNTIME is not None:
        _RUNTIME.release()
        _RUNTIME = None
