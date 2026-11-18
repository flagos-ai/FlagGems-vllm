# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import ctypes
import functools
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from triton.runtime.cache import get_cache_manager


@functools.lru_cache(maxsize=1)
def counter_memset():
    """Bind the stream-ordered PPU counter initialization."""
    try:
        memset = ctypes.CDLL("libhggc.so").hgMemsetD8Async
    except (OSError, AttributeError):
        return None
    memset.argtypes = [
        ctypes.c_uint64,
        ctypes.c_ubyte,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    memset.restype = ctypes.c_int

    return memset


@functools.lru_cache(maxsize=1)
def device_bitcode():
    """Compile the optional PPU device helpers for the fused pipeline."""
    sdk = Path(os.environ.get("PPU_SDK", "/usr/local/PPU_SDK"))
    compiler = sdk / "bin" / "clang++"
    headers = sdk / "CUDA_SDK" / "include"
    if not compiler.is_file() or not (headers / "hggc_runtime.h").is_file():
        return None
    source = Path(__file__).with_name("megamhc_device.cu")
    compiler_stat = compiler.stat()
    digest = hashlib.sha256(
        source.read_bytes()
        + f"{compiler.resolve()}:{compiler_stat.st_size}:{compiler_stat.st_mtime_ns}".encode()
    )
    version_file = sdk / "VERSION.txt"
    if version_file.is_file():
        digest.update(version_file.read_bytes())
    cache = get_cache_manager(digest.hexdigest())
    name = "megamhc-ppu001.bc"
    path = cache.get_file(name)
    if path is None:
        with tempfile.TemporaryDirectory(dir=cache.cache_dir) as temporary:
            bitcode = Path(temporary) / name
            try:
                subprocess.run(
                    [
                        str(compiler),
                        "-x",
                        "hggc",
                        "--hggc-device-only",
                        "--cuda-gpu-arch=ppu001",
                        "-std=c++17",
                        "-O2",
                        "-ffp-contract=off",
                        "-fgpu-rdc",
                        "-emit-llvm",
                        "-c",
                        "-I",
                        str(sdk / "include"),
                        "-I",
                        str(headers),
                        "-Xclang",
                        "-fno-cuda-host-device-constexpr",
                        "-D___HGGC_CG_HELPER_H___",
                        "-D__CUDACC_RDC__",
                        str(source),
                        "-o",
                        str(bitcode),
                    ],
                    check=True,
                    capture_output=True,
                )
            except (OSError, subprocess.CalledProcessError):
                return None
            path = cache.put(bitcode.read_bytes(), name, binary=True)
    return path


def clear_counters(counters, memset):
    stream = torch.cuda.current_stream(counters.device)
    status = memset(
        counters.data_ptr(),
        0,
        counters.numel() * counters.element_size(),
        stream.cuda_stream,
    )
    if status:
        raise RuntimeError(
            f"MegaMHC counter initialization failed with status {status}"
        )
