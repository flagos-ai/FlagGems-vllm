# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Load the C550 cooperative_topk kernels from the installed MACA SDK."""

import ctypes
import fcntl
import hashlib
import os
import platform
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from flaggems_vllm import runtime
from flaggems_vllm.ops.cooperative_topk import _cooperative_topk_impl
from flaggems_vllm.utils.code_cache import code_cache_dir


@lru_cache(maxsize=None)
def _is_c550(device):
    return runtime.torch_device_fn.get_device_name(device) == "MetaX C550"


@lru_cache(maxsize=1)
def _load_c550_topk():
    sdk = Path(os.environ.get("MACA_PATH", "/opt/maca")).resolve()
    compiler = sdk / "mxgpu_llvm/bin/mxcc"
    arch_tool = sdk / "bin/__metaxgpu_arch"
    if not all(
        tool.is_file() and os.access(tool, os.X_OK) for tool in (compiler, arch_tool)
    ):
        return None

    source = Path(__file__).with_suffix(".cu")
    # The SDK's target name for the same C550 differs between MACA releases.
    # Hash the compiler, SDK and visible targets as well as the kernel source.
    key = hashlib.sha256(source.read_bytes())
    key.update(platform.machine().encode())
    key.update(str(compiler).encode())
    try:
        key.update(subprocess.check_output([str(compiler), "--version"]))
        key.update(subprocess.check_output([str(arch_tool)]))
    except (OSError, subprocess.CalledProcessError):
        # An incomplete SDK installation can still have a usable Triton backend.
        return None
    version = sdk / "Version.txt"
    if version.is_file():
        key.update(version.read_bytes())

    directory = code_cache_dir() / "metax_cooperative_topk" / key.hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "cooperative_topk.so"
    with (directory / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.is_file():
            with tempfile.TemporaryDirectory(dir=directory) as tmp:
                built = Path(tmp) / library.name
                command = [
                    str(compiler),
                    "-x",
                    "maca",
                    "-offload-arch",
                    "native",
                    "-O3",
                    "-std=c++17",
                    "-fPIC",
                    "-shared",
                    f"--maca-path={sdk}",
                    str(source),
                    "-o",
                    str(built),
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(
                        f"MACA cooperative_topk build failed\n{result.stderr}"
                    )
                built.replace(library)

    launch = ctypes.CDLL(str(library)).flaggems_c550_cooperative_topk
    launch.argtypes = [ctypes.c_void_p] * 4 + [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    launch.restype = ctypes.c_char_p
    return launch


def _try_c550_topk(logits, lengths, output, workspace, k):
    rows, width = logits.shape
    if width <= k or width > 1048576:
        return False
    if not _is_c550(logits.device):
        return False
    launch = _load_c550_topk()
    if launch is None:
        return False
    error = launch(
        logits.data_ptr(),
        lengths.data_ptr(),
        output.data_ptr(),
        workspace.data_ptr(),
        rows,
        width,
        logits.stride(0),
        k,
        runtime.torch_device_fn.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"MACA cooperative_topk launch failed: {error.decode()}")
    return True


def cooperative_topk(logits, lengths, output, workspace, k, max_seq_len):
    return _cooperative_topk_impl(
        logits,
        lengths,
        output,
        workspace,
        k,
        max_seq_len,
        try_specialized=_try_c550_topk,
    )
