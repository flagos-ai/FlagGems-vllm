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

"""Small runtime shim shared by the production-shape MegaMoE candidate."""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Union

PathLike = Union[str, Path]


def _cuda_root(candidate: PathLike, source: str) -> Path:
    root = Path(candidate).expanduser().resolve()
    if not (root / "bin" / "nvcc").is_file():
        raise RuntimeError(f"{source} does not contain bin/nvcc: {root}")
    return root


def _find_cuda_home(explicit: Optional[PathLike] = None) -> Path:
    """Resolve a CUDA toolkit without modifying the process environment."""

    if explicit is not None:
        return _cuda_root(explicit, "CUDA toolkit path")

    for variable in ("CUDA_HOME", "CUDA_PATH"):
        override = os.environ.get(variable)
        if override:
            return _cuda_root(override, variable)

    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        return _cuda_root(Path(nvcc).resolve().parent.parent, "nvcc on PATH")

    try:
        torch_cpp_extension = importlib.import_module("torch.utils.cpp_extension")
        torch_cuda_home = getattr(torch_cpp_extension, "CUDA_HOME", None)
    except (ImportError, OSError):
        torch_cuda_home = None
    if torch_cuda_home:
        return _cuda_root(torch_cuda_home, "torch CUDA_HOME")

    common_roots = [Path("/usr/local/cuda"), Path("/opt/cuda")]
    for parent in (Path("/usr/local"), Path("/opt")):
        if parent.is_dir():
            common_roots.extend(sorted(parent.glob("cuda-*"), reverse=True))
    for root in common_roots:
        if (root / "bin" / "nvcc").is_file():
            return root.resolve()

    raise RuntimeError(
        "CUDA toolkit not found; pass cuda_home, set CUDA_HOME/CUDA_PATH, "
        "or put nvcc on PATH"
    )


def _nvshmem_root(candidate: PathLike, source: str) -> Path:
    root = Path(candidate).expanduser().resolve()
    missing = [name for name in ("include", "lib") if not (root / name).is_dir()]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"{source} is missing {missing_text}: {root}")
    return root


def _find_nvshmem_home(explicit: Optional[PathLike] = None) -> Path:
    """Resolve the NVSHMEM include/lib root without changing ``os.environ``."""

    if explicit is not None:
        return _nvshmem_root(explicit, "NVSHMEM installation")

    override = os.environ.get("NVSHMEM_HOME")
    if override:
        return _nvshmem_root(override, "NVSHMEM_HOME")

    try:
        nvshmem_package = importlib.import_module("nvidia.nvshmem")
    except ImportError as error:
        message = (
            "NVSHMEM installation not found; pass nvshmem_home or set NVSHMEM_HOME"
        )
        raise RuntimeError(message) from error

    package_roots = getattr(nvshmem_package, "__path__", ())
    for package_root in package_roots:
        try:
            return _nvshmem_root(package_root, "nvidia.nvshmem package")
        except RuntimeError:
            continue
    raise RuntimeError(
        "nvidia.nvshmem does not contain the required include/lib directories"
    )


def _import_env() -> dict[str, Any]:
    import torch
    import triton
    import triton.experimental.tle.language as tle
    import triton.language as tl

    return {"torch": torch, "triton": triton, "tl": tl, "tle": tle}


def _default_build_dir() -> Path:
    # Keep this import lazy. The fallback loads the same helper by file path so
    # direct execution of ``kernel.py`` need not import the package root.
    if __package__:
        from flaggems_vllm.utils.code_cache import code_cache_dir
    else:
        module_path = Path(__file__).resolve().parents[6] / "utils" / "code_cache.py"
        spec = importlib.util.spec_from_file_location(
            "_flaggems_vllm_code_cache", module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load FlagGems-vLLM code cache: {module_path}")
        code_cache = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(code_cache)
        code_cache_dir = code_cache.code_cache_dir

    return code_cache_dir() / "megamoe"


def _compile_nvshmem_host_so(
    host_src: Path,
    *,
    cuda_home: Optional[PathLike] = None,
    nvshmem_home: Optional[PathLike] = None,
    build_dir: Optional[PathLike] = None,
) -> Path:
    """Build the tiny host-side NVSHMEM wrapper once across all MPI ranks."""

    host_src = Path(host_src).expanduser().resolve()
    if not host_src.is_file():
        raise FileNotFoundError(f"NVSHMEM host source does not exist: {host_src}")

    cuda_root = _find_cuda_home(cuda_home)
    nvshmem_root = _find_nvshmem_home(nvshmem_home)
    output_dir = (
        Path(build_dir).expanduser().resolve()
        if build_dir is not None
        else _default_build_dir()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_key = hashlib.sha256()
    cache_key.update(host_src.read_bytes())
    cache_key.update(b"sm_90a|-shared|-fPIC|-rdc=true|-lnvshmem_host|-lnvshmem_device")
    nvshmem_lib = nvshmem_root / "lib"
    dependencies = {cuda_root / "bin" / "nvcc"}
    for pattern in ("libnvshmem_host.so*", "libnvshmem_device.*"):
        dependencies.update(path.resolve() for path in nvshmem_lib.glob(pattern))
    for dependency in sorted(dependencies, key=str):
        cache_key.update(str(dependency).encode())
        if dependency.exists():
            stat = dependency.stat()
            cache_key.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    digest = cache_key.hexdigest()[:16]
    so_path = output_dir / f"nvshmem_host_sm90a_{digest}.so"
    lock_path = so_path.with_suffix(".so.lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if so_path.exists():
            return so_path

        temporary_path = so_path.with_name(f"{so_path.name}.{os.getpid()}.tmp")
        cmd = [
            str(cuda_root / "bin" / "nvcc"),
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-rdc=true",
            "-arch=sm_90a",
            f"-I{nvshmem_root / 'include'}",
            f"-L{nvshmem_lib}",
            "-lnvshmem_host",
            "-lnvshmem_device",
            "-Xlinker",
            "--enable-new-dtags",
            "-Xlinker",
            "-rpath",
            "-Xlinker",
            str(nvshmem_lib),
            "-o",
            str(temporary_path),
            str(host_src),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            temporary_path.replace(so_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return so_path
