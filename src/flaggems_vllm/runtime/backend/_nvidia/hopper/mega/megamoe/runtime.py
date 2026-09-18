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
import os
import site
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_mega_site_override = os.environ.get("MEGAMOE_TORCH_SITE_PACKAGES")
MEGA_SITE = Path(_mega_site_override).resolve() if _mega_site_override else None
if MEGA_SITE is not None:
    site.addsitedir(str(MEGA_SITE))


def _find_nvshmem_home() -> Path:
    override = os.environ.get("NVSHMEM_HOME")
    if override:
        return Path(override).expanduser().resolve()

    site_roots = []
    if MEGA_SITE is not None:
        site_roots.append(MEGA_SITE)
    site_roots.extend(Path(path) for path in site.getsitepackages())
    site_roots.append(Path(site.getusersitepackages()))
    site_roots.append(
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    for root in site_roots:
        candidate = root / "nvidia" / "nvshmem"
        if candidate.is_dir():
            return candidate.resolve()
    raise RuntimeError(
        "NVSHMEM installation not found; set NVSHMEM_HOME to its include/lib root"
    )


NVSHMEM_HOME = _find_nvshmem_home()

cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8")
os.environ.setdefault("CUDA_HOME", cuda_home)
os.environ["CPATH"] = f"{cuda_home}/targets/x86_64-linux/include:" + os.environ.get(
    "CPATH", ""
)
os.environ["LD_LIBRARY_PATH"] = (
    f"{NVSHMEM_HOME / 'lib'}:{cuda_home}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
)


def _import_env() -> dict[str, Any]:
    import torch
    import triton
    import triton.experimental.tle.language as tle
    import triton.language as tl

    return {"torch": torch, "triton": triton, "tl": tl, "tle": tle}


def _compile_nvshmem_host_so(host_src: Path) -> Path:
    """Build the tiny host-side NVSHMEM wrapper once across all MPI ranks."""

    default_cache = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "flaggems_vllm"
        / "megamoe"
    )
    build_dir = Path(os.environ.get("MEGAMOE_BUILD_DIR", default_cache))
    build_dir.mkdir(parents=True, exist_ok=True)
    so_path = build_dir / "nvshmem_host_sm90a.so"
    lock_path = so_path.with_suffix(".so.lock")
    with lock_path.open("w") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                break
            except BlockingIOError:
                time.sleep(0.1)
        if (
            so_path.exists()
            and so_path.stat().st_mtime_ns >= host_src.stat().st_mtime_ns
        ):
            return so_path
        cmd = [
            str(Path(cuda_home) / "bin" / "nvcc"),
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-rdc=true",
            "-arch=sm_90a",
            f"-I{NVSHMEM_HOME / 'include'}",
            f"-L{NVSHMEM_HOME / 'lib'}",
            "-lnvshmem_host",
            "-lnvshmem_device",
            "-Xlinker",
            "-rpath",
            "-Xlinker",
            str(NVSHMEM_HOME / "lib"),
            "-o",
            str(so_path),
            str(host_src),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    return so_path
