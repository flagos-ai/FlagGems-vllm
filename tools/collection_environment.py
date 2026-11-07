"""Best-effort provenance for a baseline run; probes never require extra packages."""

import importlib
import json
import os
import platform
import subprocess
import uuid
from importlib import metadata


def _command(args, cwd=None):
    try:
        result = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _version(name):
    # NVIDIA images may provide triton via pytorch-triton or flagtree. The
    # imported module identifies the code actually used, not the wrapper package.
    if name == "triton":
        try:
            version = getattr(importlib.import_module(name), "__version__", None)
            if version:
                return str(version)
        except Exception:
            pass
    try:
        return metadata.version(name)
    except Exception:
        return None


def _driver_version(device_uuid):
    if device_uuid:
        # Some Torch builds return the bare UUID; nvidia-smi requires GPU-.
        try:
            device_uuid = "GPU-" + str(uuid.UUID(device_uuid))
        except (ValueError, AttributeError):
            pass  # Preserve GPU-/MIG- identifiers verbatim.
        value = _command(
            [
                "nvidia-smi",
                "--id",
                device_uuid,
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        )
        if value:
            return value
    # Driver version is shared by the host's NVIDIA devices. Only use an
    # unambiguous result, also covering Torch without UUID support and MIG.
    values = _command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    versions = {v.strip() for v in (values or "").splitlines() if v.strip()}
    return next(iter(versions)) if len(versions) == 1 else None


def _cpu_info():
    fields = {}
    try:
        data = json.loads(_command(["env", "LC_ALL=C", "lscpu", "--json"]) or "{}")

        def visit(rows):
            for row in rows:
                fields[row["field"].rstrip(":")] = row.get("data")
                visit(row.get("children", []))

        visit(data.get("lscpu", []))
    except (ValueError, TypeError, KeyError, AttributeError):
        pass

    def integer(key):
        try:
            return int(fields[key])
        except (KeyError, ValueError, TypeError):
            return None

    sockets, cores = integer("Socket(s)"), integer("Core(s) per socket")
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_count = None
    return {
        "cpu_model": fields.get("Model name"),
        "cpu_architecture": platform.machine(),
        "cpu_socket_count": sockets,
        "physical_cpu_core_count": sockets * cores if sockets and cores else None,
        "logical_cpu_count": os.cpu_count(),
        "cpu_threads_per_core": integer("Thread(s) per core"),
        "numa_node_count": integer("NUMA node(s)"),
        "process_cpu_affinity_count": affinity_count,
        "hypervisor_vendor": fields.get("Hypervisor vendor"),
    }


def collect_environment(torch, run):
    """Record the selected CUDA device, separately from the configured reference chip.

    Missing tools/optional dependencies are represented by null. Only explicitly
    selected environment variables are recorded, never the full process environment.
    """
    gpu = {
        "name": None,
        "uuid": None,
        "total_memory_bytes": None,
        "total_memory_gib": None,
        "compute_capability": None,
        "multiprocessor_count": None,
        "cuda_device_index": None,
        "driver_version": None,
    }
    try:
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        gpu.update(
            name=props.name,
            total_memory_bytes=props.total_memory,
            total_memory_gib=round(props.total_memory / (1024**3), 2),
            compute_capability=f"{props.major}.{props.minor}",
            multiprocessor_count=props.multi_processor_count,
            cuda_device_index=index,
        )
        uuid = getattr(props, "uuid", None)
        if uuid is not None:
            gpu["uuid"] = str(uuid)
    except Exception:
        pass
    # UUID selection respects CUDA_VISIBLE_DEVICES remapping. Never use a CUDA
    # logical ordinal as an nvidia-smi physical index (including on MIG systems).
    gpu["driver_version"] = _driver_version(gpu["uuid"])
    return {
        "hardware": {"gpu": gpu, **_cpu_info()},
        "software": {
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "packages": {
                name: _version(name) for name in ("torch", "triton", "vllm", "sglang")
            },
            "torch_cuda_version": getattr(
                getattr(torch, "version", None), "cuda", None
            ),
            "nvcc_version": _command(["nvcc", "--version"]),
            "ncu_version": _command(["ncu", "--version"]),
        },
        "run": json.loads(json.dumps(run)),
        "environment_variables": {
            name: os.environ.get(name)
            for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
        },
    }
