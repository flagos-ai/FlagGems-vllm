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

"""Standalone, non-Graph public-operator benchmark.

Run with PYTHONPATH=src python benchmark/test_fused_indexer_k_norm.py, or pytest
-q -s benchmark/test_fused_indexer_k_norm.py. INDEXER_BENCH_OUTPUT selects the
JSON destination (default: fused_indexer_k_norm_benchmark.json). Framework timer,
--iter, --warmup and --mode options do not change this dedicated protocol.
"""

import ast
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import statistics
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn

SEED = 20261106
EPS = 1e-6
WARMUP = 10
REPEAT = 50
BATCHES = 3
SOURCE_COMMIT = "2eae5ee9be3c6d01b008eb57d421545666ff8072"
NATIVE_MODULE = "vllm.models.glm5next.common.attention"
TOLERANCES = {
    torch.float32: (2e-5, 2e-5),
    torch.float16: (2e-3, 2e-3),
    torch.bfloat16: (1.6e-2, 1.6e-2),
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(root, *args):
    proc = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _identity(candidate):
    root = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__).resolve(),
        Path(inspect.getfile(inspect.unwrap(candidate))).resolve(),
        root / "src/flaggems_vllm/ops/__init__.py",
        root / "src/flaggems_vllm/__init__.py",
        root / "src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml",
    ]
    backend_config = (
        root
        / "src/flaggems_vllm/runtime/backend"
        / ("_" + flaggems_vllm.vendor_name)
        / "tune_configs.yaml"
    )
    files.append(backend_config)
    info = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": _version("triton"),
        "flagtree": _version("flagtree"),
        "vllm": _version("vllm"),
        "torch_npu": _version("torch-npu"),
        "torch_cuda_build": getattr(torch.version, "cuda", None),
        "torch_hip_build": getattr(torch.version, "hip", None),
        "device": str(flaggems_vllm.device),
        "vendor": flaggems_vllm.vendor_name,
        "package_file": flaggems_vllm.__file__,
        "candidate_module": candidate.__module__,
        "git_head": _git(root, "rev-parse", "HEAD"),
        "git_status": _git(root, "status", "--short"),
        "source_sha256": {str(p): _sha(p) for p in set(files) if p.is_file()},
        "VLLM_PLUGINS": os.environ.get("VLLM_PLUGINS"),
        "DNN_VENDOR": os.environ.get("DNN_VENDOR"),
        "frozen_vllm_source_commit": SOURCE_COMMIT,
    }
    if hasattr(torch_device_fn, "current_device"):
        info["device_index"] = torch_device_fn.current_device()
    if hasattr(torch_device_fn, "get_device_name"):
        info["device_name"] = torch_device_fn.get_device_name()
    return info


def _disable_compiler_graphs():
    # Disable optional Inductor graph capture as well as avoiding capture here.
    try:
        config = importlib.import_module("torch._inductor.config")
        config.triton.cudagraphs = False
        return "torch._inductor.config.triton.cudagraphs=False"
    except (ImportError, AttributeError) as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def _native_leaf(candidate):
    """Import the installed leaf, never reconstruct a compiled stand-in."""
    info = {"status": "N/A", "module": NATIVE_MODULE}
    try:
        module = importlib.import_module(NATIVE_MODULE)
        native = getattr(module, "_fused_indexer_k_norm")
        original = inspect.unwrap(native)
        source = textwrap.dedent(inspect.getsource(original))
        info.update(
            {
                "file": inspect.getfile(original),
                "file_sha256": _sha(inspect.getfile(original)),
                "callable_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "original_module": original.__module__,
                "compiled_wrapper": hasattr(native, "_torchdynamo_orig_callable"),
            }
        )
        expected = ast.parse(
            "F.layer_norm(x.float(), (dim,), weight, bias, eps).type_as(x)",
            mode="eval",
        ).body
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
        if (
            native is candidate
            or original is inspect.unwrap(candidate)
            or original.__module__ != NATIVE_MODULE
            or not info["compiled_wrapper"]
            or len(functions) != 1
            or len(functions[0].body) != 1
            or len(returns) != 1
            or ast.dump(returns[0].value) != ast.dump(expected)
            or original.__globals__.get("F") is not F
            or F.layer_norm.__module__ != "torch.nn.functional"
        ):
            raise ValueError(
                "native callable identity/formula/compiled-wrapper mismatch"
            )
        info["status"] = "imported_pending_per_case_numerical_qualification"
        return native, info
    except Exception as exc:
        info["reason"] = f"{type(exc).__name__}: {exc}"
        return None, info


def _oracle(x_cpu, weight_cpu, bias_cpu):
    # Start from actual stored input, explicitly honoring the source FP32 cast.
    values = x_cpu.float().double()
    centered = values - values.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    normalized = centered / torch.sqrt(variance + EPS)
    return (normalized * weight_cpu.double() + bias_cpu.double()).to(x_cpu.dtype)


def _check(fn, args, expected):
    actual = fn(*args)
    x = args[0]
    if actual.shape != x.shape or actual.dtype != x.dtype or actual.device != x.device:
        raise AssertionError("output shape/dtype/device mismatch")
    if not actual.is_contiguous():
        raise AssertionError("output must be contiguous")
    host = actual.detach().cpu()
    rtol, atol = TOLERANCES[x.dtype]
    torch.testing.assert_close(host, expected, rtol=rtol, atol=atol)
    error = host.double() - expected.double()
    return {
        "max_abs": error.abs().max().item(),
        "rms": error.square().mean().sqrt().item(),
        "rtol": rtol,
        "atol": atol,
    }


def _decomposed(x, weight, bias, dim, eps):
    return F.layer_norm(x.float(), (dim,), weight, bias, eps).type_as(x)


def _cases():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for m in (1, 8, 128, 1024, 4096):
            for gap in (0, 16):
                yield m, 128, dtype, gap, "core"
    yield 128, 127, torch.float32, 0, "tail"
    yield 128, 129, torch.bfloat16, 16, "tail"


def _inputs(m, d, dtype, gap, device, index):
    gen = torch.Generator(device="cpu").manual_seed(SEED + index)
    # Bounded reproducible values avoid FP32 overflow in this performance set.
    parent_cpu = (torch.rand((m, d + gap), generator=gen) * 6 - 3).to(dtype)
    weight_cpu = torch.rand((d,), generator=gen) * 3 - 1.5
    bias_cpu = torch.rand((d,), generator=gen) * 3 - 1.5
    expected = _oracle(parent_cpu[:, :d], weight_cpu, bias_cpu)
    parent = parent_cpu.to(device=device)
    weight = weight_cpu.to(device=device)
    bias = bias_cpu.to(device=device)
    args = (parent[:, :d], weight, bias, d, EPS)
    return args, expected, (parent, weight, bias), (parent_cpu, weight_cpu, bias_cpu)


def _unchanged(device_inputs, cpu_inputs):
    for actual, expected in zip(device_inputs, cpu_inputs):
        torch.testing.assert_close(actual.detach().cpu(), expected, rtol=0, atol=0)


def _batch_us(fn, args):
    for _ in range(WARMUP):
        fn(*args)
    torch_device_fn.synchronize()
    begin = time.perf_counter_ns()
    for _ in range(REPEAT):
        fn(*args)
    torch_device_fn.synchronize()
    return (time.perf_counter_ns() - begin) / (1000 * REPEAT)


def _summary(samples):
    return {
        "batch_us": samples,
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def _write(report):
    path = Path(
        os.environ.get("INDEXER_BENCH_OUTPUT", "fused_indexer_k_norm_benchmark.json")
    ).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


@torch.no_grad()
def run_benchmark():
    candidate = flaggems_vllm.fused_indexer_k_norm
    if not torch_device_fn.is_available():
        raise RuntimeError("accelerator unavailable; no benchmark rows executed")
    compiler_graphs = _disable_compiler_graphs()
    native, native_info = _native_leaf(candidate)
    if native is not None and compiler_graphs.startswith("unavailable"):
        native = None
        native_info.update(status="N/A", reason=compiler_graphs)
    report = {
        "schema_version": 1,
        "status": "RUNNING",
        "identity": _identity(candidate),
        "native_identity": native_info,
        "protocol": {
            "unit": "us",
            "timer": "synchronized_wall_clock_batch_mean",
            "scope": "public_operator_including_allocation_and_python_dispatch",
            "graph": False,
            "compiler_graphs": compiler_graphs,
            "warmup_per_batch": WARMUP,
            "calls_per_batch": REPEAT,
            "batches": BATCHES,
            "aggregation": "median/min/max of three batch means",
            "seed": SEED,
            "eps": EPS,
            "expected_core_rows": 30,
            "expected_tail_rows": 2,
            "sync": "runtime.torch_device_fn.synchronize before/after timed batch",
            "order": "rotate qualified methods by batch to reduce order bias",
            "baseline_qualification": "source identity plus CPU FP64 before/after",
            "normalization_factor": None,
        },
        "rows": [],
    }
    print(json.dumps({"identity": report["identity"], "native": native_info}))
    try:
        for index, (m, d, dtype, gap, group) in enumerate(_cases()):
            args, expected, device_inputs, cpu_inputs = _inputs(
                m, d, dtype, gap, flaggems_vllm.device, index
            )
            row = {
                "M": m,
                "D": d,
                "dtype": str(dtype),
                "row_gap": gap,
                "x_stride": list(args[0].stride()),
                "group": group,
                "methods": {},
            }
            report["rows"].append(row)
            # Candidate exceptions are fatal and are never baseline N/A.
            row["methods"]["candidate"] = {
                "status": "qualified",
                "accuracy": "PASS",
                "precheck": _check(candidate, args, expected),
            }
            _unchanged(device_inputs, cpu_inputs)
            methods = {"candidate": candidate, "decomposed": _decomposed}
            if native is not None:
                methods["native"] = native
            else:
                row["methods"]["native"] = {
                    "status": "N/A",
                    "reason": native_info["reason"],
                }
            for name in list(methods):
                if name == "candidate":
                    continue
                try:
                    metrics = _check(methods[name], args, expected)
                    _unchanged(device_inputs, cpu_inputs)
                    row["methods"][name] = {
                        "status": "qualified",
                        "accuracy": "PASS",
                        "precheck": metrics,
                    }
                except Exception as exc:
                    row["methods"][name] = {
                        "status": "N/A",
                        "accuracy": "FAIL",
                        "performance": "N/A",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                    del methods[name]
                    torch_device_fn.synchronize()
                    _check(candidate, args, expected)
                    _unchanged(device_inputs, cpu_inputs)
            samples = {name: [] for name in methods}
            names = list(methods)
            for batch in range(BATCHES):
                order = names[batch % len(names) :] + names[: batch % len(names)]
                for name in order:
                    if name not in methods:
                        continue
                    if name == "candidate":
                        samples[name].append(_batch_us(methods[name], args))
                    else:
                        try:
                            samples[name].append(_batch_us(methods[name], args))
                        except Exception as exc:
                            row["methods"][name].update(
                                status="N/A",
                                performance="N/A",
                                reason=f"{type(exc).__name__}: {exc}",
                            )
                            del methods[name]
                            torch_device_fn.synchronize()
                            _check(candidate, args, expected)
                            _unchanged(device_inputs, cpu_inputs)
            for name, fn in methods.items():
                if name == "candidate":
                    metrics = _check(fn, args, expected)
                else:
                    try:
                        metrics = _check(fn, args, expected)
                    except Exception as exc:
                        row["methods"][name].update(
                            status="N/A",
                            accuracy="FAIL",
                            performance="N/A",
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                        torch_device_fn.synchronize()
                        _check(candidate, args, expected)
                        continue
                row["methods"][name].update(
                    status="PASS",
                    accuracy="PASS",
                    performance="PASS",
                    postcheck=metrics,
                    **_summary(samples[name]),
                )
            _unchanged(device_inputs, cpu_inputs)
            candidate_us = row["methods"]["candidate"]["median_us"]
            for name in ("native", "decomposed"):
                baseline = row["methods"][name]
                row["S_" + name] = (
                    baseline["median_us"] / candidate_us
                    if baseline["status"] == "PASS"
                    else None
                )
            row["native_below_0_9"] = (
                row["S_native"] < 0.9 if row["S_native"] is not None else None
            )
            print(
                f"{group} M={m} D={d} {dtype} gap={gap}: "
                f"candidate={candidate_us:.3f}us "
                f"S_native={row['S_native']} S_decomposed={row['S_decomposed']}"
            )
            _write(report)
        report["summary"] = {
            "executed_rows": len(report["rows"]),
            "valid_rows": {
                name: sum(
                    row["methods"][name]["status"] == "PASS" for row in report["rows"]
                )
                for name in ("candidate", "native", "decomposed")
            },
            "core_native_below_0_9": [
                index
                for index, row in enumerate(report["rows"])
                if row["group"] == "core" and row["native_below_0_9"]
            ],
        }
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        print(f"INDEXER_BENCH_OUTPUT={_write(report)} status={report['status']}")
    return report


@pytest.mark.fused_indexer_k_norm
def test_fused_indexer_k_norm_benchmark():
    report = run_benchmark()
    assert report["status"] == "PASS"
    assert len(report["rows"]) == 32


if __name__ == "__main__":
    run_benchmark()
