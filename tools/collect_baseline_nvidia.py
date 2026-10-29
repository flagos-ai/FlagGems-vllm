#!/usr/bin/env python3

import argparse
import csv
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton

try:
    from .collection_process import run_with_cleanup
except ImportError:
    from collection_process import run_with_cleanup

try:
    from .collection_environment import collect_environment
except ImportError:
    from collection_environment import collect_environment

# NCU metric names (verified on H800/Hopper; re-check with `ncu --query-metrics`
# before switching architectures).

# CUDA Core work F_cuda: FP instruction counts; ffma/hfma count as 2 FLOP each.
FLOP_CUDA_CORE = {
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 2,
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum": 1,
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum": 2,
}

# Tensor Core work F_tensor: counter name depends on source dtype (0 if unknown).
FLOP_TENSOR_CORE = {
    "torch.float16": "sm__ops_path_tensor_src_fp16.sum",
    "torch.bfloat16": "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
    "torch.float8_e4m3fn": "sm__ops_path_tensor_src_fp8.sum",
    "torch.float8_e5m2": "sm__ops_path_tensor_src_fp8.sum",
    "torch.float64": "sm__ops_path_tensor_src_fp64.sum",
}

# Memory bytes, per-unit utilizations, and overall SM throughput (% of peak).
MEMORY_BYTES = "dram__bytes.sum"
UTIL = {
    "cuda_core": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tensor_core": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "memory_bw": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
}
# Per-kernel GPU duration: only used to split end-to-end T across kernels by ratio.
DURATION = "gpu__time_duration.sum"
KERNEL_NAME_COL = "Kernel Name"
# Byte-unit -> base-byte multiplier (ncu uses decimal: 1 Kbyte = 1000 byte).
_BYTE_UNIT_MULT = {
    "byte": 1.0,
    "Kbyte": 1e3,
    "Mbyte": 1e6,
    "Gbyte": 1e9,
    "Tbyte": 1e12,
}

# One Python module per operator exposing native()/grid()/build_inputs()/
# key_shape() (contract in baseline/__init__.py). Operators live in baseline/.
OPS_DIR = Path(__file__).resolve().parent.parent / "baseline"

# Vendor chip specs; used to derive each chip's scaling factor relative to the
# reference chip (H800 by default) for cross-chip normalization.
HARDWARE_SPECS_PATH = Path(__file__).with_name("hardware_specs.json")
DEFAULT_REFERENCE_CHIP = "H800"
# vendor_name in hardware_specs.json is already the runtime vendor_name
# (lowercase, matches flaggems_vllm).
_TENSOR_DTYPES = ("bf16", "fp16", "fp32", "tf32", "int8", "fp64", "fp8")
_VECTOR_DTYPES = ("bf16", "fp16", "fp32", "int8", "fp64", "fp8")


def _empty_result():
    return {
        "F_cuda": 0.0,
        "F_tensor": 0.0,
        "B_mem": 0.0,
        "U_cuda": 0.0,
        "U_tensor": 0.0,
        "U_mem": 0.0,
        "U_bottle_neck": 0.0,
        "bottle_neck_unit": None,
        "num_kernels": 0,
        "per_kernel": [],
    }


def _build_profile_script_ops(ops_dir, op_module, binding, dtype_str, warmup):
    """Build a self-contained temp script run by the NCU subprocess.

    It imports the operator module from ops_dir, rebuilds inputs via build_inputs,
    warms up, then wraps the single target call in the "kiro_target" NVTX range so
    NCU can pin exactly its kernels. Inputs are rebuilt before the sampled call so
    in-place operators are not polluted by warmup accumulation.
    """
    return f"""import sys
import torch

sys.path.insert(0, {str(ops_dir)!r})
import {op_module} as opmod

dtype = {dtype_str}
binding = {binding!r}
op = opmod.native()
if op is None:
    raise RuntimeError("native resolution failed: {op_module}")


def _make_args():
    return opmod.build_inputs(binding, dtype, "cuda")


args, kwargs = _make_args()
for _ in range({warmup}):
    op(*args, **kwargs)
torch.cuda.synchronize()
args, kwargs = _make_args()
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("kiro_target")
op(*args, **kwargs)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
"""


def _parse_ncu_csv(csv_text, wanted):
    rows = [r for r in csv.reader(io.StringIO(csv_text)) if r]
    if not rows:
        return {}
    header = rows[0]
    col = {m: header.index(m) for m in wanted if m in header}
    name_idx = header.index(KERNEL_NAME_COL) if KERNEL_NAME_COL in header else None

    mult = {m: 1.0 for m in col}
    data_start = 1
    if len(rows) > 1 and name_idx is not None:
        unit_row = rows[1]
        unit_name = unit_row[name_idx].strip() if name_idx < len(unit_row) else ""
        if not unit_name:
            for m, idx in col.items():
                unit = unit_row[idx].strip() if idx < len(unit_row) else ""
                mult[m] = _BYTE_UNIT_MULT.get(unit, 1.0)
            data_start = 2

    out = {m: [] for m in col}
    names = []
    for row in rows[data_start:]:
        # Skip non-kernel rows with an empty Kernel Name.
        if name_idx is not None:
            name = row[name_idx].strip() if name_idx < len(row) else ""
            if not name:
                continue
        else:
            name = ""
        # Non-numeric cells count as 0 to keep each metric aligned with kernel rows.
        for m, idx in col.items():
            val = 0.0
            if idx < len(row):
                cell = row[idx].replace(",", "").strip()
                try:
                    val = float(cell) * mult[m]
                except ValueError:
                    val = 0.0
            out[m].append(val)
        names.append(name)
    out[KERNEL_NAME_COL] = names
    return out


def _build_per_kernel(agg, tensor_metric):
    names = agg.get(KERNEL_NAME_COL, [])
    n = len(names)
    if n == 0:
        return []

    def col(m):
        vals = agg.get(m, [])
        return [vals[i] if i < len(vals) else 0.0 for i in range(n)]

    dur = col(DURATION)
    dur_sum = sum(dur)
    cuda_cols = {m: col(m) for m in FLOP_CUDA_CORE}
    tensor_col = col(tensor_metric) if tensor_metric else [0.0] * n
    mem_col = col(MEMORY_BYTES)
    ucuda_col = col(UTIL["cuda_core"])
    utensor_col = col(UTIL["tensor_core"])
    umem_col = col(UTIL["memory_bw"])

    per_kernel = []
    for i in range(n):
        f_cuda = sum(w * cuda_cols[m][i] for m, w in FLOP_CUDA_CORE.items())
        per_kernel.append(
            {
                "name": names[i],
                "F_cuda": f_cuda,
                "F_tensor": tensor_col[i],
                "B_mem": mem_col[i],
                "U_cuda": ucuda_col[i],  # already % of peak (0-100)
                "U_tensor": utensor_col[i],
                "U_mem": umem_col[i],
                "dur_frac": (dur[i] / dur_sum) if dur_sum else 0.0,
            }
        )
    return per_kernel


def profile_with_ncu(op_name, profile_script, shape, dtype_str, report_dir=None):
    tensor_metric = FLOP_TENSOR_CORE.get(dtype_str)
    metrics = list(FLOP_CUDA_CORE) + [MEMORY_BYTES, *UTIL.values(), DURATION]
    if tensor_metric:
        metrics.append(tensor_metric)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(profile_script)
        script_path = f.name

    report_dir = Path(report_dir) if report_dir else Path(tempfile.gettempdir())
    report_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{op_name}_{'x'.join(map(str, shape))}_{dtype_str.split('.')[-1]}"
    report_base = report_dir / tag

    def _err_msg(cp):
        return ((cp.stderr or "").strip() or (cp.stdout or "").strip())[:300]

    try:
        export_cmd = [
            "ncu",
            "--metrics",
            ",".join(metrics),
            "--nvtx",
            "--nvtx-include",
            "kiro_target/",
            "--force-overwrite",
            "--export",
            str(report_base),
            sys.executable,
            script_path,
        ]
        try:
            proc = run_with_cleanup(
                export_cmd, capture_output=True, text=True, timeout=600
            )
        except FileNotFoundError:
            print("  ✗ ncu command not found, skipping NCU profiling")
            return _empty_result()
        except subprocess.TimeoutExpired:
            print(f"  ✗ NCU timed out: {shape} {dtype_str}")
            return _empty_result()
        if proc.returncode != 0:
            print(
                f"  ✗ NCU profiling failed (code={proc.returncode}): "
                f"{_err_msg(proc)}"
            )
            return _empty_result()

        candidates = sorted(
            report_dir.glob(f"{tag}.ncu-rep*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"  ✗ NCU produced no report file: {report_base}.ncu-rep[z]")
            return _empty_result()
        report_path = str(candidates[0])

        imp = run_with_cleanup(
            ["ncu", "--import", report_path, "--csv", "--page", "raw"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if imp.returncode != 0:
            print(f"  ✗ NCU import failed (code={imp.returncode}): " f"{_err_msg(imp)}")
            return _empty_result()

        agg = _parse_ncu_csv(imp.stdout, metrics)

        def total(m):
            return sum(agg.get(m, []))

        def mean_pct(m):
            vals = agg.get(m, [])
            return sum(vals) / len(vals) / 100.0 if vals else 0.0

        flops_cuda = sum(w * total(m) for m, w in FLOP_CUDA_CORE.items())
        flops_tensor = total(tensor_metric) if tensor_metric else 0.0
        mem_bytes = total(MEMORY_BYTES)
        util = {dim: mean_pct(name) for dim, name in UTIL.items()}
        bottleneck = max(util, key=util.get)
        if util[bottleneck] == 0.0:  # all three zero (parse failure / metric mismatch)
            bottleneck = None

        _unit = {"cuda_core": "cuda", "tensor_core": "tensor", "memory_bw": "mem"}
        U_cuda = util["cuda_core"] * 100.0
        U_tensor = util["tensor_core"] * 100.0
        U_mem = util["memory_bw"] * 100.0
        U_bottle = util[bottleneck] * 100.0 if bottleneck else 0.0

        per_kernel = _build_per_kernel(agg, tensor_metric)

        return {
            "F_cuda": flops_cuda,
            "F_tensor": flops_tensor,
            "B_mem": mem_bytes,
            "U_cuda": U_cuda,
            "U_tensor": U_tensor,
            "U_mem": U_mem,
            "U_bottle_neck": U_bottle,
            "bottle_neck_unit": _unit.get(bottleneck),
            "num_kernels": len(per_kernel),
            "per_kernel": per_kernel,
        }
    finally:
        try:
            Path(script_path).unlink()
        except OSError:
            pass


def _finalize_record(latency_ms, ncu, config=None):
    T_us = latency_ms * 1000.0

    per_kernel = ncu.get("per_kernel", [])
    n = len(per_kernel)
    for k in per_kernel:
        frac = k.pop("dur_frac", (1.0 / n) if n else 0.0)
        k["T_us"] = T_us * frac

    record = {}
    if config is not None:
        record["config"] = config
    record["T_us"] = T_us
    record["latency_ms"] = latency_ms
    record.update(ncu)
    return record


def _ratio(num, den):
    if num is None or den is None or den == 0:
        return None
    return round(num / den, 6)


def _factors_for_block(block, ref_block):
    if not block or not ref_block:
        return None
    out = {
        "bandwidth_gbps": _ratio(
            block.get("bandwidth_gbps"), ref_block.get("bandwidth_gbps")
        ),
        "tensor": {},
        "vector": {},
    }
    for dt in _TENSOR_DTYPES:
        out["tensor"][dt] = _ratio(
            block.get("tensor", {}).get(dt), ref_block.get("tensor", {}).get(dt)
        )
    for dt in _VECTOR_DTYPES:
        out["vector"][dt] = _ratio(
            block.get("vector", {}).get(dt), ref_block.get("vector", {}).get(dt)
        )
    return out


def build_scaling_factors(
    specs_path=HARDWARE_SPECS_PATH, reference_chip=DEFAULT_REFERENCE_CHIP
):
    specs_path = Path(specs_path)
    if not specs_path.exists():
        return {
            "_error": f"spec file not found: {specs_path}, scaling factors left empty",
            "reference_chip": reference_chip,
            "chips": [],
        }
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    chips = specs.get("chips", [])
    ref = next((c for c in chips if c.get("chip") == reference_chip), None)
    if ref is None:
        return {
            "_error": f"reference chip {reference_chip} not in spec file, scaling factors left empty",
            "reference_chip": reference_chip,
            "chips": [],
        }

    entries = []
    for c in chips:
        # The reference chip is listed too (1.0 against itself, null where missing).
        entries.append(
            {
                "vendor": c.get("vendor"),
                "chip": c.get("chip"),
                "measured": _factors_for_block(c.get("measured"), ref.get("measured")),
            }
        )
    return {
        "reference_chip": reference_chip,
        "source_specs": specs_path.name,
        "chips": entries,
    }


def _dumps(obj, indent=2, level=0):
    """indent=2 JSON, but render all-scalar lists (shapes, values, ...) inline.

    Dumping e.g. lengths=[8]*1024 one element per line is pure noise; collapsing
    leaf arrays keeps the object structure readable without the vertical bloat.
    """
    pad, pad2 = " " * (indent * level), " " * (indent * (level + 1))
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [
            f"{pad2}{json.dumps(k, ensure_ascii=False)}: "
            f"{_dumps(v, indent, level + 1)}"
            for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        if all(x is None or isinstance(x, (int, float, str, bool)) for x in obj):
            return json.dumps(obj, ensure_ascii=False)
        items = [f"{pad2}{_dumps(x, indent, level + 1)}" for x in obj]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(obj, ensure_ascii=False)


def _flush_results(results, output_path):
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(_dumps(results) + "\n")
    os.replace(tmp, output_path)


def _discover_ops(ops_dir=OPS_DIR):
    if not ops_dir.is_dir():
        return []
    # Let `import <op>` find modules under ops_dir (subprocess scripts use it too).
    if str(ops_dir) not in sys.path:
        sys.path.insert(0, str(ops_dir))
    required = (
        "OP_NAME",
        "DTYPES",
        "IS_INPLACE",
        "native",
        "grid",
        "build_inputs",
        "key_shape",
    )
    discovered = []
    for path in sorted(ops_dir.glob("*.py")):
        if path.stem == "__init__":
            continue
        try:
            mod = importlib.import_module(path.stem)
        except Exception as e:  # noqa: BLE001 - one bad module must not abort the run
            print(f"\nskipping module {path.stem}: import failed ({str(e)[:80]})")
            continue
        missing = [a for a in required if not hasattr(mod, a)]
        if missing:
            print(f"\nskipping module {path.stem}: missing contract fields {missing}")
            continue
        discovered.append((mod.OP_NAME, mod))
    return discovered


def _parse_name_list(value):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    p = Path(value)
    if p.is_file():
        names = []
        for line in p.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                names.append(line)
    else:
        names = [x.strip() for x in value.split(",") if x.strip()]
    return set(names)


def _select_ops(discovered, only=None):
    available = {name for name, _ in discovered}
    if only:
        unknown = sorted(only - available)
        if unknown:
            print(
                f"warning: --ops operators not found, ignored: " f"{', '.join(unknown)}"
            )

    return [(name, mod) for name, mod in discovered if only is None or name in only]


def _collect_one_op_ops(
    op_module, report_dir, results=None, output_path=None, ops_dir=None
):
    if ops_dir is None:
        ops_dir = OPS_DIR
    op_name = op_module.OP_NAME
    mod_name = op_module.__name__
    op = op_module.native()
    if op is None:
        print(f"\nskipping {op_name}: native() resolved no callable")
        return None

    dtypes = op_module.DTYPES
    bindings = op_module.grid()

    try:
        probe_args, probe_kwargs = op_module.build_inputs(
            bindings[0], dtypes[0], "cuda"
        )
        op(*probe_args, **probe_kwargs)
        torch.cuda.synchronize()
    except (
        Exception
    ) as e:  # noqa: BLE001 - any failure means this host cannot collect it
        print(
            f"\nskipping {op_name}: pre-check failed, cannot collect on this host"
            f" ({type(e).__name__}: {str(e)[:120]})"
        )
        return None

    shapes = {}
    entry = {"native_api": f"ops.{mod_name}", "shapes": shapes}
    if results is not None:
        results[op_name] = entry
    print(f"\ncollecting {op_name}  (module {mod_name})")
    for binding in bindings:
        shape = op_module.key_shape(binding)
        per_dtype = shapes.setdefault(str(shape), {})
        for dtype in dtypes:
            dtype_str = str(dtype)
            args, kwargs = op_module.build_inputs(binding, dtype, "cuda")
            latency_ms = triton.testing.do_bench(
                lambda a=args, k=kwargs: op(*a, **k), warmup=25, rep=100
            )
            # The NCU subprocess allocates its own inputs; free ours first so a
            # large-E MoE does not OOM from holding two copies at once.
            del args, kwargs
            torch.cuda.empty_cache()
            script = _build_profile_script_ops(
                ops_dir, mod_name, binding, dtype_str, warmup=3
            )
            ncu = profile_with_ncu(
                op_name, script, shape, dtype_str, report_dir=report_dir
            )
            # Optional config(binding, dtype) describing real input/output shapes.
            cfg = None
            if hasattr(op_module, "config"):
                cfg = op_module.config(binding, dtype)
            per_dtype[dtype_str] = _finalize_record(latency_ms, ncu, config=cfg)

            bn = ncu["bottle_neck_unit"]
            suffix = f"  bottleneck={bn} ({ncu['U_bottle_neck']:.1f}%)" if bn else ""
            print(f"  {shape} {dtype_str}: {latency_ms:.4f} ms{suffix}")

        _flush_results(results, output_path)

    return entry


def _spawn_op_worker(op_name, report_dir, device, op_timeout):
    fd, tmp = tempfile.mkstemp(prefix=f"baseline_{op_name}_", suffix=".json")
    os.close(fd)
    cmd = [
        sys.executable,
        __file__,
        "--ops",
        op_name,
        "--output",
        tmp,
        "--report-dir",
        str(report_dir),
    ]
    if device is not None:
        cmd += ["--device", str(device)]
    env = dict(os.environ, BASELINE_WORKER="1")

    ok = True
    try:
        proc = run_with_cleanup(cmd, timeout=op_timeout, env=env)
        ok = proc.returncode == 0
        if not ok:
            print(
                f"  ✗ {op_name} subprocess exited abnormally (code={proc.returncode}), "
                f"skipped; completed shapes (if any) kept"
            )
    except subprocess.TimeoutExpired:
        ok = False
        print(f"  ✗ {op_name} subprocess timed out (> {op_timeout}s), skipped")

    # Read back whatever the child flushed (possibly partial), success or not.
    entry = None
    try:
        partial = json.loads(Path(tmp).read_text())
        entry = partial.get(op_name)
    except (OSError, ValueError):
        entry = None
    finally:
        try:
            Path(tmp).unlink()
        except OSError:
            pass
    return entry, ok


def _describe_callable(op):
    mod = getattr(op, "__module__", None)
    qn = getattr(op, "__qualname__", None) or getattr(op, "__name__", None)
    if mod and qn:
        return f"{mod}.{qn}"
    try:
        return repr(op)
    except Exception:  # noqa: BLE001
        return type(op).__name__


def _check_one_op(op_module, device=None):
    op_name = op_module.OP_NAME
    mod_name = op_module.__name__
    result = {
        "status": "skip",
        "op_name": op_name,
        "call": f"ops.{mod_name}",
        "shape": None,
        "dtype": None,
        "detail": "",
    }

    try:
        op = op_module.native()
    except Exception as e:  # noqa: BLE001 - native() raising also counts as uncallable
        result["status"] = "fail"
        result["detail"] = f"native() raised: {type(e).__name__}: {e}"
        return result
    if op is None:
        result["detail"] = (
            "native() returned None (intentionally skipped or unresolved)"
        )
        return result

    result["call"] = _describe_callable(op)

    try:
        binding = op_module.grid()[0]
        dtype = op_module.DTYPES[0]
        result["shape"] = op_module.key_shape(binding)
        result["dtype"] = str(dtype)
        args, kwargs = op_module.build_inputs(binding, dtype, "cuda")
    except Exception as e:  # noqa: BLE001 - building inputs failing also counts
        result["status"] = "fail"
        result["detail"] = f"input build failed: {type(e).__name__}: {e}"
        return result

    try:
        op(*args, **kwargs)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - NotImplementedError/shape/CUDA etc.
        result["status"] = "fail"
        result["detail"] = f"{type(e).__name__}: {str(e)[:200]}"
        return result

    result["status"] = "ok"
    return result


def check_ops(device=None, only=None):
    """--check entry: call each operator once, print how it is called and ok/fail;
    collect nothing and write no JSON. Works with --ops/--device.
    Returns the number of failures (nonzero exit code if any)."""
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    device_name = torch.cuda.get_device_name()

    ops_dir = OPS_DIR
    if not ops_dir.is_dir():
        raise RuntimeError(f"operator directory not found: {ops_dir}")
    ops_modules = _discover_ops(ops_dir)
    ops_modules = _select_ops(ops_modules, only=only)

    print(f"check device: {device_name}  (operator dir={ops_dir})")
    print("operator check (--check): call each op once (first shape x first dtype)\n")
    if not ops_modules:
        print("warning: no operators to check after filtering")
        return 0

    _mark = {"ok": "✓", "fail": "✗", "skip": "—"}
    _word = {"ok": "ok", "fail": "fail", "skip": "skip"}
    counts = {"ok": 0, "fail": 0, "skip": 0}
    for op_name, op_module in ops_modules:
        r = _check_one_op(op_module, device=device)
        counts[r["status"]] += 1
        sig = r["call"]
        if r["shape"] is not None:
            sig += f"  {r['shape']} {r['dtype']}"
        line = (
            f"  {_mark[r['status']]} {r['op_name']}  →  {sig}  [{_word[r['status']]}]"
        )
        if r["detail"]:
            line += f": {r['detail']}"
        print(line)

    print(
        f"\nsummary: ok {counts['ok']} / fail {counts['fail']} / "
        f"skip {counts['skip']}  ({len(ops_modules)} operators total)"
    )
    return counts["fail"]


def collect_baseline(
    output_path,
    report_dir=None,
    device=None,
    only=None,
    worker=False,
    op_timeout=1800,
    reference_chip=DEFAULT_REFERENCE_CHIP,
    task_exit_status=False,
):
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    device_name = torch.cuda.get_device_name()
    if "NVIDIA" not in device_name.upper():
        print(f"warning: current device {device_name} may not be NVIDIA hardware")

    results = {}
    output_path = Path(output_path)

    ops_dir = OPS_DIR
    if not ops_dir.is_dir():
        raise RuntimeError(f"operator directory not found: {ops_dir}")
    ops_modules = _discover_ops(ops_dir)
    ops_modules = _select_ops(ops_modules, only=only)

    if worker:
        for op_name, op_module in ops_modules:
            _collect_one_op_ops(
                op_module,
                report_dir,
                results=results,
                output_path=output_path,
                ops_dir=ops_dir,
            )
        _flush_results(results, output_path)
        return

    results["_environment"] = collect_environment(
        torch,
        {
            "device": device,
            "op_timeout_seconds": op_timeout,
            "reference_chip": reference_chip,
            "only": sorted(only) if only is not None else None,
            "latency_warmup_ms": 25,
            "latency_rep_ms": 100,
            "ncu_warmup_calls": 3,
        },
    )
    _flush_results(results, output_path)

    print(f"collect device: {device_name}  (operator dir={ops_dir})")
    sf = build_scaling_factors(reference_chip=reference_chip)
    results["_scaling_factors"] = sf
    if sf.get("_error"):
        print(f"warning: {sf['_error']}")
    else:
        print(
            f"scaling factors: reference chip {sf['reference_chip']}, "
            f"covering {len(sf['chips'])} chips"
        )
    if not ops_modules:
        print("warning: no operators to collect after filtering")
    else:
        print(
            f"operators to collect ({len(ops_modules)}): "
            f"{', '.join(name for name, _ in ops_modules)}"
        )
    failed = []
    for op_name, _ in ops_modules:
        entry, ok = _spawn_op_worker(op_name, report_dir, device, op_timeout)
        if entry is not None:
            results[op_name] = entry
        if not ok:
            failed.append(op_name)
        _flush_results(results, output_path)

    _flush_results(results, output_path)
    n_ops = sum(1 for k in results if not k.startswith("_"))
    print(f"\n✓ baseline data written: {output_path}  ({n_ops} operators)")
    if failed:
        print(
            f"⚠ {len(failed)} operator subprocess(es) failed (skipped, partial results kept): "
            f"{', '.join(failed)}"
        )
    if task_exit_status:
        return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect baseline data from NVIDIA native kernels"
    )
    parser.add_argument(
        "--output", default="op_perf_baseline.json", help="output file path"
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="directory for NCU .ncu-rep reports (open with ncu-ui)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="physical card(s) to collect on. A single card number "
        "(e.g. 0 or 3) runs single-card; comma-separated "
        "(e.g. 0,1,2,3) runs a multi-card dynamic queue, "
        "distributing operators across cards. Applied via "
        "CUDA_VISIBLE_DEVICES; default uses the default device",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a multi-card run, skipping completed operators "
        "(requires multiple card numbers in --device)",
    )
    parser.add_argument(
        "--ops",
        default=None,
        help="only collect the given operators (comma-separated, e.g. "
        "moe_sum,grouped_topk), default is all",
    )
    parser.add_argument(
        "--op-timeout",
        type=float,
        default=1800,
        help="per-operator subprocess timeout in seconds in "
        "orchestration mode (guards against CUDA hangs; default 1800)",
    )
    parser.add_argument(
        "--reference-chip",
        default=DEFAULT_REFERENCE_CHIP,
        help=f"denominator (reference) chip for scaling factors, i.e. the "
        f"card baseline is collected on; must be a chip name in "
        f"hardware_specs.json (default {DEFAULT_REFERENCE_CHIP})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only check each operator's native interface can be called "
        "(one trial call on GPU with the first shape x first dtype); "
        "print how it is called and ok/fail, collect nothing and "
        "write no JSON. Works with --ops/--device",
    )
    args = parser.parse_args()
    worker = os.environ.get("BASELINE_WORKER") == "1"
    task_exit_status = os.environ.get("BASELINE_TASK_EXIT") == "1"

    multi_gpu = args.device is not None and "," in args.device
    if multi_gpu:
        # 多卡采集暂不支持，先拦截返回。原多卡调度逻辑保留在下方注释中，
        # 待后续恢复支持时再取消注释。
        print("multi-card collection is not supported yet")
        sys.exit(1)
        # if args.check or worker:
        #     parser.error(
        #         "multi-card collection (multiple card numbers in --device) "
        #         "cannot be combined with --check or worker mode"
        #     )
        # if args.op_timeout <= 0:
        #     parser.error("--op-timeout must be positive")
        # args.devices = args.device
        # from collect_baseline_multi_gpu import collect
        #
        # try:
        #     sys.exit(collect(args))
        # except ValueError as exc:
        #     parser.error(str(exc))
    if args.resume:
        # --resume 仅服务于多卡采集，多卡暂不支持，这里一并拦截返回。
        # 原校验逻辑保留在下方注释中，待恢复多卡支持时再取消注释。
        print("--resume is not supported yet")
        sys.exit(1)
        # parser.error(
        #     "--resume requires multiple card numbers in --device (multi-card collection)"
        # )
    if args.check:
        n_fail = check_ops(device=args.device, only=_parse_name_list(args.ops))
        sys.exit(1 if n_fail else 0)
    exit_code = collect_baseline(
        args.output,
        report_dir=args.report_dir or "ncu_reports",
        device=args.device,
        only=_parse_name_list(args.ops),
        worker=worker,
        op_timeout=args.op_timeout,
        reference_chip=args.reference_chip,
        task_exit_status=task_exit_status,
    )

    sys.exit(exit_code or 0)
