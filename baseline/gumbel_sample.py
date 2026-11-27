# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Compatible vLLM native provider for the 16 immutable FP32 performance cases.

The NVIDIA collector calls native() before build_inputs(); that default remains
compatible. Other supported devices pass their actual device and expected vendor.
Candidate benchmarks pass qualify=False and explicitly qualify any native.
Set GUMBEL_NATIVE_SOURCE_ROOT to a task-private vLLM source tree containing the
three source files named in NATIVE_SOURCE to avoid changing an installed vLLM package.
The returned callable is the unmodified selected vLLM function, not a wrapper.
"""

import ast
import builtins
import copy
import hashlib
import importlib
import inspect
import json
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import torch

OP_NAME = "gumbel_sample"
DTYPES = [torch.float32]
IS_INPLACE = False
MUTATED_INPUTS = ()
NATIVE_UNAVAILABLE_REASON = None
TIMING_PROTOCOL = "triton-do-bench-no-graph-25ms-warmup-100ms-median"
SOURCE_SHA256 = "820184641a0e4ea48bd837d66724a85f1f6fc2edc3bc1354577e2d635c0a68f1"
NATIVE_SOURCE = {
    "repository": "https://github.com/vllm-project/vllm",
    "commit": "0ef2a95cc861adfae9a92ba104bdfd40ede8cbaa",
    "callable": "vllm.v1.worker.gpu.sample.gumbel.gumbel_sample",
    "files": {
        "vllm/v1/worker/gpu/sample/gumbel.py": SOURCE_SHA256,
        "vllm/v1/worker/gpu/spec_decode/draft_logits.py": (
            "5a0edde7f879c4727901b20c2d482794efe968d6480918bba4492bfcafadd647"
        ),
        "vllm/triton_utils/__init__.py": (
            "f6e03cec7d37ba3370e49771d1230999c584e2a3e4669c0f10d52b07a861b75a"
        ),
    },
}
SHAPES = (
    (1, 127),
    (1, 1024),
    (4, 4097),
    (16, 32000),
    (64, 32768),
    (1, 128256),
    (16, 128256),
    (128, 151936),
)
SEED = 0x123456789ABCDEF
_NATIVE_OP = None
_NATIVE_MODULE = None
_NATIVE_DEVICE = None
_NATIVE_IDENTITY = {"status": "not_resolved"}
_NATIVE_HASHES_CHECKED = {}
_QUALIFICATIONS = {}

# Independent integer known answers frozen before the candidate implementation.
_KAT_INPUTS = (
    (0, 0, 0, 0x8134CDF8, 0xD28AD962),
    (1, 2, 3, 0x23B53F91, 0xE3AAF818),
    (-1, 17, 1023, 0xC8C13C14, 0xD223C51C),
    (0x123456789ABCDEF, 123456789, 199999, 0x5DDE974F, 0x0547D2FE),
    (-(1 << 63), 1 << 30, 4096, 0xF1A9355B, 0x3AED160F),
    ((1 << 63) - 1, (1 << 30) + 7, 0xFFFFFFFF, 0xD952F9AA, 0x49E8EA7F),
)
_MIN_WINNER_MARGIN = 2e-3


def grid():
    return [
        {"shape": [n, v], "temperature": temperature}
        for n, v in SHAPES
        for temperature in (0.0, 0.8)
    ]


def _case(binding, dtype=torch.float32):
    shape = tuple(binding["shape"])
    temperature = binding["temperature"]
    if shape not in SHAPES or temperature not in (0.0, 0.8):
        raise ValueError(f"outside frozen gumbel performance grid: {binding}")
    if dtype != torch.float32:
        raise ValueError("gumbel native performance coverage is FP32 only")
    return *shape, temperature


def key_shape(binding):
    n, v, temperature = _case(binding)
    return f"n{n}_v{v}_t{'0' if temperature == 0 else '08'}"


def config(binding, dtype):
    """Static JSON identity; runtime paths and qualification live separately."""
    n, v, temperature = _case(binding, dtype)

    def tensor(shape, dt, stride, values):
        return {"shape": shape, "dtype": str(dt), "stride": stride, "values": values}

    return {
        "schema_version": 2,
        "case_key": key_shape(binding),
        "inputs": {
            "logits": tensor(
                [n, v],
                dtype,
                [v if n == 1 else 0, 1],
                "CPU int64 arange(V)*37%101 -> FP32 / 10; to(device); expand(N,V)",
            ),
            "expanded_idx_mapping": tensor([n], torch.int32, [1], "all zero"),
            "temperature": tensor([1], torch.float32, [1], [temperature]),
            "seed": tensor([1], torch.int64, [1], [SEED]),
            "pos": tensor([n], torch.int64, [1], "arange(N)"),
            "apply_temperature": True,
            "is_drafting": False,
            "use_fp64": False,
            "logits_cache": None,
            "logits_cache_col": None,
            "logits_cache_source": None,
            "logits_cache_row_mapping": None,
            "bias": None,
        },
        "outputs": {"shape": [n], "dtype": "torch.int64", "stride": [1]},
        "mutation": [],
        "rng": "Murmur3 fixed explicit seed/pos; identical stream replay; no generator advance",
        "semantic_reference": copy.deepcopy(NATIVE_SOURCE),
        "runtime_source_policy": "compatible-vllm-murmur3",
        "qualification_protocol": "independent-murmur-kat-cpu-winner-replay-input-and-rng-immutable-v1",
        "timing_protocol": TIMING_PROTOCOL,
        "timing": {
            "timer": "triton.testing.do_bench",
            "warmup_ms": 25,
            "rep_ms": 100,
            "return_mode": "median",
            "graph": False,
            "unit": "ms",
            "scope": "operator allocation + launches + combine; events may include host gaps",
            "reset": "none; immutable input and stateless RNG replay",
        },
    }


def build_inputs(binding, dtype, device, *, qualify=True):
    """Construct frozen inputs; candidate benchmarks opt out of auto-qualification.

    The optional keyword leaves the collector's three-argument API unchanged.
    Explicit opt-out also prevents a previous pytest case's resolved native from
    executing during candidate input setup.
    """
    n, v, temperature = _case(binding, dtype)
    cpu_logits = ((torch.arange(v, device="cpu") * 37) % 101).float() / 10
    logits = cpu_logits.to(device).expand(n, v)
    args = (
        logits,
        torch.zeros(n, dtype=torch.int32, device=device),
        torch.tensor([temperature], dtype=torch.float32, device=device),
        torch.tensor([SEED], dtype=torch.int64, device=device),
        torch.arange(n, dtype=torch.int64, device=device),
        True,
        False,
    )
    if qualify and _NATIVE_OP is not None:
        qualify_native(binding, args, {}, _NATIVE_OP)
    return args, {}


def _device_context(device, vendor):
    """Verify the requested vendor against the actual runtime and device name.

    This permits qualification attempts on previously exercised device families;
    it is not a claim that another SKU/compiler passed the per-case checks.
    """
    vendor = "thead" if vendor == "ppu" else vendor
    known_failures = {
        "iluvatar": "reference vLLM random sampling failed on the observed BI-V150 stack "
        "(expected token 60, got 90); this stack is not qualified",
    }
    if vendor in known_failures:
        raise RuntimeError(known_failures[vendor])
    expected_types = {
        "nvidia": "cuda",
        "thead": "cuda",
        "metax": "cuda",
        "hygon": "cuda",
        "mthreads": "musa",
        "ascend": "npu",
    }
    if vendor not in expected_types:
        raise RuntimeError(f"no vLLM qualification coverage for vendor {vendor!r}")
    if vendor == "mthreads" and not hasattr(torch, "musa"):
        importlib.import_module("torch_musa")
    if vendor == "ascend" and not hasattr(torch, "npu"):
        importlib.import_module("torch_npu")
    device = torch.device(device)
    if device.type != expected_types[vendor]:
        raise ValueError(
            f"vendor {vendor} requires {expected_types[vendor]}, got {device}"
        )
    api = getattr(torch, device.type, None)
    for method in (
        "is_available",
        "current_device",
        "get_device_name",
        "get_rng_state",
        "synchronize",
    ):
        if not callable(getattr(api, method, None)):
            raise RuntimeError(f"torch.{device.type}.{method} is unavailable")
    if not api.is_available():
        raise RuntimeError(f"torch.{device.type} is unavailable")
    if device.index is None:
        device = torch.device(device.type, api.current_device())
    name = api.get_device_name(device)
    lowered = name.lower()
    hip = bool(getattr(torch.version, "hip", None))
    cuda = bool(getattr(torch.version, "cuda", None))
    detected = None
    if device.type == "musa" and ("mtt" in lowered or "mthreads" in lowered):
        detected = "mthreads"
    elif device.type == "npu" and "ascend" in lowered:
        detected = "ascend"
    elif device.type == "cuda":
        if hip and (lowered == "bw" or "hygon" in lowered or "dcu" in lowered):
            detected = "hygon"
        elif not hip:
            if "ppu" in lowered:
                detected = "thead"
            elif "metax" in lowered:
                detected = "metax"
            elif cuda and any(
                tag in lowered for tag in ("nvidia", "tesla", "geforce", "quadro")
            ):
                detected = "nvidia"
    if detected != vendor:
        raise ValueError(
            f"requested vendor {vendor!r} differs from observed device: "
            f"{device}, name={name!r}, CUDA={cuda}, HIP={hip}"
        )
    return (
        device,
        api,
        {
            "vendor": detected,
            "device": str(device),
            "device_name": name,
            "device_api": f"torch.{device.type}",
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "torch_hip_version": getattr(torch.version, "hip", None),
        },
    )


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_files(root):
    """Record actual source bytes; reference hashes are not a version allowlist."""
    files = {}
    for relative in NATIVE_SOURCE["files"]:
        path = (root / relative).resolve()
        files[relative] = {"path": str(path), "sha256": _sha(path)}
    return files


def _source_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def _check_source_unchanged(files):
    for relative, entry in files.items():
        if _sha(entry["path"]) != entry["sha256"]:
            raise ValueError(f"native source changed after loading: {relative}")


def _check_interface(module, files):
    for entry in files.values():
        tree = ast.parse(Path(entry["path"]).read_bytes(), filename=entry["path"])
        for node in ast.walk(tree):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            )
            if any(
                name.split(".")[0] in ("flaggems_vllm", "flag_gems") for name in names
            ):
                raise ValueError(
                    "native source must not import the candidate implementation"
                )
    function = module.gumbel_sample
    if (
        function.__globals__ is not vars(module)
        or function.__name__ != "gumbel_sample"
        or function.__module__.startswith(("flaggems_vllm", "flag_gems"))
    ):
        raise ValueError("native callable was replaced or wrapped")
    signature = inspect.signature(function)
    parameters = tuple(signature.parameters)
    expected = (
        "logits",
        "expanded_idx_mapping",
        "temperature",
        "seed",
        "pos",
        "apply_temperature",
        "is_drafting",
    )
    if parameters[:7] != expected:
        raise ValueError(f"incompatible gumbel_sample input signature: {signature}")
    signature.bind(None, None, None, None, None, True, False)
    for name, default in (
        ("use_fp64", False),
        ("logits_cache", None),
        ("logits_cache_col", None),
        ("logits_cache_source", None),
        ("logits_cache_row_mapping", None),
        ("bias", None),
    ):
        if (
            name in signature.parameters
            and signature.parameters[name].default is not default
        ):
            raise ValueError(f"incompatible gumbel_sample default: {name}")
    gumbel_path = files["vllm/v1/worker/gpu/sample/gumbel.py"]["path"]
    draft_path = files["vllm/v1/worker/gpu/spec_decode/draft_logits.py"]["path"]
    for name, expected_path in (
        ("gumbel_sample", gumbel_path),
        ("murmur3_hash32", gumbel_path),
        ("draft_logits_row", draft_path),
    ):
        helper = getattr(module, name, None)
        helper = getattr(helper, "fn", helper)
        if not callable(helper):
            raise ValueError(f"native source lacks compatible {name} helper")
        path = inspect.getsourcefile(helper)
        if not path or str(Path(path).resolve()) != expected_path:
            raise ValueError(
                f"native {name} was redirected outside its vLLM source file"
            )
    return str(signature)


def _private_module(name, path):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    # Only private names are registered; installed vllm modules are untouched.
    sys.modules[name] = module
    return module


def _load_source(root, files):
    """Execute original source with explicit imports, without importing vLLM."""
    triton = importlib.import_module("triton")
    tl = importlib.import_module("triton.language")
    prefix = "_flaggems_native_gumbel_" + _source_digest(files)[:16]
    draft_path = root / "vllm/v1/worker/gpu/spec_decode/draft_logits.py"
    tree = ast.parse(draft_path.read_bytes(), filename=str(draft_path))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "draft_logits_row"
    )
    helper = _private_module(prefix + "_draft", draft_path)
    helper.__dict__.update(triton=triton, tl=tl)
    aliases = types.SimpleNamespace(triton=triton, tl=tl)

    def native_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] in ("flaggems_vllm", "flag_gems"):
            raise ImportError(
                "native source must not import the candidate implementation"
            )
        if name == "vllm.triton_utils" and set(fromlist) == {"triton", "tl"}:
            return aliases
        if name == "vllm.v1.worker.gpu.spec_decode.draft_logits" and tuple(
            fromlist
        ) == ("draft_logits_row",):
            return helper
        if name.startswith("vllm"):
            raise ImportError(f"unfrozen native dependency: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    guarded_builtins = {**vars(builtins), "__import__": native_import}
    helper.__dict__["__builtins__"] = guarded_builtins
    # The helper's original AST and original filename preserve source identity.
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(draft_path), "exec"),
        helper.__dict__,
    )
    path = root / "vllm/v1/worker/gpu/sample/gumbel.py"
    module = _private_module(prefix, path)
    module.__dict__["__builtins__"] = guarded_builtins
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module, {
        "kind": "vllm_source_tree",
        "dependency_loading": "original gumbel source; original draft_logits_row AST; direct triton/tl aliases",
        "files": files,
    }


def native(device="cuda", vendor="nvidia"):
    """Resolve source identity only; device qualification is a separate gate."""
    global _NATIVE_OP, _NATIVE_MODULE, _NATIVE_DEVICE, _NATIVE_IDENTITY, NATIVE_UNAVAILABLE_REASON
    _NATIVE_OP = None
    _NATIVE_MODULE = None
    _NATIVE_DEVICE = None
    _QUALIFICATIONS.clear()
    _NATIVE_HASHES_CHECKED.clear()
    identity = {}
    observed = None
    disabled = os.environ.get("GUMBEL_BENCH_DISABLE_NATIVE")
    try:
        if disabled:
            raise RuntimeError(f"disabled with evidence: {disabled}")
        device, _, observed = _device_context(device, vendor)
        source_root = os.environ.get("GUMBEL_NATIVE_SOURCE_ROOT")
        if source_root:
            root = Path(source_root).resolve()
            files = _source_files(root)
            identity = {"kind": "vllm_source_tree", "files": files}
            module, identity = _load_source(root, files)
        else:
            module = importlib.import_module("vllm.v1.worker.gpu.sample.gumbel")
            root = Path(module.__file__).resolve().parents[5]
            files = _source_files(root)
            identity = {"kind": "installed_vllm", "files": files}
        if module.triton is not importlib.import_module(
            "triton"
        ) or module.tl is not importlib.import_module("triton.language"):
            raise ValueError("native triton/tl aliases are not the recorded runtime")
        signature = _check_interface(module, files)
        _check_source_unchanged(files)
        function = module.gumbel_sample
        _NATIVE_MODULE, _NATIVE_OP = module, function
        _NATIVE_DEVICE = observed
        NATIVE_UNAVAILABLE_REASON = None
        _NATIVE_IDENTITY = {
            **identity,
            "device_identity": observed,
            "status": "resolved_unqualified",
            "semantic_reference": copy.deepcopy(NATIVE_SOURCE),
            "source_policy": "compatible-vllm-murmur3",
            "source_digest": _source_digest(files),
            "signature": signature,
            "torch_version": torch.__version__,
            "triton_version": module.triton.__version__,
            "triton_file": module.triton.__file__,
            "callable_module": function.__module__,
        }
        return function
    except Exception as exc:
        _NATIVE_OP = None
        _NATIVE_MODULE = None
        _NATIVE_DEVICE = None
        NATIVE_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
        _NATIVE_IDENTITY = {
            **identity,
            "device_identity": observed,
            "status": "unavailable",
            "reason": NATIVE_UNAVAILABLE_REASON,
            "semantic_reference": copy.deepcopy(NATIVE_SOURCE),
            "source_policy": "compatible-vllm-murmur3",
        }
        return None


def native_identity():
    return copy.deepcopy(_NATIVE_IDENTITY)


def _reference_murmur3(seed, position, offsets, domain=0):
    """Vectorized unsigned integer oracle, independent of any Triton helper."""
    import numpy as np

    mask = np.uint64(0xFFFFFFFF)
    h = np.full(np.shape(offsets), domain, dtype=np.uint64)
    seed = int(seed) & ((1 << 64) - 1)
    for word in (seed & 0xFFFFFFFF, seed >> 32, int(position) & 0xFFFFFFFF, offsets):
        k = np.asarray(word, dtype=np.uint64)
        k = (k * np.uint64(0xCC9E2D51)) & mask
        k = ((k << np.uint64(15)) | (k >> np.uint64(17))) & mask
        h ^= (k * np.uint64(0x1B873593)) & mask
        h = ((h << np.uint64(13)) | (h >> np.uint64(19))) & mask
        h = (h * np.uint64(5) + np.uint64(0xE6546B64)) & mask
    h ^= np.uint64(16)
    h ^= h >> np.uint64(16)
    h = (h * np.uint64(0x85EBCA6B)) & mask
    h ^= h >> np.uint64(13)
    h = (h * np.uint64(0xC2B2AE35)) & mask
    return (h ^ (h >> np.uint64(16))).astype(np.uint32)


def _check_known_answers():
    for seed, pos, offset, first, second in _KAT_INPUTS:
        for domain, expected in ((0, first), (0x9E3779B9, second)):
            assert int(_reference_murmur3(seed, pos, offset, domain)) == expected
            assert (
                int(_reference_murmur3(seed, pos + (1 << 32), offset, domain))
                == expected
            )


def _reference_winners(logits, mapping, temperature, seeds, positions):
    """Return independent FP32 winners and mathematical winner margins.

    log1p is evaluated in FP64 and rounded into the FP32 score. The existing
    contract's 2e-3 stable-winner margin guards compiler log/div differences;
    qualification rejects an ambiguous row rather than dropping it.
    """
    import numpy as np

    _check_known_answers()
    expected, margins = [], []
    keys = np.arange(logits.shape[1], dtype=np.uint64)
    for row, request in enumerate(mapping):
        values = np.asarray(logits[row], dtype=np.float32)
        temp = np.float32(temperature[int(request)])
        if temp == 0:
            # np.argmax returns the first maximum, including repeated maxima.
            expected.append(int(np.argmax(values)))
            margins.append(None)
            continue
        hashed = _reference_murmur3(int(seeds[int(request)]), int(positions[row]), keys)
        uniform = (hashed >> 16).astype(np.float32) * np.float32(2.0**-16)
        uniform += (
            (hashed & 0xFFFF).astype(np.float32) + np.float32(0.5)
        ) * np.float32(2.0**-32)
        with np.errstate(divide="ignore"):
            noise = -np.log(-np.log1p(-uniform.astype(np.float64)))
        scores = (values / temp + noise).astype(np.float32)
        winner = int(np.argmax(scores))
        top = np.partition(scores, -2)[-2:]
        margin = float(np.max(top) - np.min(top))
        if not margin > _MIN_WINNER_MARGIN:
            raise ValueError(
                f"independent oracle has ambiguous row {row}: margin={margin}"
            )
        expected.append(winner)
        margins.append(margin)
    return expected, margins


def _check_native_hashes(module, device):
    """Probe the resolved native's own hash helpers against independent KATs."""
    global triton, tl
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def probe(seed_ptr, pos_ptr, key_ptr, out_ptr, hash_fn: tl.constexpr):
        i = tl.arange(0, 8)
        seed = tl.load(seed_ptr + i, i < 6, 0)
        pos = tl.load(pos_ptr + i, i < 6, 0)
        key = tl.load(key_ptr + i, i < 6, 0)
        first = hash_fn(seed, pos, key)
        second = hash_fn(seed, pos, key, domain=0x9E3779B9)
        tl.store(out_ptr + i, first.to(tl.int64), i < 6)
        tl.store(out_ptr + 6 + i, second.to(tl.int64), i < 6)

    columns = [
        torch.tensor(
            [row[index] for row in _KAT_INPUTS], dtype=torch.int64, device=device
        )
        for index in range(3)
    ]
    output = torch.empty(12, dtype=torch.int64, device=device)
    probe[(1,)](*columns, output, module.murmur3_hash32)
    expected = [row[3] for row in _KAT_INPUTS] + [row[4] for row in _KAT_INPUTS]
    actual = output.cpu().tolist()
    return {
        "status": "passed" if actual == expected else "failed",
        "expected": expected,
        "actual": actual,
        "scope": "six keys, two domains",
    }


@lru_cache(maxsize=16)
def _expected_ids(n, v, temperature):
    import numpy as np

    values = ((np.arange(v, dtype=np.int64) * 37) % 101).astype(
        np.float32
    ) / np.float32(10)
    winners, _ = _reference_winners(
        np.broadcast_to(values, (n, v)),
        np.zeros(n, dtype=np.int32),
        np.array([temperature], dtype=np.float32),
        np.array([SEED], dtype=np.int64),
        np.arange(n, dtype=np.int64),
    )
    return tuple(winners)


def expected_ids(binding):
    """Independent CPU winners for this frozen 16-case performance scope only."""
    return list(_expected_ids(*_case(binding)))


def assert_output(binding, output):
    """Check a performance output; this does not replace full contract tests."""
    n, _, _ = _case(binding)
    if output.dtype != torch.int64 or output.shape != (n,):
        raise AssertionError("gumbel performance output metadata differs from contract")
    if output.detach().cpu().tolist() != expected_ids(binding):
        raise AssertionError(
            "gumbel performance output differs from independent oracle"
        )


def _tensor_snapshot(tensor):
    return (
        (
            list(tensor.shape),
            list(tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
            tensor.data_ptr(),
        ),
        tensor.detach().cpu().clone(),
    )


def _check_inputs(binding, args, kwargs):
    n, v, temperature = _case(binding, args[0].dtype)
    if len(args) != 7 or kwargs or args[5:] != (True, False):
        raise ValueError(
            "native qualification requires the frozen seven-argument no-cache call"
        )
    cfg = config(binding, args[0].dtype)
    names = ("logits", "expanded_idx_mapping", "temperature", "seed", "pos")
    for tensor, name in zip(args[:5], names):
        expected = cfg["inputs"][name]
        actual = (list(tensor.shape), list(tensor.stride()), str(tensor.dtype))
        if actual != (expected["shape"], expected["stride"], expected["dtype"]):
            raise ValueError(
                f"native input metadata differs from config: {name}: {actual}"
            )
        if tensor.device != args[0].device:
            raise ValueError("native input devices differ")
    snapshots = [_tensor_snapshot(tensor) for tensor in args[:5]]
    cpu = [snapshot[1] for snapshot in snapshots]
    values = ((torch.arange(v, device="cpu") * 37) % 101).float() / 10
    expected = (
        values.expand(n, v),
        torch.zeros(n, dtype=torch.int32),
        torch.tensor([temperature], dtype=torch.float32),
        torch.tensor([SEED], dtype=torch.int64),
        torch.arange(n, dtype=torch.int64),
    )
    if any(not torch.equal(actual, wanted) for actual, wanted in zip(cpu, expected)):
        raise ValueError("native input values differ from frozen trajectory")
    return snapshots


def qualify_native(binding, args, kwargs, op, *, allow_accuracy_warning=False):
    """Qualify independently of candidate, with strict checks by default.

    A failed device call can poison the context. Callers must save its error and
    terminate that process, never continue timing after catching this exception.
    Only the Ascend benchmark may opt into timing a numerically wrong reference;
    metadata, range, replay, mutation, source and execution errors remain fatal.
    """
    if op is not _NATIVE_OP or _NATIVE_MODULE is None:
        raise ValueError("qualify_native requires this provider's resolved callable")
    if _NATIVE_DEVICE is None:
        raise ValueError("native device identity was not resolved")
    if allow_accuracy_warning and _NATIVE_DEVICE["vendor"] != "ascend":
        raise ValueError("native accuracy warnings are permitted only for Ascend")
    device, device_api, observed = _device_context(
        args[0].device, _NATIVE_DEVICE["vendor"]
    )
    if observed != _NATIVE_DEVICE:
        raise ValueError("qualification device differs from the resolved native device")
    _check_source_unchanged(_NATIVE_IDENTITY["files"])
    saved = _check_inputs(binding, args, kwargs)
    expected, margins = _reference_winners(*(entry[1].numpy() for entry in saved))
    cpu_rng = torch.get_rng_state().clone()
    device_rng = device_api.get_rng_state(device).clone()
    n, v, temperature = _case(binding)
    hash_key = (id(op), str(device))
    if observed["vendor"] == "ascend" and temperature == 0:
        hash_kat = {
            "status": "not_applicable",
            "reason": "zero temperature uses no RNG",
        }
    else:
        if hash_key not in _NATIVE_HASHES_CHECKED:
            _NATIVE_HASHES_CHECKED[hash_key] = _check_native_hashes(
                _NATIVE_MODULE, device
            )
        hash_kat = copy.deepcopy(_NATIVE_HASHES_CHECKED[hash_key])
    if hash_kat["status"] == "failed" and not allow_accuracy_warning:
        raise AssertionError(
            "native Murmur3 known-answer check failed: " + json.dumps(hash_kat)
        )
    first = op(*args, **kwargs)
    replay = op(*args, **kwargs)
    device_api.synchronize(device)
    actual_winners = []
    for output in (first, replay):
        if (
            not isinstance(output, torch.Tensor)
            or output.dtype != torch.int64
            or output.shape != (n,)
            or output.device != device
        ):
            raise AssertionError("native output metadata differs from contract")
        actual = output.cpu().tolist()
        if any(token < 0 or token >= v for token in actual):
            raise AssertionError("native output token is outside vocabulary range")
        actual_winners.append(actual)
    if not torch.equal(first, replay):
        raise AssertionError("native fixed seed/position replay changed")
    for tensor, (metadata, contents) in zip(args[:5], saved):
        current, current_contents = _tensor_snapshot(tensor)
        if current != metadata or not torch.equal(current_contents, contents):
            raise AssertionError("native mutated a read-only input")
    if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
        device_rng, device_api.get_rng_state(device)
    ):
        raise AssertionError("native advanced a default RNG state")
    _check_source_unchanged(_NATIVE_IDENTITY["files"])
    accuracy_failures = []
    if hash_kat["status"] == "failed":
        accuracy_failures.append("native Murmur3 known-answer check failed")
    if actual_winners[0] != expected:
        accuracy_failures.append(
            "native result differs from independent Murmur3/CPU oracle"
        )
    finite_margins = [value for value in margins if value is not None]
    report = {
        "status": (
            ("accuracy_warning" if allow_accuracy_warning else "failed")
            if accuracy_failures
            else "passed"
        ),
        "accuracy_status": "failed" if accuracy_failures else "passed",
        "performance_allowed": not accuracy_failures or allow_accuracy_warning,
        "accuracy_failures": accuracy_failures,
        "case_key": key_shape(binding),
        "dtype": str(args[0].dtype),
        "device": str(device),
        "device_identity": observed,
        "oracle": "independent NumPy integer Murmur3 and FP64 log1p, FP32 score",
        "minimum_sampling_winner_margin": (
            min(finite_margins) if finite_margins else None
        ),
        "oracle_winners": expected,
        "native_winners": actual_winners[0],
        "replay_winners": actual_winners[1],
        "native_hash_kat": hash_kat,
        "replay": "passed: two identical calls",
        "inputs_immutable": True,
        "cpu_rng_immutable": True,
        "device_rng_immutable": True,
        "config_sha256": hashlib.sha256(
            json.dumps(config(binding, args[0].dtype), sort_keys=True).encode()
        ).hexdigest(),
        "native_identity": native_identity(),
    }
    _QUALIFICATIONS[(key_shape(binding), str(args[0].dtype), str(device))] = (
        copy.deepcopy(report)
    )
    if accuracy_failures and not allow_accuracy_warning:
        raise AssertionError("; ".join(accuracy_failures) + ": " + json.dumps(report))
    return report


def qualification(binding, dtype):
    """Return the resolved device's qualification receipt for a collector sidecar."""
    if _NATIVE_DEVICE is None:
        return None
    _check_source_unchanged(_NATIVE_IDENTITY["files"])
    return copy.deepcopy(
        _QUALIFICATIONS.get((key_shape(binding), str(dtype), _NATIVE_DEVICE["device"]))
    )
