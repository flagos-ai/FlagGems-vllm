# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Independent vLLM native coordinates and deterministic expansion workloads.

R values are synthetic batch sweeps, not measured model traces. G=512/P=4
comes from the official checkpoint's index_topk=2048/index_kpool=4 at
46a050bec13002ad29ee795c16e61cf4fa3c4a9c. Native source extraction is opt-in;
it is recorded separately from the installed vLLM callable.
"""

import __future__

import ast
import hashlib
import importlib
import importlib.metadata
import inspect
import os
import sys
import types
from pathlib import Path

import torch

OP_NAME = "expand_pools_and_append_tail"
DTYPES = [torch.int32, torch.int64]
IS_INPLACE = False
MUTATED_INPUTS = ()
NATIVE_UNAVAILABLE_REASON = None
NATIVE_PROVENANCE = None
NATIVE_MODULE = "vllm.models.glm5next.nvidia.ops.kpool_compress"
NATIVE_REF = "a17e79abde45715e79a00deb99ad34be49162231"
PINNED_SHA256 = "6d5ff569ca3784f9e67145ef9ba32b0d7efa0a0318e7f13e266ac15e0e3c5554"
_KERNEL_NAME = "_expand_pools_and_append_tail_kernel"
_MODEL_REF = "46a050bec13002ad29ee795c16e61cf4fa3c4a9c"


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _assert_independent(op):
    if not callable(op):
        raise TypeError("vLLM native symbol is not callable")
    module = getattr(op, "__module__", "")
    source_file = inspect.getsourcefile(op) or ""
    source = inspect.getsource(op)
    if (
        "flaggems_vllm" in module
        or "flaggems_vllm" in source_file
        or "flaggems_vllm" in source
    ):
        raise RuntimeError("native symbol resolves to the candidate implementation")
    return source_file, _sha256(source.encode())


def _pinned_native(path):
    """Compile only the two unchanged definitions from the complete pinned file."""
    path = Path(path).expanduser().resolve(strict=True)
    content = path.read_bytes()
    if _sha256(content) != PINNED_SHA256:
        raise ValueError("EXPAND_PINNED_NATIVE_FILE does not match the pinned SHA256")
    source = content.decode("utf-8")
    parsed = ast.parse(source, filename=str(path))
    names = {_KERNEL_NAME, OP_NAME}
    definitions = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(definitions) != 2 or {node.name for node in definitions} != names:
        raise ValueError("pinned file must contain exactly the two native definitions")
    # Bind the unchanged source's minimal dependencies without initializing
    # unrelated vLLM C extensions. Verify the namespace's actual import mapping.
    utils_path = Path(
        importlib.metadata.distribution("vllm").locate_file(
            "vllm/triton_utils/__init__.py"
        )
    )
    utils_source = utils_path.read_bytes()
    imports = {
        (alias.name, alias.asname)
        for node in ast.walk(ast.parse(utils_source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    if not {("triton", None), ("triton.language", "tl")} <= imports:
        raise RuntimeError("vLLM Triton namespace does not directly bind triton/tl")
    triton_module = importlib.import_module("triton")
    language_module = importlib.import_module("triton.language")
    module_name = "_expand_pinned_native_" + PINNED_SHA256[:16]
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__dict__.update(torch=torch, triton=triton_module, tl=language_module)
    # Triton/inspect must be able to recover the original source and module.
    sys.modules[module_name] = module
    try:
        exec(
            compile(
                ast.Module(body=definitions, type_ignores=[]),
                str(path),
                "exec",
                flags=__future__.annotations.compiler_flag,
                dont_inherit=True,
            ),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    op = getattr(module, OP_NAME)
    source_file, function_hash = _assert_independent(op)
    lines = source.splitlines(keepends=True)
    definition_hashes = {}
    for node in definitions:
        first = min([node.lineno] + [item.lineno for item in node.decorator_list])
        original = "".join(lines[first - 1 : node.end_lineno])
        definition_hashes[node.name] = _sha256(original.encode())
    provenance = {
        "kind": "pinned_source_extract",
        "ref": NATIVE_REF,
        "file": str(path),
        "file_sha256": PINNED_SHA256,
        "definition_sha256": definition_hashes,
        "callable_module": module_name,
        "callable_file": source_file,
        "callable_source_sha256": function_hash,
        "dependency_binding": (
            "direct imports verified against installed vllm.triton_utils source; "
            "no vLLM package initialization"
        ),
        "dependencies": {
            "torch": getattr(torch, "__file__", None),
            "vllm.triton_utils_source": str(utils_path),
            "vllm.triton_utils_sha256": _sha256(utils_source),
            "triton": getattr(triton_module, "__file__", None),
            "tl": getattr(language_module, "__file__", None),
        },
    }
    return op, provenance


def native():
    """Resolve exact vLLM native; never substitute a reference or candidate."""
    global NATIVE_UNAVAILABLE_REASON, NATIVE_PROVENANCE
    NATIVE_UNAVAILABLE_REASON = None
    NATIVE_PROVENANCE = None
    try:
        pinned = os.environ.get("EXPAND_PINNED_NATIVE_FILE")
        if pinned:
            op, NATIVE_PROVENANCE = _pinned_native(pinned)
        else:
            module = importlib.import_module(NATIVE_MODULE)
            op = getattr(module, OP_NAME)
            source_file, source_hash = _assert_independent(op)
            NATIVE_PROVENANCE = {
                "kind": "installed_vllm",
                "requested_module": NATIVE_MODULE,
                "callable_module": getattr(op, "__module__", None),
                "callable_file": source_file,
                "callable_source_sha256": source_hash,
                "module_file": getattr(module, "__file__", None),
                "requested_ref": NATIVE_REF,
                "ref_verified": False,
            }
        return op
    except Exception as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None


def grid():
    """Frozen nonempty output set, including tail-only and padded-row inputs."""
    cases = [
        (1, 512, 4, 0, "one"),
        (7, 512, 4, 0, "mixed"),
        (24, 512, 4, 0, "mixed"),
        (257, 512, 4, 0, "mixed"),
        (1, 512, 4, 0, "none"),
        (7, 512, 4, 0, "none"),
        (24, 512, 4, 0, "none"),
        (257, 512, 4, 0, "none"),
        (1, 512, 4, 0, "full"),
        (24, 512, 4, 7, "full"),
        (7, 3, 2, 3, "mixed"),
        (7, 3, 8, 0, "full"),
        (24, 128, 16, 7, "mixed"),
        (257, 128, 16, 0, "none"),
        (7, 0, 4, 3, "mixed"),
        (24, 3, 1, 7, "none"),
    ]
    return [dict(zip(("R", "G", "P", "row_padding", "tail"), case)) for case in cases]


def build_inputs(binding, dtype, device):
    """Build on CPU, copy the whole backing storage, then retain row padding."""
    if dtype not in DTYPES:
        raise TypeError("index dtype must be int32 or int64")
    rows, groups, size = (binding[key] for key in ("R", "G", "P"))
    width = max(groups + binding["row_padding"], 1)
    sentinel = binding.get("sentinel", -1)
    storage = torch.full((rows, width), -1234567, dtype=dtype)
    for row in range(rows):
        for group in range(groups):
            value = (row * 17 + group * 5) % 4096
            if group % 13 == 1:
                value = 7
            if (row + group) % 11 == 0:
                value = sentinel
            storage[row, group] = value
    tails = {
        "none": lambda row: 0,
        "one": lambda row: min(1, size - 1),
        "full": lambda row: size - 1,
        "mixed": lambda row: row % size,
    }
    remainder = tails[binding["tail"]]
    lengths = torch.tensor(
        [(8 + row % 5) * size + remainder(row) for row in range(rows)], dtype=dtype
    )
    pools = storage.to(device=device)[:, :groups]
    return (pools, lengths.to(device=device), size), {}


def key_shape(binding):
    stride = max(binding["G"] + binding["row_padding"], 1)
    return (
        f"R{binding['R']}_G{binding['G']}_P{binding['P']}"
        f"_stride{stride}_tail-{binding['tail']}"
    )


def config(binding, dtype):
    rows, groups, size = (binding[key] for key in ("R", "G", "P"))
    stride = max(groups + binding["row_padding"], 1)
    out_cols = groups * size + size - 1
    return {
        "inputs": {
            "pool_ids": {
                "shape": [rows, groups],
                "dtype": str(dtype),
                "stride": [stride, 1],
                "backing_storage_shape": [rows, stride],
            },
            "seq_lens": {"shape": [rows], "dtype": str(dtype), "stride": [1]},
            "pool_size": {"scalar": size},
        },
        "outputs": {
            "out": {
                "shape": [rows, out_cols],
                "dtype": "torch.int32",
                "stride": [out_cols, 1],
                "new_allocation": True,
            }
        },
        "dims": dict(binding),
        "tail_distribution": binding["tail"],
        "tail_remainder_rule": {
            "none": "0",
            "one": "min(1,P-1)",
            "full": "P-1",
            "mixed": "row%P",
        }[binding["tail"]],
        "tail_full_definition": "P-1 valid tail entries (maximum incomplete pool)",
        "input_construction": "deterministic CPU integer loops; copy backing storage",
        "pool_distribution": "repeated IDs, negative sentinels, no history clipping",
        "seq_lens": "(8 + row%5)*P + tail_remainder",
        "mutation": [],
        "graph": False,
        "workload_origin": {
            "R": "synthetic batch sweep, not a measured model trace",
            "G512_P4": {
                "checkpoint_ref": _MODEL_REF,
                "index_kpool": 4,
                "index_topk": 2048,
            },
            "other_shapes": "synthetic contract/diagnostic coverage",
        },
    }
