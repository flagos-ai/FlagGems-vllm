# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Load the installed vLLM ordinary-chain sampler for benchmark comparison.

Older installations accept a draft-logits tensor instead of a cache object.
The adapter unwraps that object without computing or copying tensors. Original
vLLM source can be loaded privately when unrelated package initialization fails.
Device execution and per-case mathematical qualification belong to the caller.
"""

import ast
import builtins
import hashlib
import importlib
import importlib.metadata
import inspect
import os
import sys
import types
from pathlib import Path

_MODULE = "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils"
_RELATIVE = "v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
_GUMBEL = "v1/worker/gpu/sample/gumbel.py"
_COMMON = (
    "target_logits",
    "draft_sampled",
    "cu_num_logits",
    "pos",
    "idx_mapping",
    "expanded_idx_mapping",
    "expanded_local_pos",
    "temperature",
    "seed",
    "num_speculative_steps",
)


def _file_identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _private_source(root, files):
    """Execute unchanged installed files with verified Triton import aliases."""
    utils_path = root / "triton_utils/__init__.py"
    tree = ast.parse(utils_path.read_bytes())
    imports = {
        (alias.name, alias.asname)
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    required = {
        ("triton", None),
        ("triton.language", "tl"),
        ("triton.language.extra.libdevice", "tldevice"),
    }
    if not required <= imports:
        raise ImportError("Installed vLLM does not expose the expected Triton aliases")
    files["triton_utils/__init__.py"] = _file_identity(utils_path)
    triton = importlib.import_module("triton")
    tl = importlib.import_module("triton.language")
    tldevice = importlib.import_module("triton.language.extra.libdevice")
    # This queries the real active compiler backend before enabling GPU code.
    triton.runtime.driver.active.get_current_target()
    aliases = types.SimpleNamespace(
        triton=triton, tl=tl, tldevice=tldevice, HAS_TRITON=True
    )
    modules = {}
    prefix = "_flaggems_installed_rejection_" + files[_RELATIVE]["sha256"][:16]

    def native_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.triton_utils" and set(fromlist) <= set(vars(aliases)):
            return aliases
        if name == "vllm.v1.worker.gpu.sample.gumbel" and "gumbel" in modules:
            return modules["gumbel"]
        if name.split(".")[0] in ("vllm", "flaggems_vllm", "flag_gems"):
            raise ImportError("Unsupported native dependency: " + name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    created = []
    try:
        for key, relative in (("gumbel", _GUMBEL), ("rejection", _RELATIVE)):
            path = root / relative
            files[relative] = _file_identity(path)
            name = prefix + "_" + key
            module = types.ModuleType(name)
            module.__file__ = str(path)
            module.__dict__["__builtins__"] = {
                **vars(builtins),
                "__import__": native_import,
            }
            sys.modules[name] = module
            created.append(name)
            exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
            modules[key] = module
    except Exception:
        for name in created:
            sys.modules.pop(name, None)
        raise
    return modules["rejection"]


def native():
    """Return (callable, provenance), or (None, the precise availability reason)."""
    identity = {"kind": "installed_vllm", "status": "unavailable", "files": {}}
    skip_reason = os.environ.get("FLAGGEMS_REJECTION_NATIVE_SKIP_REASON")
    if skip_reason:
        identity.update(status="explicitly_disabled", reason=skip_reason)
        return None, identity
    try:
        distribution = importlib.metadata.distribution("vllm")
        root = Path(distribution.locate_file("vllm")).resolve()
        path = root / _RELATIVE
        identity.update(version=distribution.version, root=str(root))
        if not path.is_file():
            raise ImportError("Installed vLLM lacks " + _RELATIVE)
        identity["files"][_RELATIVE] = _file_identity(path)
        try:
            module = importlib.import_module(_MODULE)
            identity["loading"] = "package_import"
        except Exception as error:
            identity["package_import_error"] = f"{type(error).__name__}: {error}"
            module = _private_source(root, identity["files"])
            identity["loading"] = "unchanged_installed_source_with_private_imports"
        function = module.rejection_sample
        actual_path = Path(inspect.getfile(function)).resolve()
        if actual_path != path:
            raise ImportError("vLLM sampler redirected to " + str(actual_path))
        signature = inspect.signature(function)
        if not set(_COMMON) <= set(signature.parameters):
            raise TypeError("Installed rejection_sample has an incompatible signature")
        legacy = "draft_logits_cache" not in signature.parameters
        if legacy and "draft_logits" not in signature.parameters:
            raise TypeError("Installed sampler lacks draft_logits/cache input")
        # Record the original sampling helper that the sampler actually calls.
        helper = getattr(module, "gumbel_block_argmax", None)
        helper = getattr(helper, "fn", helper)
        if (
            not callable(helper)
            or Path(inspect.getfile(helper)).resolve() != root / _GUMBEL
        ):
            raise ImportError("vLLM sampling helper was redirected outside its source")
        identity["files"][_GUMBEL] = _file_identity(root / _GUMBEL)
        identity.update(
            status="resolved_unqualified",
            signature=str(signature),
            callable_module=function.__module__,
            callable_file=str(actual_path),
            adapter="cache.values_to_draft_logits" if legacy else "none",
            supports_placeholder=not legacy,
        )

        def invoke(**kwargs):
            if legacy:
                kwargs = dict(kwargs)
                cache = kwargs.pop("draft_logits_cache")
                if (
                    cache is not None
                    and getattr(cache, "row_mapping", None) is not None
                ):
                    raise NotImplementedError(
                        "Installed sampler has no cache row_mapping"
                    )
                kwargs["draft_logits"] = None if cache is None else cache.values
            return function(**kwargs)

        return invoke, identity
    except Exception as error:
        identity["reason"] = f"{type(error).__name__}: {error}"
        return None, identity
