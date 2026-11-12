# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit CUDA dispatch override for symm_mem.fused_matmul_reduce_scatter.

Registration is process-wide, opt-in, and independent of ATen enable/use_gems.
It does not initialize distributed state or free collective workspaces. Callers
must enable it in every participating worker before using the collective.
"""

import importlib
import inspect
import threading
from contextlib import contextmanager

import torch

_LOCK = threading.RLock()
_LIBRARY = None
_REFERENCES = 0
_ENABLED = False
_OP_NAME = "fused_matmul_reduce_scatter"
_IMPL_KEY = f"symm_mem/{_OP_NAME}/CUDA"


def _acquire():
    global _LIBRARY, _REFERENCES
    with _LOCK:
        if _REFERENCES == 0:
            implementation_module = importlib.import_module(
                "flaggems_vllm.ops.fused_matmul_reduce_scatter"
            )
            implementation_module._get_backend()
            # Load PyTorch's schema and original implementations first. Defining
            # a second schema would conflict with later imports by vLLM/PyTorch.
            importlib.import_module("torch.distributed._symmetric_memory")
            if (
                "allow_override"
                not in inspect.signature(torch.library.Library.impl).parameters
            ):
                raise NotImplementedError(
                    "this PyTorch build does not support Library.impl(allow_override=True); "
                    "use a build with this API, or call flaggems_vllm.fused_matmul_reduce_scatter directly"
                )
            implementation = implementation_module.fused_matmul_reduce_scatter
            library = torch.library.Library("symm_mem", "IMPL")
            original_python_registration = _IMPL_KEY in getattr(
                torch.library, "_impls", ()
            )
            try:
                library.impl(_OP_NAME, implementation, "CUDA", allow_override=True)
                # PyTorch 2.8 uses a set (not a refcount) for Python registration
                # bookkeeping. The original owner must retain its entry when our
                # temporary override is destroyed. C++ still owns the override's
                # registration handle and restores the underlying kernel on reset.
                if original_python_registration:
                    library._op_impls.discard(_IMPL_KEY)
            except Exception:
                library._destroy()
                raise
            _LIBRARY = library
        _REFERENCES += 1


def _release():
    global _LIBRARY, _REFERENCES
    with _LOCK:
        if _REFERENCES == 1:
            _LIBRARY._destroy()
            _LIBRARY = None
        _REFERENCES -= 1


def enable_symm_mem():
    """Route this symm_mem op's CUDA calls to FlagGems-vllm until disabled.

    Repeated calls are idempotent. Existing torch.ops overload references continue
    to use the dispatcher. Only CUDA is overridden; other dispatch keys and the
    operator schema remain owned by PyTorch. Unsupported TLE inputs raise rather
    than falling back to Torch computation.
    """
    global _ENABLED
    with _LOCK:
        if not _ENABLED:
            _acquire()
            _ENABLED = True


def disable_symm_mem():
    """Release persistent activation, restoring PyTorch after all scopes exit.

    This does not free cached communication resources. Use the collective cache
    clearing API separately, on every group member, before destroying its group.
    """
    global _ENABLED
    with _LOCK:
        if _ENABLED:
            _release()
            _ENABLED = False


@contextmanager
def use_symm_mem():
    """Temporarily enable symm_mem routing, including nested/exceptional scopes.

    The registration affects all threads in this process while a scope or
    persistent activation is alive. It does not enable any ATen replacements.
    """
    _acquire()
    try:
        yield
    finally:
        _release()
