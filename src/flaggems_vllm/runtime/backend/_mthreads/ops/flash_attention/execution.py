# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend-local launch recording, without patching the Triton runtime."""

from contextlib import contextmanager
from contextvars import ContextVar

_recorder = ContextVar("mthreads_attention_compile_recorder", default=None)
_config = ContextVar("mthreads_attention_config", default=None)
_call_info = ContextVar("mthreads_attention_call_info", default=None)


def launch(kernel, grid, *args, **kwargs):
    recorder = _recorder.get()
    if recorder is not None:
        result = kernel.warmup(*args, grid=grid, **kwargs)
    else:
        result = kernel[grid](*args, **kwargs)
    if recorder is not None:
        result._init_handles()
        recorder.append(result)
    return result


@contextmanager
def compile_only():
    kernels = []
    token = _recorder.set(kernels)
    try:
        yield kernels
    finally:
        _recorder.reset(token)


def is_compiling():
    return _recorder.get() is not None


@contextmanager
def use_config(config):
    token = _config.set(config)
    try:
        yield
    finally:
        _config.reset(token)


def current_config():
    return _config.get()


@contextmanager
def capture_call():
    info = {}
    token = _call_info.set(info)
    try:
        yield info
    finally:
        _call_info.reset(token)


def record_call(problem, default, key):
    info = _call_info.get()
    if info is not None:
        info.update(problem=problem, default=default, key=key)
