# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import importlib

import pytest
import torch
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import make_backend
from triton.runtime.jit import compute_cache_key, create_function_from_signature

import flaggems_vllm as ops

pytestmark = pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX regression")


@pytest.mark.parametrize(
    "module_name,kernel_name,parameter",
    [
        pytest.param(
            "flaggems_vllm.runtime.backend._metax.ops.sparse_mla_decode",
            "_reduce_split_kernel",
            "Bsz",
            marks=pytest.mark.flash_mla_with_kvcache,
        )
    ],
)
def test_unused_batch_native_binding_reuses_specialization(
    module_name, kernel_name, parameter
):
    """Exercise the installed native binder, without launching a kernel."""
    kernel = getattr(importlib.import_module(module_name), kernel_name)
    backend = make_backend(GPUTarget("maca", 80, 64))
    binder = create_function_from_signature(kernel.signature, kernel.params, backend)
    values = dict.fromkeys(kernel.arg_names, None)
    keys = set()
    for batch in (*range(1, 65), 1024, 4032, 4034, 4094, 4096):
        values[parameter] = batch
        _, specialization, options = binder(**values)
        keys.add(str(compute_cache_key({}, specialization, options)))
    assert len(keys) == 1


def _cache_keys(kernel):
    return set(kernel.device_caches[torch.cuda.current_device()][0])


@pytest.mark.flash_mla_with_kvcache
@pytest.mark.parametrize(
    "real_heads,physical_heads",
    [(8, 16), (16, 16), (8, 8)],
    ids=["h8", "h16", "native_h8"],
)
def test_public_split_decode_new_batches_reuse_reduce_kernel(
    record_property, real_heads, physical_heads
):
    from .test_sparse_mla_metax import _check_split_decode as check_public

    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.sparse_mla_decode"
    )
    kernel = module._reduce_split_kernel
    assert ops.flash_mla_with_kvcache.__module__.endswith(
        "_metax.ops.sparse_mla_decode"
    )
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        check_public(1, real_heads, physical_heads)
        warm_keys = _cache_keys(kernel)
        assert warm_keys, "The public split-decode path did not hit its reduce kernel"
        observed = [(1, len(warm_keys))]
        for batch in (3, 7, 17, 31, 63):
            check_public(batch, real_heads, physical_heads)
            assert _cache_keys(kernel) == warm_keys
            observed.append((batch, len(warm_keys)))
        record_property("native_cache_keys_by_batch", repr(observed))
