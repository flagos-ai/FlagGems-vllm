# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm
from tests.test_flash_attn_varlen_func_w8a8_int8 import _run_case

pytestmark = [
    pytest.mark.flash_attn_varlen_func_w8a8,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "metax", reason="MetaX D256 decode scheduling"
    ),
]


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_decode_split_bound(paged, broadcast, dtype):
    _run_case(
        [0, 1, 1, 1, 1, 1, 1, 1] * 4,
        [1, 0, 1, 2, 31, 64, 129, 2051] * 4,
        dim=256,
        heads=4,
        kvheads=1,
        paged=paged,
        causal=True,
        dtype=dtype,
        broadcast_scales=broadcast,
        max_query_bound=1,
        max_key_bound=73728,
        metax_kwargs={"num_splits": 32},
        return_lse=True,
    )


@pytest.mark.parametrize("nk", [65536])
def test_decode_split_bound_long_context(nk):
    _run_case(
        [1] * 32,
        [nk] + [1] * 31,
        dim=256,
        heads=4,
        kvheads=1,
        paged=True,
        causal=True,
        broadcast_scales=True,
        max_key_bound=73728,
        metax_kwargs={"num_splits": 32},
        return_lse=True,
    )


@pytest.mark.parametrize("splits", [2, 3, 7, 32])
def test_decode_active_split_boundaries(splits):
    _run_case(
        [0, 1, 1] * 11,
        [1, 65, 513] * 11,
        dim=256,
        heads=8,
        kvheads=2,
        paged=True,
        causal=True,
        strided=True,
        window=(64, 0),
        cap=5.0,
        alibi=torch.tensor([0.02] * 8, device=flaggems_vllm.device),
        max_query_bound=1,
        max_key_bound=73728,
        metax_kwargs={"num_splits": splits},
        return_lse=True,
    )


def test_decode_all_empty_kv():
    _run_case(
        [1] * 33,
        [0] * 33,
        dim=256,
        heads=4,
        kvheads=1,
        paged=True,
        causal=True,
        max_query_bound=1,
        max_key_bound=73728,
        metax_kwargs={"num_splits": 32},
    )


@pytest.mark.parametrize("group", [1, 4, 8, 16])
@pytest.mark.parametrize("paged", [False, True])
def test_decode_single_value_empty_requests(group, paged):
    _run_case(
        [0, 1, 1] * 11,
        [1, 0, 1] * 11,
        dim=256,
        heads=group,
        kvheads=1,
        paged=paged,
        dtype=torch.float16,
        strided=True,
        broadcast_scales=True,
        causal=True,
        max_query_bound=1,
        max_key_bound=1,
        metax_kwargs={"num_splits": 32},
        return_lse=False,
    )


@pytest.mark.parametrize("batch", [512, 513])
def test_decode_large_batch_boundary(batch):
    _run_case(
        [1] * batch,
        [1] * batch,
        dim=256,
        heads=4,
        kvheads=1,
        paged=True,
        causal=True,
        broadcast_scales=True,
        max_key_bound=73728,
        metax_kwargs={"num_splits": 32},
    )


@pytest.mark.parametrize("length", [32, 129, 2048])
def test_decode_uniform_split_lengths(length):
    _run_case(
        [1] * 33,
        [length] * 33,
        dim=256,
        heads=4,
        kvheads=1,
        paged=True,
        causal=True,
        broadcast_scales=True,
        max_key_bound=73728,
        metax_kwargs={"num_splits": 32},
    )


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_single_value_all_empty_kv_and_descales(paged, dtype):
    # Actual max K is zero, so K/V descales have zero logical blocks.
    # A padded bound of one must not cause a read from their empty storage.
    _run_case(
        [1] * 33,
        [0] * 33,
        dim=256,
        heads=4,
        kvheads=1,
        dtype=dtype,
        paged=paged,
        causal=True,
        max_query_bound=1,
        max_key_bound=1,
        metax_kwargs={"num_splits": 32},
        return_lse=False,
    )
