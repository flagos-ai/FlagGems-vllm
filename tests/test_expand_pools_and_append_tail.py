# SPDX-License-Identifier: Apache-2.0
"""Exact pool expansion, input-storage guards, errors and isolated graph replay."""

import importlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import flaggems_vllm
from flaggems_vllm import expand_pools_and_append_tail

pytestmark = [pytest.mark.gpu, pytest.mark.expand_pools_and_append_tail]

DEVICE = flaggems_vllm.device
INT32_MAX = 2**31 - 1
GUARD_SIZE = 11


def expand_pool_oracle(pool_ids, seq_lens, pool_size):
    """Construct every output element using Python integers and CPU int64 inputs."""
    pools = pool_ids.detach().to(device="cpu", dtype=torch.int64)
    lengths = seq_lens.detach().to(device="cpu", dtype=torch.int64)
    rows, groups = pools.shape
    out_cols = groups * pool_size + pool_size - 1
    result = []
    for row_pools, length in zip(pools.tolist(), lengths.tolist()):
        for pool in row_pools:
            for offset in range(pool_size):
                result.append(pool * pool_size + offset if pool >= 0 else -1)
        tail_start = (length // pool_size) * pool_size
        tail_count = length % pool_size
        for offset in range(pool_size - 1):
            result.append(tail_start + offset if offset < tail_count else -1)
    return torch.tensor(result, dtype=torch.int32, device="cpu").reshape(rows, out_cols)


def _storage_bytes(tensor):
    return tensor.detach().cpu().view(torch.uint8).clone()


def _make_inputs(rows, groups, pool_size, pool_dtype, seq_dtype, revision, layout):
    gap = 5 if layout != "contiguous" else 0
    offset = GUARD_SIZE if layout == "offset" else 0
    row_stride = max(groups + gap, 1)
    pool_storage = torch.full(
        (offset + max(rows, 1) * row_stride + GUARD_SIZE,),
        -1234567 - revision,
        dtype=pool_dtype,
        device=DEVICE,
    )
    pools = pool_storage.as_strided((rows, groups), (row_stride, 1), offset)
    seq_storage = torch.full(
        (offset + rows + GUARD_SIZE,),
        -7654321 - revision,
        dtype=seq_dtype,
        device=DEVICE,
    )
    lengths = seq_storage.as_strided((rows,), (1,), offset)
    max_pool = (INT32_MAX - (pool_size - 1)) // pool_size
    pool_values = []
    seq_values = []
    # Mix padding, duplicate IDs, negative sentinels and the largest safe index.
    min_pool = -(2**63) if pool_dtype == torch.int64 else -(2**31)
    choices = [0, max_pool, -1, -7, min_pool, revision + 2, revision + 2]
    tail_cases = [0, pool_size - 1, pool_size, pool_size + 1, 4 * pool_size]
    for row in range(rows):
        pool_values.append(
            [
                choices[(row + group + revision) % len(choices)]
                for group in range(groups)
            ]
        )
        seq_values.append(tail_cases[(row + revision) % len(tail_cases)])
    if rows:
        pools.copy_(torch.tensor(pool_values, dtype=pool_dtype, device=DEVICE))
        lengths.copy_(torch.tensor(seq_values, dtype=seq_dtype, device=DEVICE))
    return pools, lengths, pool_storage, seq_storage


def _assert_output(result, pools, lengths, pool_size, storages):
    assert result.dtype == torch.int32
    assert result.device == pools.device
    assert result.shape == (
        pools.shape[0],
        pools.shape[1] * pool_size + pool_size - 1,
    )
    assert result.is_contiguous()
    assert result is not pools and result is not lengths
    for storage in storages:
        assert (
            result.untyped_storage().data_ptr() != storage.untyped_storage().data_ptr()
        )
    torch.testing.assert_close(
        result.cpu(), expand_pool_oracle(pools, lengths, pool_size), rtol=0, atol=0
    )


def _check_case(rows, groups, pool_size, pool_dtype, seq_dtype, layout):
    previous = None
    for revision in range(2):
        pools, lengths, pool_storage, seq_storage = _make_inputs(
            rows, groups, pool_size, pool_dtype, seq_dtype, revision, layout
        )
        storages = (pool_storage, seq_storage)
        before = [_storage_bytes(storage) for storage in storages]
        result = expand_pools_and_append_tail(pools, lengths, pool_size)
        _assert_output(result, pools, lengths, pool_size, storages)
        for storage, expected_bytes in zip(storages, before):
            assert torch.equal(_storage_bytes(storage), expected_bytes)
        if previous is not None and result.numel():
            assert (
                result.untyped_storage().data_ptr()
                != previous.untyped_storage().data_ptr()
            )
        previous = result


@pytest.mark.expand_pools_and_append_tail
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("pool_size", [1, 4, 16])
@pytest.mark.parametrize(
    "rows,groups,padded",
    [(1, 1, False), (7, 3, True), (24, 128, False), (257, 3, True)],
)
def test_matches_reference(dtype, pool_size, rows, groups, padded):
    _check_case(
        rows, groups, pool_size, dtype, dtype, "row_gap" if padded else "contiguous"
    )


@pytest.mark.parametrize("rows,groups,pool_size", [(0, 3, 4), (7, 0, 1)])
def test_empty_output(rows, groups, pool_size):
    _check_case(rows, groups, pool_size, torch.int32, torch.int32, "contiguous")


# Paired cases bound compile count while covering empty, tail-only and model paths.
@pytest.mark.parametrize(
    "rows,groups,pool_size,pool_dtype,seq_dtype,layout",
    [
        pytest.param(
            0, 0, 1, torch.int64, torch.int32, "offset", id="empty-zero-width"
        ),
        pytest.param(0, 1, 2, torch.int32, torch.int64, "row_gap", id="empty-rows"),
        pytest.param(
            1, 0, 2, torch.int64, torch.int64, "offset", id="tail-only-single"
        ),
        pytest.param(7, 0, 8, torch.int32, torch.int64, "offset", id="tail-only-mixed"),
        pytest.param(512, 0, 1, torch.int64, torch.int32, "offset", id="zero-width"),
        pytest.param(
            7, 1, 2, torch.int64, torch.int32, "offset", id="mixed-single-pool"
        ),
        pytest.param(7, 3, 8, torch.int32, torch.int64, "offset", id="mixed-offset"),
        pytest.param(512, 3, 4, torch.int64, torch.int64, "row_gap", id="many-rows"),
        pytest.param(1, 128, 16, torch.int32, torch.int32, "offset", id="large-groups"),
        pytest.param(1, 512, 4, torch.int32, torch.int32, "contiguous", id="model-r1"),
        pytest.param(7, 512, 4, torch.int64, torch.int64, "offset", id="model-r7"),
        pytest.param(24, 512, 4, torch.int32, torch.int64, "row_gap", id="model-r24"),
        pytest.param(257, 512, 4, torch.int64, torch.int32, "offset", id="model-r257"),
    ],
)
def test_contract_matrix(rows, groups, pool_size, pool_dtype, seq_dtype, layout):
    _check_case(rows, groups, pool_size, pool_dtype, seq_dtype, layout)


@pytest.mark.parametrize("pool_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_int32_boundary_and_uncropped_history(pool_size, dtype):
    max_pool = (INT32_MAX - (pool_size - 1)) // pool_size
    min_pool = -(2**63) if dtype == torch.int64 else -(2**31)
    for revision in range(2):
        pools = torch.tensor(
            [[max_pool - revision, min_pool, max_pool - revision]],
            dtype=dtype,
            device=DEVICE,
        )
        lengths = torch.tensor([INT32_MAX - revision], dtype=dtype, device=DEVICE)
        before = [_storage_bytes(value) for value in (pools, lengths)]
        result = expand_pools_and_append_tail(pools, lengths, pool_size)
        _assert_output(result, pools, lengths, pool_size, (pools, lengths))
        for value, expected_bytes in zip((pools, lengths), before):
            assert torch.equal(_storage_bytes(value), expected_bytes)
        # A zero sequence length must not truncate even the largest history pool.
        lengths.zero_()
        before = [_storage_bytes(value) for value in (pools, lengths)]
        result = expand_pools_and_append_tail(pools, lengths, pool_size)
        _assert_output(result, pools, lengths, pool_size, (pools, lengths))
        for value, expected_bytes in zip((pools, lengths), before):
            assert torch.equal(_storage_bytes(value), expected_bytes)


def test_empty_max_pool_size():
    # Exercise the largest legal scalar without allocating a huge tensor.
    _check_case(0, 0, INT32_MAX, torch.int32, torch.int64, "offset")


def test_output_guard_canaries(monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.expand_pools_and_append_tail")
    rows, groups, pool_size = 7, 3, 8
    out_cols = groups * pool_size + pool_size - 1
    original_torch = module.torch
    target_device = torch.empty(0, device=DEVICE).device
    allocations = []

    class OutputAllocator:
        def __getattr__(self, name):
            return getattr(original_torch, name)

        def empty(self, size, *args, **kwargs):
            if (
                tuple(size) == (rows, out_cols)
                and kwargs.get("dtype") == torch.int32
                and torch.device(kwargs.get("device")) == target_device
            ):
                backing = original_torch.full(
                    (GUARD_SIZE + rows * out_cols + GUARD_SIZE,),
                    -777 - len(allocations),
                    device=DEVICE,
                    dtype=torch.int32,
                )
                allocations.append((backing, backing.clone()))
                return backing[GUARD_SIZE:-GUARD_SIZE].reshape(rows, out_cols)
            return original_torch.empty(size, *args, **kwargs)

    # Scope the allocator replacement to this operator module, never torch itself.
    monkeypatch.setattr(module, "torch", OutputAllocator())
    for revision in range(2):
        pools, lengths, *storages = _make_inputs(
            rows, groups, pool_size, torch.int32, torch.int64, revision, "offset"
        )
        before = [_storage_bytes(storage) for storage in storages]
        result = expand_pools_and_append_tail(pools, lengths, pool_size)
        _assert_output(result, pools, lengths, pool_size, storages)
        for storage, expected_bytes in zip(storages, before):
            assert torch.equal(_storage_bytes(storage), expected_bytes)
    assert len(allocations) == 2
    for backing, before in allocations:
        assert torch.equal(backing[:GUARD_SIZE], before[:GUARD_SIZE])
        assert torch.equal(backing[-GUARD_SIZE:], before[-GUARD_SIZE:])


@pytest.mark.parametrize("pool_size", [4, 16])
@pytest.mark.skipif(
    DEVICE != "cuda" or flaggems_vllm.vendor_name in ("ascend", "mthreads"),
    reason="separate CUDA graph functionality requires the CUDA graph API",
)
def test_graph_replay_refreshes_history_and_tail(pool_size):
    pools = torch.tensor(
        [[0, 1, -1], [3, -1, 2], [5, 0, 1]], device=DEVICE, dtype=torch.int32
    )
    lengths = torch.tensor(
        [pool_size, pool_size + 1, 2 * pool_size - 1], device=DEVICE, dtype=torch.int32
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expand_pools_and_append_tail(pools, lengths, pool_size)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = expand_pools_and_append_tail(pools, lengths, pool_size)
    for revision in range(3):
        pools.copy_(
            torch.tensor(
                [[revision + 1, -1, 2], [-1, revision + 3, 0], [7, 2, revision]],
                device=DEVICE,
                dtype=torch.int32,
            )
        )
        lengths.copy_(
            torch.tensor(
                [2 * pool_size + revision, 3 * pool_size - revision - 1, pool_size + 2],
                device=DEVICE,
                dtype=torch.int32,
            )
        )
        expected = expand_pool_oracle(pools, lengths, pool_size)
        before = [_storage_bytes(value) for value in (pools, lengths)]
        result.fill_(-777)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
        for value, expected_bytes in zip((pools, lengths), before):
            assert torch.equal(_storage_bytes(value), expected_bytes)


@pytest.mark.parametrize(
    "invalid",
    [
        "pool_size",
        "negative_pool_size",
        "bool_pool_size",
        "float_pool_size",
        "oversize_pool_size",
        "output_columns",
        "rank",
        "seq_rank",
        "rows",
        "dtype",
        "seq_dtype",
        "bool_dtype",
        "seq_bool_dtype",
        "columns",
        "seq_stride",
    ],
)
def test_rejects_invalid_contract(invalid):
    pools = torch.zeros((2, 3), device=DEVICE, dtype=torch.int32)
    lengths = torch.ones(2, device=DEVICE, dtype=torch.int32)
    pool_size = 4
    error = ValueError
    if invalid == "pool_size":
        pool_size = 0
    elif invalid == "negative_pool_size":
        pool_size = -1
    elif invalid == "bool_pool_size":
        pool_size = True
    elif invalid == "float_pool_size":
        pool_size = 4.0
    elif invalid == "oversize_pool_size":
        pool_size = INT32_MAX + 1
    elif invalid == "output_columns":
        pools = torch.empty((0, 1), device=DEVICE, dtype=torch.int32)
        lengths = torch.empty(0, device=DEVICE, dtype=torch.int32)
        pool_size = INT32_MAX
    elif invalid == "rank":
        pools = pools[0]
    elif invalid == "seq_rank":
        lengths = lengths[:, None]
    elif invalid == "rows":
        lengths = lengths[:1]
    elif invalid in ("dtype", "bool_dtype"):
        pools = pools.to(torch.float32 if invalid == "dtype" else torch.bool)
        error = TypeError
    elif invalid in ("seq_dtype", "seq_bool_dtype"):
        lengths = lengths.to(torch.float32 if invalid == "seq_dtype" else torch.bool)
        error = TypeError
    elif invalid == "columns":
        pools = torch.zeros((2, 6), device=DEVICE, dtype=torch.int32)[:, ::2]
    else:
        lengths = torch.ones(4, device=DEVICE, dtype=torch.int32)[::2]
    with pytest.raises(error):
        expand_pools_and_append_tail(pools, lengths, pool_size)


@pytest.mark.parametrize("cpu_input", ["pools", "lengths"])
def test_rejects_cross_device(cpu_input):
    pools = torch.zeros((2, 3), device=DEVICE, dtype=torch.int32)
    lengths = torch.ones(2, device=DEVICE, dtype=torch.int64)
    if cpu_input == "pools":
        pools = pools.cpu()
    else:
        lengths = lengths.cpu()
    with pytest.raises(NotImplementedError):
        expand_pools_and_append_tail(pools, lengths, 4)


def test_rejects_cpu():
    with pytest.raises(NotImplementedError, match="accelerator"):
        expand_pools_and_append_tail(
            torch.zeros((2, 3), dtype=torch.int32), torch.ones(2, dtype=torch.int32), 4
        )


@pytest.mark.skipif(
    flaggems_vllm.vendor_name not in ("nvidia", "metax", "hygon"),
    reason="CUDA-interface stream launch regression",
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_nondefault_streams_and_offsets(dtype, offset):
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    pending = []
    for revision in range(3):
        for stream in streams:
            with torch.cuda.stream(stream):
                storage = torch.full((80,), -99, dtype=dtype, device=DEVICE)
                seq_storage = torch.full((16,), -77, dtype=dtype, device=DEVICE)
                pools = storage.as_strided((7, 3), (9, 1), offset)
                lengths = seq_storage[offset : offset + 7]
                pools.copy_(
                    torch.tensor(
                        [[revision, -1, row + 3] for row in range(7)],
                        dtype=dtype,
                        device=DEVICE,
                    )
                )
                lengths.copy_(
                    torch.tensor(
                        [4 + (row + revision) % 4 for row in range(7)],
                        dtype=dtype,
                        device=DEVICE,
                    )
                )
                result = expand_pools_and_append_tail(pools, lengths, 4)
                consumer = result.clone()
                pending.append((result, consumer, pools, lengths, storage, seq_storage))
    for stream in streams:
        stream.synchronize()
    for result, consumer, pools, lengths, storage, seq_storage in pending:
        expected = expand_pool_oracle(pools, lengths, 4)
        _assert_output(result, pools, lengths, 4, (storage, seq_storage))
        torch.testing.assert_close(consumer.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "nvidia"
    or os.environ.get("EXPAND_SECOND_DEVICE_ALLOWED") != "1",
    reason="requires a separately reserved second CUDA device",
)
def test_current_device_differs_from_inputs():
    if torch.cuda.device_count() < 2:
        pytest.skip("second CUDA device is not exposed")
    for input_device, caller_device in [(0, 1), (1, 0)]:
        pools = torch.tensor(
            [[1, -1, 3]], dtype=torch.int32, device=f"cuda:{input_device}"
        )
        lengths = torch.tensor([7], dtype=torch.int64, device=f"cuda:{input_device}")
        with torch.cuda.device(caller_device):
            for _ in range(2):
                result = expand_pools_and_append_tail(pools, lengths, 4)
                torch.testing.assert_close(
                    result.cpu(), expand_pool_oracle(pools, lengths, 4), rtol=0, atol=0
                )
                assert torch.cuda.current_device() == caller_device


@pytest.mark.skipif(
    flaggems_vllm.vendor_name not in ("nvidia", "metax", "hygon"),
    reason="CUDA-interface concurrent launch regression",
)
def test_concurrent_cold_and_warm_calls(monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.expand_pools_and_append_tail")
    if not module._CACHE_LAUNCH:
        pytest.skip("compiled launch cache is disabled on this compiler")
    monkeypatch.setattr(module, "_launch_cache", {})
    monkeypatch.setattr(module, "_LAUNCH_CACHE_LIMIT", 4)
    torch.cuda.init()
    for groups in range(1, 5):
        expand_pools_and_append_tail(
            torch.zeros((5, groups), device=DEVICE, dtype=torch.int64),
            torch.ones(5, device=DEVICE, dtype=torch.int64),
            4,
        )
    assert len(module._launch_cache) == 4
    barrier = threading.Barrier(4)

    def worker(worker_id):
        stream = torch.cuda.Stream()
        barrier.wait()
        for revision in range(3):
            with torch.cuda.stream(stream):
                pools = torch.tensor(
                    [
                        [worker_id + revision, -1, row, 2, 5] + [6] * worker_id
                        for row in range(5)
                    ],
                    dtype=torch.int64,
                    device=DEVICE,
                )
                lengths = torch.tensor(
                    [4 + (row + revision) % 4 for row in range(5)],
                    dtype=torch.int64,
                    device=DEVICE,
                )
                result = expand_pools_and_append_tail(pools, lengths, 4)
                consumer = result.clone()
                torch.testing.assert_close(
                    consumer.cpu(),
                    expand_pool_oracle(pools, lengths, 4),
                    rtol=0,
                    atol=0,
                )
        stream.synchronize()

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(worker, range(4)))
    assert len(module._launch_cache) <= 4


@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "hygon", reason="HCU storage-range specialization"
)
def test_hcu_storage_specialization_and_large_row_gap(monkeypatch):
    from triton.backends.hcu.compiler import HIPBackend

    module = importlib.import_module("flaggems_vllm.ops.expand_pools_and_append_tail")
    if not module._CACHE_LAUNCH:
        pytest.skip("compiled launch cache is disabled on this compiler")
    allocation_bytes = 2**31 + 8192
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < allocation_bytes + 256 * 1024**2:
        pytest.skip("storage-range test requires 2 GiB of free device memory")
    backing = torch.empty(allocation_bytes // 4, dtype=torch.int32, device=DEVICE)
    small_pools = torch.tensor(
        [[1, -1, 3] + [-91] * 6, [4, 2, -17] + [-91] * 6],
        dtype=torch.int32,
        device=DEVICE,
    )[:, :3]
    small_lengths = torch.tensor([5, 7], dtype=torch.int32, device=DEVICE)
    large_pools = backing.as_strided((2, 3), (9, 1), 16)
    large_lengths = backing[128:130]
    large_output = backing[256:286].view(2, 15)
    specialize = HIPBackend.get_tensor_specialization
    if specialize(small_pools) == specialize(large_pools):
        pytest.skip("HCU buffer-range specialization is disabled")
    monkeypatch.setattr(module, "_launch_cache", {})

    def check(pools, lengths):
        expected = expand_pool_oracle(pools, lengths, 4)
        original_pools, original_lengths = pools.cpu(), lengths.cpu()
        for _ in range(2):
            result = expand_pools_and_append_tail(pools, lengths, 4)
            torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(pools.cpu(), original_pools, rtol=0, atol=0)
        torch.testing.assert_close(lengths.cpu(), original_lengths, rtol=0, atol=0)

    check(small_pools, small_lengths)
    assert len(module._launch_cache) == 1
    large_pools.copy_(small_pools)
    check(large_pools, small_lengths)
    assert len(module._launch_cache) == 2
    large_lengths.copy_(small_lengths)
    check(small_pools, large_lengths)
    assert len(module._launch_cache) == 3

    backing[240:256].fill_(-119)
    backing[286:302].fill_(-119)
    original_empty = torch.empty

    def allocate(shape, *args, **kwargs):
        if (
            isinstance(shape, (tuple, list))
            and tuple(shape) == (2, 15)
            and kwargs.get("dtype") == torch.int32
        ):
            return large_output
        return original_empty(shape, *args, **kwargs)

    with monkeypatch.context() as allocator:
        allocator.setattr(torch, "empty", allocate)
        check(small_pools, small_lengths)
    assert len(module._launch_cache) == 4
    for guard in (backing[240:256], backing[286:302]):
        torch.testing.assert_close(
            guard.cpu(), torch.full((16,), -119, dtype=torch.int32), rtol=0, atol=0
        )

    # The second row starts beyond 2 GiB; row addressing must remain int64.
    gap_pools = backing.as_strided((2, 3), (2**29 + 16, 1))
    gap_pools.copy_(small_pools)
    check(gap_pools, small_lengths)
    assert len(module._launch_cache) == 5
