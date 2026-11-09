# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Contract tests for vLLM's in-place cooperative top-k interface."""

from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils
from . import conftest as cfg

pytestmark = pytest.mark.cooperative_topk
device = flaggems_vllm.device
torch_device_fn = flaggems_vllm.runtime.torch_device_fn
gpu = pytest.mark.skipif(
    device not in ("cuda", "musa", "npu") or not torch_device_fn.is_available(),
    reason="CUDA, MUSA or NPU device required",
)
KS = [512, 1024, 2048]
SHAPES = [
    (1, 513),
    (4, 4097),
    (4, 8192),
    (4, 8193),
    (4, 16384),
    (16, 16384),
    (17, 16384),
    (64, 16384),
    (4, 16385),
    (16, 16385),
    (4, 32768),
    (4, 32769),
    (1, 65539),
]
if not cfg.QUICK_MODE:
    SHAPES += [
        (16, 65536),
        (17, 65536),
        (64, 32771),
        (1, 262144),
        (4, 1048576),
        (1, 1048577),
    ]


def _buffers(logits, lengths, k):
    lengths = torch.tensor(lengths, dtype=torch.int32, device=logits.device)
    output = torch.full(
        (logits.shape[0], k), -17, dtype=torch.int32, device=logits.device
    )
    workspace = torch.empty(
        logits.shape[0] * 16384, dtype=torch.uint8, device=logits.device
    )
    return lengths, output, workspace


def _check(logits, lengths, output, k, reference_on_cpu=False):
    # Compare selected values, not tie indices or output order. Do not infer
    # padding from score values: valid entries may themselves contain -inf.
    reference_logits = logits.cpu() if reference_on_cpu else utils.to_reference(logits)
    for row, length in enumerate(lengths.reshape(-1).cpu().tolist()):
        length = max(length, 0)
        indices = output[row].to(device=reference_logits.device, dtype=torch.int64)
        if length <= k:
            expected = torch.full(
                (k,), -1, dtype=torch.int64, device=reference_logits.device
            )
            expected[:length] = torch.arange(length, device=reference_logits.device)
            utils.gems_assert_equal(indices, expected)
            continue
        assert torch.all((indices >= 0) & (indices < length))
        assert indices.unique().numel() == k
        selected = reference_logits[row, indices]
        expected = reference_logits[row, :length].topk(k, sorted=False).values
        # CUDA and CPU can order negative NaNs differently. The API promises
        # a value multiset, so check NaN multiplicity and exact non-NaN values.
        selected_nan = torch.isnan(selected)
        expected_nan = torch.isnan(expected)
        utils.gems_assert_equal(selected_nan.sum(), expected_nan.sum())
        utils.gems_assert_equal(
            selected[~selected_nan].sort().values,
            expected[~expected_nan].sort().values,
        )


@gpu
@pytest.mark.cooperative_topk
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("shape", SHAPES)
def test_cooperative_topk_random(shape, k):
    torch.manual_seed(42)
    rows, width = shape
    logits = torch.randn(rows, width, device=device)
    lengths, output, workspace = _buffers(logits, [width] * rows, k)
    before = logits.clone()
    address = output.data_ptr()
    assert (
        flaggems_vllm.cooperative_topk(logits, lengths, output, workspace, k, width)
        is None
    )
    assert output.data_ptr() == address
    _check(logits, lengths, output, k)
    torch.testing.assert_close(logits, before, rtol=0, atol=0)


@gpu
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("width", [4097, 16384, 65539])
def test_mixed_lengths_rank_two_poison_padding_and_workspace_reuse(k, width):
    # Rank-two lengths are flattened, without persistent_topk's MTP adjustment.
    host_lengths = [-3, 0, 1, k - 1, k, k + 1, width - 1, width]
    storage = torch.full((8, width + 13), float("nan"), device=device)
    logits = storage[:, :width]
    lengths, output, workspace = _buffers(logits, host_lengths, k)
    lengths = lengths.view(2, 4)
    scratch_storage = torch.full(
        (workspace.numel() + 64,), 251, dtype=torch.uint8, device=device
    )
    workspace = scratch_storage[32:-32]
    output_storage = torch.full(
        (output.numel() + 64,), -71, dtype=torch.int32, device=device
    )
    output = output_storage[32:-32].view(8, k)
    original_lengths = lengths.clone()
    for seed in (17, 29):
        torch.manual_seed(seed)
        for row, length in enumerate(host_lengths):
            logits[row, : max(length, 0)] = torch.randn(max(length, 0), device=device)
        workspace.fill_(seed)
        output.fill_(-17)
        # max_seq_len is compatibility metadata and must not truncate rows.
        flaggems_vllm.cooperative_topk(logits, lengths, output, workspace, k, 1)
        _check(logits, lengths, output, k)
        assert torch.all(scratch_storage[:32] == 251)
        assert torch.all(scratch_storage[-32:] == 251)
        assert torch.all(output_storage[:32] == -71)
        assert torch.all(output_storage[-32:] == -71)
        torch.testing.assert_close(lengths, original_lengths, rtol=0, atol=0)
    assert torch.isnan(storage[:, width:]).all()


@gpu
@pytest.mark.parametrize("k", [512] if cfg.QUICK_MODE else KS)
@pytest.mark.parametrize("width", [4097, 16384, 65539])
@pytest.mark.parametrize(
    "distribution",
    [
        "equal",
        "close",
        "infinities",
        "negative_infinity",
        "nan_few",
        "nan_many",
        "negative_nan",
        "zeros_subnormal",
        "negative_subnormal",
    ],
)
def test_exact_selection_edge_values(width, distribution, k):
    # Keep the oracle and special-value preparation independent of NPU floating
    # point arithmetic, which can flush subnormals before selection is tested.
    input_device = "cpu" if device == "npu" else device
    x = torch.arange(width, device=input_device, dtype=torch.float32)
    if distribution == "equal":
        x.fill_(-1.25)
    elif distribution == "close":
        # All scores round to the same FP16 value; selection must remain FP32.
        x = 1 + (x.remainder(1024) * torch.finfo(torch.float32).eps)
    elif distribution == "infinities":
        x.fill_(float("-inf"))
        x[::7] = float("inf")
    elif distribution == "negative_infinity":
        x.fill_(float("-inf"))
    elif distribution == "negative_nan":
        x.neg_()
        x[:17] = -float("nan")
    elif distribution.startswith("nan"):
        x.neg_()
        x[: (17 if distribution == "nan_few" else k + 11)] = float("nan")
        x[-11:] = float("inf")
    else:
        # Construct subnormal bit patterns without arithmetic that could flush
        # them to zero before the operator receives its input.
        x = torch.arange(width, device=input_device, dtype=torch.int32).remainder(7)
        if distribution == "negative_subnormal":
            x = x.bitwise_or(-2147483648)
        x = x.view(torch.float32)
        x[::5] = -0.0
        x[::11] = 0.0
    logits = x.to(device).unsqueeze(0)
    lengths, output, workspace = _buffers(logits, [width], k)
    flaggems_vllm.cooperative_topk(logits, lengths, output, workspace, k, width)
    _check(logits, lengths, output, k, reference_on_cpu=device == "npu")


@gpu
@pytest.mark.parametrize("k", [512] if cfg.QUICK_MODE else [512, 2048])
@pytest.mark.parametrize("width", [4097, 16385, 65539])
def test_arbitrary_fp32_bit_patterns(width, k):
    # Cover the whole exponent range and both NaN signs without device
    # arithmetic changing subnormal values before the selection is tested.
    generator = torch.Generator().manual_seed(31)
    bits = torch.randint(
        -(2**31),
        2**31 - 1,
        (2, width + 7),
        dtype=torch.int32,
        generator=generator,
    )
    special = torch.tensor(
        [
            0x00000000,
            0x80000000,
            0x00000001,
            0x80000001,
            0x007FFFFF,
            0x807FFFFF,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
            0xFFC00001,
        ],
        dtype=torch.int64,
    ).to(torch.int32)
    bits[:, : special.numel()] = special
    bits[:, width:] = 0x7FC00000
    logits = bits.view(torch.float32).to(device)[:, :width]
    lengths, output, workspace = _buffers(logits, [width, width - 17], k)
    flaggems_vllm.cooperative_topk(logits, lengths, output, workspace, k, width)
    _check(logits, lengths, output, k, reference_on_cpu=True)


@gpu
@pytest.mark.parametrize(
    "width,candidate_count",
    [(4097, 63), (4097, 64), (4097, 65), (65539, 255), (65539, 256), (65539, 257)],
)
def test_threshold_bucket_capacity_boundaries(width, candidate_count):
    # Distinct nearby FP32 values share a coarse bucket. The best candidates
    # must survive both sides of the compact-buffer capacity boundaries.
    rows, k, above_count = 4, 512, 480
    stride = (width + 3) // 4 * 4
    storage = torch.full((rows, stride), float("nan"))
    storage[:, :width] = -2.0
    generator = torch.Generator().manual_seed(53)
    candidates = 1 + torch.arange(candidate_count) * torch.finfo(torch.float32).eps
    for row in range(rows):
        positions = torch.randperm(width, generator=generator)
        storage[row, positions[:above_count]] = 2.0
        storage[row, positions[above_count : above_count + candidate_count]] = (
            candidates
        )
    logits = storage.to(device)[:, :width]
    lengths, output, workspace = _buffers(logits, [width] * rows, k)
    flaggems_vllm.cooperative_topk(logits, lengths, output, workspace, k, width)
    _check(logits, lengths, output, k, reference_on_cpu=True)


@gpu
@pytest.mark.parametrize("shape", [(0, 4096), (4, 0), (1, 1)])
def test_empty_and_single_element(shape):
    logits = torch.empty(shape, device=device)
    lengths, output, workspace = _buffers(logits, [shape[1]] * shape[0], 512)
    assert (
        flaggems_vllm.cooperative_topk(
            logits, lengths, output, workspace, 512, shape[1]
        )
        is None
    )
    _check(logits, lengths, output, 512)


@gpu
def test_maximum_packed_histogram_count():
    # Every entry lands in one bin. The uint16 count must still be exact at
    # the largest width dispatched to the 64-partition path.
    width, k = 1048576, 512
    logits = torch.full((1, width), 1.0, device=device)
    lengths, output, workspace = _buffers(logits, [width], k)
    flaggems_vllm.cooperative_topk(logits, lengths, output, workspace, k, width)
    _check(logits, lengths, output, k)


@gpu
@pytest.mark.parametrize(
    "broken",
    [
        "dtype",
        "k",
        "rows",
        "logits_rank",
        "logits_layout",
        "column_stride",
        "row_stride",
        "lengths_dtype",
        "lengths_shape",
        "lengths_rank",
        "lengths_stride",
        "output_dtype",
        "output_shape",
        "output_stride",
        "workspace_dtype",
        "workspace_size",
        "workspace_stride",
        "workspace_alignment",
        "float_k",
        "alias",
    ],
)
def test_invalid_inputs_rejected_before_output_mutation(broken):
    k = 512
    rows = 65 if broken == "rows" else 2
    logits = torch.randn(rows, 4096, device=device)
    lengths, output, workspace = _buffers(logits, [4096] * rows, k)
    if broken == "dtype":
        logits = logits.half()
    elif broken == "k":
        k = 256
        output = output[:, :k].contiguous()
    elif broken == "logits_rank":
        logits = logits.unsqueeze(0)
    elif broken == "logits_layout":
        if device == "npu":
            # Sparse NPU allocation is not required to test this metadata guard.
            logits = SimpleNamespace(
                device=logits.device, dtype=logits.dtype, layout=torch.sparse_coo
            )
        else:
            logits = logits.to_sparse()
    elif broken == "column_stride":
        logits = logits[:, ::2]
    elif broken == "row_stride":
        logits = logits[:1].expand(rows, -1)
    elif broken == "lengths_dtype":
        lengths = lengths.long()
    elif broken == "lengths_shape":
        lengths = lengths[:1]
    elif broken == "lengths_rank":
        lengths = lengths.view(1, 1, rows)
    elif broken == "lengths_stride":
        lengths = torch.full((rows * 2,), 4096, device=device, dtype=torch.int32)[::2]
    elif broken == "output_dtype":
        output = output.long()
    elif broken == "output_shape":
        output = output[:1]
    elif broken == "output_stride":
        output = output[:1].expand(rows, k)
    elif broken == "workspace_dtype":
        workspace = workspace.int()
    elif broken == "workspace_size":
        workspace = workspace[:-1]
    elif broken == "workspace_stride":
        workspace = torch.empty(
            workspace.numel() * 2, dtype=torch.uint8, device=device
        )[::2]
    elif broken == "workspace_alignment":
        workspace = torch.empty(
            workspace.numel() + 1, dtype=torch.uint8, device=device
        )[1:]
    elif broken == "float_k":
        k = float(k)
    elif broken == "alias":
        output = logits.view(torch.int32).reshape(-1)[: rows * k].view(rows, k)
    before = output.clone()
    with pytest.raises((ValueError, NotImplementedError)):
        flaggems_vllm.cooperative_topk(logits, lengths, output, workspace, k, 4096)
    torch.testing.assert_close(output, before, rtol=0, atol=0)


@gpu
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.complex64,
    ],
)
def test_unsupported_logits_dtype(dtype):
    metadata_only = (dtype == torch.float64 and not utils.fp64_is_supported) or (
        device == "npu" and dtype == torch.complex64
    )
    logits = torch.empty(
        (2, 4096), dtype=torch.float32 if metadata_only else dtype, device=device
    )
    lengths, output, workspace = _buffers(logits, [4096, 4096], 512)
    if metadata_only:
        # Exercise the dtype rejection without requiring unsupported storage.
        logits = SimpleNamespace(device=logits.device, dtype=dtype)
    with pytest.raises(NotImplementedError, match="float32"):
        flaggems_vllm.cooperative_topk(logits, lengths, output, workspace, 512, 4096)
    assert torch.all(output == -17)


@gpu
@pytest.mark.parametrize(
    "alias",
    ["output_lengths", "output_workspace", "workspace_logits", "workspace_lengths"],
)
def test_mutable_buffers_cannot_alias_other_arguments(alias):
    logits = torch.randn((2, 4096), device=device)
    lengths, output, workspace = _buffers(logits, [4096, 4096], 512)
    if alias == "output_lengths":
        lengths = output.reshape(-1)[:2]
    elif alias == "output_workspace":
        output = workspace.view(torch.int32)[:1024].view(2, 512)
    elif alias == "workspace_logits":
        workspace = logits.view(torch.uint8).reshape(-1)
    else:
        lengths = workspace.view(torch.int32)[:2]
    before = output.clone()
    with pytest.raises(ValueError, match="overlap"):
        flaggems_vllm.cooperative_topk(logits, lengths, output, workspace, 512, 4096)
    torch.testing.assert_close(output, before, rtol=0, atol=0)


@gpu
@pytest.mark.parametrize("argument", ["lengths", "output", "workspace"])
def test_mixed_devices_rejected(argument):
    logits = torch.randn((2, 4096), device=device)
    lengths, output, workspace = _buffers(logits, [4096, 4096], 512)
    tensors = {"lengths": lengths, "output": output, "workspace": workspace}
    tensors[argument] = tensors[argument].cpu()
    before = tensors["output"].clone()
    with pytest.raises(ValueError, match="same device"):
        flaggems_vllm.cooperative_topk(logits, **tensors, k=512, max_seq_len=4096)
    torch.testing.assert_close(tensors["output"], before, rtol=0, atol=0)


def test_cpu_is_unsupported():
    logits = torch.ones(1, 4096)
    lengths, output, workspace = _buffers(logits, [4096], 512)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.cooperative_topk(logits, lengths, output, workspace, 512, 4096)
    assert torch.all(output == -17)


@gpu
@pytest.mark.parametrize("width", [4097, 16384, 65539])
@pytest.mark.parametrize("k", [512] if cfg.QUICK_MODE else [512, 2048])
def test_device_graph_replay_with_changed_values_and_lengths(width, k):
    graph_type = getattr(
        torch_device_fn,
        {"cuda": "CUDAGraph", "musa": "MUSAGraph", "npu": "NPUGraph"}[device],
        None,
    )
    if graph_type is None or not hasattr(torch_device_fn, "graph"):
        pytest.skip(f"{device} graph capture API is unavailable")
    logits = torch.randn(4, width, device=device)
    lengths, output, workspace = _buffers(logits, [width] * 4, k)
    flaggems_vllm.cooperative_topk(logits, lengths, output, workspace, k, width)
    torch_device_fn.synchronize()
    graph = graph_type()
    with torch_device_fn.graph(graph):
        flaggems_vllm.cooperative_topk(logits, lengths, output, workspace, k, width)
    for host_lengths, close_values in (
        ([0, k - 1, k + 1, width], False),
        ([width, 0, k, width // 2], True),
        ([width, width, width, width], False),
    ):
        if close_values:
            # Force many candidates into one coarse bucket after capture,
            # then restore a dispersed distribution on the next replay.
            values = torch.arange(width, device=device, dtype=torch.float32)
            values = 1 + values.remainder(1024) * torch.finfo(torch.float32).eps
            logits.copy_(values.expand_as(logits))
        else:
            logits.normal_()
        lengths.copy_(torch.tensor(host_lengths, dtype=torch.int32, device=device))
        output.fill_(-17)
        graph.replay()
        _check(logits, lengths, output, k)


@gpu
@pytest.mark.skipif(
    torch_device_fn.device_count() < 2, reason="Two backend devices required"
)
def test_input_device_differs_from_current_device():
    with torch_device_fn.device(0):
        logits = torch.randn((4, 65539), device=f"{device}:1")
        lengths, output, workspace = _buffers(logits, [0, 512, 513, 65539], 512)
        flaggems_vllm.cooperative_topk(logits, lengths, output, workspace, 512, 65539)
        _check(logits, lengths, output, 512)
        assert torch_device_fn.current_device() == 0


@gpu
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("width", [16384, 65536])
def test_vllm_cross_agreement_when_available(k, width):
    if flaggems_vllm.vendor_name != "nvidia":
        pytest.skip("vLLM cooperative_topk requires NVIDIA")
    major, _ = torch.cuda.get_device_capability()
    if major < 9 or major == 12:
        pytest.skip("vLLM cooperative_topk requires SM90+ excluding SM12x")
    pytest.importorskip("vllm._custom_ops")
    if not hasattr(torch.ops._C, "cooperative_topk"):
        pytest.skip("vLLM build does not expose cooperative_topk")
    logits = torch.randn(4, width, device=device)
    lengths, output, workspace = _buffers(logits, [1, k + 1, width // 2, width], k)
    reference = torch.empty_like(output)
    torch.ops._C.cooperative_topk(logits, lengths, reference, workspace, k, width)
    _check(logits, lengths, reference, k)
    flaggems_vllm.cooperative_topk(logits, lengths, output, workspace, k, width)
    _check(logits, lengths, output, k)


def test_export_and_registration():
    assert "cooperative_topk" in flaggems_vllm.ops.__all__
    assert callable(flaggems_vllm.ops.cooperative_topk)
    assert any(
        name == "cooperative_topk" and fn is flaggems_vllm.cooperative_topk
        for name, fn in flaggems_vllm._FULL_CONFIG
    )
    if flaggems_vllm.vendor_name == "mthreads":
        from flaggems_vllm.runtime.backend._mthreads.ops.cooperative_topk import (
            cooperative_topk as mthreads_cooperative_topk,
        )

        assert flaggems_vllm.cooperative_topk is mthreads_cooperative_topk
