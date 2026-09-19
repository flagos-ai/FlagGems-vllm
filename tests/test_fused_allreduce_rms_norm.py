# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import inspect
import os

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.ops.fused_allreduce_rms_norm import (
    create_fused_allreduce_rms_norm_workspace,
)

try:
    from triton.language.extra.cuda.gdc import gdc_launch_dependents, gdc_wait

    HAS_PDL = True
except ImportError:
    HAS_PDL = False

HIDDEN_SIZE = 6144
RMS_EPS = 1.0e-6
SHAPES = (6, 138)
WORKSPACE_CASES = (
    pytest.param(("peer", torch.bfloat16), id="peer-bf16"),
    pytest.param(("peer", torch.float16), id="peer-fp16"),
    pytest.param(("auto", torch.bfloat16), id="auto-bf16"),
    pytest.param(("mnnvl", torch.bfloat16), id="mnnvl-bf16"),
    pytest.param(("mnnvl", torch.float16), id="mnnvl-fp16"),
)
QUANTIZATION_CASES = (
    pytest.param(False, id="no-quant"),
    pytest.param(True, id="fp8-quant"),
)
NORM_OUTPUT_CASES = (
    pytest.param(False, id="implicit-norm-out"),
    pytest.param(True, id="explicit-norm-out"),
)
PDL_CASES = (
    pytest.param(False, id="pdl-off"),
    pytest.param(
        True,
        id="pdl-on",
        marks=pytest.mark.skipif(not HAS_PDL, reason="Triton CUDA PDL is unavailable"),
    ),
)
NONDEFAULT_ARITHMETIC_CASES = (
    pytest.param(False, 0.0, id="input-precision-acc"),
    pytest.param(True, 1.0, id="weight-bias"),
    pytest.param(False, 1.0, id="input-precision-acc-weight-bias"),
)


if HAS_PDL:

    @triton.jit
    def _pdl_upstream(
        source_input,
        source_residual,
        source_gamma,
        allreduce_input,
        residual_input,
        gamma,
        H: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        columns = tl.arange(0, BLOCK_SIZE)
        mask = columns < H
        offsets = tl.program_id(0) * H + columns
        tl.store(
            allreduce_input + offsets,
            tl.load(source_input + offsets, mask=mask),
            mask=mask,
        )
        tl.store(
            residual_input + offsets,
            tl.load(source_residual + offsets, mask=mask),
            mask=mask,
        )
        if tl.program_id(0) == 0:
            tl.store(
                gamma + columns,
                tl.load(source_gamma + columns, mask=mask),
                mask=mask,
            )
        gdc_launch_dependents()

    @triton.jit
    def _pdl_downstream(
        norm_input,
        independent_input,
        output,
        H: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        columns = tl.arange(0, BLOCK_SIZE)
        mask = columns < H
        offsets = tl.program_id(0) * H + columns
        independent = tl.load(independent_input + offsets, mask=mask, other=0.0)
        gdc_wait()
        norm = tl.load(norm_input + offsets, mask=mask, other=0.0)
        tl.store(output + offsets, norm + independent, mask=mask)


def _has_distributed_launch() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) in (2, 4, 8)


@pytest.fixture(scope="module")
def distributed_context():
    if not _has_distributed_launch():
        pytest.skip("launch with torchrun and 2, 4, or 8 ranks")
    if not torch.cuda.is_available():
        pytest.fail("distributed launch requires CUDA")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.device_count() < world_size:
        pytest.fail(f"distributed launch requires {world_size} visible GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if torch.cuda.get_device_capability(local_rank)[0] < 9:
        pytest.fail("distributed launch requires NVIDIA compute capability >= 9.0")

    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", init_method="env://", device_id=device)
    rank = dist.get_rank()
    yield rank, world_size, device
    dist.barrier()
    if owns_group:
        dist.destroy_process_group()


@pytest.fixture(scope="module", params=WORKSPACE_CASES)
def workspace(request, distributed_context):
    rank, world_size, _device = distributed_context
    backend, dtype = request.param
    try:
        result = create_fused_allreduce_rms_norm_workspace(
            world_size=world_size,
            rank=rank,
            max_token_num=max(SHAPES),
            hidden_dim=HIDDEN_SIZE,
            dtype=dtype,
            group=dist.group.WORLD,
            backend=backend,
        )
    except NotImplementedError as error:
        if backend != "mnnvl":
            raise
        pytest.skip(f"MNNVL is unavailable on this topology: {error}")
    yield result
    result.destroy()


def _allocate(m: int, device: torch.device, dtype: torch.dtype):
    shape = (m, HIDDEN_SIZE)
    return {
        "input": torch.empty(shape, dtype=dtype, device=device),
        "residual_in": torch.empty(shape, dtype=dtype, device=device),
        "gamma": torch.empty((HIDDEN_SIZE,), dtype=dtype, device=device),
        "residual_out": torch.empty(shape, dtype=dtype, device=device),
        "norm_out": torch.empty(shape, dtype=dtype, device=device),
    }


def _set_varying_inputs(tensors, rank: int, step: int) -> None:
    m = tensors["input"].shape[0]
    device = tensors["input"].device
    dtype = tensors["input"].dtype
    rows = torch.arange(m, device=device)[:, None]
    columns = torch.arange(HIDDEN_SIZE, device=device)[None, :]
    payload = (
        (rows * 17 + columns * 3 + (rank + 1) * 19 + (step + 1) * 23) % 127 - 63
    ) / 256
    residual = ((rows * 11 + columns * 5 + (step + 1) * 23) % 113 - 56) / 512
    gamma = (
        1.0
        + ((torch.arange(HIDDEN_SIZE, device=device) * 7 + (step + 1) * 23) % 31 - 15)
        / 512
    )
    tensors["input"].copy_(payload.to(dtype))
    tensors["residual_in"].copy_(residual.to(dtype))
    tensors["gamma"].copy_(gamma.to(dtype))
    tensors["input"].view(torch.int16)[0, step % HIDDEN_SIZE] = -32768


def _reference(tensors, *, fp32_acc: bool = True, weight_bias: float = 0.0):
    value = tensors["input"].clone()
    value.view(torch.int16)[value.view(torch.int16) == -32768] = 0
    if fp32_acc:
        rank_sum = value.float()
        dist.all_reduce(rank_sum, op=dist.ReduceOp.SUM)
        reduced = rank_sum.to(value.dtype)
    else:
        rank_values = [torch.empty_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(rank_values, value)
        reduced = torch.zeros_like(value)
        for rank_value in rank_values:
            reduced = (reduced.float() + rank_value.float()).to(value.dtype)
    residual = (reduced.float() + tensors["residual_in"].float()).to(value.dtype)
    residual_f32 = residual.float()
    reciprocal_rms = torch.rsqrt(
        residual_f32.square().mean(dim=-1, keepdim=True) + RMS_EPS
    )
    norm = (
        residual_f32 * reciprocal_rms * (tensors["gamma"].float() + weight_bias)
    ).to(value.dtype)
    return residual, norm


def _call(
    tensors,
    workspace,
    *,
    launch_with_pdl: bool = True,
    fp32_acc: bool = True,
    pattern_code: int = 1,
    norm_out=None,
    quant_out=None,
    scale_out=None,
    scale_factor=None,
    weight_bias: float = 0.0,
):
    return flaggems_vllm.fused_allreduce_rms_norm(
        tensors["input"],
        tensors["residual_in"],
        tensors["gamma"],
        RMS_EPS,
        workspace.world_size,
        launch_with_pdl,
        fp32_acc,
        workspace.max_token_num,
        pattern_code,
        norm_out=norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
        scale_factor=scale_factor,
        weight_bias=weight_bias,
    )


def test_vllm_compatible_signature():
    parameters = inspect.signature(flaggems_vllm.fused_allreduce_rms_norm).parameters
    assert tuple(parameters) == (
        "allreduce_in",
        "residual",
        "rms_gamma",
        "rms_eps",
        "world_size",
        "launch_with_pdl",
        "fp32_acc",
        "max_token_num",
        "pattern_code",
        "norm_out",
        "quant_out",
        "scale_out",
        "scale_factor",
        "weight_bias",
    )
    defaults = tuple(parameter.default for parameter in parameters.values())
    assert defaults[:9] == (inspect.Parameter.empty,) * 9
    assert defaults[9:] == (None, None, None, None, 0.0)


def test_rejects_unvalidated_tp16():
    shape = (1, HIDDEN_SIZE)
    allreduce_input = torch.empty(shape, dtype=torch.bfloat16)
    residual = torch.empty_like(allreduce_input)
    gamma = torch.empty((HIDDEN_SIZE,), dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="validated Triton all-reduce"):
        flaggems_vllm.fused_allreduce_rms_norm(
            allreduce_input,
            residual,
            gamma,
            RMS_EPS,
            16,
            False,
            True,
            1,
            1,
        )


def _capture(call, device: torch.device):
    torch.cuda.synchronize(device)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize(device)
    dist.barrier()
    return graph


def _assert_fp8_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # The fused and reference normalization paths can round differently when a
    # value lies exactly on an E4M3 bin boundary. Require every output element
    # to be at most one adjacent E4M3 code away from the reference.
    actual_bits = actual.view(torch.uint8)
    expected_bits = expected.view(torch.uint8)
    actual_magnitude = (actual_bits & 0x7F).to(torch.int16)
    expected_magnitude = (expected_bits & 0x7F).to(torch.int16)
    same_sign = (actual_bits & 0x80) == (expected_bits & 0x80)
    both_zero = (actual_magnitude == 0) & (expected_magnitude == 0)
    adjacent = same_sign & ((actual_magnitude - expected_magnitude).abs() <= 1)
    assert bool(torch.all(adjacent | both_zero).item())


@pytest.mark.fused_allreduce_rms_norm
@pytest.mark.parametrize("m", SHAPES, ids=lambda m: f"m{m}")
@pytest.mark.parametrize("quantized", QUANTIZATION_CASES)
@pytest.mark.parametrize("explicit_norm_output", NORM_OUTPUT_CASES)
@pytest.mark.parametrize("launch_with_pdl", PDL_CASES)
def test_functional_cartesian_product(
    distributed_context,
    workspace,
    m,
    quantized,
    explicit_norm_output,
    launch_with_pdl,
):
    """Cover M x quantization x norm placement x PDL for every workspace."""
    rank, _world_size, device = distributed_context
    tensors = _allocate(m, device, workspace.dtype)
    scale_factor = (
        torch.tensor([1.25], dtype=torch.float32, device=device) if quantized else None
    )
    quant_out = (
        torch.empty_like(tensors["input"], dtype=torch.float8_e4m3fn)
        if quantized
        else None
    )
    norm_out = tensors["norm_out"] if explicit_norm_output else None
    pattern_code = 2 if quantized else 1

    _set_varying_inputs(tensors, rank, 600)
    graph = _capture(
        lambda: _call(
            tensors,
            workspace,
            launch_with_pdl=launch_with_pdl,
            pattern_code=pattern_code,
            norm_out=norm_out,
            quant_out=quant_out,
            scale_factor=scale_factor,
        ),
        device,
    )

    # Replay with different values so stale graph inputs cannot pass.
    replay_step = (
        700
        + m
        + 100 * int(quantized)
        + 10 * int(explicit_norm_output)
        + int(launch_with_pdl)
    )
    _set_varying_inputs(tensors, rank, replay_step)
    original_input = tensors["input"].clone()
    original_residual = tensors["residual_in"].clone()
    if norm_out is not None:
        norm_out.fill_(13)
        original_norm_out = norm_out.clone()
    expected_residual, expected_norm = _reference(tensors)
    expected_quant = None
    if quantized:
        expected_quant = torch.clamp(
            expected_norm.float() / scale_factor, -448.0, 448.0
        ).to(torch.float8_e4m3fn)

    try:
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize(device)

        if quantized:
            _assert_fp8_close(quant_out, expected_quant)
            if explicit_norm_output:
                torch.testing.assert_close(
                    tensors["input"], expected_residual, atol=0.04, rtol=0.04
                )
                torch.testing.assert_close(tensors["residual_in"], original_residual)
                torch.testing.assert_close(norm_out, original_norm_out)
            else:
                torch.testing.assert_close(tensors["input"], original_input)
                torch.testing.assert_close(
                    tensors["residual_in"], expected_residual, atol=0.04, rtol=0.04
                )
        elif explicit_norm_output:
            torch.testing.assert_close(
                tensors["input"], expected_residual, atol=0.04, rtol=0.04
            )
            torch.testing.assert_close(tensors["residual_in"], original_residual)
            torch.testing.assert_close(norm_out, expected_norm, atol=0.04, rtol=0.04)
        else:
            torch.testing.assert_close(
                tensors["residual_in"], expected_residual, atol=0.04, rtol=0.04
            )
            torch.testing.assert_close(
                tensors["input"], expected_norm, atol=0.04, rtol=0.04
            )
    finally:
        graph.reset()


@pytest.mark.fused_allreduce_rms_norm
def test_dynamic_shapes_and_varying_cuda_graph_replays(
    distributed_context,
    workspace,
):
    rank, _world_size, device = distributed_context
    tensors = {m: _allocate(m, device, workspace.dtype) for m in SHAPES}
    for capture_step, m in enumerate(SHAPES, start=1000):
        _set_varying_inputs(tensors[m], rank, capture_step)
    graphs = {
        m: _capture(lambda m=m: _call(tensors[m], workspace), device) for m in SHAPES
    }

    try:
        sequence = (6, 138, 6, 6, 138, 138)
        for step, m in enumerate(sequence):
            current = tensors[m]
            _set_varying_inputs(current, rank, step)
            expected_residual, expected_norm = _reference(current)
            dist.barrier()
            graphs[m].replay()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(
                current["residual_in"], expected_residual, atol=0.04, rtol=0.04
            )
            torch.testing.assert_close(
                current["input"], expected_norm, atol=0.04, rtol=0.04
            )

        state = [int(value) for value in workspace.state.cpu().tolist()]
        assert state[0] == 0
        assert state[1] in (0, 1, 2)
        assert state[2] == sequence[-1] * HIDDEN_SIZE
    finally:
        for graph in graphs.values():
            graph.reset()


@pytest.mark.fused_allreduce_rms_norm
@pytest.mark.skipif(not HAS_PDL, reason="Triton CUDA PDL is unavailable")
def test_pdl_upstream_and_downstream_chain(distributed_context, workspace):
    rank, _world_size, device = distributed_context
    m = 6
    tensors = _allocate(m, device, workspace.dtype)
    sources = _allocate(m, device, workspace.dtype)
    _set_varying_inputs(sources, rank, 211)
    independent = torch.full_like(tensors["norm_out"], 0.125)
    downstream_output = torch.empty_like(tensors["norm_out"])
    block_size = triton.next_power_of_2(HIDDEN_SIZE)

    torch.cuda.synchronize(device)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _pdl_upstream[(m,)](
            sources["input"],
            sources["residual_in"],
            sources["gamma"],
            tensors["input"],
            tensors["residual_in"],
            tensors["gamma"],
            H=HIDDEN_SIZE,
            BLOCK_SIZE=block_size,
            num_warps=8,
            launch_pdl=True,
        )
        _call(tensors, workspace)
        _pdl_downstream[(m,)](
            tensors["input"],
            independent,
            downstream_output,
            H=HIDDEN_SIZE,
            BLOCK_SIZE=block_size,
            num_warps=8,
            launch_pdl=True,
        )
    torch.cuda.synchronize(device)
    dist.barrier()

    # Change every graph input after capture. Reusing the capture-time values
    # could hide an early PDL signal that lets the downstream kernel consume
    # the preceding replay's norm output.
    _set_varying_inputs(sources, rank, 212)
    independent.fill_(0.25)
    expected_residual, expected_norm = _reference(sources)
    dist.barrier()
    try:
        graph.replay()
        torch.cuda.synchronize(device)

        torch.testing.assert_close(
            tensors["residual_in"], expected_residual, atol=0.04, rtol=0.04
        )
        torch.testing.assert_close(
            downstream_output,
            expected_norm + independent,
            atol=0.04,
            rtol=0.04,
        )
    finally:
        graph.reset()


@pytest.mark.fused_allreduce_rms_norm
@pytest.mark.parametrize("m", SHAPES, ids=lambda m: f"m{m}")
@pytest.mark.parametrize(("fp32_acc", "weight_bias"), NONDEFAULT_ARITHMETIC_CASES)
def test_nondefault_arithmetic_and_leading_shape(
    distributed_context,
    workspace,
    m,
    fp32_acc,
    weight_bias,
):
    rank, _world_size, device = distributed_context
    tensors = _allocate(m, device, workspace.dtype)
    _set_varying_inputs(tensors, rank, 307 + m)
    expected_residual, expected_norm = _reference(
        tensors,
        fp32_acc=fp32_acc,
        weight_bias=weight_bias,
    )
    leading_shape = (2, m // 2, HIDDEN_SIZE)
    dist.barrier()
    flaggems_vllm.fused_allreduce_rms_norm(
        tensors["input"].view(leading_shape),
        tensors["residual_in"].view(leading_shape),
        tensors["gamma"],
        RMS_EPS,
        workspace.world_size,
        False,
        fp32_acc,
        workspace.max_token_num,
        1,
        weight_bias=weight_bias,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        tensors["residual_in"], expected_residual, atol=0.04, rtol=0.04
    )
    torch.testing.assert_close(tensors["input"], expected_norm, atol=0.04, rtol=0.04)


@pytest.mark.fused_allreduce_rms_norm
def test_nvfp4_capability_gate(distributed_context):
    rank, world_size, device = distributed_context
    if torch.cuda.get_device_capability(device)[0] >= 10:
        pytest.skip("NVFP4 correctness requires the dedicated SM100 test job")
    tensors = _allocate(6, device, torch.bfloat16)
    _set_varying_inputs(tensors, rank, 503)
    scale_factor = torch.ones((1,), dtype=torch.float32, device=device)
    with pytest.raises(NotImplementedError, match="SM100"):
        flaggems_vllm.fused_allreduce_rms_norm(
            tensors["input"],
            tensors["residual_in"],
            tensors["gamma"],
            RMS_EPS,
            world_size,
            False,
            True,
            max(SHAPES),
            3,
            scale_factor=scale_factor,
        )


@pytest.mark.fused_allreduce_rms_norm
def test_argument_validation(distributed_context):
    _rank, world_size, device = distributed_context
    tensors = _allocate(6, device, torch.bfloat16)
    quant_out = torch.empty_like(tensors["input"], dtype=torch.float8_e4m3fn)

    with pytest.raises(ValueError, match="scale_factor is required"):
        flaggems_vllm.fused_allreduce_rms_norm(
            tensors["input"],
            tensors["residual_in"],
            tensors["gamma"],
            RMS_EPS,
            world_size,
            False,
            True,
            max(SHAPES),
            2,
            quant_out=quant_out,
        )
    with pytest.raises(ValueError, match="must not overlap"):
        flaggems_vllm.fused_allreduce_rms_norm(
            tensors["input"],
            tensors["input"],
            tensors["gamma"],
            RMS_EPS,
            world_size,
            False,
            True,
            max(SHAPES),
            1,
        )
    with pytest.raises(NotImplementedError, match="unsupported pattern_code"):
        flaggems_vllm.fused_allreduce_rms_norm(
            tensors["input"],
            tensors["residual_in"],
            tensors["gamma"],
            RMS_EPS,
            world_size,
            False,
            True,
            max(SHAPES),
            99,
        )
    with pytest.raises(ValueError, match="invalid for non-quant"):
        flaggems_vllm.fused_allreduce_rms_norm(
            tensors["input"],
            tensors["residual_in"],
            tensors["gamma"],
            RMS_EPS,
            world_size,
            False,
            True,
            max(SHAPES),
            1,
            quant_out=quant_out,
        )


@pytest.mark.fused_allreduce_rms_norm
def test_rejects_unsupported_dtype(distributed_context):
    _rank, world_size, device = distributed_context
    tensors = _allocate(6, device, torch.bfloat16)
    tensors["input"] = torch.empty((6, HIDDEN_SIZE), dtype=torch.float32, device=device)
    with pytest.raises(NotImplementedError, match="FP16/BF16"):
        flaggems_vllm.fused_allreduce_rms_norm(
            tensors["input"],
            tensors["residual_in"],
            tensors["gamma"],
            RMS_EPS,
            world_size,
            False,
            True,
            max(SHAPES),
            1,
        )
