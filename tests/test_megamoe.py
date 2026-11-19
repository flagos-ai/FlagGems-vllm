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

"""Shared tests for the public NVIDIA and Hygon MegaMoE entry points."""

from __future__ import annotations

import importlib
import inspect
import os
import socket
import subprocess
import sys
from dataclasses import FrozenInstanceError, dataclass, fields
from pathlib import Path

_IS_ACCURACY_WORKER = "--accuracy-worker" in sys.argv
if _IS_ACCURACY_WORKER:
    _local_rank = os.environ.get("LOCAL_RANK", "0")
    _cache_root = Path(
        os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache")
    )
    os.environ["TRITON_CACHE_DIR"] = str(
        _cache_root.parent / f"{_cache_root.name}-megamoe-test-rank-{_local_rank}"
    )

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

if _IS_ACCURACY_WORKER:
    torch.cuda.set_device(int(_local_rank))

# The worker cache and CUDA device must be selected before these imports.
# isort: off
import flaggems_vllm  # noqa: E402

if flaggems_vllm.vendor_name == "nvidia":
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops import (  # noqa: E402
        megamoe as hopper_megamoe,
    )
else:
    hopper_megamoe = None
from flaggems_vllm.testing.megamoe import StandaloneNvshmem  # noqa: E402
from flaggems_vllm.testing.megamoe import (  # noqa: E402
    configure_descriptor_allocator,
    interleave_l1_gate_up_rows,
    make_context,
    pool_tokens,
)
from flaggems_vllm.testing.megamoe_ppu import (  # noqa: E402
    assert_relative_l2,
    make_ppu_context,
    reference_int8_moe,
)
from flaggems_vllm.utils.triton_version_utils import has_triton_tle  # noqa: E402

# isort: on

pytestmark = pytest.mark.megamoe
HYGON_CANONICAL_DATA = Path(
    os.environ.get(
        "FLAGGEMS_MEGAMOE_DATA_DIR",
        "/workspace/data/qwen36_int8/canonical",
    )
)


def _is_hopper():
    return (
        flaggems_vllm.vendor_name == "nvidia"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
    )


def _is_hygon_gfx936():
    if flaggems_vllm.vendor_name != "hygon" or not torch.cuda.is_available():
        return False
    properties = torch.cuda.get_device_properties(0)
    return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0] == "gfx936"


def _run_torchrun(worker_arguments, *, num_ranks=2, timeout=900):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        master_port = listener.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--master-addr",
        "127.0.0.1",
        "--master-port",
        str(master_port),
        "--nproc-per-node",
        str(num_ranks),
        *worker_arguments,
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode:
        pytest.fail(completed.stdout + completed.stderr)


@pytest.mark.skipif(
    not (_is_hopper() and has_triton_tle(3, 6, 0)),
    reason="requires Hopper with FlagTree TLE",
)
def test_megamoe_fp8_is_exported_on_hopper():
    assert callable(hopper_megamoe.megamoe_fp8)
    assert flaggems_vllm.megamoe_fp8 is hopper_megamoe.megamoe_fp8


@dataclass(frozen=True)
class _AccuracyConfig:
    num_ranks: int = 2
    tokens: int = 2
    hidden: int = 128
    intermediate: int = 128
    num_experts: int = 2
    topk: int = 1
    stages: int = 2
    max_recv: int = 128


def _make_accuracy_inputs(config: _AccuracyConfig, rank: int, device):
    generator = torch.Generator().manual_seed(20261105)
    fp8 = torch.float8_e4m3fn

    inputs = torch.randn(
        config.num_ranks,
        config.tokens,
        config.hidden,
        generator=generator,
    ).mul_(0.25)
    input_scales = (
        torch.rand(
            config.num_ranks,
            config.tokens,
            config.hidden // 128,
            generator=generator,
        )
        .mul_(0.25)
        .add_(0.75)
    )
    w1_logical = torch.randn(
        config.num_experts,
        2 * config.intermediate,
        config.hidden,
        generator=generator,
    ).mul_(0.125)
    w1_scales = (
        torch.rand(
            config.num_experts,
            config.intermediate // 64,
            config.hidden // 128,
            generator=generator,
        )
        .mul_(0.25)
        .add_(0.75)
    )
    w2 = torch.randn(
        config.num_experts,
        config.hidden,
        config.intermediate,
        generator=generator,
    ).mul_(0.125)
    w2_scales = (
        torch.rand(
            config.num_experts,
            config.hidden // 128,
            config.intermediate // 128,
            generator=generator,
        )
        .mul_(0.25)
        .add_(0.75)
    )

    inputs = inputs.to(device=device, dtype=fp8)
    input_scales = input_scales.to(device)
    w1_logical = w1_logical.to(device=device, dtype=fp8)
    w1_scales = w1_scales.to(device)
    w2 = w2.to(device=device, dtype=fp8)
    w2_scales = w2_scales.to(device)
    w1_packed = interleave_l1_gate_up_rows(w1_logical)

    topk_all = torch.empty(
        config.num_ranks,
        config.tokens,
        config.topk,
        dtype=torch.int64,
    )
    for source_rank in range(config.num_ranks):
        topk_all[source_rank].fill_(1 - source_rank)
    topk_weights = torch.tensor([[0.75], [1.25]], dtype=torch.float32, device=device)

    local_slice = slice(rank, rank + 1)
    operands = (
        inputs[rank].contiguous(),
        input_scales[rank].contiguous(),
        topk_all[rank].to(device),
        topk_weights,
        w1_packed[local_slice].contiguous(),
        w1_scales[local_slice].contiguous(),
        w2[local_slice].contiguous(),
        w2_scales[local_slice].contiguous(),
    )
    reference_inputs = (
        inputs[rank],
        input_scales[rank],
        topk_all[rank].to(device),
        topk_weights,
        w1_logical,
        w1_scales,
        w2,
        w2_scales,
    )
    return operands, reference_inputs, topk_all


def _reference_megamoe(
    input_fp8,
    input_scales,
    topk_ids,
    topk_weights,
    w1,
    w1_scales,
    w2,
    w2_scales,
):
    tokens, hidden = input_fp8.shape
    intermediate = w2.shape[-1]
    output = torch.zeros(tokens, hidden, dtype=torch.float32, device=input_fp8.device)

    for token in range(tokens):
        for route in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, route])
            l1 = torch.zeros(
                2 * intermediate, dtype=torch.float32, device=input_fp8.device
            )
            for n_block in range((2 * intermediate) // 128):
                n_start = n_block * 128
                for k_block in range(hidden // 128):
                    k_start = k_block * 128
                    product = (
                        input_fp8[token, k_start : k_start + 128].float()
                        @ w1[
                            expert,
                            n_start : n_start + 128,
                            k_start : k_start + 128,
                        ]
                        .float()
                        .T
                    )
                    l1[n_start : n_start + 128] += product * (
                        input_scales[token, k_block]
                        * w1_scales[expert, n_block, k_block]
                    )

            gate, up = l1.chunk(2)
            swiglu = gate / (1.0 + torch.exp(-gate)) * up
            route_weight = topk_weights[token, route]
            segments = swiglu.reshape(intermediate // 64, 64)
            amax = segments.abs().amax(dim=1) * route_weight.abs()
            scaled = amax / 448.0
            segment_scales = torch.where(
                scaled > 0,
                torch.exp2(torch.ceil(torch.log2(scaled))),
                torch.ones_like(scaled),
            )
            quantized = (segments * (route_weight / segment_scales)[:, None]).to(
                torch.float8_e4m3fn
            )

            partial = torch.zeros(hidden, dtype=torch.float32, device=input_fp8.device)
            for n_block in range(hidden // 128):
                n_start = n_block * 128
                for k_block in range(intermediate // 128):
                    k_start = k_block * 128
                    weight_scale = w2_scales[expert, n_block, k_block]
                    low = (
                        quantized[2 * k_block].float()
                        @ w2[
                            expert,
                            n_start : n_start + 128,
                            k_start : k_start + 64,
                        ]
                        .float()
                        .T
                    )
                    high = (
                        quantized[2 * k_block + 1].float()
                        @ w2[
                            expert,
                            n_start : n_start + 128,
                            k_start + 64 : k_start + 128,
                        ]
                        .float()
                        .T
                    )
                    partial[n_start : n_start + 128] += weight_scale * (
                        low * segment_scales[2 * k_block]
                        + high * segment_scales[2 * k_block + 1]
                    )
            output[token] += partial.to(torch.bfloat16).float()

    return output.to(torch.bfloat16)


def _run_accuracy_worker() -> None:
    config = _AccuracyConfig()
    with StandaloneNvshmem() as runtime:
        assert runtime.world_size == config.num_ranks
        configure_descriptor_allocator(runtime.device)
        operands, reference_inputs, topk_all = _make_accuracy_inputs(
            config, runtime.rank, runtime.device
        )
        pool_size = pool_tokens(topk_all, config)
        context = make_context(config, runtime, pool_size)
        output = hopper_megamoe.megamoe_fp8(
            *operands,
            context,
            stages=config.stages,
        )
        torch.cuda.synchronize(runtime.device)
        reference = _reference_megamoe(*reference_inputs)

        relative_l2 = (
            output.float() - reference.float()
        ).norm() / reference.float().norm().clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(
            output.float().flatten(), reference.float().flatten(), dim=0
        )
        assert float(relative_l2) < 0.02, relative_l2
        assert float(cosine) > 0.999, cosine


@pytest.mark.skipif(
    not (_is_hopper() and has_triton_tle(3, 6, 0)),
    reason="requires Hopper with FlagTree TLE",
)
def test_megamoe_fp8_accuracy():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible Hopper GPUs")

    _run_torchrun(
        [str(Path(__file__).resolve()), "--accuracy-worker"],
    )


@pytest.mark.megamoe_int8
@pytest.mark.skipif(not _is_hygon_gfx936(), reason="requires Hygon gfx936")
def test_megamoe_int8_accuracy():
    """Reuse the shared distributed-test launcher with the W8A8 reference."""

    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible Hygon GPUs")
    if not (HYGON_CANONICAL_DATA / "manifest.json").is_file():
        pytest.skip(f"canonical W8A8 data not found at {HYGON_CANONICAL_DATA}")
    _run_torchrun(
        [
            "--module",
            "benchmark.test_megamoe",
            "--world-size",
            "2",
            "--tokens",
            "2",
            "--data-dir",
            str(HYGON_CANONICAL_DATA),
            "--warmup",
            "1",
            "--iterations",
            "1",
            "--correctness-tokens",
            "2",
            "--ncta",
            "8",
        ],
        timeout=1200,
    )


from flaggems_vllm.runtime import device  # noqa: E402
from flaggems_vllm.runtime.backend._hygon.mega.megamoe.api import (  # noqa: E402
    MegaMoEContext,
    MegaMoETileConfig,
    MegaMoEWeights,
    default_tile_config,
    megamoe_int8,
)
from flaggems_vllm.runtime.backend._hygon.mega.megamoe.kernel import (  # noqa: E402
    _local_epoch_barrier,
)
from flaggems_vllm.testing.megamoe_hygon_baseline import (  # noqa: E402
    COMMUNICATION_BACKEND,
    VLLM_REQUIRED_VERSION,
    HcuNativeEpInt8Baseline,
    NativeEpContract,
    PrequantizedHcuTritonExperts,
)


@triton.jit
def _hygon_local_barrier_fault_probe(
    done,
    released,
    sync_status,
    result,
    EPOCH: tl.constexpr,
    DECLARED_NCTA: tl.constexpr,
    NCTA_POW2: tl.constexpr,
    MAX_SPINS: tl.constexpr,
):
    ready = _local_epoch_barrier(
        done,
        released,
        sync_status,
        EPOCH,
        0,
        DECLARED_NCTA,
        NCTA_POW2,
        MAX_SPINS,
    )
    tl.store(result + tl.program_id(0), tl.where(ready, 1, -1))


@triton.jit
def _hygon_local_barrier_visibility_probe(
    payload,
    observed,
    done,
    released,
    sync_status,
    WIDTH: tl.constexpr,
    NCTA: tl.constexpr,
    NCTA_POW2: tl.constexpr,
    ITERATIONS: tl.constexpr,
    MAX_SPINS: tl.constexpr,
):
    """Exercise multi-warp producer visibility across a reusable grid barrier."""

    pid = tl.program_id(0)
    offsets = tl.arange(0, WIDTH)
    peer = (pid + 1) % NCTA
    for iteration in tl.range(0, ITERATIONS):
        epoch = iteration + 1
        values = epoch * 1_000_000 + pid * WIDTH + offsets
        tl.store(payload + pid * WIDTH + offsets, values)
        _local_epoch_barrier(
            done,
            released,
            sync_status,
            epoch,
            0,
            NCTA,
            NCTA_POW2,
            MAX_SPINS,
        )
        peer_values = tl.load(payload + peer * WIDTH + offsets)
        destination = (iteration * NCTA + pid) * WIDTH + offsets
        tl.store(observed + destination, peer_values)
        _local_epoch_barrier(
            done,
            released,
            sync_status,
            epoch,
            1,
            NCTA,
            NCTA_POW2,
            MAX_SPINS,
        )


@pytest.mark.megamoe
@pytest.mark.megamoe_int8
def test_megamoe_int8_python_contract():
    assert list(inspect.signature(megamoe_int8).parameters) == [
        "input_int8",
        "input_scales",
        "topk_ids",
        "topk_weights",
        "weights",
        "context",
        "output",
    ]
    assert [field.name for field in fields(MegaMoEWeights)] == [
        "w1_gate_t",
        "w1_up_t",
        "w2_t",
        "w1_gate_scale",
        "w1_up_scale",
        "w2_scale",
    ]
    context_fields = {field.name for field in fields(MegaMoEContext)}
    assert {
        "rank",
        "world_size",
        "tokens_per_rank",
        "tile",
        "q_local",
        "q_table",
        "route_queue",
        "sync_status",
        "output",
        "epoch",
        "lock",
        "_poisoned",
    } <= context_fields
    assert callable(MegaMoEContext.raise_if_sync_failed)


@pytest.mark.megamoe_int8
def test_hygon_barrier_uses_cooperative_fail_closed_launch():
    from flaggems_vllm.runtime.backend._hygon.mega.megamoe import kernel

    launch_source = inspect.getsource(megamoe_int8)
    publish_source = kernel._publish_epoch.src
    local_barrier_source = kernel._local_epoch_barrier.src
    rank_barrier_source = kernel._rank_epoch_barrier.src
    kernel_source = kernel.hcu_megamoe.src
    assert "launch_cooperative_grid=True" in launch_source
    assert "tl.atomic_cas" in publish_source
    assert "tl.abs(observed) < epoch" in publish_source
    assert "spins < MAX_SPINS" in local_barrier_source
    assert "_publish_epoch" in local_barrier_source
    assert "-epoch" in local_barrier_source
    assert "spins < MAX_SPINS" in rank_barrier_source
    assert "_publish_epoch" in rank_barrier_source
    assert "-epoch" in rank_barrier_source
    assert local_barrier_source.count("tl.debug_barrier()") == 2
    assert rank_barrier_source.count("tl.debug_barrier()") == 2
    local_publish = local_barrier_source.index("tl.atomic_xchg(")
    local_success = local_barrier_source.rindex("success = decision == epoch")
    local_release = local_barrier_source.rindex("return success")
    assert local_barrier_source.index("tl.debug_barrier()") < local_publish
    assert (
        local_success
        < local_barrier_source.rindex("tl.debug_barrier()")
        < local_release
    )
    rank_single_cta = rank_barrier_source.index("if NCTA == 1:")
    rank_single_cta_ready = rank_barrier_source.index("local_ready = True")
    rank_success = rank_barrier_source.rindex("success = decision == epoch")
    rank_release = rank_barrier_source.rindex("return success")
    assert (
        rank_single_cta
        < rank_barrier_source.index("tl.debug_barrier()")
        < rank_single_cta_ready
    )
    assert (
        rank_success < rank_barrier_source.rindex("tl.debug_barrier()") < rank_release
    )
    assert kernel_source.count("if not local_ready:") == 7
    assert kernel_source.count("if not rank_ready:") == 2


@pytest.mark.megamoe_int8
@pytest.mark.skipif(not _is_hygon_gfx936(), reason="requires Hygon gfx936")
def test_hygon_local_barrier_missing_cta_fails_closed():
    """A missing CTA must publish abort instead of fake success or hanging."""

    launched_ncta = 8
    declared_ncta = launched_ncta + 1
    device = torch.device("cuda:0")
    done = torch.zeros(declared_ncta, dtype=torch.int32, device=device)
    released = torch.zeros(1, dtype=torch.int32, device=device)
    sync_status = torch.zeros(1, dtype=torch.int32, device=device)
    result = torch.zeros(launched_ncta, dtype=torch.int32, device=device)

    _hygon_local_barrier_fault_probe[(launched_ncta,)](
        done,
        released,
        sync_status,
        result,
        EPOCH=1,
        DECLARED_NCTA=declared_ncta,
        NCTA_POW2=16,
        MAX_SPINS=128,
        num_warps=4,
        launch_cooperative_grid=True,
    )
    torch.cuda.synchronize(device)

    assert done.cpu().tolist() == [1] * launched_ncta + [0]
    assert int(released.item()) == -1
    assert int(sync_status.item()) == 1
    assert result.cpu().tolist() == [-1] * launched_ncta


@pytest.mark.megamoe_int8
@pytest.mark.skipif(not _is_hygon_gfx936(), reason="requires Hygon gfx936")
def test_hygon_local_barrier_preserves_multi_warp_visibility():
    """Every producer warp must finish before a peer CTA consumes its row."""

    ncta = 8
    width = 4096
    iterations = 32
    device = torch.device("cuda:0")
    payload = torch.full((ncta, width), -1, dtype=torch.int32, device=device)
    observed = torch.full(
        (iterations, ncta, width), -1, dtype=torch.int32, device=device
    )
    done = torch.zeros((2, ncta), dtype=torch.int32, device=device)
    released = torch.zeros(2, dtype=torch.int32, device=device)
    sync_status = torch.zeros(1, dtype=torch.int32, device=device)

    _hygon_local_barrier_visibility_probe[(ncta,)](
        payload,
        observed,
        done,
        released,
        sync_status,
        WIDTH=width,
        NCTA=ncta,
        NCTA_POW2=ncta,
        ITERATIONS=iterations,
        MAX_SPINS=1_000_000,
        num_warps=8,
        launch_cooperative_grid=True,
    )
    torch.cuda.synchronize(device)

    epochs = torch.arange(1, iterations + 1, dtype=torch.int32)[:, None, None]
    peers = (torch.arange(ncta, dtype=torch.int32) + 1) % ncta
    columns = torch.arange(width, dtype=torch.int32)[None, None, :]
    expected = epochs * 1_000_000 + peers[None, :, None] * width + columns
    torch.testing.assert_close(observed.cpu(), expected)
    assert int(sync_status.item()) == 0
    assert done.cpu().tolist() == [[iterations] * ncta] * 2
    assert released.cpu().tolist() == [iterations, iterations]


@pytest.mark.megamoe
@pytest.mark.megamoe_int8
@pytest.mark.parametrize(
    "tokens,expected",
    [
        (1, MegaMoETileConfig(32, 64, 128, 64, 32, 4)),
        (4, MegaMoETileConfig(32, 64, 128, 64, 32, 4)),
        (5, MegaMoETileConfig(32, 128, 128, 64, 32, 4)),
        (64, MegaMoETileConfig(32, 128, 128, 64, 32, 4)),
        (65, MegaMoETileConfig(32, 128, 128, 64, 32, 8)),
        (512, MegaMoETileConfig(32, 128, 128, 64, 32, 8)),
        (513, MegaMoETileConfig(128, 128, 128, 64, 64, 8)),
        (16384, MegaMoETileConfig(128, 128, 128, 64, 64, 8)),
    ],
)
def test_default_tile_config_boundaries(tokens, expected):
    assert default_tile_config(tokens) == expected


@pytest.mark.megamoe
@pytest.mark.megamoe_int8
@pytest.mark.parametrize(
    "tokens,error",
    [
        (False, TypeError),
        (1.0, TypeError),
        (0, ValueError),
        (-1, ValueError),
        (16385, ValueError),
    ],
)
def test_default_tile_config_rejects_unsupported_values(tokens, error):
    with pytest.raises(error):
        default_tile_config(tokens)


@pytest.mark.megamoe
@pytest.mark.megamoe_int8
def test_prepacked_weight_descriptor_is_frozen():
    marker = object()
    weights = MegaMoEWeights(marker, marker, marker, marker, marker, marker)
    with pytest.raises(FrozenInstanceError):
        weights.w2_t = object()


@pytest.mark.megamoe
@pytest.mark.megamoe_int8
def test_native_ep_baseline_contract_is_device_independent():
    """The paired benchmark contract must remain inspectable without HCU."""

    contract = NativeEpContract(
        world_size=4,
        rank=2,
        tokens_per_rank=64,
        hidden_size=2048,
        intermediate_size=512,
        num_experts=256,
        top_k=8,
    )
    assert contract.experts_per_rank == 64
    assert contract.local_expert_begin == 128
    assert contract.local_expert_end == 192
    assert VLLM_REQUIRED_VERSION == "0.18.1"
    assert COMMUNICATION_BACKEND == "allgather_reducescatter"
    assert HcuNativeEpInt8Baseline.first_input_quantization_in_timed_region is False
    assert HcuNativeEpInt8Baseline.intermediate_quantization_in_timed_region is True

    source = inspect.getsource(PrequantizedHcuTritonExperts.apply)
    assert "ignore_invalid_experts=True" in source


@pytest.mark.megamoe
@pytest.mark.megamoe_int8
def test_native_ep_baseline_contract_rejects_unsupported_world_size():
    with pytest.raises(ValueError, match="world_size"):
        NativeEpContract(
            world_size=3,
            rank=0,
            tokens_per_rank=8,
            hidden_size=2048,
            intermediate_size=512,
            num_experts=256,
            top_k=8,
        )


@pytest.mark.megamoe
@pytest.mark.megamoe_int8
def test_hygon_public_export_smoke():
    """Hygon dispatch must expose exactly the backend implementation."""

    if device.vendor_name != "hygon":
        pytest.skip("public MegaMoE dispatch is installed only on Hygon")

    import flaggems_vllm
    from flaggems_vllm.runtime.backend._hygon.ops import megamoe as backend

    assert callable(backend.megamoe_int8)
    assert flaggems_vllm.megamoe_int8 is backend.megamoe_int8


def _ppu_api():
    if flaggems_vllm.vendor_name != "thead":
        pytest.skip("requires the PPU backend")
    return importlib.import_module("flaggems_vllm.runtime.backend._thead.ops.megamoe")


def _require_ppu_device():
    api = _ppu_api()
    if not torch.cuda.is_available():
        pytest.skip("requires a PPU device")
    return api


@pytest.mark.megamoe
def test_megamoe_is_exported_on_ppu():
    api = _ppu_api()
    assert callable(api.megamoe)
    assert flaggems_vllm.megamoe is api.megamoe
    package = importlib.import_module(
        "flaggems_vllm.runtime.backend._thead.mega.megamoe"
    )
    assert package.megamoe is api.megamoe
    assert package.pack_gate_up_rows is api.pack_gate_up_rows
    assert package.peer_buffer_sizes is api.peer_buffer_sizes
    assert package.MegaMoEContext is api.MegaMoEContext
    assert package.MegaMoEConfig is api.MegaMoEConfig


@pytest.mark.parametrize("dtype", [torch.int8, torch.float32])
@pytest.mark.parametrize("shape", [(1, 32, 1), (2, 64, 3)])
def test_megamoe_w1_layout(dtype, shape):
    api = _require_ppu_device()
    source = (
        torch.arange(shape[0] * shape[1] * shape[2], dtype=torch.int32)
        .reshape(shape)
        .to(dtype)
    )
    half = shape[1] // 2
    expected = torch.stack(
        (
            source[:, :half].reshape(shape[0], half // 8, 8, shape[2]),
            source[:, half:].reshape(shape[0], half // 8, 8, shape[2]),
        ),
        dim=2,
    ).reshape(shape)
    actual = api.pack_gate_up_rows(source.cuda())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape,error",
    [
        ((4, 8), "rank 3"),
        ((1, 15, 2), "even row count"),
        ((1, 20, 2), "divisible by 8"),
    ],
)
def test_megamoe_w1_layout_rejects_invalid_shape(shape, error):
    api = _ppu_api()
    with pytest.raises(ValueError, match=error):
        api.pack_gate_up_rows(torch.empty(shape, dtype=torch.int8))


@pytest.mark.parametrize(
    "values,error_type",
    [
        ((0, 256, 128, 8, 2, 2), ValueError),
        ((True, 256, 128, 8, 2, 2), ValueError),
        ((17, 192, 128, 8, 2, 2), NotImplementedError),
        ((17, 256, 192, 8, 2, 2), NotImplementedError),
        ((17, 256, 128, 7, 2, 2), ValueError),
        ((17, 256, 128, 8, 9, 2), ValueError),
    ],
)
def test_megamoe_rejects_unsupported_contract(values, error_type):
    api = _ppu_api()
    with pytest.raises(error_type):
        api.peer_buffer_sizes(*values)


def test_megamoe_peer_capacity_covers_routing_skew():
    sizes = _ppu_api().peer_buffer_sizes(17, 256, 384, 10, 3, 2)
    assert sizes["tok_tab"] == 17 * 256
    assert sizes["cb_tab"] == 17 * 3 * 256 * 2
    assert sizes["meta_tab"] >= 2 * 17 * 3 * 12
    assert set(sizes) == {
        "tok_tab",
        "sf_tab",
        "w_tab",
        "route_tab",
        "cb_tab",
        "sig_tab",
        "meta_tab",
    }


@pytest.mark.parametrize("ep", [2, 4])
@pytest.mark.parametrize("layout", ["aligned", "mixed", "no_collective"])
def test_megamoe_context_agrees_on_return_layout(ep, layout):
    api = _require_ppu_device()
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_name(device) != "PPU-ZW810E":
        pytest.skip("requires the tuned PPU return path")
    pool_tokens = 1 << 20
    block_m = 128 if ep == 2 else 64
    storage = torch.empty(24, dtype=torch.int8, device=device)
    meta = torch.empty((pool_tokens // block_m, 32), dtype=torch.int32, device=device)
    for rank in range(ep):
        offset = 8 if layout == "mixed" and rank == ep - 1 else 0
        votes = []

        def all_ranks_true(local):
            votes.append(local)
            # EP2 first checks the group-wide partial-reduction capacity.
            return (ep == 2 and len(votes) == 1) or layout == "aligned"

        context = api.MegaMoEContext(
            config=api.MegaMoEConfig(16384, 2048, 512, 256, 8, ep, rank),
            device=device,
            buffers={"l1_tok": storage[offset : offset + 16], "l2_meta": meta},
            peer_tables={},
            options=dict(
                POOL_TOKENS=pool_tokens, BLOCK_M=block_m, AGGREGATE_COMPLETION=True
            ),
            reset_table=meta,
            reset_programs=0,
            barrier=None,
            owners=(),
            output=meta,
            all_ranks_true=None if layout == "no_collective" else all_ranks_true,
        )
        if layout == "no_collective":
            assert not votes
        else:
            assert votes == ([True] if ep == 2 else []) + [offset == 0]
        assert context._gather_input is (layout == "aligned")
        assert context._ep2_partial_reduce is (ep == 2 and layout != "no_collective")


@pytest.mark.parametrize("shape", [(1, 128, 128, 4, 1), (17, 256, 384, 5, 3)])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("routing", ["normal", "skew", "dropped"])
def test_megamoe_int8_correctness(shape, ids_dtype, routing):
    api = _require_ppu_device()
    t, h, i, e, k = shape
    rng = torch.Generator().manual_seed(918)
    x = torch.randint(-8, 9, (t, h), dtype=torch.int8, generator=rng)
    scales = torch.rand(t, 1, generator=rng) * 0.02 + 0.01
    w1 = torch.randint(-8, 9, (e, 2 * i, h), dtype=torch.int8, generator=rng)
    s1 = torch.rand(e, 2 * i, 1, generator=rng) * 0.02 + 0.01
    w2 = torch.randint(-8, 9, (e, h, i), dtype=torch.int8, generator=rng)
    s2 = torch.rand(e, h, 1, generator=rng) * 0.02 + 0.01
    router = torch.rand(t, k, generator=rng)
    ids = torch.randint(e, (t, k), dtype=ids_dtype, generator=rng)
    if routing == "skew":
        ids.fill_(e - 1)
    elif routing == "dropped":
        ids.fill_(-1)
    expected = reference_int8_moe(x, scales, w1, s1, w2, s2, router, ids)
    context = make_ppu_context(
        api.MegaMoEConfig(t, h, i, e, k),
        torch.device("cuda", torch.cuda.current_device()),
    )
    actual = api.megamoe(
        x.cuda(),
        scales.cuda(),
        api.pack_gate_up_rows(w1.cuda()),
        api.pack_gate_up_rows(s1.cuda()),
        w2.cuda(),
        s2.cuda(),
        router.cuda(),
        ids.cuda(),
        context=context,
    )
    assert actual is context.output
    assert actual.dtype == torch.bfloat16 and actual.shape == (t, h)
    actual = actual.cpu()
    if routing == "dropped":
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert_relative_l2(actual, expected)


def test_megamoe_context_reuse_with_changing_routes():
    api = _require_ppu_device()
    t, h, i, e, k = 17, 256, 384, 5, 3
    device = torch.device("cuda", torch.cuda.current_device())
    context = make_ppu_context(api.MegaMoEConfig(t, h, i, e, k), device)
    rng = torch.Generator().manual_seed(123)
    x = torch.randint(-8, 9, (t, h), dtype=torch.int8, generator=rng)
    scales = torch.rand(t, 1, generator=rng) * 0.02 + 0.01
    w1 = torch.randint(-8, 9, (e, 2 * i, h), dtype=torch.int8, generator=rng)
    s1 = torch.rand(e, 2 * i, 1, generator=rng) * 0.02 + 0.01
    w2 = torch.randint(-8, 9, (e, h, i), dtype=torch.int8, generator=rng)
    s2 = torch.rand(e, h, 1, generator=rng) * 0.02 + 0.01
    inputs = (
        x.to(device),
        scales.to(device),
        api.pack_gate_up_rows(w1.to(device)),
        api.pack_gate_up_rows(s1.to(device)),
        w2.to(device),
        s2.to(device),
    )
    pointers = {name: tensor.data_ptr() for name, tensor in context.buffers.items()}
    for routing in ("normal", "skew", "dropped", "normal"):
        ids = torch.randint(e, (t, k), generator=rng, dtype=torch.int32)
        if routing == "skew":
            ids.fill_(e - 1)
        elif routing == "dropped":
            ids.fill_(-1)
        router = torch.rand(t, k, generator=rng)
        expected = reference_int8_moe(x, scales, w1, s1, w2, s2, router, ids)
        # Poison both state and payload to expose missing reset/overwrite paths.
        for tensor in context.buffers.values():
            tensor.fill_(7)
        actual = api.megamoe(
            *inputs, router.to(device), ids.to(device), context=context
        )
        assert actual is context.output
        if routing == "dropped":
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        else:
            assert_relative_l2(actual.cpu(), expected)
        assert pointers == {
            name: tensor.data_ptr() for name, tensor in context.buffers.items()
        }


def test_megamoe_rejects_mismatched_context():
    from dataclasses import replace

    api = _require_ppu_device()
    device = torch.device("cuda", torch.cuda.current_device())
    context = make_ppu_context(api.MegaMoEConfig(1, 128, 128, 4, 1), device)
    inputs = (
        torch.empty((1, 128), device=device, dtype=torch.int8),
        torch.empty((1, 1), device=device, dtype=torch.float32),
        torch.empty((4, 256, 128), device=device, dtype=torch.int8),
        torch.empty((4, 256, 1), device=device, dtype=torch.float32),
        torch.empty((4, 128, 128), device=device, dtype=torch.int8),
        torch.empty((4, 128, 1), device=device, dtype=torch.float32),
        torch.empty((1, 1), device=device, dtype=torch.float32),
        torch.empty((1, 1), device=device, dtype=torch.int32),
    )
    with pytest.raises(TypeError, match="MegaMoEContext"):
        api.megamoe(*inputs, context=None)
    with pytest.raises(ValueError, match="context dimensions and device"):
        api.megamoe(
            *inputs, context=replace(context, config=replace(context.config, tokens=2))
        )
    with pytest.raises(ValueError, match="context dimensions and device"):
        api.megamoe(*inputs, context=replace(context, device=torch.device("cpu")))


@pytest.mark.parametrize("num_ranks", [2, 4, 8])
@pytest.mark.parametrize("model_shape", [False, True], ids=["odd", "model"])
def test_megamoe_expert_parallel(num_ranks, model_shape):
    _require_ppu_device()
    if torch.cuda.device_count() < num_ranks:
        pytest.skip(f"requires {num_ranks} visible PPU devices")
    from benchmark.test_megamoe import _PPUBenchmarkConfig, _launch_ppu_megamoe

    config = _PPUBenchmarkConfig(
        num_ranks=num_ranks,
        tokens=32 if model_shape else 17,
        hidden=2048 if model_shape else 256,
        intermediate=512 if model_shape else 384,
        num_experts=256 if model_shape else num_ranks * 5,
        topk=8 if model_shape else 3,
        stages=4,
        warmup=0,
        iterations=1,
        timeout=120,
        ppu=True,
        validate=True,
    )
    assert _launch_ppu_megamoe(config) > 0


@pytest.mark.parametrize("tokens", [216, 448])
def test_megamoe_ep4_repeated_combine(tokens):
    _require_ppu_device()
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four visible PPU devices")
    from benchmark.test_megamoe import _PPUBenchmarkConfig, _launch_ppu_megamoe

    config = _PPUBenchmarkConfig(
        num_ranks=4,
        tokens=tokens,
        hidden=2048,
        intermediate=512,
        num_experts=256,
        topk=8,
        stages=4,
        warmup=10,
        iterations=500,
        timeout=120,
        ppu=True,
        validate=True,
    )
    assert _launch_ppu_megamoe(config) > 0


def test_megamoe_launcher_parses_every_rank():
    from benchmark.test_megamoe import _parse_rank_latencies

    stdout = "\n".join(
        [
            "MEGAMOE_RESULT scope=kernel rank=1 world=2 latency_us=1500.000",
            "MEGAMOE_RESULT scope=kernel rank=0 world=2 latency_us=1250.000",
        ]
    )
    assert _parse_rank_latencies(stdout, 2) == (1.25, 1.5)
    with pytest.raises(RuntimeError, match="every rank"):
        _parse_rank_latencies(stdout.splitlines()[0], 2)
    with pytest.raises(RuntimeError, match="every rank"):
        _parse_rank_latencies("\n".join([stdout.splitlines()[0]] * 2), 2)


def test_megamoe_worker_cli_preserves_test_config(monkeypatch):
    from benchmark.test_megamoe import (
        _PPUBenchmarkConfig,
        _parse_ppu_worker_config,
        _ppu_worker_arguments,
    )

    config = _PPUBenchmarkConfig(
        num_ranks=2,
        tokens=17,
        hidden=256,
        intermediate=384,
        num_experts=10,
        topk=3,
        stages=4,
        warmup=2,
        iterations=3,
        timeout=90,
        ppu=True,
        validate=True,
    )
    monkeypatch.setattr("sys.argv", ["worker", *_ppu_worker_arguments(config)])
    assert _parse_ppu_worker_config() == config


def test_megamoe_fast_silu_preserves_bf16_domain():
    """Compare the optimized BF16 map with the original FP32 div_rn formula."""
    api = _require_ppu_device()
    import triton
    import triton.language as tl

    kernel = importlib.import_module(
        api.megamoe.__module__.rsplit(".", 1)[0] + ".kernel"
    )

    @triton.jit
    def evaluate(reference, actual, fallback, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        gate = index.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        up = tl.full((BLOCK,), 1.0, tl.float32)
        expected = tl.div_rn(gate, 1.0 + tl.exp(-gate)).to(tl.bfloat16)
        optimized = kernel.swiglu_bf16(gate, up, True)
        original = kernel.swiglu_bf16(gate, up, False)
        tl.store(reference + index, expected.to(tl.uint16, bitcast=True))
        tl.store(actual + index, optimized.to(tl.uint16, bitcast=True))
        tl.store(fallback + index, original.to(tl.uint16, bitcast=True))

    reference = torch.empty(65536, device="cuda", dtype=torch.int16)
    actual = torch.empty_like(reference)
    fallback = torch.empty_like(reference)
    evaluate[(256,)](reference, actual, fallback, 256)
    reference = reference.cpu().to(torch.int32) & 65535
    finite_input = (torch.arange(65536, dtype=torch.int32) & 0x7F80) != 0x7F80
    for result in (actual, fallback):
        result = result.cpu().to(torch.int32) & 65535
        torch.testing.assert_close(
            result[finite_input], reference[finite_input], rtol=0, atol=0
        )
        # NaN payload/sign are unspecified; preserve all output classifications.
        for values in (reference, result):
            is_special = (values & 0x7F80) == 0x7F80
            classification = is_special.to(torch.int32) + (
                is_special & ((values & 127) != 0)
            ).to(torch.int32)
            if values is reference:
                expected_classification = classification
            else:
                torch.testing.assert_close(
                    classification, expected_classification, rtol=0, atol=0
                )


if __name__ == "__main__" and _IS_ACCURACY_WORKER:
    _run_accuracy_worker()
