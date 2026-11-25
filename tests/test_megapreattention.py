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

"""Distributed accuracy, Graph replay and shared synthetic benchmark inputs."""

import json
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl

from . import conftest as cfg
from .accuracy_utils import gems_assert_close

pytestmark = pytest.mark.megapreattention


@dataclass(frozen=True)
class Config:
    num_q_heads: int
    num_kv_heads: int
    head_dim: int = 256
    rotary_dim: int = 64
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    eps: float = 1e-6
    global_kv_heads: int = 2

    @property
    def q_size(self):
        return self.num_q_heads * self.head_dim

    @property
    def kv_size(self):
        return self.num_kv_heads * self.head_dim

    @property
    def packed_size(self):
        return 2 * (self.q_size + self.kv_size)


@dataclass
class Inputs:
    hidden: torch.Tensor
    weight: torch.Tensor
    positions: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    rope: torch.Tensor
    cache: torch.Tensor
    slots: torch.Tensor
    scale: torch.Tensor
    group: object
    config: Config


@dataclass
class Output:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    gate: torch.Tensor  # [tokens, local_q_heads, head_dim], possibly strided.
    workspace: tuple = ()

    def tensors(self):
        return self.query, self.key, self.value, self.gate


def make_synthetic_weights(model_dims, head_dim):
    hidden_size, q_heads, kv_heads = model_dims
    generator = torch.Generator().manual_seed(20261010)
    return {
        "q_proj.weight": torch.randn(
            2 * q_heads * head_dim,
            hidden_size,
            generator=generator,
            dtype=torch.bfloat16,
        )
        * 0.02,
        "k_proj.weight": torch.randn(
            kv_heads * head_dim, hidden_size, generator=generator, dtype=torch.bfloat16
        )
        * 0.02,
        "v_proj.weight": torch.randn(
            kv_heads * head_dim, hidden_size, generator=generator, dtype=torch.bfloat16
        )
        * 0.02,
        "q_norm.weight": torch.randn(
            head_dim, generator=generator, dtype=torch.bfloat16
        )
        * 0.1,
        "k_norm.weight": torch.randn(
            head_dim, generator=generator, dtype=torch.bfloat16
        )
        * 0.1,
    }


def run_candidate(inputs):
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megapreattention import (
        megapreattention,
    )

    c = inputs.config
    return Output(
        *megapreattention(
            inputs.hidden,
            inputs.weight,
            inputs.positions,
            inputs.q_norm,
            inputs.k_norm,
            inputs.rope,
            inputs.cache,
            inputs.slots,
            inputs.group,
            num_heads=c.num_q_heads * inputs.group.size(),
            num_kv_heads=c.global_kv_heads,
            head_dim=c.head_dim,
            rotary_dim=c.rotary_dim,
            mrope_section=c.mrope_section,
            eps=c.eps,
        )
    )


def make_inputs(
    tokens,
    tp,
    rank,
    *,
    edge=False,
    model_dims=(2048, 16, 2),
    head_dim=256,
    rotary_dim=64,
    mrope_section=(11, 11, 10),
    eps=1e-6,
):
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    hidden_size, q_heads, kv_heads = model_dims
    config = Config(
        q_heads // tp,
        max(1, kv_heads // tp),
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        mrope_section=tuple(mrope_section),
        eps=eps,
        global_kv_heads=kv_heads,
    )
    weights = make_synthetic_weights(model_dims, head_dim)
    qwidth = q_heads * head_dim * 2 // tp
    kv_rank = rank // max(1, tp // kv_heads)
    kv_start = kv_rank * config.kv_size
    local = torch.cat(
        (
            weights["q_proj.weight"][rank * qwidth : (rank + 1) * qwidth],
            weights["k_proj.weight"][kv_start : kv_start + config.kv_size],
            weights["v_proj.weight"][kv_start : kv_start + config.kv_size],
        )
    ).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(20261010 + rank + tokens)
    hidden = (
        torch.randn(
            (tokens // tp, hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.05
    )
    position_base = torch.arange(tokens, device="cuda", dtype=torch.int64)
    positions = position_base.repeat(3, 1)
    slots = position_base.clone()
    if edge:
        storage = torch.empty((3, tokens + 7), device="cuda", dtype=torch.int64)
        positions = storage[:, :tokens]
        positions[0], positions[1], positions[2] = (
            position_base,
            position_base + 13,
            position_base + 29,
        )
        slot_storage = torch.empty(tokens * 2, device="cuda", dtype=torch.int64)
        slots = slot_storage[::2]
        slots.copy_(position_base * 2 + 15)
        slots[-1] = -1
    count = tokens * 2 + 128 if edge else tokens + 128
    blocks = (count + 127) // 128
    cache = torch.full(
        (blocks, config.num_kv_heads, 128, 2 * head_dim),
        0.375,
        device="cuda",
        dtype=torch.bfloat16,
    )
    if edge:
        cache = torch.full(
            (128, config.num_kv_heads, blocks, 2 * head_dim),
            0.375,
            device="cuda",
            dtype=torch.bfloat16,
        ).permute(2, 1, 0, 3)
    rope = MRotaryEmbedding(
        head_dim,
        rotary_dim,
        262144,
        1.0e7,
        True,
        torch.bfloat16,
        mrope_section=list(mrope_section),
        mrope_interleaved=True,
    )
    rope.cos_sin_cache = rope.cos_sin_cache.to("cuda")
    inputs = Inputs(
        hidden,
        local,
        positions,
        weights["q_norm.weight"].to("cuda"),
        weights["k_norm.weight"].to("cuda"),
        rope.cos_sin_cache,
        cache,
        slots,
        torch.ones(1, device="cuda"),
        dist.group.WORLD,
        config,
    )
    return inputs, rope


def independent_reference(inputs):
    """Oracle starts at original shards; never reads candidate scratch/results."""
    c, t = inputs.config, inputs.positions.shape[1]
    hidden = inputs.hidden.new_empty((t, inputs.hidden.shape[1]))
    dist.all_gather_into_tensor(hidden, inputs.hidden, group=inputs.group)
    qkv = hidden @ inputs.weight.t()
    qgate, key, value = qkv.split((2 * c.q_size, c.kv_size, c.kv_size), dim=-1)
    query, gate = qgate.view(t, c.num_q_heads, 2 * c.head_dim).chunk(2, dim=-1)
    frequency = torch.arange(c.rotary_dim // 2, device="cuda")
    axis = torch.zeros_like(frequency)
    axis[1 : 3 * c.mrope_section[1] : 3] = 1
    axis[2 : 3 * c.mrope_section[2] : 3] = 2
    position = inputs.positions[axis].t()
    cos = inputs.rope[position, frequency].float().unsqueeze(1)
    sin = inputs.rope[position, frequency + c.rotary_dim // 2].float().unsqueeze(1)

    def norm_rope(x, weight):
        x = x.float()
        normalized = (
            (
                x
                * torch.rsqrt(x.square().mean(-1, keepdim=True) + c.eps)
                * (weight.float() + 1)
            )
            .to(torch.bfloat16)
            .float()
        )
        first, second = normalized[..., : c.rotary_dim].chunk(2, dim=-1)
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )
        return torch.cat((rotated, normalized[..., c.rotary_dim :]), dim=-1).to(
            torch.bfloat16
        )

    query = norm_rope(query, inputs.q_norm).reshape(t, c.q_size)
    key = norm_rope(key.view(t, c.num_kv_heads, c.head_dim), inputs.k_norm)
    valid = inputs.slots >= 0
    slots = inputs.slots[valid]
    block, offset = slots // inputs.cache.shape[2], slots % inputs.cache.shape[2]
    inputs.cache[block, :, offset, : c.head_dim] = key[valid]
    inputs.cache[block, :, offset, c.head_dim :] = value.view(
        t, c.num_kv_heads, c.head_dim
    )[valid]
    return Output(query, key.reshape(t, c.kv_size), value, gate)


# Keep the repository BF16 relative tolerance and bound near-zero differences.
BF16_ATOL = 1e-2


def check_outputs(actual, expected, actual_cache, expected_cache):
    for a, b in zip(actual.tensors(), expected.tensors()):
        gems_assert_close(a, b, torch.bfloat16, atol=BF16_ATOL)
        assert torch.isfinite(a).all()
    gems_assert_close(actual_cache, expected_cache, torch.bfloat16, atol=BF16_ATOL)


def check_unaligned_inputs(inputs):
    """Exercise cached TMA inputs and Graph replay with only rank zero offset."""
    for field in ("hidden", "weight"):
        value = getattr(inputs, field)
        offset = int(dist.get_rank(inputs.group) == 0)
        storage = torch.empty(
            value.numel() + offset, device=value.device, dtype=value.dtype
        )
        changed = storage[offset:].view(value.shape)
        changed.copy_(value)
        candidate = replace(
            inputs, **{field: changed}, cache=torch.full_like(inputs.cache, 0.375)
        )
        assert changed.is_contiguous() and changed.data_ptr() % 16 == offset * 2
        # An aligned invocation has already populated the runtime cache.
        run_candidate(candidate)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run_candidate(candidate)
        try:
            for scale in (-0.75, -1.25):
                changed.mul_(scale)
                candidate.cache.fill_(0.375)
                reference_cache = torch.full_like(candidate.cache, 0.375)
                # Keep the oracle GEMM aligned while exercising candidate staging.
                reference_weight = candidate.weight
                if field == "weight" and offset:
                    reference_weight = reference_weight.clone()
                expected = independent_reference(
                    replace(candidate, weight=reference_weight, cache=reference_cache)
                )
                graph.replay()
                torch.cuda.synchronize()
                check_outputs(actual, expected, candidate.cache, reference_cache)
        finally:
            graph.reset()


def _worker(
    rank,
    tp,
    tokens,
    mode,
    iterations,
    warmup,
    edge,
    check_only,
    directory,
    model_dims,
    kernel_config,
):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=tp,
        init_method=f"file://{directory}/rdzv",
        device_id=torch.device("cuda", rank),
        timeout=timedelta(seconds=180),
    )
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config

    native_context = set_current_vllm_config(VllmConfig())
    native_context.__enter__()
    source, rope = make_inputs(
        tokens, tp, rank, edge=edge, model_dims=model_dims, **kernel_config
    )
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops import (
        megapreattention as impl,
    )

    names = ["megapreattention"] if check_only else ["native", "megapreattention"]
    inputs = {
        name: replace(source, cache=torch.full_like(source.cache, 0.375))
        for name in names
    }

    def candidate_with_other_current_device(value):
        # Exercise creation, cached calls and Graph capture with a different
        # current device. The API must restore the caller's device on return.
        other = (rank + 1) % tp
        with torch.cuda.device(other):
            result = run_candidate(value)
            assert torch.cuda.current_device() == other
        assert all(tensor.device == value.hidden.device for tensor in result.tensors())
        assert all(
            w.communication.device == value.hidden.device
            for w in impl._windows.values()
        )
        return result

    functions = {
        "megapreattention": (
            candidate_with_other_current_device if check_only else run_candidate
        ),
    }
    if not check_only:
        from benchmark.test_megapreattention import (
            NativeWindow,
            measure_window,
            summarize_samples,
        )

        native = NativeWindow(inputs["native"], rope, edge=edge)
        functions["native"] = native
    # Original-input oracle is independent of both native and candidate scratch.
    oracle_cache = torch.full_like(source.cache, 0.375)
    oracle = independent_reference(replace(source, cache=oracle_cache))
    for name in names:
        actual = functions[name](inputs[name])
        torch.cuda.synchronize()
        check_outputs(actual, oracle, inputs[name].cache, oracle_cache)
    for _ in range(warmup):
        for name in names:
            functions[name](inputs[name])
    torch.cuda.synchronize()
    graphs, outputs = {}, {}
    if mode == "cudagraph":
        for name in names:
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[name] = functions[name](inputs[name])
            graphs[name] = graph

    calls = {
        name: (
            graphs[name].replay
            if mode == "cudagraph"
            else lambda name=name: functions[name](inputs[name])
        )
        for name in names
    }
    # Changed input and poisoned payload must work with the same registered runtime/Graph.
    for scale in (-0.75, -1.25):
        source.hidden.mul_(scale)
        oracle_cache.fill_(0.375)
        oracle = independent_reference(replace(source, cache=oracle_cache))
        for window in impl._windows.values():
            window.communication.buffer.fill_(
                float("nan")
            )  # test-only, never production
        for name in names:
            inputs[name].cache.fill_(0.375)
            if mode == "cudagraph":
                graphs[name].replay()
                actual = outputs[name]
            else:
                actual = functions[name](inputs[name])
            torch.cuda.synchronize()
            check_outputs(actual, oracle, inputs[name].cache, oracle_cache)
    # Unsupported metadata is rejected before any collective initialization.
    unsupported = {}
    check_metadata = (
        check_only
        and tp == 2
        and tokens == 1024
        and model_dims == (2048, 16, 2)
        and kernel_config["eps"] == 1e-6
    )
    if check_metadata:
        check_unaligned_inputs(inputs["megapreattention"])
    for name in ("megapreattention",) if check_metadata else ():
        value = inputs[name]
        bad_cases = {
            "fp32": replace(value, hidden=value.hidden.float()),
            "strided_hidden": replace(
                value,
                hidden=torch.empty(
                    (tokens // tp, value.hidden.shape[1] * 2),
                    device="cuda",
                    dtype=torch.bfloat16,
                )[:, ::2],
            ),
            "requires_grad": replace(
                value, weight=value.weight.clone().requires_grad_(True)
            ),
            "wrong_heads": replace(
                value,
                config=replace(value.config, num_q_heads=value.config.num_q_heads + 1),
            ),
            "empty": replace(
                value,
                hidden=value.hidden[:0],
                positions=value.positions[:, :0],
                slots=value.slots[:0],
            ),
        }
        unsupported[name] = {}
        for label, bad in bad_cases.items():
            try:
                run_candidate(bad)
            except AssertionError:
                unsupported[name][label] = True
            else:
                raise AssertionError(f"{name} silently accepted unsupported {label}")
        del bad_cases

    samples = {name: {"wall": [], "device": []} for name in names}
    if not check_only:
        samples = measure_window(calls, names, iterations)
    record = {
        "rank": rank,
        "samples": samples,
        "unsupported": unsupported,
        "jit_free": None if check_only else True,
        "native_source": None if check_only else native.source_evidence,
    }
    ranks = [None] * tp
    dist.all_gather_object(ranks, record)
    dist.barrier()
    torch.cuda.synchronize()
    for graph in graphs.values():
        graph.reset()
    outputs.clear()
    graphs.clear()
    calls.clear()
    with torch.cuda.device((rank + 1) % tp):
        impl.clear_megapreattention_cache()
        assert torch.cuda.current_device() == (rank + 1) % tp
    if rank == 0:
        metrics = {} if check_only else summarize_samples(ranks, names, iterations)
        report = {
            "tp": tp,
            "tokens": tokens,
            "mode": mode,
            "edge": edge,
            "scope": "BF16 megapreattention window, not complete model inference",
            "correctness_passed": True,
            "measurement_jit_free": None if check_only else True,
            "iterations": iterations,
            "warmup": warmup,
            "metrics": metrics,
            "ranks": ranks,
            "fixture": {"input_origin": "seeded synthetic weights and hidden"},
        }
        (Path(directory) / "report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    dist.destroy_process_group()
    native_context.__exit__(None, None, None)


def run_distributed(
    tp,
    tokens,
    mode,
    *,
    iterations=100,
    warmup=20,
    rounds=5,
    edge=False,
    check_only=False,
    model_dims=(2048, 16, 2),
    head_dim=256,
    rotary_dim=64,
    mrope_section=(11, 11, 10),
    eps=1e-6,
):
    """Run independent rounds with seeded synthetic inputs."""
    import torch.multiprocessing as mp

    model_dims = tuple(model_dims)
    kernel_config = dict(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        mrope_section=tuple(mrope_section),
        eps=eps,
    )
    reports = []
    for _ in range(rounds):
        with tempfile.TemporaryDirectory(prefix="megapreattention-") as directory:
            context = mp.spawn(
                _worker,
                args=(
                    tp,
                    tokens,
                    mode,
                    iterations,
                    warmup,
                    edge,
                    check_only,
                    directory,
                    model_dims,
                    kernel_config,
                ),
                nprocs=tp,
                join=False,
            )
            deadline = time.monotonic() + 600
            try:
                while not context.join(timeout=1):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "megapreattention worker round exceeded 600s"
                        )
                reports.append(
                    json.loads((Path(directory) / "report.json").read_text())
                )
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
    summary = {}
    if not check_only:
        from benchmark.test_megapreattention import summarize_rounds

        summary = summarize_rounds(reports, mode)
    result = {
        "model_dims": model_dims,
        "kernel_config": kernel_config,
        "tp": tp,
        "tokens": tokens,
        "mode": mode,
        "correctness_passed": True,
        "measurement_jit_free": None if check_only else True,
        "summary": summary,
        "reports": reports,
    }
    return result


def require_hopper(tp=1):
    if not torch.cuda.is_available() or torch.cuda.device_count() < tp:
        pytest.skip(f"requires {tp} CUDA GPUs")
    if any(torch.cuda.get_device_capability(i) != (9, 0) for i in range(tp)):
        pytest.skip("requires SM90 GPUs")


# The shared worker checks changed-input Graph replay and poisoned transport.
# Cover each scheduling path without a second private-input test suite.
CASES = [
    ((2048, 16, 2), 2, 1024, "cudagraph", True),
    ((5120, 24, 4), 2, 1024, "cudagraph", False),
    ((3072, 32, 8), 2, 1024, "cudagraph", True),
]
if not cfg.QUICK_MODE:
    CASES += [
        ((2048, 16, 2), 2, 1280, "cudagraph", True),
        ((2048, 16, 2), 2, 32768, "cudagraph", True),
        ((2048, 16, 2), 4, 1536, "operator", True),
        ((2048, 16, 2), 4, 32768, "cudagraph", True),
        ((2048, 16, 2), 8, 2048, "cudagraph", True),
        ((2048, 16, 2), 8, 32768, "operator", False),
        ((5120, 24, 4), 4, 1024, "operator", True),
        ((5120, 24, 4), 8, 32768, "cudagraph", True),
        ((1024, 32, 16), 4, 1024, "cudagraph", True),
    ]


@pytest.mark.parametrize("model_dims,tp,tokens,mode,edge", CASES)
def test_megapreattention(model_dims, tp, tokens, mode, edge):
    require_hopper(tp)
    result = run_distributed(
        tp,
        tokens,
        mode,
        warmup=3,
        rounds=1,
        edge=edge,
        check_only=True,
        model_dims=model_dims,
    )
    assert result["correctness_passed"]
    assert all(len(r["ranks"]) == tp for r in result["reports"])


def test_megapreattention_metadata_bounds():
    require_hopper()

    from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megapreattention import (
        _initialize_metadata,
    )

    words, programs = 2050, 132
    flags = torch.full((words + 8,), 17, device="cuda", dtype=torch.uint64)
    counters = torch.full((programs + 8,), 17, device="cuda", dtype=torch.int64)
    _initialize_metadata[(triton.cdiv(words, 1024),)](
        flags[:words], counters[:programs], words, COUNTERS=programs, num_warps=4
    )
    torch.cuda.synchronize()
    assert flags.cpu().tolist() == [0] * words + [17] * 8
    assert counters.cpu().tolist() == [0] * programs + [17] * 8


@triton.jit
def _probe_packed_input_offsets(
    tiles, vectors, out, TILE_WORDS: tl.constexpr, offset_fn: tl.constexpr
):
    i = tl.arange(0, 4)
    offsets = offset_fn(tl.load(tiles + i), tl.load(vectors + i), TILE_WORDS)
    tl.store(out + i, offsets)


def test_megapreattention_packed_input_offset_bounds():
    require_hopper()
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megapreattention import (
        _packed_input_word_offset,
    )

    tile_words = 128 * 5120 // 2
    boundary = 2**31
    tile, remainder = divmod(boundary, tile_words)
    tiles = [tile, tile, tile + 1, 8191]
    vectors = [remainder // 4 - 1, remainder // 4, 0, tile_words // 4 - 1]
    actual = torch.empty(4, dtype=torch.int64, device="cuda")
    _probe_packed_input_offsets[(1,)](
        torch.tensor(tiles, dtype=torch.int32, device="cuda"),
        torch.tensor(vectors, dtype=torch.int32, device="cuda"),
        actual,
        TILE_WORDS=tile_words,
        offset_fn=_packed_input_word_offset,
        num_warps=1,
    )
    assert actual.tolist() == [
        boundary - 4,
        boundary,
        (tile + 1) * tile_words,
        1048576 * 5120 // 2 - 4,
    ]


ROPE_CASES = [
    (2, (16, 8, 8), 1e-5),
    (4, (8, 12, 12), 1e-4),
]


@pytest.mark.parametrize("tp,sections,eps", ROPE_CASES)
def test_megapreattention_rope_parameters(tp, sections, eps):
    require_hopper(tp)
    result = run_distributed(
        tp,
        1024,
        "cudagraph",
        warmup=3,
        rounds=1,
        edge=True,
        check_only=True,
        mrope_section=sections,
        eps=eps,
    )
    assert result["correctness_passed"]
