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

"""TP2 test helpers adapted from MegaPreAttention PR #1007 (89b0a51)."""

import hashlib
import json
import statistics
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton
import triton.language as tl

import flaggems_vllm

from ..accuracy_utils import gems_assert_close


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
    group: object
    config: Config


@dataclass
class Output:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    gate: torch.Tensor  # [tokens, local_q_heads, head_dim], possibly strided.

    def tensors(self):
        return self.query, self.key, self.value, self.gate


def load_fixture(path=None, model_dims=(2048, 16, 2), head_dim=256):
    hidden_size, q_heads, kv_heads = model_dims
    if path is not None:
        fixture = torch.load(path, map_location="cpu", weights_only=True)
        for name, tensor in fixture["weights"].items():
            digest = hashlib.sha256(
                tensor.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            assert digest == fixture["metadata"]["weight_sha256"][name]
        expected = {
            "q_proj.weight": (2 * q_heads * head_dim, hidden_size),
            "k_proj.weight": (kv_heads * head_dim, hidden_size),
            "v_proj.weight": (kv_heads * head_dim, hidden_size),
            "q_norm.weight": (head_dim,),
            "k_norm.weight": (head_dim,),
        }
        for name, shape in expected.items():
            if tuple(fixture["weights"][name].shape) != shape:
                raise ValueError(
                    f"fixture {name} does not match model dimensions {model_dims}"
                )
        return fixture
    generator = torch.Generator().manual_seed(20261010)
    weights = {
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
    return {
        "weights": weights,
        "metadata": {"input_origin": "seeded synthetic weights and hidden"},
    }


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


BF16_ATOL = 0.01


def check_outputs(actual, expected, inputs, expected_cache):
    for a, b in zip(actual.tensors()[:2], expected.tensors()[:2]):
        gems_assert_close(a, b, torch.bfloat16, atol=BF16_ATOL)
        assert torch.isfinite(a).all()
    torch.testing.assert_close(actual.value, expected.value, rtol=0, atol=0)
    torch.testing.assert_close(actual.gate, expected.gate, rtol=0, atol=0)
    gems_assert_close(inputs.cache, expected_cache, torch.bfloat16, atol=BF16_ATOL)
    block_size = inputs.cache.shape[2]
    written = torch.zeros(
        (inputs.cache.shape[0], block_size),
        device=inputs.cache.device,
        dtype=torch.bool,
    )
    slots = inputs.slots[inputs.slots >= 0]
    written[slots // block_size, slots % block_size] = True
    untouched = inputs.cache.transpose(1, 2)[~written]
    torch.testing.assert_close(
        untouched, torch.full_like(untouched, 0.375), rtol=0, atol=0
    )


def make_inputs(tokens, rank, model_dims, edge, fixture_path, sections):
    hidden_size, q_heads, kv_heads = model_dims
    c = Config(
        q_heads // 2,
        max(1, kv_heads // 2),
        mrope_section=sections,
        global_kv_heads=kv_heads,
    )
    weights = load_fixture(fixture_path, model_dims)["weights"]
    kv_rank = rank if kv_heads > 1 else 0
    weight = torch.cat(
        (
            weights["q_proj.weight"].chunk(2)[rank],
            weights["k_proj.weight"].narrow(0, kv_rank * c.kv_size, c.kv_size),
            weights["v_proj.weight"].narrow(0, kv_rank * c.kv_size, c.kv_size),
        )
    ).cuda()
    generator = torch.Generator(device="cuda").manual_seed(20261010 + rank + tokens)
    hidden = (
        torch.randn(
            tokens // 2,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.05
    )
    base = torch.arange(tokens, device="cuda")
    positions = base.repeat(3, 1)
    slots = base.clone()
    blocks = (tokens + 127) // 128 + 1
    if edge:
        positions = torch.empty((3, tokens + 7), device="cuda", dtype=torch.int64)[
            :, :tokens
        ]
        positions[0], positions[1], positions[2] = base, base + 13, base + 29
        slots = torch.empty(tokens * 2, device="cuda", dtype=torch.int64)[::2]
        slots.copy_(2 * base + 15)
        slots[-1] = -1
        blocks = (2 * tokens + 127) // 128 + 1
        cache = torch.full(
            (128, c.num_kv_heads, blocks, 512),
            0.375,
            device="cuda",
            dtype=torch.bfloat16,
        ).permute(2, 1, 0, 3)
    else:
        cache = torch.full(
            (blocks, c.num_kv_heads, 128, 512),
            0.375,
            device="cuda",
            dtype=torch.bfloat16,
        )
    # The PR uses vLLM's MRotaryEmbedding to build this NeoX cos/sin table.
    # Generate the same frequencies in the test without a vLLM extension.
    frequency = 1e7 ** (-torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    phase = torch.arange(tokens + 29, dtype=torch.float32)[:, None] * frequency
    rope = torch.cat((phase.cos(), phase.sin()), dim=1).to("cuda", torch.bfloat16)
    return Inputs(
        hidden,
        weight,
        positions,
        weights["q_norm.weight"].cuda(),
        weights["k_norm.weight"].cuda(),
        rope,
        cache,
        slots,
        dist.group.WORLD,
        c,
    )


def run_candidate(inputs, **overrides):
    c = inputs.config
    kwargs = dict(
        num_heads=c.num_q_heads * 2,
        num_kv_heads=c.global_kv_heads,
        head_dim=c.head_dim,
        rotary_dim=c.rotary_dim,
        mrope_section=c.mrope_section,
        eps=c.eps,
    )
    kwargs.update(overrides)
    return Output(
        *flaggems_vllm.megapreattention(
            inputs.hidden,
            inputs.weight,
            inputs.positions,
            inputs.q_norm,
            inputs.k_norm,
            inputs.rope,
            inputs.cache,
            inputs.slots,
            inputs.group,
            **kwargs,
        )
    )


@triton.jit
def _store_kv(
    key,
    value,
    cache,
    slots,
    K_WIDTH: tl.constexpr,
    V_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + token)
    d = tl.arange(0, triton.next_power_of_2(K_WIDTH))
    # Cache layout is [blocks, heads, block_size, 2 * head_dim].
    target = (
        (slot // BLOCK_SIZE) * BLOCK_SIZE * K_WIDTH * 2
        + (d // 256) * BLOCK_SIZE * 512
        + (slot % BLOCK_SIZE) * 512
        + d % 256
    )
    valid = (d < K_WIDTH) & (slot >= 0)
    k = tl.load(key + token * K_WIDTH + d, valid, other=0)
    v = tl.load(value + token * V_STRIDE + d, valid, other=0)
    tl.store(cache + target, k, valid)
    tl.store(cache + target + 256, v, valid)


def run_unfused(inputs):
    """Performance baseline for the contiguous benchmark layout."""
    from torch.distributed import _symmetric_memory as symm_mem

    c = inputs.config
    tokens = inputs.positions.shape[1]
    group_name = inputs.group.group_name
    symm_mem.enable_symm_mem_for_group(group_name)
    _, (qkv,) = torch.ops.symm_mem.fused_all_gather_matmul(
        inputs.hidden, [inputs.weight.t()], 0, group_name
    )
    qgate, key, value = qkv.split((2 * c.q_size, c.kv_size, c.kv_size), dim=-1)
    query, gate = qgate.view(tokens, c.num_q_heads, 2 * c.head_dim).chunk(2, dim=-1)
    query = flaggems_vllm.gemma_rms_norm(
        query.contiguous().view(-1, c.head_dim), inputs.q_norm, c.eps
    ).view(tokens, c.q_size)
    key = flaggems_vllm.gemma_rms_norm(
        key.contiguous().view(-1, c.head_dim), inputs.k_norm, c.eps
    ).view(tokens, c.kv_size)
    flaggems_vllm.mrope(
        query,
        key,
        inputs.rope,
        inputs.positions,
        *c.mrope_section,
        c.head_dim,
        c.rotary_dim,
        is_interleaved=True,
    )
    _store_kv[(tokens,)](
        key,
        value,
        inputs.cache,
        inputs.slots,
        c.kv_size,
        value.stride(0),
        inputs.cache.shape[2],
        num_warps=4,
    )
    return Output(query, key, value, gate)


def _check(actual, expected, inputs, ref_cache):
    check_outputs(actual, expected, inputs, ref_cache)
    return {
        name: (a.float() - b.float()).abs().max().item()
        for name, a, b in zip(
            ("query", "key", "value", "gate", "kv_cache"),
            (*actual.tensors(), inputs.cache),
            (*expected.tensors(), ref_cache),
        )
    }


def _worker(
    rank,
    directory,
    tokens,
    model_dims,
    edge,
    sections,
    fixture_path,
    warmup,
    iterations,
    larger_tokens,
):
    import pytest

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{directory}/rdzv",
        rank=rank,
        world_size=2,
        device_id=device,
        timeout=timedelta(seconds=90),
    )
    graphs = {}
    try:
        inputs = make_inputs(tokens, rank, model_dims, edge, fixture_path, sections)
        if not iterations:
            with pytest.raises(NotImplementedError, match="requires BF16"):
                run_candidate(replace(inputs, hidden=inputs.hidden.float()))
            with pytest.raises(NotImplementedError, match="requires TP2, D256 and R64"):
                run_candidate(inputs, rotary_dim=32)
            with pytest.raises(ValueError, match="invalid packed"):
                run_candidate(replace(inputs, weight=inputs.weight[:-1]))
            with pytest.raises(ValueError, match="invalid MRoPE"):
                run_candidate(inputs, mrope_section=(0, 0, 0))

        reference = replace(inputs, cache=torch.full_like(inputs.cache, 0.375))
        expected = independent_reference(reference)
        calls = {"candidate": (run_candidate, inputs)}
        if iterations:
            calls["baseline"] = (
                run_unfused,
                replace(inputs, cache=torch.full_like(inputs.cache, 0.375)),
            )
        if larger_tokens is not None:
            larger = make_inputs(larger_tokens, rank, model_dims, edge, None, sections)
            larger_reference = replace(
                larger, cache=torch.full_like(larger.cache, 0.375)
            )
            larger_expected = independent_reference(larger_reference)
            # Reserve SymmMem's maximum workspace before any graph references it.
            run_candidate(larger)
            torch.cuda.synchronize()
        errors, outputs = {}, {}
        for name, (call, args) in calls.items():
            errors[name] = _check(call(args), expected, args, reference.cache)
            for _ in range(warmup):
                call(args)
            torch.cuda.synchronize()
            dist.barrier(device_ids=[rank])
            graph = graphs[name] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[name] = call(args)
        if larger_tokens is not None:
            larger_actual = run_candidate(larger)
            _check(larger_actual, larger_expected, larger, larger_reference.cache)
        inputs.hidden.mul_(-0.75)
        reference.cache.fill_(0.375)
        expected = independent_reference(reference)
        for name, (_, args) in calls.items():
            args.cache.fill_(0.375)
            graphs[name].replay()
            torch.cuda.synchronize()
            for tensor, error in _check(
                outputs[name], expected, args, reference.cache
            ).items():
                errors[name][tensor] = max(errors[name][tensor], error)

        samples = {name: [] for name in calls}
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        order = list(calls)
        from triton.runtime.jit import JITFunction

        def unexpected_compile(*args, **kwargs):
            raise AssertionError("measurement triggered Triton JIT compilation")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(JITFunction, "_do_compile", unexpected_compile)
            for i in range(iterations):
                for name in order if i % 2 == 0 else reversed(order):
                    dist.barrier(device_ids=[rank])
                    torch.cuda.synchronize()
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end))
        record = {"rank": rank, "max_abs_error": errors, "samples_ms": samples}
        (Path(directory) / f"rank{rank}.json").write_text(json.dumps(record))
    finally:
        for graph in graphs.values():
            graph.reset()
        torch.cuda.synchronize()
        dist.destroy_process_group()


def run_tp2(
    tokens,
    *,
    model_dims=(2048, 16, 2),
    edge=False,
    sections=(11, 11, 10),
    fixture_path=None,
    warmup=2,
    iterations=0,
    larger_tokens=None,
):
    with tempfile.TemporaryDirectory(prefix="megapreattention-ppu-") as directory:
        context = mp.spawn(
            _worker,
            args=(
                directory,
                tokens,
                model_dims,
                edge,
                sections,
                fixture_path,
                warmup,
                iterations,
                larger_tokens,
            ),
            nprocs=2,
            join=False,
        )
        deadline = time.monotonic() + 240
        try:
            while not context.join(timeout=1):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "MegaPreAttention TP2 workers exceeded 240 seconds"
                    )
            return [
                json.loads((Path(directory) / f"rank{rank}.json").read_text())
                for rank in range(2)
            ]
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()


def summarize_round(ranks, name):
    samples = [rank["samples_ms"][name] for rank in ranks]
    means = [statistics.mean(s) for s in samples]
    rank_max_samples = [max(sample) for sample in zip(*samples)]
    p10, *_, p90 = (
        statistics.quantiles(rank_max_samples, n=10, method="inclusive")
        if len(rank_max_samples) > 1
        else [rank_max_samples[0]] * 2
    )
    return {
        "rank_mean_ms": means,
        "rank_max_mean_ms": statistics.mean(rank_max_samples),
        "rank_max_p50_ms": statistics.median(rank_max_samples),
        "rank_max_p10_ms": p10,
        "rank_max_p90_ms": p90,
    }
