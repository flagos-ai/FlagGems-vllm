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

"""Distributed test helpers adapted from MegaPreAttention PR #1007 (89b0a51)."""

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
from flaggems_vllm.ops.mrope import _axis_for_pair
from flaggems_vllm.testing import assert_close


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


def independent_reference(inputs, reference_group=None):
    """Oracle starts at original shards; never reads candidate scratch/results."""
    c, t = inputs.config, inputs.positions.shape[1]
    if reference_group is None:
        hidden = inputs.hidden.new_empty((t, inputs.hidden.shape[1]))
        dist.all_gather_into_tensor(hidden, inputs.hidden, group=inputs.group)
    else:
        shards = [None] * inputs.group.size()
        dist.all_gather_object(shards, inputs.hidden.cpu(), group=reference_group)
        hidden = torch.cat(shards).to(inputs.hidden.device)
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
        assert_close(a, b, torch.bfloat16, atol=BF16_ATOL)
        assert torch.isfinite(a).all()
    torch.testing.assert_close(actual.value, expected.value, rtol=0, atol=0)
    torch.testing.assert_close(actual.gate, expected.gate, rtol=0, atol=0)
    assert_close(inputs.cache, expected_cache, torch.bfloat16, atol=BF16_ATOL)
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


def make_inputs(tokens, rank, model_dims, edge, fixture_path, sections, *, tp):
    hidden_size, q_heads, kv_heads = model_dims
    c = Config(
        q_heads // tp,
        max(1, kv_heads // tp),
        mrope_section=sections,
        global_kv_heads=kv_heads,
    )
    weights = load_fixture(fixture_path, model_dims)["weights"]
    kv_rank = rank // max(1, tp // kv_heads)
    weight = torch.cat(
        (
            weights["q_proj.weight"].chunk(tp)[rank],
            weights["k_proj.weight"].narrow(0, kv_rank * c.kv_size, c.kv_size),
            weights["v_proj.weight"].narrow(0, kv_rank * c.kv_size, c.kv_size),
        )
    ).cuda()
    generator = torch.Generator(device="cuda").manual_seed(20261010 + rank + tokens)
    hidden = (
        torch.randn(
            tokens // tp,
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
        if rank == 0:
            # Contiguous shards can still have an unaligned base address.
            storage = hidden.new_empty((hidden.numel() + 1,))
            shifted = storage[1:].view_as(hidden)
            shifted.copy_(hidden)
            hidden = shifted
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
        num_heads=c.num_q_heads * inputs.group.size(),
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
def _megapreattention_post(
    projected,
    positions,
    q_norm,
    k_norm,
    rope,
    kv_cache,
    slots,
    query,
    key,
    tokens,
    eps,
    PROJ_STRIDE: tl.constexpr,
    POS_AXIS_STRIDE: tl.constexpr,
    POS_TOKEN_STRIDE: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    CACHE_BLOCK_STRIDE: tl.constexpr,
    CACHE_HEAD_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    CACHE_CAPACITY: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    SECTION_H: tl.constexpr,
    SECTION_W: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    token = (tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
    head = tl.program_id(1)
    d = tl.arange(0, 256)
    valid = token < tokens
    is_query = head < Q_HEADS
    kv_head = head - Q_HEADS
    column = tl.where(is_query, head * 512, Q_HEADS * 512 + kv_head * 256)
    norm = q_norm if is_query else k_norm
    src = projected + token[:, None] * PROJ_STRIDE + column

    x = tl.load(src + d[None, :], valid[:, None], other=0).to(tl.float32)
    rms = tl.rsqrt(tl.sum(x * x, 1) / 256.0 + eps)
    w = tl.load(norm + d).to(tl.float32)
    normalized = (x * rms[:, None] * (1.0 + w[None, :])).to(tl.bfloat16)

    # NeoX pairs the two halves of the rotary prefix. Round both inputs to
    # BF16 before rotation, matching the separate Gemma RMSNorm output.
    partner_d = d ^ 32
    partner_x = tl.load(
        src + partner_d[None, :], valid[:, None] & (d[None, :] < 64), other=0
    ).to(tl.float32)
    partner_w = tl.load(norm + partner_d, d < 64, other=0).to(tl.float32)
    partner = (partner_x * rms[:, None] * (1.0 + partner_w[None, :])).to(tl.bfloat16)
    pair = d % 32
    axis = _axis_for_pair(pair, None, 0, SECTION_H, SECTION_W, True, False, 256)
    pos = tl.load(
        positions + token[:, None] * POS_TOKEN_STRIDE + axis[None, :] * POS_AXIS_STRIDE,
        valid[:, None] & (d[None, :] < 64),
        other=0,
    )
    phase = rope + pos * ROPE_STRIDE + pair[None, :]
    rotate = valid[:, None] & (d[None, :] < 64)
    cos = tl.load(phase, rotate, other=1).to(tl.float32)
    sin = tl.load(phase + 32, rotate, other=0).to(tl.float32)
    signed_sin = tl.where(d[None, :] < 32, -sin, sin)
    rotated = normalized.to(tl.float32) * cos + partner.to(tl.float32) * signed_sin
    result = tl.where(d[None, :] < 64, rotated.to(tl.bfloat16), normalized)

    if is_query:
        offset = (token[:, None] * Q_HEADS + head) * 256 + d[None, :]
        tl.store(query + offset, result, valid[:, None])
    else:
        offset = (token[:, None] * KV_HEADS + kv_head) * 256 + d[None, :]
        tl.store(key + offset, result, valid[:, None])
        slot = tl.load(slots + token * SLOT_STRIDE, valid, other=-1)
        cache_valid = valid & (slot >= 0) & (slot < CACHE_CAPACITY)
        cache_offset = (
            (slot // CACHE_BLOCK_SIZE) * CACHE_BLOCK_STRIDE
            + kv_head * CACHE_HEAD_STRIDE
            + (slot % CACHE_BLOCK_SIZE) * CACHE_TOKEN_STRIDE
        )
        target = kv_cache + cache_offset[:, None] + d[None, :]
        v = tl.load(src + KV_HEADS * 256 + d[None, :], cache_valid[:, None], other=0)
        tl.store(target, result, cache_valid[:, None])
        tl.store(target + 256, v, cache_valid[:, None])


def run_symmmem_post(inputs):
    """Benchmark the original PR #1226 implementation."""
    from torch.distributed import _symmetric_memory as symm_mem

    c = inputs.config
    tokens = inputs.positions.shape[1]
    group_name = inputs.group.group_name
    symm_mem.enable_symm_mem_for_group(group_name)
    _, (projected,) = torch.ops.symm_mem.fused_all_gather_matmul(
        inputs.hidden, [inputs.weight.t()], 0, group_name
    )
    query = inputs.hidden.new_empty((tokens, c.q_size))
    key = inputs.hidden.new_empty((tokens, c.kv_size))
    block_t = 4
    _megapreattention_post[
        (triton.cdiv(tokens, block_t), c.num_q_heads + c.num_kv_heads)
    ](
        projected,
        inputs.positions,
        inputs.q_norm,
        inputs.k_norm,
        inputs.rope,
        inputs.cache,
        inputs.slots,
        query,
        key,
        tokens,
        c.eps,
        projected.stride(0),
        *inputs.positions.stride(),
        inputs.rope.stride(0),
        inputs.slots.stride(0),
        *inputs.cache.stride()[:3],
        inputs.cache.shape[2],
        inputs.cache.shape[0] * inputs.cache.shape[2],
        c.num_q_heads,
        c.num_kv_heads,
        c.mrope_section[1],
        c.mrope_section[2],
        block_t,
        num_warps=4,
        enable_fp_fusion=False,
    )
    value = projected[:, -c.kv_size :]
    gate = projected[:, : 2 * c.q_size].view(tokens, c.num_q_heads, 2 * c.head_dim)
    return Output(query, key, value, gate[..., c.head_dim :])


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
    tp,
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
        world_size=tp,
        device_id=device,
        timeout=timedelta(seconds=90),
    )
    # PCCL 2.1 on this PPU can fault in the reference AllGather at TP3.
    reference_group = dist.new_group(backend="gloo") if tp & (tp - 1) else None
    graphs = {}
    try:
        inputs = make_inputs(
            tokens, rank, model_dims, edge, fixture_path, sections, tp=tp
        )
        if not iterations:
            with pytest.raises(NotImplementedError, match="requires BF16"):
                run_candidate(replace(inputs, hidden=inputs.hidden.float()))
            with pytest.raises(NotImplementedError, match="requires D256 and R64"):
                run_candidate(inputs, rotary_dim=32)
            with pytest.raises(ValueError, match="invalid packed"):
                run_candidate(replace(inputs, weight=inputs.weight[:-1]))
            with pytest.raises(ValueError, match="invalid MRoPE"):
                run_candidate(inputs, mrope_section=(0, 0, 0))

        reference = replace(inputs, cache=torch.full_like(inputs.cache, 0.375))
        expected = independent_reference(reference, reference_group)
        calls = {"candidate": (run_candidate, inputs)}
        if iterations:
            calls["baseline"] = (
                run_symmmem_post,
                replace(inputs, cache=torch.full_like(inputs.cache, 0.375)),
            )
        if larger_tokens is not None:
            larger = make_inputs(
                larger_tokens, rank, model_dims, edge, None, sections, tp=tp
            )
            larger_reference = replace(
                larger, cache=torch.full_like(larger.cache, 0.375)
            )
            larger_expected = independent_reference(larger_reference, reference_group)
            # Initialize the larger workspace before capture.
            run_candidate(larger)
            torch.cuda.synchronize()
        errors, outputs = {}, {}
        for name, (call, args) in calls.items():
            errors[name] = _check(call(args), expected, args, reference.cache)
            for _ in range(warmup):
                call(args)
            torch.cuda.synchronize()
            if reference_group is None:
                dist.barrier(device_ids=[rank])
            else:
                dist.barrier(group=reference_group)
            graph = graphs[name] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[name] = call(args)
        if larger_tokens is not None:
            larger_actual = run_candidate(larger)
            _check(larger_actual, larger_expected, larger, larger_reference.cache)
        inputs.hidden.mul_(-0.75)
        reference.cache.fill_(0.375)
        expected = independent_reference(reference, reference_group)
        for name, (_, args) in calls.items():
            args.cache.fill_(0.375)
            graphs[name].replay()
            torch.cuda.synchronize()
            for tensor, error in _check(
                outputs[name], expected, args, reference.cache
            ).items():
                errors[name][tensor] = max(errors[name][tensor], error)

        if not iterations:
            # Replay without a rank barrier between launches. Each launch must
            # protect the peer's remaining readers of the previous payload.
            changed = inputs.hidden.clone()
            for _ in range(4):
                changed.mul_(-0.5)
            reference.cache.fill_(0.375)
            expected = independent_reference(
                replace(reference, hidden=changed), reference_group
            )
            inputs.cache.fill_(0.375)
            for _ in range(4):
                inputs.hidden.mul_(-0.5)
                graphs["candidate"].replay()
            torch.cuda.synchronize()
            for tensor, error in _check(
                outputs["candidate"], expected, inputs, reference.cache
            ).items():
                errors["candidate"][tensor] = max(errors["candidate"][tensor], error)

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
        from flaggems_vllm.runtime.backend._thead.ops.megapreattention import (
            clear_megapreattention_cache,
        )

        clear_megapreattention_cache()
        dist.destroy_process_group()


def run_distributed(
    tokens,
    *,
    tp=2,
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
                tp,
                model_dims,
                edge,
                sections,
                fixture_path,
                warmup,
                iterations,
                larger_tokens,
            ),
            nprocs=tp,
            join=False,
        )
        deadline = time.monotonic() + 240
        try:
            while not context.join(timeout=1):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"MegaPreAttention TP{tp} workers exceeded 240 seconds"
                    )
            return [
                json.loads((Path(directory) / f"rank{rank}.json").read_text())
                for rank in range(tp)
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
