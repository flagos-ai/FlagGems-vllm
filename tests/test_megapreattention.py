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

import hashlib
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

import flaggems_vllm

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
    mrope_mode: str = "qwen"

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


def load_fixture_weights(path, model_dims, head_dim):
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
        return fixture["weights"]
    return make_synthetic_weights(model_dims, head_dim)


def run_candidate(inputs, **overrides):
    if flaggems_vllm.vendor_name == "thead":
        megapreattention = flaggems_vllm.megapreattention
        overrides.setdefault("mrope_mode", inputs.config.mrope_mode)
    else:
        from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megapreattention import (
            megapreattention,
        )

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
            **kwargs,
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
    mrope_mode="qwen",
    fixture_path=None,
    multimodal=False,
):
    from vllm.model_executor.layers.rotary_embedding import get_rope

    hidden_size, q_heads, kv_heads = model_dims
    config = Config(
        q_heads // tp,
        max(1, kv_heads // tp),
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        mrope_section=tuple(mrope_section),
        eps=eps,
        global_kv_heads=kv_heads,
        mrope_mode=mrope_mode,
    )
    weights = load_fixture_weights(fixture_path, model_dims, head_dim)
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
    if edge and flaggems_vllm.vendor_name == "thead" and rank == 0:
        storage = hidden.new_empty(hidden.numel() + 1)
        shifted = storage[1:].view_as(hidden)
        shifted.copy_(hidden)
        hidden = shifted
    if multimodal:
        positions[1].add_(13)
        positions[2].add_(29)
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
    rope = get_rope(
        head_dim,
        tokens + 29,
        is_neox_style=True,
        rope_parameters={
            "rope_type": "openpangu" if mrope_mode == "interleaved" else "default",
            "rope_theta": 1.0e7,
            "partial_rotary_factor": rotary_dim / head_dim,
            "mrope_section": list(mrope_section),
            "mrope_interleaved": True,
        },
        dtype=torch.bfloat16,
    )
    # Keep vLLM's native axis selection and match the kernels' FP32
    # intermediates rather than rounding both rotation products to BF16.
    rope.apply_rotary_emb.enable_fp32_compute = True
    rope.cos_sin_cache = rope.cos_sin_cache.to("cuda", dtype=torch.bfloat16)
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


def independent_reference(inputs, rope, reference_group=None):
    """Oracle starts at original shards; never reads candidate scratch/results."""
    t = inputs.positions.shape[1]
    if reference_group is None:
        hidden = inputs.hidden.new_empty((t, inputs.hidden.shape[1]))
        dist.all_gather_into_tensor(hidden, inputs.hidden, group=inputs.group)
    else:
        shards = [None] * inputs.group.size()
        dist.all_gather_object(shards, inputs.hidden.cpu(), group=reference_group)
        hidden = torch.cat(shards).to(inputs.hidden.device)
    qkv = hidden @ inputs.weight.t()
    return _reference_postprocess(inputs, rope, qkv)


def _reference_postprocess(inputs, rope, qkv):
    c, t = inputs.config, inputs.positions.shape[1]
    qgate, key, value = qkv.split((2 * c.q_size, c.kv_size, c.kv_size), dim=-1)
    query, gate = qgate.view(t, c.num_q_heads, 2 * c.head_dim).chunk(2, dim=-1)

    def normalize(x, weight):
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + c.eps)
        return (x * (weight.float() + 1)).to(torch.bfloat16)

    query = normalize(query, inputs.q_norm).reshape(t, c.q_size)
    key = normalize(key.view(t, c.num_kv_heads, c.head_dim), inputs.k_norm)
    # The OpenPangu subclass overrides forward, while Qwen supplies forward_native.
    apply_rope = rope.forward if c.mrope_mode == "interleaved" else rope.forward_native
    query, key = apply_rope(inputs.positions, query, key)
    valid = inputs.slots >= 0
    slots = inputs.slots[valid]
    block, offset = slots // inputs.cache.shape[2], slots % inputs.cache.shape[2]
    inputs.cache[block, :, offset, : c.head_dim] = key[valid]
    inputs.cache[block, :, offset, c.head_dim :] = value.view(
        t, c.num_kv_heads, c.head_dim
    )[valid]
    return Output(query, key.reshape(t, c.kv_size), value, gate, (qkv,))


# Keep the repository BF16 relative tolerance and bound near-zero differences.
BF16_ATOL = 1e-2


def check_outputs(actual, expected, actual_cache, expected_cache):
    for a, b in zip(actual.tensors(), expected.tensors()):
        gems_assert_close(a, b, torch.bfloat16, atol=BF16_ATOL)
        assert torch.isfinite(a).all()
    gems_assert_close(actual_cache, expected_cache, torch.bfloat16, atol=BF16_ATOL)


def check_unaligned_inputs(inputs, rope):
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
                    replace(candidate, weight=reference_weight, cache=reference_cache),
                    rope,
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
    oracle = independent_reference(replace(source, cache=oracle_cache), rope)
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
        oracle = independent_reference(replace(source, cache=oracle_cache), rope)
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
        check_unaligned_inputs(inputs["megapreattention"], rope)
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


def _check_split_rope_output(actual, expected, config):
    actual = actual.reshape(-1, config.head_dim)
    expected = expected.reshape(-1, config.head_dim)
    rotary_dim = config.rotary_dim
    gems_assert_close(
        actual[:, rotary_dim:],
        expected[:, rotary_dim:],
        torch.bfloat16,
        atol=BF16_ATOL,
    )
    actual = actual[:, :rotary_dim].reshape(-1, 2, rotary_dim // 2).float()
    expected = expected[:, :rotary_dim].reshape_as(actual).float()
    # Near cancellation, a BF16 rounding step in the input can dominate
    # one rotated component. Bound that error by the scale of its pair.
    pair_scale = expected.abs().amax(dim=1, keepdim=True)
    tolerance = torch.maximum(BF16_ATOL + 0.016 * expected.abs(), 2**-7 * pair_scale)
    error = (actual - expected).abs() / tolerance
    torch.testing.assert_close(error, error.clamp_max(1), rtol=0, atol=0)


def _check_ppu_outputs(actual, expected, inputs, ref_cache, *, split_rope=None):
    if split_rope is not None:
        (projection,) = actual.workspace
        (reference_projection,) = expected.workspace
        torch.testing.assert_close(
            projection, reference_projection, rtol=2**-7, atol=2e-5
        )
        # Normalization can amplify small differences between the two GEMMs.
        # Check the split epilogue against the native reference on its own projection.
        split_inputs = replace(inputs, cache=torch.full_like(ref_cache, 0.375))
        expected = _reference_postprocess(split_inputs, split_rope, projection)
        ref_cache = split_inputs.cache
        _check_split_rope_output(actual.query, expected.query, inputs.config)
        _check_split_rope_output(actual.key, expected.key, inputs.config)
    else:
        check_outputs(actual, expected, inputs.cache, ref_cache)

    torch.testing.assert_close(actual.value, expected.value, rtol=0, atol=0)
    torch.testing.assert_close(actual.gate, expected.gate, rtol=0, atol=0)
    block_size = inputs.cache.shape[2]
    written = torch.zeros(
        (inputs.cache.shape[0], block_size),
        device=inputs.cache.device,
        dtype=torch.bool,
    )
    valid = inputs.slots >= 0
    slots = inputs.slots[valid]
    blocks, offsets = slots // block_size, slots % block_size
    head_dim = inputs.config.head_dim
    cached = inputs.cache[blocks, :, offsets]
    tokens = inputs.positions.shape[1]
    torch.testing.assert_close(
        (cached[..., :head_dim], cached[..., head_dim:]),
        (
            actual.key.reshape(tokens, -1, head_dim)[valid],
            actual.value.reshape(tokens, -1, head_dim)[valid],
        ),
        rtol=0,
        atol=0,
    )
    written[blocks, offsets] = True
    untouched = inputs.cache.transpose(1, 2)[~written]
    torch.testing.assert_close(
        untouched, torch.full_like(untouched, 0.375), rtol=0, atol=0
    )

    return {
        name: (a.float() - b.float()).abs().max().item()
        for name, a, b in zip(
            ("query", "key", "value", "gate", "kv_cache"),
            (*actual.tensors(), inputs.cache),
            (*expected.tensors(), ref_cache),
        )
    }


def _ppu_worker(
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
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config

    from flaggems_vllm.runtime.backend._thead.ops.megapreattention import (
        clear_megapreattention_cache,
    )

    larger_tokens = kernel_config.get("larger_tokens")
    input_options = {
        name: value for name, value in kernel_config.items() if name != "larger_tokens"
    }
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
    native_context = set_current_vllm_config(VllmConfig())
    native_context.__enter__()
    try:
        inputs, rope = make_inputs(
            tokens, tp, rank, model_dims=model_dims, edge=edge, **input_options
        )
        if check_only:
            with pytest.raises(NotImplementedError, match="requires BF16"):
                run_candidate(replace(inputs, hidden=inputs.hidden.float()))
            with pytest.raises(NotImplementedError, match="requires D256 and R64"):
                run_candidate(inputs, rotary_dim=32)
            with pytest.raises(ValueError, match="invalid packed"):
                run_candidate(replace(inputs, weight=inputs.weight[:-1]))
            with pytest.raises(ValueError, match="invalid MRoPE"):
                run_candidate(inputs, mrope_section=(0, 0, 0))

            with pytest.raises(ValueError, match="invalid MRoPE mode"):
                run_candidate(inputs, mrope_mode="unknown")
            with pytest.raises(ValueError, match="missing time section"):
                run_candidate(
                    inputs, mrope_section=(0, 16, 16), mrope_mode="interleaved"
                )

        reference = replace(inputs, cache=torch.full_like(inputs.cache, 0.375))
        expected = independent_reference(reference, rope, reference_group)
        calls = {"megapreattention": (run_candidate, inputs)}
        if not check_only:
            from benchmark.test_megapreattention import (
                SplitWindow,
                measure_window,
                summarize_samples,
            )

            calls["native"] = (
                SplitWindow(inputs, rope),
                replace(inputs, cache=torch.full_like(inputs.cache, 0.375)),
            )
        if larger_tokens is not None:
            larger, larger_rope = make_inputs(
                larger_tokens,
                tp,
                rank,
                model_dims=model_dims,
                edge=edge,
                **input_options,
            )
            larger_reference = replace(
                larger, cache=torch.full_like(larger.cache, 0.375)
            )
            larger_expected = independent_reference(
                larger_reference, larger_rope, reference_group
            )
            # Initialize the larger workspace before capture.
            run_candidate(larger)
            torch.cuda.synchronize()
        errors, outputs = {}, {}
        for name, (call, args) in calls.items():
            errors[name] = _check_ppu_outputs(
                call(args),
                expected,
                args,
                reference.cache,
                split_rope=rope if name == "native" else None,
            )
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
            _check_ppu_outputs(
                larger_actual, larger_expected, larger, larger_reference.cache
            )
        inputs.hidden.mul_(-0.75)
        reference.cache.fill_(0.375)
        expected = independent_reference(reference, rope, reference_group)
        for name, (_, args) in calls.items():
            args.cache.fill_(0.375)
            graphs[name].replay()
            torch.cuda.synchronize()
            for tensor, error in _check_ppu_outputs(
                outputs[name],
                expected,
                args,
                reference.cache,
                split_rope=rope if name == "native" else None,
            ).items():
                errors[name][tensor] = max(errors[name][tensor], error)

        if check_only:
            # Replay without a rank barrier between launches. Each launch must
            # protect the peer's remaining readers of the previous payload.
            changed = inputs.hidden.clone()
            for _ in range(4):
                changed.mul_(-0.5)
            reference.cache.fill_(0.375)
            expected = independent_reference(
                replace(reference, hidden=changed), rope, reference_group
            )
            inputs.cache.fill_(0.375)
            for _ in range(4):
                inputs.hidden.mul_(-0.5)
                graphs["megapreattention"].replay()
            torch.cuda.synchronize()
            for tensor, error in _check_ppu_outputs(
                outputs["megapreattention"], expected, inputs, reference.cache
            ).items():
                errors["megapreattention"][tensor] = max(
                    errors["megapreattention"][tensor], error
                )

        rank_path = Path(directory) / f"rank{rank}.json"
        if check_only:
            rank_path.write_text(json.dumps({"rank": rank, "max_abs_error": errors}))
        else:
            names = list(calls)
            timings = measure_window(
                {name: graphs[name].replay for name in names}, names, iterations
            )
            rank_path.write_text(
                json.dumps({"rank": rank, "max_abs_error": errors, "samples": timings})
            )
        if reference_group is None:
            dist.barrier(device_ids=[rank])
        else:
            dist.barrier(group=reference_group)
        if rank == 0:
            ranks = [
                json.loads((Path(directory) / f"rank{i}.json").read_text())
                for i in range(tp)
            ]
            report = {"ranks": ranks}
            if not check_only:
                report["metrics"] = summarize_samples(ranks, names, iterations)
            (Path(directory) / "report.json").write_text(json.dumps(report))
    finally:
        for graph in graphs.values():
            graph.reset()
        torch.cuda.synchronize()
        clear_megapreattention_cache()
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
    mrope_mode="qwen",
    fixture_path=None,
    multimodal=False,
    larger_tokens=None,
):
    """Run independent rounds with a shared input and reference implementation."""
    import torch.multiprocessing as mp

    model_dims = tuple(model_dims)
    kernel_config = dict(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        mrope_section=tuple(mrope_section),
        eps=eps,
    )
    worker = _worker
    if flaggems_vllm.vendor_name == "thead":
        if mode != "cudagraph":
            raise ValueError("PPU MegaPreAttention measurement requires cudagraph mode")
        worker = _ppu_worker
        kernel_config.update(
            mrope_mode=mrope_mode,
            fixture_path=fixture_path,
            multimodal=multimodal,
            larger_tokens=larger_tokens,
        )
    reports = []
    for _ in range(rounds):
        with tempfile.TemporaryDirectory(prefix="megapreattention-") as directory:
            context = mp.spawn(
                worker,
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
        "fixture": fixture_path or "seeded synthetic",
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


def require_ppu(tp):
    if flaggems_vllm.vendor_name != "thead" or torch.cuda.device_count() < tp:
        pytest.skip(f"requires {tp} T-Head PPUs")


@pytest.mark.megapreattention
@pytest.mark.parametrize("mode", ["qwen", "interleaved"])
@pytest.mark.parametrize("sections", [(11, 11, 10), (8, 12, 12), (16, 8, 8)])
def test_megapreattention_ppu_rope_modes(mode, sections):
    require_ppu(2)
    run_distributed(
        2,
        258,
        "cudagraph",
        rounds=1,
        warmup=2,
        check_only=True,
        model_dims=(320, 8, 2),
        mrope_mode=mode,
        mrope_section=sections,
        multimodal=True,
    )


@pytest.mark.parametrize(
    "tp,tokens,model_dims,edge,sections",
    [
        (2, 1024, (2048, 16, 2), False, (11, 11, 10)),
        (2, 768, (2048, 16, 4), True, (8, 12, 12)),
        (2, 256, (256, 4, 1), True, (16, 8, 8)),
        (3, 384, (256, 6, 3), False, (11, 11, 10)),
        (4, 2048, (256, 8, 4), True, (8, 12, 12)),
        (8, 4096, (256, 16, 2), True, (16, 8, 8)),
        (16, 2048, (256, 16, 2), True, (16, 8, 8)),
        (1, 1, (256, 4, 2), True, (11, 11, 10)),
        (2, 34, (320, 8, 2), True, (16, 8, 8)),
        (2, 128, (256, 4, 1), False, (11, 11, 10)),
        (3, 387, (256, 6, 3), False, (16, 8, 8)),
        (4, 1036, (320, 8, 4), True, (8, 12, 12)),
        (8, 520, (256, 16, 2), True, (16, 8, 8)),
    ],
)
def test_megapreattention_ppu(tp, tokens, model_dims, edge, sections):
    require_ppu(tp)
    from flaggems_vllm.runtime.backend._thead.ops import megapreattention

    assert flaggems_vllm.megapreattention is megapreattention
    assert flaggems_vllm.ops.__all__.count("megapreattention") == 1
    run_distributed(
        tp,
        tokens,
        "cudagraph",
        rounds=1,
        warmup=2,
        check_only=True,
        model_dims=model_dims,
        edge=edge,
        mrope_section=sections,
    )


@pytest.mark.parametrize("tokens", [34, 256])
def test_megapreattention_replay_after_larger_shape(tokens):
    require_ppu(2)
    run_distributed(
        2, tokens, "cudagraph", rounds=1, warmup=2, check_only=True, larger_tokens=8192
    )


def test_megapreattention_large_cache_offsets():
    require_ppu(1)
    from flaggems_vllm.runtime.backend._thead.ops.megapreattention_kernel import (
        _cache_offset,
    )

    @triton.jit
    def probe(output, slot):
        head = tl.program_id(0)
        offset = _cache_offset(slot, head, 1 << 32, 1 << 30, 512, 1 << 21)
        tl.store(output + head, offset)

    output = torch.empty(4, dtype=torch.int64, device="cuda")
    probe[(4,)](output, (1 << 21) + 7)
    assert output.tolist() == [(1 << 32) + (head << 30) + 7 * 512 for head in range(4)]
