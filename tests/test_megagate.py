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

"""Contract and Hopper correctness tests for the public MegaGate API."""

from __future__ import annotations

import ast
import inspect

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate import (
    MegaGateContext,
    create_megagate_context,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate import (
    dsv4 as dsv4_module,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate import (
    megagate,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate.api import (
    EXPERTS,
    HIDDEN,
    PROFILE,
    ROUTED_SCALING_FACTOR,
    SUPPORTED_TOKENS,
    TOPK,
)
from flaggems_vllm.utils.tle_capabilities import (
    supports_megagate_tle,
    supports_module_wide_named_barriers,
)

pytestmark = pytest.mark.megagate

DSV4_EAGER_TOKENS = (
    1028,
    2048,
    3076,
    3805,
    4100,
    4835,
    7884,
    8221,
    13021,
    16384,
)


def _has_hopper_megagate() -> bool:
    if flaggems_vllm.vendor_name != "nvidia" or not torch.cuda.is_available():
        return False
    try:
        if tuple(torch.cuda.get_device_capability()) != (9, 0):
            return False
        import triton.experimental.tle.language as tle
    except (ImportError, RuntimeError):
        return False
    return supports_megagate_tle(tle) and supports_module_wide_named_barriers()


HOPPER_MEGAGATE_AVAILABLE = _has_hopper_megagate()


def _imported_modules(source: str) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


@pytest.mark.megagate
def test_megagate_dsv4_public_contract() -> None:
    assert PROFILE == "dsv4_flash"
    assert (HIDDEN, EXPERTS, TOPK) == (4096, 256, 6)
    assert ROUTED_SCALING_FACTOR == 1.5
    assert SUPPORTED_TOKENS == DSV4_EAGER_TOKENS

    context = MegaGateContext()
    factory_context = create_megagate_context()
    assert context.profile == PROFILE
    assert factory_context.profile == PROFILE
    assert context.prepared_tokens == ()
    assert context.kernel_count_per_call == 1
    assert {
        tokens: dsv4_module.LEAF_IMPLEMENTATION_BY_TOKENS[tokens]
        for tokens in SUPPORTED_TOKENS
    } == {
        **{tokens: "clusterless" for tokens in SUPPORTED_TOKENS[:8]},
        **{tokens: "quadmath" for tokens in SUPPORTED_TOKENS[8:]},
    }


def test_megagate_api_signatures_keep_prepare_outside_hot_path() -> None:
    prepare = inspect.signature(MegaGateContext.prepare).parameters
    assert list(prepare) == [
        "self",
        "x",
        "gate_weight",
        "correction_bias",
        "routed_scaling_factor",
    ]
    assert prepare["routed_scaling_factor"].kind is inspect.Parameter.KEYWORD_ONLY
    assert prepare["routed_scaling_factor"].default == ROUTED_SCALING_FACTOR

    hot_path = inspect.signature(megagate).parameters
    assert list(hot_path) == [
        "x",
        "gate_weight",
        "correction_bias",
        "context",
        "routed_scaling_factor",
        "out",
    ]
    assert hot_path["context"].kind is inspect.Parameter.KEYWORD_ONLY
    assert hot_path["context"].default is inspect.Parameter.empty
    assert hot_path["out"].default is None


def test_megagate_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unsupported MegaGate profile"):
        create_megagate_context(profile="not-a-profile")


def test_megagate_rejects_unknown_context_before_tensor_access() -> None:
    with pytest.raises(TypeError, match="context must be MegaGateContext"):
        megagate(None, None, None, context=object())


def test_megagate_context_close_releases_host_state() -> None:
    context = MegaGateContext()
    context.close()
    assert context.prepared_tokens == ()


@pytest.mark.parametrize("tokens", [1, 1024, 4101, 16385])
def test_megagate_exact_token_dispatch_fails_closed_on_cpu(tokens: int) -> None:
    # Meta tensors keep this source/API test cheap and prove that an unsupported
    # exact T is rejected before any CUDA or TLE operation is attempted.
    x = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device="meta")
    gate_weight = torch.empty((EXPERTS, HIDDEN), dtype=torch.bfloat16, device="meta")
    correction_bias = torch.empty((EXPERTS,), dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match=rf"rejects T={tokens}"):
        megagate(
            x,
            gate_weight,
            correction_bias,
            context=MegaGateContext(),
        )


def test_megagate_supported_token_rejects_non_cuda_abi() -> None:
    tokens = SUPPORTED_TOKENS[0]
    x = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device="meta")
    gate_weight = torch.empty((EXPERTS, HIDDEN), dtype=torch.bfloat16, device="meta")
    correction_bias = torch.empty((EXPERTS,), dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="violates the DSV4 ABI"):
        megagate(
            x,
            gate_weight,
            correction_bias,
            context=MegaGateContext(),
        )


def test_megagate_source_is_library_only_eager_single_launch() -> None:
    source = inspect.getsource(dsv4_module)
    imported = _imported_modules(source)
    assert not any(
        module == "vllm" or module.startswith("vllm.") for module in imported
    )
    assert not any(
        module == "deep_gemm" or module.startswith("deep_gemm.") for module in imported
    )

    tree = ast.parse(source)
    call_names = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for name in [_dotted_name(node.func)]
        if name is not None
    }
    assert "torch.cuda.CUDAGraph" not in call_names
    assert "torch.cuda.graph" not in call_names
    assert not any(name.endswith(".replay") for name in call_names)

    runner_launches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Subscript)
        and _dotted_name(node.func.value) == "leaf.runner"
    ]
    assert len(runner_launches) == 1


@pytest.mark.parametrize("tokens", DSV4_EAGER_TOKENS)
@pytest.mark.skipif(
    not HOPPER_MEGAGATE_AVAILABLE,
    reason="MegaGate correctness requires NVIDIA SM90 and its TLE APIs",
)
def test_megagate_dsv4_hopper_correctness_and_public_export(tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20261112)
    x = torch.randn(
        (tokens, HIDDEN),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    gate_weight = torch.randn(
        (EXPERTS, HIDDEN),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).mul_(HIDDEN**-0.5)
    correction_bias = torch.randn(
        (EXPERTS,),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).mul_(0.01)

    assert flaggems_vllm.megagate is megagate
    assert flaggems_vllm.create_megagate_context is create_megagate_context

    context = flaggems_vllm.create_megagate_context(profile=PROFILE)
    context.prepare(
        x,
        gate_weight,
        correction_bias,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
    )
    assert context.prepared_tokens == (tokens,)

    weights_out = torch.empty((tokens, TOPK), dtype=torch.float32, device="cuda")
    indices_out = torch.empty((tokens, TOPK), dtype=torch.int64, device="cuda")
    weights, indices = flaggems_vllm.megagate(
        x,
        gate_weight,
        correction_bias,
        context=context,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        out=(weights_out, indices_out),
    )
    torch.cuda.synchronize()

    assert weights is weights_out
    assert indices is indices_out
    assert tuple(weights.shape) == (tokens, TOPK)
    assert tuple(indices.shape) == (tokens, TOPK)
    assert weights.dtype == torch.float32
    assert indices.dtype == torch.int64
    assert bool(torch.all((indices >= 0) & (indices < EXPERTS)))
    sorted_indices = indices.sort(dim=1).values
    assert bool(torch.all(sorted_indices[:, 1:] != sorted_indices[:, :-1]))

    try:
        logits = torch.mm(x, gate_weight.T, out_dtype=torch.float32)
    except TypeError:
        logits = x.float() @ gate_weight.float().T
    raw_scores = torch.nn.functional.softplus(logits).sqrt()
    ranking = raw_scores + correction_bias
    # vLLM's fused_topk_bias resolves equal ranking scores by preferring the
    # smaller expert id.  A stable descending sort preserves the original
    # ascending expert order and therefore models that tie rule exactly;
    # torch.topk does not define the order of equal elements.
    expected_indices = torch.argsort(
        ranking,
        dim=1,
        descending=True,
        stable=True,
    )[
        :, :TOPK
    ].to(torch.int64)
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)

    expected_weights = raw_scores.gather(1, expected_indices)
    expected_weights = (
        expected_weights / expected_weights.sum(dim=1, keepdim=True)
    ) * ROUTED_SCALING_FACTOR
    torch.testing.assert_close(weights, expected_weights, rtol=5e-3, atol=5e-4)

    if tokens == DSV4_EAGER_TOKENS[0]:
        rebound_x = x.clone()
        rebound_weight = gate_weight.clone()
        rebound_bias = correction_bias.clone()
        rebound_weights = torch.empty_like(weights)
        rebound_indices = torch.empty_like(indices)

        # Clearing address-bound descriptors and preparing an already-compiled
        # leaf must fully rebind both X and W before the next hot-path call.
        context.clear_cached_descriptors()
        context.prepare(rebound_x, rebound_weight, rebound_bias)
        assert context._weight_descriptor_entries[64] is not None
        actual_weights, actual_indices = flaggems_vllm.megagate(
            rebound_x,
            rebound_weight,
            rebound_bias,
            context=context,
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
            out=(rebound_weights, rebound_indices),
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual_weights, weights, rtol=0, atol=0)
        torch.testing.assert_close(actual_indices, indices, rtol=0, atol=0)

        allocated_weights, allocated_indices = flaggems_vllm.megagate(
            rebound_x,
            rebound_weight,
            rebound_bias,
            context=context,
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(allocated_weights, weights, rtol=0, atol=0)
        torch.testing.assert_close(allocated_indices, indices, rtol=0, atol=0)


@pytest.mark.skipif(
    not HOPPER_MEGAGATE_AVAILABLE,
    reason="MegaGate stream contract requires NVIDIA SM90 and its TLE APIs",
)
def test_megagate_requires_prepare_on_each_cuda_stream() -> None:
    tokens = DSV4_EAGER_TOKENS[0]
    x = torch.zeros((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    gate_weight = torch.zeros((EXPERTS, HIDDEN), dtype=torch.bfloat16, device="cuda")
    correction_bias = torch.zeros((EXPERTS,), dtype=torch.float32, device="cuda")
    context = create_megagate_context()
    context.prepare(x, gate_weight, correction_bias)

    default_stream = torch.cuda.current_stream()
    other_stream = torch.cuda.Stream()
    other_stream.wait_stream(default_stream)
    with torch.cuda.stream(other_stream):
        outputs = (
            torch.empty((tokens, TOPK), dtype=torch.float32, device="cuda"),
            torch.empty((tokens, TOPK), dtype=torch.int64, device="cuda"),
        )
        with pytest.raises(
            dsv4_module.UnpreparedShapeError, match="current CUDA stream"
        ):
            megagate(
                x,
                gate_weight,
                correction_bias,
                context=context,
                out=outputs,
            )
        context.prepare(x, gate_weight, correction_bias)
        weights, indices = megagate(
            x,
            gate_weight,
            correction_bias,
            context=context,
            out=outputs,
        )
    default_stream.wait_stream(other_stream)
    assert weights is outputs[0]
    assert indices is outputs[1]


@pytest.mark.skipif(
    not HOPPER_MEGAGATE_AVAILABLE,
    reason="MegaGate prepare rollback requires NVIDIA SM90 and its TLE APIs",
)
def test_megagate_prepare_failure_rolls_back_context(monkeypatch) -> None:
    tokens = DSV4_EAGER_TOKENS[0]
    x = torch.zeros((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    gate_weight = torch.zeros((EXPERTS, HIDDEN), dtype=torch.bfloat16, device="cuda")
    correction_bias = torch.zeros((EXPERTS,), dtype=torch.float32, device="cuda")
    clusterless = dsv4_module._load_leaves()["clusterless"]

    def fail_warmup(*args, **kwargs):
        raise RuntimeError("injected compile failure")

    monkeypatch.setattr(
        clusterless._b128_clusterless_last_arriver_kernel,
        "warmup",
        fail_warmup,
    )
    context = create_megagate_context()
    with pytest.raises(RuntimeError, match="injected compile failure"):
        context.prepare(x, gate_weight, correction_bias)

    assert context.prepared_tokens == ()
    assert context._device_index is None
    assert context._modules is None
    assert context._workspaces == {}
    assert context._weight_descriptor_entries == {64: None, 256: None}
