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
import importlib
import inspect
from pathlib import Path

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
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate import megagate
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate.api import (
    EXPERTS,
    HIDDEN,
    PROFILE,
    PROFILES,
    ROUTED_SCALING_FACTOR,
    SUPPORTED_TOKENS,
    TOPK,
    MegaGateContextError,
    UnpreparedShapeError,
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

EXPECTED_SOURCE_LEAVES = {
    "dsv4_flash": {
        **{tokens: "v363" for tokens in DSV4_EAGER_TOKENS[:8]},
        **{tokens: "v376" for tokens in DSV4_EAGER_TOKENS[8:]},
    },
    "dsv32": {
        **{tokens: "v371" for tokens in (1028, 1756, 2048, 3076, 3805, 8276)},
        **{tokens: "v372" for tokens in (12292, 12372)},
        **{tokens: "v435" for tokens in (14305, 16384)},
    },
    "qwen_h2048_e256_k8": {
        **{tokens: "v366" for tokens in (1034, 1035, 1036, 1352, 2048, 3104)},
        **{tokens: "v365" for tokens in (4107, 4108, 4138)},
        **{tokens: "v364" for tokens in (4434, 4435, 4436)},
        **{tokens: "v363" for tokens in (9309, 9330, 11469, 13421, 13422)},
        16384: "v367",
    },
    "qwen_h4096_e512_k10": {
        **{tokens: "v371" for tokens in (1035, 1036)},
        2048: "v400",
        **{tokens: "v412" for tokens in (4107, 4108)},
        **{tokens: "v414" for tokens in (4434, 4435)},
        5240: "v431",
        8214: "v394",
        **{tokens: "v430" for tokens in (13421, 13422)},
        16384: "v415",
    },
}

# One representative exact-T case per production schedule family.  The full
# 50-shape performance matrix lives in the dedicated acceptance harness; this
# smaller set makes every distinct fused leaf compile and prove its numerical
# semantics in ordinary Hopper CI.
SCHEDULE_FAMILY_CASES = (
    ("dsv32", 1028, "v371"),
    ("dsv32", 12292, "v372"),
    ("dsv32", 14305, "v435"),
    ("qwen_h2048_e256_k8", 1034, "v366"),
    ("qwen_h2048_e256_k8", 4107, "v365"),
    ("qwen_h2048_e256_k8", 4434, "v364"),
    ("qwen_h2048_e256_k8", 9309, "v363"),
    ("qwen_h2048_e256_k8", 16384, "v367"),
    ("qwen_h4096_e512_k10", 1035, "v371"),
    ("qwen_h4096_e512_k10", 2048, "v400"),
    ("qwen_h4096_e512_k10", 4107, "v412"),
    ("qwen_h4096_e512_k10", 4434, "v414"),
    ("qwen_h4096_e512_k10", 5240, "v431"),
    ("qwen_h4096_e512_k10", 8214, "v394"),
    ("qwen_h4096_e512_k10", 13421, "v430"),
    ("qwen_h4096_e512_k10", 16384, "v415"),
)

LEAF_FILES_BY_PROFILE = {
    "dsv4_flash": ("dsv4_clusterless.py", "dsv4_quadmath.py"),
    "dsv32": (
        "dsv32_clusterless.py",
        "dsv32_dualmath.py",
        "dsv32_quadmath.py",
    ),
    "qwen_h2048_e256_k8": tuple(
        f"qwen_h2048_v{version}.py" for version in range(363, 368)
    ),
    "qwen_h4096_e512_k10": (
        "qwen_h4096_e64_cluster8.py",
        "qwen_h4096_t2048.py",
        "qwen_h4096_t410x.py",
        "qwen_h4096_t443x.py",
        "qwen_h4096_t5240.py",
        "qwen_h4096_t8214.py",
        "qwen_h4096_t1342x.py",
        "qwen_h4096_t16384.py",
    ),
}

# Some entry modules have two mutually exclusive ABI branches, but every
# runtime path executes exactly one of these runner sites.
RUNNER_LAUNCH_SITES_BY_PROFILE = {
    "dsv4_flash": 1,
    "dsv32": 2,
    "qwen_h2048_e256_k8": 2,
    "qwen_h4096_e512_k10": 1,
}


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


def test_megagate_registry_preserves_final50_provenance() -> None:
    assert set(PROFILES) == set(EXPECTED_SOURCE_LEAVES)
    assert sum(len(profile.supported_tokens) for profile in PROFILES.values()) == 50
    for name, expected in EXPECTED_SOURCE_LEAVES.items():
        profile = PROFILES[name]
        assert dict(profile.source_leaf_by_tokens) == expected
        assert profile.supported_tokens == tuple(expected)


@pytest.mark.parametrize("profile_name", tuple(EXPECTED_SOURCE_LEAVES))
def test_megagate_profile_module_matches_registry(profile_name: str) -> None:
    profile = PROFILES[profile_name]
    module = importlib.import_module(profile.module)
    assert module.PROFILE == profile.name
    assert (module.HIDDEN, module.EXPERTS, module.TOPK) == (
        profile.hidden,
        profile.experts,
        profile.topk,
    )
    assert module.ROUTED_SCALING_FACTOR == profile.routed_scaling_factor
    assert tuple(module.SUPPORTED_TOKENS) == profile.supported_tokens
    assert module.TOKEN_TO_LEAF == dict(profile.source_leaf_by_tokens)
    assert module.MegaGateContextError is MegaGateContextError
    assert module.UnpreparedShapeError is UnpreparedShapeError

    context = create_megagate_context(profile=profile_name)
    assert context.profile == profile_name
    assert context.profile_spec is profile
    assert context.supported_tokens == profile.supported_tokens
    assert context.source_leaf_by_tokens == dict(profile.source_leaf_by_tokens)
    context.close()


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
    assert prepare["routed_scaling_factor"].default is None
    assert prepare["correction_bias"].default is None

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
    assert hot_path["routed_scaling_factor"].default is None
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


@pytest.mark.parametrize("profile_name", tuple(EXPECTED_SOURCE_LEAVES))
def test_megagate_source_is_library_only_eager_single_launch(
    profile_name: str,
) -> None:
    profile = PROFILES[profile_name]
    module = importlib.import_module(profile.module)
    source = inspect.getsource(module)
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

    run_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run"
    ]
    assert len(run_functions) == 1
    runner_launches = [
        node
        for node in ast.walk(run_functions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Subscript)
        and _dotted_name(node.func.value) == "leaf.runner"
    ]
    assert len(runner_launches) == RUNNER_LAUNCH_SITES_BY_PROFILE[profile_name]
    assert module.MegaGateContext.kernel_count_per_call == 1
    assert not any(
        name.endswith(".warmup")
        for node in ast.walk(run_functions[0])
        if isinstance(node, ast.Call)
        for name in [_dotted_name(node.func)]
        if name is not None
    )

    kernels_dir = Path(module.__file__).with_name("kernels")
    for leaf_file in LEAF_FILES_BY_PROFILE[profile_name]:
        leaf_source = (kernels_dir / leaf_file).read_text(encoding="utf-8")
        leaf_imports = _imported_modules(leaf_source)
        assert not any(
            imported_module == "vllm" or imported_module.startswith("vllm.")
            for imported_module in leaf_imports
        )
        assert not any(
            imported_module == "deep_gemm" or imported_module.startswith("deep_gemm.")
            for imported_module in leaf_imports
        )
        leaf_calls = {
            name
            for node in ast.walk(ast.parse(leaf_source))
            if isinstance(node, ast.Call)
            for name in [_dotted_name(node.func)]
            if name is not None
        }
        assert "torch.cuda.CUDAGraph" not in leaf_calls
        assert "torch.cuda.graph" not in leaf_calls
        assert not any(name.endswith(".replay") for name in leaf_calls)


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


def _stable_descending_indices(values: torch.Tensor, topk: int) -> torch.Tensor:
    """Match the production tie rule: equal scores prefer smaller expert id."""

    return torch.argsort(values, dim=1, descending=True, stable=True)[:, :topk]


def _reference_non_dsv4_routing(
    profile_name: str,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    profile = PROFILES[profile_name]
    if profile.routing == "grouped_sigmoid_bias":
        assert correction_bias is not None
        # DSV3.2 GateLinear keeps its model BF16 producer boundary.  Routing
        # itself is FP32, including sigmoid, correction bias, and normalization.
        logits = torch.mm(x, gate_weight.T).float()
        raw_scores = logits.sigmoid()
        ranking = raw_scores + correction_bias
        group_scores = (
            ranking.view(x.shape[0], 8, 32)
            .sort(dim=2, descending=True, stable=True)
            .values[:, :, :2]
            .sum(dim=2)
        )
        group_indices = _stable_descending_indices(group_scores, 4)
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_indices, True)
        expert_mask = group_mask.unsqueeze(2).expand(-1, -1, 32).reshape_as(ranking)
        ranking = ranking.masked_fill(~expert_mask, float("-inf"))
        indices = _stable_descending_indices(ranking, profile.topk)
        weights = raw_scores.gather(1, indices)
    else:
        assert profile.routing == "softmax"
        assert correction_bias is None
        # Qwen's production baseline is BF16 ReplicatedLinear followed by an
        # FP32 router.  Preserve that producer boundary before selecting and
        # normalizing experts; an FP32-accumulator-only oracle can disagree at
        # top-k boundaries and would not validate the chain being replaced.
        logits = torch.mm(x, gate_weight.T).float()
        indices = _stable_descending_indices(logits, profile.topk)
        # Selecting from the full softmax and then renormalizing is exactly a
        # softmax over the selected logits, while avoiding a large temporary.
        weights = torch.softmax(logits.gather(1, indices), dim=1)

    weights = weights / weights.sum(dim=1, keepdim=True)
    weights = weights * profile.routed_scaling_factor
    return weights.float(), indices.to(torch.int32)


@pytest.mark.parametrize(
    "profile_name,tokens,source_leaf",
    SCHEDULE_FAMILY_CASES,
    ids=[
        f"{profile_name}-{source_leaf}-T{tokens}"
        for profile_name, tokens, source_leaf in SCHEDULE_FAMILY_CASES
    ],
)
@pytest.mark.skipif(
    not HOPPER_MEGAGATE_AVAILABLE,
    reason="MegaGate correctness requires NVIDIA SM90 and its TLE APIs",
)
def test_megagate_new_schedule_family_hopper_correctness(
    profile_name: str,
    tokens: int,
    source_leaf: str,
) -> None:
    profile = PROFILES[profile_name]
    assert profile.source_leaf_by_tokens[tokens] == source_leaf
    generator = torch.Generator(device="cuda").manual_seed(
        20261112 + tokens + profile.hidden
    )
    x = torch.randn(
        (tokens, profile.hidden),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    gate_weight = torch.randn(
        (profile.experts, profile.hidden),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).mul_(profile.hidden**-0.5)
    correction_bias = None
    if profile.routing == "grouped_sigmoid_bias":
        correction_bias = torch.randn(
            (profile.experts,),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        ).mul_(0.01)

    context = create_megagate_context(profile=profile_name)
    try:
        context.prepare(
            x,
            gate_weight,
            correction_bias,
            routed_scaling_factor=profile.routed_scaling_factor,
        )
        assert context.prepared_tokens == (tokens,)
        assert context.source_leaf_by_tokens[tokens] == source_leaf

        actual_weights, actual_indices = megagate(
            x,
            gate_weight,
            correction_bias,
            context=context,
            routed_scaling_factor=profile.routed_scaling_factor,
        )
        torch.cuda.synchronize()
        expected_weights, expected_indices = _reference_non_dsv4_routing(
            profile_name,
            x,
            gate_weight,
            correction_bias,
        )
        torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
        torch.testing.assert_close(
            actual_weights,
            expected_weights,
            rtol=2.0e-4,
            atol=2.0e-4,
        )
        torch.testing.assert_close(
            actual_weights.sum(dim=1),
            torch.full(
                (tokens,),
                profile.routed_scaling_factor,
                dtype=torch.float32,
                device="cuda",
            ),
            rtol=0,
            atol=2.0e-3,
        )
    finally:
        context.close()


@pytest.mark.skipif(
    not HOPPER_MEGAGATE_AVAILABLE,
    reason="MegaGate stream contract requires NVIDIA SM90 and its TLE APIs",
)
@pytest.mark.parametrize(
    "profile_name,tokens",
    (
        ("dsv4_flash", 1028),
        ("dsv32", 1028),
        ("qwen_h2048_e256_k8", 4434),
        ("qwen_h4096_e512_k10", 1035),
    ),
)
def test_megagate_requires_prepare_on_each_cuda_stream(
    profile_name: str, tokens: int
) -> None:
    profile = PROFILES[profile_name]
    x = torch.zeros((tokens, profile.hidden), dtype=torch.bfloat16, device="cuda")
    gate_weight = torch.zeros(
        (profile.experts, profile.hidden), dtype=torch.bfloat16, device="cuda"
    )
    correction_bias = (
        torch.zeros((profile.experts,), dtype=torch.float32, device="cuda")
        if profile.routing.endswith("_bias")
        else None
    )
    context = create_megagate_context(profile=profile_name)
    context.prepare(
        x,
        gate_weight,
        correction_bias,
        routed_scaling_factor=profile.routed_scaling_factor,
    )

    default_stream = torch.cuda.current_stream()
    other_stream = torch.cuda.Stream()
    other_stream.wait_stream(default_stream)
    with torch.cuda.stream(other_stream):
        outputs = (
            torch.empty((tokens, profile.topk), dtype=torch.float32, device="cuda"),
            torch.empty(
                (tokens, profile.topk),
                dtype=torch.int64 if profile_name == "dsv4_flash" else torch.int32,
                device="cuda",
            ),
        )
        with pytest.raises(UnpreparedShapeError, match="current CUDA stream"):
            megagate(
                x,
                gate_weight,
                correction_bias,
                context=context,
                routed_scaling_factor=profile.routed_scaling_factor,
                out=outputs,
            )
        context.prepare(
            x,
            gate_weight,
            correction_bias,
            routed_scaling_factor=profile.routed_scaling_factor,
        )
        weights, indices = megagate(
            x,
            gate_weight,
            correction_bias,
            context=context,
            routed_scaling_factor=profile.routed_scaling_factor,
            out=outputs,
        )
    default_stream.wait_stream(other_stream)
    assert weights is outputs[0]
    assert indices is outputs[1]
    context.close()


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
