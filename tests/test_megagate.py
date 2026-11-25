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

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate import MegaGateContext
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate import (
    api as megagate_api,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate import (
    create_megagate_context,
    megagate,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate import (
    sqrt_softplus_h4096_e256_k6 as dsv4_module,
)
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
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate.capabilities import (
    _has_module_wide_named_barrier_support,
    is_megagate_available,
    supports_megagate_tle,
    supports_module_wide_named_barriers,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate.registry import (
    ShapeKey,
    get_kernel_plan,
    get_profile,
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


def _has_hopper_megagate() -> bool:
    if flaggems_vllm.vendor_name != "nvidia" or not torch.cuda.is_available():
        return False
    if tuple(torch.cuda.get_device_capability()) != (9, 0):
        return False
    return is_megagate_available()


HOPPER_MEGAGATE_AVAILABLE = _has_hopper_megagate()


def test_megagate_tle_capability_is_operator_local() -> None:
    def api(*args, **kwargs):
        raise AssertionError("capability probe executed a TLE primitive")

    language = SimpleNamespace(
        pipe=api,
        device_mesh=api,
        MeshConfig=api,
        distributed_barrier=api,
        shard_id=api,
        gpu=SimpleNamespace(
            alloc=api,
            local_ptr=api,
            copy=api,
            warp_specialize=api,
            alloc_barrier=api,
            barrier_wait=api,
            wgmma=api,
            wgmma_wait=api,
            smem=object(),
        ),
    )

    assert supports_megagate_tle(language)
    assert not supports_megagate_tle(None)
    for namespace in (language, language.gpu):
        for name, value in vars(namespace).copy().items():
            if callable(value):
                setattr(namespace, name, object())
                assert not supports_megagate_tle(language), name
            delattr(namespace, name)
            before = vars(namespace).copy()
            assert not supports_megagate_tle(language), name
            assert vars(namespace) == before
            setattr(namespace, name, value)


def test_megagate_named_barrier_capability_requires_fixed_contract() -> None:
    def api(*args, **kwargs):
        raise AssertionError("capability probe executed a compiler API")

    def reserve(_semantic, count, _generator=None):
        raise AssertionError("capability probe executed the frontend allocator")

    core = SimpleNamespace(
        _FIRST_VIRTUAL_NAMED_BARRIER_ID=16,
        _reserve_named_barrier_ids=reserve,
    )
    libtriton = SimpleNamespace(
        tle=SimpleNamespace(
            passes=SimpleNamespace(add_allocate_named_barriers=api),
        )
    )

    assert _has_module_wide_named_barrier_support(core, libtriton)

    core._FIRST_VIRTUAL_NAMED_BARRIER_ID = 15
    assert not _has_module_wide_named_barrier_support(core, libtriton)
    core._FIRST_VIRTUAL_NAMED_BARRIER_ID = 16

    def legacy_reserve(_semantic, count):
        raise AssertionError("capability probe executed the legacy allocator")

    core._reserve_named_barrier_ids = legacy_reserve
    assert not _has_module_wide_named_barrier_support(core, libtriton)
    core._reserve_named_barrier_ids = object()
    assert not _has_module_wide_named_barrier_support(core, libtriton)
    core._reserve_named_barrier_ids = reserve

    libtriton.tle.passes.add_allocate_named_barriers = object()
    assert not _has_module_wide_named_barrier_support(core, libtriton)


def test_megagate_named_barrier_capability_is_cached() -> None:
    first = supports_module_wide_named_barriers()
    before = supports_module_wide_named_barriers.cache_info()
    assert supports_module_wide_named_barriers() is first
    after = supports_module_wide_named_barriers.cache_info()
    assert after.misses == before.misses
    assert after.hits == before.hits + 1


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
    assert set(dsv4_module.LEAF_IMPLEMENTATION_BY_TOKENS) == set(SUPPORTED_TOKENS)
    assert set(dsv4_module.LEAF_IMPLEMENTATION_BY_TOKENS.values()) == {
        "clusterless",
        "quadmath",
    }


@pytest.mark.parametrize("profile_name", tuple(PROFILES))
def test_megagate_profile_shape_plans_match_modules(profile_name: str) -> None:
    profile = PROFILES[profile_name]
    actual_leaves = {}
    implementation_modules = set()
    for tokens in profile.supported_tokens:
        plan = get_kernel_plan(profile.shape_key(tokens))
        module = importlib.import_module(plan.implementation_module)
        implementation_modules.add(module)
        assert (module.HIDDEN, module.EXPERTS, module.TOPK) == (
            profile.hidden,
            profile.experts,
            profile.topk,
        )
        assert module.ROUTED_SCALING_FACTOR == profile.routed_scaling_factor
        assert tokens in module.SUPPORTED_TOKENS
        assert module.TOKEN_TO_LEAF[tokens] == plan.source_leaf
        assert module.MegaGateContextError is MegaGateContextError
        assert module.UnpreparedShapeError is UnpreparedShapeError
        actual_leaves[tokens] = plan.source_leaf
    assert len(implementation_modules) == 1
    assert actual_leaves == dict(profile.source_leaf_by_tokens)

    context = create_megagate_context(profile=profile_name)
    assert context.profile == profile_name
    assert context.profile_spec is profile
    assert context.supported_tokens == profile.supported_tokens
    assert context.source_leaf_by_tokens == dict(profile.source_leaf_by_tokens)
    context.close()


def test_megagate_profile_aliases_are_not_dispatch_key_fields() -> None:
    assert {"model", "module", "name", "profile"}.isdisjoint(
        ShapeKey.__dataclass_fields__
    )
    assert "module" not in type(PROFILES["dsv4_flash"]).__dataclass_fields__
    for alias, canonical, tokens in (
        ("dsv4", "dsv4_flash", 1028),
        ("deepseek_v4", "dsv4_flash", 1028),
        ("deepseek_v3_2", "dsv32", 1028),
        ("qwen_h2048", "qwen_h2048_e256_k8", 1034),
        ("qwen_h4096", "qwen_h4096_e512_k10", 1035),
    ):
        alias_profile = get_profile(alias)
        canonical_profile = get_profile(canonical)
        alias_shape = alias_profile.shape_key(tokens)
        canonical_shape = canonical_profile.shape_key(tokens)
        assert alias_shape == canonical_shape
        assert get_kernel_plan(alias_shape) is get_kernel_plan(canonical_shape)


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
    ]
    assert hot_path["context"].kind is inspect.Parameter.KEYWORD_ONLY
    assert hot_path["context"].default is inspect.Parameter.empty
    assert hot_path["routed_scaling_factor"].default is None


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


def test_megagate_prepares_exact_plans_and_caches_device_capability(
    monkeypatch,
) -> None:
    class Tensor:
        def __init__(self, shape):
            self.shape = shape
            self.ndim = len(shape)
            self.dtype = torch.bfloat16
            self.device = torch.device("cuda", 0)

    capability_queries = []
    plans = []
    closed = []

    def get_capability(device):
        capability_queries.append(device)
        return (9, 0)

    class HostContext:
        def prepare(self, x, weight, bias, *, routed_scaling_factor, plan):
            if not plans:
                plans.append(None)
                raise RuntimeError("injected prepare failure")
            plans.append(plan)

        def run(self, x, weight, bias, *, routed_scaling_factor):
            return x, weight

        def close(self):
            closed.append(self)

    monkeypatch.setattr(
        megagate_api,
        "torch",
        SimpleNamespace(
            Tensor=Tensor,
            cuda=SimpleNamespace(get_device_capability=get_capability),
        ),
    )
    monkeypatch.setattr(
        megagate_api,
        "_load_plan_module",
        lambda plan: SimpleNamespace(MegaGateContext=HostContext),
    )
    context = MegaGateContext()
    first, second = (Tensor((tokens, HIDDEN)) for tokens in SUPPORTED_TOKENS[:2])
    weight = Tensor((EXPERTS, HIDDEN))

    with pytest.raises(RuntimeError, match="injected prepare failure"):
        context.prepare(first, weight)
    assert context.prepared_tokens == ()
    with pytest.raises(UnpreparedShapeError):
        megagate(first, weight, context=context)

    for x in (first, second):
        context.prepare(x, weight)
        assert plans[-1] is get_kernel_plan(context.profile_spec.shape_key(x.shape[0]))
        assert megagate(x, weight, context=context) == (x, weight)
    assert context.prepared_tokens == SUPPORTED_TOKENS[:2]
    assert capability_queries == [first.device, first.device]

    context.close()
    assert context.prepared_tokens == ()
    assert len(closed) == 1
    with pytest.raises(UnpreparedShapeError):
        megagate(first, weight, context=context)
    assert capability_queries == [first.device, first.device]


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
    with pytest.raises(ValueError, match="requires CUDA inputs"):
        megagate(
            x,
            gate_weight,
            correction_bias,
            context=MegaGateContext(),
        )


@pytest.mark.parametrize("tokens", (1028, 16384))
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

    weights, indices = flaggems_vllm.megagate(
        x,
        gate_weight,
        correction_bias,
        context=context,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
    )
    torch.cuda.synchronize()

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
        with pytest.raises(UnpreparedShapeError, match="current CUDA stream"):
            megagate(
                x,
                gate_weight,
                correction_bias,
                context=context,
                routed_scaling_factor=profile.routed_scaling_factor,
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
        )
    default_stream.wait_stream(other_stream)
    assert tuple(weights.shape) == (tokens, profile.topk)
    assert tuple(indices.shape) == (tokens, profile.topk)
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
