# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the generic offline port, without operator GPU recipes."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

try:
    HAS_FLAGTREE = importlib.util.find_spec("triton.flagtune") is not None
except ModuleNotFoundError:
    HAS_FLAGTREE = False

pytestmark = pytest.mark.skipif(
    not HAS_FLAGTREE, reason="generic offline contracts require optional FlagTree"
)

if HAS_FLAGTREE:
    from flaggems_vllm.flagtune.offline.contracts.operator_contract import (
        OperatorConfigError,
        load_operator_benchmark_spec,
        public_operator_name,
        resolve_public_operator,
    )
    from flaggems_vllm.flagtune.offline.train import config_space
    from flaggems_vllm.flagtune.offline.train.route import common, resolver


def _recipe():
    """A synthetic floating-point operation, not a W8A8 training contract."""
    return {
        "schema_version": 3,
        "op_id": "flaggems_vllm/test_operator",
        "variants": {
            "general": {
                "inputs": {"n": {"min": 1}},
                "params": {"BLOCK_SIZE": {"values": [64, 128]}},
                "features": [{"name": "n", "op": "ident", "args": ["n"]}],
            }
        },
        "pretune": {
            "shape": {"fields": {"n": {"type": "int", "min": 1}}, "identity": ["n"]},
            "dispatch": {"policy": "first_match", "order": ["general"]},
            "benchmark": {
                "tensors": {
                    "x": {"factory": "randn", "shape": ["n"], "dtype": "runtime"}
                },
                "invoke": {"kind": "flaggems_vllm_public", "args": ["x"]},
            },
        },
    }


def test_public_operator_namespace_does_not_import_flaggems():
    sentinel = lambda: None
    assert public_operator_name("flaggems_vllm/test_operator") == "test_operator"
    assert (
        resolve_public_operator(
            SimpleNamespace(test_operator=sentinel), "flaggems_vllm/test_operator"
        )
        is sentinel
    )
    with pytest.raises(OperatorConfigError, match="expected 'flaggems_vllm'"):
        public_operator_name("flaggems/test_operator")


def test_recipe_compiles_with_real_flagtree_schema(tmp_path):
    path = tmp_path / "synthetic.yaml"
    path.write_text(yaml.safe_dump(_recipe()))
    spec = load_operator_benchmark_spec(path)
    assert spec.op_id == "flaggems_vllm/test_operator"
    assert spec.public_operator_name == "test_operator"
    assert spec.resolve_variant({"n": 64}) == "general"
    assert spec.shape.identity == ("n",)
    assert [item.name for item in spec.benchmark.tensors] == ["x"]


@pytest.mark.parametrize("kind", ["flag_gems_public", "python_import", "os.system"])
def test_recipe_rejects_other_invocation_mechanisms(tmp_path, kind):
    payload = _recipe()
    payload["pretune"]["benchmark"]["invoke"]["kind"] = kind
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(OperatorConfigError, match="flaggems_vllm_public"):
        load_operator_benchmark_spec(path)


def test_generic_port_does_not_register_mm_routes(monkeypatch):
    assert common.BACKEND_MODULES == {}
    assert not resolver.has_route_resolver("flaggems_vllm/mm")
    monkeypatch.setattr(resolver, "ROUTE_RESOLVERS", {})
    resolver.register_route_resolver(
        "flaggems_vllm/test_operator",
        lambda a, b, context: {"variant": context["variant"], "lhs": a},
    )
    assert resolver.resolve_route(
        "flaggems_vllm/test_operator",
        {"lhs": 3},
        runtime_context={"variant": "general"},
    ) == {"variant": "general", "lhs": 3}
    with pytest.raises(ValueError, match="no route resolver"):
        resolver.resolve_route("flaggems_vllm/mm", 1, 2)


def test_recipe_identity_retains_explicit_layouts_without_guessing_axes():
    values = {
        "M": 8,
        "N": 16,
        "K": 32,
        "B_layout": "transposed_2d",
        "stride_b0": 1,
        "stride_b1": 16,
    }
    assert common.recipe_layout_metadata(values) == {
        "layouts": {"B": "transposed_2d"},
        "stride_rules": {"stride_b0": 1, "stride_b1": 16},
    }
    assert common.recipe_layout_metadata({"M": 8, "N": 16, "K": 32}) == {
        "layouts": {},
        "stride_rules": {},
    }
    identity = common.make_recipe_id(
        "flaggems_vllm/test_operator",
        "metax",
        values,
        ["float16"],
        ["float16"],
        "general",
    )
    assert identity != common.make_recipe_id(
        "flaggems_vllm/test_operator",
        "metax",
        {**values, "stride_b1": 32},
        ["float16"],
        ["float16"],
        "general",
    )


@pytest.fixture
def runtime_binding(monkeypatch):
    import flaggems_vllm

    observed = {}
    declared = [object(), object()]
    expanded = [object(), object(), object()]
    pre_hook = object()
    operator = lambda: None
    expanded_mode = object()

    def configs_for_mode(op_name, mode):
        observed["mode_lookup"] = (op_name, mode)
        return expanded, "expanded"

    tuner = SimpleNamespace(
        _flagtune_op_name="test_legacy",
        _flagtune_expand_op_name="test_expand",
        _flagtune_yaml_path="actual-space.yaml",
        _flagtune_pre_hook=pre_hook,
        _flagtune_default_configs=declared,
        _flagtune_configs_for_mode=configs_for_mode,
    )

    def find_target(public, op_id, variant):
        observed["binding"] = (public, op_id, variant)
        return object(), tuner

    def get_configs(op_name, **kwargs):
        observed["get_configs"] = (op_name, kwargs)
        return expanded

    runtime = ModuleType("flaggems_vllm.runtime")
    runtime.device = SimpleNamespace(vendor_name="metax")
    runtime.TuningMode = SimpleNamespace(EXPANDED=expanded_mode)
    runtime.ops_get_configs = get_configs
    libentry = ModuleType("flaggems_vllm.flagtune.offline.runtime.tuner")
    libentry.find_flagtune_benchmark_target = find_target
    monkeypatch.setitem(sys.modules, "flaggems_vllm.runtime", runtime)
    monkeypatch.setitem(
        sys.modules, "flaggems_vllm.flagtune.offline.runtime.tuner", libentry
    )
    monkeypatch.setattr(flaggems_vllm, "runtime", runtime, raising=False)
    monkeypatch.setattr(flaggems_vllm, "test_operator", operator, raising=False)
    return SimpleNamespace(
        observed=observed,
        tuner=tuner,
        expanded=expanded,
        runtime=runtime,
        libentry=libentry,
        operator=operator,
    )


def test_runtime_candidates_reuse_actual_binding_and_mode(runtime_binding):
    binding = runtime_binding
    op_id = "flaggems_vllm/test_operator"
    assert (
        config_space.runtime_op_name_for_variant(op_id, "general", "maca")
        == "test_expand"
    )
    assert (
        config_space.runtime_configs_for_variant(op_id, "general", platform="metax")
        == binding.expanded
    )
    assert binding.observed["binding"] == (binding.operator, op_id, "general")
    assert binding.observed["mode_lookup"] == (
        "test_legacy",
        binding.runtime.TuningMode.EXPANDED,
    )
    assert "get_configs" not in binding.observed


def test_runtime_override_retains_bound_pre_hook(runtime_binding):
    binding = runtime_binding
    config_space.runtime_configs_for_variant(
        "flaggems_vllm/test_operator", "general", yaml_path="override.yaml"
    )
    assert binding.observed["get_configs"] == (
        "test_expand",
        {"yaml_path": "override.yaml", "pre_hook": binding.tuner._flagtune_pre_hook},
    )
    assert binding.tuner._flagtune_yaml_path == "actual-space.yaml"


def test_runtime_lookup_rejects_wrong_backend_and_ambiguous_binding(runtime_binding):
    with pytest.raises(ValueError, match="active backend"):
        config_space.runtime_configs_for_variant(
            "flaggems_vllm/test_operator", "general", platform="hygon"
        )

    def ambiguous(*args):
        raise RuntimeError("ambiguous variant binding")

    runtime_binding.libentry.find_flagtune_benchmark_target = ambiguous
    with pytest.raises(RuntimeError, match="ambiguous"):
        config_space.runtime_configs_for_variant(
            "flaggems_vllm/test_operator", "general"
        )
    assert (
        config_space.runtime_configs_for_variant(
            "flaggems_vllm/not_exported", "general"
        )
        is None
    )


def test_cli_roots_and_actual_flagtree_training_api():
    expected_root = Path(__file__).resolve().parents[2]
    for name in ("pretune", "train", "rebench", "compare", "package"):
        module = importlib.import_module(f"flaggems_vllm.flagtune.offline.cli.{name}")
        if hasattr(module, "PROJECT_ROOT"):
            assert module.PROJECT_ROOT == expected_root
        assert callable(module.main)

    train = importlib.import_module("flaggems_vllm.flagtune.offline.cli.train")
    ranker = importlib.import_module("triton.flagtune.training.ranker")
    args = train.build_parser().parse_args(
        [
            "--shape-config",
            "synthetic-shapes.yaml",
            "--flagtune-config",
            "synthetic.yaml",
            "--variant",
            "general",
            "--model-version",
            "1.0.0",
        ]
    )
    train.validate_args(args)
    options = train._training_options(args)
    assert isinstance(options, ranker.XGBoostTrainingOptions)
    assert options.seed == 2026
    inspect.signature(ranker.train_xgboost_ranker).bind(
        object(), Path("rows.jsonl"), options
    )
    inspect.signature(ranker.export_ranker_model).bind(
        object(),
        object(),
        Path("out"),
        {},
        identity=object(),
        dtypes=["float16"],
        gpu={},
        model_version="1.0.0",
    )
