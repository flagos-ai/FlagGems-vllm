# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free contracts for the actual online inference module.

The optional FlagTree loader/proposer and tensors are mocked; no GPU, Torch,
model files, network, or dataset is required. Dtype tests check separation and
forwarding, not whether an installed FlagTree release supports FP8 archives.
Run directly with Python or through unittest discovery.
"""

import ast
import importlib.util
import os
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src/flaggems_vllm"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


class Config:
    def __init__(self, block, pre_hook=None):
        self.kwargs = {"BLOCK": block}
        self.num_warps = 4
        self.num_stages = 2
        self.num_ctas = 1
        self.pre_hook = pre_hook

    def all_kwargs(self):
        return {
            **self.kwargs,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "num_ctas": self.num_ctas,
        }


@dataclass(frozen=True)
class Identity:
    platform_key: str
    op_id: str
    variant: str
    dtype_key: str

    @property
    def artifact_key(self):
        return "/".join((self.platform_key, self.op_id, self.variant, self.dtype_key))


def tensor(dtype, device="test:0"):
    return types.SimpleNamespace(dtype=dtype, device=device)


def infer_dtypes(values):
    return tuple(
        value.dtype
        for value in (getattr(v, "base", v) for v in values)
        if hasattr(value, "dtype")
    )


class InferenceContracts(unittest.TestCase):
    def setUp(self):
        modules = {
            name: package(name)
            for name in (
                "flaggems_vllm",
                "flaggems_vllm.flagtune",
                "flaggems_vllm.flagtune.inference",
                "triton",
                "triton.flagtune",
                "triton.flagtune.runtime",
                "triton.flagtune.contract",
            )
        }
        # Keep these dependency-free contracts isolated when real FlagTree
        # modules have already been loaded by other tests in the same process.
        modules.update(
            {
                name: None
                for name in sys.modules
                if name.startswith(("triton.flagtune.", "flaggems_vllm."))
                and name not in modules
            }
        )
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.runtime = load_source("flaggems_vllm.runtime", SRC / "runtime/flagtune.py")
        self.runtime.device = types.SimpleNamespace(vendor_name="testvendor")
        modules["flaggems_vllm"].runtime = self.runtime
        load_source(
            "flaggems_vllm.flagtune.inference.status",
            SRC / "flagtune/inference/status.py",
        )
        self.cm = load_source(
            "flaggems_vllm.flagtune.inference.cost_model",
            SRC / "flagtune/inference/cost_model.py",
        )
        self.cm.print_status = Mock()
        self.dtype_resolvers = self.cm
        self.proposer_api = types.ModuleType("triton.flagtune.runtime.proposer")
        self.proposer_api.load_model_bundle = Mock()
        self.proposer_api.make_config_proposer = Mock()
        sys.modules[self.proposer_api.__name__] = self.proposer_api
        self.identity_api = types.ModuleType("triton.flagtune.contract.identity")
        self.identity_api.ModelIdentity = Identity
        self.identity_api.discover_gpu_metadata = lambda: {
            "platform_key": "test-platform"
        }
        self.identity_api.make_dtype_key = lambda values: "-".join(map(str, values))
        sys.modules[self.identity_api.__name__] = self.identity_api
        self.configs = [Config(16), Config(32)]
        self.tuner = types.SimpleNamespace(
            _flagtune_op_id="flaggems_vllm/w8a8_block_matmul",
            _flagtune_variant="test_general",
            _flagtune_op_name="w8a8_block_int8_matmul",
            _flagtune_expand_op_name="test_general",
            _flagtune_pre_hook=None,
            _flagtune_dtype_resolver=None,
            arg_names=["A", "B", "As", "Bs", "C", "M"],
            nargs={
                "A": tensor("int8"),
                "B": tensor("int8"),
                "As": tensor("float32"),
                "Bs": tensor("float32"),
                "C": tensor("float16"),
                "M": 7,
            },
            early_config_prune=None,
            _flagtune_configs_for_mode=lambda *args: (self.configs, None),
        )
        self.default = Mock(return_value=("default", {}))
        self.fallback = Mock(return_value=("fallback", {}))
        self.bench = Mock(return_value=[1.0, 0.9, 1.1])

    def run_policy(self):
        return self.cm.run_policy(
            self.tuner,
            self.bench,
            self.configs,
            (),
            {},
            supports_cost_model=True,
            infer_tensor_dtypes=infer_dtypes,
            default_policy=self.default,
            legacy_fallback=self.fallback,
        )

    def loaded(self, proposer, converter=None):
        identity = Identity("test-platform", "test/op", "general", "i8-f16")
        variant = types.SimpleNamespace(
            param_names=["BLOCK", "num_warps", "num_stages", "num_ctas"],
            param_space=types.SimpleNamespace(validate=lambda record: True),
            to_config=converter or (lambda record: Config(record["BLOCK"])),
        )
        return identity, proposer, variant

    def failure_key(self):
        return self.cm.failure_key(self.tuner, self.tuner.nargs)

    def test_auto_failure_isolated_by_dtype_and_device(self):
        with patch.object(
            self.cm, "load_model", side_effect=FileNotFoundError("missing")
        ) as loader:
            self.run_policy()
            failed = self.failure_key()
            self.tuner.nargs["A"].dtype = "float8_e4m3fn"
            self.assertFalse(self.cm.is_auto_disabled(*self.failure_key()))
            self.run_policy()
            for value in self.tuner.nargs.values():
                if hasattr(value, "device"):
                    value.device = "other:0"
            self.assertFalse(self.cm.is_auto_disabled(*self.failure_key()))
            self.run_policy()
            self.assertEqual(loader.call_count, 3)
            self.assertTrue(self.cm.is_auto_disabled(*failed))

    def test_canonical_failure_isolated_by_dtype_and_platform(self):
        with patch.object(
            self.cm, "ensure_proposer", side_effect=FileNotFoundError("missing")
        ) as loader:
            self.run_policy()
            first = self.failure_key()
            self.assertEqual(first[0], "test-platform")
            self.tuner.nargs["A"].dtype = "float8_e4m3fn"
            self.run_policy()
            second = self.failure_key()
            self.assertNotEqual(first[3], second[3])
            self.identity_api.discover_gpu_metadata = lambda: {
                "platform_key": "other-platform"
            }
            self.tuner.nargs["A"].device = "other:0"
            self.run_policy()
            third = self.failure_key()
            self.assertEqual(third[0], "other-platform")
            self.assertEqual(loader.call_count, 3)
            self.assertEqual(len(self.cm._COST_MODEL_DISABLED_OPS), 3)

    def test_auto_failure_does_not_retry_after_elapsed_time(self):
        with (
            patch("time.monotonic", return_value=100) as clock,
            patch.object(
                self.cm, "ensure_proposer", side_effect=FileNotFoundError("missing")
            ) as loader,
        ):
            self.assertEqual(self.run_policy()[0], "fallback")
            key = self.failure_key()
            self.assertEqual(key[0], "test-platform")
            for now in (160, 3600, 1_000_000):
                clock.return_value = now
                self.assertEqual(self.run_policy()[0], "fallback")
                self.assertTrue(self.cm.is_auto_disabled(*key))
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(self.cm.print_status.call_count, 1)

    def test_required_can_migrate_discovery_failure_without_reenabling_auto(self):
        with (
            patch.object(self.cm, "flagtune_error_types", return_value=(ValueError,)),
            patch.object(
                self.identity_api,
                "discover_gpu_metadata",
                side_effect=[
                    ValueError("device unavailable"),
                    {"platform_key": "test-platform"},
                ],
            ) as discovery,
            patch.object(
                self.cm, "ensure_proposer", return_value=("proposer", "variant")
            ) as loader,
            patch.object(self.cm, "run_proposer", return_value=(self.configs[0], {})),
        ):
            self.assertEqual(self.run_policy()[0], "fallback")
            provisional = self.failure_key()
            self.assertEqual(self.run_policy()[0], "fallback")
            self.assertEqual(discovery.call_count, 1)
            os.environ["USE_FLAGTUNE_COST_MODEL"] = "1"
            self.assertEqual(self.run_policy()[0], self.configs[0])
            canonical = self.failure_key()
            self.assertEqual(canonical[0], "test-platform")
            self.assertNotIn(provisional, self.cm._COST_MODEL_DISABLED_OPS)
            self.assertTrue(self.cm.is_auto_disabled(*canonical))
            os.environ.pop("USE_FLAGTUNE_COST_MODEL")
            self.assertEqual(self.run_policy()[0], "fallback")
            self.assertEqual(loader.call_count, 1)

    def test_auto_failure_fuses_only_affected_variant(self):
        with patch.object(
            self.cm, "load_model", side_effect=FileNotFoundError("missing")
        ) as loader:
            self.assertEqual(self.run_policy()[0], "fallback")
            self.assertEqual(self.run_policy()[0], "fallback")
            self.assertEqual(loader.call_count, 1)
            self.assertTrue(self.cm.is_auto_disabled(*self.failure_key()))
            self.tuner._flagtune_variant = "other_variant"
            self.assertFalse(self.cm.is_auto_disabled(*self.failure_key()))
            self.run_policy()
            self.assertEqual(loader.call_count, 2)
        self.assertEqual(self.fallback.call_count, 3)

    def test_required_missing_model_raises_without_fusing_or_fallback(self):
        os.environ["USE_FLAGTUNE_COST_MODEL"] = "1"
        with patch.object(
            self.cm, "load_model", side_effect=FileNotFoundError("missing")
        ):
            with self.assertRaises(FileNotFoundError):
                self.run_policy()
        self.assertFalse(self.cm._COST_MODEL_DISABLED_OPS)
        self.fallback.assert_not_called()
        self.default.assert_not_called()

    def test_required_ignores_prior_auto_fuse(self):
        self.cm._COST_MODEL_DISABLED_OPS.add(self.failure_key())
        os.environ["USE_FLAGTUNE_COST_MODEL"] = "1"
        with patch.object(
            self.cm, "load_model", side_effect=FileNotFoundError("missing")
        ) as loader:
            with self.assertRaises(FileNotFoundError):
                self.run_policy()
        loader.assert_called_once()
        self.fallback.assert_not_called()

    def test_ordinary_legacy_error_is_not_swallowed(self):
        os.environ["USE_FLAGTUNE"] = "0"
        self.default.side_effect = RuntimeError("device failed")
        with self.assertRaisesRegex(RuntimeError, "device failed"):
            self.run_policy()
        self.fallback.assert_not_called()
        self.assertFalse(self.cm._COST_MODEL_DISABLED_OPS)

    def test_unified_flagtree_boundary_preserves_cancellation(self):
        with patch.object(self.cm, "load_model", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_policy()
        self.fallback.assert_not_called()
        self.assertFalse(self.cm._COST_MODEL_DISABLED_OPS)

    def test_expanded_mode_does_not_load_or_predict(self):
        os.environ["USE_FLAGTUNE_COST_MODEL"] = "0"
        with patch.object(self.cm, "load_model") as loader:
            self.assertEqual(self.run_policy()[0], "default")
        loader.assert_not_called()
        self.fallback.assert_not_called()

    def test_singleton_domain_does_not_load_model(self):
        self.configs = [Config(16)]
        os.environ["USE_FLAGTUNE_COST_MODEL"] = "1"
        with patch.object(self.cm, "load_model") as loader:
            self.assertEqual(self.run_policy()[0], "default")
        loader.assert_not_called()

    def test_original_prune_receives_live_named_arguments(self):
        seen = []

        def prune(configs, arguments, **kwargs):
            seen.append((arguments, kwargs))
            return [c for c in configs if c.kwargs["BLOCK"] == 32]

        self.tuner.early_config_prune = prune
        result = self.cm.runtime_candidates(self.tuner, "op", {"M": 13})
        self.assertEqual(result, [self.configs[1]])
        self.assertIs(result[0], self.configs[1])
        self.assertEqual(seen[0][0]["M"], 13)
        self.assertEqual(seen[0][1], {"M": 13})

    def test_empty_pruned_domain_is_rejected(self):
        self.tuner.early_config_prune = lambda *args, **kwargs: []
        with self.assertRaisesRegex(RuntimeError, "no legal configs"):
            self.cm.runtime_candidates(self.tuner, "op", {})

    def test_proposer_cannot_benchmark_pruned_candidate(self):
        bad = Config(128).all_kwargs()

        def proposer(bench, arguments, domain, meta):
            bench(bad)
            return [bad]

        with self.assertRaisesRegex(ValueError, "pruned runtime domain"):
            self.cm.run_proposer(
                self.tuner, self.bench, self.configs, {}, self.loaded(proposer)
            )
        self.bench.assert_not_called()

    def test_proposer_return_cannot_bypass_domain_check(self):
        with self.assertRaisesRegex(ValueError, "pruned runtime domain"):
            self.cm.run_proposer(
                self.tuner,
                self.bench,
                self.configs,
                {},
                self.loaded(lambda *args: [Config(128).all_kwargs()]),
            )
        self.bench.assert_not_called()

    def test_tle_descriptor_pre_hook_is_reapplied(self):
        hook = Mock()
        self.tuner._flagtune_pre_hook = hook
        config, _ = self.cm.run_proposer(
            self.tuner,
            self.bench,
            self.configs,
            {},
            self.loaded(lambda *args: [self.configs[0].all_kwargs()]),
        )
        self.assertIs(config.pre_hook, hook)
        self.assertIs(self.bench.call_args.args[0].pre_hook, hook)
        self.assertIsNot(config, self.configs[0])
        self.assertIsNone(self.configs[0].pre_hook)
        self.assertEqual(config.all_kwargs(), self.configs[0].all_kwargs())

    def test_original_config_hook_is_preserved_over_converter_or_fallback(self):
        original = Mock()
        self.configs[0].pre_hook = original
        self.tuner._flagtune_pre_hook = Mock()
        converter = Mock(return_value=Config(16, Mock()))
        best, _ = self.cm.run_proposer(
            self.tuner,
            self.bench,
            self.configs,
            {},
            self.loaded(lambda *args: [self.configs[0].all_kwargs()], converter),
        )
        self.assertIs(best, self.configs[0])
        self.assertIs(best.pre_hook, original)
        self.assertIs(self.bench.call_args.args[0], self.configs[0])
        converter.assert_not_called()

    def test_schema_missing_meta_cannot_alias_distinct_candidates(self):
        candidates = [Config(16), Config(16)]
        candidates[0].kwargs.update(VECTOR=False, CHUNKS=1)
        candidates[1].kwargs.update(VECTOR=True, CHUNKS=4)
        proposer = Mock(return_value=[candidates[0].all_kwargs()])
        with self.assertRaisesRegex(ValueError, "aliases distinct runtime candidates"):
            self.cm.run_proposer(
                self.tuner, self.bench, candidates, {}, self.loaded(proposer)
            )
        proposer.assert_not_called()
        self.bench.assert_not_called()

    def test_runtime_candidates_outside_model_space_are_filtered(self):
        calls = []

        def proposer(bench, arguments, domain, meta):
            calls.append(domain)
            return domain

        loaded = self.loaded(proposer)
        loaded[2].param_space.validate = lambda record: record["BLOCK"] == 32
        self.configs[1].kwargs["VECTOR"] = True
        best, _ = self.cm.run_proposer(self.tuner, self.bench, self.configs, {}, loaded)
        self.assertEqual([r["BLOCK"] for r in calls[0]], [32])
        self.assertIs(best, self.configs[1])
        self.assertTrue(best.kwargs["VECTOR"])

    def test_empty_model_domain_is_rejected_before_proposing(self):
        proposer = Mock()
        loaded = self.loaded(proposer)
        loaded[2].param_space.validate = lambda record: False
        with self.assertRaisesRegex(ValueError, "no runtime candidates satisfy"):
            self.cm.run_proposer(self.tuner, self.bench, self.configs, {}, loaded)
        proposer.assert_not_called()
        self.bench.assert_not_called()

    def test_model_converter_cannot_change_launch_parameters(self):
        illegal = Config(16)
        illegal.num_warps = 8
        converter = Mock(return_value=illegal)
        best, _ = self.cm.run_proposer(
            self.tuner,
            self.bench,
            self.configs,
            {},
            self.loaded(lambda *args: [self.configs[0].all_kwargs()], converter),
        )
        self.assertIs(best, self.configs[0])
        self.assertEqual(self.bench.call_args.args[0].num_warps, 4)
        converter.assert_not_called()

    def test_backend_meta_survives_projection_without_conversion(self):
        original = self.configs[0]
        original.kwargs.update(VECTOR=True, CHUNKS=4)
        converter = Mock(return_value=Config(16))
        best, _ = self.cm.run_proposer(
            self.tuner,
            self.bench,
            self.configs,
            {},
            self.loaded(lambda *args: [original.all_kwargs()], converter),
        )
        self.assertIs(best, original)
        self.assertEqual(best.kwargs["VECTOR"], True)
        self.assertEqual(best.kwargs["CHUNKS"], 4)
        converter.assert_not_called()

    def test_identity_distinguishes_dtype_output_variant_and_platform(self):
        identity = lambda vendor="test": self.cm.identity_cache_key(
            self.tuner, self.tuner.nargs, vendor
        )
        keys = {identity()}
        self.tuner.nargs["A"].dtype = "float8_e4m3fn"
        keys.add(identity())
        self.tuner.nargs["A"].dtype = "float8_e5m2"
        keys.add(identity())
        self.tuner.nargs["C"].dtype = "bfloat16"
        keys.add(identity())
        self.tuner._flagtune_variant = "other"
        keys.add(identity())
        keys.add(identity("other-platform"))
        self.tuner._flagtune_op_id = "flaggems_vllm/other_operator"
        keys.add(identity())
        self.assertEqual(len(keys), 7)

    def test_descriptor_uses_base_tensor_dtype(self):
        key = self.cm.identity_cache_key(self.tuner, self.tuner.nargs, "test")
        self.tuner.nargs["A"] = types.SimpleNamespace(
            base=self.tuner.nargs["A"], block_shape=[16, 128]
        )
        descriptor_key = self.cm.identity_cache_key(
            self.tuner, self.tuner.nargs, "test"
        )
        self.assertEqual(descriptor_key, key)
        self.tuner.nargs["A"].base.dtype = "float8_e4m3fn"
        self.assertNotEqual(
            self.cm.identity_cache_key(self.tuner, self.tuner.nargs, "test"), key
        )

    def test_load_model_passes_actual_tensor_dtypes_to_identity(self):
        self.tuner.nargs["A"] = types.SimpleNamespace(base=tensor("float8_e4m3fn"))
        self.tuner.nargs["B"] = tensor("float8_e4m3fn")
        self.tuner.nargs["C"] = tensor("bfloat16")
        key = self.cm.identity_cache_key(self.tuner, self.tuner.nargs, "test")
        with patch.object(
            self.cm, "ensure_proposer", return_value=(object(), object())
        ):
            identity, _, _ = self.cm.load_model(
                self.tuner, key, self.tuner.nargs, infer_dtypes
            )
        self.assertEqual(
            identity.dtype_key,
            "float8_e4m3fn-float8_e4m3fn-float32-float32-bfloat16",
        )
        self.assertEqual(identity.variant, self.tuner._flagtune_variant)

    def test_proposer_cache_separates_package_and_model_versions(self):
        identity = Identity("test-platform", "test/op", "general", "i8-f16")
        loaded = types.SimpleNamespace(
            package_version="1.0.0", model_version="1.0.0", variant=object()
        )
        self.proposer_api.load_model_bundle.return_value = loaded
        self.proposer_api.make_config_proposer.side_effect = lambda *a, **k: object()
        first = self.cm.ensure_proposer(identity)
        self.assertIs(self.cm.ensure_proposer(identity)[0], first[0])
        loaded.package_version = "1.0.1"
        second = self.cm.ensure_proposer(identity)
        self.assertIsNot(first[0], second[0])
        loaded.model_version = "1.1.0"
        third = self.cm.ensure_proposer(identity)
        self.assertIsNot(second[0], third[0])
        self.assertEqual(self.proposer_api.make_config_proposer.call_count, 3)

    def test_nonfinite_or_empty_benchmark_is_rejected(self):
        for values in ([], [float("nan")], [float("inf")], [1.0, float("inf")]):
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                self.cm.benchmark_config(lambda config: values, Config(16))

    def test_tensor_resolver_matches_offline_input_then_output_order(self):
        executor_path = SRC / "flagtune/offline/runtime/executor.py"
        tree = ast.parse(executor_path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_model_dtypes_for_result"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                function,
            ],
            type_ignores=[],
        )
        namespace = {}
        exec(
            compile(ast.fix_missing_locations(module), str(executor_path), "exec"),
            namespace,
        )
        offline_dtypes = namespace["_model_dtypes_for_result"](
            "general",
            ["int8", "int8", "float32", "float32"],
            ["float16"],
            self.tuner.nargs,
            stage="public",
        )
        # Most INT8 JITs put C before the scales, unlike the training contract.
        arguments = {
            name: self.tuner.nargs[name] for name in ("A", "B", "C", "As", "Bs")
        }
        resolver = self.dtype_resolvers.tensor_dtypes("A", "B", "As", "Bs", "C")
        self.assertEqual(list(resolver(arguments)), offline_dtypes)
        self.tuner._flagtune_dtype_resolver = resolver
        self.tuner.arg_names = list(arguments)
        identity_key = self.cm.identity_cache_key(self.tuner, arguments, "test")
        with patch.object(self.cm, "ensure_proposer", return_value=(None, None)):
            identity, _, _ = self.cm.load_model(
                self.tuner, identity_key, arguments, infer_dtypes
            )
        self.assertEqual(identity.dtype_key, "-".join(offline_dtypes))

    def test_tensor_resolver_keeps_real_descriptor_fp8_dtype(self):
        arguments = {
            "A": types.SimpleNamespace(base=tensor("float8_e5m2")),
            "C": tensor("bfloat16"),
        }
        resolver = self.dtype_resolvers.tensor_dtypes("A", "C")
        self.assertEqual(resolver(arguments), ("float8_e5m2", "bfloat16"))

    def test_tensor_resolver_rejects_missing_or_ambiguous_roles(self):
        for names in ((), ("A", "A"), ("",), (1,)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.dtype_resolvers.tensor_dtypes(*names)
        resolver = self.dtype_resolvers.tensor_dtypes("A")
        with self.assertRaisesRegex(ValueError, "missing"):
            resolver({})
        with self.assertRaisesRegex(TypeError, "no tensor dtype"):
            resolver({"A": object()})

    def test_unsupported_fp8_dependency_is_not_coerced_in_required_mode(self):
        class UnsupportedDtypeError(ValueError):
            pass

        self.tuner.nargs["A"] = tensor("float8_e4m3fn")
        os.environ["USE_FLAGTUNE_COST_MODEL"] = "1"
        reject = Mock(side_effect=UnsupportedDtypeError("unsupported tensor dtype"))
        with (
            patch.object(self.identity_api, "make_dtype_key", reject),
            patch.object(
                self.cm, "flagtune_error_types", return_value=(UnsupportedDtypeError,)
            ),
            self.assertRaisesRegex(UnsupportedDtypeError, "unsupported tensor dtype"),
        ):
            self.run_policy()
        self.assertIn("float8_e4m3fn", reject.call_args.args[0])
        self.proposer_api.load_model_bundle.assert_not_called()
        self.fallback.assert_not_called()

    def test_auto_falls_back_when_dependency_rejects_fp8_identity(self):
        class UnsupportedDtypeError(ValueError):
            pass

        self.tuner.nargs["A"] = tensor("float8_e4m3fn")
        with (
            patch.object(
                self.identity_api,
                "make_dtype_key",
                side_effect=UnsupportedDtypeError("unsupported tensor dtype"),
            ),
            patch.object(
                self.cm, "flagtune_error_types", return_value=(UnsupportedDtypeError,)
            ),
        ):
            self.assertEqual(self.run_policy()[0], "fallback")
        self.proposer_api.load_model_bundle.assert_not_called()
        self.assertTrue(self.cm.is_auto_disabled(*self.failure_key()))

    def test_strict_benchmark_flag_restored_on_model_error(self):
        self.tuner._flagtune_strict_benchmark = False
        with (
            patch.object(self.cm, "load_model", return_value=object()),
            patch.object(self.cm, "run_proposer", side_effect=ValueError("broken")),
            self.assertRaisesRegex(ValueError, "broken"),
        ):
            self.run_policy()
        self.assertFalse(self.tuner._flagtune_strict_benchmark)


if __name__ == "__main__":
    unittest.main(verbosity=2)
