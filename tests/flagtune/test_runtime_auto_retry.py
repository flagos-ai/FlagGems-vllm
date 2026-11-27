"""CPU contracts for the actual migrated definitions with device/cache doubles."""

import ast
import hashlib
import importlib.util
import inspect
import logging
import math
import os
import sys
import time
import unittest
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from functools import cached_property
from itertools import starmap
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2] / "src/flaggems_vllm"
spec = importlib.util.spec_from_file_location(
    "native_runtime_mode", ROOT / "runtime/flagtune.py"
)
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)


class Tensor:
    def __init__(self, dtype):
        self.dtype = dtype
        self.device = "test:0"


class TensorDescriptor:
    def __init__(self, base):
        self.base = base


class Config:
    def __init__(self, kwargs, num_warps=4, num_stages=1, pre_hook=None):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_ctas = 1
        self.pre_hook = pre_hook

    def all_kwargs(self):
        return dict(
            self.kwargs,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
            num_ctas=1,
        )

    def __repr__(self):
        return repr(self.all_kwargs())


class JIT:
    def __init__(self):
        self.__name__ = "shared_kernel"
        self.cache_key = "same_jit"
        self.arg_names = ["A", "B", "C", "M"]
        self.launches = []

    def run(self, *args, **kwargs):
        self.launches.append((args, kwargs))
        return "kernel"


class Autotuner:
    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        *pos,
        pre_hook=None,
        post_hook=None,
        prune_configs_by=None,
        do_bench=None,
        **kw,
    ):
        self.fn = self.base_fn = fn
        self.arg_names = arg_names
        self.configs = configs
        self.pre_hook = pre_hook or (lambda args, **kw: None)
        self.post_hook = post_hook
        self.do_bench = do_bench
        self.early_config_prune = (prune_configs_by or {}).get("early_config_prune")
        self.measured = []

    def prune_configs(self, kwargs):
        return (
            self.early_config_prune(self.configs, self.nargs, **kwargs)
            if self.early_config_prune
            else self.configs
        )

    def _bench(self, *args, config, **kwargs):
        self.measured.append(config)
        return [config.kwargs["BLOCK"]] * 3


class BenchmarkMode(str, Enum):
    EVENT = "event"
    REPLAY = "replay"


class Protocol:
    def __init__(self, mode, warmup, rep, retries):
        self.requested_mode = mode
        self.values = (mode, warmup, rep, retries)

    def cache_key(self):
        return self.values


def resolve_benchmarker(mode, *, warmup_ms, measurement_ms, n_retries):
    return SimpleNamespace(
        protocol=Protocol(mode, warmup_ms, measurement_ms, n_retries),
        benchmark=lambda f, quantiles: f(),
    )


# Preserve process imports: these doubles are active only inside each test.
_boundary_names = (
    "triton",
    "triton.testing",
    "torch",
    "triton.tools.tensor_descriptor",
)
_saved_modules = {name: sys.modules.get(name) for name in _boundary_names}

triton = ModuleType("triton")
triton.Config = Config
triton.runtime = SimpleNamespace(Autotuner=Autotuner, JITFunction=JIT)
triton.testing = SimpleNamespace(do_bench=lambda *a, **kw: None)
testing = ModuleType("triton.testing")
testing.do_bench_cudagraph = lambda *a, **kw: None
sys.modules["triton"] = triton
sys.modules["triton.testing"] = testing
sys.modules["torch"] = SimpleNamespace(Tensor=Tensor)
sys.modules["triton.tools.tensor_descriptor"] = SimpleNamespace(
    TensorDescriptor=TensorDescriptor
)

selected = {
    "LibTuner",
    "use_run_mode",
    "use_benchmark_protocol",
    "benchmark_config",
    "LibTunerRunMode",
    "_infer_tensor_dtypes",
    "_supports_flagtune_cost_model",
    "_flagtune_legacy_fallback",
    "flagtune_policy",
    "default_strategy",
    "log2_strategy",
    "align32_strategy",
    "default_policy",
    "libtuner",
}
tree = ast.parse((ROOT / "utils/libentry.py").read_text())
helper_trees = [
    ast.parse((ROOT / "flagtune/inference/cost_model.py").read_text()),
    ast.parse((ROOT / "flagtune/offline/runtime/tuner.py").read_text()),
]
definitions = [node for helper in helper_trees for node in helper.body] + tree.body
filtered = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *[
            n
            for n in definitions
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in selected
        ],
    ],
    type_ignores=[],
)
ns = dict(
    globals(),
    hashlib=hashlib,
    inspect=inspect,
    math=math,
    contextmanager=contextmanager,
    cached_property=cached_property,
    starmap=starmap,
    libcache=defaultdict(dict),
    major_version=3,
    minor_version=6,
    abstractmethod=lambda x: x,
    _HAS_FLAGTREE_FLAGTUNE=True,
    logger=logging.getLogger("test"),
    BENCHMARK_CACHE_SCHEMA_VERSION=2,
    DEFAULT_BENCHMARK_RETRIES=10,
)
runtime.device = SimpleNamespace(vendor_name="testvendor")
cmns = dict(
    runtime=runtime,
    time=time,
    _COST_MODEL_IDENTITIES={},
    _COST_MODEL_DISABLED_OPS=set(),
)
cmtree = ast.parse((ROOT / "flagtune/inference/cost_model.py").read_text())
cmnames = {
    "identity_cache_key",
    "failure_key",
    "is_auto_disabled",
}
cmast = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *[
            n
            for n in cmtree.body
            if isinstance(n, ast.FunctionDef) and n.name in cmnames
        ],
    ],
    type_ignores=[],
)
exec(
    compile(
        ast.fix_missing_locations(cmast),
        "<actual AUTO identity/failure helpers>",
        "exec",
    ),
    cmns,
)
ns["cost_model"] = SimpleNamespace(**cmns)
ns.update({name: cmns[name] for name in cmnames})
exec(
    compile(
        ast.fix_missing_locations(filtered),
        "<actual migrated libentry definitions>",
        "exec",
    ),
    ns,
)
_fake_modules = {name: sys.modules[name] for name in _boundary_names}
for _name, _module in _saved_modules.items():
    if _module is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module

Tuner = ns["LibTuner"]
RunMode = ns["LibTunerRunMode"]


class PortTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, _fake_modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        runtime._include_ops = None
        ns["libcache"].clear()
        cmns["_COST_MODEL_DISABLED_OPS"].clear()
        cmns["_COST_MODEL_IDENTITIES"].clear()

    def tearDown(self):
        self.env.stop()

    def make(self, variant="general", rep=20, graph=True, **kw):
        return ns["libtuner"](
            configs=[Config({"BLOCK": 16}), Config({"BLOCK": 32})],
            key=["M"],
            strategy="align32",
            rep=rep,
            use_cuda_graph=graph,
            flagtune_op_name="w8a8_block_matmul",
            flagtune_op_id="flaggems_vllm/w8a8_block_matmul",
            flagtune_variant=variant,
            **kw,
        )(JIT())

    def test_modes_and_registry(self):
        self.assertEqual(
            runtime.resolve_tuning_mode("x", supports_cost_model=True),
            runtime.TuningMode.COST_MODEL,
        )
        self.assertEqual(
            runtime.resolve_cost_model_intent(supports_cost_model=True),
            runtime.CostModelIntent.AUTO,
        )
        os.environ["USE_FLAGTUNE_COST_MODEL"] = "1"
        self.assertEqual(
            runtime.resolve_cost_model_intent(supports_cost_model=True),
            runtime.CostModelIntent.REQUIRED,
        )
        os.environ["USE_FLAGTUNE_COST_MODEL"] = "0"
        self.assertEqual(
            runtime.resolve_tuning_mode("x", supports_cost_model=True),
            runtime.TuningMode.EXPANDED,
        )
        os.environ["USE_FLAGTUNE"] = "0"
        self.assertEqual(
            runtime.resolve_tuning_mode("x", supports_cost_model=True),
            runtime.TuningMode.DEFAULT,
        )
        self.assertIn("w8a8_block_matmul", runtime.SUPPORTED_FLAGTUNE_OPS)
        self.assertEqual(
            runtime.DEFAULT_FLAGTUNE_INCLUDE, runtime.get_default_flagtune_include()
        )
        os.environ["USE_FLAGTUNE"] = "bad"
        with self.assertRaises(ValueError):
            runtime.resolve_tuning_mode("x")

    def test_old_and_native_policy_signatures(self):
        @Tuner.register_policy("old_four")
        def old(bench, configs, args, kwargs):
            return "four", {}

        @Tuner.register_policy("native_five")
        def new(self, bench, configs, args, kwargs):
            return self, {}

        a = self.make(policy="old_four")
        b = self.make(policy="native_five")
        self.assertEqual(a.policy(None, [], (), {})[0], "four")
        self.assertIs(b.policy(None, [], (), {})[0], b)

    def test_dtype_roles_and_descriptor(self):
        tuner = self.make()
        a = {"A": Tensor("int8"), "B": Tensor("int8"), "C": Tensor("float16"), "M": 33}
        b = dict(a, A=Tensor("float8_e4m3fn"), B=Tensor("float8_e4m3fn"))
        c = dict(b, C=Tensor("bfloat16"))
        self.assertNotEqual(tuner.get_key(a), tuner.get_key(b))
        self.assertNotEqual(tuner.get_key(b), tuner.get_key(c))
        descriptor = dict(b, A=TensorDescriptor(b["A"]), B=TensorDescriptor(b["B"]))
        self.assertEqual(tuner.get_key(b), tuner.get_key(descriptor))
        self.assertEqual(
            ns["_infer_tensor_dtypes"](descriptor.values()),
            ("float8_e4m3fn", "float8_e4m3fn", "float16"),
        )

    def test_normalized_selection_exact_measurement(self):
        tuner = self.make()
        self.assertEqual(tuner.get_key({"M": 33}), tuner.get_key({"M": 34}))
        self.assertNotEqual(
            tuner.get_benchmark_key({"M": 33}), tuner.get_benchmark_key({"M": 34})
        )

    def test_variant_and_native_protocol_cache_separation(self):
        a = self.make()
        b = self.make("swap")
        c = self.make(rep=100)
        d = self.make(graph=False)
        e = self.make(rep=100, graph=False)
        self.assertNotEqual(a.config_table_name, b.config_table_name)
        self.assertNotEqual(a.benchmark_table_name, b.benchmark_table_name)
        self.assertNotEqual(a.config_table_name, c.config_table_name)
        self.assertNotEqual(d.config_table_name, e.config_table_name)
        self.assertNotEqual(
            a.get_benchmark_key({"M": 32}), c.get_benchmark_key({"M": 32})
        )
        self.assertNotEqual(
            a.get_benchmark_key({"M": 32}), d.get_benchmark_key({"M": 32})
        )

    def test_scoped_protocol_restores_even_after_error(self):
        tuner = self.make()
        before = (
            tuner.config_table_name,
            tuner.cache,
            tuner.do_bench,
            tuner._benchmark_protocol,
        )
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with ns["use_benchmark_protocol"](tuner, "replay", 5, 100, 3):
                self.assertNotEqual(tuner._benchmark_protocol, before[-1])
                raise RuntimeError("sentinel")
        self.assertEqual(
            (
                tuner.config_table_name,
                tuner.cache,
                tuner.do_bench,
                tuner._benchmark_protocol,
            ),
            before,
        )

    def test_scoped_run_mode_cache_and_hook(self):
        events = []
        tuner = self.make()
        tuner.configs[0].pre_hook = lambda args: events.append(args["BLOCK"])
        args = (Tensor("int8"), Tensor("int8"), Tensor("float16"), 33)
        with ns["use_run_mode"](tuner, RunMode.EXHAUSTIVE_COLLECTION):
            tuner.run(*args)
        self.assertEqual(tuner._run_mode, RunMode.NORMAL)
        self.assertEqual(tuner.benchmark_success_count, 2)
        self.assertEqual(len(tuner.cache), 0)
        self.assertEqual(events, [16])
        with ns["use_run_mode"](tuner, RunMode.FORCE_POLICY):
            tuner.run(*args)
        self.assertEqual(tuner.benchmark_success_count, 0)
        self.assertEqual(tuner.benchmark_cache_hit_count, 2)
        self.assertEqual(events, [16, 16])

    def test_fixed_config_inherits_native_graph_or_event(self):
        seen = []
        original = ns["resolve_benchmarker"]

        def record(mode, **kwargs):
            seen.append(mode)
            return original(mode, **kwargs)

        ns["resolve_benchmarker"] = record
        try:
            for graph in (True, False):
                tuner = self.make(graph=graph)
                ns["benchmark_config"](
                    tuner, tuner.configs[0], warmup=5, rep=20, args=(), meta={}
                )
        finally:
            ns["resolve_benchmarker"] = original
        self.assertEqual(seen, [BenchmarkMode.REPLAY, BenchmarkMode.EVENT])

    def test_source_compatibility_guards(self):
        txt = (ROOT / "utils/libentry.py").read_text()
        for marker in (
            'device == "cpu"',
            "HIPBackend.get_tensor_specialization(arg)",
            "launch_pre_hooks",
            "config.pre_hook(full_nargs)",
            "self.pre_hook(full_nargs, reset_only=True)",
        ):
            self.assertIn(marker, txt)
        self.assertNotIn("except _TritonOutOfResources", txt)

    def test_direct_tuner_reuses_cached_fallback_after_elapsed_time(self):
        tuner = self.make()
        args = (Tensor("int8"), Tensor("int8"), Tensor("float16"), 33)
        tuner._flagtune_configs_for_mode = lambda *a: (tuner.configs, "default")
        arguments = dict(zip(tuner.arg_names, args))
        key = cmns["failure_key"](tuner, arguments)
        cmns["_COST_MODEL_DISABLED_OPS"].add(key)
        calls = []
        tuner.policy = lambda bench, configs, args, kwargs: (
            calls.append(True) or configs[0],
            {},
        )
        with patch.object(time, "monotonic", return_value=100):
            tuner.apply_flagtune(arguments)
            tuner.run(*args)
            self.assertEqual(len(calls), 1)
            tuner.run(*args)
            self.assertEqual(len(calls), 1)
        with patch.object(time, "monotonic", return_value=160):
            tuner.run(*args)
            self.assertEqual(len(calls), 1)

    def test_entry_failure_invalidates_compiled_and_CPU_cache_once(self):
        entry_class = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LibEntry"
        )
        method = next(
            n
            for n in entry_class.body
            if isinstance(n, ast.FunctionDef) and n.name == "_apply_flagtune"
        )
        scope = dict(ns)
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                "<actual LibEntry selection>",
                "exec",
            ),
            scope,
        )
        selected = []
        tuner = SimpleNamespace(
            _flagtune_op_id="test/op",
            _flagtune_variant="general",
            apply_flagtune=lambda arguments: selected.append(arguments) or False,
            fn=JIT(),
        )
        args = (Tensor("int8"), Tensor("int8"), Tensor("float16"), 33)
        tuner.arg_names = tuner.fn.arg_names
        entry = SimpleNamespace(
            fn=tuner,
            arg_names=tuner.fn.arg_names,
            kernel_cache=({"cached": 1},),
            _cpu_cache={"cached": 1},
        )
        key = cmns["failure_key"](tuner, dict(zip(entry.arg_names, args)))
        cmns["_COST_MODEL_DISABLED_OPS"].add(key)
        # A newly failed identity clears previously compiled model launches.
        with patch.object(time, "monotonic", return_value=160):
            scope["_apply_flagtune"](entry, args, {})
        self.assertFalse(entry.kernel_cache[0])
        self.assertFalse(entry._cpu_cache)
        self.assertIs(selected[0]["A"], args[0])
        # A stable failure state does not repeatedly discard compiled kernels.
        entry.kernel_cache[0]["cached"] = 1
        scope["_apply_flagtune"](entry, args, {})
        self.assertTrue(entry.kernel_cache[0])
        # A failure in an unrelated identity must not flush healthy dtype caches.
        args[0].dtype = "float8_e4m3fn"
        scope["_apply_flagtune"](entry, args, {})
        entry.kernel_cache[0]["healthy"] = 1
        args[0].dtype = "float8_e5m2"
        scope["_apply_flagtune"](entry, args, {})
        self.assertIn("healthy", entry.kernel_cache[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
