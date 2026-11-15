"""Offline regression tests for the strict gumbel NVIDIA-baseline adapter.

Every fixture in this file is synthetic.  It exercises only
``benchmark/gumbel_baseline.py`` through its public helper and requires neither
torch nor an accelerator.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmark" / "gumbel_baseline.py"
BENCHMARK_TEST_PATH = (
    Path(__file__).resolve().parents[1] / "benchmark" / "test_gumbel_sample.py"
)
SPEC = importlib.util.spec_from_file_location("gumbel_baseline_offline", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load baseline helper from {MODULE_PATH}")
BASELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASELINE
SPEC.loader.exec_module(BASELINE)


CASE_KEY = "tokens=16_vocab=256_temp=0.8_stride=0_seed=4660"
DTYPE = "torch.float32"
TIMING_PROTOCOL = {
    "timer": "cuda-events",
    "warmup": 10,
    "batches": 3,
    "iterations": 100,
    "graph": False,
    "reset": "replay-fixed-stateless-rng",
    "unit": "ms",
}
CONFIG = {
    "logits": {"shape": [16, 256], "stride": [0, 1], "dtype": DTYPE},
    "temperature": {"scalar": 0.8, "apply_temperature": True},
    "rng": {"seed": 4660, "position_mode": "fixed", "draft": False},
    "source": {"mode": "request", "row_mapping": "identity", "cache": False},
    "timing_protocol": TIMING_PROTOCOL,
}

SYNTHETIC_CASE_KEYS = tuple(
    f"n{tokens}_v{vocab}_t{temperature}"
    for tokens, vocab in (
        (1, 127),
        (1, 1024),
        (4, 4097),
        (16, 32000),
        (64, 32768),
        (1, 128256),
        (16, 128256),
        (128, 151936),
    )
    for temperature in ("0", "08")
)


def _load_record_function():
    """Extract benchmark _record without importing its torch/triton module."""
    tree = ast.parse(BENCHMARK_TEST_PATH.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_record"
    )
    updates = []

    def update_result(op, data):
        updates.append((op, data))

    conftest = types.SimpleNamespace(
        Config=types.SimpleNamespace(
            bench_level=types.SimpleNamespace(value="comprehensive")
        ),
        update_result=update_result,
    )
    namespace = {
        "__name__": "_gumbel_record_output_contract",
        "_DETAIL": None,
        "_NATIVE_SKIP_REASONS": [],
        "conftest": conftest,
        "json": __import__("json"),
        "os": os,
    }
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), str(BENCHMARK_TEST_PATH), "exec"
        ),
        namespace,
    )
    return namespace["_record"], updates


def write_synthetic_standard_output(jsonl_path):
    """Run the actual _record body over 16 synthetic latency-only rows."""
    record, updates = _load_record_function()
    rows = []
    prior_output = os.environ.get("GUMBEL_BENCH_OUTPUT")
    os.environ["GUMBEL_BENCH_OUTPUT"] = str(jsonl_path)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for index, case_key in enumerate(SYNTHETIC_CASE_KEYS):
                has_compared = index % 2 == 0
                candidate_ms = (index + 1) / 1000
                compared = 0.8 if has_compared else None
                normalization = {
                    "compared_speedup": compared,
                    "base_raw_speedup": 0.4,
                    "base_latency_ms": candidate_ms * 0.4,
                    "base_scaling_factor": 0.5 if has_compared else None,
                    "base_shape_key": case_key,
                    "base_data_source": {
                        "kind": "synthetic",
                        "fixture": "output-01",
                    },
                    "base_miss_reason": (
                        None if has_compared else "synthetic target factor unavailable"
                    ),
                }
                metric = {
                    "shape_detail": case_key,
                    "latency": candidate_ms,
                    "latency_base": None,
                    "speedup": None,
                    "timing_protocol": "synthetic-no-graph-v1",
                    "workload_config": {"synthetic_case": case_key},
                    "native_identity": {"status": "unavailable", "synthetic": True},
                    **normalization,
                }
                row = {
                    "synthetic": True,
                    "case_key": case_key,
                    "dtype": "float32",
                    "candidate": {
                        "latency_ms": candidate_ms,
                        "timing_protocol": "synthetic-no-graph-v1",
                    },
                    "native_ratio": None,
                    "normalized_ratio": compared,
                    "normalization": normalization,
                }
                rows.append(row)
                record(row, metric, "synthetic native unavailable")
    finally:
        if prior_output is None:
            os.environ.pop("GUMBEL_BENCH_OUTPUT", None)
        else:
            os.environ["GUMBEL_BENCH_OUTPUT"] = prior_output

    if len(updates) != 1:
        raise AssertionError(
            f"_record registered {len(updates)} dtype groups, expected 1"
        )
    return updates[0][1], rows


def synthetic_base_data(*, bottleneck="cuda"):
    """Return a fully qualified, synthetic H800 baseline payload."""
    return {
        "_environment": {"hardware": {"gpu": {"name": "NVIDIA H800"}}},
        "_scaling_factors": {
            "reference_chip": "H800",
            "chips": [
                {
                    "vendor": "ascend",
                    "chip": "Ascend 910B4",
                    "measured": {
                        "bandwidth_gbps": 0.5,
                        "vector": {"fp32": 0.5},
                        "tensor": {"fp32": 0.25},
                    },
                },
                {
                    "vendor": "nvidia",
                    "chip": "H20",
                    "measured": {
                        "bandwidth_gbps": 0.5,
                        "vector": {"fp32": 0.5},
                        "tensor": {"fp32": 0.5},
                    },
                },
            ],
        },
        "gumbel_sample": {
            "shapes": {
                CASE_KEY: {
                    DTYPE: {
                        "status": "ok",
                        "config": copy.deepcopy(CONFIG),
                        "timing_protocol": copy.deepcopy(TIMING_PROTOCOL),
                        "latency_ms": 10.0,
                        "bottle_neck_unit": bottleneck,
                    }
                }
            }
        },
    }


def normalize(
    data=None,
    *,
    case_key=CASE_KEY,
    dtype=DTYPE,
    config=None,
    candidate_ms=25.0,
    vendor="ascend",
    chip="Huawei Ascend-910 B4",
    base_data_source="synthetic offline fixture",
    timing_protocol=None,
):
    return BASELINE.normalize_case(
        synthetic_base_data() if data is None else data,
        case_key,
        dtype,
        copy.deepcopy(CONFIG) if config is None else config,
        candidate_ms,
        vendor,
        chip,
        base_data_source=base_data_source,
        timing_protocol=timing_protocol,
    )


class GumbelBaselineOfflineTests(unittest.TestCase):
    def test_capacity_and_interface_suffixes_are_not_discarded(self):
        canonical = BASELINE.canonical_chip
        self.assertNotEqual(
            canonical("NVIDIA A100 40GB PCIe"), canonical("A100 80GB SXM")
        )
        self.assertNotEqual(
            canonical("NVIDIA A100 40GB SXM"), canonical("A100 80GB SXM")
        )
        self.assertNotEqual(canonical("NVIDIA H800 80GB"), canonical("H800"))

    def assert_missing(self, result):
        self.assertIsNone(result["compared_speedup"])
        self.assertIsNotNone(result["base_miss_reason"])

    def test_valid_vector_factor_returns_normalized_and_raw_ratios(self):
        result = normalize()

        self.assertAlmostEqual(result["base_raw_speedup"], 0.4)
        self.assertAlmostEqual(result["compared_speedup"], 0.8)
        self.assertEqual(result["base_latency_ms"], 10.0)
        self.assertEqual(result["base_scaling_factor"], 0.5)
        self.assertEqual(result["base_shape_key"], CASE_KEY)
        self.assertEqual(result["base_data_source"], "synthetic offline fixture")
        self.assertIsNone(result["base_miss_reason"])
        self.assertEqual(result["factor_resource"], "vector")

    def test_actual_record_emits_one_dtype_group_with_all_synthetic_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            jsonl_path = Path(directory) / "synthetic-gumbel.jsonl"
            detail, expected_rows = write_synthetic_standard_output(jsonl_path)

            self.assertEqual(detail["op_name"], "gumbel_sample")
            self.assertEqual(detail["dtype"], "torch.float32")
            self.assertEqual(detail["mode"], "kernel")
            self.assertEqual(detail["level"], "comprehensive")
            self.assertEqual(
                detail["native_baseline_skip_reason"], "synthetic native unavailable"
            )
            self.assertEqual(len(detail["result"]), 16)
            self.assertEqual(
                {metric["shape_detail"] for metric in detail["result"]},
                set(SYNTHETIC_CASE_KEYS),
            )
            self.assertEqual(len(SYNTHETIC_CASE_KEYS), 16)

            output_rows = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(output_rows, expected_rows)
            metrics = {metric["shape_detail"]: metric for metric in detail["result"]}
            for index, case_key in enumerate(SYNTHETIC_CASE_KEYS):
                with self.subTest(case_key=case_key):
                    metric = metrics[case_key]
                    has_compared = index % 2 == 0
                    self.assertEqual(metric["latency"], (index + 1) / 1000)
                    self.assertIsNone(metric["latency_base"])
                    self.assertIsNone(metric["speedup"])
                    self.assertEqual(
                        metric["compared_speedup"], 0.8 if has_compared else None
                    )
                    self.assertEqual(metric["base_raw_speedup"], 0.4)
                    self.assertAlmostEqual(
                        metric["base_latency_ms"], (index + 1) / 2500
                    )
                    self.assertEqual(metric["base_shape_key"], case_key)
                    self.assertEqual(
                        metric["base_data_source"],
                        {"kind": "synthetic", "fixture": "output-01"},
                    )
                    self.assertEqual(
                        metric["base_miss_reason"],
                        None if has_compared else "synthetic target factor unavailable",
                    )

    def test_tensor_and_memory_records_select_their_own_resources(self):
        tensor = normalize(synthetic_base_data(bottleneck="tensor"))
        memory = normalize(synthetic_base_data(bottleneck="mem"))

        self.assertAlmostEqual(tensor["base_scaling_factor"], 0.25)
        self.assertAlmostEqual(tensor["compared_speedup"], 1.6)
        self.assertEqual(tensor["factor_resource"], "tensor")
        self.assertAlmostEqual(memory["base_scaling_factor"], 0.5)
        self.assertAlmostEqual(memory["compared_speedup"], 0.8)
        self.assertEqual(memory["factor_resource"], "bandwidth_gbps")

    def test_exact_case_dtype_and_config_are_required(self):
        wrong_case = BASELINE.normalize_case(
            synthetic_base_data(),
            CASE_KEY + "_different",
            DTYPE,
            copy.deepcopy(CONFIG),
            25.0,
            "ascend",
            "Ascend 910B4",
        )
        wrong_dtype = BASELINE.normalize_case(
            synthetic_base_data(),
            CASE_KEY,
            "torch.float16",
            copy.deepcopy(CONFIG),
            25.0,
            "ascend",
            "Ascend 910B4",
        )
        self.assert_missing(wrong_case)
        self.assert_missing(wrong_dtype)
        self.assertIsNone(wrong_case["base_raw_speedup"])
        self.assertIsNone(wrong_dtype["base_raw_speedup"])

        for field, changed in (
            ("temperature", {"scalar": 0.0, "apply_temperature": True}),
            ("logits", {"shape": [16, 256], "stride": [256, 1], "dtype": DTYPE}),
            ("rng", {"seed": 4661, "position_mode": "fixed", "draft": False}),
        ):
            with self.subTest(field=field):
                config = copy.deepcopy(CONFIG)
                config[field] = changed
                result = BASELINE.normalize_case(
                    synthetic_base_data(),
                    CASE_KEY,
                    DTYPE,
                    config,
                    25.0,
                    "ascend",
                    "Ascend 910B4",
                )
                self.assert_missing(result)
                self.assertIsNone(result["base_raw_speedup"])

    def test_op_name_is_not_fallback_matched(self):
        data = synthetic_base_data()
        data["some_other_operator"] = data.pop("gumbel_sample")

        result = normalize(data)

        self.assert_missing(result)
        self.assertIsNone(result["base_raw_speedup"])

    def test_protocol_must_match_config_record_and_measurement(self):
        data = synthetic_base_data()
        data["gumbel_sample"]["shapes"][CASE_KEY][DTYPE]["timing_protocol"] = {
            **TIMING_PROTOCOL,
            "iterations": 101,
        }
        record_mismatch = normalize(data)
        caller_mismatch = normalize(timing_protocol={**TIMING_PROTOCOL, "graph": True})

        self.assert_missing(record_mismatch)
        self.assert_missing(caller_mismatch)
        self.assertIsNone(record_mismatch["base_raw_speedup"])
        self.assertIsNone(caller_mismatch["base_raw_speedup"])

    def test_missing_or_invalid_factor_leaves_only_qualified_raw_ratio(self):
        for label, factor in (
            ("missing", None),
            ("zero", 0.0),
            ("negative", -0.5),
            ("nan", math.nan),
            ("infinite", math.inf),
        ):
            with self.subTest(factor=label):
                data = synthetic_base_data()
                data["_scaling_factors"]["chips"][0]["measured"]["vector"][
                    "fp32"
                ] = factor
                result = normalize(data)
                self.assert_missing(result)
                self.assertAlmostEqual(result["base_raw_speedup"], 0.4)
                self.assertEqual(result["base_latency_ms"], 10.0)
                self.assertIsNone(result["base_scaling_factor"])

    def test_nominal_only_or_ambiguous_factor_is_rejected(self):
        nominal_only = synthetic_base_data()
        target = nominal_only["_scaling_factors"]["chips"][0]
        target["nominal"] = target.pop("measured")
        nominal_result = normalize(nominal_only)

        duplicate = synthetic_base_data()
        duplicate["_scaling_factors"]["chips"].append(
            copy.deepcopy(duplicate["_scaling_factors"]["chips"][0])
        )
        ambiguous_result = normalize(duplicate)

        self.assert_missing(nominal_result)
        self.assert_missing(ambiguous_result)
        self.assertAlmostEqual(nominal_result["base_raw_speedup"], 0.4)
        self.assertAlmostEqual(ambiguous_result["base_raw_speedup"], 0.4)

    def test_exact_sku_matching_never_borrows_another_model(self):
        for chip in ("Ascend 910B1", "Huawei Ascend 910C 64GB"):
            with self.subTest(chip=chip):
                result = normalize(chip=chip)
                self.assert_missing(result)
                self.assertAlmostEqual(result["base_raw_speedup"], 0.4)

        h20z = BASELINE.normalize_case(
            synthetic_base_data(),
            CASE_KEY,
            DTYPE,
            copy.deepcopy(CONFIG),
            25.0,
            "nvidia",
            "NVIDIA H20Z 96GB",
        )
        wrong_vendor = normalize(vendor="nvidia", chip="Ascend 910B4")
        self.assert_missing(h20z)
        self.assert_missing(wrong_vendor)
        self.assertAlmostEqual(h20z["base_raw_speedup"], 0.4)
        self.assertAlmostEqual(wrong_vendor["base_raw_speedup"], 0.4)

    def test_reference_chip_must_identify_the_gpu_that_produced_the_baseline(self):
        data = synthetic_base_data()
        data["_environment"]["hardware"]["gpu"]["name"] = "NVIDIA H100 80GB"

        result = normalize(data)

        self.assert_missing(result)
        self.assertAlmostEqual(result["base_raw_speedup"], 0.4)

    def test_invalid_candidate_latency_is_an_error(self):
        for candidate_ms in (0.0, -1.0, math.nan, math.inf, True, "25"):
            with self.subTest(candidate_ms=candidate_ms):
                with self.assertRaises(ValueError):
                    BASELINE.normalize_case(
                        synthetic_base_data(),
                        CASE_KEY,
                        DTYPE,
                        copy.deepcopy(CONFIG),
                        candidate_ms,
                        "ascend",
                        "Ascend 910B4",
                    )

    def test_invalid_baseline_latency_and_bottleneck_do_not_normalize(self):
        for label, value in (
            ("zero", 0.0),
            ("negative", -1.0),
            ("nan", math.nan),
            ("infinite", math.inf),
        ):
            with self.subTest(latency=label):
                data = synthetic_base_data()
                data["gumbel_sample"]["shapes"][CASE_KEY][DTYPE]["latency_ms"] = value
                result = normalize(data)
                self.assert_missing(result)
                self.assertIsNone(result["base_raw_speedup"])

        data = synthetic_base_data(bottleneck="unknown")
        result = normalize(data)
        self.assert_missing(result)
        self.assertAlmostEqual(result["base_raw_speedup"], 0.4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
