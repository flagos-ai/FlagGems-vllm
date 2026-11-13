# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Standard no-Graph benchmark for the preallocated, three-argument API.

The output is fully overwritten and never read, so timed replays are
idempotent. Poison/reset and oracle checks happen outside timing for both
providers. Allocation is outside timing for both providers.
"""

import importlib
import math
from pathlib import Path

import pytest
import torch

import flaggems_vllm as gems
from baseline import fused_kimi_k3_mla_kv_concat as workload
from benchmark import conftest as report
from benchmark.base import Benchmark
from benchmark.consts import BenchMode
from tests.test_fused_kimi_k3_mla_kv_concat import (
    assert_inputs_unchanged,
    assert_output,
    reference_output,
    snapshot_inputs,
)


def _native_slot(*args, **kwargs):
    raise RuntimeError("native slot must be intercepted before timing")


class KVConcatBenchmark(Benchmark):
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_DTYPES = workload.DTYPES
    DEFAULT_SHAPE_DESC = "T/H/layout; fixed 128/64/192; preallocated output"

    def set_shapes(self, shape_file_path=None):
        # Baseline grid is the default single source. Explicit shape files use
        # the normal framework parser with [T, H, layout] configuration rows.
        if (
            shape_file_path
            and shape_file_path != self.DEFAULT_SHAPE_FILES
            and Path(shape_file_path).resolve()
            != Path(__file__).with_name("core_shapes.yaml").resolve()
        ):
            super().set_shapes(shape_file_path)
            if any(len(shape) != 3 for shape in self.shapes):
                raise ValueError("KV concat shape files require [T, H, layout] rows")
            self.shapes = [
                dict(zip(("T", "H", "layout"), shape)) for shape in self.shapes
            ]
        else:
            self.shapes = workload.grid()

    def init_user_config(self):
        super().init_user_config()
        if report.Config.mode != BenchMode.KERNEL:
            raise ValueError("KV concat requires --mode kernel (no Graph)")
        self.native_op = None
        self.native_reason = "native timing disabled by --metrics"
        if "latency_base" in self.to_bench_metrics:
            self.native_op = workload.native()
            self.native_reason = workload.NATIVE_UNAVAILABLE_REASON
        self.row_native_reason = None

    def run(self):
        # The parent matches unique tensor shapes even if layout differs. Filter
        # offline records per configuration before its normal lookup/timing path.
        self.offline_data = report.Config.base_data
        self.offline_rows = []  # Keep identities alive for the parent's index cache.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(report.Config, "base_data", self.offline_data)
            if report.Config.user_desired_metrics == ["latency"]:
                self.offline_data = None
                patch.setattr(report.Config, "base_data", None)
            super().run()

    def matching_offline_data(self, binding, dtype):
        """Reject shape-only matches across strides or allocating API variants."""
        data = self.offline_data
        if data is None:
            return None, "offline baseline not enabled"
        expected = workload.config(binding, dtype)
        entry = data.get(self.op_name, {})
        matches = {}
        for key, per_dtype in entry.get("shapes", {}).items():
            record = per_dtype.get(str(dtype), {})
            config = record.get("config", {})
            if (
                config.get("inputs") == expected["inputs"]
                and config.get("outputs") == expected["outputs"]
                and config.get("mutation") == expected["mutation"]
                and config.get("layout") == expected["layout"]
                and config.get("graph") is False
            ):
                matches[key] = {str(dtype): record}
        reason = None
        if len(matches) != 1:
            reason = f"offline baseline requires one exact input/layout/mutation match; got {len(matches)}"
            matches = {}
        filtered = {**data, self.op_name: {**entry, "shapes": matches}}
        self.offline_rows.append(filtered)
        return filtered, reason

    def tuning_metadata(self, args):
        """Read the actual cached selection outside timing, including PE reuse."""
        n, p, out = args
        if not out.numel():
            return {"path": "empty", "pe_logical_load_bytes": 0}
        ascend = gems.vendor_name == "ascend"
        module_name = (
            "flaggems_vllm.runtime.backend._ascend.ops.fused_kimi_k3_mla_kv_concat"
            if ascend
            else "flaggems_vllm.ops.fused_kimi_k3_mla_kv_concat"
        )
        module = importlib.import_module(module_name)
        kernel = module._kv_concat_ascend_kernel if ascend else module._kv_concat_kernel
        tuner = kernel.fn
        tune_args = {
            "N": n.view(torch.int16),
            "P": p.view(torch.int16),
            "OUT": out.view(torch.int16),
            "T": n.shape[0],
            "H": n.shape[1],
            "NS0": n.stride(0),
            "NS1": n.stride(1),
            "PS0": p.stride(0),
        }
        if not ascend:
            del tune_args["T"]
            tune_args["OUTPUT_ELEMENTS"] = out.numel()
        config = tuner.cache[tuner.get_key(tune_args)]
        heads_per_program = config.kwargs.get("HEADS_PER_PROGRAM", 1)
        pe_loads = 2 * n.shape[0] * 64 * math.ceil(n.shape[1] / heads_per_program)
        return {
            "path": (
                "ascend_token_head"
                if ascend
                else ("head_row" if config.kwargs["ROW_LAYOUT"] else "linear_output")
            ),
            "META": config.kwargs,
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
            "pe_logical_load_bytes": pe_loads,
            "kernel_logical_traffic_bytes": 2 * n.numel() + pe_loads + 2 * out.numel(),
        }

    def get_input_iter(self, dtype):
        misses, rows = [], []
        for binding in self.shapes:
            args, kwargs = workload.build_inputs(binding, dtype, self.device)
            self.binding = binding
            expected = reference_output(*args)
            saved = snapshot_inputs(args)
            pointer = args[2].data_ptr()
            # Candidate/reference failures propagate; never classify them as
            # native unavailability. Two poisons check complete overwrites.
            for poison in (0x5A5A, -12345):
                args[2].view(torch.int16).fill_(poison)
                result = self.gems_op(*args, **kwargs)
                assert_output(result, args[2], expected)
                assert args[2].data_ptr() == pointer
                assert_inputs_unchanged(saved)
            self.expected = expected
            self.saved = saved
            self.row_native_reason = self.native_reason or workload.unsupported_reason(
                binding, dtype
            )
            offline_reason = None
            if hasattr(self, "offline_data"):
                report.Config.base_data, offline_reason = self.matching_offline_data(
                    binding, dtype
                )
            yield (*args, kwargs)
            shape = self.record_shapes(*args, **kwargs)
            rows.append(
                {
                    "shape": shape,
                    "config": workload.config(binding, dtype),
                    "tuning": self.tuning_metadata(args),
                    "native_reason": self.row_native_reason,
                    "offline_baseline_reason": offline_reason,
                    "candidate_timing_reason": (
                        "empty output: no device launch to time"
                        if not args[2].numel()
                        else None
                    ),
                }
            )
            if self.row_native_reason:
                misses.append({"shape": shape, "reason": self.row_native_reason})
        report.update_result(
            self.op_name,
            {
                "dtype": str(dtype),
                "workload_details": rows,
                "native_provenance": workload.NATIVE_PROVENANCE,
            },
        )
        if misses:
            report.update_result(
                self.op_name,
                {
                    "dtype": str(dtype),
                    "native_baseline_skip_reason": "; ".join(
                        dict.fromkeys(row["reason"] for row in misses)
                    ),
                    "native_baseline_misses": misses,
                },
            )

    def record_shapes(self, *args, **kwargs):
        # Shapes alone collide for the three layout variants at T=128,H=3.
        return [
            super().record_shapes(*args, **kwargs),
            {
                "layout": self.binding["layout"],
                "strides": [list(x.stride()) for x in args],
            },
        ]

    def get_latency(self, op, *args, **kwargs):
        # A profiler cannot assign device latency to an empty, no-launch API.
        if not args[2].numel():
            return None
        if op is not self.torch_op:
            latency = super().get_latency(op, *args, **kwargs)
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"invalid candidate latency: {latency!r}")
            return latency
        if self.native_op is None or self.row_native_reason:
            return None
        phase = "native qualification"
        try:
            pointer = args[2].data_ptr()
            for poison in (0x5A5A, -12345):
                args[2].view(torch.int16).fill_(poison)
                result = self.native_op(*args, **kwargs)
                assert_output(result, args[2], self.expected)
                assert args[2].data_ptr() == pointer
                assert_inputs_unchanged(self.saved)
            phase = "native timing"
            latency = super().get_latency(self.native_op, *args, **kwargs)
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"invalid native latency: {latency!r}")
        except Exception as error:
            self.row_native_reason = f"{phase}: {error!r}"
            # A rejected native must not leave mutated inputs for the candidate.
            for base, before in self.saved:
                base.view(torch.int16).copy_(before.to(base.device))
            return None
        return latency


@pytest.mark.fused_kimi_k3_mla_kv_concat
def test_fused_kimi_k3_mla_kv_concat():
    KVConcatBenchmark(
        op_name=workload.OP_NAME,
        torch_op=_native_slot,
        gems_op=gems.fused_kimi_k3_mla_kv_concat,
        is_inplace=True,
    ).run()
