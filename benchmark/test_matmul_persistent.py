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

"""Non-Graph matmul benchmark for NVIDIA and Ascend."""

import inspect
import math
import warnings
from pathlib import Path

import pytest
import torch

import flaggems_vllm as gems
from baseline import matmul_persistent as workload
from tests.test_matmul_persistent import assert_case
from tests.test_matmul_persistent_ascend import (
    assert_ascend_output,
    build_ascend_inputs,
)

from . import base, consts
from .conftest import Config
from .consts import lookup_base_record_ex


def _native_reference():
    try:
        from vllm.model_executor.determinism.batch_invariant import (
            matmul_batch_invariant as native,
        )
    except Exception as exc:
        warnings.warn(
            f"vLLM matmul_batch_invariant unavailable: {type(exc).__name__}: {exc}; "
            "using a matching --base-data record or the Python baseline."
        )
        return None
    if not callable(native) or native is gems.matmul_persistent:
        warnings.warn(
            "An independent vLLM native reference is unavailable; "
            "using a matching --base-data record or the Python baseline."
        )
        return None
    return native


class MatmulPersistentBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPE_DESC = "a.ndim, *a.shape, b.ndim, *b.shape"

    def set_shapes(self, shape_file_path=None):
        if shape_file_path and Path(shape_file_path).resolve() not in (
            Path(self.DEFAULT_SHAPE_FILES).resolve(),
            Path(__file__).with_name("core_shapes.yaml").resolve(),
        ):
            super().set_shapes(shape_file_path)
        else:
            self.shapes = [tuple(workload.key_shape(b)) for b in workload.grid()]
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def init_user_config(self):
        super().init_user_config()
        self._native_op = self.torch_op
        if self.mode not in (consts.BenchMode.KERNEL, consts.BenchMode.OPERATOR):
            raise ValueError("Use non-Graph kernel/operator timing")

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            self.binding = workload.binding_from_shape(shape)
            if gems.vendor_name == "ascend":
                yield (*build_ascend_inputs(self.binding, dtype, gems.device), {})
            else:
                args, kwargs = workload.build_inputs(self.binding, dtype, gems.device)
                yield (*args, kwargs)

    def _local_reference_kind(self):
        if gems.vendor_name == "ascend":
            return "torch_npu_matmul"
        return workload.NATIVE_KIND

    def _python_baseline(self):
        if gems.vendor_name == "ascend":
            return torch.matmul
        if not hasattr(self, "_baseline_op"):
            try:
                reference = workload.native()
                if not callable(reference) or reference is self.gems_op:
                    raise RuntimeError("No independent Python baseline callable")
            except Exception as exc:
                warnings.warn(
                    f"Python baseline unavailable: {type(exc).__name__}: {exc}"
                )
                reference = None
            self._baseline_op = reference
        return self._baseline_op

    def _use_reference(self, kind, reference):
        self.torch_op = reference
        self._shape_detail["comparison_source"] = kind
        self._shape_detail["native_kind"] = (
            kind if reference is not None else "unavailable"
        )
        self._shape_detail["native_source_sha256"] = (
            workload.NATIVE_SOURCE_SHA256
            if reference is not None and kind == workload.NATIVE_KIND
            else None
        )

    def record_shapes(self, *args, **kwargs):
        dtype = args[0].dtype
        # Keep the semantic collector key first for the shared JSON matcher.
        detail = dict(
            key_shape=torch.Size(workload.key_shape(self.binding)),
            inputs=super().record_shapes(*args, **kwargs),
            config=workload.config(self.binding, dtype),
            candidate_file=inspect.getfile(gems.matmul_persistent),
            graph=False,
        )
        self._shape_detail = detail
        self._has_json_reference = False
        if Config.base_data is not None:
            record, reason = lookup_base_record_ex(
                Config.base_data, self.op_name, dtype, detail, self.base_variant
            )
            if record is not None and record.get("latency_ms") is not None:
                actual = record.get("config", {})
                if any(
                    actual.get(k) != detail["config"][k] for k in ("inputs", "outputs")
                ):
                    raise ValueError("--base-data input/output contract differs")
                latency = record["latency_ms"]
                if (
                    isinstance(latency, bool)
                    or not isinstance(latency, (int, float))
                    or not math.isfinite(latency)
                    or latency <= 0
                ):
                    raise ValueError(
                        "--base-data latency_ms must be finite and positive"
                    )
                self._has_json_reference = True
            else:
                detail["json_baseline_miss_reason"] = reason or "latency_ms missing"
        # Reconsider the native reference for every shape and dtype.
        if self._native_op is not None:
            self._use_reference("installed_vllm", self._native_op)
        elif self._has_json_reference:
            self._use_reference("json_baseline", None)
        else:
            reference = self._python_baseline()
            self._use_reference(
                (
                    self._local_reference_kind()
                    if reference is not None
                    else "unavailable"
                ),
                reference,
            )
        return detail

    def get_latency(self, op, *args, **kwargs):
        # Candidate correctness errors must never trigger reference fallback.
        is_ascend = gems.vendor_name == "ascend"
        if is_ascend:
            a, b = args
            expected = a.cpu().double() @ b.cpu().double()
        else:
            expected = assert_case(*args, **kwargs)
        is_reference = op is self.torch_op
        before = [
            (value, value.clone())
            for value in (*args, *kwargs.values())
            if isinstance(value, torch.Tensor)
        ]
        try:
            # Correctness checks and input snapshots stay outside timing.
            result = op(*args, **kwargs)
            if is_ascend:
                assert_ascend_output(result, expected, args[0])
            else:
                torch.testing.assert_close(
                    result, expected, rtol=0, atol=0, equal_nan=True
                )
            assert all(
                torch.equal(value, saved) for value, saved in before
            ), "Callable modified inputs"
            latency = super().get_latency(op, *args, **kwargs)
            assert all(
                torch.equal(value, saved) for value, saved in before
            ), "Timed callable modified inputs"
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"Invalid latency: {latency!r}")
            return latency
        except Exception as exc:
            if not is_reference:
                raise
            for value, saved in before:
                value.copy_(saved)
            kind = self._shape_detail["native_kind"]
            reason = f"{type(exc).__name__}: {exc}"
            self._shape_detail.setdefault("reference_failures", {})[kind] = reason
            warnings.warn(f"{kind} reference failed: {reason}")
            # A matching JSON record takes priority over timing the Python port.
            if not self._has_json_reference and op is self._native_op:
                reference = self._python_baseline()
                if reference is not None and reference is not op:
                    self._use_reference(self._local_reference_kind(), reference)
                    return self.get_latency(reference, *args, **kwargs)
            self._use_reference(
                "json_baseline" if self._has_json_reference else "unavailable", None
            )
            if is_ascend and not self._has_json_reference:
                raise RuntimeError(
                    "No usable Ascend timing reference for "
                    f"{self._shape_detail['key_shape']} ({args[0].dtype}): {reason}"
                ) from exc
            return None


@pytest.mark.matmul_persistent
def test_matmul_persistent_benchmark():
    if gems.vendor_name == "ascend":
        assert torch.npu.is_available(), "Ascend NPU is unavailable"
    elif gems.vendor_name == "nvidia":
        assert torch.cuda.is_available(), "NVIDIA CUDA is unavailable"
    else:
        pytest.skip("NVIDIA CUDA or Ascend NPU required")
    bench = MatmulPersistentBenchmark(
        op_name=workload.OP_NAME,
        torch_op=_native_reference(),
        gems_op=gems.matmul_persistent,
        dtypes=workload.DTYPES,
        is_inplace=False,
    )
    bench.run()
