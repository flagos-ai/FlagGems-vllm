# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Same-run vLLM port versus the public matmul entry; no Graph timing."""

import inspect
from pathlib import Path

import pytest
import torch

import flaggems_vllm as gems
from baseline import matmul_persistent as workload
from tests.test_matmul_persistent import assert_case

from . import base, consts
from .conftest import Config


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
        if self.mode not in (consts.BenchMode.KERNEL, consts.BenchMode.OPERATOR):
            raise ValueError("Use non-Graph kernel/operator timing")

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            self.binding = workload.binding_from_shape(shape)
            if Config.base_data is not None:
                record = (
                    Config.base_data.get(workload.OP_NAME, {})
                    .get("shapes", {})
                    .get(str(workload.key_shape(self.binding)), {})
                    .get(str(dtype))
                )
                if record is not None:
                    expected = workload.config(self.binding, dtype)
                    actual = record.get("config", {})
                    if any(actual.get(k) != expected[k] for k in ("inputs", "outputs")):
                        raise ValueError("--base-data input/output contract differs")
            args, kwargs = workload.build_inputs(self.binding, dtype, gems.device)
            yield (*args, kwargs)

    def record_shapes(self, *args, **kwargs):
        return dict(
            key_shape=torch.Size(workload.key_shape(self.binding)),
            inputs=super().record_shapes(*args, **kwargs),
            config=workload.config(self.binding, args[0].dtype),
            native_kind=workload.NATIVE_KIND,
            candidate_file=inspect.getfile(gems.matmul_persistent),
            graph=False,
        )

    def get_latency(self, op, *args, **kwargs):
        # assert_case uses both public and baseline entries before either timed call.
        assert_case(*args, **kwargs)
        return super().get_latency(op, *args, **kwargs)


@pytest.mark.matmul_persistent
def test_matmul_persistent_benchmark():
    if gems.vendor_name != "nvidia" or not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if "H20" not in props.name.split() or (props.major, props.minor) != (9, 0):
        pytest.skip("Current performance claim is H20 only")
    bench = MatmulPersistentBenchmark(
        op_name=workload.OP_NAME,
        torch_op=workload.native(),
        gems_op=gems.matmul_persistent,
        dtypes=workload.DTYPES,
        is_inplace=False,
    )
    bench.run()
