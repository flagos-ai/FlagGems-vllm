# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Same-device ported vLLM Triton baseline versus the public fused_q API."""

import inspect
from pathlib import Path

import flaggems_vllm as gems
import pytest
import torch

from baseline import fused_q as workload
from tests.test_fused_q import assert_benchmark_case

from . import base, consts
from .conftest import Config


class FusedQBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPE_DESC = "N, H, NoPE, RoPE, index_heads, index_head_dim"

    def set_shapes(self, shape_file_path=None):
        if shape_file_path and Path(shape_file_path).resolve() not in (
            Path(self.DEFAULT_SHAPE_FILES).resolve(),
            Path(__file__).with_name("core_shapes.yaml").resolve(),
        ):
            super().set_shapes(shape_file_path)
        else:
            # Collector and benchmark share this sole default workload matrix.
            self.shapes = [tuple(workload.key_shape(b)) for b in workload.grid()]
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def init_user_config(self):
        super().init_user_config()
        if self.mode not in (consts.BenchMode.KERNEL, consts.BenchMode.OPERATOR):
            raise ValueError(
                "fused_q performance requires non-Graph kernel/operator mode"
            )

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            self.binding = workload.binding_from_shape(shape)
            if Config.base_data is not None:
                entry = Config.base_data.get(workload.OP_NAME, {}).get("shapes", {})
                record = entry.get(str(workload.key_shape(self.binding)), {}).get(
                    str(dtype)
                )
                if record is not None:
                    expected = workload.config(self.binding, dtype)
                    actual = record.get("config", {})
                    if any(actual.get(k) != expected[k] for k in ("inputs", "outputs")):
                        raise ValueError(
                            "--base-data matched a shape but has a different input/output contract"
                        )
            args, kwargs = workload.build_inputs(self.binding, dtype, gems.device)
            yield (*args, kwargs)

    def record_shapes(self, *args, **kwargs):
        # The first torch.Size is an explicit semantic key, not a tensor shape.
        # This supports the six-dimensional collector key without changing the
        # project's baseline matcher or inventing an input tensor.
        return dict(
            key_shape=torch.Size(workload.key_shape(self.binding)),
            inputs=super().record_shapes(*args, **kwargs),
            config=workload.config(self.binding, kwargs["q_pe"].dtype),
            native_kind=workload.NATIVE_KIND,
            native_source_sha256=workload.NATIVE_SOURCE_SHA256,
            candidate_file=inspect.getfile(gems.fused_q),
            graph=False,
        )

    def get_latency(self, op, *args, **kwargs):
        # References, input cloning and precision checks stay outside timing.
        before = {
            k: v.clone() for k, v in kwargs.items() if isinstance(v, torch.Tensor)
        }
        assert_benchmark_case(kwargs, op(*args, **kwargs))
        for key, value in before.items():
            assert torch.equal(kwargs[key], value), "Input modified: " + key
        return super().get_latency(op, *args, **kwargs)


@pytest.mark.fused_q
def test_fused_q_benchmark():
    if gems.vendor_name != "nvidia" or not torch.cuda.is_available():
        pytest.skip("The fused_q benchmark targets NVIDIA CUDA")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (props.name, props.major, props.minor, props.multi_processor_count) != (
        "NVIDIA H20",
        9,
        0,
        78,
    ):
        pytest.skip("The frozen performance claim is limited to full H20")
    bench = FusedQBenchmark(
        op_name="fused_q",
        torch_op=workload.native(),
        gems_op=gems.fused_q,
        dtypes=workload.DTYPES,
        is_inplace=workload.IS_INPLACE,
    )
    bench.run()
