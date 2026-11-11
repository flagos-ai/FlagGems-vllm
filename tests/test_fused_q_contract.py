# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Collector contract checks; these do not replace device numerical tests."""

import inspect

import pytest
import torch

from baseline import fused_q as workload

pytestmark = pytest.mark.fused_q


def test_fused_q_baseline_contract():
    assert workload.OP_NAME == "fused_q"
    assert workload.DTYPES == [torch.bfloat16]
    assert workload.IS_INPLACE is False
    assert workload.native() is workload.fused_q
    bindings = workload.grid()
    keys = [tuple(workload.key_shape(b)) for b in bindings]
    assert len(keys) == len(set(keys)) == 21
    for binding in bindings:
        cfg = workload.config(binding, torch.bfloat16)
        key = workload.key_shape(binding)
        assert cfg["dims"]["descriptor"] == key
        assert key[2:] == [512, 64, 32, 128]
        assert cfg["inputs"]["q_scale"]["value"] == 0.5
        assert cfg["outputs"]["mqa_q"]["shape"] == [key[0], key[1], 576]


def test_fused_q_baseline_inputs_cpu():
    # One bounded allocation verifies input order/metadata without any CUDA.
    binding = workload.grid()[0]
    args, values = workload.build_inputs(binding, torch.bfloat16, "cpu")
    inspect.signature(workload.native()).bind(*args, **values)
    cfg = workload.config(binding, torch.bfloat16)
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            assert list(value.shape) == cfg["inputs"][key]["shape"]
            assert str(value.dtype) == cfg["inputs"][key]["dtype"]
        else:
            assert value == cfg["inputs"][key]["scalar"]


def test_fused_q_baseline_rejects_unbenchmarked_dtype():
    with pytest.raises(ValueError, match="BF16"):
        workload.build_inputs(workload.grid()[0], torch.float16, "cpu")


@pytest.mark.parametrize(
    "shape", [[1], [0, 16, 512, 64, 32, 128], [True, 16, 512, 64, 32, 128]]
)
def test_fused_q_baseline_rejects_invalid_shape(shape):
    with pytest.raises(ValueError):
        workload.binding_from_shape(shape)
