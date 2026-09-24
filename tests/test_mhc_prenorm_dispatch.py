# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""The loop-exit completion path requires the entire reviewed configuration."""

import importlib
from contextlib import nullcontext

import pytest
import torch


class Metadata:
    def __init__(self, shape, dtype=torch.bfloat16):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device("cuda", 0)

    def stride(self):
        return (self.shape[1], 1)


@pytest.mark.parametrize(
    "tokens,config,expected",
    [
        (64, (64, 128, 64, 4, 3), True),
        (96, (64, 128, 32, 4, 3), False),
        (128, (64, 128, 32, 4, 3), False),
        (96, (64, 128, 64, 4, 3), False),
        (64, (32, 128, 64, 4, 3), False),
        (64, (64, 64, 64, 4, 3), False),
        (64, (64, 128, 32, 4, 3), False),
        (64, (64, 128, 64, 8, 3), False),
        (64, (64, 128, 64, 4, 1), False),
        (64, (64, 128, 64, 4, 2), False),
        (64, (64, 128, 64, 4, 4), False),
    ],
)
def test_two_step_completion_requires_reviewed_config(
    monkeypatch, tokens, config, expected
):
    module = importlib.import_module("flaggems_vllm.ops.mhc.mhc_prenorm")
    monkeypatch.setattr(
        module, "_PRENORM_CONFIGS", {tokens: module._PrenormConfig(*config)}
    )
    monkeypatch.setattr(module.torch_device_fn, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        module, "_get_packed_fn", lambda fn: (Metadata((32768, 32)), None)
    )
    monkeypatch.setattr(
        torch, "empty", lambda shape, dtype, device: Metadata(shape, dtype)
    )
    monkeypatch.setattr(module, "TensorDescriptor", lambda *args, **kwargs: object())
    launches = []

    class Capture:
        def __getitem__(self, grid):
            def record(*args, **kwargs):
                launches.append((grid, kwargs))

            return record

    monkeypatch.setattr(module, "_mhc_prenorm_gemm_kernel", Capture())
    partial, square = module._mhc_prenorm_gemm_impl(
        Metadata((tokens, 16384)), Metadata((24, 16384), torch.float32)
    )
    assert partial.shape == (config[2], tokens, 32)
    assert square.shape == (config[2], tokens)
    assert len(launches) == 1
    _, kwargs = launches[0]
    assert kwargs["TWO_STEP_COMPLETION"] is expected
    assert kwargs["LAUNCH_PDL"] is False and kwargs["launch_pdl"] is False
    assert kwargs["num_warps"] == config[3]
    assert kwargs["num_stages"] == config[4]
