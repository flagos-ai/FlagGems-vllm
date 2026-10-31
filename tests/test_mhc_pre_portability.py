# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""mHC pre candidate policies use shape metadata without device execution."""

import importlib
import importlib.util
import sys
from types import SimpleNamespace

import pytest

pre = importlib.import_module("flaggems_vllm.ops.mhc.mhc_pre")

_HC4_DEFAULTS = [
    (256, 4, 1),
    (256, 8, 1),
    (512, 4, 1),
    (512, 8, 1),
    (1024, 4, 1),
    (1024, 8, 1),
    (1024, 16, 1),
    (256, 4, 2),
    (512, 4, 2),
    (512, 8, 2),
]
_GENERIC_DEFAULTS = [
    (256, 4, 1),
    (256, 8, 1),
    (512, 4, 1),
    (512, 8, 1),
    (1024, 4, 1),
    (1024, 8, 1),
]


def _config_values(configs):
    assert all(config.num_ctas == 1 for config in configs)
    return [
        (config.kwargs["BLOCK_H"], config.num_warps, config.num_stages)
        for config in configs
    ]


def _source_import(monkeypatch, vendor, autotune):
    monkeypatch.setattr(pre.runtime, "device", SimpleNamespace(vendor_name=vendor))
    monkeypatch.setattr(pre.triton, "autotune", autotune)
    name = "flaggems_vllm.ops.mhc._pre_shape_import"
    spec = importlib.util.spec_from_file_location(name, pre.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "hidden,expected_blocks",
    [
        (1, [1, 1]),
        (63, [64, 64]),
        (64, [64, 64]),
        (65, [128, 128]),
        (127, [128, 128]),
        (128, [128, 128]),
        (129, [128, 128, 256, 256, 256]),
        (256, [256, 256, 256]),
        (512, [256, 256, 512, 512, 256, 512, 512]),
        (1024, [item[0] for item in _HC4_DEFAULTS]),
        (8192, [item[0] for item in _HC4_DEFAULTS]),
    ],
)
def test_kunlun_hc4_prunes_through_real_autotuner_abi(
    monkeypatch, hidden, expected_blocks
):
    native_autotune = pre.triton.autotune
    registrations = []

    def capture(**kwargs):
        registrations.append(kwargs)
        return native_autotune(**kwargs)

    module = _source_import(monkeypatch, "kunlunxin", capture)
    tuner = module.mhc_pre_fused_kernel_hc_mult_4
    assert registrations[0]["prune_configs_by"] == {
        "early_config_prune": module._mhc_pre_hc4_config_prune
    }
    assert registrations[0]["do_bench"] is module._mhc_pre_do_bench
    assert "prune_configs_by" not in registrations[1]
    original = _config_values(tuner.configs)
    assert original == [(128, 4, 1), (128, 8, 1)] + _HC4_DEFAULTS

    # Use the actual positional JIT argument names and Autotuner.prune_configs,
    # which supplies its named nargs mapping plus runtime keyword arguments.
    positional = [None] * len(tuner.arg_names)
    positional[tuner.arg_names.index("hidden_size")] = hidden
    positional[tuner.arg_names.index("hc_hidden_size")] = 4 * hidden
    positional[tuner.arg_names.index("num_tokens")] = 3
    tuner.nargs = dict(zip(tuner.arg_names, positional))
    calls = []
    prune = tuner.early_config_prune

    def observe(configs, named_args, **kwargs):
        assert configs is tuner.configs and named_args is tuner.nargs
        assert named_args["hidden_size"] == hidden
        assert kwargs == {"grid": (3,), "warmup": False}
        calls.append(True)
        return prune(configs, named_args, **kwargs)

    monkeypatch.setattr(tuner, "early_config_prune", observe)
    selected = tuner.prune_configs({"grid": (3,), "warmup": False})
    assert calls == [True]
    assert [config.kwargs["BLOCK_H"] for config in selected] == expected_blocks
    assert {config.num_warps for config in selected} >= {4, 8}
    cap = pre.triton.next_power_of_2(hidden)
    assert all(
        0 < config.kwargs["BLOCK_H"] <= cap
        and config.kwargs["BLOCK_H"] & (config.kwargs["BLOCK_H"] - 1) == 0
        for config in selected
    )
    assert _config_values(tuner.configs) == original
    if hidden >= 256:
        assert all(config.kwargs["BLOCK_H"] >= 256 for config in selected)
    if hidden >= 1024:
        assert _config_values(selected) == _HC4_DEFAULTS
        assert all(a is b for a, b in zip(selected, tuner.configs[2:]))


@pytest.mark.parametrize("vendor", ["nvidia", "metax", "mthreads", "ascend", "thead"])
def test_other_vendors_keep_exact_defaults_and_legacy_constructor(monkeypatch, vendor):
    native_autotune = pre.triton.autotune
    registrations = []

    # Reject even optional/None pruning or benchmark keywords at the actual
    # source-import boundary, then delegate to the installed real Triton.
    def legacy_autotune(configs, key):
        registrations.append((configs, key))
        return native_autotune(configs=configs, key=key)

    module = _source_import(monkeypatch, vendor, legacy_autotune)
    assert len(registrations) == 2
    assert _config_values(registrations[0][0]) == _HC4_DEFAULTS
    assert _config_values(registrations[1][0]) == _GENERIC_DEFAULTS
    assert registrations[0][1] == ["hidden_size", "num_tokens_bucket"]
    assert registrations[1][1] == ["hidden_size", "num_tokens_bucket", "HC"]
    for kernel in (
        module.mhc_pre_fused_kernel_hc_mult_4,
        module.mhc_pre_generic_kernel,
    ):
        assert kernel.early_config_prune is None
        assert getattr(kernel, "_do_bench", None) is None
