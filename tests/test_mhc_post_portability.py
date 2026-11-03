# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""mHC host routing with real tensor metadata and optional compiler APIs."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

post = importlib.import_module("flaggems_vllm.ops.mhc.mhc_post")


class _Launch:
    def __init__(self, call):
        self.call = call

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.call(grid, *args, **kwargs)

        return launch


def _inputs(hc):
    return (
        torch.empty(2, 128, device="cuda:0", dtype=torch.bfloat16),
        torch.empty(2, hc, 128, device="cuda:0", dtype=torch.bfloat16),
        torch.empty(2, hc, 1, device="cuda:0", dtype=torch.float32),
        torch.empty(2, hc, hc, device="cuda:0", dtype=torch.float32),
    )


@pytest.mark.parametrize(
    "vendor,hc,expected",
    [("nvidia", 4, "hc4"), ("nvidia", 2, "generic"), ("thead", 4, "ppu")],
)
@pytest.mark.parametrize("api", ["no_knobs", "no_autotuning", "no_adjust"])
def test_mhc_post_routes_without_optional_flagtree_api(
    monkeypatch, vendor, hc, expected, api
):
    mode = FakeTensorMode()
    monkeypatch.setattr(post.runtime, "device", SimpleNamespace(vendor_name=vendor))
    if api == "no_knobs":
        monkeypatch.delattr(post.triton, "knobs", raising=False)
    elif api == "no_autotuning":
        monkeypatch.setattr(post.triton, "knobs", SimpleNamespace(), raising=False)
    else:
        monkeypatch.setattr(
            post.triton,
            "knobs",
            SimpleNamespace(autotuning=SimpleNamespace()),
            raising=False,
        )
    calls = []

    def record(name):
        def call(grid, a, b, c, d, out, sn, sj, si, **kwargs):
            assert grid({"BLOCK_H": 64}) == ((2, 2) if hc == 4 else (2, hc, 2))
            assert (sn, sj, si) == (hc * hc, hc, 1)
            assert kwargs == ({"H": 128} if hc == 4 else {"H": 128, "HC": hc})
            calls.append(name)

        return call

    monkeypatch.setattr(post, "mhc_post_kernel_hc_mult_4", _Launch(record("hc4")))
    monkeypatch.setattr(post, "mhc_post_kernel_generic", _Launch(record("generic")))
    ppu = ModuleType("flaggems_vllm.runtime.backend._thead.ops.mhc_post")

    def ppu_call(a, b, c, d, out, sn, sj, si, hidden):
        assert (sn, sj, si, hidden) == (16, 4, 1, 128)
        calls.append("ppu")

    ppu.mhc_post_hc4_tokloop = ppu_call
    monkeypatch.setitem(sys.modules, ppu.__name__, ppu)
    with mode:
        args = _inputs(hc)
        output = post.mhc_post(*args)
        assert output.shape == args[1].shape and output.dtype == torch.bfloat16
        assert output.device == args[1].device
    assert calls == [expected]
    if api == "no_knobs":
        assert not hasattr(post.triton, "knobs")
    elif api == "no_autotuning":
        assert not hasattr(post.triton.knobs, "autotuning")
    else:
        assert not hasattr(post.triton.knobs.autotuning, "adjust_block_size")


@pytest.mark.parametrize("replace_owner", [False, True])
def test_mhc_post_restores_original_optional_owner_on_launch_failure(
    monkeypatch, replace_owner
):
    monkeypatch.setattr(post.runtime, "device", SimpleNamespace(vendor_name="nvidia"))
    original = object()
    owner = SimpleNamespace(adjust_block_size=original)
    monkeypatch.setattr(
        post.triton, "knobs", SimpleNamespace(autotuning=owner), raising=False
    )
    replacement = SimpleNamespace(
        autotuning=SimpleNamespace(adjust_block_size=object())
    )
    replacement_value = replacement.autotuning.adjust_block_size
    error = RuntimeError("device launch failed")
    calls = []

    def fail(*args, **kwargs):
        assert owner.adjust_block_size is False
        calls.append(True)
        if replace_owner:
            post.triton.knobs = replacement
        raise error

    monkeypatch.setattr(post, "mhc_post_kernel_hc_mult_4", _Launch(fail))
    with FakeTensorMode(), pytest.raises(RuntimeError) as caught:
        post.mhc_post(*_inputs(4))
    assert caught.value is error
    assert calls == [True]
    assert owner.adjust_block_size is original
    assert replacement.autotuning.adjust_block_size is replacement_value


def test_mhc_post_restores_optional_setting_after_success(monkeypatch):
    monkeypatch.setattr(post.runtime, "device", SimpleNamespace(vendor_name="nvidia"))
    original = object()
    owner = SimpleNamespace(adjust_block_size=original)
    monkeypatch.setattr(
        post.triton, "knobs", SimpleNamespace(autotuning=owner), raising=False
    )
    calls = []

    def launch(*args, **kwargs):
        assert owner.adjust_block_size is False
        calls.append(True)

    monkeypatch.setattr(post, "mhc_post_kernel_hc_mult_4", _Launch(launch))
    with FakeTensorMode():
        post.mhc_post(*_inputs(4))
    assert calls == [True]
    assert owner.adjust_block_size is original
