# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Independent CPU integer oracle; no native execution in accuracy tests."""

import importlib

import pytest
import torch

import flaggems_vllm as gems
from baseline import fused_kimi_k3_mla_kv_concat as workload

pytestmark = pytest.mark.fused_kimi_k3_mla_kv_concat


def reference_output(k_nope, k_pe, k_out):
    n = k_nope.cpu().view(torch.int16)
    p = k_pe.cpu().view(torch.int16)
    return torch.cat((n, p[:, None, :].expand(n.shape[0], n.shape[1], 64)), dim=-1)


def assert_output(result, out, expected):
    assert result is None
    assert out.is_contiguous()
    assert torch.equal(out.cpu().view(torch.int16), expected)


def snapshot_inputs(args):
    saved = []
    for source in args[:2]:
        base = source
        while base._base is not None:
            base = base._base
        saved.append((base, base.cpu().view(torch.int16).clone()))
    return saved


def assert_inputs_unchanged(saved):
    for base, before in saved:
        assert torch.equal(base.cpu().view(torch.int16), before)


@pytest.mark.parametrize("dtype", workload.DTYPES)
@pytest.mark.parametrize("binding", workload.grid(), ids=workload.key_shape)
@pytest.mark.fused_kimi_k3_mla_kv_concat
def test_fused_kimi_k3_mla_kv_concat(dtype, binding):
    args, kwargs = workload.build_inputs(binding, dtype, gems.device)
    expected = reference_output(*args)
    saved = snapshot_inputs(args)
    out = args[2]
    pointer = out.data_ptr()
    for poison in (0x5A5A, -12345):
        out.view(torch.int16).fill_(poison)
        result = gems.fused_kimi_k3_mla_kv_concat(*args, **kwargs)
        assert_output(result, out, expected)
        assert out.data_ptr() == pointer
        assert_inputs_unchanged(saved)


@pytest.mark.parametrize("dtype", workload.DTYPES)
def test_empty_does_not_launch(dtype, monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.fused_kimi_k3_mla_kv_concat")
    args, _ = workload.build_inputs(
        {"T": 0, "H": 3, "layout": "gather"}, dtype, gems.device
    )

    class NoLaunch:
        def __getitem__(self, grid):
            pytest.fail("empty operation attempted a kernel launch")

    monkeypatch.setattr(module, "_kv_concat_kernel", NoLaunch())
    assert gems.fused_kimi_k3_mla_kv_concat(*args) is None
    assert args[2].numel() == 0


@pytest.mark.parametrize(
    "broken",
    [
        "nope_rank",
        "pe_rank",
        "out_rank",
        "nope_width",
        "pe_width",
        "out_width",
        "tokens",
        "heads",
        "dtype",
        "out_dtype",
        "unsupported_dtype",
        "nope_last_stride",
        "pe_last_stride",
        "out_contiguous",
        "device",
        "alias_nope",
        "alias_pe",
        "partial_alias",
    ],
)
def test_invalid_inputs_do_not_mutate(broken):
    args, _ = workload.build_inputs(
        {"T": 7, "H": 3, "layout": "packed"}, torch.float16, gems.device
    )
    n, p, out = args
    if broken == "nope_rank":
        n = n[0]
    elif broken == "pe_rank":
        p = p[:, None, :]
    elif broken == "out_rank":
        out = out[0]
    elif broken == "nope_width":
        n = n[..., :127]
    elif broken == "pe_width":
        p = p[:, :63]
    elif broken == "out_width":
        out = out[..., :191]
    elif broken == "tokens":
        p = p[:6]
    elif broken == "heads":
        out = out[:, :2]
    elif broken == "dtype":
        p = p.bfloat16()
    elif broken == "out_dtype":
        out = out.bfloat16()
    elif broken == "unsupported_dtype":
        n, p, out = (x.float() for x in args)
    elif broken == "nope_last_stride":
        n = torch.empty((7, 3, 256), device=gems.device, dtype=n.dtype)[..., ::2]
    elif broken == "pe_last_stride":
        p = torch.empty((7, 128), device=gems.device, dtype=p.dtype)[:, ::2]
    elif broken == "out_contiguous":
        out = torch.empty((7, 3, 384), device=gems.device, dtype=n.dtype)[..., ::2]
    elif broken == "device":
        p = p.cpu()
    elif broken == "alias_nope":
        n = out[..., :128]
    elif broken == "alias_pe":
        p = out[:, 0, 128:]
    else:
        storage = torch.empty(out.numel() + 20, dtype=n.dtype, device=gems.device)
        out = storage[: out.numel()].view_as(out)
        n = torch.as_strided(storage, n.shape, (384, 128, 1), 10)
    before = out.cpu().view(torch.int16).clone()
    with pytest.raises((ValueError, TypeError)):
        gems.fused_kimi_k3_mla_kv_concat(n, p, out)
    assert torch.equal(out.cpu().view(torch.int16), before)


def test_cpu_unsupported():
    args, _ = workload.build_inputs(
        {"T": 1, "H": 1, "layout": "contiguous"}, torch.float16, "cpu"
    )
    with pytest.raises(NotImplementedError, match="accelerator"):
        gems.fused_kimi_k3_mla_kv_concat(*args)


@pytest.mark.parametrize("dtype", workload.DTYPES)
def test_disjoint_views_of_same_storage(dtype):
    storage = torch.empty(4 * 3 * 192 + 4 * 3 * 128, dtype=dtype, device=gems.device)
    out = storage[: 4 * 3 * 192].view(4, 3, 192)
    n = storage[4 * 3 * 192 :].view(4, 3, 128)
    n.view(torch.int16).fill_(-32768)
    p = torch.empty(4, 64, dtype=dtype, device=gems.device)
    p.view(torch.int16).fill_(32705)
    expected = reference_output(n, p, out)
    assert_output(gems.fused_kimi_k3_mla_kv_concat(n, p, out), out, expected)


@pytest.mark.parametrize("dtype", workload.DTYPES)
@pytest.mark.parametrize("layout", ["transposed", "broadcast"])
def test_read_only_source_stride_variants(dtype, layout):
    nbits = torch.randint(-32768, 32768, (3, 7, 128), dtype=torch.int16)
    pbits = torch.randint(-32768, 32768, (7, 64), dtype=torch.int16)
    n = nbits.to(gems.device).view(dtype).transpose(0, 1)
    p = pbits.to(gems.device).view(dtype)
    if layout == "broadcast":
        n = n[:, :1].expand(7, 3, 128)
        p = p[:1].expand(7, 64)
    out = torch.empty(7, 3, 192, device=gems.device, dtype=dtype)
    expected = reference_output(n, p, out)
    saved = snapshot_inputs((n, p, out))
    assert_output(gems.fused_kimi_k3_mla_kv_concat(n, p, out), out, expected)
    assert_inputs_unchanged(saved)


@pytest.mark.parametrize("dtype", workload.DTYPES)
def test_wide_index_kernel(dtype):
    # Exercise the large-output index specialization without allocating GBs.
    module = importlib.import_module("flaggems_vllm.ops.fused_kimi_k3_mla_kv_concat")
    args, _ = workload.build_inputs(
        {"T": 7, "H": 3, "layout": "padded"}, dtype, gems.device
    )
    n, p, out = args
    expected = reference_output(*args)
    saved = snapshot_inputs(args)
    with gems.runtime.torch_device_fn.device(n.device):
        module._kv_concat_kernel[
            lambda meta: (
                (7 * 3 * (256 if meta["ROW_LAYOUT"] else 192) + meta["BLOCK"] - 1)
                // meta["BLOCK"],
            )
        ](
            n.view(torch.int16),
            p.view(torch.int16),
            out.view(torch.int16),
            out.numel(),
            3,
            n.stride(0),
            n.stride(1),
            p.stride(0),
            WIDE_INDEX=True,
        )
    assert_output(None, out, expected)
    assert_inputs_unchanged(saved)
