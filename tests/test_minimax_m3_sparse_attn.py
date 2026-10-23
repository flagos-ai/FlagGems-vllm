# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import importlib

import pytest
import torch

import flaggems_vllm
from tests import conftest as cfg
from tests.minimax_m3_sparse_attn_utils import dense_reference, make_inputs

op_module = importlib.import_module("flaggems_vllm.ops.minimax_m3_sparse_attn")
pytestmark = pytest.mark.minimax_m3_sparse_attn
cuda_only = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "nvidia" or not torch.cuda.is_available(),
    reason="requires NVIDIA SM80+",
)


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0)])
def test_bf16_request_preserves_precision(capability):
    assert op_module._resolve_qk_dtype("bf16", torch.bfloat16, capability) == "bf16"


@pytest.mark.parametrize("capability", [(8, 0), (8, 6)])
def test_explicit_fp8_request_uses_int8_with_bf16_storage(capability):
    assert op_module._resolve_qk_dtype("fp8", torch.bfloat16, capability) == "int8"
    with pytest.raises(ValueError, match="upstream precision planner"):
        op_module._resolve_qk_dtype("fp8", torch.float8_e4m3fn, capability)


def test_precision_rejects_incompatible_storage():
    with pytest.raises(ValueError, match="typed float8"):
        op_module._resolve_qk_dtype("fp8", torch.bfloat16, (9, 0))
    with pytest.raises(ValueError, match="BF16 KV"):
        op_module._resolve_qk_dtype("int8", torch.float8_e4m3fn, (9, 0))
    with pytest.raises(NotImplementedError, match="SM80"):
        op_module._resolve_qk_dtype("bf16", torch.bfloat16, (7, 5))


@cuda_only
@pytest.mark.parametrize("d", [96, 128])
@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize(
    "mode", ["compact", "scattered", "holes", "duplicates", "empty"]
)
def test_selected_dense_and_redzones(d, kv_heads, mode):
    if cfg.QUICK_MODE and kv_heads == 2 and mode not in ("compact", "holes"):
        pytest.skip("non-quick case")
    args = make_inputs(d=d, kv_heads=kv_heads, mode=mode, topk=17, strided=True)
    q = args[0]
    guard = torch.full((q.numel() + 128,), 123, device=q.device, dtype=q.dtype)
    out = guard[64:-64].view_as(q)
    returned = flaggems_vllm.minimax_m3_sparse_attn(
        *args, output=out, chunks=3, schedule="selected"
    )
    assert returned is out
    torch.testing.assert_close(out.float(), dense_reference(args), atol=0.01, rtol=0)
    assert bool((guard[:64] == 123).all() & (guard[-64:] == 123).all())


@cuda_only
@pytest.mark.parametrize("precision", ["bf16", "fp8", "int8"])
def test_precision_and_scalar_scales(precision):
    if precision == "fp8" and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FP8 hardware required for typed FP8 fixture")
    dtype = torch.float8_e4m3fn if precision == "fp8" else torch.bfloat16
    args = make_inputs(dtype=dtype, mode="scattered")
    q = args[0]
    out = torch.empty_like(q)
    ks = torch.tensor(0.75, device=q.device, dtype=torch.float32)
    vs = torch.tensor(1.25, device=q.device, dtype=torch.float32)
    flaggems_vllm.minimax_m3_sparse_attn(
        *args, output=out, qk_dtype=precision, k_scale=ks, v_scale=vs
    )
    torch.testing.assert_close(
        out.float(), dense_reference(args, k_scale=ks, v_scale=vs), atol=0.02, rtol=0
    )


@cuda_only
@pytest.mark.parametrize("d", [96, 128])
def test_graph_recomputes_union_predicate(d):
    args = make_inputs(qlens=(384, 256), d=d, kv_heads=2, dtype=torch.bfloat16)
    q, _, top, bt, cu, seq, prefix, *_ = args
    out = torch.empty_like(q)
    flaggems_vllm.minimax_m3_sparse_attn(*args, output=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flaggems_vllm.minimax_m3_sparse_attn(*args, output=out)
    for mode in ("scattered", "empty", "duplicates", "compact", "holes"):
        update = make_inputs(
            qlens=(256, 384), d=d, kv_heads=2, dtype=torch.bfloat16, mode=mode
        )
        q.copy_(update[0])
        args[1].copy_(update[1])
        top.copy_(update[2])
        bt.copy_(update[3])
        cu.copy_(update[4])
        seq.copy_(update[5])
        prefix.copy_(update[6])
        graph.replay()
        sample = [0, 255, 256, q.shape[0] - 1]
        torch.testing.assert_close(
            out[sample].float(), dense_reference(args, sample), atol=0.01, rtol=0
        )


@cuda_only
def test_empty_and_unsupported_metadata():
    args = list(make_inputs())
    q = args[0]
    with pytest.raises(ValueError, match="provided together"):
        flaggems_vllm.minimax_m3_sparse_attn(
            *args, output=torch.empty_like(q), k_scale=torch.ones((), device=q.device)
        )
    with pytest.raises(NotImplementedError, match="contiguous"):
        flaggems_vllm.minimax_m3_sparse_attn(
            *args,
            output=torch.empty_like(q).transpose(0, 1).contiguous().transpose(0, 1),
        )
    args[0] = q[:0]
    args[2] = args[2][:, :0]
    args[7] = 0
    out = torch.empty_like(args[0])
    assert flaggems_vllm.minimax_m3_sparse_attn(*args, output=out) is out


@cuda_only
def test_nonfinite_matches_selected_schedule():
    args = make_inputs(qlens=(384,), mode="compact")
    args[0][0, 0, 0] = float("nan")
    args[0][1, 0, 0] = float("inf")
    auto, selected = torch.empty_like(args[0]), torch.empty_like(args[0])
    for output, schedule in ((auto, "auto"), (selected, "selected")):
        flaggems_vllm.minimax_m3_sparse_attn(
            *args, output=output, schedule=schedule, chunks=1
        )
    torch.testing.assert_close(auto, selected, atol=0.01, rtol=0, equal_nan=True)


def test_top_level_export_without_fictitious_aten_registration():
    assert flaggems_vllm.minimax_m3_sparse_attn is op_module.minimax_m3_sparse_attn
    assert "minimax_m3_sparse_attn" not in dict(flaggems_vllm._FULL_CONFIG)
