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


import inspect

import pytest
import torch

import flaggems_vllm

from . import conftest as cfg
from .marlin_moe_utils import (
    FP8_CONFIGS,
    FP8_ORIGINAL_CONFIGS,
    check_decode_edges,
    check_exact_decode,
    check_graph_mutation,
    check_large_expert_stride,
    check_large_route_reduction,
    check_mutation,
    check_output,
    check_public,
    check_result,
    check_scale_strides,
    decode_weight,
    make_inputs,
    reference,
    supported_device,
)

pytestmark = pytest.mark.fused_marlin_moe_w8a16_fp8
MUSA_ONLY = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "mthreads", reason="Moore Threads backend contract"
)
DTYPES = [torch.bfloat16, torch.float16]
ACTIVE_FP8_CONFIGS = FP8_CONFIGS[:2] if cfg.QUICK_MODE else FP8_CONFIGS


@pytest.mark.skipif(
    not supported_device(), reason="requires Hopper or a vendor override"
)
@pytest.mark.parametrize(
    "config",
    ACTIVE_FP8_CONFIGS,
    ids=lambda shape: ("original" if shape in FP8_ORIGINAL_CONFIGS else "ordinary")
    + "-"
    + "x".join(map(str, shape)),
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_marlin_moe_w8a16_fp8(config, dtype):
    group_size = 64 if config[2] == 64 else 128
    if group_size == 64 and flaggems_vllm.vendor_name == "nvidia":
        pytest.skip("generic NVIDIA FP8 path requires group size at least 128")
    args, weights = make_inputs(config, dtype, "fp8", group_size)

    check_public(args, weights)


@MUSA_ONLY
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", ["out", "inplace", "alias"])
@pytest.mark.parametrize("shape", [(4, 256, 512), (1, 1024, 1024), (65, 256, 512)])
def test_output(dtype, mode, shape):
    check_output("fp8", dtype, shape, mode)


@MUSA_ONLY
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("field", ["w1", "w2", "w1_scale", "w2_scale"])
def test_same_address_mutation(dtype, field):
    check_mutation("fp8", dtype, field)


@MUSA_ONLY
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
def test_decode_edges(dtype, group_size):
    check_decode_edges("fp8", dtype, group_size)


@MUSA_ONLY
@pytest.mark.parametrize("dtype", DTYPES)
def test_expert_address_above_two_gib(dtype):
    check_large_expert_stride("fp8", dtype)


@MUSA_ONLY
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("experts", [3, 17, 65])
def test_non_power_of_two_experts(dtype, experts):
    args, weights = make_inputs((33, experts, 128, 256, 2), dtype, "fp8")
    check_public(args, weights)


@MUSA_ONLY
def test_route_reduction_above_int32():
    check_large_route_reduction(2, 4096)


@MUSA_ONLY
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", ["default", "out", "inplace", "alias"])
def test_empty(dtype, mode):
    args, _ = make_inputs((0, 2, 128, 256, 1), dtype, "fp8")
    hs = args["hidden_states"]
    destination = None
    if mode in ("out", "alias"):
        destination = torch.empty_like(hs) if mode == "out" else hs
        args["output"] = destination
    elif mode == "inplace":
        destination = hs
        args["inplace"] = True
    result = flaggems_vllm.fused_marlin_moe(**args)
    assert (
        result.shape == hs.shape
        and result.dtype == dtype
        and result.device == hs.device
    )
    if destination is not None:
        assert result is destination


@MUSA_ONLY
@pytest.mark.parametrize("dtype", DTYPES)
def test_invalid_output(dtype):
    args, _ = make_inputs((4, 2, 128, 256, 1), dtype, "fp8")
    hs = args["hidden_states"]
    invalid = (
        torch.empty((4, 64), device=hs.device, dtype=dtype),
        torch.empty(hs.shape, device=hs.device, dtype=torch.float32),
        torch.empty(hs.shape, device="cpu", dtype=dtype),
        torch.empty((4, 256), device=hs.device, dtype=dtype)[:, ::2],
    )
    for output in invalid:
        with pytest.raises(ValueError):
            flaggems_vllm.fused_marlin_moe(**args, output=output)
    with pytest.raises(ValueError):
        flaggems_vllm.fused_marlin_moe(**args, output=hs, inplace=True)


@MUSA_ONLY
@pytest.mark.parametrize(
    "option", ["activation", "g_idx1", "input_dtype", "bias1", "is_k_full"]
)
def test_unsupported_options(option):
    args, _ = make_inputs((1, 2, 128, 256, 1), torch.bfloat16, "fp8")
    values = {
        "activation": "relu",
        "g_idx1": torch.zeros(128, device=flaggems_vllm.device, dtype=torch.int32),
        "input_dtype": torch.float8_e4m3fn,
        "bias1": torch.zeros((2, 512), device=flaggems_vllm.device),
        "is_k_full": False,
    }
    args[option] = values[option]
    with pytest.raises(NotImplementedError):
        flaggems_vllm.fused_marlin_moe(**args)


@MUSA_ONLY
def test_public_backend_binding():
    source = inspect.getfile(inspect.unwrap(flaggems_vllm.fused_marlin_moe))
    assert "/_mthreads/" in source.replace("\\", "/")


@MUSA_ONLY
@pytest.mark.parametrize("dtype", DTYPES)
def test_precision_entry(dtype):
    args, weights = make_inputs((4, 4, 256, 512, 2), dtype, "fp8")
    fn = flaggems_vllm.fused_marlin_moe_w8a16_fp8
    assert "/_mthreads/" in inspect.getfile(inspect.unwrap(fn)).replace("\\", "/")
    expected = reference(args, weights)
    direct_args = {
        key: value
        for key, value in args.items()
        if key not in ("bias1", "bias2", "quant_type_id")
    }
    result = fn(**direct_args)
    check_result(result, expected, args["hidden_states"])


@MUSA_ONLY
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_tokens", [512, 513, 1024, 1025, 4096, 32768])
def test_batch_boundaries(dtype, num_tokens):
    args, weights = make_inputs((num_tokens, 8, 1024, 1024, 2), dtype, "fp8")
    check_public(args, weights)


@MUSA_ONLY
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fp32_scales", [False, True])
@pytest.mark.parametrize(
    "num_tokens,group_size", [(1, 256), (16, 384), (64, 256), (1025, 128), (1025, 384)]
)
def test_group_alignment(dtype, fp32_scales, num_tokens, group_size):
    args, _ = make_inputs(
        (num_tokens, 2, 2 * group_size, 2 * group_size, 1), dtype, "fp8", group_size
    )
    args["hidden_states"].mul_(0.1)
    args["topk_ids"] = (
        torch.arange(num_tokens, device=flaggems_vllm.device) % 2
    ).reshape(-1, 1)
    args["topk_weights"] = torch.full((num_tokens, 1), 0.7, device=flaggems_vllm.device)
    for key in ("w1", "w2"):
        q = args[key]
        args[key] = (torch.randn(q.shape, device=q.device) * 4).to(torch.float8_e4m3fn)
        scale = args[f"{key}_scale"]
        args[f"{key}_scale"] = (
            (torch.rand(scale.shape, device=scale.device) + 0.5) * 0.005
        ).to(torch.float32 if fp32_scales else dtype)
    weights = tuple(
        decode_weight(args[key], args[f"{key}_scale"], "fp8", group_size, dtype)
        for key in ("w1", "w2")
    )
    check_public(args, weights)


@MUSA_ONLY
@pytest.mark.parametrize("dtype", DTYPES)
def test_exhaustive_decode(dtype):
    check_exact_decode("fp8", dtype)


@MUSA_ONLY
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("group_size", [64, 384])
def test_scale_strides(dtype, group_size):
    check_scale_strides("fp8", dtype, group_size)


@MUSA_ONLY
def test_graph_replay_mutation():
    check_graph_mutation("fp8")
