# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Metadata-only regressions; runnable without importing Torch or initializing CUDA.

python -m pytest --noconftest -c /dev/null tests/test_fused_moe_policy_cpu.py
"""

import ast
import enum
import functools
import math
import numbers
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).parents[1] / "src/flaggems_vllm/ops/fused_moe.py"


class TensorMetadata:
    def __init__(self, shape=(64, 384), dtype="bf16", device="cuda"):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)
        self.device = SimpleNamespace(type=device, index=0)

    def size(self, axis):
        return self.shape[axis]


@pytest.fixture
def policy():
    functions = {
        "_is_w8a8_quant_sm90_device",
        "_is_w8a8_tp8_h100_device",
        "_supports_w8a8_quant_fusion",
        "_supports_w8a8_native_div_quant_fusion",
        "_native_cuda_int8_quant_div_supported",
        "_int8_quant_dispatch_policy",
        "_get_w8a8_tp8_h100_config",
        "_get_config_dtype_str",
        "_validate_swigluoai_uninterleave_params",
    }
    nodes = []
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name == "_Int8QuantRecipe":
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.startswith("_INT8_QUANT_")
            for t in node.targets
        ):
            nodes.append(node)
    kernels = {
        "aten::div.Tensor": "SafeKernelFunction(debug='registered at "
        "/__w/pytorch/pytorch/build/aten/src/ATen/RegisterCUDA_0.cpp:8338')",
        "aten::div.Scalar": "SafeKernelFunction(debug='registered at "
        "/__w/pytorch/pytorch/build/aten/src/ATen/RegisterCompositeExplicitAutograd_0.cpp:4047')",
    }
    torch = SimpleNamespace(
        __version__="2.13.0+cu129",
        bfloat16="bf16",
        float16="fp16",
        float32="fp32",
        float="fp32",
        Tensor=TensorMetadata,
        dtype=object,
        finfo=lambda _: SimpleNamespace(max=3.4028234663852886e38),
        version=SimpleNamespace(
            git_version="cf30153c4c131c8164ee7798e5022d810682e2cb",
            cuda="12.9",
            hip=None,
        ),
        cuda=SimpleNamespace(
            get_device_capability=lambda _: (9, 0),
            get_device_name=lambda _: "NVIDIA H100 80GB HBM3",
        ),
        library=SimpleNamespace(get_kernel=lambda op, _: kernels[op]),
    )
    package = SimpleNamespace(current_work_registrar=None)
    ns = dict(
        torch=torch,
        IntEnum=enum.IntEnum,
        Optional=__import__("typing").Optional,
        Any=object,
        math=math,
        numbers=numbers,
        functools=functools,
        sys=SimpleNamespace(modules={"flag_gems": package}),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return SimpleNamespace(ns=ns, torch=torch, package=package, kernels=kernels)


def test_live_registration_and_foreign_overrides(policy):
    ns = policy.ns
    recipe = ns["_Int8QuantRecipe"]
    resolve = ns["_int8_quant_dispatch_policy"]
    assert resolve() == recipe.TORCH_ALL
    keys = set(ns["_INT8_QUANT_REQUIRED_FG_KEYS"])
    policy.package.current_work_registrar = SimpleNamespace(get_all_keys=lambda: keys)
    assert resolve() == recipe.FG_ALL
    keys.remove("div.Tensor")
    assert resolve() is None
    keys.remove("div.Scalar")
    assert resolve() == recipe.FG_NATIVE_DIV
    for op in policy.kernels:
        original = policy.kernels[op]
        policy.kernels[op] = "third-party override"
        assert resolve() is None
        policy.kernels[op] = original
        assert resolve() == recipe.FG_NATIVE_DIV
    for key in tuple(keys):
        keys.remove(key)
        assert resolve() is None
        keys.add(key)
    keys.update(ns["_INT8_QUANT_DIV_FG_KEYS"])
    assert resolve() == recipe.FG_ALL


@pytest.mark.parametrize(
    "registrar", [object(), SimpleNamespace(get_all_keys=lambda: None)]
)
def test_unknown_registrar(policy, registrar):
    policy.package.current_work_registrar = registrar
    assert policy.ns["_int8_quant_dispatch_policy"]() is None


@pytest.mark.parametrize(
    "field,value", [("git_version", "other"), ("cuda", "13.0"), ("hip", "rocm")]
)
def test_pinned_native_runtime(policy, field, value):
    policy.package.current_work_registrar = SimpleNamespace(
        get_all_keys=lambda: policy.ns["_INT8_QUANT_CORE_FG_KEYS"]
    )
    setattr(policy.torch.version, field, value)
    assert policy.ns["_int8_quant_dispatch_policy"]() is None


def test_other_torch_version(policy):
    policy.package.current_work_registrar = SimpleNamespace(
        get_all_keys=lambda: policy.ns["_INT8_QUANT_CORE_FG_KEYS"]
    )
    policy.torch.__version__ = "2.14.0"
    assert policy.ns["_int8_quant_dispatch_policy"]() is None


@pytest.mark.parametrize(
    "tensor,expected",
    [
        (TensorMetadata(), True),
        (TensorMetadata((8192, 6144)), True),
        (TensorMetadata((32768, 384)), True),
        (TensorMetadata((32769, 384)), False),
        (TensorMetadata((64, 385)), False),
        (TensorMetadata((64, 1, 384)), False),
        (TensorMetadata(dtype="fp16"), False),
        (TensorMetadata(dtype="fp32"), False),
        (TensorMetadata(device="cpu"), False),
        (TensorMetadata(device="musa"), False),
        (TensorMetadata(device="xpu"), False),
    ],
)
def test_native_div_shape_dtype_guard(policy, tensor, expected):
    assert policy.ns["_supports_w8a8_native_div_quant_fusion"](tensor) == expected


@pytest.mark.parametrize(
    "name,capability", [("NVIDIA H800", (9, 0)), ("NVIDIA H100 80GB HBM3", (8, 0))]
)
def test_exact_product_guard(policy, name, capability):
    policy.torch.cuda.get_device_name = lambda _: name
    policy.torch.cuda.get_device_capability = lambda _: capability
    assert not policy.ns["_supports_w8a8_native_div_quant_fusion"](TensorMetadata())


@pytest.mark.parametrize("tokens", [0, 1, 8, 15, 65, 128, 1023, 8193, 32768])
def test_tuning_envelope_fallback(policy, tokens):
    assert policy.ns["_get_w8a8_tp8_h100_config"](tokens, "gemm1") is None


@pytest.mark.parametrize(
    "tokens", [16, 17, 24, 32, 63, 64, 1024, 2048, 4095, 4096, 8192]
)
def test_stage_configs_share_alignment(policy, tokens):
    get = policy.ns["_get_w8a8_tp8_h100_config"]
    first, second = get(tokens, "gemm1"), get(tokens, "gemm2")
    assert first["BLOCK_SIZE_M"] == second["BLOCK_SIZE_M"]
    if tokens <= 64:
        assert first["SWAP_AB"] and "SWAP_AB" not in second
        assert first["BLOCK_SIZE_K"] == second["BLOCK_SIZE_K"] == 128
        assert not second.get("INT32_DOWN", False)
    else:
        assert second["INT32_DOWN"]
        assert first["BLOCK_SIZE_K"] == second["BLOCK_SIZE_K"] == 64


def test_w8a8_never_selects_plain_half_config(policy):
    assert (
        policy.ns["_get_config_dtype_str"](dtype="bf16", use_int8_w8a8=True)
        == "int8_w8a8"
    )


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize(
    "value", [1e39, -1e39, float("nan"), float("inf"), True, None, 10**1000]
)
def test_parameters_rejected_before_tensor_work(policy, index, value):
    args = [1.702, 1.0, 7.0]
    args[index] = value
    with pytest.raises(ValueError):
        policy.ns["_validate_swigluoai_uninterleave_params"](*args)


def test_new_kernel_bodies_do_not_add_torch_compute():
    tree = ast.parse(SOURCE.read_text())
    names = {
        "_is_w8a8_quant_sm90_device",
        "_is_w8a8_tp8_h100_device",
        "_supports_w8a8_quant_fusion",
        "_supports_w8a8_native_div_quant_fusion",
        "_native_cuda_int8_quant_div_supported",
        "_int8_quant_dispatch_policy",
        "_get_w8a8_tp8_h100_config",
    }
    allowed = {
        "torch.cuda.get_device_capability",
        "torch.cuda.get_device_name",
        "torch.library.get_kernel",
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            for call in ast.walk(node):
                if isinstance(call, ast.Call):
                    name = ast.unparse(call.func)
                    if name.startswith("torch."):
                        assert name in allowed, (node.name, name)
