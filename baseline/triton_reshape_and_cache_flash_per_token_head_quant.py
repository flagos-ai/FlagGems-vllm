# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Workloads, independent FP32 reference, and installed vLLM ABI adapter."""

import hashlib
import importlib
import importlib.metadata
import inspect
from pathlib import Path

import torch

OP_NAME = "triton_reshape_and_cache_flash_per_token_head_quant"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (2, 3, 4, 5)
NATIVE_MODULE = "vllm.v1.attention.ops.triton_reshape_and_cache_flash"
NATIVE_UNAVAILABLE_REASON = None
NATIVE_PROVENANCE = None

ARGUMENTS = (
    "key",
    "value",
    "key_cache",
    "value_cache",
    "k_scale_cache",
    "v_scale_cache",
    "slot_mapping",
)
SHAPE_FIELDS = ("T", "H", "Dk", "Dv", "S", "layout", "slots", "cache")
_CORE = (
    (1, 1, 64, 64, 16, "contiguous", "linear"),
    (1, 8, 128, 128, 16, "contiguous", "linear"),
    (128, 3, 80, 128, 16, "strided", "random"),
    (256, 8, 160, 96, 32, "strided", "random"),
    (1024, 8, 128, 128, 32, "contiguous", "linear"),
)
_EXTRA = (
    (64, 3, 32, 512, 16, "strided", "random"),
    (64, 3, 512, 32, 16, "strided", "random"),
    (64, 3, 512, 512, 32, "contiguous", "linear"),
)
CORE_SHAPES = tuple((*shape, kind) for shape in _CORE for kind in ("int8", "fp8"))
EXTRA_SHAPES = tuple((*shape, kind) for shape in _EXTRA for kind in ("int8", "fp8"))
_SCALE_SENTINEL = -17.5


def cache_dtype(binding):
    if binding["cache"] == "int8":
        return torch.int8
    if binding["cache"] == "fp8":
        if "fp8_dtype" in binding:
            return binding["fp8_dtype"]
        import flaggems_vllm

        # Match the neighboring FlagGems cache path. AMD platforms may use
        # either E4M3 format, depending on the device generation.
        if flaggems_vllm.vendor_name == "amd":
            if torch.cuda.get_device_properties().gcnArchName.split(":")[0] == "gfx942":
                return torch.float8_e4m3fnuz
        if flaggems_vllm.vendor_name == "hygon":
            return torch.float8_e4m3fnuz
        return torch.float8_e4m3fn
    raise ValueError(f"unknown cache kind: {binding['cache']}")


def native():
    global NATIVE_UNAVAILABLE_REASON, NATIVE_PROVENANCE
    NATIVE_UNAVAILABLE_REASON = NATIVE_PROVENANCE = None
    try:
        module = importlib.import_module(NATIVE_MODULE)
        op = getattr(module, OP_NAME)
        signature = inspect.signature(op)
        names = tuple(signature.parameters)
        if names == ARGUMENTS:
            abi = "7-argument"
            modes = None
            signature.bind(*([None] * 7))
        elif names == (*ARGUMENTS, "kv_quant_mode"):
            abi = "8-argument (KVQuantMode)"
            from vllm.v1.kv_cache_interface import KVQuantMode

            modes = {
                "int8": KVQuantMode.INT8_PER_TOKEN_HEAD,
                "fp8": KVQuantMode.FP8_PER_TOKEN_HEAD,
            }
            mode_positional = (
                signature.parameters["kv_quant_mode"].kind
                == inspect.Parameter.POSITIONAL_ONLY
            )
            if mode_positional:
                signature.bind(*([None] * 8))
            else:
                signature.bind(*([None] * 7), kv_quant_mode=None)
        else:
            raise ValueError(f"unsupported native signature {signature}")
        source = inspect.getsource(op)
        if op.__module__ != NATIVE_MODULE or "flaggems_vllm" in source:
            raise ValueError("native symbol is redirected from its vLLM implementation")
        path = Path(inspect.getfile(op))
        NATIVE_PROVENANCE = {
            "abi": abi,
            "signature": str(signature),
            "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "version": importlib.metadata.version("vllm"),
        }
    except Exception as exc:
        NATIVE_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
        return None

    def bind(args):
        dtype = args[2].dtype
        if args[6].stride(0) != 1:
            raise ValueError("native slot_mapping must have unit stride")
        if dtype != torch.int8:
            native_dtype = getattr(module, "FP8_DTYPE", None)
            if native_dtype is None:
                native_dtype = module.current_platform.fp8_dtype()
            if dtype != native_dtype:
                raise ValueError(
                    f"native FP8 encoding is {native_dtype}, cache uses {dtype}"
                )
        if modes is None:
            return op, {}
        mode = modes["int8" if dtype == torch.int8 else "fp8"]
        if mode_positional:
            return lambda *native_args: op(*native_args, mode), {}
        return op, {"kv_quant_mode": mode}

    # The collector calls native() with the original seven arguments. The
    # benchmark can bind the mode once, outside qualification and timing.
    def invoke(*args):
        bound_op, kwargs = bind(args)
        return bound_op(*args, **kwargs)

    invoke.bind = bind
    return invoke


def grid():
    return [dict(zip(SHAPE_FIELDS, shape)) for shape in CORE_SHAPES + EXTRA_SHAPES]


def key_shape(binding):
    return "/".join(str(binding[field]) for field in SHAPE_FIELDS)


def _strides(shape, order, gap):
    step = 1
    strides = [0] * len(shape)
    for level, dim in enumerate(reversed(order)):
        strides[dim] = step
        step = step * max(shape[dim], 1) + gap * (level + 1)
    return tuple(strides)


def _specs(binding, dtype):
    t, h, dk, dv, s = (binding[name] for name in SHAPE_FIELDS[:5])
    p = (max(t, 1) + s - 1) // s + 3
    strided = binding["layout"] == "strided"
    quant = cache_dtype(binding)
    specs = []
    for shape, order, gap, kind in (
        ((t, h, dk), (0, 1, 2), 3, dtype),
        ((t, h, dv), (1, 0, 2), 5, dtype),
        ((p, s, h, dk), (0, 2, 1, 3), 7, quant),
        ((p, s, h, dv), (0, 1, 2, 3), 11, quant),
        ((p, s, h), (1, 0, 2), 2, torch.float32),
        ((p, s, h), (2, 0, 1), 4, torch.float32),
        ((t,), (0,), 0, binding.get("slot_dtype", torch.int64)),
    ):
        if not strided:
            order, gap = tuple(range(len(shape))), 0
        stride = _strides(shape, order, gap)
        if len(shape) == 1:
            stride = (binding.get("slot_stride", 1),)
        specs.append((shape, stride, kind))
    return specs


def config(binding, dtype):
    return {
        "slots": binding["slots"],
        "native_abi": NATIVE_PROVENANCE["abi"] if NATIVE_PROVENANCE else None,
        "inputs": {
            name: {"shape": list(shape), "stride": list(stride), "dtype": str(kind)}
            for name, (shape, stride, kind) in zip(ARGUMENTS, _specs(binding, dtype))
        },
    }


def build_inputs(binding, dtype, device):
    specs = _specs(binding, dtype)
    t, h, _ = specs[0][0]
    p, s = specs[2][0][:2]
    gen = torch.Generator().manual_seed(143)
    buffers = []
    for index, (shape, stride, kind) in enumerate(specs):
        offset = index + 3
        count = offset + sum((max(1, n) - 1) * step for n, step in zip(shape, stride))
        bytes_cache = index in (2, 3) and kind != torch.int8
        storage_dtype = torch.uint8 if bytes_cache else kind
        sentinel = 0x55 if bytes_cache else -91 if index in (2, 3) else _SCALE_SENTINEL
        storage = torch.full((count + 33,), sentinel, dtype=storage_dtype)
        view = storage.as_strided(shape, stride, offset)
        if index in (0, 1):
            amplitudes = (torch.arange(t) % 5 + 1)[:, None] * (torch.arange(h) + 1)[
                None, :
            ]
            sample = torch.randn(shape, generator=gen) * (
                amplitudes[..., None] * (4**index)
            )
            view.copy_(sample)
        if index == 6:
            if binding["slots"] == "random":
                # Force both page edges into the sample and reserve one untouched page.
                pool = torch.arange((p - 1) * s)
                edges = torch.tensor(sorted({0, s - 1, s}))
                pool = pool[~torch.isin(pool, edges)]
                slots = torch.cat(
                    (edges, pool[torch.randperm(len(pool), generator=gen)])
                )[:t]
                slots[5::5] = -7
            elif binding["slots"] == "negative":
                slots = torch.full((t,), -7)
            else:
                slots = torch.arange(t)
            view.copy_(slots)
        storage = storage.to(device)
        if bytes_cache:
            storage = storage.view(kind)
        buffers.append(storage.as_strided(shape, stride, offset))
    return tuple(buffers), {}


def reference(args):
    """CPU FP32 math and indexed writes into complete copies of the four buffers."""
    from baseline.triton_reshape_and_cache_flash import backing, encode_fp8_cpu

    key, value, kc, vc, ks, vs, slots = args
    valid = slots.cpu().long()
    take = (valid >= 0) & (valid < kc.shape[0] * kc.shape[1])
    pages, offsets = valid[take] // kc.shape[1], valid[take] % kc.shape[1]
    fp8 = kc.dtype != torch.int8
    qmax = {
        torch.int8: 127.0,
        torch.float8_e4m3fn: 448.0,
        torch.float8_e4m3fnuz: 224.0,
    }[kc.dtype]
    data_refs, scale_refs = [], []
    for source, cache, scale_cache in ((key, kc, ks), (value, vc, vs)):
        rows = source.cpu().float()[take]
        scale = (rows.abs().amax(dim=-1) / qmax).clamp_min_(1e-6)
        values = rows * scale.reciprocal()[..., None]
        if fp8:
            codes = encode_fp8_cpu(values.clamp(-qmax, qmax), cache.dtype)
        else:
            adjusted = torch.where(values >= 0, values + 0.5, values - 0.5)
            codes = adjusted.clamp(-128, 127).to(torch.int8).view(torch.uint8)

        data = backing(cache).view(torch.uint8).cpu().clone()
        scales = backing(scale_cache).cpu().clone()
        data.as_strided(cache.shape, cache.stride(), cache.storage_offset())[
            pages, offsets
        ] = codes
        scales.as_strided(
            scale_cache.shape, scale_cache.stride(), scale_cache.storage_offset()
        )[pages, offsets] = scale
        data_refs.append(data)
        scale_refs.append(scales)
    return (*data_refs, *scale_refs)


def snapshot_outputs(args):
    from baseline.triton_reshape_and_cache_flash import backing

    return tuple(backing(args[i]).view(torch.uint8) for i in MUTATED_INPUTS)


def assert_outputs(args, expected, *, scale_rtol=0):
    for index, (actual, want) in enumerate(zip(snapshot_outputs(args), expected)):
        actual = actual.cpu().view(want.dtype)
        if index < 2:
            torch.testing.assert_close(actual, want, rtol=0, atol=0)
        else:
            untouched = want == _SCALE_SENTINEL
            torch.testing.assert_close(
                actual[untouched], want[untouched], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual[~untouched], want[~untouched], rtol=scale_rtol, atol=0
            )
