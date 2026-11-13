# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Frozen native coordinates and workloads for ordinary/global-FP8 KV writes."""

import hashlib
import importlib
import importlib.metadata
import inspect
import math

import torch

OP_NAME = "triton_reshape_and_cache_flash"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
IS_INPLACE = True
MUTATED_INPUTS = (2, 3)
NATIVE_MODULE = "vllm.v1.attention.ops.triton_reshape_and_cache_flash"
SAMPLED_SHA256 = "f18224e51162e37c14a2e2b3045bca55baf0b207b1d203ad6ee7d8ed417139eb"
NATIVE_UNAVAILABLE_REASON = None
NATIVE_PROVENANCE = None


def native():
    global NATIVE_UNAVAILABLE_REASON, NATIVE_PROVENANCE
    NATIVE_UNAVAILABLE_REASON = None
    NATIVE_PROVENANCE = None
    try:
        module = importlib.import_module(NATIVE_MODULE)
        op = getattr(module, OP_NAME)
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = "native symbol is not callable"
        return None
    if "flaggems_vllm" in op.__module__ or "flaggems_vllm" in inspect.getsource(op):
        raise RuntimeError("native is redirected to candidate")
    from pathlib import Path

    path = Path(inspect.getsourcefile(op))
    NATIVE_PROVENANCE = {
        "module": op.__module__,
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sampled_sha256": SAMPLED_SHA256,
        "version": importlib.metadata.version("vllm"),
        "signature": str(inspect.signature(op)),
        "fp8_dtype": str(module.current_platform.fp8_dtype()),
    }
    if tuple(inspect.signature(op).parameters) != (
        "key",
        "value",
        "key_cache",
        "value_cache",
        "slot_mapping",
        "kv_cache_dtype",
        "k_scale",
        "v_scale",
    ):
        raise RuntimeError("native signature drift: diff before benchmarking")
    return op


def grid():
    """One matrix shared by baseline and standard benchmark; no random duplicates."""
    return [
        dict(
            T=t,
            H=h,
            D=d,
            BS=bs,
            layout=layout,
            padding=padding,
            distribution=distribution,
            cache=cache,
        )
        for t, h, d, bs, padding in (
            (1, 8, 128, 16, 0),
            (17, 3, 80, 7, 13),
            (128, 8, 128, 16, 0),
            (1024, 8, 128, 32, 0),
        )
        for layout in ("flash", "head_major")
        for distribution in ("contiguous", "random")
        for cache in ("auto", "fp8", "uint8")
    ]


def fp8_dtype(vendor=None):
    if vendor is None:
        import flaggems_vllm

        vendor = flaggems_vllm.vendor_name
    return torch.float8_e4m3fnuz if vendor in ("hygon", "amd") else torch.float8_e4m3fn


def backing(tensor):
    """View entire allocation, including prefix/suffix and stride gaps."""
    return tensor.as_strided(
        (tensor.untyped_storage().nbytes() // tensor.element_size(),), (1,), 0
    )


def encode_fp8_cpu(value, dtype):
    """Independent nearest-representable search, using a decoded E4M3 table."""
    fnuz = dtype == torch.float8_e4m3fnuz
    bias = 8 if fnuz else 7
    count = 128 if fnuz else 127
    table = torch.tensor(
        [
            (
                (code % 8) * 2.0 ** (1 - bias - 3)
                if code < 8
                else (1 + (code % 8) / 8) * 2.0 ** (code // 8 - bias)
            )
            for code in range(count)
        ],
        dtype=torch.float64,
    )
    flat = value.double().abs().clamp(max=table[-1]).contiguous()
    upper = torch.searchsorted(table, flat).clamp(0, count - 1)
    lower = (upper - 1).clamp(min=0)
    dl, du = flat - table[lower], table[upper] - flat
    code = torch.where((du < dl) | ((du == dl) & ((upper % 2) == 0)), upper, lower)
    code = code | (torch.signbit(value).long() * 128)
    if fnuz:
        code = torch.where((code & 127) == 0, 0, code)
    code = torch.where(torch.isnan(value), 128 if fnuz else 127, code)
    return code.to(torch.uint8)


def _device_copy(tensor, device):
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        return tensor.view(torch.uint8).to(device).view(tensor.dtype)
    return tensor.to(device)


def build_inputs(binding, dtype, device):
    t, h, d, bs = (binding[name] for name in ("T", "H", "D", "BS"))
    padding = binding.get("padding", 0)
    fmt = binding.get("format", fp8_dtype())
    cache_kind = binding.get("cache", "auto")
    target = (
        dtype
        if cache_kind == "auto"
        else (torch.uint8 if cache_kind == "uint8" else fmt)
    )
    source_dtype = fmt if binding.get("prequantized", False) else dtype
    sources = []
    for index in range(2):
        step = h * d + padding * (index + 1)
        raw = torch.arange((t + 2) * step, dtype=torch.float32)
        raw = ((raw * (index + 3)) % 2047 - 1023) / 8
        if binding.get("edges", False):
            edge = torch.tensor(
                [
                    0.0,
                    -0.0,
                    2**-20,
                    -(2**-9),
                    2**-6,
                    1.0625,
                    1.1875,
                    240.0,
                    448.0,
                    464.0,
                    1e6,
                    -1e6,
                    float("inf"),
                    -float("inf"),
                    float("nan"),
                    25.375,
                    -25.375,
                ]
            )
            raw[: len(edge)] = edge
            # Repeat edge values into the first consumed row, too.
            raw[step : step + min(len(edge), h * d)] = edge[: h * d]
        if source_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            raw = encode_fp8_cpu(raw, source_dtype).view(source_dtype)
        else:
            raw = raw.to(source_dtype)
        raw = _device_copy(raw, device)
        sources.append(raw.as_strided((t, h, d), (step, d, 1), step))
    blocks = math.ceil(max(t, 1) / bs) + 4
    x = binding.get("x", 8 if d % 8 == 0 else 1)
    if binding["layout"] == "head_major":
        head_stride = d * bs
        block_stride = h * head_stride + padding
        kshape, kstride = (blocks, h, d // x, bs, x), (
            block_stride,
            head_stride,
            bs * x,
            x,
            1,
        )
        vshape, vstride = (blocks, h, d, bs), (block_stride, head_stride, bs, 1)
    else:
        head_stride = d
        page_stride = h * d
        block_stride = bs * page_stride + padding
        kshape = vshape = (blocks, bs, h, d)
        kstride = vstride = (block_stride, page_stride, head_stride, 1)
    caches = []
    for shape, stride in ((kshape, kstride), (vshape, vstride)):
        count = (blocks + 2) * block_stride
        if target in (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.uint8):
            storage = torch.full((count,), 0x55, dtype=torch.uint8).view(target)
        else:
            storage = torch.full((count,), -113.25, dtype=target)
        storage = _device_copy(storage, device)
        caches.append(storage.as_strided(shape, stride, block_stride))
    n = binding.get("slots", t)
    available = (blocks - 2) * bs
    if binding.get("distribution", "random") == "contiguous":
        slots = torch.arange(n, dtype=torch.int64) + bs
    else:
        slots = (
            torch.randperm(available, generator=torch.Generator().manual_seed(810))[:n]
            + bs
        )
    if n > 2:
        slots[1::5] = -1
    if binding.get("all_negative", False):
        slots.fill_(-7)
    # Source and cache identity are independent, even when dtype/shape match.
    args = (
        *sources,
        *caches,
        slots.to(dtype=binding.get("slot_dtype", torch.int64), device=device),
        binding.get("selector", "auto" if cache_kind == "auto" else "fp8"),
        torch.tensor(binding.get("k_scale", 0.5), dtype=torch.float32, device=device),
        torch.tensor(binding.get("v_scale", 2.0), dtype=torch.float32, device=device),
    )
    return args, {}


def reference(args):
    """CPU indexing oracle, returning complete cache allocation bytes."""
    key, value, kc, vc, slots, selector, ks, vs = args
    outputs = []
    for source, cache, scale in ((key, kc, ks), (value, vc, vs)):
        storage = (
            backing(cache).view(torch.uint8).detach().cpu().clone().view(cache.dtype)
        )
        view = storage.as_strided(cache.shape, cache.stride(), cache.storage_offset())
        target = fp8_dtype() if cache.dtype == torch.uint8 else cache.dtype
        fp8 = target in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        if fp8:
            write_view = view.view(torch.uint8)
            if source.dtype == target:
                values = source.view(torch.uint8).cpu()
            else:
                values = (
                    source.view(torch.uint8).cpu().view(source.dtype).float()
                    if source.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
                    else source.cpu().float()
                )
                if selector.startswith("fp8"):
                    values = values / scale.cpu()
                values = encode_fp8_cpu(values, target)
        else:
            write_view, values = view, source.cpu().to(cache.dtype)
        bs = kc.shape[3] if kc.ndim == 5 else kc.shape[1]
        for token, slot in enumerate(slots.cpu().tolist()):
            if slot < 0:
                continue
            block, offset = divmod(slot, bs)
            if kc.ndim == 5:
                if cache.ndim == 5:
                    write_view[block, :, :, offset, :] = values[token].reshape(
                        cache.shape[1], cache.shape[2], cache.shape[4]
                    )
                else:
                    write_view[block, :, :, offset] = values[token]
            else:
                write_view[block, offset] = values[token]
        outputs.append(storage.view(torch.uint8))
    return outputs


def key_shape(binding):
    return "_".join(f"{key}={value}" for key, value in sorted(binding.items()))


def config(binding, dtype):
    t, h, d, bs = (binding[name] for name in ("T", "H", "D", "BS"))
    pad = binding.get("padding", 0)
    blocks = math.ceil(max(t, 1) / bs) + 4
    kind = binding.get("cache", "auto")
    fmt = binding.get("format", fp8_dtype())
    output_dtype = (
        dtype if kind == "auto" else (torch.uint8 if kind == "uint8" else fmt)
    )
    source_dtype = fmt if binding.get("prequantized", False) else dtype
    x = binding.get("x", 8 if d % 8 == 0 else 1)
    if binding["layout"] == "head_major":
        cb, ch = h * d * bs + pad, d * bs
        shapes = [(blocks, h, d // x, bs, x), (blocks, h, d, bs)]
        strides = [(cb, ch, bs * x, x, 1), (cb, ch, bs, 1)]
    else:
        cb = bs * h * d + pad
        shapes = [(blocks, bs, h, d)] * 2
        strides = [(cb, h * d, d, 1)] * 2
    inputs = {}
    for i, name in enumerate(("key", "value")):
        step = h * d + pad * (i + 1)
        inputs[name] = {
            "shape": [t, h, d],
            "stride": [step, d, 1],
            "dtype": str(source_dtype),
            "storage_offset": step,
            "backing_elements": (t + 2) * step,
        }
    for i, name in enumerate(("key_cache", "value_cache")):
        inputs[name] = {
            "shape": list(shapes[i]),
            "stride": list(strides[i]),
            "dtype": str(output_dtype),
            "storage_offset": cb,
            "backing_elements": (blocks + 2) * cb,
            "mutation": True,
            "same_storage_after": True,
        }
    inputs["slot_mapping"] = {
        "shape": [binding.get("slots", t)],
        "stride": [1],
        "dtype": str(binding.get("slot_dtype", torch.int64)),
        "distribution": binding.get("distribution", "random"),
        "seed": 810,
        "negative_rule": "positions 1::5=-1 when N>2; optional all_negative",
        "positive_rule": "unique slots in [BS,(B-1)*BS); two cache pages stay untouched",
    }
    for name, default in (("k_scale", 0.5), ("v_scale", 2.0)):
        inputs[name] = {
            "shape": [],
            "dtype": "torch.float32",
            "value": binding.get(name, default),
        }
    selector = binding.get("selector", "auto" if kind == "auto" else "fp8")
    return {
        "inputs": inputs,
        "kv_cache_dtype": selector,
        "dims": {
            name: str(value) if isinstance(value, torch.dtype) else value
            for name, value in binding.items()
        },
        "format": str(fmt) if kind != "auto" else None,
        "cache_layout": binding["layout"],
        "mutation": [2, 3],
        "return": None,
        "independent_allocations": True,
        "graph": False,
        "poison": "0x55 bytes for FP8/uint8, -113.25 for ordinary dtype",
        "timing": "restore both full cache allocations outside every warmup/event interval",
    }


# Include already-quantized source input in native collection and benchmarks.
DTYPES.append(fp8_dtype())
