"""Narrow SM90 W8A8 INT8 decode API; unsupported inputs raise."""

from functools import lru_cache

import torch
import triton.language as tl

from flaggems_vllm.ops.triton_scaled_mm import scaled_mm_kernel


@lru_cache(maxsize=16)
def _capability(device_index):
    # Hardware metadata only; warmup primes this before FULL capture.
    return torch.cuda.get_device_capability(device_index)


def _require(condition, message, kind=ValueError):
    if not condition:
        raise kind(message)


def int8_scaled_mm_decode(a, b, scale_a, scale_b, out_dtype=torch.bfloat16, bias=None):
    """INT8[M,1024] @ column-major INT8[1024,6144], M1..64 -> BF16.

    Dynamic per-token FP32 scale_a[M,1], channel FP32 scale_b[6144,1],
    no bias. Fresh contiguous output. No streams, owner cache or fallback.
    Native SM90 epilogue order is scale_b*INT32acc -> scale_a -> BF16.
    """
    _require(
        all(isinstance(v, torch.Tensor) for v in (a, b, scale_a, scale_b)),
        "four tensors required",
        TypeError,
    )
    _require(
        a.ndim == b.ndim == scale_a.ndim == scale_b.ndim == 2,
        "2D operands/scales required",
    )
    m, k = a.shape
    _require(
        1 <= m <= 64 and k == 1024 and b.shape == (1024, 6144),
        "unsupported INT8 decode shape",
        NotImplementedError,
    )
    _require(
        a.dtype == b.dtype == torch.int8
        and scale_a.dtype == scale_b.dtype == torch.float32
        and out_dtype == torch.bfloat16
        and bias is None,
        "INT8/FP32/BF16 unbiased contract required",
        TypeError,
    )
    _require(
        all(v.is_cuda and v.device == a.device for v in (a, b, scale_a, scale_b)),
        "one CUDA device required",
        NotImplementedError,
    )
    _require(
        torch.version.hip is None and torch.version.cuda is not None,
        "NVIDIA CUDA component only",
        NotImplementedError,
    )
    _require(
        a.stride() == (1024, 1) and b.stride() == (1, 1024),
        "native A/B strides required",
    )
    _require(
        scale_a.shape == (m, 1)
        and scale_b.shape == (6144, 1)
        and scale_a.stride() == scale_b.stride() == (1, 1),
        "native per-token/channel scale layout required",
    )
    _require(
        all(v.storage_offset() == 0 for v in (a, b, scale_a, scale_b)),
        "zero storage offsets required",
    )
    _require(
        _capability(a.device.index) == (9, 0),
        "SM90 component only",
        NotImplementedError,
    )
    out = torch.empty((m, 6144), dtype=out_dtype, device=a.device)
    scaled_mm_kernel[(96,)](
        a,
        b,
        scale_a,
        scale_b,
        out,
        None,
        m,
        6144,
        1024,
        1024,
        1,
        1,
        1024,
        6144,
        1,
        tl.int32,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=256,
        BLOCK_SIZE_SCALE_A=64,
        BLOCK_SIZE_SCALE_B=64,
        SCALE_B_FIRST=True,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["int8_scaled_mm_decode"]
