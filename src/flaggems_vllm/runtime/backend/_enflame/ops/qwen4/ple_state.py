# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-vendor row I/O kernels for the Qwen4 PLE short-conv state.

The short-conv cache is exposed as a non-contiguous logical
``[rows, hidden, width]`` view of physical ``[rows, width, hidden]`` storage.
FlagGems 5.3/5.4 materializes that whole multi-GiB view for ``index_select``.
These public entries therefore use the compact stride-aware Triton kernel on
an accelerator satisfying the guard and fail closed elsewhere. Only NVIDIA
H100 has current correctness/performance evidence for this extraction.

The scatter write mask supplied by PLE prevents NULL/padded rows from writing
the reserved row zero; out-of-range rows are ignored.  The caller must supply
that mask explicitly so the original ``-1`` index remains distinguishable from
valid row zero.  Duplicate destinations are supported when the mask keeps only
the desired (normally last) writer.  There is no Torch compute fallback.

GCU adaptation (enflame backend copy):
* int64 layout gate. torch_gcu stores device int64 tensors as compact 4-byte
  payload words unless ``TORCH_GCU_ENABLE_INT64_AND_UINT64`` is exactly
  ``"1"``/``"true"``, in which case the true 8-byte layout is active
  (compiler_issues/triton_i64_narrow_load.md). The index load therefore runs
  at the declared int64 width only under the true layout (``I64_TRUE``) and
  falls back to the proven scalar int32 pointer bitcast for the compact
  layout, so both environments read the intended row indices; the loaded
  value is widened to int64 only for the scalar row-base address. Row indices
  stay in the i32 safe domain (< 2**31).
* Affine fast path. When the inner strides are the natural ones for the
  enumeration orientation (``stride(1) == 1 and stride(2) == hidden`` when
  hidden is fastest, or the mirror for width-fastest), the per-element
  ``%``/``//`` decomposition and three-term i64 vector addressing collapse to
  a flat ``row_base + offsets``, which both removes the vector ALU and lets
  the backend see contiguous row segments (measured 1.10-1.34x on the core
  gather shapes, GCU400).
* Adaptive block. ``BLOCK_SIZE`` follows ``next_power_of_2(row_elements)``
  capped at 1024 instead of a fixed 1024, cutting the masked-out lane waste
  for the 195-element PLE rows to 256 lanes.
* Row batching. GCU400 shows a sharp wave-contention penalty once the row
  grid exceeds the 48-program band, so batches larger than 48 rows launch
  two rows per program (``ROWS_PER_PROG=2``): 64-row gather 7.2 -> 4.2us and
  64-row scatter 7.3 -> 5.8us (paired same-process do_bench, GCU400).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

HAS_TRITON = True

_BLOCK_CAP = 1024
_ROWS_BATCH_THRESHOLD = 48
_ROWS_PER_PROG = 2


def _gcu_real_int64_storage() -> bool:
    # torch_gcu narrows device int64 tensors to 4-byte payload words unless
    # TORCH_GCU_ENABLE_INT64_AND_UINT64 is exactly "1"/"true" (the true 8-byte
    # layout); calibrated in compiler_issues/triton_i64_narrow_load.md.
    return os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") in ("1", "true")


def _is_triton_device(*tensors: torch.Tensor) -> bool:
    return bool(
        HAS_TRITON
        and tensors
        and len({tensor.device for tensor in tensors}) == 1
        and all(tensor.device.type not in ("cpu", "meta") for tensor in tensors)
    )


if HAS_TRITON:

    @triton.jit
    def _ple_state_gather_kernel_3d(
        state_ptr,
        indices_ptr,
        output_ptr,
        indices_stride,
        state_stride0,
        state_stride1,
        state_stride2,
        output_stride0,
        output_stride1,
        output_stride2,
        num_cache_rows,
        num_rows,
        hidden_size: tl.constexpr,
        state_width: tl.constexpr,
        HIDDEN_FASTEST: tl.constexpr,
        INNER_AFFINE: tl.constexpr,
        I64_TRUE: tl.constexpr,
        ROWS_PER_PROG: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < hidden_size * state_width
        for i in tl.static_range(ROWS_PER_PROG):
            row = pid * ROWS_PER_PROG + i
            if row < num_rows:
                if I64_TRUE:
                    index = tl.load(indices_ptr + row * indices_stride).to(tl.int64)
                else:
                    slot_ptr = indices_ptr.to(tl.pointer_type(tl.int32))
                    index = tl.load(slot_ptr + row * indices_stride).to(tl.int64)
                valid_index = (index >= 0) & (index < num_cache_rows)
                safe_index = tl.minimum(tl.maximum(index, 0), num_cache_rows - 1)
                if INNER_AFFINE:
                    source = state_ptr + safe_index * state_stride0 + offsets
                    destination = output_ptr + row * output_stride0 + offsets
                else:
                    if HIDDEN_FASTEST:
                        hidden = offsets % hidden_size
                        width = offsets // hidden_size
                    else:
                        hidden = offsets // state_width
                        width = offsets % state_width
                    source = (
                        state_ptr
                        + safe_index * state_stride0
                        + hidden * state_stride1
                        + width * state_stride2
                    )
                    destination = (
                        output_ptr
                        + row * output_stride0
                        + hidden * output_stride1
                        + width * output_stride2
                    )
                values = tl.load(source, mask=valid & valid_index, other=0)
                tl.store(destination, values, mask=valid)

    @triton.jit
    def _ple_state_scatter_kernel_3d(
        state_ptr,
        indices_ptr,
        rows_ptr,
        indices_stride,
        write_mask_ptr,
        write_mask_stride,
        state_stride0,
        state_stride1,
        state_stride2,
        rows_stride0,
        rows_stride1,
        rows_stride2,
        num_cache_rows,
        num_rows,
        hidden_size: tl.constexpr,
        state_width: tl.constexpr,
        HIDDEN_FASTEST: tl.constexpr,
        DST_AFFINE: tl.constexpr,
        I64_TRUE: tl.constexpr,
        ROWS_PER_PROG: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < hidden_size * state_width
        for i in tl.static_range(ROWS_PER_PROG):
            row = pid * ROWS_PER_PROG + i
            if row < num_rows:
                row_valid = valid & tl.load(write_mask_ptr + row * write_mask_stride)
                if I64_TRUE:
                    index = tl.load(indices_ptr + row * indices_stride).to(tl.int64)
                else:
                    slot_ptr = indices_ptr.to(tl.pointer_type(tl.int32))
                    index = tl.load(slot_ptr + row * indices_stride).to(tl.int64)
                row_valid &= (index >= 0) & (index < num_cache_rows)
                safe_index = tl.minimum(tl.maximum(index, 0), num_cache_rows - 1)
                if HIDDEN_FASTEST:
                    hidden = offsets % hidden_size
                    width = offsets // hidden_size
                else:
                    hidden = offsets // state_width
                    width = offsets % state_width
                source = (
                    rows_ptr
                    + row * rows_stride0
                    + hidden * rows_stride1
                    + width * rows_stride2
                )
                if DST_AFFINE:
                    destination = state_ptr + safe_index * state_stride0 + offsets
                else:
                    destination = (
                        state_ptr
                        + safe_index * state_stride0
                        + hidden * state_stride1
                        + width * state_stride2
                    )
                values = tl.load(source, mask=row_valid, other=0)
                tl.store(destination, values, mask=row_valid)

else:  # pragma: no cover - exercised by import-only/CPU environments
    _ple_state_gather_kernel_3d = None
    _ple_state_scatter_kernel_3d = None


def _validate_state_tensor(state: torch.Tensor, name: str) -> None:
    if state.ndim != 3:
        raise ValueError(f"PLE {name} must be a rank-3 [rows, hidden, width] tensor")
    if state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"PLE {name} must use float16, bfloat16, or float32; got {state.dtype}"
        )


def _empty_state_rows_like(state: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Allocate compact rows with the same dense inner layout as ``state``."""

    if state.stride(1) <= state.stride(2):
        return torch.empty(
            (num_rows, state.shape[2], state.shape[1]),
            dtype=state.dtype,
            device=state.device,
        ).transpose(1, 2)
    return torch.empty(
        (num_rows, state.shape[1], state.shape[2]),
        dtype=state.dtype,
        device=state.device,
    )


def _inner_affine(
    tensor: torch.Tensor,
    hidden_size: int,
    state_width: int,
    hidden_fastest: bool,
) -> bool:
    """True when the inner strides make the flat enumeration a row offset."""

    stride1, stride2 = tensor.stride(1), tensor.stride(2)
    if hidden_fastest:
        return stride1 == 1 and stride2 == hidden_size
    return stride2 == 1 and stride1 == state_width


def _launch_geometry(row_elements: int, num_rows: int):
    block_size = min(_BLOCK_CAP, triton.next_power_of_2(row_elements))
    num_warps = max(1, block_size // 128)
    rows_per_prog = _ROWS_PER_PROG if num_rows > _ROWS_BATCH_THRESHOLD else 1
    grid = (
        triton.cdiv(num_rows, rows_per_prog),
        triton.cdiv(row_elements, block_size),
    )
    return block_size, num_warps, rows_per_prog, grid


def ple_state_gather(
    state: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor | None = None,
    *,
    indices_are_safe: bool = False,
) -> torch.Tensor:
    """Gather rows; original ``-1``/out-of-range indices produce zero rows.

    ``indices_are_safe`` is retained for call-site compatibility; it never
    bypasses the original-index validity check.
    """

    _validate_state_tensor(state, "state")
    if not _is_triton_device(state, indices):
        raise RuntimeError("Qwen4 PLE gather requires a Triton accelerator")
    if indices.ndim != 1 or indices.device != state.device:
        raise ValueError(
            "PLE gather indices must be a one-dimensional same-device tensor"
        )
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("PLE gather indices must be int32 or int64")
    if state.shape[0] == 0:
        raise ValueError("PLE gather requires at least one cache row")
    expected_shape = (indices.numel(),) + tuple(state.shape[1:])
    if output is not None:
        _validate_state_tensor(output, "state output")
        if output.shape != expected_shape:
            raise ValueError(
                "PLE state gather output has an invalid shape: "
                f"got {tuple(output.shape)}, expected {expected_shape}"
            )
        if output.device != state.device or output.dtype != state.dtype:
            raise ValueError(
                "PLE state gather output must match state device and dtype"
            )

    if not indices.numel() or state.shape[1] == 0 or state.shape[2] == 0:
        if output is None:
            output = _empty_state_rows_like(state, indices.numel())
        return output
    if output is None:
        output = _empty_state_rows_like(state, indices.numel())
    hidden_size = state.shape[1]
    state_width = state.shape[2]
    row_elements = hidden_size * state_width
    block_size, num_warps, rows_per_prog, grid = _launch_geometry(
        row_elements, indices.numel()
    )
    hidden_fastest = state.stride(1) <= state.stride(2)
    inner_affine = _inner_affine(
        state, hidden_size, state_width, hidden_fastest
    ) and _inner_affine(output, hidden_size, state_width, hidden_fastest)
    _ple_state_gather_kernel_3d[grid](
        state,
        indices,
        output,
        indices.stride(0),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        state.shape[0],
        indices.numel(),
        hidden_size=hidden_size,
        state_width=state_width,
        HIDDEN_FASTEST=hidden_fastest,
        INNER_AFFINE=inner_affine,
        I64_TRUE=indices.dtype == torch.int64 and _gcu_real_int64_storage(),
        ROWS_PER_PROG=rows_per_prog,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def ple_state_scatter_(
    state: torch.Tensor,
    indices: torch.Tensor,
    rows: torch.Tensor,
    *,
    write_mask: torch.Tensor | None = None,
    indices_are_safe: bool = False,
) -> torch.Tensor:
    """Write rows through an explicit NULL/padding/duplicate write mask.

    ``indices_are_safe`` is retained for call-site compatibility; it never
    bypasses the original-index validity check.
    """

    _validate_state_tensor(state, "state")
    _validate_state_tensor(rows, "state rows")
    if not _is_triton_device(state, indices, rows):
        raise RuntimeError("Qwen4 PLE scatter requires a Triton accelerator")
    if indices.ndim != 1 or indices.device != state.device:
        raise ValueError(
            "PLE scatter indices must be a one-dimensional same-device tensor"
        )
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("PLE scatter indices must be int32 or int64")
    if state.shape[0] == 0:
        raise ValueError("PLE scatter requires at least one cache row")
    if rows.shape[0] != indices.numel() or tuple(rows.shape[1:]) != tuple(
        state.shape[1:]
    ):
        raise ValueError(
            "PLE state scatter rows and indices have incompatible shapes: "
            f"rows={tuple(rows.shape)}, indices={indices.numel()}, "
            f"state={tuple(state.shape)}"
        )
    if rows.device != state.device or rows.dtype != state.dtype:
        raise ValueError("PLE state scatter rows must match state device and dtype")

    if write_mask is None:
        raise NotImplementedError(
            "Qwen4 PLE scatter requires an explicit write_mask; "
            "derive duplicate/null semantics in the caller"
        )
    if write_mask.ndim != 1 or write_mask.numel() != indices.numel():
        raise ValueError("PLE state scatter write_mask must match indices")
    if write_mask.device != state.device or write_mask.dtype != torch.bool:
        raise ValueError("PLE state scatter write_mask must be same-device bool")

    if not indices.numel() or state.shape[1] == 0 or state.shape[2] == 0:
        return state

    hidden_size = state.shape[1]
    state_width = state.shape[2]
    row_elements = hidden_size * state_width
    block_size, num_warps, rows_per_prog, grid = _launch_geometry(
        row_elements, indices.numel()
    )
    hidden_fastest = state.stride(1) <= state.stride(2)
    _ple_state_scatter_kernel_3d[grid](
        state,
        indices,
        rows,
        indices.stride(0),
        write_mask,
        write_mask.stride(0),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        rows.stride(0),
        rows.stride(1),
        rows.stride(2),
        state.shape[0],
        indices.numel(),
        hidden_size=hidden_size,
        state_width=state_width,
        HIDDEN_FASTEST=hidden_fastest,
        DST_AFFINE=_inner_affine(state, hidden_size, state_width, hidden_fastest),
        I64_TRUE=indices.dtype == torch.int64 and _gcu_real_int64_storage(),
        ROWS_PER_PROG=rows_per_prog,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return state


_PUBLIC_OPS = ("ple_state_gather", "ple_state_scatter_")


def _install() -> None:
    """Swap the GCU-adapted entry points into the generic module.

    Tests and benchmarks import ``from flaggems_vllm.ops.qwen4.ple_state
    import ...``, which resolves against the already-imported generic
    module, so the SpecOpRegistrar top-level namespace swap alone cannot
    reach them. Replace the attributes on that module in place, mirroring
    the kpool backend override precedent: only a pristine generic function
    is swapped, so explicit user or test patches always win.
    """
    import sys

    mod = sys.modules.get("flaggems_vllm.ops.qwen4.ple_state")
    if mod is None:
        return
    for name in _PUBLIC_OPS:
        current = getattr(mod, name, None)
        if current is not None and getattr(current, "__module__", None) == (
            "flaggems_vllm.ops.qwen4.ple_state"
        ):
            setattr(mod, name, globals()[name])


try:
    _install()
except Exception:
    # pytest assertion-rewrite 上下文中 @triton.jit 源定位可能失败
    # (collect 期动态触碰 generic 模块触发);跳过接线不影响 op 路由
    # (SpecOpRegistrar 已保证),直连 import 入口的接线由运行期重试兜底。
    pass
__all__ = ["ple_state_gather", "ple_state_scatter_"]
