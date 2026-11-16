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

"""MetaX values-only top-k/top-p filtering.

Semantics follow vLLM apply_top_k_top_p_pytorch at ee0da84ab. Single rows use
integer-key tile sorting and stable parallel merges. The batched algorithm uses
counting sort for exactly BF16-representable values; other
FP32 inputs use the existing FlagGems radix kernels without token indices.
"""

import importlib
import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _order_key(values):
    raw = values.to(tl.uint32, bitcast=True)
    return tl.where((raw & 0x80000000) != 0, ~raw, raw ^ 0x80000000)


@triton.jit
def _filter_order_key(values):
    # Float comparisons tie signed zeros. Keep the reversible sort key intact.
    raw = values.to(tl.uint32, bitcast=True)
    return tl.where(raw == 0x80000000, 0x80000000, _order_key(values))


@triton.jit
def _fill_zero(ptr, size: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + i, 0, mask=i < size)


@triton.jit
def _count_bf16(logits, counts, invalid, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // tl.cdiv(N, BLOCK)
    col = (pid % tl.cdiv(N, BLOCK)) * BLOCK + tl.arange(0, BLOCK)
    valid = col < N
    value = tl.load(logits + row * N + col, mask=valid, other=0.0)
    bf16 = value.to(tl.bfloat16)
    bad = tl.sum(((bf16.to(tl.float32) != value) & valid).to(tl.int32), 0)
    tl.atomic_max(invalid, bad, sem="relaxed")
    raw = bf16.to(tl.uint16, bitcast=True).to(tl.uint32)
    key = tl.where((raw & 0x8000) != 0, (~raw) & 0xFFFF, raw ^ 0x8000)
    tl.atomic_add(counts + row * 65536 + key, 1, mask=valid, sem="relaxed")


@triton.jit
def _expand_counted(prefix, out, invalid, N: tl.constexpr, BLOCK: tl.constexpr):
    if tl.load(invalid) == 0:
        pid = tl.program_id(0)
        row = pid // tl.cdiv(N, BLOCK)
        pos = (pid % tl.cdiv(N, BLOCK)) * BLOCK + tl.arange(0, BLOCK)
        valid = pos < N
        lo = tl.zeros((BLOCK,), tl.int32)
        hi = tl.full((BLOCK,), 65536, tl.int32)
        for _ in range(17):
            mid = (lo + hi) // 2
            value = tl.load(
                prefix + row * 65536 + mid, mask=valid & (mid < 65536), other=2147483647
            )
            right = value <= pos
            lo = tl.where(right, mid + 1, lo)
            hi = tl.where(right, hi, mid)
        raw = tl.where(lo < 0x8000, (~lo) & 0xFFFF, lo ^ 0x8000)
        values = (raw.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
        tl.store(out + row * N + pos, values, mask=valid)


# Reuse existing FlagGems Triton helpers, not its PyTorch dispatch wrapper.
# Imported only in this MetaX backend; other vendors do not require FlagGems.
_sort_ops = None


@triton.jit
def _guarded_hist(logits, hist, invalid, M: tl.constexpr, N: tl.constexpr):
    if tl.load(invalid) != 0:
        pid = tl.program_id(0)
        row = pid % M
        cols = (pid // M) * 8192 + tl.arange(0, 8192)
        valid = cols < N
        values = tl.load(logits + row * N + cols, mask=valid, other=0.0)
        keys = _order_key(values)
        bins = tl.arange(0, 16)
        for pass_id in tl.static_range(8):
            digit = (keys >> (pass_id * 4)) & 15
            counts = tl.histogram(digit, 16, mask=valid)
            tl.atomic_add(hist + row * 128 + pass_id * 16 + bins, counts, sem="relaxed")


@triton.jit
def _hist_prefix(hist, prefix):
    pid = tl.program_id(0)
    bins = tl.arange(0, 16)
    value = tl.load(hist + pid * 16 + bins)
    tl.store(prefix + pid * 16 + bins, (tl.cumsum(value, 0) - value).to(tl.uint32))


@triton.jit
def _guarded_zero(ptr, invalid, size: tl.constexpr, BLOCK: tl.constexpr):
    if tl.load(invalid) != 0:
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(ptr + i, 0, mask=i < size)


@triton.jit
def _guarded_sweep(
    inp,
    out,
    prefix,
    status,
    invalid,
    M: tl.constexpr,
    N: tl.constexpr,
    PASS: tl.constexpr,
):
    if tl.load(invalid) != 0:
        _sort_ops.sweep(
            inp,
            None,
            out,
            None,
            prefix,
            status,
            8,
            PASS,
            PASS * 4,
            M,
            N,
            tl.cdiv(N, 2048),
            2048,
            8,
            4,
            False,
        )


@libentry()
@triton.jit
def _sort_value_tiles(logits, out, N: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(logits + col, mask=col < N, other=0.0)
    key = tl.where(col < N, _order_key(value), 0xFFFFFFFF).to(tl.uint32)
    key = tl.sort(key, descending=False)
    raw = tl.where((key & 0x80000000) != 0, key ^ 0x80000000, ~key)
    tl.store(out + col, raw.to(tl.float32, bitcast=True), mask=col < N)


@libentry()
@triton.jit
def _merge_value_runs(
    source,
    out,
    N: tl.constexpr,
    RUN: tl.constexpr,
    STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = col < N
    value = tl.load(source + col, mask=valid, other=0.0)
    key = _order_key(value)
    base = (col // (2 * RUN)) * (2 * RUN)
    left = col - base < RUN
    other = base + tl.where(left, RUN, 0)
    length = tl.minimum(tl.maximum(N - other, 0), RUN)
    lo = tl.zeros((BLOCK,), tl.int32)
    hi = length
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        other_value = tl.load(
            source + other + mid, mask=valid & (mid < length), other=0.0
        )
        other_key = _order_key(other_value)
        # Left run wins ties: lower_bound for left, upper_bound for right.
        advance = tl.where(left, other_key < key, other_key <= key) & (mid < length)
        lo = tl.where(advance, mid + 1, lo)
        hi = tl.where(advance, hi, mid)
    dest = base + col % RUN + lo
    tl.store(out + dest, value, mask=valid)


def _sorted_values_single_row(logits):
    n = logits.shape[1]
    block = min(4096, triton.next_power_of_2(n))
    result = torch.empty_like(logits)
    _sort_value_tiles[(triton.cdiv(n, block),)](logits, result, n, block)
    if n <= block:
        return result
    scratch = torch.empty_like(logits)
    run = block
    while run < n:
        _merge_value_runs[(triton.cdiv(n, 256),)](
            result, scratch, n, run, run.bit_length(), 256
        )
        result, scratch = scratch, result
        run *= 2
    return result


def _sorted_values(logits):
    if logits.shape[0] == 1:
        return _sorted_values_single_row(logits)
    global _sort_ops
    import flag_gems

    if _sort_ops is None:
        _sort_ops = importlib.import_module("flag_gems.ops.sort")

    m, n = logits.shape
    counts = torch.empty((m, 65536), dtype=torch.int32, device=logits.device)
    invalid = torch.empty((), dtype=torch.int32, device=logits.device)
    _fill_zero[(triton.cdiv(m * 65536, 1024),)](counts, m * 65536, 1024)
    _fill_zero[(1,)](invalid, 1, 1)
    _count_bf16[(m * triton.cdiv(n, 256),)](logits, counts, invalid, n, 256)
    prefix = flag_gems.cumsum(counts, dim=-1, dtype=torch.int32)

    hist = torch.empty((m, 8, 16), dtype=torch.int32, device=logits.device)
    hist_prefix = torch.empty_like(hist, dtype=torch.uint32)
    _fill_zero[(triton.cdiv(m * 8 * 16, 1024),)](hist, m * 8 * 16, 1024)
    _guarded_hist[(m * triton.cdiv(n, 8192), 1, 1)](logits, hist, invalid, m, n)
    _hist_prefix[(m * 8,)](hist, hist_prefix)
    scratch = torch.empty_like(logits)
    result = torch.empty_like(logits)
    status_size = m * 16 * triton.cdiv(n, 2048)
    status = torch.empty((status_size,), dtype=torch.uint32, device=logits.device)
    current = logits
    for pass_id in range(8):
        out = scratch if pass_id % 2 == 0 else result
        _guarded_zero[(triton.cdiv(status_size, 1024),)](
            status, invalid, status_size, 1024
        )
        _guarded_sweep[(m * triton.cdiv(n, 2048), 2)](
            current, out, hist_prefix, status, invalid, m, n, pass_id
        )
        current = out
    _expand_counted[(m * triton.cdiv(n, 256),)](prefix, result, invalid, n, 256)
    return result


@triton.jit
def _extract_top_k(
    values, k, thresholds, N: tl.constexpr, M: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.arange(0, BLOCK)
    valid = row < M
    ki = tl.load(k + row, mask=valid, other=1)
    x = tl.load(values + row * N + N - ki, mask=valid, other=0.0)
    tl.store(thresholds + row, x, mask=valid)


@triton.jit
def _mask_sorted_top_k(values, thresholds, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // tl.cdiv(N, BLOCK)
    col = (pid % tl.cdiv(N, BLOCK)) * BLOCK + tl.arange(0, BLOCK)
    threshold = tl.load(thresholds + row)
    valid = col < N
    x = tl.load(values + row * N + col, mask=valid, other=0.0)
    tl.store(
        values + row * N + col, tl.where(x < threshold, -float("inf"), x), mask=valid
    )


@triton.jit
def _top_k_only_mask(logits, thresholds, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // tl.cdiv(N, BLOCK)
    col = (pid % tl.cdiv(N, BLOCK)) * BLOCK + tl.arange(0, BLOCK)
    valid = col < N
    x = tl.load(logits + row * N + col, mask=valid, other=0.0)
    threshold = tl.load(thresholds + row)
    tl.store(
        logits + row * N + col, tl.where(x < threshold, -float("inf"), x), mask=valid
    )


@triton.jit
def _find_first_keep(
    cumulative_ptr,
    top_p_ptr,
    first_keep_ptr,
    n: tl.constexpr,
    m: tl.constexpr,
    block_rows: tl.constexpr,
    search_steps: tl.constexpr,
):
    rows = tl.arange(0, block_rows)
    valid = rows < m
    cutoff = 1.0 - tl.load(top_p_ptr + rows, mask=valid, other=1.0)
    lo = tl.zeros((block_rows,), dtype=tl.int32)
    hi = tl.full((block_rows,), n - 1, dtype=tl.int32)
    for _ in range(search_steps):
        mid = (lo + hi) // 2
        value = tl.load(cumulative_ptr + rows * n + mid, mask=valid, other=1.0)
        remove = (value <= cutoff) & (mid < n - 1)
        lo = tl.where(remove, mid + 1, lo)
        hi = tl.where(remove, hi, mid)
    tl.store(first_keep_ptr + rows, lo, mask=valid)


@triton.jit
def _threshold_counts(
    logits_ptr,
    sorted_logits_ptr,
    first_keep_ptr,
    less_counts_ptr,
    equal_counts_ptr,
    n: tl.constexpr,
    block_n: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n: tl.constexpr = tl.cdiv(n, block_n)
    row = pid // grid_n
    cols = (pid % grid_n) * block_n + tl.arange(0, block_n)
    valid = cols < n
    first_keep = tl.load(first_keep_ptr + row)
    threshold = tl.load(sorted_logits_ptr + row * n + first_keep)
    values = tl.load(logits_ptr + row * n + cols, mask=valid, other=0.0)
    count_offset = row * grid_n + pid % grid_n
    tl.store(
        less_counts_ptr + count_offset,
        tl.sum(
            ((_filter_order_key(values) < _filter_order_key(threshold)) & valid).to(
                tl.int32
            ),
            axis=0,
        ),
    )
    tl.store(
        equal_counts_ptr + count_offset,
        tl.sum(
            ((_filter_order_key(values) == _filter_order_key(threshold)) & valid).to(
                tl.int32
            ),
            axis=0,
        ),
    )


@triton.jit
def _threshold_prefix_stats(
    less_counts_ptr,
    equal_counts_ptr,
    total_less_ptr,
    prefix_equal_ptr,
    grid_n: tl.constexpr,
    block_counts: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_counts)
    valid = cols < grid_n
    offsets = row * grid_n + cols
    less = tl.load(less_counts_ptr + offsets, mask=valid, other=0)
    equal = tl.load(equal_counts_ptr + offsets, mask=valid, other=0)
    total_less = tl.sum(less, axis=0)
    exclusive_equal = tl.cumsum(equal, axis=0) - equal
    tl.store(total_less_ptr + row, total_less)
    tl.store(prefix_equal_ptr + offsets, exclusive_equal, mask=valid)


@triton.jit
def _stable_threshold_mask(
    logits_ptr,
    sorted_logits_ptr,
    first_keep_ptr,
    less_counts_ptr,
    equal_counts_ptr,
    n: tl.constexpr,
    block_n: tl.constexpr,
    count_block: tl.constexpr,
    top_k_threshold_ptr,
    has_k: tl.constexpr,
    total_less_ptr,
    prefix_equal_ptr,
    use_prefix: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n: tl.constexpr = tl.cdiv(n, block_n)
    chunk = pid % grid_n
    row = pid // grid_n
    cols = chunk * block_n + tl.arange(0, block_n)
    valid = cols < n
    first_keep = tl.load(first_keep_ptr + row)
    threshold = tl.load(sorted_logits_ptr + row * n + first_keep)

    if use_prefix:
        total_less = tl.load(total_less_ptr + row)
        prior_equal = tl.load(prefix_equal_ptr + row * grid_n + chunk)
    else:
        count_cols = tl.arange(0, count_block)
        count_valid = count_cols < grid_n
        count_offsets = row * grid_n + count_cols
        less_counts = tl.load(
            less_counts_ptr + count_offsets, mask=count_valid, other=0
        )
        equal_counts = tl.load(
            equal_counts_ptr + count_offsets, mask=count_valid, other=0
        )
        total_less = tl.sum(less_counts, axis=0)
        prior_equal = tl.sum(
            tl.where(count_valid & (count_cols < chunk), equal_counts, 0), axis=0
        )
    tie_remove_count = first_keep - total_less

    offsets = row * n + cols
    values = tl.load(logits_ptr + offsets, mask=valid)
    is_equal = (_filter_order_key(values) == _filter_order_key(threshold)) & valid
    local_equal_rank = tl.cumsum(is_equal.to(tl.int32), axis=0) - 1
    remove_equal = is_equal & (prior_equal + local_equal_rank < tie_remove_count)
    remove = (_filter_order_key(values) < _filter_order_key(threshold)) | remove_equal
    if has_k:
        top_k_threshold = tl.load(top_k_threshold_ptr + row)
        remove = remove | (values < top_k_threshold)
    tl.store(
        logits_ptr + offsets,
        tl.where(remove, -float("inf"), values),
        mask=valid,
    )


def top_k_top_p(
    logits: torch.Tensor,
    p: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
    *,
    topk_len_override: int | None = None,
    p_is_one_override: bool | None = None,
) -> torch.Tensor:
    """Return stable top-k/top-p filtered logits, preserving the input.

    The argument order and out-of-place result match the Ascend API. Its
    two optional launch overrides are explicitly unsupported on MetaX.

    Args:
        logits: Contiguous [batch, vocab] FP32 device tensor; forward only.
        k: Optional contiguous int32/int64 [batch], with 1 <= k <= vocab (validated by the caller).
        p: Optional contiguous FP32 [batch], with 0 <= p <= 1 (validated by the caller).

    Returns:
        A new tensor, with excluded entries replaced by -inf. The input
        logits and filter metadata remain unchanged.

    Raises:
        NotImplementedError: Unsupported metadata; no torch compute fallback.

    Numerically equal logits, including signed zeros, are filtered in
    increasing token index. At the nucleus cutoff, later tokens win ties.
    vLLM does not specify a stable token order for tied logits; this
    implementation makes that order explicit.
    FP32 softmax/cumsum rounding can change the mask at a nucleus boundary
    compared with another reduction implementation.
    Filter values are not read back to the host for range validation.

    Single rows use integer-key tile sorting and stable merges. Larger
    batches select counting sort with a device-side exact BF16 check, or
    values-only radix sorting otherwise. All paths support graph capture
    and replay without synchronizing filter values with the host.
    """
    logger.debug("GEMS_METAX TOP_K_TOP_P")
    if k is None and p is None:
        raise ValueError("p and k cannot both be None")
    if topk_len_override is not None or p_is_one_override is not None:
        raise NotImplementedError("launch overrides are only supported by Ascend")
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise NotImplementedError("requires rank-2 FP32 logits")
    m, n = logits.shape
    if not logits.is_contiguous() or m > 64 or n > 262144 or n == 0:
        raise NotImplementedError("requires contiguous M<=64 and 1<=V<=262144")
    if logits.device.type != "cuda":
        raise NotImplementedError("requires MetaX CUDA-compatible tensors")
    for value, dtypes in ((k, (torch.int32, torch.int64)), (p, (torch.float32,))):
        if value is not None and (
            value.shape != (m,)
            or value.device != logits.device
            or value.dtype not in dtypes
            or not value.is_contiguous()
        ):
            raise NotImplementedError("unsupported filter metadata")
    if m == 0:
        return logits.clone()
    import flag_gems

    with torch_device_fn.device(logits.device):
        logits = logits.clone()
        sorted_logits = _sorted_values(logits)
        thresholds = torch.empty((m,), dtype=torch.float32, device=logits.device)
        grid_n = triton.cdiv(n, 256)
        grid = (m * grid_n,)
        if k is not None:
            _extract_top_k[(1,)](
                sorted_logits, k, thresholds, n, m, triton.next_power_of_2(m)
            )
            _mask_sorted_top_k[grid](sorted_logits, thresholds, n, 256)
        if p is None:
            _top_k_only_mask[grid](logits, thresholds, n, 256)
            return logits
        cumulative = flag_gems.softmax(sorted_logits, dim=-1, half_to_float=False)
        flag_gems.cumsum_out(cumulative, dim=-1, out=cumulative)
        first_keep = torch.empty((m,), dtype=torch.int32, device=logits.device)
        _find_first_keep[(1,)](
            cumulative,
            p,
            first_keep,
            n,
            m,
            triton.next_power_of_2(m),
            max(1, (n - 1).bit_length()),
        )
        less = torch.empty((m, grid_n), dtype=torch.int32, device=logits.device)
        equal = torch.empty_like(less)
        _threshold_counts[grid](logits, sorted_logits, first_keep, less, equal, n, 256)
        # Precomputing row statistics avoids repeated O(grid_n) reductions
        # in every mask CTA. Keep small batches/vocabs on the original path:
        # the extra launch and workspace allocations otherwise cost more.
        use_prefix = m >= 32 and n >= 131072
        total_less = prefix_equal = None
        if use_prefix:
            total_less = torch.empty((m,), dtype=torch.int32, device=logits.device)
            prefix_equal = torch.empty_like(equal)
            _threshold_prefix_stats[(m,)](
                less,
                equal,
                total_less,
                prefix_equal,
                grid_n,
                triton.next_power_of_2(grid_n),
                num_warps=4,
            )
        _stable_threshold_mask[grid](
            logits,
            sorted_logits,
            first_keep,
            less,
            equal,
            n,
            256,
            triton.next_power_of_2(grid_n),
            thresholds,
            k is not None,
            total_less,
            prefix_equal,
            use_prefix,
        )
    return logits
