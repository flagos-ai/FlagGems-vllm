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

"""Host dispatch for MixedQuantSparseFlashMLA on Ascend 950DT (A5).

The gather dominates the cost, so the dispatch answers two questions: how many
heads one program should sweep, and whether the resulting task count already
fills the cores.  The gather always gets its own pass.

Because all N1 heads share one KV tile under MLA absorb, the head block is also
the tile re-read factor: BLOCK_H=16 with N1=64 makes the attention read every
token's tile four times.  So the dispatch takes the widest head block that
still keeps the grid busy, and splits K only when none of them does.

    widest BLOCK_H with tasks >= num_cores   gather, contiguous attention
    else widest BLOCK_H whose split stays    gather, split-K attention, combine
      >= MIN_SPLIT_DEPTH tokens deep
    else narrowest BLOCK_H, split as far     gather, split-K attention, combine
      as it goes

where ``tasks = T1 * N1 / BLOCK_H``.  At N1=64 on 32 cores this picks
BLOCK_H=64 for T1 >= 32, 32 at T1=16, 16 at T1=8, and split-K below that.  The
depth floor earns its place: at T1=2 a 32-head block needs 8 splits of 256
tokens and costs 32.5us, while a 16-head block with 4 splits of 512 costs
28.4us.  Every configuration this rule picks is within 1% of the best of the
60 that a ``block_h x path x splits`` sweep covers.

``splits`` is always a whole number of ``block_s`` tiles, which keeps the main
kernel's inner loop unmasked.

Measured on a 950DT against ``aclnnMixedQuantSparseFlashMla`` given the same
sparse indices, at K=2048, N1=64, D=512, seqlen_kv=8192, block_size=64, as
device kernel time per call in us (``do_bench_npu``, which is what
``benchmark/`` reports in its default ``--mode kernel``)::

    T1             1      2      4      8     16     32
    ours        25.2   28.5   39.7   60.7   92.2  159.7
      gather     9.3   10.1   16.6   29.6   55.5  108.8
      attention 16.1   18.1   22.9   30.9   36.6   50.3
    vendor      41.8   42.3   42.3   42.4   44.0   43.7
    speedup     1.66x  1.48x  1.07x  0.70x  0.48x  0.27x

where speedup is vendor/ours, so above 1 is faster than the vendor.

The vendor is flat in T1 because it parallelizes rows across cores and 32 rows
still fit, in one kernel that reads the pages once.  This op wins up to T1=4,
where the vendor leaves cores idle, and loses above it on the gather, which is
68% of kernel time at T1=32.

That 108.8us is close to the floor this KV layout allows.  A 544B row straddles
two 512B HBM granules, so each gathered token costs two accesses no matter how
the kernel asks for it, and the measurements agree: dropping either the 512B
data load or the 32B scale load leaves 58.3us, exactly half, even though their
byte counts differ 16-fold.  Three ways to spend less all measured worse or
flat, at T1=32:

  * merging the two reads into one 1024B load.  Still two granules, and 102.7us
    against 67.9us for a single 512B load at the same tile, so wider costs
    more rather than less.
  * sorting or clustering the sparse indices.  109.9us uniform, 108.3us sorted,
    108.2us for whole contiguous pages: the cost follows the number of granule
    accesses, not their locality.
  * a wider gather grid.  64 programs is the plateau and the curve is a tail
    effect, not bandwidth: 128, 192 and 256 programs all land at 107.9us, while
    80 and 96 cost 165.1 and 140.7us because the last partial wave stalls the
    kernel.

Removing the 128 MB staging round trip itself is the only remaining lever, and
both forms of it measure worse:

  * staging fp8 and dequantizing inside the attention.  The gather drops to
    63.7us, but the attention rises from 50.0us to 120.8us: the dequant wants
    the gather's 64 Vector programs, and in the attention it runs on a
    32-program Cube-bound grid inside the matmul dependency chain.
  * the fused ``scattered`` kernels, which read the pages in place and stay
    reachable through ``path=``.  639us at the best of eight tiles, and
    identical at 32 and 64 programs, so they are bound per program rather than
    on memory parallelism.  BLOCK_S=128 and BLOCK_H=64 do not compile there, so
    fusion also cannot use the wide head block that makes the split pipeline
    fast.  At T1=1 ``scattered_splitk`` is the cheaper path in wall-clock only,
    where it saves one launch (115us against 135us per call); under graph
    capture that launch is replayed rather than issued from Python, and the
    kernel time is what remains.
"""

import math

import torch

from flaggems_vllm.runtime.backend._ascend.a5.ops.mixed_quant_sparse_flash_mla.attention import (  # noqa: E501
    contig_attn_kernel,
    contig_splitk_main_kernel,
    splitk_combine_kernel,
)
from flaggems_vllm.runtime.backend._ascend.a5.ops.mixed_quant_sparse_flash_mla.common import (  # noqa: E501
    BLOCK_H,
    BLOCK_S,
    CMP_KV_ROW_BYTES,
    GATHER_CORES,
    GROUP_SIZE,
    MAX_SPLITK_BLOCK_H,
    MIN_SPLIT_DEPTH,
    MULTIBUFFER_OPTS,
    N_GROUPS,
    NUM_CORES,
    ROW_STRIDE_DATA,
    ROW_STRIDE_SCALE,
    D,
    head_blocks,
    log2_block_size,
    ori_cache_views,
    pick_block_h,
    pick_splits,
    validate_ori_sparse_args,
)
from flaggems_vllm.runtime.backend._ascend.a5.ops.mixed_quant_sparse_flash_mla.gather import (  # noqa: E501
    gather_dequant_kernel,
)
from flaggems_vllm.runtime.backend._ascend.a5.ops.mixed_quant_sparse_flash_mla.scattered import (  # noqa: E501
    scattered_attn_kernel,
    scattered_splitk_main_kernel,
)

__all__ = [
    "MixedQuantSparseFlashMlaPlan",
    "PATHS",
    "mixed_quant_sparse_flash_mla",
]

PATHS = (
    "auto",
    "gather_contig",
    "gather_contig_splitk",
    "scattered_splitk",
    "scattered",
)

_SPLITK_PATHS = ("gather_contig_splitk", "scattered_splitk")
_GATHER_PATHS = ("gather_contig", "gather_contig_splitk")

# A contiguous attention tile of 128 tokens halves the loop trip count over the
# 64 the gather uses; the gather cannot follow because [128, 544] of staged FP8
# plus its bf16 expansion overflows the UB.
_ATTN_BLOCK_S = 128


def _choose_dispatch(t1, n1, k_total, num_cores, block_s, block_h=None):
    """Return ``(path, splits, block_h)`` for the measured dispatch above.

    ``block_h=None`` lets the widest head block win; passing one pins it and
    only the path and split count are chosen.
    """
    candidates = head_blocks(n1) if block_h is None else (int(block_h),)

    # Widest head block that fills the grid on its own.  A split would only
    # add a combine, and a narrower block would re-read the KV tile.
    for bh in candidates:
        if t1 * (n1 // bh) >= num_cores:
            return "gather_contig", 1, bh

    # Nothing fills it, so K has to be split.  Widening the head block costs
    # splits, which is only worth it while each split stays deep.
    for bh in candidates:
        if bh > MAX_SPLITK_BLOCK_H:
            continue
        splits = pick_splits(num_cores, t1 * (n1 // bh), k_total, block_s)
        if splits > 1 and k_total // splits >= MIN_SPLIT_DEPTH:
            return "gather_contig_splitk", splits, bh

    # No split is both deep and wide: take the narrowest block, which needs the
    # fewest splits to fill the grid, and accept whatever depth that leaves.
    bh = min(candidates)
    splits = pick_splits(num_cores, t1 * (n1 // bh), k_total, block_s)
    if splits > 1:
        return "gather_contig_splitk", splits, bh
    return "gather_contig", 1, bh


class MixedQuantSparseFlashMlaPlan:
    """A compiled dispatch plus the workspaces for one decode shape.

    Holding the plan across steps keeps the staging buffer, the split-K
    partials and the strided KV views allocated, which matters because the
    kernels themselves run in tens of microseconds.  The plan is bound to the
    shape it was built from and rejects a differently shaped call.
    """

    def __init__(
        self,
        q,
        *,
        ori_kv,
        ori_sparse_indices,
        ori_block_table,
        ori_topk_length=None,
        sinks=None,
        softmax_scale=None,
        return_softmax_lse=False,
        num_cores=NUM_CORES,
        gather_cores=GATHER_CORES,
        block_h=None,
        block_s=BLOCK_S,
        splits=None,
        path="auto",
    ):
        if block_h is not None:
            block_h = int(block_h)
        validate_ori_sparse_args(
            q,
            ori_kv,
            ori_sparse_indices,
            ori_block_table,
            ori_topk_length,
            sinks,
            block_h if block_h is not None else pick_block_h(q.shape[1], BLOCK_H),
            block_s,
        )
        if path not in PATHS:
            raise ValueError(f"path must be one of {PATHS}, got {path!r}")
        if num_cores < 1:
            raise ValueError(f"num_cores must be positive, got {num_cores}")

        self.t1, self.n1 = q.shape[0], q.shape[1]
        self.k_total = ori_sparse_indices.shape[1]
        self.block_size = ori_kv.shape[1]
        self.log2_block_size = log2_block_size(self.block_size)
        self.max_blocks = ori_block_table.shape[1]
        self.device = q.device
        self.has_topk = ori_topk_length is not None
        self.has_sinks = sinks is not None
        self.return_softmax_lse = bool(return_softmax_lse)
        self.softmax_scale = (
            1.0 / math.sqrt(D) if softmax_scale is None else float(softmax_scale)
        )
        self.num_cores = int(num_cores)
        self.gather_cores = int(gather_cores)
        self.block_s = int(block_s)

        auto_path, auto_splits, auto_block_h = _choose_dispatch(
            self.t1, self.n1, self.k_total, self.num_cores, self.block_s, block_h
        )
        self.block_h = auto_block_h
        self.n_head_groups = self.n1 // self.block_h
        self.path = auto_path if path == "auto" else path
        if splits is not None:
            self.splits = int(splits)
        elif self.path == auto_path:
            self.splits = auto_splits
        else:
            self.splits = pick_splits(
                self.num_cores,
                self.t1 * self.n_head_groups,
                self.k_total,
                self.block_s,
            )

        if self.path in _SPLITK_PATHS and self.block_h > MAX_SPLITK_BLOCK_H:
            raise NotImplementedError(
                f"block_h={self.block_h} does not fit the split-K main kernel, "
                f"which carries a [block_h, {D}] fp32 partial in UB; "
                f"the limit is {MAX_SPLITK_BLOCK_H}"
            )

        if self.path in _SPLITK_PATHS:
            if self.splits < 1:
                raise ValueError(f"splits must be positive, got {self.splits}")
            if self.k_total % self.splits != 0:
                raise ValueError(
                    f"K={self.k_total} must be divisible by splits={self.splits}"
                )
            k_per_split = self.k_total // self.splits
            if k_per_split % self.block_s != 0:
                raise ValueError(
                    f"K/splits={k_per_split} must be a multiple of "
                    f"block_s={self.block_s}"
                )
        else:
            self.splits = 1

        # The 128-token attention tile needs K to be a whole number of tiles.
        self.attn_block_s = (
            _ATTN_BLOCK_S if self.k_total % _ATTN_BLOCK_S == 0 else self.block_s
        )

        # Workspaces, allocated once per plan.
        if self.path in _GATHER_PATHS:
            self.deq_buffer = torch.empty(
                self.t1, self.k_total, D, dtype=torch.bfloat16, device=self.device
            )
            self.valid_buffer = torch.empty(
                self.t1, self.k_total, dtype=torch.int8, device=self.device
            )
        else:
            self.deq_buffer = None
            self.valid_buffer = None

        if self.path in _SPLITK_PATHS:
            partial_shape = (self.t1, self.n_head_groups, self.splits, self.block_h)
            self.ws_acc = torch.empty(
                *partial_shape, D, dtype=torch.float32, device=self.device
            )
            self.ws_m = torch.empty(
                *partial_shape, dtype=torch.float32, device=self.device
            )
            self.ws_l = torch.empty(
                *partial_shape, dtype=torch.float32, device=self.device
            )
        else:
            self.ws_acc = None
            self.ws_m = None
            self.ws_l = None

        # Stand-ins for the optional pointers.  The kernels never dereference
        # them, but a launch still needs a valid address.
        self._dummy_f32 = torch.empty(1, dtype=torch.float32, device=self.device)
        self._dummy_i32 = torch.empty(1, dtype=torch.int32, device=self.device)

        self._kv_key = None
        self._kv_views = None

    # ── Internals ────────────────────────────────────────────────────────

    def _views(self, ori_kv):
        """Strided FP8/scale views, rebuilt only when the cache moves."""
        key = (ori_kv.data_ptr(), ori_kv.shape[0], ori_kv.shape[1])
        if key != self._kv_key:
            self._kv_views = ori_cache_views(ori_kv)
            self._kv_key = key
        return self._kv_views

    def _check(self, q, ori_kv, ori_sparse_indices, ori_block_table, topk, sinks):
        """Reject a call the plan's buffers and compiled constexprs do not fit."""
        got = (
            tuple(q.shape),
            q.dtype,
            q.device,
            ori_kv.shape[1],
            ori_sparse_indices.shape[1],
            ori_block_table.shape[1],
            topk is not None,
            sinks is not None,
        )
        want = (
            (self.t1, self.n1, D),
            torch.bfloat16,
            self.device,
            self.block_size,
            self.k_total,
            self.max_blocks,
            self.has_topk,
            self.has_sinks,
        )
        if got != want:
            names = (
                "q shape",
                "q dtype",
                "device",
                "block_size",
                "K",
                "max_blocks",
                "has topk",
                "has sinks",
            )
            diff = ", ".join(
                f"{n}: want {w}, got {g}" for n, w, g in zip(names, want, got) if w != g
            )
            raise ValueError(f"plan is for a different call ({diff})")

    def _run_gather(self, ori_data, ori_scale, si, bt, topk):
        n_s_tiles = self.k_total // self.block_s
        cores = min(self.gather_cores, self.t1 * n_s_tiles)
        gather_dequant_kernel[(cores,)](
            ori_data,
            ori_scale,
            si,
            bt,
            topk if self.has_topk else self._dummy_i32,
            self.deq_buffer,
            self.valid_buffer,
            T1=self.t1,
            K_TOTAL=self.k_total,
            MAX_BLK=self.max_blocks,
            BS=self.block_size,
            LOG2_BS=self.log2_block_size,
            BLOCK_S=self.block_s,
            NUM_CORES=cores,
            D_C=D,
            NG=N_GROUPS,
            GS=GROUP_SIZE,
            ROW_STRIDE_DATA=ROW_STRIDE_DATA,
            ROW_STRIDE_SCALE=ROW_STRIDE_SCALE,
            HAS_TOPK=self.has_topk,
            **MULTIBUFFER_OPTS,
        )

    def _run_combine(self, out, lse_buf):
        cores = min(self.num_cores, self.t1 * self.n_head_groups)
        splitk_combine_kernel[(cores,)](
            self.ws_acc,
            self.ws_m,
            self.ws_l,
            out,
            lse_buf,
            T1=self.t1,
            N1_C=self.n1,
            BLOCK_H=self.block_h,
            SPLITS=self.splits,
            NUM_CORES=cores,
            D_C=D,
            RETURN_LSE=self.return_softmax_lse,
        )

    # ── Entry point ──────────────────────────────────────────────────────

    def __call__(
        self,
        q,
        *,
        ori_kv,
        ori_sparse_indices,
        ori_block_table,
        ori_topk_length=None,
        sinks=None,
        out=None,
        softmax_lse=None,
    ):
        """Run the plan.  Returns ``(out, lse)``; ``lse`` is empty unless asked for."""
        self._check(
            q, ori_kv, ori_sparse_indices, ori_block_table, ori_topk_length, sinks
        )

        q = q.contiguous()
        si = ori_sparse_indices.contiguous()
        bt = ori_block_table.contiguous()
        topk = (
            ori_topk_length.contiguous().reshape(-1)
            if self.has_topk
            else self._dummy_i32
        )
        sinks_t = sinks.contiguous().reshape(-1) if self.has_sinks else self._dummy_f32

        ori_data, ori_scale = self._views(ori_kv)

        if out is None:
            out = torch.empty(
                self.t1, self.n1, D, dtype=torch.bfloat16, device=self.device
            )
        elif out.shape != (self.t1, self.n1, D) or out.dtype != torch.bfloat16:
            raise ValueError(
                f"out must be [{self.t1}, {self.n1}, {D}] bfloat16, got "
                f"{tuple(out.shape)} {out.dtype}"
            )

        if self.return_softmax_lse:
            if softmax_lse is None:
                softmax_lse = torch.empty(
                    1, self.t1, self.n1, dtype=torch.float32, device=self.device
                )
            elif (
                softmax_lse.numel() != self.t1 * self.n1
                or softmax_lse.dtype != torch.float32
            ):
                raise ValueError(
                    f"softmax_lse must hold {self.t1 * self.n1} float32 entries, "
                    f"got {softmax_lse.numel()} {softmax_lse.dtype}"
                )
            lse_buf = softmax_lse
        else:
            softmax_lse = torch.empty(0, dtype=torch.float32, device=self.device)
            lse_buf = self._dummy_f32

        if self.path in _GATHER_PATHS:
            self._run_gather(ori_data, ori_scale, si, bt, topk)

        if self.path == "gather_contig":
            cores = min(self.num_cores, self.t1 * self.n_head_groups)
            contig_attn_kernel[(cores,)](
                q,
                self.deq_buffer,
                self.valid_buffer,
                sinks_t,
                out,
                lse_buf,
                self.softmax_scale,
                T1=self.t1,
                N1_C=self.n1,
                K_TOTAL=self.k_total,
                BLOCK_H=self.block_h,
                BLOCK_S=self.attn_block_s,
                NUM_CORES=cores,
                D_C=D,
                HAS_SINKS=self.has_sinks,
                RETURN_LSE=self.return_softmax_lse,
                **MULTIBUFFER_OPTS,
            )
        elif self.path == "gather_contig_splitk":
            cores = min(self.num_cores, self.t1 * self.n_head_groups * self.splits)
            contig_splitk_main_kernel[(cores,)](
                q,
                self.deq_buffer,
                self.valid_buffer,
                sinks_t,
                self.ws_acc,
                self.ws_m,
                self.ws_l,
                self.softmax_scale,
                T1=self.t1,
                N1_C=self.n1,
                K_TOTAL=self.k_total,
                BLOCK_H=self.block_h,
                BLOCK_S=self.block_s,
                SPLITS=self.splits,
                NUM_CORES=cores,
                D_C=D,
                HAS_SINKS=self.has_sinks,
                **MULTIBUFFER_OPTS,
            )
            self._run_combine(out, lse_buf)
        elif self.path == "scattered_splitk":
            cores = min(self.num_cores, self.t1 * self.n_head_groups * self.splits)
            scattered_splitk_main_kernel[(cores,)](
                q,
                ori_data,
                ori_scale,
                si,
                bt,
                topk,
                sinks_t,
                self.ws_acc,
                self.ws_m,
                self.ws_l,
                self.softmax_scale,
                T1=self.t1,
                N1_C=self.n1,
                K_TOTAL=self.k_total,
                MAX_BLK=self.max_blocks,
                BS=self.block_size,
                LOG2_BS=self.log2_block_size,
                BLOCK_H=self.block_h,
                BLOCK_S=self.block_s,
                SPLITS=self.splits,
                NUM_CORES=cores,
                D_C=D,
                NG=N_GROUPS,
                GS=GROUP_SIZE,
                ROW_STRIDE_DATA=ROW_STRIDE_DATA,
                ROW_STRIDE_SCALE=ROW_STRIDE_SCALE,
                HAS_TOPK=self.has_topk,
                HAS_SINKS=self.has_sinks,
                **MULTIBUFFER_OPTS,
            )
            self._run_combine(out, lse_buf)
        else:  # "scattered"
            cores = min(self.num_cores, self.t1 * self.n_head_groups)
            scattered_attn_kernel[(cores,)](
                q,
                ori_data,
                ori_scale,
                si,
                bt,
                topk,
                sinks_t,
                out,
                lse_buf,
                self.softmax_scale,
                T1=self.t1,
                N1_C=self.n1,
                K_TOTAL=self.k_total,
                MAX_BLK=self.max_blocks,
                BS=self.block_size,
                LOG2_BS=self.log2_block_size,
                BLOCK_H=self.block_h,
                BLOCK_S=self.block_s,
                NUM_CORES=cores,
                D_C=D,
                NG=N_GROUPS,
                GS=GROUP_SIZE,
                ROW_STRIDE_DATA=ROW_STRIDE_DATA,
                ROW_STRIDE_SCALE=ROW_STRIDE_SCALE,
                HAS_TOPK=self.has_topk,
                HAS_SINKS=self.has_sinks,
                RETURN_LSE=self.return_softmax_lse,
                **MULTIBUFFER_OPTS,
            )

        return out, softmax_lse


_PLAN_CACHE = {}
_PLAN_CACHE_LIMIT = 32


def _cached_plan(key, build):
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        if len(_PLAN_CACHE) >= _PLAN_CACHE_LIMIT:
            _PLAN_CACHE.clear()
        plan = build()
        _PLAN_CACHE[key] = plan
    return plan


def mixed_quant_sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    softmax_scale=None,
    quant_mode=1,
    return_softmax_lse=False,
    num_cores=NUM_CORES,
    block_h=BLOCK_H,
    block_s=BLOCK_S,
    splits=None,
    path="auto",
):
    """Sparse MLA decode attention over an FP8 paged KV cache.

    Replaces ``cann_ops_transformer.mixed_quant_sparse_flash_mla``
    (``aclnnMixedQuantSparseFlashMla``) in the ORI_SPARSE decode call form.

    Args:
        q: [T1, N1, 512] bf16 query, TND layout.
        ori_kv: [block_num, block_size, 1, 544] uint8 FP8 KV cache, PA_BBND.
        cmp_kv: FP4 compressed side; not implemented.
        ori_sparse_indices: [T1, K] int32 token indices into the logical cache.
        ori_block_table: [T1, max_blocks] int32 logical-to-physical page map.
        ori_topk_length: [T1] int32 valid prefix of each row of the indices.
            When omitted, only the ``index >= 0`` test gates a token.
        sinks: [N1] fp32 per-head bias seeding the online softmax row max.
        softmax_scale: defaults to ``512 ** -0.5``.
        quant_mode: must be 1.
        return_softmax_lse: also return the log-sum-exp.
        num_cores: grid width; defaults to the 32 Cube cores of a 950DT.
        block_h: query heads per task, must divide N1.
        block_s: sparse tokens per tile, must divide K.
        splits: K splits for the split-K paths; derived from the core count
            when omitted.
        path: force one of ``PATHS`` instead of the measured dispatch.

    Returns:
        ``(out, lse)`` with ``out`` [T1, N1, 512] bf16 and ``lse``
        [1, T1, N1] fp32, or an empty fp32 tensor when not requested.

    Raises:
        NotImplementedError: for the compressed KV side or a quant mode other
            than 1, both of which have a different byte layout.
        ValueError: on a shape, dtype or tuning argument this op cannot run.
    """
    if quant_mode != 1:
        raise NotImplementedError(
            f"quant_mode {quant_mode} has a different KV row layout; "
            "only quant_mode=1 (544B FP8 rows) is implemented"
        )
    if cmp_kv is not None or cmp_sparse_indices is not None:
        raise NotImplementedError(
            f"the compressed KV side ({CMP_KV_ROW_BYTES}B FP4 rows) is not "
            "implemented"
        )
    if cmp_block_table is not None or cmp_topk_length is not None:
        raise NotImplementedError("the compressed KV side is not implemented")
    if ori_kv is None:
        raise ValueError("ori_kv is required")

    key = (
        tuple(q.shape),
        q.device,
        ori_kv.shape[0],
        ori_kv.shape[1],
        None if ori_sparse_indices is None else ori_sparse_indices.shape[1],
        None if ori_block_table is None else ori_block_table.shape[1],
        ori_topk_length is not None,
        sinks is not None,
        softmax_scale,
        bool(return_softmax_lse),
        num_cores,
        block_h,
        block_s,
        splits,
        path,
    )

    plan = _cached_plan(
        key,
        lambda: MixedQuantSparseFlashMlaPlan(
            q,
            ori_kv=ori_kv,
            ori_sparse_indices=ori_sparse_indices,
            ori_block_table=ori_block_table,
            ori_topk_length=ori_topk_length,
            sinks=sinks,
            softmax_scale=softmax_scale,
            return_softmax_lse=return_softmax_lse,
            num_cores=num_cores,
            block_h=block_h,
            block_s=block_s,
            splits=splits,
            path=path,
        ),
    )
    return plan(
        q,
        ori_kv=ori_kv,
        ori_sparse_indices=ori_sparse_indices,
        ori_block_table=ori_block_table,
        ori_topk_length=ori_topk_length,
        sinks=sinks,
    )
