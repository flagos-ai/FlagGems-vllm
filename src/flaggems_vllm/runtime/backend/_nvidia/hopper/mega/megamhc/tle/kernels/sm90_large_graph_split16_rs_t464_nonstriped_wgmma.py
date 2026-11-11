"""Production T=464 split16 RS variant with CTA-local token ownership.

T=464 leaves 48 inactive rows in the eighth M64 tile.  The generic
split16 kernel uses striped role ownership, which spreads those inactive rows
over many CTAs and prevents a CTA-uniform tail skip.  This probe maps each
split's four routing/hidden roles to four adjacent rows.  The twelve empty
splits in the final tile still join both cooperative rendezvous, but skip the
projection reduction, Sinkhorn, and hidden/RMSNorm work after publishing the
required tail-barrier arrival.

All 128 CTAs are projection producers; the exact kernel makes that fact a
compile-time constant and removes the generic consumer branch.  Production
dispatch selects this module only for the exact T=464 Graph bucket.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Mapping


TOKENS = 464
_EXPECTED_IMPLEMENTATION_SHA256 = (
    "9aca5026b6bfdd29f54278a9971c06ddb26a8b49b01082e7433bc39a6c5460cc"
)
_EXACT_ENVIRONMENT = {
    "DG_TLE_MHC_LARGE_GRAPH_ROLE4": "1",
    "DG_TLE_MHC_LARGE_GRAPH_ROLE4_LOGICAL_TOKENS": str(TOKENS),
    "DG_TLE_MHC_LARGE_GRAPH_GRID_CTAS": "128",
    "DG_TLE_MHC_LARGE_GRAPH_TRUE_K64_STAGE": "1",
    "DG_TLE_MHC_LARGE_GRAPH_COEFF_CACHE": "0",
    "DG_TLE_MHC_LARGE_GRAPH_WEIGHT_CACHE": "0",
    "DG_TLE_MHC_LARGE_GRAPH_PTX_SINKHORN": "1",
    "DG_TLE_MHC_LARGE_GRAPH_NON_STRIPED_TAIL": "1",
    "DG_TLE_MHC_LARGE_GRAPH_EARLY_PRE_SYNC": "1",
    "DG_TLE_MHC_LARGE_GRAPH_PROBE_PHASE": "4",
    "DG_TLE_MHC_LARGE_GRAPH_ROLE_OVERLAP": "0",
    "DG_TLE_MHC_LARGE_GRAPH_ALLOCATED_ROLE_BARRIERS": "0",
    "DG_TLE_MHC_LARGE_GRAPH_SHARED_ROLE_FLAGS": "0",
    "DG_TLE_MHC_LARGE_GRAPH_TILE_LOCAL_BARRIER": "1",
    "DG_TLE_MHC_LARGE_GRAPH_FAST_GRID_BARRIER": "0",
    "DG_TLE_MHC_LARGE_GRAPH_EXACT_T448": "0",
    "DG_TLE_MHC_LARGE_GRAPH_EXACT_T512": "0",
    "DG_TLE_MHC_LARGE_GRAPH_SPLIT16_RS_MATH_REGS": "128",
}


for _name, _expected in _EXACT_ENVIRONMENT.items():
    _actual = os.environ.get(_name)
    if _actual not in (None, _expected):
        raise RuntimeError(
            f"{_name}={_actual!r}, but T464 non-striped requires "
            f"{_expected!r}"
        )
    os.environ[_name] = _expected


import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from .. import fused_post_gemm_t64 as _t64  # noqa: E402
from . import sm90_large_graph_split16_rs_wgmma as _implementation  # noqa: E402


_base = _implementation._base
_IMPLEMENTATION_SOURCE = Path(_implementation.__file__).resolve()
_IMPLEMENTATION_SHA256 = hashlib.sha256(
    _IMPLEMENTATION_SOURCE.read_bytes()
).hexdigest()
if _IMPLEMENTATION_SHA256 != _EXPECTED_IMPLEMENTATION_SHA256:
    raise RuntimeError(
        "T464 admission pins split16 RS implementation SHA256 "
        f"{_EXPECTED_IMPLEMENTATION_SHA256}, got {_IMPLEMENTATION_SHA256}"
    )
if _implementation.TOKENS != TOKENS:
    raise RuntimeError(
        f"T464 admission imported T={_implementation.TOKENS} implementation"
    )
# Keep the eight projection tiles x sixteen splits as the complete grid.
_implementation.GRID_CTAS = _implementation.PRODUCER_CTAS
_implementation.JIT_GRID_CTAS = tl.constexpr(_implementation.PRODUCER_CTAS)
_base.GRID_CTAS = _implementation.PRODUCER_CTAS
_base.JIT_GRID_CTAS = tl.constexpr(_implementation.PRODUCER_CTAS)
_original_hidden_tail = _base._pipe_role4_hidden_tail


@triton.jit
def _t464_tile_barrier(counter_base, thread):
    tile = tl.program_id(0) // _implementation.JIT_NUM_SPLITS
    tile_counter = counter_base + tile * 16
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader, pending;
          .reg .u64 old, one, target, current;
          setp.eq.u32 leader, $2, 0;
          setp.ne.u32 pending, $2, $2;
          mov.u64 old, 0;
          mov.u64 one, 1;
          @leader atom.release.gpu.global.add.u64 old, [$1], one;
          @leader or.b64 target, old, 15;
          @leader add.u64 target, target, 1;
        DG_TLE_MHC_T464_TILE_POW2_WAIT:
          @leader ld.acquire.gpu.global.u64 current, [$1];
          @leader setp.lt.u64 pending, current, target;
          @pending bra DG_TLE_MHC_T464_TILE_POW2_WAIT;
          mov.u64 $0, old;
        }
        """,
        constraints="=l,l,r",
        args=[tl.cast(tile_counter, tl.uint64), thread],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _t464_active_cta_hidden_tail(
    weight_shared,
    pre_shared,
    new_residual_ptr,
    rmsnorm_weight_ptr,
    y_bf16_ptr,
    pid,
    tile,
    split,
    rmsnorm_eps,
    rmsnorm_scale,
):
    first_token = (
        tile * _implementation.JIT_BLOCK_M
        + split * _base.JIT_NUM_ROLES
    )
    if first_token < _implementation.JIT_TOKENS:
        _original_hidden_tail(
            weight_shared,
            pre_shared,
            new_residual_ptr,
            rmsnorm_weight_ptr,
            y_bf16_ptr,
            pid,
            tile,
            split,
            rmsnorm_eps,
            rmsnorm_scale,
        )


@triton.jit
def _t464_nonstriped_math(
    partials_ptr,
    q_stages,
    fn_stages,
    ready0,
    ready1,
    done0,
    done1,
    producer_ready,
    grid_acquire_ready,
    pre_shared,
    mix_scales_ptr,
    mix_bases_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    tile,
    split,
    pid,
    is_producer,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
):
    accumulator = tl.zeros(
        (_implementation.JIT_BLOCK_M, _implementation.JIT_BLOCK_N),
        tl.float32,
    )
    if is_producer:
        for hb in tl.static_range(
            0, _implementation.JIT_HIDDEN_BLOCKS_PER_SPLIT
        ):
            if hb % 2 == 0:
                _implementation.tle.gpu.barrier_wait(ready0)
            else:
                _implementation.tle.gpu.barrier_wait(ready1)
            for channel in tl.static_range(0, _implementation.JIT_HC_MULT):
                q_registers = tl.load(
                    _implementation.tle.gpu.local_ptr(
                        q_stages.slot(
                            (hb % 2) * _implementation.JIT_HC_MULT + channel
                        )
                    )
                ).to(tl.float32)
                accumulator = _implementation.tle.gpu.wgmma(
                    q_registers,
                    fn_stages.slot(
                        (hb % 2) * _implementation.JIT_HC_MULT + channel
                    ),
                    accumulator,
                    input_precision="tf32",
                    trans_b=True,
                )
                accumulator = _implementation.tle.gpu.wgmma_wait(
                    0, accumulator
                )
            if hb % 2 == 0:
                _implementation.tle.gpu.barrier_wait(done0)
            else:
                _implementation.tle.gpu.barrier_wait(done1)

        m = (
            tile * _implementation.JIT_BLOCK_M
            + tl.arange(0, _implementation.JIT_BLOCK_M)
        )
        n = tl.arange(0, _implementation.JIT_BLOCK_N)
        tl.store(
            partials_ptr
            + split
            * _implementation.JIT_TOKENS
            * _implementation.JIT_BLOCK_N
            + m[:, None] * _implementation.JIT_BLOCK_N
            + n[None, :],
            accumulator,
            mask=(m[:, None] < _implementation.JIT_TOKENS)
            & (n[None, :] < _implementation.JIT_PROJECTION_OUTPUTS),
        )

    _implementation.tle.gpu.barrier_wait(producer_ready)
    if _implementation.JIT_PROBE_PHASE >= 2:
        _implementation.tle.gpu.barrier_wait(grid_acquire_ready)
        if _implementation.JIT_PROBE_PHASE >= 3:
            physical_thread = _t64._physical_thread_id_x()
            lane = physical_thread % 32
            role = (
                physical_thread - _implementation.JIT_ROOT_THREADS
            ) // 32
            first_token = (
                tile * _implementation.JIT_BLOCK_M
                + split * _base.JIT_NUM_ROLES
            )
            if first_token < _implementation.JIT_TOKENS:
                token = first_token + role
                packed_sum = 0.0
                for source_split in tl.static_range(
                    0, _implementation.JIT_NUM_SPLITS
                ):
                    packed_sum += tl.load(
                        partials_ptr
                        + source_split
                        * _implementation.JIT_TOKENS
                        * _implementation.JIT_BLOCK_N
                        + token * _implementation.JIT_BLOCK_N
                        + lane,
                        mask=lane
                        <= _implementation.JIT_PROJECTION_OUTPUTS,
                        other=0.0,
                    )
                _base._pipe_role4_routing_finalize(
                    pre_shared,
                    pre_shared,
                    producer_ready,
                    producer_ready,
                    producer_ready,
                    producer_ready,
                    mix_scales_ptr,
                    mix_bases_ptr,
                    new_post_mix_ptr,
                    new_comb_res_mix_ptr,
                    packed_sum,
                    token,
                    role,
                    lane,
                    True,
                    hc_norm_eps,
                    hc_pre_eps,
                    hc_post_scale,
                    sinkhorn_eps,
                )
            else:
                # Root still waits at the full-tail barrier in every CTA.
                _base._full_tail_barrier_arrive()


_base._pipe_role4_hidden_tail = _t464_active_cta_hidden_tail
_base._monotonic_grid_arrive_and_wait = _t464_tile_barrier
_implementation._split16_rs_math = _t464_nonstriped_math


@triton.jit
def _t464_allproducer_kernel(
    x_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_res_mix_ptr,
    fn_ptr,
    mix_scales_ptr,
    mix_bases_ptr,
    rmsnorm_weight_ptr,
    new_residual_ptr,
    new_post_mix_ptr,
    new_comb_res_mix_ptr,
    y_bf16_ptr,
    partials_ptr,
    grid_counter_ptr,
    hc_norm_eps,
    hc_pre_eps,
    hc_post_scale,
    sinkhorn_eps,
    rmsnorm_eps,
    rmsnorm_scale,
):
    pid = tl.program_id(0)
    tile = pid // _implementation.JIT_NUM_SPLITS
    split = pid % _implementation.JIT_NUM_SPLITS

    q_stages = _implementation.tle.gpu.alloc(
        [
            _implementation.JIT_PIPE_STAGES,
            _implementation.JIT_BLOCK_M,
            _implementation.JIT_BLOCK_K,
        ],
        dtype=tl.bfloat16,
        layout=None,
        scope=_implementation.tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    fn_stages = _implementation.tle.gpu.alloc(
        [
            _implementation.JIT_PIPE_STAGES,
            _implementation.JIT_BLOCK_N,
            _implementation.JIT_BLOCK_K,
        ],
        dtype=tl.float32,
        layout=None,
        scope=_implementation.tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    post_shared = _implementation.tle.gpu.alloc(
        [_implementation.JIT_BLOCK_M, _implementation.JIT_HC_MULT],
        dtype=tl.float32,
        layout=None,
        scope=_implementation.tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    comb_shared = _implementation.tle.gpu.alloc(
        [
            _implementation.JIT_BLOCK_M,
            _implementation.JIT_HC_MULT * _implementation.JIT_HC_MULT,
        ],
        dtype=tl.float32,
        layout=None,
        scope=_implementation.tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pre_shared = _implementation.tle.gpu.alloc(
        [4, _implementation.JIT_HC_MULT],
        dtype=tl.float32,
        layout=None,
        scope=_implementation.tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    weight_dummy = _implementation.tle.gpu.alloc(
        [1],
        dtype=tl.bfloat16,
        layout=None,
        scope=_implementation.tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    producer_ready = _implementation.tle.gpu.alloc_barrier(
        arrive_count=_implementation.JIT_PHYSICAL_THREADS
    )
    grid_acquire_ready = _implementation.tle.gpu.alloc_barrier(
        arrive_count=_implementation.JIT_PHYSICAL_THREADS
    )
    coeff_ready = _implementation.tle.gpu.alloc_barrier(
        arrive_count=_implementation.JIT_ROOT_THREADS
    )
    ready0 = _implementation.tle.gpu.alloc_barrier(
        arrive_count=_implementation.JIT_PHYSICAL_THREADS
    )
    ready1 = _implementation.tle.gpu.alloc_barrier(
        arrive_count=_implementation.JIT_PHYSICAL_THREADS
    )
    done0 = _implementation.tle.gpu.alloc_barrier(
        arrive_count=_implementation.JIT_PHYSICAL_THREADS
    )
    done1 = _implementation.tle.gpu.alloc_barrier(
        arrive_count=_implementation.JIT_PHYSICAL_THREADS
    )

    _implementation.tle.gpu.warp_specialize(
        [
            (
                _implementation._split16_rs_root,
                (
                    x_ptr, residual_ptr, post_mix_ptr, comb_res_mix_ptr,
                    fn_ptr, new_residual_ptr, partials_ptr, q_stages,
                    fn_stages, ready0, ready1, done0, done1, producer_ready,
                    grid_acquire_ready, coeff_ready, post_shared, comb_shared,
                    weight_dummy, pre_shared, rmsnorm_weight_ptr, y_bf16_ptr,
                    grid_counter_ptr, tile, split, pid, True,
                    rmsnorm_eps, rmsnorm_scale,
                ),
            ),
            (
                _t464_nonstriped_math,
                (
                    partials_ptr, q_stages, fn_stages, ready0, ready1,
                    done0, done1, producer_ready, grid_acquire_ready,
                    pre_shared, mix_scales_ptr, mix_bases_ptr,
                    new_post_mix_ptr, new_comb_res_mix_ptr, tile, split, pid,
                    True, hc_norm_eps, hc_pre_eps, hc_post_scale,
                    sinkhorn_eps,
                ),
            ),
        ],
        [_implementation.JIT_MATH_WARPS],
        [_implementation.JIT_MATH_REGS],
    )


@dataclass(frozen=True)
class LargeGraphSplit16RST464Plan:
    implementation: _implementation.LargeGraphSplit16RSPlan
    metadata: dict[str, object]

    @property
    def grid_ctas(self) -> int:
        return self.implementation.grid_ctas

    @property
    def partials(self):
        return self.implementation.partials

    @property
    def workspace_tensors(self):
        return self.implementation.workspace_tensors

    @property
    def counter_tensors(self):
        return self.implementation.counter_tensors

    def compile(self):
        return _t464_allproducer_kernel.warmup(
            *self.implementation._kernel_args(),
            grid=(_implementation.GRID_CTAS,),
            num_warps=_implementation.ROOT_WARPS,
            num_stages=1,
            num_ctas=1,
            launch_cooperative_grid=True,
        )

    def run(self) -> None:
        _t464_allproducer_kernel[(_implementation.GRID_CTAS,)](
            *self.implementation._kernel_args(),
            num_warps=_implementation.ROOT_WARPS,
            num_stages=1,
            num_ctas=1,
            launch_cooperative_grid=True,
        )


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> LargeGraphSplit16RST464Plan:
    implementation = _implementation.prepare_tle_sm90_mega_mhc(inputs, outputs)
    metadata = dict(implementation.metadata)
    metadata.update({
        "admission_shape": "T464",
        "admission_source": str(Path(__file__).resolve()),
        "implementation_source": str(_IMPLEMENTATION_SOURCE),
        "implementation_source_sha256": _IMPLEMENTATION_SHA256,
        "pipeline_variant": (
            "t464-allproducer-nonstriped-active-cta-skip-tilebar"
        ),
        "kernel_symbol": "_t464_allproducer_kernel",
        "CTA": _implementation.PRODUCER_CTAS,
        "grid_ctas": _implementation.PRODUCER_CTAS,
        "required_total_resident_ctas": _implementation.PRODUCER_CTAS,
        "consumer_only_ctas": 0,
        "audited_spill_admission": True,
        "expected_spills": 2,
        "expected_local_bytes_per_thread": 8,
        "experimental": False,
        "dispatch_enabled": True,
    })
    return LargeGraphSplit16RST464Plan(
        implementation=implementation,
        metadata=metadata,
    )


__all__ = [
    "LargeGraphSplit16RST464Plan",
    "TOKENS",
    "prepare_tle_sm90_mega_mhc",
]
