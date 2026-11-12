"""Research T=400 Graph kernel with native raw M64N24 projection math.

The current all-producer/non-striped Graph producer, software tile barrier,
CTA-local routing, and hidden tail are unchanged.  Only the four-warp math
partition is replaced: the existing N32 shared B stage feeds native Hopper
``m64n24k8`` instructions with a persistent twelve-register accumulator.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import triton
import triton.language as tl

from . import sm90_eager_p384_rs_wgmma as _tail_helper_module
from . import sm90_large_graph_split16_rs_t400_nonstriped_wgmma as _t400
from . import sm90_wgmma_m64n24_raw_probe as _raw_module
from .sm90_wgmma_m64n24_raw_probe import (
    _raw_m64n24k64_accumulate,
    _store_raw_n24_partials,
)

TOKENS = _t400.TOKENS
_implementation = _t400._implementation
_base = _t400._base
_t64 = _t400._t64
_T400_SOURCE = Path(_t400.__file__).resolve()
_T400_SHA256 = hashlib.sha256(_T400_SOURCE.read_bytes()).hexdigest()
_EXPECTED_T400_SHA256 = (
    "74b1da8bf6bff10cce9a8f54fedf0165df211c823e62cf34eed9aac857de2602"
)
if _T400_SHA256 != _EXPECTED_T400_SHA256:
    raise RuntimeError(
        "raw N24 candidate pins T400 producer source SHA256 "
        f"{_EXPECTED_T400_SHA256}, got {_T400_SHA256}"
    )

_RAW_HELPER_SOURCE = Path(_raw_module.__file__).resolve()
_RAW_HELPER_SHA256 = hashlib.sha256(_RAW_HELPER_SOURCE.read_bytes()).hexdigest()
_EXPECTED_RAW_HELPER_SHA256 = (
    "550e368ed415c45f04c87112b10f3c261c6ff3c239f3613d0a384c881f887a5f"
)
if _RAW_HELPER_SHA256 != _EXPECTED_RAW_HELPER_SHA256:
    raise RuntimeError(
        "raw N24 candidate pins helper source SHA256 "
        f"{_EXPECTED_RAW_HELPER_SHA256}, got {_RAW_HELPER_SHA256}"
    )

_TAIL_HELPER_SOURCE = Path(_tail_helper_module.__file__).resolve()
_TAIL_HELPER_SHA256 = hashlib.sha256(_TAIL_HELPER_SOURCE.read_bytes()).hexdigest()
_EXPECTED_TAIL_HELPER_SHA256 = (
    "3507caadf1195a029c3fd2d84320a65c7f98b79d33a1a13ae22f79de32d587a7"
)
if _TAIL_HELPER_SHA256 != _EXPECTED_TAIL_HELPER_SHA256:
    raise RuntimeError(
        "raw N24 candidate pins the serial-quad tail helper SHA256 "
        f"{_EXPECTED_TAIL_HELPER_SHA256}, got {_TAIL_HELPER_SHA256}"
    )


@triton.jit
def _t400_raw_n24_math(
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
    values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if is_producer:
        for hb in tl.static_range(0, _implementation.JIT_HIDDEN_BLOCKS_PER_SPLIT):
            if hb % 2 == 0:
                _implementation.tle.gpu.barrier_wait(ready0)
            else:
                _implementation.tle.gpu.barrier_wait(ready1)
            for channel in tl.static_range(0, _implementation.JIT_HC_MULT):
                q_address = _implementation._quad._shared_address(
                    _implementation.tle.gpu.local_ptr(
                        q_stages.slot((hb % 2) * _implementation.JIT_HC_MULT + channel),
                        (0, 0),
                    )
                )
                fn_address = _implementation._quad._shared_address(
                    _implementation.tle.gpu.local_ptr(
                        fn_stages.slot(
                            (hb % 2) * _implementation.JIT_HC_MULT + channel
                        ),
                        (0, 0),
                    )
                )
                values = _raw_m64n24k64_accumulate(
                    q_address,
                    fn_address,
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                    values[5],
                    values[6],
                    values[7],
                    values[8],
                    values[9],
                    values[10],
                    values[11],
                )
            if hb % 2 == 0:
                _implementation.tle.gpu.barrier_wait(done0)
            else:
                _implementation.tle.gpu.barrier_wait(done1)

        split_partials = (
            partials_ptr
            + split * _implementation.JIT_TOKENS * _implementation.JIT_BLOCK_N
        )
        _ = _store_raw_n24_partials(
            tl.cast(split_partials, tl.uint64),
            tile,
            _implementation.JIT_TOKENS,
            values,
        )

    _implementation.tle.gpu.barrier_wait(producer_ready)
    if _implementation.JIT_PROBE_PHASE >= 2:
        physical_thread = _t64._physical_thread_id_x()
        _ = _t400._t400_tile_barrier(tl.cast(partials_ptr, tl.uint64), physical_thread)
        _implementation.tle.gpu.barrier_wait(grid_acquire_ready)
        if _implementation.JIT_PROBE_PHASE >= 3:
            lane = physical_thread % 32
            role = (physical_thread - _implementation.JIT_ROOT_THREADS) // 32
            first_token = (
                tile * _implementation.JIT_BLOCK_M + split * _base.JIT_NUM_ROLES
            )
            if first_token < _implementation.JIT_TOKENS:
                token = first_token + role
                packed_sum = 0.0
                for source_split in tl.static_range(0, _implementation.JIT_NUM_SPLITS):
                    packed_sum += tl.load(
                        partials_ptr
                        + source_split
                        * _implementation.JIT_TOKENS
                        * _implementation.JIT_BLOCK_N
                        + token * _implementation.JIT_BLOCK_N
                        + lane,
                        mask=lane <= _implementation.JIT_PROJECTION_OUTPUTS,
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
                _base._full_tail_barrier_arrive()


# The owning kernel resolves this global when the fresh-process candidate is
# specialized, so every allocation/producer/barrier remains byte-identical.
_t400._t400_nonstriped_math = _t400_raw_n24_math


def prepare_tle_sm90_mega_mhc(
    inputs: Mapping[str, object], outputs: Mapping[str, object]
):
    plan = _t400.prepare_tle_sm90_mega_mhc(inputs, outputs)
    plan.metadata.update(
        {
            "admission_source": str(Path(__file__).resolve()),
            "base_t400_source": str(_T400_SOURCE),
            "base_t400_source_sha256": _T400_SHA256,
            "raw_helper_source": str(_RAW_HELPER_SOURCE),
            "raw_helper_source_sha256": _RAW_HELPER_SHA256,
            "tail_helper_source": str(_TAIL_HELPER_SOURCE),
            "tail_helper_source_sha256": _TAIL_HELPER_SHA256,
            "pipeline_variant": "t400-graph-raw-m64n24k8",
            "projection_wgmma_shape": "m64n24k8",
            "projection_accumulator_registers_per_thread": 12,
            "expected_spills": 0,
            "expected_local_bytes_per_thread": 0,
            "audited_spill_admission": False,
            "experimental": False,
            "dispatch_enabled": True,
        }
    )
    return plan


__all__ = ["TOKENS", "prepare_tle_sm90_mega_mhc"]
