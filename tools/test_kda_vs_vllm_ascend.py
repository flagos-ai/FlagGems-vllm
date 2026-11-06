#!/usr/bin/env python3
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

"""Compare FlagGems Ascend KDA ops against the AscendC operators Kimi K3 calls.

Kimi K3's delta-attention layer on Ascend is
``vllm_ascend/ops/kimi_kda.py::AscendKimiK3DeltaAttention``. Its KDA math goes
through ``vllm_ascend/ops/kda.py``, which wraps **AscendC** kernels from the
installed ``fla_npu`` package -- not the Triton files under
``vllm_ascend/ops/triton/kda/``, which are an unused fallback. This script
targets the AscendC operators:

  ===================  ========================================  =================================
  Stage                AscendC operator (what Kimi K3 calls)     FlagGems counterpart
  ===================  ========================================  =================================
  prefill KDA          ``fla_npu...chunk_kda_fwd`` via            ``_ascend.FLA.chunk_kda``
                       ``run_chunk_kda``                          (BT=16 vs AscendC BT=64)
  decode / spec KDA    ``fla_npu...recurrent_kda`` via            ``_ascend.FLA.fused_recurrent_kda``
                       ``run_recurrent_kda``                      (added for this gap)
  conv prefill         ``fla_npu...causal_conv1d_fn``             ``_ascend.ops.causal_conv1d_fn``
  conv decode          ``fla_npu...causal_conv1d_update``         ``_ascend.ops.causal_conv1d_update``
  ===================  ========================================  =================================

Both KDA entry points fuse q/k L2-norm, the bounded ("safe") decay gate
``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` and beta's sigmoid, so the
raw projections are passed in and the comparison covers the whole fused op.

Shapes default to the Kimi K3 KDA layer: ``head_dim=128``, ``num_heads=64``
sharded over TP (``--tp`` picks the local head count), ``conv_size=4``, and
a ``[pool, H, V, K]`` fp32 recurrent state cache.

Timing uses ``torch_npu.profiler`` with ``schedule(warmup=5, active=5)``,
summing every kernel's ``Duration(us)`` per iteration so a multi-kernel
implementation is charged its full device cost.

Caveat: numbers drift when many ops are swept in one process -- the same
``recurrent_kda`` bs=128 case measured 97.7 us with ``--ops recurrent_kda``
and 364 us inside a full sweep. Compare an op against its own reference
within one run, and re-measure a single op with ``--ops`` before acting on
its absolute latency.

Usage::

    PYTHONPATH=src:FlagGems/src python tools/test_kda_vs_vllm_ascend.py
    PYTHONPATH=... python tools/test_kda_vs_vllm_ascend.py --ops recurrent_kda
    PYTHONPATH=... python tools/test_kda_vs_vllm_ascend.py --skip-perf --tp 8
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass
from typing import Callable

import torch

try:
    import torch_npu
except ImportError as exc:  # pragma: no cover - environment guard
    print(f"FATAL: torch_npu unavailable: {exc}", file=sys.stderr)
    sys.exit(2)

# Kimi K3 KDA layer geometry (vllm KimiLinearConfig / Kimi K3 checkpoint).
HEAD_DIM = 128
NUM_HEADS_TOTAL = 64
CONV_SIZE = 4
GATE_LOWER_BOUND = -5.0
ASCENDC_CHUNK_SIZE = 64   # vllm_ascend.ops.kda.KDA_CHUNK_SIZE
FLAGGEMS_CHUNK_SIZE = 16  # FlagGems TLE prefill is specialized for BT=16
NULL_BLOCK_ID = 0         # vllm.v1.attention.backends.utils.NULL_BLOCK_ID
PAD_SLOT_ID = -1

WARMUP = 5
ACTIVE = 5

ALL_OPS = ("chunk_kda", "recurrent_kda", "conv_prefill", "conv_decode")


def _missing_dependency(exc: BaseException) -> str | None:
    """Return the message if a failure is really an absent build dependency.

    Triton reports a kernel that cannot compile as ``CompilationError`` with
    the real ``ModuleNotFoundError`` attached as its cause, so walk the chain.
    """
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ModuleNotFoundError, ImportError)):
            return str(current)
        current = current.__cause__ or current.__context__
    # Triton may also fold the cause into the rendered message only.
    text = str(exc)
    if "ModuleNotFoundError" in text and "is not installed" in text:
        return text.split("ModuleNotFoundError", 1)[1].strip(" (\"')\n")
    return None


def _import_sides():
    """Import both operator sets, reporting a usable message if either is absent."""
    try:
        from flaggems_vllm.ops.FLA.index import prepare_chunk_indices
        from flaggems_vllm.runtime.backend._ascend.FLA import (
            chunk_kda,
            fused_recurrent_kda,
        )
        from flaggems_vllm.runtime.backend._ascend.ops import (
            causal_conv1d_fn,
            causal_conv1d_update,
        )
    except ImportError as exc:
        print(
            f"FATAL: cannot import flaggems_vllm ({exc}).\n"
            "Add both repos to PYTHONPATH, e.g.\n"
            "  PYTHONPATH=src:FlagGems/src python tools/test_kda_vs_vllm_ascend.py",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        # The AscendC path Kimi K3 actually dispatches through.
        from fla_npu.ops.ascendc import (
            causal_conv1d_fn as ascendc_conv_fn,
            causal_conv1d_update as ascendc_conv_update,
        )
        from vllm_ascend.ops.kda import run_chunk_kda, run_recurrent_kda
    except ImportError as exc:
        print(
            f"FATAL: cannot import the AscendC KDA operators ({exc}).\n"
            "This comparison needs fla_npu and vllm_ascend installed.",
            file=sys.stderr,
        )
        sys.exit(2)

    return dict(
        prepare_chunk_indices=prepare_chunk_indices,
        chunk_kda=chunk_kda,
        fused_recurrent_kda=fused_recurrent_kda,
        causal_conv1d_fn=causal_conv1d_fn,
        causal_conv1d_update=causal_conv1d_update,
        run_chunk_kda=run_chunk_kda,
        run_recurrent_kda=run_recurrent_kda,
        ascendc_conv_fn=ascendc_conv_fn,
        ascendc_conv_update=ascendc_conv_update,
    )


# --------------------------------------------------------------------------- #
# profiling
# --------------------------------------------------------------------------- #
def profile_device_us(step: Callable[[], None], trace_root: str, tag: str) -> tuple[float, int]:
    """Return (mean per-iteration device time in us, kernels per iteration)."""
    trace_dir = os.path.join(trace_root, tag)
    shutil.rmtree(trace_dir, ignore_errors=True)
    os.makedirs(trace_dir, exist_ok=True)

    # Compile/autotune outside the measured window.
    for _ in range(WARMUP):
        step()
    torch.npu.synchronize()

    try:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=WARMUP, active=ACTIVE, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir),
        ) as prof:
            for _ in range(WARMUP + ACTIVE):
                step()
                torch.npu.synchronize()
                prof.step()

        pattern = os.path.join(trace_dir, "*_ascend_pt", "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if not files:
            return float("nan"), 0
        total_us, rows = 0.0, 0
        with open(files[0], newline="") as handle:
            for row in csv.DictReader(handle):
                key = next((k for k in row if k and "Duration" in k), None)
                if key is None:
                    continue
                try:
                    total_us += float(row[key])
                except (TypeError, ValueError):
                    continue
                rows += 1
        if rows == 0:
            return float("nan"), 0
        return total_us / ACTIVE, round(rows / ACTIVE)
    finally:
        # Traces are the bulk of this script's disk use; drop them once parsed.
        shutil.rmtree(trace_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# case plumbing
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    """One shape of one op pair: correctness check plus two timed closures."""

    op: str
    shape: str
    check: Callable[[], tuple[bool, str]]
    gems_step: Callable[[], None]
    ref_step: Callable[[], None]
    ref_label: str
    notes: str = ""


@dataclass
class Result:
    op: str
    shape: str
    ref_label: str
    passed: bool
    detail: str
    # Unavailable in this environment (e.g. a kernel needing FlagTree TLE).
    # Reported separately so a missing dependency is not read as a wrong result.
    skipped: bool = False
    gems_us: float = float("nan")
    ref_us: float = float("nan")
    gems_kernels: int = 0
    ref_kernels: int = 0
    notes: str = ""

    @property
    def speedup(self) -> float:
        if self.ref_us != self.ref_us or self.gems_us != self.gems_us or self.gems_us <= 0:
            return float("nan")
        return self.ref_us / self.gems_us


def compare(name: str, got: torch.Tensor, want: torch.Tensor, rtol: float, atol: float) -> tuple[bool, str]:
    """Compare on CPU in fp32 and report the worst deviation either way."""
    got_c, want_c = got.detach().float().cpu(), want.detach().float().cpu()
    if got_c.shape != want_c.shape:
        return False, f"{name}: shape {tuple(got_c.shape)} != {tuple(want_c.shape)}"
    if not torch.isfinite(got_c).all():
        return False, f"{name}: non-finite values in FlagGems output"
    diff = (got_c - want_c).abs()
    max_abs = diff.max().item()
    scale = want_c.abs().max().item()
    max_rel = max_abs / scale if scale > 0 else 0.0
    ok = torch.allclose(got_c, want_c, rtol=rtol, atol=atol)
    return ok, f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"


# --------------------------------------------------------------------------- #
# case builders
# --------------------------------------------------------------------------- #
def _kda_inputs(heads, tokens, device, dtype, seed=1234):
    """Raw KDA projections as the layer hands them to the fused operators."""
    torch.manual_seed(seed)
    q = torch.randn(1, tokens, heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(1, tokens, heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(1, tokens, heads, HEAD_DIM, dtype=dtype, device=device)
    # The gate is a raw projection; the bounded activation is fused in-kernel.
    raw_g = -torch.rand(1, tokens, heads, HEAD_DIM, dtype=dtype, device=device) * 0.05
    raw_beta = torch.rand(1, tokens, heads, dtype=torch.float32, device=device)
    a_log = torch.randn(heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(heads * HEAD_DIM, dtype=torch.float32, device=device) * 0.1
    return q, k, v, raw_g, raw_beta, a_log, dt_bias


def build_chunk_kda_cases(ops, device, heads, dtype) -> list[Case]:
    """Prefill KDA: FlagGems chunk_kda vs AscendC chunk_kda_fwd."""
    cases = []
    # Ragged prefill batches as the scheduler produces them.
    for cu_list in ([0, 128, 384], [0, 1024], [0, 31, 96], [0, 512, 1536, 4096]):
        tokens = cu_list[-1]
        n = len(cu_list) - 1
        q, k, v, raw_g, raw_beta, a_log, dt_bias = _kda_inputs(heads, tokens, device, dtype)
        cu = torch.tensor(cu_list, dtype=torch.int64, device=device)
        # state_v_first=True: the recurrent cache stores [N, H, V, K].
        init_vk = torch.randn(n, heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device)

        ci_ascendc = ops["prepare_chunk_indices"](cu, ASCENDC_CHUNK_SIZE)
        ci_gems = ops["prepare_chunk_indices"](cu, FLAGGEMS_CHUNK_SIZE)
        # run_chunk_kda types cu_seqlens/chunk_indices as Sequence[int] and
        # pre-activates beta on the host; honor that exact contract.
        cu_host = list(cu_list)
        ci_host = ci_ascendc.flatten().cpu().tolist()

        def ref_call(q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta, init=init_vk,
                     cu_host=cu_host, ci_host=ci_host, a_log=a_log, dt_bias=dt_bias):
            return ops["run_chunk_kda"](
                q, k, v, raw_g, raw_beta.sigmoid(), init.clone(),
                cu_host, ci_host, a_log, dt_bias, lower_bound=GATE_LOWER_BOUND,
            )

        def gems_call(q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta, init=init_vk,
                      cu=cu, ci=ci_gems, a_log=a_log, dt_bias=dt_bias):
            return ops["chunk_kda"](
                q, k, v, raw_g, raw_beta.to(q.dtype),
                scale=HEAD_DIM**-0.5,
                initial_state=init.clone(),
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                lower_bound=GATE_LOWER_BOUND,
                state_v_first=True,
                cu_seqlens=cu,
                chunk_indices=ci,
                chunk_size=FLAGGEMS_CHUNK_SIZE,
                A_log=a_log,
                dt_bias=dt_bias,
            )

        def check(gems_call=gems_call, ref_call=ref_call):
            got_o, got_s = gems_call()
            want_o, want_s = ref_call()
            torch.npu.synchronize()
            # FlagGems runs BT=16, AscendC BT=64: the chunked recurrence is
            # mathematically equal but accumulates in a different order, so
            # bf16 outputs differ at roughly bf16 epsilon.
            ok_o, det_o = compare("out", got_o, want_o, rtol=3e-2, atol=3e-2)
            ok_s, det_s = compare("state", got_s, want_s, rtol=3e-2, atol=3e-2)
            return ok_o and ok_s, f"{det_o} | {det_s}"

        cases.append(
            Case(
                op="chunk_kda",
                shape=f"T={tokens} H={heads} D={HEAD_DIM} seqs={n}",
                check=check,
                gems_step=lambda f=gems_call: f(),
                ref_step=lambda f=ref_call: f(),
                ref_label="AscendC chunk_kda_fwd (run_chunk_kda)",
                notes=f"BT={FLAGGEMS_CHUNK_SIZE} vs AscendC BT={ASCENDC_CHUNK_SIZE}",
            )
        )
    return cases


def build_recurrent_kda_cases(ops, device, heads, dtype) -> list[Case]:
    """Decode / speculative KDA: FlagGems fused_recurrent_kda vs AscendC recurrent_kda."""
    cases = []
    # (decode batch, speculative tokens per request); num_spec=0 is plain decode.
    for batch, num_spec in ((4, 0), (32, 0), (128, 0), (8, 3)):
        tokens = batch * (num_spec + 1)
        q, k, v, raw_g, raw_beta, a_log, dt_bias = _kda_inputs(
            heads, tokens, device, dtype, seed=11
        )
        pool = max(2 * tokens + 2, 64)
        base = torch.randn(pool, heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device)

        if num_spec:
            # Speculative decode: one state column per candidate token, and
            # the accepted count picks which column seeds the recurrence.
            cu = torch.arange(0, tokens + 1, num_spec + 1, dtype=torch.int32, device=device)
            idx = torch.arange(
                1, batch * (num_spec + 1) + 1, dtype=torch.int32, device=device
            ).view(batch, num_spec + 1)
            accepted = torch.full((batch,), num_spec + 1, dtype=torch.int32, device=device)
        else:
            cu = torch.arange(batch + 1, dtype=torch.int32, device=device)
            # Slot 0 is NULL_BLOCK_ID; live slots start at 1.
            idx = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
            accepted = None

        ref_state = base.clone()
        gems_state = base.clone()

        def ref_call(q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta, state=ref_state,
                     cu=cu, idx=idx, a_log=a_log, dt_bias=dt_bias, accepted=accepted):
            return ops["run_recurrent_kda"](
                q, k, v, raw_g, raw_beta.sigmoid(), state, cu, idx, a_log, dt_bias,
                lower_bound=GATE_LOWER_BOUND, num_accepted_tokens=accepted,
            )

        def gems_call(q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta, state=gems_state,
                      cu=cu, idx=idx, a_log=a_log, dt_bias=dt_bias, accepted=accepted):
            return ops["fused_recurrent_kda"](
                q, k, v, raw_g, raw_beta,
                scale=HEAD_DIM**-0.5,
                initial_state=state,
                inplace_final_state=True,
                cu_seqlens=cu,
                ssm_state_indices=idx,
                num_accepted_tokens=accepted,
                use_qk_l2norm_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                A_log=a_log,
                dt_bias=dt_bias,
                safe_gate=True,
                lower_bound=GATE_LOWER_BOUND,
            )

        def check(gems_call=gems_call, ref_call=ref_call,
                  a=gems_state, b=ref_state, base=base):
            # Both update state in place, so reset before each side runs.
            a.copy_(base)
            b.copy_(base)
            got, _ = gems_call()
            want = ref_call()
            torch.npu.synchronize()
            ok_o, det_o = compare("out", got, want, rtol=2e-3, atol=2e-3)
            ok_s, det_s = compare("state", a, b, rtol=2e-3, atol=2e-3)
            return ok_o and ok_s, f"{det_o} | {det_s}"

        label = f"decode bs={batch}" if not num_spec else f"spec bs={batch} k={num_spec}"
        cases.append(
            Case(
                op="recurrent_kda",
                shape=f"{label} T={tokens} H={heads} D={HEAD_DIM}",
                check=check,
                gems_step=lambda f=gems_call: f(),
                ref_step=lambda f=ref_call: f(),
                ref_label="AscendC recurrent_kda (run_recurrent_kda)",
            )
        )
    return cases


def build_conv_cases(ops, device, heads, dtype, decode: bool) -> list[Case]:
    """Short causal conv over the packed q/k/v projection (3 * H * D channels)."""
    cases = []
    dim = 3 * heads * HEAD_DIM
    shapes = ((4, 0), (32, 0), (128, 0)) if decode else ([0, 128, 384], [0, 1024], [0, 512, 1536, 4096])

    for spec in shapes:
        torch.manual_seed(77)
        if decode:
            batch = spec[0]
            tokens = batch
            qsl = torch.arange(batch + 1, dtype=torch.int32, device=device)
        else:
            cu_list = spec
            tokens = cu_list[-1]
            batch = len(cu_list) - 1
            qsl = torch.tensor(cu_list, dtype=torch.int32, device=device)

        # AscendC takes token-major x [T, dim] and a packed [W, dim] weight
        # (see kimi_kda._pack_conv_weights), with the cache as [pool, W-1, dim].
        x = torch.randn(tokens, dim, dtype=dtype, device=device)
        weight_t = torch.randn(CONV_SIZE, dim, dtype=dtype, device=device) * 0.1
        pool = max(2 * batch + 2, 64)
        conv_base = torch.randn(pool, CONV_SIZE - 1, dim, dtype=dtype, device=device)
        # Slot 0 is the null block for conv states too (null_block_id=0).
        cache_idx = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
        has_initial = torch.ones(batch, dtype=torch.bool, device=device)

        ref_state = conv_base.clone()
        gems_state = conv_base.clone()

        # FlagGems keeps the upstream vLLM contract: weight is (dim, width) and
        # conv_states (pool, dim, state_len) with a unit dim stride, so
        # transposed views of the AscendC-layout tensors satisfy both without
        # copying and both sides touch exactly the same memory.
        #
        # x differs by entry point: causal_conv1d_fn is varlen and takes
        # (dim, total_tokens), while causal_conv1d_update is per-request and
        # takes token-major [batch, dim]. Transposing x for update would make
        # it read `dim` sequences of `batch` channels.
        # Bind x/weight per case: a free variable here would make every case
        # reuse the last loop iteration's tensors.
        # Returns (x, conv_state, weight) in FlagGems' expected layouts.
        def gems_views(state, x=x, w=weight_t, decode=decode):
            gx = x if decode else x.transpose(0, 1)
            return gx, state.transpose(1, 2), w.transpose(0, 1)

        if decode:
            # FlagGems' causal_conv1d_update writes its result in place over
            # `x` and returns that same buffer, so each side must convolve its
            # own copy of the input -- otherwise whichever runs first silently
            # feeds the other a pre-convolved `x`.
            def ref_call(x=x, state=ref_state, w=weight_t, idx=cache_idx, qsl=qsl):
                return ops["ascendc_conv_update"](
                    x.clone(), state, w, bias=None, activation="silu",
                    conv_state_indices=idx, num_accepted_tokens=None,
                    query_start_loc=qsl, max_query_len=1, null_block_id=0,
                    out=torch.empty_like(x),
                )

            def gems_call(state=gems_state, idx=cache_idx, qsl=qsl, views=gems_views):
                gx, gs, gw = views(state)
                return ops["causal_conv1d_update"](
                    gx.clone(), gs, gw, bias=None, activation="silu",
                    conv_state_indices=idx, query_start_loc=qsl, max_query_len=1,
                )
            label = f"decode bs={batch}"
        else:
            def ref_call(x=x, state=ref_state, w=weight_t, idx=cache_idx,
                         qsl=qsl, has_init=has_initial):
                return ops["ascendc_conv_fn"](
                    x, w, None, conv_states=state, query_start_loc=qsl,
                    cache_indices=idx, has_initial_state=has_init,
                    activation="silu", pad_slot_id=PAD_SLOT_ID, null_block_id=0,
                )

            def gems_call(state=gems_state, idx=cache_idx, qsl=qsl,
                          has_init=has_initial, views=gems_views):
                gx, gs, gw = views(state)
                return ops["causal_conv1d_fn"](
                    gx, gw, None, conv_states=gs, query_start_loc=qsl,
                    cache_indices=idx, has_initial_state=has_init,
                    activation="silu", pad_slot_id=PAD_SLOT_ID,
                )
            label = f"prefill T={tokens} seqs={batch}"

        def check(gems_call=gems_call, ref_call=ref_call,
                  a=gems_state, b=ref_state, base=conv_base):
            a.copy_(base)
            b.copy_(base)
            got = gems_call()
            want = ref_call()
            torch.npu.synchronize()
            if isinstance(got, tuple):
                got = got[0]
            if isinstance(want, tuple):
                want = want[0]
            # FlagGems returns channel-first output; bring it back to [T, dim].
            if got.shape != want.shape and got.shape == want.shape[::-1]:
                got = got.transpose(0, 1)
            ok_o, det_o = compare("out", got, want, rtol=2e-2, atol=2e-2)
            ok_s, det_s = compare("conv_state", a, b, rtol=2e-2, atol=2e-2)
            return ok_o and ok_s, f"{det_o} | {det_s}"

        op_name = "conv_decode" if decode else "conv_prefill"
        ref_name = "causal_conv1d_update" if decode else "causal_conv1d_fn"
        cases.append(
            Case(
                op=op_name,
                shape=f"{label} dim={dim} W={CONV_SIZE}",
                check=check,
                gems_step=lambda f=gems_call: f(),
                ref_step=lambda f=ref_call: f(),
                ref_label=f"AscendC {ref_name}",
            )
        )
    return cases


def build_cases(ops, device, selected, heads, dtype) -> list[Case]:
    cases: list[Case] = []
    if "chunk_kda" in selected:
        cases += build_chunk_kda_cases(ops, device, heads, dtype)
    if "recurrent_kda" in selected:
        cases += build_recurrent_kda_cases(ops, device, heads, dtype)
    if "conv_prefill" in selected:
        cases += build_conv_cases(ops, device, heads, dtype, decode=False)
    if "conv_decode" in selected:
        cases += build_conv_cases(ops, device, heads, dtype, decode=True)
    return cases


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_report(results: list[Result], skip_perf: bool) -> None:
    width = 124
    print("\n" + "=" * width)
    print("FUNCTIONAL  (FlagGems vs the AscendC operator Kimi K3 calls)")
    print("=" * width)
    print(f"{'op':<15} {'shape':<40} {'result':<7} detail")
    print("-" * width)
    for r in results:
        verdict = "SKIP" if r.skipped else ("PASS" if r.passed else "FAIL")
        print(f"{r.op:<15} {r.shape:<40} {verdict:<7} {r.detail}")

    if not skip_perf:
        print("\n" + "=" * width)
        print("PERFORMANCE  (torch_npu.profiler, device kernel time, warmup=5 active=5)")
        print("=" * width)
        print(
            f"{'op':<15} {'shape':<40} {'gems us':>10} {'k':>3} "
            f"{'ascendc us':>11} {'k':>3} {'speedup':>8}"
        )
        print("-" * width)
        for r in results:
            if not r.passed:
                why = "unavailable" if r.skipped else "functional FAIL"
                print(f"{r.op:<15} {r.shape:<40} {f'(not timed: {why})':>35}")
                continue
            sp = r.speedup
            print(
                f"{r.op:<15} {r.shape:<40} {r.gems_us:>10.3f} {r.gems_kernels:>3} "
                f"{r.ref_us:>11.3f} {r.ref_kernels:>3} "
                f"{(f'{sp:.2f}x' if sp == sp else 'n/a'):>8}"
            )
        print("-" * width)
        print("'k' = kernels per iteration. speedup = AscendC device time / FlagGems device time.")
        print("Reference operators:")
        for label in sorted({
            f"  {r.op}: {r.ref_label}" + (f"  [{r.notes}]" if r.notes else "")
            for r in results
        }):
            print(label)

    failed = [r for r in results if not r.passed and not r.skipped]
    skipped = [r for r in results if r.skipped]
    graded = len(results) - len(skipped)
    print("\n" + "=" * width)
    print(f"SUMMARY: {graded - len(failed)}/{graded} comparable cases functionally correct"
          + (f"  ({len(skipped)} unavailable in this build)" if skipped else ""))
    if skipped:
        print("unavailable (missing dependency, not a correctness result):")
        for r in skipped:
            print(f"  {r.op} [{r.shape}] -> {r.detail}")
    if failed:
        print("failed cases:")
        for r in failed:
            print(f"  {r.op} [{r.shape}] -> {r.detail}")
    elif not skip_perf:
        ok = [r for r in results if r.speedup == r.speedup]
        if ok:
            geo = 1.0
            for r in ok:
                geo *= r.speedup
            geo **= 1.0 / len(ok)
            print(f"geometric-mean speedup over {len(ok)} cases: {geo:.2f}x")
    print("=" * width)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare FlagGems Ascend KDA ops against the AscendC operators Kimi K3 calls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ops", default="all", help=f"comma-separated subset of: {','.join(ALL_OPS)}")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tp", type=int, default=8,
                        help=f"TP size; local heads = {NUM_HEADS_TOTAL}/tp")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--skip-perf", action="store_true", help="run correctness checks only")
    parser.add_argument("--trace-root", default=None, help="profiler scratch dir (default: a temp dir)")
    args = parser.parse_args()

    if not torch.npu.is_available():
        print("FATAL: no NPU available", file=sys.stderr)
        return 2
    if args.tp <= 0 or NUM_HEADS_TOTAL % args.tp:
        print(f"FATAL: tp must divide {NUM_HEADS_TOTAL}", file=sys.stderr)
        return 2

    if args.ops == "all":
        selected = set(ALL_OPS)
    else:
        selected = {o.strip() for o in args.ops.split(",") if o.strip()}
        unknown = selected - set(ALL_OPS)
        if unknown:
            print(f"FATAL: unknown ops {sorted(unknown)}; valid: {','.join(ALL_OPS)}", file=sys.stderr)
            return 2

    ops = _import_sides()
    device = torch.device(args.device)
    torch.npu.set_device(device)
    dtype = getattr(torch, args.dtype)
    heads = NUM_HEADS_TOTAL // args.tp

    print(f"device    : {args.device} ({torch_npu.npu.get_device_name(device.index or 0)})")
    print(f"torch     : {torch.__version__}  torch_npu: {torch_npu.__version__}")
    print(f"model     : Kimi K3 KDA, head_dim={HEAD_DIM}, heads={NUM_HEADS_TOTAL} "
          f"-> local={heads} (tp={args.tp}), conv_size={CONV_SIZE}")
    print(f"ops       : {','.join(sorted(selected))}   dtype: {args.dtype}")

    owns_trace_root = args.trace_root is None
    trace_root = args.trace_root or tempfile.mkdtemp(prefix="kda_prof_")
    os.makedirs(trace_root, exist_ok=True)

    results: list[Result] = []
    try:
        # The fused TLE prefill path rejects grad mode outright.
        with torch.inference_mode():
            cases = build_cases(ops, device, selected, heads, dtype)
            print(f"cases     : {len(cases)}\n")

            for i, case in enumerate(cases, 1):
                print(f"[{i}/{len(cases)}] {case.op}  {case.shape}", flush=True)
                try:
                    passed, detail = case.check()
                except Exception as exc:  # noqa: BLE001 - classify, keep going
                    # A kernel this build cannot compile (the FlagGems conv
                    # needs FlagTree TLE) is not a correctness result either
                    # way. Triton wraps the cause in a CompilationError, so
                    # inspect the chain rather than only the raised type.
                    missing = _missing_dependency(exc)
                    if missing is not None:
                        results.append(
                            Result(case.op, case.shape, case.ref_label, False,
                                   f"unavailable: {missing}",
                                   skipped=True, notes=case.notes)
                        )
                    else:
                        traceback.print_exc()
                        results.append(
                            Result(case.op, case.shape, case.ref_label, False,
                                   f"exception during check: {type(exc).__name__}: {exc}",
                                   notes=case.notes)
                        )
                    continue

                result = Result(case.op, case.shape, case.ref_label, passed, detail, notes=case.notes)
                if passed and not args.skip_perf:
                    try:
                        tag = f"{case.op}_{i}"
                        result.gems_us, result.gems_kernels = profile_device_us(
                            case.gems_step, trace_root, f"gems_{tag}"
                        )
                        result.ref_us, result.ref_kernels = profile_device_us(
                            case.ref_step, trace_root, f"ref_{tag}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        traceback.print_exc()
                        result.detail += f" | profiling failed: {type(exc).__name__}: {exc}"
                results.append(result)
    finally:
        if owns_trace_root:
            shutil.rmtree(trace_root, ignore_errors=True)

    print_report(results, args.skip_perf)
    return 0 if all(r.passed or r.skipped for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
