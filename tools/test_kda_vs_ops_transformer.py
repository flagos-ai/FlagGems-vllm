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

"""Compare the AscendC operators Kimi K3 calls (via ``fla_npu``) against the
equivalent ``cann_ops_transformer`` (ops-transformer) operators.

Kimi K3's delta-attention layer on Ascend is
``vllm_ascend/ops/kimi_kda.py::AscendKimiK3DeltaAttention``. Its KDA math goes
through ``vllm_ascend/ops/kda.py``, which wraps **AscendC** kernels from the
installed ``fla_npu`` package. ``ops-transformer`` ships its own, separately
built AscendC implementations of the same four operators under
``cann_ops_transformer``/``cann_ops_transformer_custom``. This script treats
``fla_npu`` as the reference side (what Kimi K3 actually dispatches through
today) and times/validates ``cann_ops_transformer`` against it:

  ===================  ==========================================  ==================================
  Stage                AscendC operator (what Kimi K3 calls)        ops-transformer counterpart
  ===================  ==========================================  ==================================
  prefill KDA          ``fla_npu...chunk_kda_fwd`` via               ``cann_ops_transformer.chunk_kda_fwd``
                       ``run_chunk_kda`` (BT=64)                      (BT=64, same chunk size)
  decode / spec KDA    ``fla_npu...recurrent_kda`` via               ``cann_ops_transformer.recurrent_kda``
                       ``run_recurrent_kda``
  conv prefill         ``fla_npu...causal_conv1d_fn``                ``cann_ops_transformer.causal_conv1d_fn``
  conv decode          ``fla_npu...causal_conv1d_update``            ``cann_ops_transformer.causal_conv1d_update``
  ===================  ==========================================  ==================================

Calling-convention differences the case builders paper over (see
``tools/test_kda_vs_vllm_ascend.py`` for the FlagGems-side equivalents of
these same notes):

- ``run_chunk_kda``/``run_recurrent_kda`` require ``beta`` pre-activated by
  the caller (``raw_beta.sigmoid()``); ``chunk_kda_fwd``/``recurrent_kda``
  take the raw projection (``recurrent_kda`` can also fold the sigmoid
  in-kernel via ``use_beta_sigmoid_in_kernel``, which this script sets to
  keep the comparison on raw beta for both sides).
- ``run_chunk_kda`` wants ``cu_seqlens``/``chunk_indices`` as host Python
  lists; ``chunk_kda_fwd`` accepts the same host sequences directly.
- ``chunk_kda_fwd`` returns a fixed 12-item tuple
  (``attn_out, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h,
  initial_state``); only the first two are compared here.
- ``causal_conv1d_update`` on the ``fla_npu`` side accepts an ``out=``
  buffer for a true in-place write; ``cann_ops_transformer``'s version always
  allocates its own output, so both sides are compared as plain return
  values rather than via shared storage.
- ``cann_ops_transformer``'s ``causal_conv1d_update`` infers decode mode from
  a 3D ``x`` with ``seq_len == 1`` (``[batch, 1, dim]``); its 2D
  ``[cu_seq_len, dim]`` path is reserved for ``num_accepted_tokens``
  speculative decoding. ``fla_npu``'s plain-decode ``x`` is flat
  ``[batch, dim]``, so this reshapes a view of it to 3D for the
  ``cann_ops_transformer`` call only, and squeezes the result back before
  comparing.

Shapes default to the Kimi K3 KDA layer: ``head_dim=128``, ``num_heads=96``
sharded over TP (``--tp`` picks the local head count; 8 matches the 950DT
mixed-deployment TP), ``conv_size=4``, and a ``[pool, H, V, K]`` fp32
recurrent state cache. Request/token batch sizes follow the 950DT mixed
Prefill/Decode deployment documented in vllm-ascend's
``docs/source/tutorials/models/Kimi-K3.md`` (section 5.1, 950DT products):
``--max-num-seqs 16``, ``--max-num-batched-tokens 24576``, and DSpark
speculative decoding with ``num_speculative_tokens=7``.

Timing uses ``torch_npu.profiler`` with ``schedule(warmup=5, active=5)``,
summing every kernel's ``Duration(us)`` per iteration so a multi-kernel
implementation is charged its full device cost.

Caveat: numbers drift when many ops are swept in one process -- compare an
op against its own reference within one run, and re-measure a single op
with ``--ops`` before acting on its absolute latency.

Usage::

    # Unlike tools/test_kda_vs_vllm_ascend.py, this script needs
    # cann_ops_transformer, which this environment makes importable only via
    # a PYTHONPATH entry (e.g. $ASCEND_TOOLKIT/python/site-packages). Extend
    # $PYTHONPATH rather than overwriting it, or that entry is dropped and
    # `import cann_ops_transformer` fails with a plain ModuleNotFoundError.
    PYTHONPATH=src:FlagGems/src:$PYTHONPATH python tools/test_kda_vs_ops_transformer.py
    PYTHONPATH=...:$PYTHONPATH python tools/test_kda_vs_ops_transformer.py --ops recurrent_kda
    PYTHONPATH=...:$PYTHONPATH python tools/test_kda_vs_ops_transformer.py --skip-perf --tp 8

Process isolation for conv_prefill/conv_decode: ``cann_ops_transformer``'s
public ``causal_conv1d_fn``/``causal_conv1d_update`` dispatch through CANN's
own built-in ``CausalConv1d`` AscendC op (shipped in ``libopapi.so``) -- a
working, separately-built-in op, nothing this script or ops-transformer needs
to (re)build. ``fla_npu`` ships its *own*, differently-shaped ``CausalConv1d``
op (12 inputs, vs. the built-in op's 8) under its wheel-embedded vendor OPP,
registered via ``ASCEND_CUSTOM_OPP_PATH`` at ``import fla_npu.ops.ascendc``
time. Ascend's native op registry binds the op-type name ``CausalConv1d`` to
one implementation per process on first reference (import, not even first
launch), so importing both packages in one interpreter always resolves it to
whichever package was imported first; there is no in-process ordering or
env-var trick that lets both resolve correctly afterward. conv_prefill/
conv_decode therefore run each side (ref =
fla_npu, ot = cann_ops_transformer) in its own subprocess via this script's
``--internal-conv-worker`` mode (see ``_build_conv_tensors``/
``_run_conv_worker``); the parent spawns both, loads back the saved tensors
and timing, and compares/report as usual. chunk_kda_fwd and recurrent_kda are
unaffected (no such name collision) and still run fully in-process.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess
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

# Kimi K3 KDA layer geometry (vllm-ascend tests/e2e/pull_request/four_card/
# test_kimi_k3.py's linear_attn_config, which keeps production tensor widths:
# head_dim=128, num_heads=96, short_conv_kernel_size=4).
HEAD_DIM = 128
NUM_HEADS_TOTAL = 96
CONV_SIZE = 4
GATE_LOWER_BOUND = -5.0
ASCENDC_CHUNK_SIZE = 64  # vllm_ascend.ops.kda.KDA_CHUNK_SIZE; ops-transformer defaults to the same BT.
NULL_BLOCK_ID = 0  # vllm.v1.attention.backends.utils.NULL_BLOCK_ID
PAD_SLOT_ID = -1

# Ascend 950DT mixed Prefill/Decode deployment (vllm-ascend
# docs/source/tutorials/models/Kimi-K3.md section 5.1, 950DT products,
# four-node, Node 0 command): TP=8, --max-num-seqs 16,
# --max-num-batched-tokens 24576, --max-model-len 131072, DSpark speculative
# decoding with num_speculative_tokens=7.
K3_950DT_MAX_NUM_SEQS = 16
K3_950DT_MAX_NUM_BATCHED_TOKENS = 24576
K3_950DT_MAX_MODEL_LEN = 131072
K3_950DT_NUM_SPECULATIVE_TOKENS = 7

WARMUP = 5
ACTIVE = 5

ALL_OPS = ("chunk_kda", "recurrent_kda", "conv_prefill", "conv_decode")

#  (T, H, seq) chunk prefill cases, zipped (not a cartesian product):
#  common chunk-prefill token budgets paired with the TP-sharded local head
#  counts (tp=4/8/16 of NUM_HEADS_TOTAL=96) and small request counts. The
#  4th T re-cycles the first (H, seq) pair since the lists have different
#  lengths. Matches tools/test_kda_vs_vllm_ascend.py's sweep exactly, so the
#  two scripts' reports cover the same shapes.
CHUNK_KDA_T_VALUES = (8192, 16384, 24576, 32768)
CHUNK_KDA_H_VALUES = (24, 12, 6)
CHUNK_KDA_SEQ_VALUES = (2, 4, 8)


def _missing_dependency(exc: BaseException) -> str | None:
    """Return the message if a failure is really an absent build dependency.

    Besides an absent Python module, a ``cann_ops_transformer`` install that
    is missing a device/dtype's compiled AscendC kernel binary raises a
    ``RuntimeError`` with "Cannot find binary for op" rather than an
    ImportError -- this is an install/build gap (this op's binaries were
    never compiled for this device), not a wrong result, so it is reported
    the same way.
    """
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ModuleNotFoundError, ImportError)):
            return str(current)
        current = current.__cause__ or current.__context__
    text = str(exc)
    if "ModuleNotFoundError" in text and "is not installed" in text:
        return text.split("ModuleNotFoundError", 1)[1].strip(" (\"')\n")
    if "Cannot find binary for op" in text or "Cannot find any bin for" in text:
        return "cann_ops_transformer install is missing a compiled kernel binary for this op/dtype/shape"
    return None


def _import_sides():
    """Import both operator sets, reporting a usable message if either is absent.

    Deliberately omits fla_npu.ops.ascendc's and cann_ops_transformer's conv
    functions: merely *importing* fla_npu.ops.ascendc.causal_conv1d_fn (never
    even calling it) is enough to bind the native op-type name "CausalConv1d"
    to fla_npu's 12-input kernel for the rest of the process, after which
    cann_ops_transformer.causal_conv1d_fn/update can never resolve to the
    correct built-in op again -- see the module docstring's "Process
    isolation" note. build_conv_cases instead runs each side in its own
    subprocess via --internal-conv-worker, which imports only one package.
    """
    try:
        from vllm_ascend.ops.kda import run_chunk_kda, run_recurrent_kda
    except ImportError as exc:
        print(
            f"FATAL: cannot import the AscendC KDA operators ({exc}).\n"
            "This comparison needs fla_npu and vllm_ascend installed.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        from cann_ops_transformer.ops import (
            chunk_kda_fwd as ot_chunk_kda_fwd,
            recurrent_kda as ot_recurrent_kda,
        )
    except ImportError as exc:
        print(
            f"FATAL: cannot import cann_ops_transformer ({exc}).\n"
            "Install the ops-transformer torch_extension build, e.g.\n"
            "  bash ops-transformer/build.sh --torch_extension\n"
            "  python3 -m pip install ops-transformer/build_out/cann_ops_transformer*.whl",
            file=sys.stderr,
        )
        sys.exit(2)

    return dict(
        run_chunk_kda=run_chunk_kda,
        run_recurrent_kda=run_recurrent_kda,
        ot_chunk_kda_fwd=ot_chunk_kda_fwd,
        ot_recurrent_kda=ot_recurrent_kda,
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
    ot_step: Callable[[], None]
    ref_step: Callable[[], None]
    ref_label: str
    notes: str = ""
    # conv_prefill/conv_decode run ref/ot in separate subprocesses (see the
    # module docstring's "Process isolation" note) and already have timing
    # from that run; perf_result, if set, short-circuits main()'s normal
    # profile_device_us(ot_step)/profile_device_us(ref_step) calls with
    # (ot_us, ot_kernels, ref_us, ref_kernels) gathered during check().
    perf_result: Callable[[], tuple[float, int, float, int]] | None = None


@dataclass
class Result:
    op: str
    shape: str
    ref_label: str
    passed: bool
    detail: str
    # Unavailable in this environment (e.g. a kernel needing a build this
    # environment lacks). Reported separately so a missing dependency is not
    # read as a wrong result.
    skipped: bool = False
    ot_us: float = float("nan")
    ref_us: float = float("nan")
    ot_kernels: int = 0
    ref_kernels: int = 0
    notes: str = ""

    @property
    def speedup(self) -> float:
        if self.ref_us != self.ref_us or self.ot_us != self.ot_us or self.ot_us <= 0:
            return float("nan")
        return self.ref_us / self.ot_us


def compare(name: str, got: torch.Tensor, want: torch.Tensor, rtol: float, atol: float) -> tuple[bool, str]:
    """Compare on CPU in fp32 and report the worst deviation either way."""
    got_c, want_c = got.detach().float().cpu(), want.detach().float().cpu()
    if got_c.shape != want_c.shape:
        return False, f"{name}: shape {tuple(got_c.shape)} != {tuple(want_c.shape)}"
    if not torch.isfinite(got_c).all():
        return False, f"{name}: non-finite values in ops-transformer output"
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
    """Prefill KDA: ops-transformer chunk_kda_fwd vs AscendC chunk_kda_fwd (fla_npu).

    Both sides use chunk_size=64 (vllm_ascend.ops.kda.KDA_CHUNK_SIZE and
    cann_ops_transformer.chunk_kda_fwd's default), so -- unlike the FlagGems
    Triton comparison in tools/test_kda_vs_vllm_ascend.py -- there is no
    chunking-order mismatch to tolerate; this uses a tighter comparison
    tolerance accordingly.

    Shapes sweep common chunk-prefill sizes rather than the heads/batch
    implied by ``--tp``: T in {8k,16k,24k,32k}, H in {24,12,6}, seq in
    {2,4,8}, zipped pairwise (cycling the shorter lists) rather than a full
    cross product -- the same sweep as
    ``tools/test_kda_vs_vllm_ascend.py::build_chunk_kda_cases``, so both
    scripts report on identical shapes. Each case's T tokens are split evenly
    across `seq` chunk-aligned requests.
    """
    cases = []
    for i, tokens in enumerate(CHUNK_KDA_T_VALUES):
        case_heads = CHUNK_KDA_H_VALUES[i % len(CHUNK_KDA_H_VALUES)]
        n = CHUNK_KDA_SEQ_VALUES[i % len(CHUNK_KDA_SEQ_VALUES)]
        # Chunk-align each request's length so both sides see whole chunks
        # per request.
        per_seq = (tokens // n // ASCENDC_CHUNK_SIZE) * ASCENDC_CHUNK_SIZE
        cu_list = [j * per_seq for j in range(n)] + [tokens]

        q, k, v, raw_g, raw_beta, a_log, dt_bias = _kda_inputs(case_heads, tokens, device, dtype)
        cu = torch.tensor(cu_list, dtype=torch.int64, device=device)
        # state_v_first=True: the recurrent cache stores [N, H, V, K].
        init_vk = torch.randn(n, case_heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device)

        # run_chunk_kda types cu_seqlens/chunk_indices as Sequence[int] and
        # pre-activates beta on the host; honor that exact contract.
        from flaggems_vllm.ops.FLA.index import prepare_chunk_indices

        ci = prepare_chunk_indices(cu, ASCENDC_CHUNK_SIZE)
        cu_host = list(cu_list)
        ci_host = ci.flatten().cpu().tolist()

        def ref_call(q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta, init=init_vk,
                     cu_host=cu_host, ci_host=ci_host, a_log=a_log, dt_bias=dt_bias):
            return ops["run_chunk_kda"](
                q, k, v, raw_g, raw_beta.sigmoid(), init.clone(),
                cu_host, ci_host, a_log, dt_bias, lower_bound=GATE_LOWER_BOUND,
            )

        # cann_ops_transformer.chunk_kda_fwd has no use_qk_l2norm_in_kernel
        # knob at all (unlike recurrent_kda) -- run_chunk_kda (vllm_ascend)
        # is a thin wrapper that L2-normalizes q/k on the host via
        # vllm.third_party.flash_linear_attention.ops.l2norm.l2norm_fwd
        # before calling this exact op, so this must call the same helper
        # or the comparison is against un-normalized q/k.
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

        def ot_call(q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta, init=init_vk,
                    cu_host=cu_host, ci_host=ci_host, a_log=a_log, dt_bias=dt_bias):
            outputs = ops["ot_chunk_kda_fwd"](
                l2norm_fwd(q.contiguous()), l2norm_fwd(k.contiguous()), v, raw_g,
                raw_beta.sigmoid(), HEAD_DIM**-0.5,
                initial_state=init.clone(),
                output_final_state=True,
                safe_gate=True,
                lower_bound=GATE_LOWER_BOUND,
                use_gate_in_kernel=True,
                A_log=a_log,
                dt_bias=dt_bias,
                chunk_size=ASCENDC_CHUNK_SIZE,
                cu_seqlens=cu_host,
                chunk_indices=ci_host,
                state_v_first=True,
                layout="BSND",
            )
            attn_out, final_state = outputs[0], outputs[1]
            return attn_out, final_state

        def check(ot_call=ot_call, ref_call=ref_call):
            got_o, got_s = ot_call()
            want_o, want_s = ref_call()
            torch.npu.synchronize()
            # Both sides use the identical BT=64 chunking, so this should be
            # close to bf16 round-trip precision rather than needing the
            # looser tolerance a BT=16-vs-BT=64 mismatch would require.
            ok_o, det_o = compare("out", got_o, want_o, rtol=1e-2, atol=1e-2)
            ok_s, det_s = compare("state", got_s, want_s, rtol=1e-2, atol=1e-2)
            return ok_o and ok_s, f"{det_o} | {det_s}"

        cases.append(
            Case(
                op="chunk_kda",
                shape=f"T={tokens} H={case_heads} D={HEAD_DIM} seqs={n}",
                check=check,
                ot_step=lambda f=ot_call: f(),
                ref_step=lambda f=ref_call: f(),
                ref_label="AscendC chunk_kda_fwd (fla_npu, via run_chunk_kda)",
                notes=f"both BT={ASCENDC_CHUNK_SIZE}",
            )
        )
    return cases


def build_recurrent_kda_cases(ops, device, heads, dtype) -> list[Case]:
    """Decode / speculative KDA: ops-transformer recurrent_kda vs AscendC recurrent_kda (fla_npu).

    Decode batches are bounded by the 950DT mixed-deployment
    --max-num-seqs=16 (one DP rank serves at most 16 concurrent requests);
    the speculative case uses the deployed DSpark num_speculative_tokens=7.
    """
    cases = []
    # (decode batch, speculative tokens per request); num_spec=0 is plain decode.
    max_seqs = K3_950DT_MAX_NUM_SEQS
    for batch, num_spec in (
        (4, 0),
        (max_seqs // 2, 0),
        (max_seqs, 0),
        (max_seqs, K3_950DT_NUM_SPECULATIVE_TOKENS),
    ):
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
        ot_state = base.clone()

        # run_recurrent_kda takes cu_seqlens/ssm_state_indices/a_log/dt_bias
        # as positional args and requires beta pre-activated by the caller.
        def ref_call(q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta, state=ref_state,
                     cu=cu, idx=idx, a_log=a_log, dt_bias=dt_bias, accepted=accepted):
            return ops["run_recurrent_kda"](
                q, k, v, raw_g, raw_beta.sigmoid(), state, cu, idx, a_log, dt_bias,
                lower_bound=GATE_LOWER_BOUND, num_accepted_tokens=accepted,
            )

        # cann_ops_transformer.recurrent_kda keeps the same parameter names
        # as FlagGems' Triton fused_recurrent_kda (use_qk_l2norm_in_kernel,
        # use_beta_sigmoid_in_kernel, inplace_final_state, state_v_first,
        # ...), all kwonly -- run_recurrent_kda (vllm_ascend) is in fact a
        # thin wrapper calling this exact fla_npu op with
        # state_v_first=True, use_beta_sigmoid_in_kernel=False, and a
        # host-presigmoided beta, so mirror that contract here rather than
        # letting the kernel sigmoid raw beta a second time.
        def ot_call(q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta, state=ot_state,
                    cu=cu, idx=idx, a_log=a_log, dt_bias=dt_bias, accepted=accepted):
            return ops["ot_recurrent_kda"](
                q, k, v, raw_g, raw_beta.sigmoid(), state,
                cu_seqlens=cu,
                ssm_state_indices=idx,
                A_log=a_log,
                dt_bias=dt_bias,
                num_accepted_tokens=accepted,
                layout="BSND",
                scale=HEAD_DIM**-0.5,
                output_final_state=True,
                inplace_final_state=True,
                state_v_first=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=False,
                safe_gate=True,
                lower_bound=GATE_LOWER_BOUND,
            )

        def check(ot_call=ot_call, ref_call=ref_call,
                  a=ot_state, b=ref_state, base=base):
            # Both update state in place, so reset before each side runs.
            a.copy_(base)
            b.copy_(base)
            got, _ = ot_call()
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
                ot_step=lambda f=ot_call: f(),
                ref_step=lambda f=ref_call: f(),
                ref_label="AscendC recurrent_kda (fla_npu, via run_recurrent_kda)",
            )
        )
    return cases


def _conv_shapes(heads, decode: bool) -> tuple[tuple[int, int], ...] | tuple[tuple[int, int, int], ...]:
    """Shape sweep shared by build_conv_cases and the subprocess worker.

    Decode batches are bounded by the 950DT mixed-deployment
    --max-num-seqs=16. Prefill shapes follow the same (T, H, seq) chunk
    prefill sweep as build_chunk_kda_cases: T in {8k,16k,24k,32k} zipped
    with H in {24,12,6} and seq in {2,4,8} (not a cartesian product) -- the
    same sweep as tools/test_kda_vs_vllm_ascend.py::build_conv_cases, so both
    scripts report on identical shapes.
    """
    max_seqs = K3_950DT_MAX_NUM_SEQS
    if decode:
        return ((4, heads), (max_seqs // 2, heads), (max_seqs, heads))
    n_prefill = len(CHUNK_KDA_T_VALUES)
    return tuple(
        (CHUNK_KDA_T_VALUES[i], CHUNK_KDA_H_VALUES[i % len(CHUNK_KDA_H_VALUES)],
         CHUNK_KDA_SEQ_VALUES[i % len(CHUNK_KDA_SEQ_VALUES)])
        for i in range(n_prefill)
    )


def _build_conv_tensors(decode: bool, spec, dtype, device):
    """Deterministically rebuild one conv case's tensors from (decode, spec).

    Both the in-process case enumeration and the out-of-process
    --internal-conv-worker call this so ref and ot see byte-identical inputs
    despite running in separate interpreters (same seed, same shape, same
    construction order -- no tensor is ever passed across the process
    boundary). Returns a dict with every tensor/metadata field build_conv_cases
    and _run_conv_worker need.
    """
    torch.manual_seed(77)
    if decode:
        batch, case_heads = spec
        tokens = batch
        qsl = torch.arange(batch + 1, dtype=torch.int32, device=device)
    else:
        tokens, case_heads, n = spec
        # Chunk-align each request's length, matching build_chunk_kda_cases.
        per_seq = (tokens // n // ASCENDC_CHUNK_SIZE) * ASCENDC_CHUNK_SIZE
        cu_list = [j * per_seq for j in range(n)] + [tokens]
        batch = n
        qsl = torch.tensor(cu_list, dtype=torch.int32, device=device)

    dim = 3 * case_heads * HEAD_DIM
    # Both sides take token-major x [T, dim] and a (kW, dim) weight (see
    # kimi_kda._pack_conv_weights), with the cache as [pool, kW-1, dim].
    x = torch.randn(tokens, dim, dtype=dtype, device=device)
    weight = torch.randn(CONV_SIZE, dim, dtype=dtype, device=device) * 0.1
    pool = max(2 * batch + 2, 64)
    conv_base = torch.randn(pool, CONV_SIZE - 1, dim, dtype=dtype, device=device)
    # Slot 0 is the null block for conv states too (null_block_id=0).
    cache_idx = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    # fla_npu's has_initial_state is bool (1=use cached history); the
    # cann_ops_transformer op documents the identical 1/0 semantics but
    # requires int32 ("initialStateModeOptional dtype must be INT32").
    has_initial = torch.ones(batch, dtype=torch.bool, device=device)

    label = f"decode bs={batch}" if decode else f"prefill T={tokens} seqs={batch}"
    return dict(
        tokens=tokens, batch=batch, dim=dim, label=label,
        x=x, weight=weight, conv_base=conv_base, cache_idx=cache_idx,
        has_initial=has_initial, qsl=qsl,
    )


def _run_conv_worker(side: str, decode: bool, spec, dtype_name: str, device_str: str,
                      skip_perf: bool, out_path: str) -> None:
    """Build one conv case and run exactly one side (ref=fla_npu, ot=cann_ops_transformer).

    Entry point for ``--internal-conv-worker``; never called in-process.
    Importing only one of {fla_npu, cann_ops_transformer} per subprocess is
    the whole point -- see the module docstring's "Process isolation" note
    for why both can't be imported together and get correct results.
    """
    if side not in ("ref", "ot"):
        print(f"FATAL: --internal-conv-worker side must be ref|ot, got {side!r}", file=sys.stderr)
        sys.exit(2)

    device = torch.device(device_str)
    torch.npu.set_device(device)
    dtype = getattr(torch, dtype_name)
    t = _build_conv_tensors(decode, spec, dtype, device)
    x, weight, conv_base = t["x"], t["weight"], t["conv_base"]
    cache_idx, has_initial, qsl = t["cache_idx"], t["has_initial"], t["qsl"]
    state = conv_base.clone()

    if side == "ref":
        from fla_npu.ops.ascendc import causal_conv1d_fn as ascendc_conv_fn, causal_conv1d_update as ascendc_conv_update
        if decode:
            def call(x=x, state=state, w=weight, idx=cache_idx, qsl=qsl):
                return ascendc_conv_update(
                    x.clone(), state, w, bias=None, activation="silu",
                    conv_state_indices=idx, num_accepted_tokens=None,
                    query_start_loc=qsl, max_query_len=1, null_block_id=0,
                    out=torch.empty_like(x),
                )
        else:
            def call(x=x, state=state, w=weight, idx=cache_idx, qsl=qsl, has_init=has_initial):
                return ascendc_conv_fn(
                    x, w, None, conv_states=state, query_start_loc=qsl,
                    cache_indices=idx, has_initial_state=has_init,
                    activation="silu", pad_slot_id=PAD_SLOT_ID, null_block_id=0,
                )
    else:
        import cann_ops_transformer
        has_initial_i32 = has_initial.to(torch.int32)
        if decode:
            # cann_ops_transformer's causal_conv1d_update infers decode mode
            # from a 3D x with seq_len==1 ([batch, 1, dim]); the 2D
            # [cu_seq_len, dim] path is reserved for num_accepted_tokens
            # speculative decoding. Reshape fla_npu's flat [batch, dim] x
            # into the documented 3D decode shape for this side only --
            # same storage, just a view.
            def call(x=x, state=state, w=weight, idx=cache_idx, qsl=qsl):
                return cann_ops_transformer.causal_conv1d_update(
                    x.clone().unsqueeze(1).contiguous(), state, w, bias=None, activation="silu",
                    conv_state_indices=idx, query_start_loc=qsl, max_query_len=1,
                    null_block_id=0,
                )
        else:
            # cann_ops_transformer's causal_conv1d_fn takes bias as a
            # required positional (pass None explicitly -- it has no default).
            def call(x=x, state=state, w=weight, idx=cache_idx, qsl=qsl, has_init=has_initial_i32):
                return cann_ops_transformer.causal_conv1d_fn(
                    x, w, None, state, qsl,
                    cache_indices=idx, has_initial_state=has_init,
                    activation="silu", pad_slot_id=PAD_SLOT_ID, null_block_id=0,
                )

    def step():
        state.copy_(conv_base)
        out = call()
        if isinstance(out, tuple):
            out = out[0]
        return out

    out = step()
    torch.npu.synchronize()
    # decode's ot call feeds a 3D [batch, 1, dim] x (see above); drop the
    # singleton seq dim so the parent compares against fla_npu's 2D output.
    if side == "ot" and decode and out.dim() == 3 and out.shape[1] == 1:
        out = out.squeeze(1)

    us, kernels = float("nan"), 0
    if not skip_perf:
        trace_root = tempfile.mkdtemp(prefix=f"kda_ot_prof_conv_worker_{side}_")
        try:
            us, kernels = profile_device_us(step, trace_root, side)
        finally:
            shutil.rmtree(trace_root, ignore_errors=True)
            state.copy_(conv_base)
            out2 = step()
            torch.npu.synchronize()
            if side == "ot" and decode and out2.dim() == 3 and out2.shape[1] == 1:
                out2 = out2.squeeze(1)
            out = out2  # use the post-profiling run's output for the saved tensors

    torch.save(
        {"out": out.detach().cpu(), "state": state.detach().cpu(), "us": us, "kernels": kernels},
        out_path,
    )


def build_conv_cases(ops, device, heads, dtype, decode: bool) -> list[Case]:
    """Short causal conv over the packed q/k/v projection (3 * H * D channels).

    conv_prefill/conv_decode's ref (fla_npu) and ot (cann_ops_transformer)
    sides run in separate subprocesses -- see the module docstring's
    "Process isolation" note. ``ops["ascendc_conv_fn"]``/``ops["ot_conv_fn"]``
    etc. are unused here; ``_run_conv_worker`` re-imports each side itself.
    """
    cases = []
    shapes = _conv_shapes(heads, decode)

    dtype_name = "bfloat16" if dtype is torch.bfloat16 else "float16"
    device_str = str(device)

    for spec in shapes:
        # Metadata only (label/dim/batch/tokens) -- the actual tensors are
        # rebuilt independently inside each subprocess by _build_conv_tensors.
        meta = _build_conv_tensors(decode, spec, dtype, torch.device("cpu"))
        dim, label = meta["dim"], meta["label"]

        def run_subprocess(side, spec=spec, skip_perf=False):
            with tempfile.NamedTemporaryFile(suffix=f"_{side}.pt", delete=False) as f:
                out_path = f.name
            try:
                cmd = [
                    sys.executable, os.path.abspath(__file__),
                    "--internal-conv-worker", side,
                    "--internal-conv-decode", "1" if decode else "0",
                    "--internal-conv-spec", repr(spec),
                    "--internal-conv-out", out_path,
                    "--dtype", dtype_name,
                    "--device", device_str,
                ]
                if skip_perf:
                    cmd.append("--skip-perf")
                env = dict(os.environ)
                if side == "ot":
                    # fla_npu's site .pth hook prepends its own wheel-embedded
                    # vendor OPP to ASCEND_CUSTOM_OPP_PATH at interpreter
                    # startup, unconditionally, before any of this script's
                    # code runs -- even though this worker never imports
                    # fla_npu. Left enabled, it binds the native op-type name
                    # "CausalConv1d" to fla_npu's 12-input kernel instead of
                    # the correct CANN built-in (8-input) one that
                    # cann_ops_transformer.causal_conv1d_fn/update dispatch
                    # through. The ref side needs the opposite: fla_npu's own
                    # docs warn its ASCEND_CUSTOM_OPP_PATH prepend can be "too
                    # late" if CANN's runtime initializes first, so the .pth
                    # (which runs before even `import torch_npu`) must stay
                    # enabled there.
                    #
                    # FLA_NPU_DISABLE_PTH alone stops the *child's* .pth from
                    # running, but env=dict(os.environ) above already copied
                    # forward whatever ASCEND_CUSTOM_OPP_PATH *this* process
                    # picked up at its own startup (e.g. if this orchestrator
                    # was itself launched without FLA_NPU_DISABLE_PTH=1, as
                    # it will be for every normal `python
                    # test_kda_vs_ops_transformer.py` invocation) -- the
                    # disable flag can't retroactively un-poison a value
                    # that's already in the copied dict, so pop it too.
                    env["FLA_NPU_DISABLE_PTH"] = "1"
                    env.pop("ASCEND_CUSTOM_OPP_PATH", None)
                else:
                    env.pop("FLA_NPU_DISABLE_PTH", None)
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if proc.returncode != 0 or not os.path.exists(out_path):
                    raise RuntimeError(
                        f"conv worker ({side}) failed, exit {proc.returncode}:\n"
                        f"{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
                    )
                return torch.load(out_path, map_location="cpu")
            finally:
                try:
                    os.unlink(out_path)
                except OSError:
                    pass

        def check(spec=spec):
            ref = run_subprocess("ref", spec, skip_perf=True)
            ot = run_subprocess("ot", spec, skip_perf=True)
            ok_o, det_o = compare("out", ot["out"], ref["out"], rtol=2e-2, atol=2e-2)
            ok_s, det_s = compare("conv_state", ot["state"], ref["state"], rtol=2e-2, atol=2e-2)
            # perf_result() below reruns both subprocesses with profiling on
            # if the case passes and --skip-perf wasn't given; this keeps
            # check() itself fast and side-effect-free.
            return ok_o and ok_s, f"{det_o} | {det_s}"

        def perf_result(spec=spec):
            ref = run_subprocess("ref", spec, skip_perf=False)
            ot = run_subprocess("ot", spec, skip_perf=False)
            return ot["us"], ot["kernels"], ref["us"], ref["kernels"]

        op_name = "conv_decode" if decode else "conv_prefill"
        ref_name = "causal_conv1d_update" if decode else "causal_conv1d_fn"
        cases.append(
            Case(
                op=op_name,
                shape=f"{label} dim={dim} W={CONV_SIZE}",
                check=check,
                ot_step=lambda: None,
                ref_step=lambda: None,
                ref_label=f"AscendC {ref_name} (fla_npu)",
                perf_result=perf_result,
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
    print("FUNCTIONAL  (ops-transformer vs the AscendC operator Kimi K3 calls via fla_npu)")
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
            f"{'op':<15} {'shape':<40} {'ot us':>10} {'k':>3} "
            f"{'fla_npu us':>11} {'k':>3} {'speedup':>8}"
        )
        print("-" * width)
        for r in results:
            if not r.passed:
                why = "unavailable" if r.skipped else "functional FAIL"
                print(f"{r.op:<15} {r.shape:<40} {f'(not timed: {why})':>35}")
                continue
            sp = r.speedup
            print(
                f"{r.op:<15} {r.shape:<40} {r.ot_us:>10.3f} {r.ot_kernels:>3} "
                f"{r.ref_us:>11.3f} {r.ref_kernels:>3} "
                f"{(f'{sp:.2f}x' if sp == sp else 'n/a'):>8}"
            )
        print("-" * width)
        print("'k' = kernels per iteration. speedup = fla_npu (AscendC) device time / ops-transformer device time.")
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
        description="Compare ops-transformer Ascend KDA ops against the AscendC operators Kimi K3 calls (fla_npu).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ops", default="all", help=f"comma-separated subset of: {','.join(ALL_OPS)}")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tp", type=int, default=8,
                        help=f"TP size; local heads = {NUM_HEADS_TOTAL}/tp")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--skip-perf", action="store_true", help="run correctness checks only")
    parser.add_argument("--trace-root", default=None, help="profiler scratch dir (default: a temp dir)")
    # Internal re-entry point used by build_conv_cases to run the ref
    # (fla_npu) or ot (cann_ops_transformer) side of conv_prefill/conv_decode
    # in its own subprocess -- see the module docstring's "Process isolation"
    # note. Not meant to be passed by hand.
    parser.add_argument("--internal-conv-worker", default=None, choices=["ref", "ot"], help=argparse.SUPPRESS)
    parser.add_argument("--internal-conv-decode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-conv-spec", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-conv-out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.internal_conv_worker is not None:
        if not torch.npu.is_available():
            print("FATAL: no NPU available", file=sys.stderr)
            return 2
        import ast
        _run_conv_worker(
            args.internal_conv_worker,
            args.internal_conv_decode == "1",
            ast.literal_eval(args.internal_conv_spec),
            args.dtype,
            args.device,
            args.skip_perf,
            args.internal_conv_out,
        )
        return 0

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
    trace_root = args.trace_root or tempfile.mkdtemp(prefix="kda_ot_prof_")
    os.makedirs(trace_root, exist_ok=True)

    results: list[Result] = []
    try:
        # The fused kernels reject grad mode outright.
        with torch.inference_mode():
            cases = build_cases(ops, device, selected, heads, dtype)
            print(f"cases     : {len(cases)}\n")

            for i, case in enumerate(cases, 1):
                print(f"[{i}/{len(cases)}] {case.op}  {case.shape}", flush=True)
                try:
                    passed, detail = case.check()
                except Exception as exc:  # noqa: BLE001 - classify, keep going
                    # A kernel this build cannot run (e.g. an unbuilt
                    # ops-transformer op) is not a correctness result either
                    # way.
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
                        if case.perf_result is not None:
                            (result.ot_us, result.ot_kernels,
                             result.ref_us, result.ref_kernels) = case.perf_result()
                        else:
                            tag = f"{case.op}_{i}"
                            result.ot_us, result.ot_kernels = profile_device_us(
                                case.ot_step, trace_root, f"ot_{tag}"
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
