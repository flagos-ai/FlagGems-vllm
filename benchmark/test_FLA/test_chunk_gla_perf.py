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

import pytest
import torch
import torch.nn.functional as F
import triton

import flaggems_vllm
from benchmark.conftest import Config
from flaggems_vllm.ops.FLA.chunk_gla import chunk_gla as gems_chunk_gla

# optional FLA reference
_HAS_FLA_CHUNK = False
_fla_chunk_gla = None

try:
    from fla.ops.gla import chunk_gla as _fla_chunk_gla

    _HAS_FLA_CHUNK = True
except Exception:
    _HAS_FLA_CHUNK = False


def _fla_chunk_wrapper(q, k, v, g, **kwargs):
    if not _HAS_FLA_CHUNK:
        raise RuntimeError("fla chunk_gla is unavailable")

    return _fla_chunk_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=kwargs.get("scale", None),
        initial_state=kwargs.get("initial_state", None),
        output_final_state=kwargs.get("output_final_state", False),
        state_v_first=kwargs.get("state_v_first", False),
        cu_seqlens=kwargs.get("cu_seqlens", None),
    )


def _precompile():
    """
    Force:
    1. Triton JIT compile
    2. autotune cache population
    3. CUDA context stabilization
    """
    print("[precompile] warming up kernels & autotune cache ...")

    device = flaggems_vllm.device
    dtype = torch.float16

    B, T, H, D = 1, 512, 8, 64

    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, D, device=device, dtype=dtype))

    kwargs = {
        "scale": D**-0.5,
        "initial_state": None,
        "output_final_state": False,
        "state_v_first": False,
        "cu_seqlens": None,
        "cu_seqlens_cpu": None,
    }

    # run multiple times to fully warm autotune
    for _ in range(5):
        if _HAS_FLA_CHUNK:
            _fla_chunk_wrapper(q, k, v, g, **kwargs)
        gems_chunk_gla(q, k, v, g, **kwargs)

    torch.cuda.synchronize()


def _bench_ms(fn):
    if Config.mode.value == "kernel":
        return triton.testing.do_bench(
            fn,
            warmup=Config.warm_up,
            rep=Config.repetition,
            return_mode="median",
        )


def _build_inputs(B, T, H, D, dtype, requires_grad=False):
    device = flaggems_vllm.device
    q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    g_logit = torch.randn(
        B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad
    )

    kwargs = {
        "scale": D**-0.5,
        "initial_state": None,
        "output_final_state": False,
        "state_v_first": False,
        "cu_seqlens": None,
        "cu_seqlens_cpu": None,
    }
    if requires_grad:
        return q, k, v, g_logit, kwargs
    else:
        g = F.logsigmoid(g_logit)
        return q, k, v, g, kwargs


# ---------------------------
# fwd+bwd benchmark helper
# ---------------------------
def _bench_fwd_bwd_ms(fn, q, k, v, g_logit, kwargs):
    """Measure forward + backward pass time for a chunk_gla function.

    ``g_logit`` is the pre-logsigmoid raw tensor (a leaf).  F.logsigmoid is
    called *inside* the timed closure so each backward iteration gets a fresh
    computation graph.
    """

    params = [q, k, v, g_logit]

    def _fwd_bwd():
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        g = F.logsigmoid(g_logit)
        out = fn(q, k, v, g, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        loss.backward()

    if Config.mode.value == "kernel":
        return triton.testing.do_bench(
            _fwd_bwd,
            warmup=Config.warm_up,
            rep=Config.repetition,
            return_mode="median",
        )


_SHAPES = [
    # (1, 4096, 32, 512),
    # (2, 2048, 16, 512),
    (1, 8192, 96, 128),
    (2, 16384, 16, 128),
    (4, 2048, 16, 128),
    (4, 4096, 64, 128),
    (8, 2048, 32, 256),
    (2, 2048, 16, 512),
    (4, 1024, 8, 512),
    (8, 1024, 8, 64),
]

_DTYPES = [
    torch.bfloat16,
    # torch.float32,
    # torch.float16,
]


def _print_header(title, subtitle):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"  {subtitle}")
    print(
        f"  mode={Config.mode.value}  warmup={Config.warm_up}  iter={Config.repetition}"
    )
    print(f"{'=' * 70}")

    if _HAS_FLA_CHUNK:
        print(
            f"{'B':>3} {'T':>6} {'H':>4} {'D':>4} {'dtype':>8} "
            f"{'fla(ms)':>10} {'gems(ms)':>10} {'gems/fla':>10}"
        )
    else:
        print(f"{'B':>3} {'T':>6} {'H':>4} {'D':>4} {'dtype':>8} " f"{'gems(ms)':>10}")


def _print_row(B, T, H, D, dtype, ms_fla, ms_gems):
    dtype_str = str(dtype).split(".")[-1]
    if _HAS_FLA_CHUNK:
        speedup = ms_fla / ms_gems if ms_gems > 0 else float("inf")
        print(
            f"{B:>3} {T:>6} {H:>4} {D:>4} {dtype_str:>8} "
            f"{ms_fla:>10.3f} {ms_gems:>10.3f} {speedup:>10.2f}x"
        )
    else:
        print(f"{B:>3} {T:>6} {H:>4} {D:>4} {dtype_str:>8} " f"{ms_gems:>10.3f}")


# ---------------------------
# unified benchmark (fwd then fwd+bwd)
# ---------------------------
@pytest.mark.skipif(
    flaggems_vllm.device != "cuda",
    reason="benchmark requires CUDA device",
)
@pytest.mark.chunk_gla
def test_perf_chunk_gla():

    # ============================================================
    # Part 1: forward only
    # ============================================================
    _print_header(
        "chunk_gla benchmark — FWD ONLY",
        "provider: fla_chunk vs flaggems_chunk",
    )

    _precompile()
    torch.cuda.synchronize()

    for dtype in _DTYPES:
        print("\ndtype:", dtype)
        for B, T, H, D in _SHAPES:
            q, k, v, g, kwargs = _build_inputs(B, T, H, D, dtype)

            ms_fla = None
            if _HAS_FLA_CHUNK:
                ms_fla = _bench_ms(lambda: _fla_chunk_wrapper(q, k, v, g, **kwargs))
            ms_gems = _bench_ms(lambda: gems_chunk_gla(q, k, v, g, **kwargs))
            _print_row(B, T, H, D, dtype, ms_fla, ms_gems)

    # ============================================================
    # Part 2: forward + backward
    # ============================================================
    _print_header(
        "chunk_gla benchmark — FWD + BWD",
        "provider: fla_chunk vs flaggems_chunk  (forward + backward)",
    )

    # warmup: run fwd+bwd on a small shape so backward kernels compile
    print("[precompile] warming up fwd+bwd kernels ...")
    wq, wk, wv, wg_logit, wkwargs = _build_inputs(
        1, 256, 4, 64, torch.float16, requires_grad=True
    )
    for _ in range(3):
        if _HAS_FLA_CHUNK:
            wg = F.logsigmoid(wg_logit)
            out = _fla_chunk_wrapper(wq, wk, wv, wg, **wkwargs)
            if isinstance(out, tuple):
                out = out[0]
            out.sum().backward()
            for p in [wq, wk, wv, wg_logit]:
                if p.grad is not None:
                    p.grad.zero_()
        wg = F.logsigmoid(wg_logit)
        out = gems_chunk_gla(wq, wk, wv, wg, **wkwargs)
        if isinstance(out, tuple):
            out = out[0]
        out.sum().backward()
        for p in [wq, wk, wv, wg_logit]:
            if p.grad is not None:
                p.grad.zero_()
    torch.cuda.synchronize()

    for dtype in _DTYPES:
        print("\ndtype:", dtype)
        for B, T, H, D in _SHAPES:
            q, k, v, g_logit, kwargs = _build_inputs(
                B, T, H, D, dtype, requires_grad=True
            )

            ms_fla = None
            if _HAS_FLA_CHUNK:
                ms_fla = _bench_fwd_bwd_ms(_fla_chunk_wrapper, q, k, v, g_logit, kwargs)
            ms_gems = _bench_fwd_bwd_ms(gems_chunk_gla, q, k, v, g_logit, kwargs)
            _print_row(B, T, H, D, dtype, ms_fla, ms_gems)

    print(f"\n{'=' * 70}")
    print("  All done. ")
    print(f"{'=' * 70}\n")
