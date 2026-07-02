import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import triton

import flaggems_vllm
from benchmark.conftest import Config
from flaggems_vllm.ops.FLA.chunk_gla import chunk_gla as gems_chunk_gla

# Try importing fla chunk_gla from installed package only.
_HAS_FLA_CHUNK = False
_fla_chunk_gla = None

try:
    from fla.ops.gla import chunk_gla as _fla_chunk_gla

    _HAS_FLA_CHUNK = True
except Exception:
    _HAS_FLA_CHUNK = False


def _torch_naive_recurrent_gla(
    q,
    k,
    v,
    g,
    scale=None,
    initial_state=None,
    output_final_state=False,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
):
    del cu_seqlens_cpu

    if state_v_first:
        raise ValueError("state_v_first=True is not supported in torch naive baseline")
    if cu_seqlens is not None:
        raise ValueError("cu_seqlens is not supported in torch naive baseline")

    dtype = q.dtype
    qf = q.float()
    kf = k.float()
    vf = v.float()
    gf = g.float()

    B, T, H, K = qf.shape
    V = vf.shape[-1]
    if scale is None:
        scale = K**-0.5

    if initial_state is None:
        h = torch.zeros((B, H, K, V), device=q.device, dtype=torch.float32)
    else:
        h = initial_state.float().clone()

    out = torch.zeros((B, T, H, V), device=q.device, dtype=torch.float32)
    for t in range(T):
        qt = qf[:, t] * float(scale)
        kt = kf[:, t]
        vt = vf[:, t]
        gt = gf[:, t].exp()
        h = h * gt[..., None] + kt[..., None] * vt[..., None, :]
        out[:, t] = (qt[..., None] * h).sum(-2)

    final_state = h if output_final_state else None
    return out.to(dtype), final_state


def _fla_chunk_wrapper(
    q,
    k,
    v,
    g,
    scale=None,
    initial_state=None,
    output_final_state=False,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
):
    del cu_seqlens_cpu
    if not _HAS_FLA_CHUNK:
        raise RuntimeError("fla chunk_gla is unavailable")
    return _fla_chunk_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
    )


def _load_test_reference_impl():
    repo_root = Path(__file__).resolve().parents[2]
    test_file = repo_root / "benchmark" / "test_FLA" / "test_chunk_gla_perf.py"
    spec = importlib.util.spec_from_file_location("_chunk_gla_test_ref", str(test_file))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load benchmark/test_FLA/test_chunk_gla_perf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_torch_reference_consistency():
    module = _load_test_reference_impl()
    if hasattr(module, "_HAS_FLA_NAIVE"):
        module._HAS_FLA_NAIVE = False
    if not hasattr(module, "_reference_chunk_gla"):
        print("[consistency] skip: _reference_chunk_gla not found")
        return
    dtype = torch.bfloat16
    device = flaggems_vllm.device
    B, T, H, D = 1, 16, 2, 32

    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, D, device=device, dtype=dtype))
    h0 = torch.randn(B, H, D, D, device=device, dtype=torch.float32)
    scale = D**-0.5

    out_a, ht_a = _torch_naive_recurrent_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
    )
    out_b, ht_b = module._reference_chunk_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        state_v_first=False,
        cu_seqlens=None,
    )

    torch.testing.assert_close(out_a, out_b, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ht_a, ht_b, atol=1e-6, rtol=1e-6)
    print(
        "[consistency] torch naive == tests/test_FLA/test_chunk_gla.py reference: PASS"
    )


def _bench_ms(fn):
    if Config.mode.value == "kernel":
        return triton.testing.do_bench(
            fn,
            warmup=Config.warm_up,
            rep=Config.repetition,
            return_mode="median",
        )

    for _ in range(Config.warm_up):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(Config.repetition):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / Config.repetition


def _build_inputs(B, T, H, D, dtype):
    device = flaggems_vllm.device
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
    return q, k, v, g, kwargs


@pytest.mark.skipif(
    flaggems_vllm.device != "cuda", reason="benchmark requires CUDA device"
)
@pytest.mark.chunk_gla
def test_perf_chunk_gla():
    _check_torch_reference_consistency()

    shapes = [
        # small / correctness-friendly
        (2, 512, 8, 64),
        (4, 1024, 8, 64),
        # small batch + medium / long context
        (1, 2048, 8, 64),
        (1, 4096, 16, 64),
    ]
    dtypes = [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]

    print("\n[chunk_gla 3-way benchmark]")
    print(f"mode={Config.mode.value} warmup={Config.warm_up} iter={Config.repetition}")
    if _HAS_FLA_CHUNK:
        print("provider: torch_naive vs fla_chunk vs flaggems_chunk")
        print(
            f"{'B':>3} {'T':>6} {'H':>4} {'D':>4} {'dtype':>8} "
            f"{'torch(ms)':>12} {'fla(ms)':>10} {'gems(ms)':>10} "
            f"{'gems/torch':>11} {'gems/fla':>10}"
        )
    else:
        print("provider: torch_naive vs flaggems_chunk (fla_chunk unavailable)")
        print(
            f"{'B':>3} {'T':>6} {'H':>4} {'D':>4} {'dtype':>8} "
            f"{'torch(ms)':>12} {'gems(ms)':>10} {'gems/torch':>11}"
        )

    for dtype in dtypes:
        print("精度：", dtype)
        for B, T, H, D in shapes:
            q, k, v, g, kwargs = _build_inputs(B, T, H, D, dtype)

            ms_torch = _bench_ms(
                lambda: _torch_naive_recurrent_gla(q, k, v, g, **kwargs)
            )
            ms_fla = None
            if _HAS_FLA_CHUNK:
                ms_fla = _bench_ms(lambda: _fla_chunk_wrapper(q, k, v, g, **kwargs))
            ms_gems = _bench_ms(lambda: gems_chunk_gla(q, k, v, g, **kwargs))

            speedup_vs_torch = ms_torch / ms_gems if ms_gems > 0 else float("inf")
            if _HAS_FLA_CHUNK and ms_fla is not None:
                speedup_vs_fla = ms_fla / ms_gems if ms_gems > 0 else float("inf")
                print(
                    f"{B:>3} {T:>6} {H:>4} {D:>4} {str(dtype).split('.')[-1]:>8} "
                    f"{ms_torch:>12.3f} {ms_fla:>10.3f} {ms_gems:>10.3f} "
                    f"{speedup_vs_torch:>11.2f}x {speedup_vs_fla:>10.2f}x"
                )
            else:
                print(
                    f"{B:>3} {T:>6} {H:>4} {D:>4} {str(dtype).split('.')[-1]:>8} "
                    f"{ms_torch:>12.3f} {ms_gems:>10.3f} {speedup_vs_torch:>11.2f}x"
                )
