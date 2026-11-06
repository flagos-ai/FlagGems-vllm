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

"""Performance benchmarks for the Ascend MHC operator family in
``flaggems_vllm.runtime.backend._ascend.mhc``.

Baselines are the pure-PyTorch references (``*_ref``) shipped in the same
modules. SpeedUp = latency_torch_ref / latency_triton.

Usage:
    python benchmark/test_ascend_mhc_pre_clamp_sinkhorn_backward.py
"""

import argparse
import importlib.util  # noqa: E402
import sys  # noqa: E402
import time
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import torch
import torch_npu  # noqa: F401

_MHC_DIR = (
    Path(__file__).resolve().parent.parent
    / "src/flaggems_vllm/runtime/backend/_ascend/mhc"
)

# Register stub parent packages so the backend op modules can be loaded
# directly by path (with their relative imports intact) without pulling in
# the full flaggems_vllm package, which requires vLLM.
for _pkg in [
    "flaggems_vllm",
    "flaggems_vllm.runtime",
    "flaggems_vllm.runtime.backend",
    "flaggems_vllm.runtime.backend._ascend",
]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = types.ModuleType(_pkg)

# The leaf package must be a real package (with __path__) for relative
# imports inside the op modules to resolve.
_mhc_pkg = types.ModuleType("flaggems_vllm.runtime.backend._ascend.mhc")
_mhc_pkg.__path__ = [str(_MHC_DIR)]
sys.modules["flaggems_vllm.runtime.backend._ascend.mhc"] = _mhc_pkg


def _load(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(
        f"flaggems_vllm.runtime.backend._ascend.mhc.{mod_name}",
        _MHC_DIR / file_name,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_pre_bwd_mod = _load(
    "ascend_mhc_pre_clamp_sinkhorn_backward", "mhc_pre_clamp_sinkhorn_backward.py"
)

mhc_pre_clamp_sinkhorn_backward = _pre_bwd_mod.mhc_pre_clamp_sinkhorn_backward
mhc_pre_clamp_sinkhorn_backward_ref = _pre_bwd_mod.mhc_pre_clamp_sinkhorn_backward_ref

DEVICE = "npu"
HC_MULT = 4
HC_MIX = HC_MULT * (HC_MULT + 2)  # 24
WARMUP = 10
REP = 50


def gen_pre_inputs(t, d, dtype=torch.bfloat16):
    torch.manual_seed(42)
    x = torch.randn((t, HC_MULT, d), dtype=dtype, device=DEVICE)
    phi = torch.randn((HC_MIX, HC_MULT * d), dtype=torch.float32, device=DEVICE) * 0.02
    alpha = torch.randn((3,), dtype=torch.float32, device=DEVICE) * 0.1
    base = torch.randn((HC_MIX,), dtype=torch.float32, device=DEVICE) * 0.1
    return x, phi, alpha, base


def bench(fn, args):
    for _ in range(WARMUP):
        fn(*args)
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(REP):
        fn(*args)
    torch.npu.synchronize()
    return (time.perf_counter() - start) / REP * 1e3  # ms


def _run(op_name, ref_fn, tri_fn, arg_gen, shapes):
    print(f"\n=== {op_name} (SpeedUp = latency_ref / latency_triton) ===")
    print(f"{'shape':>16} {'ref (ms)':>10} {'triton (ms)':>12} {'speedup':>8}")
    speedups = []
    for t, d in shapes:
        args = arg_gen(t, d)
        lat_ref = bench(ref_fn, args)
        lat_tri = bench(tri_fn, args)
        speedup = lat_ref / lat_tri
        speedups.append(speedup)
        print(f"{f'({t},{d})':>16} {lat_ref:>10.3f} {lat_tri:>12.3f} {speedup:>8.2f}")
    geo = 1.0
    for s in speedups:
        geo *= s
    geo **= 1.0 / len(speedups)
    print(f"{'geomean':>16} {'':>10} {'':>12} {geo:>8.2f}")
    return geo


def _torch_pre_forward(x, phi, alpha, base, norm_eps=1e-6, hc_eps=1e-6):
    """Test/benchmark-only torch forward producing the saved intermediates
    that mhc_pre_clamp_sinkhorn_backward consumes (avoids depending on the
    forward Triton op in this PR)."""
    T, N, D = x.shape
    x_flat = x.reshape(T, N * D).float()
    ms = x_flat.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(ms + norm_eps).squeeze(-1)
    x_scaled = x_flat * inv_rms.unsqueeze(-1)
    mixes = x_scaled @ phi.float().t()
    a = alpha.float()
    b = base.float()
    pre = torch.sigmoid(mixes[:, :N] * a[0] + b[:N]) + hc_eps
    logits = (mixes[:, 2 * N :] * a[2] + b[2 * N :]).reshape(T, N, N)
    return inv_rms, x_scaled, mixes, logits, pre


def bench_mhc_pre_backward(shapes):
    def arg_gen(t, d):
        x, phi, alpha, base = gen_pre_inputs(t, d)
        inv_rms, x_scaled, mixes, h_res_logits, pre = _torch_pre_forward(
            x, phi, alpha, base
        )
        torch.manual_seed(7)
        grad_y = torch.randn((t, d), dtype=torch.bfloat16, device=DEVICE)
        grad_post = torch.randn((t, HC_MULT), dtype=torch.float32, device=DEVICE)
        grad_comb = torch.randn(
            (t, HC_MULT, HC_MULT), dtype=torch.float32, device=DEVICE
        )
        saved = (inv_rms, x_scaled, mixes, h_res_logits, pre)
        grads = (grad_y, grad_post, grad_comb)
        # ref takes (x, phi, alpha, base, grads...); triton takes saved tensors too
        ref_args = (x, phi, alpha, base) + grads
        tri_args = (x, phi, alpha, base) + saved + grads
        return ref_args, tri_args

    print(
        "\n=== mhc_pre_clamp_sinkhorn_backward "
        "(SpeedUp = latency_ref / latency_triton) ==="
    )
    print(f"{'shape':>16} {'ref (ms)':>10} {'triton (ms)':>12} {'speedup':>8}")
    speedups = []
    for t, d in shapes:
        ref_args, tri_args = arg_gen(t, d)
        lat_ref = bench(mhc_pre_clamp_sinkhorn_backward_ref, ref_args)
        lat_tri = bench(mhc_pre_clamp_sinkhorn_backward, tri_args)
        speedup = lat_ref / lat_tri
        speedups.append(speedup)
        print(f"{f'({t},{d})':>16} {lat_ref:>10.3f} {lat_tri:>12.3f} {speedup:>8.2f}")
    geo = 1.0
    for s in speedups:
        geo *= s
    geo **= 1.0 / len(speedups)
    print(f"{'geomean':>16} {'':>10} {'':>12} {geo:>8.2f}")
    return geo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["all", "mhc_pre_backward"], default="all")
    args = parser.parse_args()

    bwd_shapes = [
        (512, 1280),
        (1024, 2560),
        (4096, 1280),
        (4096, 3584),
        (8192, 2560),
    ]

    results = {}
    if args.op in ("all", "mhc_pre_backward"):
        results["mhc_pre_backward"] = bench_mhc_pre_backward(bwd_shapes)

    print("\n=== summary (geomean speedup) ===")
    for k, v in results.items():
        print(f"{k:35s} {v:6.2f}x")


if __name__ == "__main__":
    main()
