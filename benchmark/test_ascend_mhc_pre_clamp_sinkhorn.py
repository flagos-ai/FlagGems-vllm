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
    python benchmark/test_ascend_mhc_pre_clamp_sinkhorn.py
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


_pre_mod = _load("ascend_mhc_pre_clamp_sinkhorn", "mhc_pre_clamp_sinkhorn.py")

mhc_pre_clamp_sinkhorn = _pre_mod.mhc_pre_clamp_sinkhorn
mhc_pre_clamp_sinkhorn_ref = _pre_mod.mhc_pre_clamp_sinkhorn_ref

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


def bench_mhc_pre(shapes):
    return _run(
        "mhc_pre_clamp_sinkhorn",
        mhc_pre_clamp_sinkhorn_ref,
        mhc_pre_clamp_sinkhorn,
        gen_pre_inputs,
        shapes,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=["all", "mhc_pre"], default="all")
    args = parser.parse_args()

    fwd_shapes = [
        (512, 1280),
        (1024, 2560),
        (4096, 1280),
        (4096, 3584),
        (8192, 2560),
        (16384, 3584),
    ]

    results = {}
    if args.op in ("all", "mhc_pre"):
        results["mhc_pre"] = bench_mhc_pre(fwd_shapes)

    print("\n=== summary (geomean speedup) ===")
    for k, v in results.items():
        print(f"{k:35s} {v:6.2f}x")


if __name__ == "__main__":
    main()
