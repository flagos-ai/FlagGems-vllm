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

import importlib

import torch

from flaggems_vllm.ops.mhc.mhc_pre_broadcast import derive_fn_broadcast

OP_NAME = "mhc_pre_broadcast"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# (N, hidden_size) — mirrors the shape set used by the existing pytest
# benchmark (benchmark/test_mhc_pre_broadcast.py, MHCPreBroadcastBenchmark
# .set_shapes).
_SHAPES = [
    (1, 1280),
    (8, 1280),
    (512, 1280),
    (512, 2560),
    (512, 4096),
    (1024, 1280),
    (1024, 2560),
    (1024, 4096),
    (2048, 1280),
    (2048, 2560),
    (2048, 4096),
    (8192, 1280),
    (8192, 2560),
    (8192, 4096),
]

# Fixed in the pytest benchmark's constructor defaults
# (hc_mult=4, sinkhorn_repeat=10); mirrored here so baseline inputs match.
_HC_MULT = 4
_SINKHORN_REPEAT = 10
_RMS_EPS = 1e-6
_HC_PRE_EPS = 1e-6
_HC_SINKHORN_EPS = 1e-6
_HC_POST_MULT_VALUE = 1.0

# nvidia ships a dedicated tilelang kernel for the broadcast fast path
# (hc_mult == 4, all residual streams identical at model entry):
# vllm.model_executor.kernels.mhc.tilelang.mhc_pre_broadcast_tilelang.
# Mirrors benchmark/test_mhc_pre_broadcast.py's _NATIVE_MHC_PRE_BROADCAST:
# hygon (aiter, only mhc_pre_aiter) and thead (only MHCPreOp/MHCPostOp/
# MHCFusedPostPreOp/HCHeadOp) were checked and confirmed to have no
# broadcast variant, so this baseline is NV-only, consistent with the
# "baseline/ covers the CUDA baseline only" scope.
_NATIVE_MODULE = "vllm.model_executor.kernels.mhc.tilelang"
_NATIVE_ATTR = "mhc_pre_broadcast_tilelang"


def native():
    """Resolve the NV-native mhc_pre_broadcast_tilelang kernel; None if
    unavailable. Unlike the pytest benchmark, this baseline does not fall
    back to the PyTorch reference implementation when the native kernel is
    missing — per convention it should return None so the collector skips
    this shape/vendor rather than recording a torch-reference latency."""
    try:
        mod = importlib.import_module(_NATIVE_MODULE)
    except ImportError:
        return None
    op = getattr(mod, _NATIVE_ATTR, None)
    return op if callable(op) else None


def grid():
    return [{"shape": list(s)} for s in _SHAPES]


def _unpack(binding):
    n, hidden_size = binding["shape"]
    return n, hidden_size


def build_inputs(binding, dtype, device):
    n, hidden_size = _unpack(binding)
    hc_mult = _HC_MULT
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    torch.manual_seed(42)
    residual = torch.randn((n, hidden_size), dtype=torch.float, device=device).to(
        dtype
    )
    fn = (
        torch.randn((hc_mult3, hc_mult * hidden_size), dtype=torch.float, device=device)
        * 1e-4
    )
    fn_broadcast = derive_fn_broadcast(fn, hc_mult)
    hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1
    norm_weight = torch.randn((hidden_size,), dtype=torch.float, device=device).to(
        dtype
    )

    # Same positional order as mhc_pre_broadcast / the pytest benchmark's
    # get_input_iter; a vendor-native callable must accept this same order.
    args = (
        residual,
        fn_broadcast,
        hc_scale,
        hc_base,
        _RMS_EPS,
        _HC_PRE_EPS,
        _HC_SINKHORN_EPS,
        _HC_POST_MULT_VALUE,
        _SINKHORN_REPEAT,
        norm_weight,
    )
    return args, {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    n, hidden_size = _unpack(binding)
    hc_mult = _HC_MULT
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    dt = str(dtype)

    return {
        "inputs": {
            "residual": {"shape": [n, hidden_size], "dtype": dt},
            "fn_broadcast": {
                "shape": [hc_mult3, hc_mult * hidden_size],
                "dtype": "torch.float32",
                "note": "derived from fn via derive_fn_broadcast(fn, hc_mult)",
            },
            "hc_scale": {"shape": [3], "dtype": "torch.float32"},
            "hc_base": {"shape": [hc_mult3], "dtype": "torch.float32"},
            "rms_eps": {"scalar": _RMS_EPS},
            "hc_pre_eps": {"scalar": _HC_PRE_EPS},
            "hc_sinkhorn_eps": {"scalar": _HC_SINKHORN_EPS},
            "hc_post_mult_value": {"scalar": _HC_POST_MULT_VALUE},
            "sinkhorn_repeat": {"scalar": _SINKHORN_REPEAT},
            "norm_weight": {"shape": [hidden_size], "dtype": dt},
        },
        "dims": {
            "N": n,
            "hidden_size": hidden_size,
            "hc_mult": hc_mult,
            "hc_mult3": hc_mult3,
        },
    }

