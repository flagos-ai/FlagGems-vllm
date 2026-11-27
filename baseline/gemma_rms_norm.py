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
import importlib.util
import sys
from pathlib import Path

import torch

OP_NAME = "gemma_rms_norm"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = None

EPS = 1e-5

# The vLLM Triton reference (pure Triton, no ROCm-specific dependency, so it
# also runs on NVIDIA). Loaded by file path: importing it as a package member
# would execute vllm.models.minimax_m3.__init__, which pulls in the NVIDIA
# model classes and their heavy transitive dependencies.
NATIVE_MODULE = "vllm.models.minimax_m3.amd.ops.gemma_rmsnorm"
NATIVE_FILE = Path("models") / "minimax_m3" / "amd" / "ops" / "gemma_rmsnorm.py"

# Shapes aligned to powers of two and the hidden dimensions of the
# Gemma-series models; identical to the benchmark shape grid (M, N).
GEMMA_RMS_NORM_NS = [128, 256, 1024, 16384] + [
    1152,
    2048,
    2560,
    3072,
    3584,
    3840,
    4608,
    5376,
]
GEMMA_RMS_NORM_MS = [1, 256, 1024]
GEMMA_RMS_NORM_SHAPES = [(m, n) for m in GEMMA_RMS_NORM_MS for n in GEMMA_RMS_NORM_NS]


def native():
    """Resolve the vLLM Triton gemma_rmsnorm baseline; no torch fallback.

    The module lives under an AMD path but is a pure-Triton implementation
    with no ROCm-specific dependency, so it imports and runs on NVIDIA. It is
    loaded by file path because importing it as a package member would execute
    vllm.models.minimax_m3.__init__, which pulls in the NVIDIA model classes
    and their heavy transitive dependencies.
    """
    global NATIVE_UNAVAILABLE_REASON
    NATIVE_UNAVAILABLE_REASON = None
    try:
        vllm = importlib.import_module("vllm")
        path = Path(vllm.__path__[0]) / NATIVE_FILE
        name = "_fgv_vllm_native_gemma_rmsnorm"
        module = sys.modules.get(name)
        if module is None:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(name, None)
                raise
        op = getattr(module, "gemma_rmsnorm", None)
        if not callable(op):
            raise TypeError(f"{NATIVE_MODULE} has no gemma_rmsnorm")
        return op
    except Exception as exc:
        NATIVE_UNAVAILABLE_REASON = f"{NATIVE_MODULE}: {type(exc).__name__}: {exc}"
        return None


def grid():
    return [{"shape": [m, n]} for m, n in GEMMA_RMS_NORM_SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["shape"]
    x = torch.randn((m, n), dtype=dtype, device=device)
    w = torch.randn((n,), dtype=dtype, device=device)
    return (x, w, EPS), {}


def key_shape(binding):
    return list(binding["shape"])


def config(binding, dtype):
    m, n = binding["shape"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, n], "dtype": dt},
            "w": {"shape": [n], "dtype": dt},
            "eps": {"scalar": EPS},
        },
        "outputs": {
            "y": {
                "shape": [m, n],
                "dtype": dt,
                "alias": "fresh; no input alias",
            },
        },
        "dims": {"m": m, "n": n},
        "compute_dtype": "torch.float32",
        "affine": "y = x * rsqrt(mean(x^2) + eps) * (1 + w)",
    }
