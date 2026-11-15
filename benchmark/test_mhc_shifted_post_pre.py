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

"""Performance test for the shifted mHC post->pre operator.

The baseline is vLLM's production kernel for the DSV4.1 boundary (the fused
TileLang / DeepGEMM path).  Shapes and inputs mirror
``baseline/mhc_shifted_post_pre.py`` so the benchmark and the CUDA baseline
collector time the same workload; a missing native kernel leaves ``torch_op``
as ``None``, in which case the case reports latency only.  vLLM's torch
reference is the CPU/fallback implementation used for numeric checks in
``tests/``: it upcasts every stage to fp32 and fuses nothing, so timing against
it would produce a large and meaningless SpeedUp.

Run eagerly: ``--mode kernel`` (the default).  The operator dispatches into
several sub-operators with host-side autotune caches, so a cold CUDA-graph
capture invalidates the stream and a warm one only measures the frozen
steady-state.
"""

import importlib
import warnings

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.mhc.mhc_shifted_post_pre import mhc_shifted_post_pre

from . import base

VENDOR = flaggems_vllm.vendor_name

_HC = 4
_MIX = _HC * (_HC + 2)
_SEED = 0

# DSV4/DSV4.1 boundary shapes.  Keep in sync with
# baseline/mhc_shifted_post_pre.py and benchmark/core_shapes.yaml.
SHAPES = [
    (1, 1280),
    (16, 1280),
    (512, 1280),
    (512, 7168),
    (4096, 7168),
    (128, 5120),
    (2048, 5120),
]

DTYPES = [torch.bfloat16]

# Vendor-native baseline, one entry per vendor (nvidia mirrors
# baseline/mhc_shifted_post_pre.py::native()).  A vendor missing here has no
# native kernel and the benchmark reports the FlagGems latency only.
_NATIVE_MHC_SHIFTED = {
    "nvidia": (
        "vllm.models.deepseek_v41.nvidia.ops.mhc",
        "mhc_shifted_post_pre",
    ),
}


def _get_baseline_op():
    """vLLM's production kernel for this vendor, or None when it ships none."""
    entry = _NATIVE_MHC_SHIFTED.get(VENDOR)
    if entry is None:
        return None
    module_name, attr = entry
    try:
        return getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        warnings.warn(
            f"mhc_shifted_post_pre benchmark: {VENDOR} declares a native "
            f"baseline at {module_name}.{attr} but it could not be loaded "
            f"({exc!r}); the case runs without a baseline, so latency_base and "
            "speedup stay empty.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _build_inputs(tokens, hidden, dtype, device):
    """Mirror baseline/mhc_shifted_post_pre.py::build_inputs exactly."""
    gen = torch.Generator(device=device).manual_seed(_SEED)
    x = torch.randn(tokens, hidden, dtype=dtype, device=device, generator=gen)
    residual = torch.randn(
        tokens, _HC, hidden, dtype=dtype, device=device, generator=gen
    )
    post_layer_mix = (
        torch.rand(tokens, _HC, 1, dtype=torch.float32, device=device, generator=gen)
        * 0.3
    )
    comb_res_mix = (
        torch.randn(tokens, _HC, _HC, dtype=torch.float32, device=device, generator=gen)
        * 0.2
    )
    fn = (
        torch.randn(
            _MIX, _HC * hidden, dtype=torch.float32, device=device, generator=gen
        )
        * 0.02
    )
    hc_scale = torch.tensor([0.1, 0.3, 0.2], dtype=torch.float32, device=device)
    hc_base = torch.randn(_MIX, dtype=torch.float32, device=device, generator=gen) * 0.1
    pre_mix = (
        torch.rand(tokens, _HC, dtype=torch.float32, device=device, generator=gen) + 0.5
    )
    norm_weight = (
        torch.randn(hidden, dtype=dtype, device=device, generator=gen) * 0.3 + 1
    )
    args = (
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        1e-6,  # rms_eps
        1e-6,  # hc_pre_eps
        1e-6,  # hc_sinkhorn_eps
        1.0,  # hc_post_mult_value
        5,  # sinkhorn_repeat
    )
    kwargs = {
        "pre_mix": pre_mix,
        "norm_weight": norm_weight,
        "norm_eps": 1e-6,
        "capture_aux": True,
    }
    return args, kwargs


class MHCShiftedPostPreBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "N, H"

    # core_shapes.yaml holds the recommended shapes; these are the fallback and
    # must stay identical to it and to baseline/mhc_shifted_post_pre.py.
    DEFAULT_SHAPES = SHAPES
    DEFAULT_DTYPES = DTYPES

    def get_input_iter(self, dtype):
        for tokens, hidden in self.shapes:
            args, kwargs = _build_inputs(tokens, hidden, dtype, self.device)
            yield (*args, kwargs)


@pytest.mark.mhc_shifted_post_pre
def test_mhc_shifted_post_pre():
    bench = MHCShiftedPostPreBenchmark(
        op_name="mhc_shifted_post_pre",
        torch_op=_get_baseline_op(),
        gems_op=mhc_shifted_post_pre,
        is_inplace=False,
    )
    bench.run()
