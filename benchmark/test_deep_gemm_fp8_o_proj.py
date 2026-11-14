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

"""Benchmark the DeepSeek-V4 FP8 o_proj against vLLM's DeepGEMM path.

Both sides receive identical tensors: the raw BF16 attention output, the RoPE
table, the block-wise FP8 ``wo_a`` weight and the same ``wo_b`` callable
(identity here, so the projection's GEMM is what gets measured). vLLM runs
``deep_gemm_fp8_o_proj`` (inverse RoPE + quant + ``fp8_einsum``); FlagGems-vllm
runs ``flaggems_vllm.deep_gemm_fp8_o_proj``. On Hopper the latter fuses the
quantization into the GEMM K loop, so this is fused vs unfused for the same
math.

Shape = ``(tokens, n_groups, group_dim, o_lora_rank)`` with the model's
``head_dim = 512``, ``nope_dim = 448``, ``rope_dim = 64``.
"""

import glob
import json
import os
import shutil

import pytest
import torch

import flaggems_vllm

from . import base

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
MAX_POS = 65536
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def is_hopper_available():
    if flaggems_vllm.vendor_name != "nvidia" or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 90 and major * 10 + minor < 100


HOPPER_AVAILABLE = is_hopper_available()


def _ensure_deep_gemm_env():
    """Make ``nvcc`` discoverable for DeepGEMM's JIT, wherever it lives.

    The pip CUDA toolkit (``nvidia/cuXX``) ships ``bin/nvcc`` without adding it
    to ``PATH``, and its headers can lag that nvcc by a minor release, which
    trips CCCL's hard compatibility error. Both are fine for compiling these
    kernels, so point DeepGEMM at the toolkit and relax that check. DeepGEMM
    caches the nvcc path when its extension loads, so this runs before the
    ``vllm.utils.deep_gemm`` import below.
    """
    if shutil.which("nvcc"):
        return True
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and os.path.exists(os.path.join(cuda_home, "bin", "nvcc")):
        return True

    import sysconfig

    purelib = sysconfig.get_paths()["purelib"]
    for nvcc in sorted(
        glob.glob(os.path.join(purelib, "nvidia", "cu*", "bin", "nvcc"))
    ):
        home = os.path.dirname(os.path.dirname(nvcc))
        os.environ.setdefault("CUDA_HOME", home)
        os.environ["PATH"] = (
            os.path.dirname(nvcc) + os.pathsep + os.environ.get("PATH", "")
        )
        os.environ.setdefault(
            "NVCC_PREPEND_FLAGS", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
        )
        return True
    return False


NVCC_READY = _ensure_deep_gemm_env()


try:
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import compute_fp8_einsum_recipe
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
        deep_gemm_fp8_o_proj as vllm_deep_gemm_fp8_o_proj,
    )
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    HAS_VLLM = is_deep_gemm_supported()
except ImportError:
    HAS_VLLM = False


DEEP_GEMM_READY = HAS_VLLM and NVCC_READY


def _is_metax():
    """MetaX has no vLLM DeepGEMM path, so the benchmark only reports latency."""
    return flaggems_vllm.vendor_name == "metax"


def _identity(flat):
    return flat


class _QuantWoA(torch.nn.Module):
    """FP8 block-wise ``wo_a`` holder, mirroring vLLM's module layout."""

    def __init__(self, weight, scale):
        super().__init__()
        self.weight = weight
        self.weight_scale_inv = scale


_WOA_CACHE = {}


def _wo_a(weight, scale):
    """Reuse one holder per tensor pair: the harness only forwards tensors."""
    key = (weight.data_ptr(), scale.data_ptr())
    holder = _WOA_CACHE.get(key)
    if holder is None:
        holder = _QuantWoA(weight, scale)
        _WOA_CACHE[key] = holder
    return holder


def _make_cos_sin_cache(max_pos, device):
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, ROPE_DIM, 2, device=device).float() / ROPE_DIM)
    )
    freqs = torch.outer(torch.arange(max_pos, device=device).float(), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def _per_block_cast_to_fp8(x, block_n=128, block_k=128):
    """128x128 block-wise FP8 quant of a 2D tensor; returns (data, scales)."""
    m, n = x.shape
    x_view = x.view(m // block_n, block_n, n // block_k, block_k)
    amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    scale = amax / FP8_MAX
    return (x_view / scale).to(torch.float8_e4m3fn).view(m, n), scale.view(
        m // block_n, n // block_k
    )


def _make_inputs(shape, device, seed=0):
    tokens, n_groups, group_dim, o_lora_rank = shape
    heads_per_group = group_dim // HEAD_DIM
    torch.manual_seed(seed)

    o = torch.randn(
        (tokens, n_groups * heads_per_group, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    positions = torch.randint(0, MAX_POS, (tokens,), device=device, dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(MAX_POS, device)

    weight = torch.randn(
        (n_groups, o_lora_rank, group_dim), device=device, dtype=torch.bfloat16
    )
    weight_fp8 = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty(
        (n_groups, o_lora_rank // 128, group_dim // 128),
        device=device,
        dtype=torch.float32,
    )
    # One group at a time keeps the fp32 dequant scratch small.
    for group in range(n_groups):
        weight_fp8[group], weight_scale[group] = _per_block_cast_to_fp8(weight[group])

    recipe, tma_aligned = (
        compute_fp8_einsum_recipe() if HAS_VLLM else ((1, 128, 128), False)
    )
    return (
        o,
        positions,
        cos_sin_cache,
        weight_fp8,
        weight_scale,
        n_groups,
        heads_per_group,
        recipe,
        tma_aligned,
    )


def _vllm_o_proj_wrapper(
    o,
    positions,
    cos_sin_cache,
    weight,
    weight_scale,
    n_groups,
    heads_per_group,
    recipe,
    tma_aligned,
):
    wo_a = _wo_a(weight, weight_scale)
    return vllm_deep_gemm_fp8_o_proj(
        o,
        positions,
        cos_sin_cache,
        wo_a,
        _identity,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        o_lora_rank=weight.shape[1],
        einsum_recipe=recipe,
        tma_aligned_scales=tma_aligned,
    )


def _gems_o_proj_wrapper(
    o,
    positions,
    cos_sin_cache,
    weight,
    weight_scale,
    n_groups,
    heads_per_group,
    recipe,
    tma_aligned,
):
    wo_a = _wo_a(weight, weight_scale)
    return flaggems_vllm.deep_gemm_fp8_o_proj(
        o,
        positions,
        cos_sin_cache,
        wo_a,
        _identity,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        o_lora_rank=weight.shape[1],
        einsum_recipe=recipe,
        tma_aligned_scales=False,
    )


class DeepGemmFP8OProjBenchmark(base.Benchmark):
    """Compare fused FlagGems o_proj against vLLM's DeepGEMM implementation."""

    DEFAULT_METRICS = base.consts.DEFAULT_METRICS[:] + ["tflops"]
    DEFAULT_SHAPE_DESC = "tokens, n_groups, group_dim, o_lora_rank"

    def set_more_shapes(self):
        # Decode / DSpark verify / C32 / prefill points for both DGX SKUs.
        tokens = (1, 4, 16, 64, 128, 512, 2048, 8192)
        return [(t, g, 4096, 1024) for g in (8, 16) for t in tokens]

    def get_input_iter(self, cur_dtype):
        del cur_dtype
        device = flaggems_vllm.device
        for shape in self.shapes:
            args = _make_inputs(shape, device)
            # Compile and tune both implementations outside the timed region.
            if DEEP_GEMM_READY and HOPPER_AVAILABLE:
                _vllm_o_proj_wrapper(*args)
            _gems_o_proj_wrapper(*args)
            torch.cuda.synchronize()
            yield args

    def get_tflops(self, op, *args, **kwargs):
        del op
        o, _, _, weight = args[:4]
        tokens = o.shape[0]
        n_groups, o_lora_rank, group_dim = weight.shape
        return 2.0 * tokens * n_groups * o_lora_rank * group_dim


@pytest.mark.deep_gemm_fp8_o_proj
@pytest.mark.skipif(
    not (HOPPER_AVAILABLE or _is_metax()),
    reason="requires an NVIDIA Hopper or a MetaX GPU",
)
def test_perf_deep_gemm_fp8_o_proj_gems_vs_vllm():
    # vLLM's op is only importable from a build that ships DeepSeek-V4, and its
    # DeepGEMM path still needs a working nvcc. Without both, report the
    # FlagGems latency only (torch_op=None) and, when --base-data is given,
    # compare against that baseline instead of an in-process vLLM run.
    torch_op = _vllm_o_proj_wrapper if (DEEP_GEMM_READY and HOPPER_AVAILABLE) else None
    bench = DeepGemmFP8OProjBenchmark(
        op_name="deep_gemm_fp8_o_proj",
        torch_op=torch_op,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_o_proj_wrapper)
    bench.run()


# H100 native DeepGEMM records collected with ``baseline/deep_gemm_fp8_o_proj.py``
# on an H100 80GB (vllm 0.24.0, do_bench, wo_b = identity).
H100_BASELINE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "base_data",
    "deep_gemm_fp8_o_proj_h100.json",
)

# Shapes produced by this benchmark: set_more_shapes() tokens plus the core
# 4096-token point, for both group counts.
_BASELINE_TOKENS = (1, 4, 16, 64, 128, 512, 2048, 4096, 8192)
_BASELINE_GROUPS = (8, 16)


@pytest.mark.deep_gemm_fp8_o_proj
@pytest.mark.skipif(_is_metax() is False, reason="MetaX-only: H100-equivalent speedup")
def test_perf_deep_gemm_fp8_o_proj_metax_vs_h100():
    """Cross-chip speedup against the H100 DeepGEMM baseline.

    MetaX has no native implementation to time in-process, so the reference is
    the H100 record collected with the same inputs and timing method. The
    harness prints ``vs Base = (base_ms / latency) / factor``, where ``factor``
    is the C550/H100 peak ratio of the record's bottleneck resource from
    ``_scaling_factors`` -- the 折算 the acceptance doc prescribes.
    """
    with open(H100_BASELINE) as handle:
        data = json.load(handle)

    # The baseline must cover every shape this benchmark emits, otherwise the
    # harness reports a miss instead of a speedup for that point.
    records = data["deep_gemm_fp8_o_proj"]["shapes"]
    heads_per_group = 4096 // HEAD_DIM
    missing = []
    for tokens in _BASELINE_TOKENS:
        for groups in _BASELINE_GROUPS:
            key = str([tokens, groups * heads_per_group, HEAD_DIM])
            if key not in records:
                missing.append(key)
    assert not missing, f"H100 baseline is missing shapes: {missing}"

    if base.Config.base_data is None:
        base.Config.base_data = data

    bench = DeepGemmFP8OProjBenchmark(
        op_name="deep_gemm_fp8_o_proj",
        torch_op=None,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_o_proj_wrapper)
    bench.run()
