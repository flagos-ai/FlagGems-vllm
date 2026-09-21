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
import yaml

import flaggems_vllm
from flaggems_vllm.utils.device_info import get_device_capability

from . import base

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
QUANT_GROUP_SIZE = 128

HAS_NATIVE_FP8 = hasattr(torch, "float8_e4m3fn") and (
    flaggems_vllm.SUPPORTED_FP8_DTYPE == torch.float8_e4m3fn
    or (flaggems_vllm.vendor_name == "mthreads" and get_device_capability() >= (3, 1))
)

try:
    from vllm.models.deepseek_v4.common.ops import (
        fused_inv_rope_fp8_quant as vllm_fused_inv_rope_fp8_quant,
    )
except ImportError:
    try:
        from vllm.v1.attention.ops.deepseek_v4_ops import (
            fused_inv_rope_fp8_quant as vllm_fused_inv_rope_fp8_quant,
        )
    except ImportError:
        vllm_fused_inv_rope_fp8_quant = None


ORIGINAL_SHAPES = [
    (num_tokens, num_heads, n_groups)
    for num_tokens in (1, 7, 32, 128)
    for num_heads, n_groups in ((32, 4), (64, 8), (128, 8))
] + [(1, 8, 1), (16, 64, 8), (256, 64, 8)]
# Medium and long prefill batches extend the original decode and short prefill set.
SUPPLEMENTAL_SHAPES = [(num_tokens, 64, 8) for num_tokens in (512, 1024, 2048, 4096)]


def _make_cos_sin_cache(max_pos, rope_dim, device):
    half = rope_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _input_fn(shape, dtype, device):
    num_tokens, num_heads, n_groups, tma_aligned_scales = shape
    heads_per_group = num_heads // n_groups
    max_pos = max(4096, num_tokens * 2)

    o = torch.randn(num_tokens, num_heads, HEAD_DIM, dtype=dtype, device=device)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.long, device=device
    )
    cos_sin_cache = _make_cos_sin_cache(max_pos, ROPE_DIM, torch.device(device))

    yield (
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        NOPE_DIM,
        ROPE_DIM,
        QUANT_GROUP_SIZE,
        tma_aligned_scales,
    )


def _gems_fused_inv_rope_fp8_quant(
    o,
    positions,
    cos_sin_cache,
    n_groups,
    heads_per_group,
    nope_dim,
    rope_dim,
    quant_group_size,
    tma_aligned_scales,
):
    return flaggems_vllm.fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=quant_group_size,
        dtype=torch.float8_e4m3fn,
        tma_aligned_scales=tma_aligned_scales,
    )


def _vllm_fused_inv_rope_fp8_quant(
    o,
    positions,
    cos_sin_cache,
    n_groups,
    heads_per_group,
    nope_dim,
    rope_dim,
    quant_group_size,
    tma_aligned_scales,
):
    return vllm_fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=quant_group_size,
        tma_aligned_scales=tma_aligned_scales,
    )


def _validate_baseline(device):
    """Verify the selected baseline's output contract before measuring it."""
    source = vllm_fused_inv_rope_fp8_quant.__module__
    if source.startswith("flaggems_vllm"):
        raise RuntimeError("The vLLM baseline was redirected to FlagGems-vllm")
    print(f"vLLM baseline: {source}.fused_inv_rope_fp8_quant (packed UE8M0 scales)")

    o = torch.ones(1, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)
    positions = torch.zeros(1, dtype=torch.int64, device=device)
    cache = _make_cos_sin_cache(1, ROPE_DIM, torch.device(device))
    out, scale = _vllm_fused_inv_rope_fp8_quant(
        o,
        positions,
        cache,
        1,
        1,
        NOPE_DIM,
        ROPE_DIM,
        QUANT_GROUP_SIZE,
        True,
    )

    # A unit input gives scale 2**-8, whose UE8M0 exponent byte is 119.
    out_float = out.float()
    valid = (
        out.shape == (1, 1, HEAD_DIM)
        and out.dtype == torch.float8_e4m3fn
        and scale.shape == (1, 1, 1)
        and scale.dtype == torch.int32
        and torch.equal(scale, torch.full_like(scale, 0x77777777))
        and torch.equal(out_float, torch.full_like(out_float, 256.0))
    )
    if not valid:
        raise RuntimeError(
            f"Baseline {source} has incompatible packed quantization semantics. "
            "FlagGems requires power-of-two UE8M0 scales. "
            "This baseline is not an equivalent comparison."
        )


class FusedInvRopeFP8QuantBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [(*shape, True) for shape in ORIGINAL_SHAPES + SUPPLEMENTAL_SHAPES]
    DEFAULT_SHAPE_DESC = "num_tokens, num_heads, n_groups, tma_aligned_scales"

    def set_more_shapes(self):
        # GenericBenchmark's extra shapes do not describe this operator's inputs.
        return []

    def set_shapes(self, shape_file_path):
        with open(shape_file_path) as shape_file:
            config = yaml.safe_load(shape_file)
        if not any(key in config for key in (self.op_name, type(self).__name__)):
            # The shared Benchmark entry contains generic tensor shapes.
            self.shapes = list(self.DEFAULT_SHAPES)
            self.shape_desc = self.DEFAULT_SHAPE_DESC
            return
        super().set_shapes(shape_file_path)

    def init_user_config(self):
        super().init_user_config()
        if any(len(shape) != 4 for shape in self.shapes):
            raise ValueError(f"Invalid fused_inv_rope_fp8_quant shapes: {self.shapes}")
        if any(shape[3] is not True for shape in self.shapes):
            raise ValueError(
                "fused_inv_rope_fp8_quant benchmarks require "
                "tma_aligned_scales=True (packed scales)"
            )

    def get_input_iter(self, dtype):
        _validate_baseline(self.device)
        yield from super().get_input_iter(dtype)


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.skipif(
    vllm_fused_inv_rope_fp8_quant is None,
    reason="vLLM fused_inv_rope_fp8_quant not installed",
)
def test_fused_inv_rope_fp8_quant():
    bench = FusedInvRopeFP8QuantBenchmark(
        op_name="fused_inv_rope_fp8_quant",
        input_fn=_input_fn,
        torch_op=_vllm_fused_inv_rope_fp8_quant,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_inv_rope_fp8_quant)
    bench.run()
