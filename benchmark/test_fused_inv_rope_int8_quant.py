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

import flaggems_vllm

from . import base

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
QUANT_GROUP_SIZE = 128
INT8_ABS_MAX = 127.0


def _make_cos_sin_cache(max_pos, rope_dim, device):
    half = rope_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    freqs = torch.outer(
        torch.arange(max_pos, device=device, dtype=torch.float32), inv_freq
    )
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _input_fn(shape, dtype, device):
    num_tokens, num_heads, n_groups = shape
    heads_per_group = num_heads // n_groups
    max_pos = max(4096, num_tokens * 2)
    activation = torch.randn(
        num_tokens, num_heads, HEAD_DIM, dtype=dtype, device=device
    )
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.long, device=device
    )
    cos_sin_cache = _make_cos_sin_cache(max_pos, ROPE_DIM, torch.device(device))
    yield (
        activation,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        NOPE_DIM,
        ROPE_DIM,
        QUANT_GROUP_SIZE,
    )


def _unfused_inv_rope_int8_quant(
    activation,
    positions,
    cos_sin_cache,
    n_groups,
    heads_per_group,
    nope_dim,
    rope_dim,
    quant_group_size,
):
    # vLLM has no fused int8 inverse-RoPE, so the baseline is this unfused path.
    half = rope_dim // 2
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos = cos_sin[:, :half].repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = -cos_sin[:, half:].repeat_interleave(2, dim=-1).unsqueeze(1)
    rotated = activation[..., nope_dim:].float()
    even = rotated[..., ::2]
    odd = rotated[..., 1::2]
    partner = torch.stack((-odd, even), dim=-1).flatten(-2)
    rotated = rotated * cos + partner * sin
    merged = torch.cat((activation[..., :nope_dim].float(), rotated), dim=-1)
    num_tokens, _, head_dim = merged.shape
    width = heads_per_group * head_dim
    blocks = merged.reshape(
        num_tokens, n_groups, width // quant_group_size, quant_group_size
    )
    absmax = blocks.abs().amax(dim=-1)
    scale = absmax / INT8_ABS_MAX
    scaled = torch.where(absmax.unsqueeze(-1) == 0, 0.0, blocks / scale.unsqueeze(-1))
    codes = torch.floor(scaled + 0.5).clamp(-128, 127).to(torch.int8)
    return codes.reshape(num_tokens, n_groups, width), scale


def _gems_fused_inv_rope_int8_quant(
    activation,
    positions,
    cos_sin_cache,
    n_groups,
    heads_per_group,
    nope_dim,
    rope_dim,
    quant_group_size,
):
    return flaggems_vllm.fused_inv_rope_int8_quant(
        activation,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=quant_group_size,
    )


class FusedInvRopeInt8QuantBenchmark(base.GenericBenchmark):
    # Upstream fused_inv_rope shapes, without the FP8 tma_aligned_scales flag.
    DEFAULT_SHAPES = [(1, 8, 1), (16, 64, 8)]
    DEFAULT_SHAPE_DESC = "num_tokens, num_heads, n_groups"

    def set_more_shapes(self):
        return [
            (128, 128, 8),
            (512, 64, 8),
            (512, 128, 8),
            (1024, 64, 8),
            (1024, 128, 8),
            (4096, 64, 8),
            (4096, 128, 8),
            (8192, 64, 8),
            (8192, 128, 8),
        ]

    def init_user_config(self):
        super().init_user_config()
        self.shapes = [shape for shape in self.shapes if len(shape) == 3]


@pytest.mark.fused_inv_rope_int8_quant
@pytest.mark.skipif(
    flaggems_vllm.device != "npu", reason="ascend int8 inverse-RoPE quant"
)
def test_fused_inv_rope_int8_quant():
    bench = FusedInvRopeInt8QuantBenchmark(
        op_name="fused_inv_rope_int8_quant",
        input_fn=_input_fn,
        torch_op=_unfused_inv_rope_int8_quant,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_inv_rope_int8_quant)
    bench.run()
