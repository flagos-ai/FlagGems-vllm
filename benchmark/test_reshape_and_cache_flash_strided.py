# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm

from . import base
from .conftest import Config


def write_reference(key, value, key_cache, value_cache, slots):
    blocks = slots // key_cache.shape[1]
    offsets = slots % key_cache.shape[1]
    key_cache[blocks, offsets] = key
    value_cache[blocks, offsets] = value


class StridedFlashCacheBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [(1, 2, 256), (8, 2, 256), (128, 2, 256), (1024, 2, 256)]
    DEFAULT_SHAPE_DESC = "tokens, heads, head_dim"

    def init_user_config(self):
        self.mode = Config.mode
        self.set_dtypes(Config.user_desired_dtypes)
        self.set_metrics(Config.user_desired_metrics)
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, dtype):
        block_size = 16
        for tokens, heads, dim in self.shapes:
            key = torch.randn(tokens, heads, dim, device=self.device, dtype=dtype)
            value = torch.randn_like(key)
            combined = torch.zeros(
                (tokens + block_size - 1) // block_size,
                heads,
                block_size,
                2 * dim,
                device=self.device,
                dtype=dtype,
            ).transpose(1, 2)
            slots = torch.arange(tokens, device=self.device, dtype=torch.int64)
            yield key, value, combined[..., :dim], combined[..., dim:], slots


@pytest.mark.reshape_and_cache_flash_strided
@pytest.mark.skipif(
    flaggems_vllm.device != "ptpu",
    reason="The strided Flash cache writer targets Sunrise PTPU",
)
def test_reshape_and_cache_flash_strided():
    benchmark = StridedFlashCacheBenchmark(
        op_name="reshape_and_cache_flash_strided",
        torch_op=write_reference,
        gems_op=flaggems_vllm.reshape_and_cache_flash_strided,
        dtypes=[torch.bfloat16],
        is_inplace=True,
    )
    benchmark.run()
