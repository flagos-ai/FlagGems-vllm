# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm

from . import base


def _reference(logits, bitmask, mapping, offsets, mask_stride):
    req = mapping // mask_stride
    pos = mapping % mask_stride
    start = offsets[req]
    rows = (start + pos).long()
    active = pos < offsets[req + 1] - start
    token = torch.arange(logits.shape[-1], device=logits.device)
    blocked = (bitmask[:, token // 32] >> (token % 32)) & 1 == 0
    indices = (blocked & active[:, None]).nonzero(as_tuple=True)
    logits[rows[indices[0]], indices[1]] = -torch.inf


class V2GrammarBitmaskBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 167), (4, 151936), (32, 151939)]

    def get_input_iter(self, dtype):
        for rows, vocab in self.shapes:
            yield (
                torch.zeros(rows, vocab, dtype=dtype, device=self.device),
                torch.randint(
                    -(1 << 31),
                    1 << 31,
                    (rows, (vocab + 31) // 32),
                    dtype=torch.int32,
                    device=self.device,
                ),
                torch.arange(rows, dtype=torch.int32, device=self.device),
                torch.arange(rows + 1, dtype=torch.int32, device=self.device),
                1,
            )


@pytest.mark.v2_apply_grammar_bitmask
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "kunlunxin", reason="Kunlunxin required"
)
def test_v2_apply_grammar_bitmask():
    benchmark = V2GrammarBitmaskBenchmark(
        op_name="v2_apply_grammar_bitmask", torch_op=_reference, dtypes=[torch.float32]
    )
    benchmark.set_gems(flaggems_vllm.v2_apply_grammar_bitmask)
    benchmark.run()
