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
from flaggems_vllm.ops.persistent_topk import persistent_topk

from . import base

device = flaggems_vllm.device

HAS_VLLM = False
try:
    import vllm._custom_ops  # noqa: F401

    HAS_VLLM = True
except (ImportError, AttributeError):
    pass

STRIDE = 262144
K = 512


def _baseline_persistent_topk(
    logits, lengths, indices, workspace, max_seq_len, seq_lens
):
    torch.ops._C.persistent_topk(logits, lengths, indices, workspace, K, max_seq_len)
    return indices


def _gems_decode(logits, lengths, indices, workspace, max_seq_len, seq_lens):
    persistent_topk(logits, lengths, indices, workspace, K, max_seq_len=max_seq_len)
    return indices


class PersistentTopKBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_rows, seq_len, max_seq_len"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            # Medium path shapes (commented out: not real-world scenarios)
            # (1, 10000, 10000),
            # (1, 15000, 15000),
            # (1, 20000, 20000),
            # (1, 32768, 32768),
            # (4, 10000, 10000),
            # (8, 10000, 10000),
            # (12, 20000, 20000),
            # (20, 32768, 32768),
            # Decode path shapes
            (1, 4102, 4102),
            (4, 4102, 4102),
            (10, 1055, 1055),
            (12, 4105, 4105),
            (20, 4105, 4105),
            (28, 4109, 4109),
            (1, 8192, 8192),
            # Large path lower bound
            (1, 32773, 32773),
            # Large path: num_rows=1..32, seq_len=262144
            *[(nr, 262144, 1048576) for nr in range(1, 33)],
            (496, 262144, 1048576),
            (512, 262144, 1048576),
        ]
        self.hetero_shapes = [
            (496, 1048576),
            (496, 1),
        ]

    def get_input_iter(self, dtype):
        for num_rows, seq_len, max_seq_len in self.shapes:
            torch.manual_seed(torch.randint(0, 2**31, (1,)).item())
            logits = torch.full(
                (num_rows, STRIDE),
                float("-inf"),
                dtype=torch.float32,
                device=self.device,
            )
            logits[:, :seq_len] = torch.randn(num_rows, seq_len, device=self.device)

            lengths = torch.full(
                (num_rows,), seq_len, dtype=torch.int32, device=self.device
            )
            indices = torch.empty((num_rows, K), dtype=torch.int32, device=self.device)
            workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=self.device)
            seq_lens = torch.full(
                (num_rows,), seq_len, dtype=torch.int32, device=self.device
            )

            yield logits, lengths, indices, workspace, max_seq_len, seq_lens

        # Heterogeneous batches (FlagOSTune production shapes)
        for num_rows, max_len in self.hetero_shapes:
            torch.manual_seed(torch.randint(0, 2**31, (1,)).item())
            lengths_values = torch.linspace(
                1, min(max_len, STRIDE), num_rows, dtype=torch.int32, device=self.device
            )
            logits = torch.full(
                (num_rows, STRIDE),
                float("-inf"),
                dtype=torch.float32,
                device=self.device,
            )
            for i, sl in enumerate(lengths_values):
                logits[i, :sl] = torch.randn(sl, device=self.device)

            lengths = lengths_values
            indices = torch.empty((num_rows, K), dtype=torch.int32, device=self.device)
            workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=self.device)
            seq_lens = lengths_values

            yield logits, lengths, indices, workspace, max_len, seq_lens


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")
@pytest.mark.persistent_topk
def test_persistent_topk():
    bench = PersistentTopKBenchmark(
        op_name="persistent_topk",
        torch_op=_baseline_persistent_topk,
        gems_op=_gems_decode,
        dtypes=[torch.float32],
    )
    bench.run()
