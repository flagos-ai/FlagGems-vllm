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

"""Benchmark the multi-rank Hopper MegaMoE kernel on representative shapes."""

import sys

import pytest
import torch

from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamoe import (
    MegaMoEConfig,
    launch_megamoe,
    resolve_mpirun,
)

from . import base
from .conftest import Config
from .consts import DEFAULT_ITER_TIME, DEFAULT_WARMUP_TIME, BenchMode

OP_NAME = "megamoe"
NUM_RANKS = 8
STAGES = 4
DEFAULT_SHAPES = [
    (512, 2048, 512, 256, 8, NUM_RANKS),
    (1024, 2048, 512, 256, 8, NUM_RANKS),
    (2048, 2048, 512, 256, 8, NUM_RANKS),
]
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 30
DEFAULT_TIMEOUT_S = 1800


def _iteration_count(option, configured, default, fallback):
    explicitly_set = any(
        arg == option or arg.startswith(f"{option}=") for arg in sys.argv
    )
    return configured if explicitly_set or configured != default else fallback


class MegaMoEBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency"]
    DEFAULT_DTYPES = [torch.float8_e4m3fn]
    DEFAULT_SHAPE_DESC = "tokens, hidden, intermediate, num_experts, topk, num_ranks"

    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            torch_op=launch_megamoe,
            gems_op=launch_megamoe,
            dtypes=self.DEFAULT_DTYPES,
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, dtype):
        yield from self.shapes

    def get_latency(
        self,
        op,
        tokens,
        hidden,
        intermediate,
        num_experts,
        topk,
        num_ranks,
    ):
        if Config.mode is not BenchMode.KERNEL:
            raise ValueError("MegaMoE only supports --mode kernel")
        config = MegaMoEConfig(
            num_ranks=num_ranks,
            tokens=tokens,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_experts=num_experts,
            topk=topk,
            stages=STAGES,
            drop_rate=0,
            benchmark=True,
            warmup=_iteration_count(
                "--warmup", Config.warm_up, DEFAULT_WARMUP_TIME, DEFAULT_WARMUP
            ),
            iterations=_iteration_count(
                "--iter", Config.repetition, DEFAULT_ITER_TIME, DEFAULT_ITERS
            ),
            reduce="mean",
            gpu_start_barrier=True,
            timeout=DEFAULT_TIMEOUT_S,
        )
        return op(config).slowest_rank_latency_ms


@pytest.mark.megamoe
def test_megamoe():
    if Config.query:
        MegaMoEBenchmark().run()
        return
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        pytest.skip(f"requires SM90, got SM{major}0")
    if torch.cuda.device_count() < NUM_RANKS:
        pytest.skip(f"requires {NUM_RANKS} visible GPUs")
    if resolve_mpirun() is None:
        pytest.skip("requires mpirun on PATH")

    MegaMoEBenchmark().run()
