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

"""Benchmark trace replay against the upstream vLLM Triton implementation."""

import pytest
import torch

from flaggems_vllm import apply_trace_tokens as flaggems_apply_trace_tokens

from . import base

try:
    from vllm.v1.worker.gpu.sample.trace_replay import (
        apply_trace_tokens as vllm_reference,
    )
except (ImportError, OSError):
    vllm_reference = None



class TraceReplayBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [
        (n, 2048) for n in (1, 2, 3, 5, 32, 33, 127, 128, 129, 256, 257, 4096)
    ]
    DEFAULT_SHAPE_DESC = "num_reqs, max_model_len"

    def get_latency(self, op, sampled, *args, **kwargs):
        # Replaying fixed trace tokens is idempotent. Isolate implementations,
        # then let the base class handle all timing modes.
        return super().get_latency(op, sampled.clone(), *args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(self.DEFAULT_SHAPES)

    def get_input_iter(self, dtype):
        for num_reqs, max_model_len in self.shapes:
            sampled = torch.full((num_reqs,), -7, dtype=dtype, device=self.device)
            mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
            traces = torch.arange(
                num_reqs * max_model_len, dtype=torch.int32, device=self.device
            ).reshape(num_reqs, max_model_len)
            lengths = torch.full((num_reqs,), 3, dtype=torch.int32, device=self.device)
            prompt = torch.full((num_reqs,), 5, dtype=torch.int32, device=self.device)
            total = prompt + 1
            if num_reqs >= 3:
                # Exercise replayed, skipped and untraced requests.
                mapping[0] = 1
                mapping[1] = -1
                mapping[2] = 0
                lengths[0] = 0

            arguments = (mapping, traces, lengths, total, prompt)
            if vllm_reference is not None:
                candidate_sampled = sampled.clone()
                reference_sampled = sampled.clone()
                assert (
                    flaggems_apply_trace_tokens(candidate_sampled, *arguments)
                    is None
                )
                assert vllm_reference(reference_sampled, *arguments) is None
                torch.testing.assert_close(
                    candidate_sampled, reference_sampled, rtol=0, atol=0
                )
            yield (sampled, *arguments)


@pytest.mark.apply_trace_tokens
def test_trace_replay_perf():
    bench = TraceReplayBenchmark(
        op_name="apply_trace_tokens",
        torch_op=vllm_reference,
        dtypes=[torch.int64],
        is_inplace=True,
    )
    bench.set_gems(flaggems_apply_trace_tokens)
    bench.run()
