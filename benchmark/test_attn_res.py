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

import math
import statistics

import pytest
import torch

import flaggems_vllm

try:
    from vllm.models.kimi_k3.nvidia.ops.attn_res import attn_res as vllm_attn_res

    HAS_VLLM_ATTN_RES = True
except (ImportError, OSError):
    vllm_attn_res = None
    HAS_VLLM_ATTN_RES = False

from . import base

EPS = 1e-5
HIDDEN_SIZE = 7168
MAX_BLOCKS = 8
CASE_NAMES = ("write", "post", "common", "final")
TOKEN_SHAPES = (
    ("c1_decode", 1),
    ("c64_decode", 64),
    ("c1_prefill", 7680),
    ("c64_prefill", 16384),
)


def _do_bench_cudagraph_pair(
    baseline_fn,
    gems_fn,
    rep_ms: int,
) -> tuple[float, float]:
    """Measure two captured graphs with balanced replay ordering."""
    functions = (baseline_fn, gems_fn)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for fn in functions:
            fn()
        stream.synchronize()

        estimates = []
        for fn in functions:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(5):
                fn()
            end_event.record()
            end_event.synchronize()
            estimates.append(start_event.elapsed_time(end_event) / 5)

        max_estimate_ms = max(estimates)
        repeats = (
            1000 if max_estimate_ms == 0 else max(1, int(rep_ms / max_estimate_ms))
        )
        graphs = []
        for fn in functions:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(repeats):
                    fn()
            graphs.append(graph)
        stream.synchronize()

        samples = ([], [])
        for retry in range(10):
            order = (0, 1) if retry % 2 == 0 else (1, 0)
            events = {}
            for provider_idx in order:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                graphs[provider_idx].replay()
                end_event.record()
                events[provider_idx] = (start_event, end_event)
            stream.synchronize()
            for provider_idx, (start_event, end_event) in events.items():
                samples[provider_idx].append(
                    start_event.elapsed_time(end_event) / repeats
                )

    return statistics.median(samples[0]), statistics.median(samples[1])


def _cases(
    case_names: tuple[str, ...] = CASE_NAMES,
) -> list[tuple[str, int, int, int, bool, bool]]:
    cases = []
    for workload, num_tokens in TOKEN_SHAPES:
        if "write" in case_names:
            cases.extend(
                (
                    f"{workload}/block{num_blocks}_write",
                    num_tokens,
                    num_blocks,
                    num_blocks,
                    num_blocks > 0,
                    True,
                )
                for num_blocks in range(MAX_BLOCKS)
            )
        if "post" in case_names:
            cases.extend(
                (
                    f"{workload}/post_block{num_blocks}",
                    num_tokens,
                    num_blocks,
                    -1,
                    False,
                    True,
                )
                for num_blocks in range(1, MAX_BLOCKS + 1)
            )
        if "common" in case_names:
            cases.extend(
                (
                    f"{workload}/common_block{num_blocks}",
                    num_tokens,
                    num_blocks,
                    -1,
                    True,
                    True,
                )
                for num_blocks in range(1, MAX_BLOCKS + 1)
            )
        if "final" in case_names:
            cases.append((f"{workload}/final", num_tokens, 8, -1, True, False))
    return cases


def _call_flaggems_vllm(vllm_args, flaggems_vllm_args):
    del vllm_args
    return flaggems_vllm.attn_res(*flaggems_vllm_args)


def _call_vllm(vllm_args, flaggems_vllm_args):
    del flaggems_vllm_args
    return vllm_attn_res(*vllm_args)


class AttnResBenchmark(base.Benchmark):
    def __init__(self, case_names: tuple[str, ...] = CASE_NAMES):
        selected_cases = frozenset(case_names)
        self.case_names = tuple(name for name in CASE_NAMES if name in selected_cases)
        super().__init__(
            "attn_res",
            _call_vllm,
            [torch.bfloat16],
            gems_op=_call_flaggems_vllm,
        )

    def set_shapes(self, shape_file_path=None):
        del shape_file_path
        self.shapes = _cases(self.case_names)
        points_per_workload = MAX_BLOCKS * sum(
            name != "final" for name in self.case_names
        ) + ("final" in self.case_names)
        assert len(self.shapes) == len(TOKEN_SHAPES) * points_per_workload
        self.shape_desc = "case, tokens, num_blocks, write_idx, has_delta, output_norm"
        self._current_case = None

    def get_paired_latency(self, baseline_op, gems_op, *args, **kwargs):
        if base.Config.mode != base.consts.BenchMode.CUDAGRAPH:
            return None
        baseline_fn = lambda: baseline_op(*args, **kwargs)
        gems_fn = lambda: gems_op(*args, **kwargs)
        return _do_bench_cudagraph_pair(
            baseline_fn,
            gems_fn,
            base.Config.repetition,
        )

    def get_input_iter(self, dtype):
        torch.manual_seed(2026)
        for (
            case_name,
            num_tokens,
            num_blocks,
            block_write_idx,
            has_delta,
            apply_output_norm,
        ) in self.shapes:
            prefix = torch.randn(
                (num_tokens, HIDDEN_SIZE),
                device=self.device,
                dtype=dtype,
            )
            # Keep graph replays idempotent while preserving the HAS_DELTA path.
            delta = torch.zeros_like(prefix) if has_delta else None
            blocks = torch.randn(
                (num_tokens, MAX_BLOCKS, HIDDEN_SIZE),
                device=self.device,
                dtype=dtype,
            )
            norm_weight = 1 + 0.1 * torch.randn(
                (HIDDEN_SIZE,),
                device=self.device,
                dtype=dtype,
            )
            qk_weight = torch.randn(
                (HIDDEN_SIZE,),
                device=self.device,
                dtype=dtype,
            ) / math.sqrt(HIDDEN_SIZE)
            output_norm_weight = (
                1
                + 0.1
                * torch.randn(
                    (HIDDEN_SIZE,),
                    device=self.device,
                    dtype=dtype,
                )
                if apply_output_norm
                else None
            )
            self._current_case = case_name
            vllm_args = (
                prefix,
                delta,
                blocks,
                norm_weight,
                qk_weight,
                output_norm_weight,
                num_blocks,
                block_write_idx,
                EPS,
                EPS,
            )
            flaggems_vllm_args = (
                prefix.clone(),
                delta,
                blocks.clone(),
                norm_weight,
                qk_weight,
                output_norm_weight,
                num_blocks,
                block_write_idx,
                EPS,
                EPS,
            )
            yield vllm_args, flaggems_vllm_args

    def record_shapes(self, *args, **kwargs):
        del kwargs
        vllm_args = args[0]
        return {
            "case": self._current_case,
            "tokens": vllm_args[0].shape[0],
            "hidden_size": vllm_args[0].shape[1],
            "max_blocks": vllm_args[2].shape[1],
            "num_blocks": vllm_args[6],
            "block_write_idx": vllm_args[7],
            "has_delta": vllm_args[1] is not None,
            "output_norm": vllm_args[5] is not None,
        }


@pytest.mark.attn_res
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0)
    or not HAS_VLLM_ATTN_RES,
    reason="requires CUDA SM90 and vLLM Kimi K3 AttnRes",
)
@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_attn_res_benchmark(case_name):
    AttnResBenchmark((case_name,)).run()
