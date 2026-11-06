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

"""Paired TP benchmark using the repository shape/metric/reporting framework.

Unlike the default single-device timer, this adapter takes the slowest rank's
mean latency, then averages five independent worker rounds. --warmup/--iter
are counts for this distributed operator, matching the MegaMoE convention.
"""

import hashlib
import inspect
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

from tests.megapreattention_utils import Output, run_distributed

from . import base
from .conftest import Config


class NativeWindow:
    """Library AG/GEMM + original vLLM projection method + native cache op.

    The text baseline enables vLLM's own fast fused norm/RoPE/gate path.
    Multimodal checks use its original non-fused MRoPE path instead.
    """

    def __init__(self, inputs, rope, *, edge=False):
        from vllm import _custom_ops
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm
        from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

        symm.enable_symm_mem_for_group(inputs.group.group_name)
        qnorm = SimpleNamespace(
            weight=inputs.q_norm, variance_epsilon=inputs.config.eps
        )
        knorm = SimpleNamespace(
            weight=inputs.k_norm, variance_epsilon=inputs.config.eps
        )
        qnorm_call = lambda x: GemmaRMSNorm.forward_native(qnorm, x)
        knorm_call = lambda x: GemmaRMSNorm.forward_native(knorm, x)
        # The projection method reads .weight even for its fused fast path.
        qnorm_call.weight, qnorm_call.variance_epsilon = (
            inputs.q_norm,
            inputs.config.eps,
        )
        knorm_call.weight, knorm_call.variance_epsilon = (
            inputs.k_norm,
            inputs.config.eps,
        )
        self.context = SimpleNamespace(
            use_fused_qk_norm_rope_gate=not edge,
            attn_output_gate=True,
            q_size=inputs.config.q_size,
            kv_size=inputs.config.kv_size,
            num_heads=inputs.config.num_q_heads,
            num_kv_heads=inputs.config.num_kv_heads,
            head_dim=inputs.config.head_dim,
            q_norm=qnorm_call,
            k_norm=knorm_call,
            rotary_emb=rope,
        )
        self.method, self.cache_op = (
            Qwen3NextAttention._project_qkv_gate,
            _custom_ops.reshape_and_cache_flash,
        )
        self.scale = inputs.scale
        self.group_name = inputs.group.group_name
        self.post = torch.compile(self._postprocess, fullgraph=True, dynamic=False)
        self.source_evidence = {
            "projection_method_sha256": hashlib.sha256(
                inspect.getsource(self.method).encode()
            ).hexdigest(),
            "projection_file": inspect.getfile(self.method),
            "text_fused_path": not edge,
            "cache_op": "vllm._custom_ops.reshape_and_cache_flash",
            "communication_gemm": "torch.ops.symm_mem.fused_all_gather_matmul.default",
            "compile": "fullgraph=True,dynamic=False; postprocessing only",
        }

    def _postprocess(self, qkv, positions, cache, slots):
        q, k, v, gate = self.method(self.context, qkv, positions)
        key_cache, value_cache = cache.transpose(1, 2).split(
            self.context.head_dim, dim=-1
        )
        self.cache_op(
            k.view(-1, self.context.num_kv_heads, self.context.head_dim),
            v.view(-1, self.context.num_kv_heads, self.context.head_dim),
            key_cache,
            value_cache,
            slots,
            "auto",
            self.scale,
            self.scale,
        )
        return (
            q,
            k,
            v,
            gate.view(q.shape[0], self.context.num_heads, self.context.head_dim),
        )

    def __call__(self, inputs):
        gathered, products = torch.ops.symm_mem.fused_all_gather_matmul.default(
            inputs.hidden, [inputs.weight.t()], 0, self.group_name
        )
        values = self.post(
            products[0], inputs.positions, inputs.cache, inputs.slots.contiguous()
        )
        return Output(*values, (gathered, products[0]))


def distribution(samples):
    ordered = sorted(samples)

    def percentile(q):
        index = (len(ordered) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    median = statistics.median(samples)
    span = (percentile(0.9) - percentile(0.1)) / median
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": median,
        "span": span,
        "stable": len(samples) >= 10 and span <= 0.05,
        "samples_ms": samples,
    }


def measure_window(calls, names, iterations):
    from torch._dynamo.utils import counters as dynamo_counters
    from triton.runtime.jit import JITFunction

    samples = {name: {"wall": [], "device": []} for name in names}
    compilations = []
    original_compile = JITFunction._do_compile

    def track_compile(*args, **kwargs):
        compilations.append(True)
        return original_compile(*args, **kwargs)

    before_dynamo = dict(dynamo_counters["stats"])
    JITFunction._do_compile = track_compile
    try:
        for repeat in range(iterations):
            shift = repeat % len(names)
            for name in names[shift:] + names[:shift]:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                dist.barrier()
                torch.cuda.synchronize()
                begin = time.perf_counter_ns()
                start.record()
                calls[name]()
                end.record()
                torch.cuda.synchronize()
                samples[name]["wall"].append((time.perf_counter_ns() - begin) / 1.0e6)
                samples[name]["device"].append(start.elapsed_time(end))
    finally:
        JITFunction._do_compile = original_compile
    assert not compilations, "measurement triggered TLE/Triton JIT"
    assert before_dynamo == dict(
        dynamo_counters["stats"]
    ), "measurement triggered Dynamo recompilation"
    return samples


def summarize_samples(ranks, names, iterations):
    metrics = {}
    for name in names:
        metrics[name] = {}
        for kind in ("wall", "device"):
            per_rank_means = [statistics.mean(r["samples"][name][kind]) for r in ranks]
            rank_max = [
                max(r["samples"][name][kind][i] for r in ranks)
                for i in range(iterations)
            ]
            metrics[name][kind] = {
                "slowest_rank_mean_ms": max(per_rank_means),
                "rank_mean_ms": per_rank_means,
                "rank_max_distribution": distribution(rank_max),
            }
    return metrics


def summarize_rounds(reports, mode):
    summary = {}
    kind = "wall" if mode == "operator" else "device"
    for name in ("native", "megapreattention"):
        times = [r["metrics"][name][kind]["slowest_rank_mean_ms"] for r in reports]
        summary[name] = {
            "latency_ms": statistics.mean(times),
            "round_latency_ms": times,
            "stable_all": all(
                r["metrics"][name][kind]["rank_max_distribution"]["stable"]
                for r in reports
            ),
        }
    candidate = summary["megapreattention"]
    candidate["speedup"] = summary["native"]["latency_ms"] / candidate["latency_ms"]
    candidate["performance_threshold_passed"] = candidate["speedup"] >= 0.95
    candidate["validated_stable_ratio"] = (
        summary["native"]["stable_all"] and candidate["stable_all"]
    )
    return summary


class MegaPreAttentionBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))

    def __init__(self):
        super().__init__("megapreattention", self.native, gems_op=self.candidate)
        self.paired_results = {}

    @staticmethod
    def native():
        """Baseline label; worker ranks execute the native window."""

    @staticmethod
    def candidate():
        """Candidate label; worker ranks execute megapreattention."""

    def get_input_iter(self, dtype):
        yield from self.shapes

    def get_latency(
        self, op, tokens, tp, hidden, q_heads, kv_heads, head_dim, rotary_dim
    ):
        if not torch.cuda.is_available() or torch.cuda.device_count() < tp:
            pytest.skip(f"requires {tp} CUDA GPUs")
        if any(torch.cuda.get_device_capability(i) != (9, 0) for i in range(tp)):
            pytest.skip("requires SM90 GPUs")
        mode = Config.mode.value
        if mode not in ("kernel", "operator", "cudagraph"):
            raise ValueError(
                "megapreattention requires kernel, operator or cudagraph mode"
            )
        warmup = (
            Config.warm_up
            if any(a == "--warmup" or a.startswith("--warmup=") for a in sys.argv)
            else 20
        )
        iterations = (
            Config.repetition
            if any(a == "--iter" or a.startswith("--iter=") for a in sys.argv)
            else 100
        )
        key = (
            tokens,
            tp,
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            mode,
            warmup,
            iterations,
        )
        if key not in self.paired_results:
            self.paired_results[key] = run_distributed(
                tp,
                tokens,
                mode,
                warmup=warmup,
                iterations=iterations,
                rounds=5,
                model_dims=(hidden, q_heads, kv_heads),
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                mrope_section=(
                    (11, 11, 10)
                    if rotary_dim == 64
                    else (
                        rotary_dim // 6,
                        rotary_dim // 6,
                        rotary_dim // 2 - 2 * (rotary_dim // 6),
                    )
                ),
            )
        name = "native" if op == self.native else "megapreattention"
        return self.paired_results[key]["summary"][name]["latency_ms"]


@pytest.mark.megapreattention
def test_megapreattention():
    MegaPreAttentionBenchmark().run()
