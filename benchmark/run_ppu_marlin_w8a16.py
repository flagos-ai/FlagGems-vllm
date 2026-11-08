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

"""Reproduce the PPU INT8/FP8 53-shape BF16 comparison from PR #752.

Run from the repository root with DNN_VENDOR=thead, VLLM_PLUGINS=fl,
VLLM_FL_USE_FLAGGEMS_VLLM=1 and PYTHONPATH=src. Weight preparation and JIT
are outside timing; every GPU operation in the candidate call is inside.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from unittest.mock import patch

import torch
import triton
import triton.testing
from vllm.model_executor.layers.fused_moe import fused_moe as native
from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import (
    moe_align_block_size_flaggems,
    moe_sum_flaggems,
)
from vllm_fl.ops.fused_moe.activation import apply_moe_activation

import flaggems_vllm
from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_FP8_E4M3, QUANT_TYPE_UINT8B128

GROUP_SIZE = 128
BASELINE_SHA256 = "57a3f7efc9624f5c437908892ed5109a81ba0dbf45e1c6ac8a1438a23131f207"


def make_weights(fp8):
    """Use the frozen PPU generator; decode identical BF16 baseline weights."""
    torch.manual_seed(7)
    weights, scales, decoded = [], [], []
    for output_size, input_size in ((512, 4096), (4096, 256)):
        raw = torch.randint(
            0, 256, (256, output_size, input_size), device="cuda", dtype=torch.uint8
        )
        if fp8:
            raw = torch.where((raw & 127) == 127, raw - 1, raw)
            weight = raw.view(torch.float8_e4m3fn)
        else:
            weight = raw
        scale = (
            torch.rand((256, output_size, input_size // GROUP_SIZE), device="cuda")
            * 0.001
            + 0.0005
        ).to(torch.bfloat16)
        reference = torch.empty_like(weight, dtype=torch.bfloat16)
        for expert in range(256):
            values = weight[expert].float()
            if not fp8:
                values = values - 128
            reference[expert] = (
                values * scale[expert].float().repeat_interleave(GROUP_SIZE, dim=-1)
            ).to(torch.bfloat16)
        weights.append(weight)
        scales.append(scale)
        decoded.append(reference)
    return weights, scales, decoded


def baseline(x, weights, routing_weights, ids):
    # Restore all auxiliary bindings after each call. GEMMs and dispatch stay
    # in the upstream vLLM module, including when the candidate is timed next.
    with patch.multiple(
        native,
        moe_align_block_size=moe_align_block_size_flaggems,
        apply_moe_activation=apply_moe_activation,
    ), patch.object(native.ops, "moe_sum", moe_sum_flaggems):
        return native.fused_experts(x, *weights, routing_weights, ids)


def relative_errors(actual, expected):
    delta = actual.float() - expected.float()
    rms = (
        delta.square().mean().sqrt()
        / expected.float().square().mean().sqrt().clamp_min(1e-12)
    )
    peak = delta.abs().max() / expected.float().abs().max().clamp_min(1e-12)
    return rms.item(), peak.item()


def run(precision, shapes, rep):
    fp8 = precision == "fp8"
    quant = QUANT_TYPE_FP8_E4M3 if fp8 else QUANT_TYPE_UINT8B128
    weights, scales, decoded = make_weights(fp8)
    rows = []
    for shape in shapes:
        m = shape["shape"][0]
        torch.manual_seed(7 + m)
        x = torch.randn((m, 4096), device="cuda", dtype=torch.bfloat16) * 0.1
        ids = torch.rand((m, 256), device="cuda").topk(6, dim=-1).indices
        routing_weights = torch.randn((m, 6), device="cuda").softmax(-1)

        def call_baseline():
            return baseline(x, decoded, routing_weights, ids)

        def call_candidate():
            return flaggems_vllm.fused_marlin_moe(
                hidden_states=x,
                w1=weights[0],
                w2=weights[1],
                w1_scale=scales[0],
                w2_scale=scales[1],
                topk_weights=routing_weights,
                topk_ids=ids,
                bias1=None,
                bias2=None,
                quant_type_id=quant,
                group_size=GROUP_SIZE,
            )

        errors = relative_errors(call_candidate(), call_baseline())
        if not (errors[0] < 0.01 and errors[1] < 0.02):
            raise AssertionError((precision, m, errors))
        functions = {"bf16": call_baseline, "candidate": call_candidate}
        trials = {name: [] for name in functions}
        for trial in range(3):
            order = list(functions) if trial % 2 == 0 else list(reversed(functions))
            for name in order:
                trials[name].append(
                    triton.testing.do_bench_cudagraph(functions[name], rep=rep)
                )
        times = {name: statistics.median(values) for name, values in trials.items()}
        row = dict(
            shape=shape["shape"],
            count=shape["count"],
            trials_ms=trials,
            latency_ms=times,
            relative_rms=errors[0],
            relative_peak=errors[1],
            speedup=times["bf16"] / times["candidate"],
        )
        rows.append(row)
        print(precision, m, times, row["speedup"], flush=True)
    count = sum(row["count"] for row in rows)
    total = {
        name: sum(row["count"] * row["latency_ms"][name] for row in rows)
        for name in ("bf16", "candidate")
    }
    speeds = [row["speedup"] for row in rows]
    summary = dict(
        shape_count=len(rows),
        sum_count=count,
        mean_speedup=statistics.mean(speeds),
        count_weighted_mean_speedup=sum(row["count"] * row["speedup"] for row in rows)
        / count,
        minimum=min(speeds),
        maximum=max(speeds),
        faster=sum(s > 1 for s in speeds),
        cumulative_ms=total,
        cumulative_speedup=total["bf16"] / total["candidate"],
    )
    return dict(rows=rows, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("int8", "fp8", "both"), default="both")
    parser.add_argument(
        "--shape-file",
        type=Path,
        default=Path(__file__).parent / "results/ppu_marlin_w8a16_20261107/shapes.json",
    )
    parser.add_argument(
        "--tokens", help="Optional comma-separated M subset for a smoke run"
    )
    parser.add_argument(
        "--rep", type=int, default=100, help="Milliseconds per timing round"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if flaggems_vllm.vendor_name != "thead":
        raise RuntimeError("This benchmark requires the T-Head PPU environment")
    source_hash = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    if source_hash != BASELINE_SHA256:
        raise RuntimeError("vLLM baseline source differs from the frozen comparison")
    shapes = json.loads(args.shape_file.read_text())
    if len(shapes) != 53 or sum(row["count"] for row in shapes) != 10105:
        raise ValueError("Expected the frozen 53-shape, 10105-call histogram")
    if any(row["shape"][1:] != [256, 4096, 256, 6] for row in shapes):
        raise ValueError("Expected E=256, K=4096, N=256, topk=6")
    if args.tokens:
        tokens = set(map(int, args.tokens.split(",")))
        shapes = [row for row in shapes if row["shape"][0] in tokens]
        if {row["shape"][0] for row in shapes} != tokens:
            raise ValueError("M subset is absent from the frozen histogram")
    result = dict(
        baseline_source_sha256=source_hash, rep_ms=args.rep, rounds=3, results={}
    )
    for precision in ("int8", "fp8") if args.precision == "both" else (args.precision,):
        result["results"][precision] = run(precision, shapes, args.rep)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value["summary"] for key, value in result["results"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
