# SPDX-License-Identifier: Apache-2.0
"""Compare the pinned Hygon vendor baseline and FlagTree MegaMHC with and without CUDA Graph."""

import importlib.metadata
import importlib.util
import json
import os
import statistics
from functools import partial

import pytest
import torch

import flaggems_vllm
from tests.test_megamhc import (
    MAX_DIFF,
    OUTPUTS,
    difference,
    invoke,
    make_inputs,
    reference,
)

CORE_TOKENS = (1, 24, 64, 128, 512, 1028, 2048, 3076, 16384)
GRAPH_CALLS = 16
GRAPH_WARMUP = 5
SAMPLES = 21
EAGER_TRIALS = 3
EAGER_WARMUP_MS = 100
EAGER_MEASURE_MS = 300
VLLM_VERSION = "0.29.0"
AITER_VERSION = "0.1.5+dtk2604.torch2110.2608272215.g561f76"
TORCH_VERSION = "2.11.0+das.opt1.dtk2604.2608241728.gfb4aa7"
BASELINE_IMAGE = (
    "harbor.sourcefind.cn:5443/dcu/admin/base/custom:"
    "vllm0.29.0-ubuntu22.04-dtk26.04-py3.10-20260831-qwen3.8flashnext"
)


def frozen_baseline():
    from aiter.ops.mhc import mhc_pre
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.kernels.mhc import mhc_post_aiter

    for package, expected in (
        ("vllm", VLLM_VERSION),
        ("aiter", AITER_VERSION),
        ("torch", TORCH_VERSION),
    ):
        installed = importlib.metadata.version(package)
        if installed != expected:
            raise RuntimeError(
                f"Expected {package} {expected} from {BASELINE_IMAGE}, got {installed}"
            )

    def vendor_megamhc(
        x,
        residual,
        post,
        comb,
        weight,
        scale,
        bias,
        rms_eps,
        pre_eps,
        sinkhorn_eps,
        multiplier,
        iterations,
        _splits,
        _tile_n,
        norm,
        norm_eps,
    ):
        updated = mhc_post_aiter(x, residual, post, comb)
        # The bundled vLLM pre wrapper passes RMSNorm arguments this AITER
        # build does not accept. Call its registered op and apply RMSNorm here.
        new_post, new_comb, layer_input = mhc_pre(
            updated,
            weight,
            scale,
            bias,
            rms_eps,
            pre_eps,
            sinkhorn_eps,
            multiplier,
            iterations,
        )
        normalized = rocm_aiter_ops.rms_norm(layer_input, norm, norm_eps)
        return updated, new_post, new_comb, normalized

    return vendor_megamhc


def capture(op, inputs, stream):
    for _ in range(3):
        invoke(op, inputs)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(GRAPH_CALLS):
            outputs = invoke(op, inputs)
    return graph, outputs


def verify_outputs(outputs, expected):
    errors = {
        name: difference(actual, target)
        for name, actual, target in zip(OUTPUTS, outputs, expected, strict=True)
    }
    if not all(error < MAX_DIFF for error in errors.values()):
        raise ValueError(f"MegaMHC output exceeded the reference threshold: {errors}")
    return max(errors.values())


def prepare_candidate(candidate, inputs, prepare):
    if prepare is None:
        return candidate
    prepared = prepare(inputs[4])
    return partial(candidate, prepared=prepared)


def compare(baseline, candidate, tokens, *, prepare=None):
    inputs = make_inputs(tokens)
    candidate = prepare_candidate(candidate, inputs, prepare)
    expected = reference(inputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        base_graph, base_output = capture(baseline, inputs, stream)
        test_graph, test_output = capture(candidate, inputs, stream)
        for _ in range(GRAPH_WARMUP):
            base_graph.replay()
            test_graph.replay()
        stream.synchronize()
        base_error = verify_outputs(base_output, expected)
        test_error = verify_outputs(test_output, expected)
        stream.synchronize()

        events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
        tasks = ((base_graph, []), (test_graph, []))
        for sample in range(SAMPLES):
            order = tasks if sample % 2 == 0 else reversed(tasks)
            for graph, times in order:
                start, end = events
                start.record(stream)
                graph.replay()
                end.record(stream)
                end.synchronize()
                times.append(start.elapsed_time(end) * 1000 / GRAPH_CALLS)

    latency_base = statistics.median(tasks[0][1])
    latency_test = statistics.median(tasks[1][1])
    return {
        "tokens": tokens,
        "baseline_us": latency_base,
        "flagtree_us": latency_test,
        "speedup": latency_base / latency_test,
        "baseline_max_error": base_error,
        "flagtree_max_error": test_error,
    }


def compare_eager(baseline, candidate, tokens, *, prepare=None):
    import triton.testing

    inputs = make_inputs(tokens)
    candidate = prepare_candidate(candidate, inputs, prepare)
    expected = reference(inputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    operators = (baseline, candidate)
    samples = ([], [])
    with torch.cuda.stream(stream):
        errors = [verify_outputs(invoke(op, inputs), expected) for op in operators]
        stream.synchronize()
        for trial in range(EAGER_TRIALS):
            order = (0, 1) if (trial + tokens) % 2 == 0 else (1, 0)
            for index in order:
                op = operators[index]
                elapsed = triton.testing.do_bench(
                    lambda: invoke(op, inputs),
                    warmup=EAGER_WARMUP_MS,
                    rep=EAGER_MEASURE_MS,
                    return_mode="median",
                )
                samples[index].append(elapsed * 1000)

    latency_base, latency_test = map(statistics.median, samples)
    return {
        "tokens": tokens,
        "baseline_us": latency_base,
        "flagtree_us": latency_test,
        "speedup": latency_base / latency_test,
        "baseline_max_error": errors[0],
        "flagtree_max_error": errors[1],
        "baseline_trials_us": samples[0],
        "flagtree_trials_us": samples[1],
    }


class HygonMegaMHCBenchmark:
    def __init__(self, baseline, candidate, prepare=None):
        self.baseline = baseline
        self.candidate = candidate
        self.prepare = prepare

    @torch.inference_mode()
    def run(self, selected_mode):
        print(
            json.dumps(
                {
                    "device": torch.cuda.get_device_properties(0).gcnArchName,
                    "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
                    "baseline": "Hygon vLLM image: AITER post + pre + weighted RMSNorm",
                    "baseline_image": BASELINE_IMAGE,
                    "vllm": VLLM_VERSION,
                    "aiter": AITER_VERSION,
                    "torch": TORCH_VERSION,
                    "flagtree": importlib.metadata.version("flagtree"),
                    "graph_calls": GRAPH_CALLS,
                    "graph_warmup": GRAPH_WARMUP,
                    "samples": SAMPLES,
                    "mode": selected_mode,
                    "variant": "prepared" if self.prepare else "raw",
                    "eager_trials": EAGER_TRIALS,
                    "eager_warmup_ms": EAGER_WARMUP_MS,
                    "eager_measure_ms": EAGER_MEASURE_MS,
                }
            ),
            flush=True,
        )
        runners = {"graph": compare, "eager": compare_eager}
        modes = runners if selected_mode == "both" else (selected_mode,)
        speedups = {mode: [] for mode in modes}
        for tokens in CORE_TOKENS:
            results = {}
            for mode in modes:
                result = runners[mode](
                    self.baseline, self.candidate, tokens, prepare=self.prepare
                )
                speedups[mode].append(result["speedup"])
                results[mode] = result
            row = (
                {"tokens": tokens, **results}
                if selected_mode == "both"
                else results[selected_mode]
            )
            print(json.dumps(row), flush=True)

        summary = {
            mode: {
                "arithmetic_mean_speedup": statistics.mean(values),
                "geomean_speedup": statistics.geometric_mean(values),
                "wins": sum(value > 1 for value in values),
            }
            for mode, values in speedups.items()
        }
        row = summary if selected_mode == "both" else summary[selected_mode]
        print(json.dumps(row), flush=True)


@pytest.mark.megamhc
def test_megamhc_performance():
    if flaggems_vllm.vendor_name == "hygon":
        if any(importlib.util.find_spec(name) is None for name in ("vllm", "aiter")):
            pytest.skip(f"requires AITER bundled in {BASELINE_IMAGE}")
        if torch.cuda.device_count() != 1:
            pytest.fail("Expose one Hygon DCU via HIP_VISIBLE_DEVICES")
        mode = os.environ.get("MEGAMHC_MODE", "both")
        if mode not in {"graph", "eager", "both"}:
            pytest.fail("MEGAMHC_MODE must be graph, eager, or both")
        baseline = frozen_baseline()
        variant = os.environ.get("MEGAMHC_VARIANT", "raw")
        if variant not in {"raw", "prepared"}:
            pytest.fail("MEGAMHC_VARIANT must be raw or prepared")
        prepare = (
            partial(flaggems_vllm.mega_mhc_prepare, max_decode_tokens=128)
            if variant == "prepared"
            else None
        )
        HygonMegaMHCBenchmark(baseline, flaggems_vllm.mega_mhc, prepare).run(mode)
    else:
        pytest.skip("MegaMHC performance is currently configured for Hygon")
