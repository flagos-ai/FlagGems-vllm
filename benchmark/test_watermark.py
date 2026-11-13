# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Benchmark draft context-mask latency across accelerator backends."""

import pytest
import torch

import flaggems_vllm
from benchmark import base
from benchmark.conftest import Config
from flaggems_vllm import draft_watermarking_mask

try:
    from vllm.v1.worker.gpu.sample.watermark import (
        draft_watermarking_mask as vllm_reference,
    )
except (ImportError, OSError):
    vllm_reference = None


SHAPES = (
    [
        (n, history, 4, 8, 3, 8192, False, "fresh")
        for n in (1, 8, 32, 128, 512)
        for history in (64, 512, 2048, 8192)
    ]
    + [(n, 32768, 4, 8, 3, 8192, False, "fresh") for n in (32, 512)]
    + [(32, 8192, width, 8, 3, 8192, False, "fresh") for width in (1, 8, 16)]
    + [
        (32, 8192, 4, draft_steps, step, window, False, "fresh")
        for draft_steps, step, window in (
            (1, 0, 8192),
            (16, 15, 8192),
            (8, 3, 512),
            (8, 3, None),
        )
    ]
    + [
        (32, 8192, 4, 8, 3, 8192, include_prompt, "prompt_match")
        for include_prompt in (False, True)
    ]
    + [(32, 8192, 4, 8, 3, 8192, False, "all_equal")]
)


def make_inputs(shape, device=flaggems_vllm.device):
    n, history, width, draft_steps, step, window, include_prompt, pattern = shape
    generator = torch.Generator().manual_seed(20261107)
    tokens = torch.randint(
        0, 32000, (n, history + width + 8), generator=generator, dtype=torch.int32
    )
    contexts = torch.randint(
        0, 32000, (n, width), generator=generator, dtype=torch.int64
    )
    prior = torch.randint(
        0, 32000, (n, draft_steps, width), generator=generator, dtype=torch.int64
    )
    if pattern == "prompt_match":
        tokens[:, :width] = contexts
    elif pattern == "all_equal":
        tokens.zero_()
        contexts.zero_()
    inputs = (
        tokens,
        torch.arange(n, dtype=torch.int32),
        torch.full((n,), 4, dtype=torch.int32),
        torch.full((n,), history, dtype=torch.int32),
        prior,
        contexts,
        torch.full((n,), step, dtype=torch.int64),
        torch.ones(n, dtype=torch.bool),
    )
    return (*(tensor.to(device) for tensor in inputs), window, include_prompt)


def case_scenario(shape):
    _, history, width, draft_steps, step, window, include_prompt, pattern = shape
    if pattern == "prompt_match":
        return "prompt_included" if include_prompt else "prompt_excluded"
    if pattern == "all_equal":
        return "all_tokens_equal"
    if width != 4:
        return "context_width_sweep"
    if window is None:
        return "unlimited_history"
    if window != 8192:
        return "short_window"
    if draft_steps != 8 or step != 3:
        return "draft_steps_sweep"
    if history <= 512:
        return "short_history"
    if history == 2048:
        return "medium_history"
    return "window_length_history" if history == 8192 else "beyond_window_history"


class DraftWatermarkBenchmark(base.Benchmark):
    DEFAULT_SHAPES = SHAPES
    DEFAULT_SHAPE_DESC = (
        "requests, history, context_width, draft_steps, step, max_history, "
        "include_prompt, pattern"
    )

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def record_shapes(self, *args, **kwargs):
        return {
            "case": self._case_shape,
            "scenario": case_scenario(self._case_shape),
            "inputs": super().record_shapes(*args, **kwargs),
        }

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            self._case_shape = shape
            inputs = make_inputs(shape, self.device)
            prior = inputs[4].clone()
            if vllm_reference is not None:
                expected = vllm_reference(*inputs)
                expected_prior = inputs[4].clone()
                inputs[4].copy_(prior)
                actual = draft_watermarking_mask(*inputs)
                assert torch.equal(actual, expected)
                assert torch.equal(inputs[4], expected_prior)
            else:
                actual = draft_watermarking_mask(*inputs)
                assert actual.shape == (shape[0],)
                assert actual.dtype == torch.bool
            inputs[4].copy_(prior)
            yield inputs

    def run(self):
        if vllm_reference is not None or Config.query:
            return super().run()
        self.init_user_config()
        rows = []
        for dtype in self.to_bench_dtypes:
            for inputs in self.get_input_iter(dtype):
                args, kwargs = self.unpack_to_args_kwargs(inputs)
                latency_us = self.get_latency(self.gems_op, *args, **kwargs) * 1000
                rows.append((self._case_shape, latency_us))
        print(
            "\nDraft watermark mask: FlagGems-vllm latency (vLLM reference unavailable)"
        )
        print("Requests  History  Width  Scenario                 FlagGems-vllm (us)")
        print("-" * 72)
        for shape, latency_us in rows:
            print(
                f"{shape[0]:>8}  {shape[1]:>7}  {shape[2]:>5}  "
                f"{case_scenario(shape):<24} {latency_us:>18.3f}"
            )


@pytest.mark.draft_watermarking_mask
def test_draft_watermarking_mask_perf():
    bench = DraftWatermarkBenchmark(
        op_name="draft_watermarking_mask",
        torch_op=vllm_reference,
        dtypes=[torch.int64],
        is_inplace=True,
    )
    bench.set_gems(draft_watermarking_mask)
    bench.run()
