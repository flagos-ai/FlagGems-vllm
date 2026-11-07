# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Draft context-mask latency against the vLLM implementation."""

import pytest
import torch

from benchmark import base
from flaggems_vllm import draft_watermarking_mask

try:
    from vllm.v1.worker.gpu.sample.watermark import (
        draft_watermarking_mask as vllm_reference,
    )
except (ImportError, OSError):
    vllm_reference = None

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# N, committed length, context width, draft capacity, step, window, prompt scope,
# repeated-context distribution, and history storage.
SHAPES = (
    [
        (n, history, 4, 8, 3, 8192, False, "fresh", "cuda")
        for n in (1, 8, 32, 128, 512)
        for history in (64, 512, 2048, 8192, 32768)
    ]
    + [
        (32, 8192, width, 8, 3, 8192, include, pattern, "cuda")
        for width in (4, 8)
        for include in (False, True)
        for pattern in ("early", "late", "draft", "mixed")
    ]
    + [
        (n, history, 4, 8, 3, 8192, False, "fresh", "uva")
        for n in (1, 32, 128)
        for history in (512, 8192)
    ]
    + [
        (32, 8192, 4, draft_steps, step, window, False, "fresh", "cuda")
        for draft_steps, step, window in (
            (1, 0, 8192),
            (4, 3, 8192),
            (16, 15, 8192),
            (8, 3, 512),
            (8, 3, None),
        )
    ]
    + [(32, 8192, width, 8, 3, 8192, False, "fresh", "cuda") for width in (1, 2, 16)]
    + [
        (n, history, 4, 8, 3, 8192, False, "fresh", "cuda", 32780)
        for n in (1, 32, 128)
        for history in (64, 512, 8192)
    ]
    + [
        (n, 8192, 4, 8, 3, 8192, False, pattern, "cuda")
        for n in (32, 512)
        for pattern in ("small_vocab", "all_equal", "anchor_collision")
    ]
    + [
        (n, history, 4, 8, 3, 8192, False, "fresh", "cuda", 32768)
        for n in (32, 128)
        for history in (8192, 512)
    ]
)


def make_inputs(shape, device="cuda"):
    (
        n,
        history,
        width,
        draft_steps,
        step,
        window,
        include,
        pattern,
        storage,
        *capacity,
    ) = shape
    generator = torch.Generator().manual_seed(20261107)
    tokens = torch.randint(
        0,
        32000,
        (n, capacity[0] if capacity else history + width + 8),
        generator=generator,
        dtype=torch.int32,
    )
    contexts = torch.randint(
        0, 32000, (n, width), generator=generator, dtype=torch.int64
    )
    prior = torch.randint(
        0, 32000, (n, draft_steps, width), generator=generator, dtype=torch.int64
    )
    for row in range(n):
        row_pattern = (
            ("fresh", "early", "draft", "disabled")[row % 4]
            if pattern == "mixed"
            else pattern
        )
        if row_pattern == "early":
            tokens[row, 4 : 4 + width] = contexts[row]
        elif row_pattern == "late":
            tokens[row, history - width - 1 : history - 1] = contexts[row]
        elif row_pattern == "draft":
            prior[row, 0] = contexts[row]
    if pattern == "small_vocab":
        tokens.remainder_(32)
        contexts.remainder_(32)
        prior.remainder_(32)
    elif pattern in ("all_equal", "anchor_collision"):
        tokens.zero_()
        contexts.zero_()
        if pattern == "anchor_collision":
            contexts[:, 0] = 17
    enabled = torch.ones(n, dtype=torch.bool)
    if pattern == "mixed":
        enabled[3::4] = False
    inputs = [
        tokens,
        torch.arange(n, dtype=torch.int32),
        torch.full((n,), 4, dtype=torch.int32),
        torch.full((n,), history, dtype=torch.int32),
        prior,
        contexts,
        torch.tensor(step, dtype=torch.int64).expand(n),
        enabled,
    ]
    inputs = [t.to(device) for t in inputs]
    inputs[6] = torch.tensor(step, dtype=torch.int64, device=device).expand(n)
    if storage == "uva":
        from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool

        pool = UvaBufferPool(tokens.shape, tokens.dtype, max_concurrency=1)
        inputs[0] = pool.copy_to_uva(tokens)
        inputs[0]._watermark_history_owner = pool
    return (*inputs, window, include)


def case_scenario(shape):
    _, history, width, draft_steps, step, window, _, pattern, storage, *capacity = shape
    if storage == "uva":
        return "uva_history"
    if pattern != "fresh":
        return {
            "early": "early_history_match",
            "late": "late_history_match",
            "draft": "prior_draft_match",
            "mixed": "mixed_requests",
            "small_vocab": "small_vocabulary",
            "all_equal": "all_tokens_equal",
            "anchor_collision": "anchor_collision",
        }[pattern]
    if capacity:
        return "aligned_capacity" if capacity[0] == 32768 else "fixed_capacity"
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
        "include_prompt, pattern, storage, optional_capacity"
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
        if vllm_reference is None:
            pytest.skip("requires vLLM with draft_watermarking_mask")
        for shape in self.shapes:
            self._case_shape = shape
            inputs = make_inputs(shape, self.device)
            prior = inputs[4].clone()
            expected = vllm_reference(*inputs)
            expected_prior = inputs[4].clone()
            inputs[4].copy_(prior)
            actual = draft_watermarking_mask(*inputs)
            assert torch.equal(actual, expected)
            assert torch.equal(inputs[4], expected_prior)
            inputs[4].copy_(prior)
            yield inputs


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
