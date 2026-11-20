# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

"""Indexer latency from populated FP8 K cache to top-2048 indices."""

import pytest
import torch

from tests.test_flash_topk import _call, _case

from . import base


def _native_prefill(case, gathered, scale_bytes):
    from vllm import _custom_ops as ops
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    ops.cp_gather_indexer_k_quant_cache(
        case["kv"], gathered, scale_bytes, case["table"], case["offsets"]
    )
    logits = fp8_fp4_mqa_logits(
        (case["q"], None),
        (gathered, scale_bytes.view(torch.float32).view(-1)),
        case["weights"],
        case["starts"],
        case["ends"],
        clean_logits=False,
    )
    ops.top_k_per_row_prefill(
        logits,
        case["starts"],
        case["ends"],
        case["out"],
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        2048,
    )


def _native_decode(case, schedule, select):
    from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits

    lengths = case["lens"].view(case["q"].shape[:2])
    logits = fp8_fp4_paged_mqa_logits(
        (case["q"], None),
        case["kv"].unsqueeze(-2),
        case["weights"],
        lengths,
        case["table"],
        schedule,
        case["max_context"],
        clean_logits=False,
    )
    select(logits, lengths, case["out"])


def _make_decode_selector(case, device):
    width = case["max_context"]
    try:
        from vllm.model_executor.layers.indexer_topk import get_indexer_topk
    except ImportError:
        # vLLM 0.24 exposes two CUDA selectors directly.
        workspace = torch.empty((1024 * 1024,), dtype=torch.uint8, device=device)
        op = (
            torch.ops._C.cooperative_topk
            if case["q"].shape[0] <= 32
            else torch.ops._C.persistent_topk
        )

        def select_v024(logits, lengths, out):
            op(logits, lengths, out, workspace, 2048, width)

        return select_v024

    selector = get_indexer_topk("auto")

    def select_auto(logits, lengths, out):
        selector(logits, lengths, lengths.shape[1], out, 2048, width)

    return select_auto


class FlashTopKBenchmark(base.Benchmark):
    def __init__(self, *, prefill, **kwargs):
        self.prefill = prefill
        super().__init__(**kwargs)

    def set_shapes(self, shape_file_path=None):
        if self.prefill:
            self.shapes = [
                ((512,), (512,)),
                ((8192 + 32,), (32,)),
                ((32768 + 128,), (128,)),
                ((65536 + 128,), (128,)),
                ((122880 + 128, 81920 + 64), (128, 64)),
            ]
        else:
            self.shapes = [
                ((2048,), (1,)),
                ((4096,) * 8, (1,) * 8),
                ((8192,) * 32, (1,) * 32),
                ((65536,) * 32, (1,) * 32),
                ((131072,), (1,)),
            ]

    def get_input_iter(self, dtype):
        from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata

        for lengths, counts in self.shapes:
            case = _case(self.prefill, lengths, counts, dyadic=False)
            case["table"] = case["table"].contiguous()
            if self.prefill:
                gather = torch.empty(
                    (sum(lengths), 128), device=self.device, dtype=dtype
                )
                scales = torch.empty(
                    (sum(lengths), 4), device=self.device, dtype=torch.uint8
                )
                yield (case,), (gather, scales)
            else:
                lens = case["lens"].view(case["q"].shape[:2])
                schedule = get_paged_mqa_logits_metadata(
                    lens,
                    64,
                    torch.cuda.get_device_properties(lens.device).multi_processor_count,
                )
                yield (case,), (schedule, _make_decode_selector(case, self.device))


def flash_topk_benchmark(prefill):
    pytest.importorskip("vllm")
    from vllm import _custom_ops  # noqa: F401
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device("cuda:0"))
    return FlashTopKBenchmark(
        prefill=prefill,
        op_name="flash_topk_prefill" if prefill else "flash_topk_decode",
        torch_op=lambda packet, aux: (_native_prefill if prefill else _native_decode)(
            packet[0], *aux
        ),
        gems_op=lambda packet, aux: _call(prefill, packet[0]),
        dtypes=[torch.float8_e4m3fn],
    )
