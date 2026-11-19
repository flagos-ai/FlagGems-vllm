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

"""Benchmark FlagGems-vllm persistent_topk against the vendor-native kernel."""

import pytest
import torch

import flaggems_vllm

from . import base

VENDOR = flaggems_vllm.vendor_name


K = 512
WORKSPACE_BYTES = 1 << 20

_HOMO_SHAPES = [
    # DeepSeek-V4.1-Flash, 1k context, ratio 1
    (67, 1044, 65536, 65536),
    (128, 1044, 65536, 65536),
    (256, 1044, 65536, 65536),
    (508, 1044, 65536, 65536),
    # DeepSeek-V4.1-Flash, 1k context, ratio 2
    (67, 522, 32768, 32768),
    (128, 522, 32768, 32768),
    (256, 522, 32768, 32768),
    (508, 522, 32768, 32768),
    # DeepSeek-V4.1-Flash, 4k context (ratio 1 / ratio 2)
    (16, 4096, 65536, 65536),
    (125, 4096, 65536, 65536),
    (48, 2048, 32768, 32768),
    (120, 2048, 32768, 32768),
    # DeepSeek-V4.1-Flash, 32k context (ratio 1 / ratio 2)
    (3, 32769, 65536, 65536),
    (5, 16384, 32768, 32768),
    # DeepSeek-V4-Flash
    (12, 4105, 4105, 262144),
    (24, 4108, 4108, 262144),
    (10, 1055, 1055, 262144),
    (1, 32773, 32773, 262144),
]


def _get_supported_dtypes():
    return [torch.float32]


def _input_fn(shape, dtype, device):

    num_rows, seq_len, max_seq_len, stride = shape

    logits = torch.full((num_rows, stride), float("-inf"), dtype=dtype, device=device)
    logits[:, :seq_len] = torch.randn(num_rows, seq_len, dtype=dtype, device=device)
    lengths = torch.full((num_rows,), seq_len, dtype=torch.int32, device=device)

    indices = torch.empty((num_rows, K), dtype=torch.int32, device=device)
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)

    yield logits, lengths, indices, workspace, K, max_seq_len


def _native_persistent_topk():
    """Resolve torch.ops._C.persistent_topk when this vendor's vLLM exposes it."""
    try:
        import vllm._custom_ops  # noqa: F401

        if VENDOR == "metax":
            try:
                import mcoplib._C  # noqa: F401
            except ImportError:
                pass
        op = getattr(getattr(torch.ops, "_C", None), "persistent_topk", None)
        return op if callable(op) else None
    except Exception:
        return None


def _vllm_native_op(logits, lengths, indices, workspace, k, max_seq_len):

    torch.ops._C.persistent_topk(logits, lengths, indices, workspace, k, max_seq_len)

    return indices


def _torch_persistent_topk(logits, lengths, indices, workspace, k, max_seq_len):
    """torch reference (benchmark-only): top-k over each row's valid prefix.

    Masks out the -inf tail beyond ``lengths[r]`` and pads the result with -1,
    matching persistent_topk's semantics (ids unsorted).
    """
    width = logits.shape[1]
    cols = torch.arange(width, device=logits.device)
    valid = cols[None, :] < lengths[:, None].to(cols.dtype)
    masked = torch.where(valid, logits, torch.full_like(logits, float("-inf")))
    take = min(k, width)
    vals, idx = masked.topk(take, dim=-1, sorted=False)
    idx = torch.where(
        vals != float("-inf"),
        idx.to(torch.int32),
        torch.full_like(idx, -1, dtype=torch.int32),
    )
    indices[:, :take] = idx
    if take < k:
        indices[:, take:] = -1
    return indices


def _get_baseline_op():

    if _native_persistent_topk() is not None:
        return _vllm_native_op

    return _torch_persistent_topk


class PersistentTopKBenchmark(base.GenericBenchmarkExcluse1D):

    DEFAULT_SHAPE_DESC = "num_rows, seq_len, max_seq_len, stride"

    def init_user_config(self):
        super().init_user_config()
        if self.torch_op is None:

            self.to_bench_metrics = [
                metric
                for metric in self.to_bench_metrics
                if metric not in ("latency_base", "speedup")
            ]

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path

        self.shapes = list(_HOMO_SHAPES)


@pytest.mark.persistent_topk
def test_persistent_topk():
    baseline_op = _get_baseline_op()

    bench = PersistentTopKBenchmark(
        input_fn=_input_fn,
        op_name="persistent_topk",
        torch_op=baseline_op,
        gems_op=flaggems_vllm.persistent_topk,
        dtypes=_get_supported_dtypes(),
    )

    bench.run()
