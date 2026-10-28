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

"""Benchmark FlagGems-vllm persistent_topk against the vLLM native kernel.

Only NVIDIA and MetaX vLLM builds ship a native persistent_topk. On the other
vendors the Gems kernel is timed on its own: the Torch and Gems speedup columns
stay N/A and only the vs-baseline ratio from --base-data is reported. There is
deliberately no torch.topk reference: a torch-built baseline is not a
meaningful performance reference.

Every shape is (num_rows, seq_len, max_seq_len, stride): seq_len is the length
the kernel scans, max_seq_len is the value it receives, stride is the width of
the logits buffer.
"""

import pytest
import torch

import flaggems_vllm

from . import base

VENDOR = flaggems_vllm.vendor_name

# NVIDIA and MetaX expose persistent_topk through vLLM custom ops.
VLLM_NATIVE_VENDORS = {
    "nvidia",
    "metax",
}


# -----------------------------------------------------------------------------
# Vendor-specific baseline import
# -----------------------------------------------------------------------------

if VENDOR in VLLM_NATIVE_VENDORS:
    # Registers the torch.ops._C::persistent_topk schema.
    import vllm._custom_ops  # noqa: F401

if VENDOR == "metax":
    # MetaX serves the kernel from mcoplib rather than from vLLM's own
    # compiled extension.
    try:
        import mcoplib._C  # noqa: F401
    except ImportError:
        pass


# -----------------------------------------------------------------------------
# Shape grid
# -----------------------------------------------------------------------------

K = 512
# The recorded vLLM baseline allocates a 1 MB scratch buffer.
WORKSPACE_BYTES = 1 << 20

_HOMO_SHAPES = [
    # Dispatch note. vLLM's auto policy prefers cooperative_topk for every
    # batch with num_rows <= AUTO_COOPERATIVE_MAX_ROWS (64) and only sends
    # larger batches to persistent_topk (see indexer_topk.py). cooperative_topk
    # is a newer kernel: before it existed all of these rows ran on
    # persistent_topk, and FlagGems-vllm has no cooperative_topk counterpart.
    # Whether one will be written is undecided, so the small-batch rows are
    # kept - they are the workload this operator owns whenever cooperative_topk
    # is unavailable, rather than a pinned-backend corner case.
    # DeepSeek-V4.1-Flash, 1k context: the closest thing to production we have,
    # taken from a p1024d1024 run (prompt 1024, concurrency 512, 3 rounds)
    # filtered by the serving-start sentinel - 11,136 real calls over
    # num_rows 67-508. seq_len is the p50 of each ratio class (1044 for ratio
    # 1, 522 for ratio 2); stride == max_seq_len because compress_ratio scales
    # both.
    (67, 1044, 65536, 65536),
    (128, 1044, 65536, 65536),
    (256, 1044, 65536, 65536),
    (508, 1044, 65536, 65536),
    (67, 522, 32768, 32768),
    (128, 522, 32768, 32768),
    (256, 522, 32768, 32768),
    (508, 522, 32768, 32768),
    # DeepSeek-V4.1-Flash, 4k context, one batch pair per ratio class.
    (16, 4096, 65536, 65536),
    (125, 4096, 65536, 65536),
    (48, 2048, 32768, 32768),
    (120, 2048, 32768, 32768),
    # DeepSeek-V4.1-Flash, 32k context, one row per ratio class.
    (3, 32769, 65536, 65536),
    (5, 16384, 32768, 32768),
    # DeepSeek-V4-Flash keeps a 262144-wide buffer and passes the real context;
    # 4k dominates its traffic (78.8%), then ~1k (15.1%) and 32k (6.1%).
    (12, 4105, 4105, 262144),
    (24, 4108, 4108, 262144),
    (10, 1055, 1055, 262144),
    (1, 32773, 32773, 262144),
]


def _get_supported_dtypes():
    """Return the dtypes benchmarked for persistent_topk."""
    return [torch.float32]


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------


def _input_fn(shape, dtype, device):
    """Build the inputs for one shape.

    shape is (num_rows, seq_len, max_seq_len, stride): every row has the same
    seq_len, and only the first seq_len entries of each logits row are finite.
    """
    num_rows, seq_len, max_seq_len, stride = shape

    logits = torch.full((num_rows, stride), float("-inf"), dtype=dtype, device=device)
    logits[:, :seq_len] = torch.randn(num_rows, seq_len, dtype=dtype, device=device)
    lengths = torch.full((num_rows,), seq_len, dtype=torch.int32, device=device)

    indices = torch.empty((num_rows, K), dtype=torch.int32, device=device)
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)

    yield logits, lengths, indices, workspace, K, max_seq_len


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------


def _vllm_native_op(logits, lengths, indices, workspace, k, max_seq_len):
    """vLLM native persistent_topk baseline.

    Used by NVIDIA and MetaX. Writing into indices is the operator's output
    contract, and every call overwrites the whole buffer, so the inputs do not
    need cloning.
    """
    torch.ops._C.persistent_topk(logits, lengths, indices, workspace, k, max_seq_len)

    return indices


def _get_baseline_op():
    """Select the baseline implementation for the current vendor.

    Unlike fused_add_rms_norm there is no PyTorch reference to fall back to: a
    torch-built baseline is not a meaningful performance reference. Vendors
    without a native implementation get None and the framework decides how to
    handle the missing baseline.
    """
    if VENDOR in VLLM_NATIVE_VENDORS:
        return _vllm_native_op

    return None


# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------


class PersistentTopKBenchmark(base.GenericBenchmarkExcluse1D):
    """Benchmark FlagGems-vllm persistent_topk.

    NVIDIA:
        vLLM native vs FlagGems-vllm

    MetaX:
        mcoplib native vs FlagGems-vllm

    Other vendors:
        gems kernel only; baseline columns stay N/A
    """

    DEFAULT_SHAPE_DESC = "num_rows, seq_len, max_seq_len, stride"

    def init_user_config(self):
        super().init_user_config()
        if self.torch_op is None:
            # No vendor-native kernel on this platform: timing it would raise,
            # and a gems-vs-gems ratio says nothing. Keep the latency metric and
            # drop the two baseline ones; compared_speedup is computed from the
            # frozen data outside this list.
            self.to_bench_metrics = [
                metric
                for metric in self.to_bench_metrics
                if metric not in ("latency_base", "speedup")
            ]

    def set_shapes(self, shape_file_path=None):
        """Shape list, kept in sync with baseline/persistent_topk.py::grid()."""
        _ = shape_file_path

        self.shapes = list(_HOMO_SHAPES)


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------


@pytest.mark.persistent_topk
def test_persistent_topk():
    baseline_op = _get_baseline_op()

    bench = PersistentTopKBenchmark(
        input_fn=_input_fn,
        op_name="persistent_topk",
        # GenericBenchmark calls this field torch_op, but here it represents
        # the vendor-native baseline implementation.
        torch_op=baseline_op,
        # Top-level FlagGems API selects the current vendor backend.
        gems_op=flaggems_vllm.persistent_topk,
        dtypes=_get_supported_dtypes(),
    )

    bench.run()
