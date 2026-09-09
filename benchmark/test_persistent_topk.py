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

import pytest
import torch

import flaggems_vllm

from . import base

# Bind the top-level entry: on vendor backends with a specialized
# implementation (e.g. _hygon/fused), flaggems_vllm.persistent_topk is the
# vendor version (replaced at import time); on NVIDIA it is the generic one.
persistent_topk = flaggems_vllm.persistent_topk

device = flaggems_vllm.device

# The vLLM native op is used as the baseline where available (NVIDIA); on
# platforms without torch.ops._C.persistent_topk (e.g. Hygon vllm-hcu) the
# benchmark falls back to a torch.topk reference below.
HAS_VLLM = False
try:
    import vllm._custom_ops  # noqa: F401

    # `vllm._custom_ops` may import cleanly even when the compiled C++ extension
    # is missing, in which case the native op raises NotImplementedError at
    # dispatch time. Probe one tiny call so the benchmark falls back (rather
    # than errors) when the kernel is not actually available.
    _probe_logits = torch.zeros(1, 4102, dtype=torch.float32, device="cuda")
    _probe_lengths = torch.tensor([4102], dtype=torch.int32, device="cuda")
    torch.ops._C.persistent_topk(
        _probe_logits,
        _probe_lengths,
        torch.empty((1, 512), dtype=torch.int32, device="cuda"),
        torch.empty(2 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        512,
        4102,
    )
    HAS_VLLM = True
except (ImportError, AttributeError, NotImplementedError, RuntimeError):
    pass

STRIDE = 262144
K = 512

# CUDA-Graph-safe lengths (torch.topk fallback only): the reference needs
# host-side seq_lens to decide batched-vs-padded and to slice logits. Reading
# the device tensor back (seq_lens.tolist()) inside CUDA Graph capture is
# illegal on HIP, and even a warmed cache does not help: torch.topk's own
# graph capture (small slices first, then large) leaves its private memory
# pool reused by a later shape's seq_lens tensor, whose content is then
# observed as 0/garbage on the first D2H read -> baseline collapses to
# ~0.0018ms. Fix: precompute the host list once per shape in get_input_iter,
# keyed by tensor id; _host_lengths only looks up the dict and never reads
# the device tensor in the measured path.
_HLEN_CACHE: dict = {}


def _host_lengths(seq_lens):
    return _HLEN_CACHE[id(seq_lens)]


def _baseline_persistent_topk(
    logits, lengths, indices, workspace, max_seq_len, seq_lens
):
    """torch.topk baseline: vLLM native op where available, else a torch.topk
    reference.

    Matches persistent_topk semantics: top-k column indices in arbitrary
    order (sorted=False), -1 padding when seq_len < k.

    The torch.topk fallback is batched when all seq_lens are equal, else a
    single batched topk over the -inf-padded rows (the heterogeneous rows are
    -inf-padded to the full stride by get_input_iter, so one batched call
    returns the per-row top-min(K, seq_len) real entries, and pad slots are
    masked back to -1 via the returned values). A per-row baseline loop is
    avoided: its sequential launches inflated the measured latency and made
    the SpeedUp comparison unreliable.
    """
    if HAS_VLLM:
        torch.ops._C.persistent_topk(
            logits, lengths, indices, workspace, K, max_seq_len
        )
        return indices
    lens = _host_lengths(seq_lens)
    first = lens[0]
    if all(x == first for x in lens):
        k = min(K, first)
        if k > 0:
            _, idx = logits[:, :first].topk(k, dim=-1, sorted=False)
            indices[:, :k] = idx
        indices[:, k:] = -1
        return indices
    vals, idx = logits.topk(K, dim=-1, sorted=False)
    real = vals > -1e30
    indices[:, :] = torch.where(real, idx, -1)
    return indices


def _gems_decode(logits, lengths, indices, workspace, max_seq_len, seq_lens):
    persistent_topk(logits, lengths, indices, workspace, K, max_seq_len=max_seq_len)
    return indices


class PersistentTopKBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_rows, seq_len, max_seq_len"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            # Medium path shapes (commented out: not real-world scenarios)
            # (1, 10000, 10000),
            # (1, 15000, 15000),
            # (1, 20000, 20000),
            # (1, 32768, 32768),
            # (4, 10000, 10000),
            # (8, 10000, 10000),
            # (12, 20000, 20000),
            # (20, 32768, 32768),
            # Decode path shapes
            (1, 4102, 4102),
            (4, 4102, 4102),
            (10, 1055, 1055),
            (12, 4105, 4105),
            (20, 4105, 4105),
            (28, 4109, 4109),
            (1, 8192, 8192),
            # Large path lower bound
            (1, 32773, 32773),
            # Large path: num_rows=1..32, seq_len=262144
            *[(nr, 262144, 1048576) for nr in range(1, 33)],
            (496, 262144, 1048576),
            (512, 262144, 1048576),
        ]
        self.hetero_shapes = [
            (496, 1048576),
            (496, 1),
        ]

    def get_input_iter(self, dtype):
        for num_rows, seq_len, max_seq_len in self.shapes:
            torch.manual_seed(torch.randint(0, 2**31, (1,)).item())
            logits = torch.full(
                (num_rows, STRIDE),
                float("-inf"),
                dtype=torch.float32,
                device=self.device,
            )
            logits[:, :seq_len] = torch.randn(num_rows, seq_len, device=self.device)

            lengths = torch.full(
                (num_rows,), seq_len, dtype=torch.int32, device=self.device
            )
            indices = torch.empty((num_rows, K), dtype=torch.int32, device=self.device)
            # vLLM native op uses its own layout (1MB suffices); the torch.topk
            # fallback runs the vendor kernels, which need 2MB headroom.
            workspace = torch.empty(
                (1 if HAS_VLLM else 2) * 1024 * 1024,
                dtype=torch.uint8,
                device=self.device,
            )
            seq_lens = torch.full(
                (num_rows,), seq_len, dtype=torch.int32, device=self.device
            )
            # precompute host lengths once per shape; _host_lengths only looks
            # this up (never reads the device tensor in the measured path)
            _HLEN_CACHE[id(seq_lens)] = [seq_len] * num_rows

            yield logits, lengths, indices, workspace, max_seq_len, seq_lens

        # Heterogeneous batches (FlagOSTune production shapes)
        for num_rows, max_len in self.hetero_shapes:
            torch.manual_seed(torch.randint(0, 2**31, (1,)).item())
            lengths_values = torch.linspace(
                1, min(max_len, STRIDE), num_rows, dtype=torch.int32, device=self.device
            )
            logits = torch.full(
                (num_rows, STRIDE),
                float("-inf"),
                dtype=torch.float32,
                device=self.device,
            )
            for i, sl in enumerate(lengths_values):
                logits[i, :sl] = torch.randn(sl, device=self.device)

            lengths = lengths_values
            indices = torch.empty((num_rows, K), dtype=torch.int32, device=self.device)
            workspace = torch.empty(
                (1 if HAS_VLLM else 2) * 1024 * 1024,
                dtype=torch.uint8,
                device=self.device,
            )
            seq_lens = lengths_values
            _HLEN_CACHE[id(seq_lens)] = lengths_values.tolist()

            yield logits, lengths, indices, workspace, max_len, seq_lens


@pytest.mark.persistent_topk
def test_persistent_topk():
    bench = PersistentTopKBenchmark(
        op_name="persistent_topk",
        torch_op=_baseline_persistent_topk,
        gems_op=_gems_decode,
        dtypes=[torch.float32],
    )
    bench.run()
