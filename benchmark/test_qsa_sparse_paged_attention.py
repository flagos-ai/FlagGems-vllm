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

"""QSA performance against a vendor-native op or collected NVIDIA baseline."""

import pytest
import torch

import flaggems_vllm
from baseline import qsa_sparse_paged_attention as qsa_baseline

from . import base, consts

VENDOR = flaggems_vllm.vendor_name

pytestmark = pytest.mark.skipif(
    VENDOR == "thead",
    reason="T-Head retains the legacy QSA interface benchmarked in test_qsa_attention.py",
)

vendor_qsa = None
if VENDOR == "nvidia":
    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        qsa_sparse_paged_attention as vendor_qsa,
    )


def _get_baseline_op():
    """Select the vendor-native baseline when one is configured."""
    return vendor_qsa


class QSASparsePagedAttentionBenchmark(base.Benchmark):
    def __init__(self, reference):
        super().__init__(
            qsa_baseline.OP_NAME,
            reference,
            qsa_baseline.DTYPES,
            gems_op=flaggems_vllm.qsa_sparse_paged_attention,
        )

    def init_user_config(self):
        super().init_user_config()
        if base.Config.mode == consts.BenchMode.CUDAGRAPH:
            raise ValueError(
                "QSA baseline comparison uses non-graph timing; use --mode kernel"
            )

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            args, kwargs = qsa_baseline.build_inputs(
                {"shape": shape}, dtype, flaggems_vllm.device
            )
            yield (*args, kwargs)


@pytest.mark.qsa_sparse_paged_attention
@torch.inference_mode()
def test_qsa_sparse_paged_attention_benchmark():
    baseline_op = _get_baseline_op()
    if VENDOR == "nvidia":
        assert callable(baseline_op), "vLLM native QSA attention is not available"
    QSASparsePagedAttentionBenchmark(baseline_op).run()
