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

import inspect
import itertools
import math
from importlib import import_module

import pytest
import torch
import torch.nn.functional as F

import flaggems_vllm
from benchmark.base import Benchmark

try:
    chunk_kda = import_module(
        f"flaggems_vllm.runtime.backend._{flaggems_vllm.vendor_name}.FLA"
    ).chunk_kda
except ModuleNotFoundError:
    from flaggems_vllm.ops.FLA import chunk_kda

VENDOR = flaggems_vllm.vendor_name

chunk_kda_vllm_fused_gate = None
supports_lower_bound = False
l2norm_fwd = None

# chunk_kda moved across vLLM releases; probe each known location in turn.
_VLLM_KDA_MODULES = [
    "vllm.third_party.flash_linear_attention.ops.kda",  # 0.26+
    "vllm.model_executor.layers.fla.ops.kda",  # 0.25
]

if VENDOR == "nvidia":
    for _module_name in _VLLM_KDA_MODULES:
        try:
            _kda_module = import_module(_module_name)
        except ImportError:
            continue
        break
    else:
        raise ImportError(
            "No vLLM chunk_kda module found; tried %s" % _VLLM_KDA_MODULES
        )

    # The model layer calls the fused-gate entry, not plain chunk_kda: the
    # latter expects an already-activated gate and an already-sigmoided beta,
    # and silently drops A_log / dt_bias / lower_bound through **kwargs.
    chunk_kda_vllm_fused_gate = getattr(_kda_module, "chunk_kda_with_fused_gate", None)
    if callable(chunk_kda_vllm_fused_gate):
        supports_lower_bound = (
            "lower_bound" in inspect.signature(chunk_kda_vllm_fused_gate).parameters
        )

elif VENDOR == "ascend":
    import torch_npu

    torch_npu.npu.config.allow_internal_format = True

    try:
        from vllm_ascend.utils import enable_custom_op

        custom_ops_enabled = enable_custom_op()
    except ImportError:
        custom_ops_enabled = False

    try:
        from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
    except ImportError:
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

LOWER_BOUND = -5.0
CHUNK_SIZE = 64  # vllm-ascend 0.26.0rc1 _KDA_CHUNK_SIZE; fwd supports 32/64/128
DEFAULT_H = 96
DEFAULT_D = 128
FIXED_CASES = [
    [8192],
]
VARLEN_CASES = [
    [1300, 547, 2048, 963, 271, 3063],
    [1024] * 8,
]
FLASHKDA_CASES = FIXED_CASES + VARLEN_CASES


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------


def _chunk_kda_inference_op(*args, **kwargs):
    with torch.inference_mode():
        return chunk_kda(*args, **kwargs)


def _chunk_kda_vllm_inference_op(*args, **kwargs):
    q, k, v, g, beta = args
    beta_sigmoid = beta.float().sigmoid()
    fused_kwargs = {
        "scale": kwargs["scale"],
        "initial_state": kwargs["initial_state"],
        "output_final_state": kwargs.get("output_final_state", True),
        "use_qk_l2norm_in_kernel": kwargs.get("use_qk_l2norm_in_kernel", True),
        "cu_seqlens": kwargs.get("cu_seqlens"),
    }
    if supports_lower_bound:
        fused_kwargs["lower_bound"] = kwargs.get("lower_bound", LOWER_BOUND)

    with torch.inference_mode():
        return chunk_kda_vllm_fused_gate(
            q,
            k,
            v,
            g,
            beta_sigmoid,
            kwargs["A_log"],
            kwargs["dt_bias"],
            **fused_kwargs,
        )


def _canonical_chunk_indices(cu_seqlens, chunk_size):
    """Build the AscendC chunk_indices: [seq_id, chunk_id] per chunk."""
    return [
        value
        for seq_id, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:]))
        for chunk_id in range((end - start + chunk_size - 1) // chunk_size)
        for value in (seq_id, chunk_id)
    ]


def _chunk_kda_ascendc_inference_op(*args, **kwargs):
    """vllm-ascend AscendC chunk_kda baseline.

    Mirrors vllm_ascend/ops/kimi_kda.py on 0.26.0rc1:
    - kda_gate_cumsum performs gate activation and chunk-local cumsum;
    - q/k l2norm and beta sigmoid happen at the model layer;
    - chunk_kda_fwd returns 12 values; take output and final state.
    """
    q, k, v, g, beta = args
    with torch.inference_mode():
        cu_seqlens = kwargs.get("cu_seqlens")
        cu_seqlens_list = None if cu_seqlens is None else cu_seqlens.tolist()
        chunk_indices = _canonical_chunk_indices(cu_seqlens_list, CHUNK_SIZE)

        q_n = l2norm_fwd(q.contiguous())
        k_n = l2norm_fwd(k.contiguous())
        beta_s = beta.float().sigmoid()
        gate_cumsum = torch.ops._C_ascend.kda_gate_cumsum(
            g.contiguous(),
            CHUNK_SIZE,
            A_log=kwargs["A_log"].reshape(-1).contiguous(),
            dt_bias=kwargs["dt_bias"].contiguous(),
            cu_seqlens=cu_seqlens_list,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=kwargs.get("lower_bound", LOWER_BOUND),
            layout="BSND",
        )
        result = torch.ops._C_ascend.chunk_kda_fwd(
            q_n,
            k_n,
            v.contiguous(),
            gate_cumsum,
            beta_s,
            kwargs["scale"],
            CHUNK_SIZE,
            layout="BSND",
            initial_state=kwargs["initial_state"],
            output_final_state=True,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_indices,
            return_intermediate=False,
        )
        return result[0], result[1]


def _get_baseline_op():
    if VENDOR == "nvidia":
        return _chunk_kda_vllm_inference_op

    if VENDOR == "ascend":
        return _chunk_kda_ascendc_inference_op

    return None


class ChunkKDABenchmark(Benchmark):
    """Benchmark FlagGems-vllm chunk_kda against the vendor-native baseline.

    NVIDIA:
        vLLM native chunk_kda_with_fused_gate

    Ascend:
        vllm-ascend AscendC (kda_gate_cumsum + chunk_kda_fwd)

    Other vendors:
        no native baseline: only FlagGems latency is measured, and when
        --base-data is given the H800 latency scaled by _scaling_factors is
        reported as the 'vs Base' comparison instead.
    """

    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    BASELINE_METRICS = ("latency_base", "speedup")
    DEFAULT_SHAPES = [
        tuple(itertools.chain(seq_lens, [DEFAULT_H, DEFAULT_D]))
        for seq_lens in FLASHKDA_CASES
    ]
    DEFAULT_SHAPE_DESC = "seq_lens..., H, D"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.torch_op is None:
            self.metrics = [
                metric for metric in self.metrics if metric not in self.BASELINE_METRICS
            ]
            self.to_bench_metrics = self.metrics

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            *seq_lens, H, D = shape
            yield self._build_inputs(seq_lens, H, D, cur_dtype)

    def _build_inputs(self, seq_lens, H, D, dtype):
        device = flaggems_vllm.device
        T_total = sum(seq_lens)
        N = len(seq_lens)
        scale = 1.0 / math.sqrt(D)

        q = F.normalize(
            torch.randn((1, T_total, H, D), dtype=torch.float32, device=device),
            p=2.0,
            dim=-1,
        ).to(dtype)
        k = F.normalize(
            torch.randn((1, T_total, H, D), dtype=torch.float32, device=device),
            p=2.0,
            dim=-1,
        ).to(dtype)
        v = torch.randn((1, T_total, H, D), dtype=dtype, device=device)
        g = torch.randn((1, T_total, H, D), dtype=dtype, device=device)
        beta = torch.randn((1, T_total, H), dtype=dtype, device=device)
        A_log = torch.rand(H, dtype=torch.float32, device=device)
        dt_bias = torch.rand(H, D, dtype=torch.float32, device=device)
        initial_state = (
            torch.arange(N * H * D * D, dtype=torch.float32, device=device)
            .reshape(N, H, D, D)
            .to(dtype)
            .float()
        )

        cu_seqlens = None
        if N > 1:
            cu_seqlens = torch.tensor(
                [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
                dtype=torch.long,
                device=device,
            )

        return (
            q,
            k,
            v,
            g,
            beta,
            {
                "scale": scale,
                "initial_state": initial_state,
                "output_final_state": True,
                "use_qk_l2norm_in_kernel": True,
                "use_gate_in_kernel": True,
                "use_beta_sigmoid_in_kernel": True,
                "safe_gate": True,
                "lower_bound": LOWER_BOUND,
                "A_log": A_log,
                "dt_bias": dt_bias,
                "state_v_first": True,
                "cu_seqlens": cu_seqlens,
                "chunk_size": 32 if VENDOR == "mthreads" else 16,
            },
        )


@pytest.mark.chunk_kda
def test_chunk_kda():
    baseline_op = _get_baseline_op()

    if VENDOR == "nvidia":
        assert callable(
            chunk_kda_vllm_fused_gate,
        ), "vLLM native chunk_kda_with_fused_gate is not available"

    elif VENDOR == "ascend":
        if not custom_ops_enabled:
            pytest.skip(
                "vllm_ascend custom ops are not registered, so "
                "torch.ops._C_ascend.kda_gate_cumsum/chunk_kda_fwd cannot be "
                "used (A5 disables custom ops, or vllm_ascend_C was built "
                "without KDA support)."
            )
        assert callable(l2norm_fwd), "vLLM l2norm_fwd is not available"
        assert hasattr(
            torch.ops._C_ascend,
            "kda_gate_cumsum",
        ), "torch.ops._C_ascend.kda_gate_cumsum is not available"
        assert hasattr(
            torch.ops._C_ascend,
            "chunk_kda_fwd",
        ), "torch.ops._C_ascend.chunk_kda_fwd is not available"

    bench = ChunkKDABenchmark(
        op_name="chunk_kda",
        torch_op=baseline_op,
    )
    bench.set_gems(_chunk_kda_inference_op)
    bench.run()
