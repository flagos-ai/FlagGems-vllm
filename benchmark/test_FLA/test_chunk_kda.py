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
import os
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
        try:
            from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
        except ImportError:
            l2norm_fwd = None

    # The AscendC KDA prefill kernel: this is the operator vllm-ascend itself
    # calls (``vllm_ascend.ops.kda.run_chunk_kda`` -> ``fla_npu.ops.ascendc``),
    # so it is the true native baseline. Importing it is not enough -- the
    # aclnn entry behind it only exists on A5-class parts -- so availability is
    # probed with one live 64-token call (see _fla_npu_kda_available).
    try:
        from fla_npu.ops.ascendc import chunk_kda_fwd as fla_npu_chunk_kda
    except ImportError as exc:
        fla_npu_chunk_kda = None
        fla_npu_chunk_kda_error = f"import failed: {exc!r}"
    else:
        fla_npu_chunk_kda_error = None

    # vllm-ascend's Triton KDA prefill entry: its software fallback, used when
    # the AscendC kernel is unavailable.
    try:
        from vllm_ascend.ops.triton.kda.kda import (
            chunk_kda_with_fused_gate as chunk_kda_vllm_ascend,
        )
    except ImportError:
        chunk_kda_vllm_ascend = None

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


def _chunk_kda_fla_npu_inference_op(*args, **kwargs):
    """AscendC chunk_kda baseline (the operator vllm-ascend ships in production).

    ``chunk_kda_fwd`` applies the bounded gate in-kernel from the raw gate plus
    ``A_log``/``dt_bias``. Its in-kernel L2-norm/beta-sigmoid variant
    (``aclnnChunkKdaFwdV2``) is not available here, so -- exactly like
    vllm-ascend's own ``run_chunk_kda`` -- q/k normalisation and beta's sigmoid
    are done at this layer and only the gate is fused. ``dt_bias`` must be
    flattened to ``[H*K]`` fp32; a 2-D tensor is rejected with 161002.
    """
    q, k, v, g, beta = args
    cu_seqlens = kwargs.get("cu_seqlens")
    with torch.inference_mode():
        out = fla_npu_chunk_kda(
            torch.nn.functional.normalize(q.float(), p=2.0, dim=-1, eps=1e-6).to(
                q.dtype
            ),
            torch.nn.functional.normalize(k.float(), p=2.0, dim=-1, eps=1e-6).to(
                k.dtype
            ),
            v,
            g,
            beta.float().sigmoid(),
            kwargs["scale"],
            CHUNK_SIZE,
            layout="BSND",
            initial_state=kwargs["initial_state"],
            output_final_state=kwargs.get("output_final_state", True),
            cu_seqlens=None if cu_seqlens is None else cu_seqlens.tolist(),
            disable_recompute=True,
            return_intermediate_states=True,
            state_v_first=True,
            safe_gate=True,
            lower_bound=kwargs.get("lower_bound", LOWER_BOUND),
            use_gate_in_kernel=True,
            A_log=kwargs["A_log"].reshape(-1).contiguous(),
            # The aclnn entry requires the gate bias flattened to [H*K] fp32;
            # a 2-D [H, K] tensor is rejected with 161002 (PARAM_INVALID).
            dt_bias=kwargs["dt_bias"].reshape(-1).contiguous(),
        )
    return out[0], out[1]


_fla_npu_kda_probe = None


def _fla_npu_kda_available():
    """Whether the AscendC chunk_kda kernel actually executes on this device.

    The import succeeds even where the aclnn kernel is absent, so the check is
    one live call on a minimal K=V=128 shape; the aclnn entry rejects the launch
    immediately with "GetWorkspaceSize failed: 161002" on parts without it.
    """
    global _fla_npu_kda_probe
    if _fla_npu_kda_probe is None:
        if fla_npu_chunk_kda is None:
            _fla_npu_kda_probe = (False, fla_npu_chunk_kda_error or "not importable")
        else:
            try:
                torch.manual_seed(0)
                dev = flaggems_vllm.device
                t, h, d = CHUNK_SIZE, 1, 128
                q = torch.randn(1, t, h, d, device=dev, dtype=torch.bfloat16)
                init = torch.zeros(1, h, d, d, device=dev, dtype=torch.float32)
                fla_npu_chunk_kda(
                    q,
                    q.clone(),
                    q.clone(),
                    q.clone(),
                    torch.sigmoid(torch.randn(1, t, h, device=dev)),
                    1.0 / math.sqrt(d),
                    CHUNK_SIZE,
                    layout="BSND",
                    initial_state=init,
                    output_final_state=True,
                    disable_recompute=True,
                    return_intermediate_states=True,
                    safe_gate=True,
                    lower_bound=LOWER_BOUND,
                    use_gate_in_kernel=True,
                    A_log=torch.rand(h, device=dev, dtype=torch.float32),
                    dt_bias=torch.randn(h * d, device=dev, dtype=torch.float32),
                    state_v_first=True,
                )
                torch.npu.synchronize()
                _fla_npu_kda_probe = (True, None)
            except Exception as exc:
                _fla_npu_kda_probe = (False, f"{type(exc).__name__}: {exc}")
    return _fla_npu_kda_probe


def _chunk_kda_vllm_ascend_inference_op(*args, **kwargs):
    """vllm-ascend Triton KDA prefill baseline.

    ``chunk_kda_with_fused_gate`` takes the raw gate plus
    ``A_log``/``dt_bias``/``lower_bound`` and does the q/k L2 norm in-kernel;
    beta is sigmoided by the caller, matching the AscendC baseline above.
    """
    q, k, v, g, beta = args
    with torch.inference_mode():
        return chunk_kda_vllm_ascend(
            q,
            k,
            v,
            g,
            beta.float().sigmoid(),
            kwargs["A_log"].reshape(-1).contiguous(),
            kwargs["dt_bias"].contiguous(),
            scale=kwargs["scale"],
            initial_state=kwargs["initial_state"],
            output_final_state=kwargs.get("output_final_state", True),
            use_qk_l2norm_in_kernel=kwargs.get("use_qk_l2norm_in_kernel", True),
            cu_seqlens=kwargs.get("cu_seqlens"),
            safe_gate=True,
            lower_bound=kwargs.get("lower_bound", LOWER_BOUND),
        )


def _ascendc_available():
    """The AscendC KDA custom ops are registered and usable."""
    return (
        custom_ops_enabled
        and callable(l2norm_fwd)
        and hasattr(torch.ops._C_ascend, "kda_gate_cumsum")
        and hasattr(torch.ops._C_ascend, "chunk_kda_fwd")
    )


# The Triton KDA entry is a correctness-equivalent native baseline, but on
# Ascend 910B it cannot run the benchmark's core shapes: T=8192 / H=96 / D=128
# pushes a 32768-block grid through its chunk_gla_fwd_o stage, which trips the
# device watchdog ("aicore execution times out", ~2 s per launch) while this
# repo's kernel finishes the same workload in ~39 ms. It is therefore opt-in,
# so the benchmark skips with a precise reason instead of dying on a timeout.
_VLLM_ASCEND_KDA_ENV = "FLAGGEMS_BENCH_VLLM_ASCEND_KDA"


def _vllm_ascend_kda_opt_in():
    value = os.environ.get(_VLLM_ASCEND_KDA_ENV, "")
    return value.strip().lower() in ("1", "true", "on", "yes")


def _get_baseline_op():
    if VENDOR == "nvidia":
        return _chunk_kda_vllm_inference_op

    if VENDOR == "ascend":
        if _ascendc_available():
            return _chunk_kda_ascendc_inference_op
        if _fla_npu_kda_available()[0]:
            return _chunk_kda_fla_npu_inference_op
        if chunk_kda_vllm_ascend is not None and _vllm_ascend_kda_opt_in():
            return _chunk_kda_vllm_ascend_inference_op
        return None

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
        if baseline_op is None:
            # ChunkKDABenchmark drops the baseline metrics when torch_op is
            # None, so still measure the FlagGems latency and say why there is
            # no SpeedUp column.
            _, fla_npu_reason = _fla_npu_kda_available()
            print(f"[chunk_kda] AscendC chunk_kda_fwd unusable: {fla_npu_reason}")
            if chunk_kda_vllm_ascend is not None:
                print(
                    "[chunk_kda] vllm-ascend's Triton chunk_kda fallback is "
                    "opt-in because it trips the 910B device watchdog on these "
                    f"shapes ({_VLLM_ASCEND_KDA_ENV}=1 to time it). "
                    "Reporting FlagGems latency only."
                )
            else:
                print(
                    "[chunk_kda] no vllm-ascend chunk_kda baseline available; "
                    "reporting FlagGems latency only."
                )
        chosen = {
            _chunk_kda_ascendc_inference_op: "torch.ops._C_ascend KDA ops",
            _chunk_kda_fla_npu_inference_op: "fla_npu AscendC chunk_kda_fwd",
            _chunk_kda_vllm_ascend_inference_op: "vllm-ascend Triton chunk_kda",
        }.get(baseline_op, "unknown")
        print(f"[chunk_kda] baseline: {chosen}")

    bench = ChunkKDABenchmark(
        op_name="chunk_kda",
        torch_op=baseline_op,
    )
    bench.set_gems(_chunk_kda_inference_op)
    bench.run()
