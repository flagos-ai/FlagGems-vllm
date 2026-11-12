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

from . import base, consts

VENDOR = flaggems_vllm.vendor_name

MODEL_TOP_K_HIDDEN = [
    (6, 4096),
    (8, 6144),
    (4, 6144),
    (16, 7168),
]

NUM_TOKENS = [512, 2048, 8192, 16384]


# Benchmark shapes: (num_tokens, topk, hidden_size).
MOE_SUM_SHAPES = [
    (num_tokens, topk, hidden_size)
    for topk, hidden_size in MODEL_TOP_K_HIDDEN
    for num_tokens in NUM_TOKENS
]


# -----------------------------------------------------------------------------
# Vendor-specific baseline import
# -----------------------------------------------------------------------------

# A platform build may ship FlagGems without the vendor-native vocabulary
# package (e.g. lightop on Hygon). A failed import leaves the handle None and
# degrades the run to latency-only mode instead of failing collection.
vendor_ops = None
torch_npu = None
hygon_op = None
iluvatar_moe_reduce_sum = None
maybe_fast_moe_sum = None
_BASELINE_IMPORT_ERROR = None

if VENDOR == "nvidia":
    try:
        from vllm import _custom_ops as vendor_ops
    except ImportError as exc:
        _BASELINE_IMPORT_ERROR = exc

elif VENDOR == "ascend":
    try:
        import torch_npu
    except ImportError as exc:
        _BASELINE_IMPORT_ERROR = exc

elif VENDOR == "hygon":
    try:
        from lightop import op as hygon_op
    except ImportError as exc:
        _BASELINE_IMPORT_ERROR = exc

elif VENDOR == "iluvatar":
    try:
        from ixformer.inference.functions.moe import (
            moe_output_reduce_sum as iluvatar_moe_reduce_sum,
        )
    except ImportError as exc:
        _BASELINE_IMPORT_ERROR = exc

elif VENDOR == "mthreads":
    try:
        from vllm_musa.jit_kernel.csrc.moe import maybe_fast_moe_sum
    except ImportError as exc:
        _BASELINE_IMPORT_ERROR = exc


# -----------------------------------------------------------------------------
# Vendor capabilities
# -----------------------------------------------------------------------------


def _normalized_shape(shape):
    """Normalize a benchmark shape to (num_tokens, topk, hidden_size).

    Two-dimensional shapes use topk=1, matching _input_fn.
    """
    return (shape[0], 1, shape[1]) if len(shape) == 2 else shape


def _mthreads_shape_supported(shape):
    """Return whether vLLM-MUSA musa_fast_moe_sum supports the shape."""
    _, topk, hidden_size = _normalized_shape(shape)

    # The native fast path is a statically-unrolled topk reduction and was
    # validated on real MoE shapes (topk <= 16). Hidden rows use vectorized
    # loads, so keep them aligned to 8 elements.
    return topk <= 16 and hidden_size > 0 and hidden_size % 8 == 0


def _iluvatar_shape_supported(shape):
    """Return whether Iluvatar moe_output_reduce_sum supports the shape."""
    _, topk, hidden_size = _normalized_shape(shape)

    # The Iluvatar native kernel was validated on MoE shapes with topk <= 16
    # and an even hidden size.
    return topk <= 16 and hidden_size > 0 and hidden_size % 2 == 0


def _ascend_shape_supported(shape):
    """Return whether the Ascend MoeTokenUnpermute supports the shape."""
    _, topk, _ = _normalized_shape(shape)

    # torch_npu.npu_moe_token_unpermute rejects topK > 512.
    return topk <= 512


def _get_supported_dtypes():
    """Return dtypes supported by both FlagGems and the selected baseline."""
    if VENDOR == "iluvatar":
        # Iluvatar moe_output_reduce_sum supports FP16/BF16 only.
        return [
            dtype
            for dtype in consts.FLOAT_DTYPES
            if dtype in (torch.float16, torch.bfloat16)
        ]

    # Ascend, Hygon, MThreads, NVIDIA and T-Head native kernels support the
    # normal benchmark dtype set.
    return consts.FLOAT_DTYPES


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------


def _input_fn(shape, dtype, device):
    shape = _normalized_shape(shape)
    num_tokens, topk, hidden_size = shape
    input_tensor = torch.randn(
        num_tokens,
        topk,
        hidden_size,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )

    output_tensor = torch.empty(
        num_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )
    yield input_tensor, output_tensor


# -----------------------------------------------------------------------------
# Native argument caches
# -----------------------------------------------------------------------------

# Some native kernels need extra tensors (identity indices, ones/zeros) that
# are not part of the benchmark input. Build them once per input buffer and
# reuse them so their allocation does not land in the timed region.
_AUX_CACHE = {}


def _cache_key(input_tensor):
    return (
        input_tensor.data_ptr(),
        tuple(input_tensor.shape),
        input_tensor.dtype,
        input_tensor.device,
    )


def _cached_aux(input_tensor, factory):
    key = _cache_key(input_tensor)
    aux = _AUX_CACHE.get(key)
    if aux is None:
        aux = factory()
        _AUX_CACHE[key] = aux
    return aux


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------


def _vllm_native_op(input_tensor, output_tensor):
    """vLLM native moe_sum baseline (NVIDIA and T-Head)."""
    vendor_ops.moe_sum(input_tensor, output_tensor)


def _hygon_native_op(input_tensor, output_tensor):
    """Hygon lightop native moe_sum baseline.

    With bias / expert_mask / num_local_tokens set to None and factor=1.0,
    lightop reduces to out[t, h] = sum_k input[t, k, h].
    """
    hygon_op.moe_sum(
        input=input_tensor,
        output=output_tensor,
        bias=None,
        expert_mask=None,
        num_local_tokens=None,
        factor=1.0,
    )


def _ascend_native_op(input_tensor, output_tensor):
    """Ascend MoeTokenUnpermute baseline used as a plain topk-sum.

    torch_npu.npu_moe_token_unpermute allocates its own output, so the
    pre-allocated output_tensor is intentionally unused.
    """
    del output_tensor

    num_tokens, topk, hidden_size = input_tensor.shape

    def _make_args():
        # Identity indices make the unpermute a no-op; unit probs turn the
        # weighted reduction into the same plain topk-sum as FlagGems.
        sorted_indices = torch.arange(
            num_tokens * topk,
            dtype=torch.int32,
            device=input_tensor.device,
        )
        probs = torch.ones(
            (num_tokens, topk),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        return sorted_indices, probs

    sorted_indices, probs = _cached_aux(input_tensor, _make_args)

    # Do not cache this view: it keeps the (potentially very large) input
    # tensor alive for the lifetime of the process.
    permuted = input_tensor.view(num_tokens * topk, hidden_size)

    torch_npu.npu_moe_token_unpermute(
        permuted,
        sorted_indices,
        probs=probs,
        padded_mode=False,
        restore_shape=None,
    )


def _iluvatar_native_op(input_tensor, output_tensor):
    """Iluvatar ixformer native moe_sum baseline.

    Neutral parameters (unit weights, all-false mask, zero residual, scaling
    1.0) make the native op compute the same plain topk-sum as FlagGems.
    """
    num_tokens, topk, hidden_size = input_tensor.shape

    def _make_args():
        router_weights = torch.ones(
            (num_tokens, topk),
            dtype=torch.float32,
            device=input_tensor.device,
        )
        mask = torch.zeros(
            num_tokens * topk,
            dtype=torch.bool,
            device=input_tensor.device,
        )
        residual = torch.zeros(
            (num_tokens, hidden_size),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        return router_weights, mask, residual

    router_weights, mask, residual = _cached_aux(input_tensor, _make_args)

    iluvatar_moe_reduce_sum(
        input_tensor,
        router_weights,
        output_tensor,
        mask,
        residual,
        1.0,
    )


def _mthreads_native_op(input_tensor, output_tensor):
    """vLLM-MUSA native fast moe_sum baseline.

    The MUSA fast-path guard and the lazy JIT build were already probed on
    these buffers by MoeSumBenchmark.get_input_iter.
    """
    torch.ops.vllm.musa_fast_moe_sum(output_tensor, input_tensor)


def _get_baseline_op():
    """Select the baseline implementation for the current vendor.

    Returns None when the vendor has no native op or when its library is not
    installed in this environment; the caller then runs latency-only.
    """
    if VENDOR == "nvidia":
        return _vllm_native_op if vendor_ops is not None else None

    if VENDOR == "ascend":
        return _ascend_native_op if torch_npu is not None else None

    if VENDOR == "hygon":
        return _hygon_native_op if hygon_op is not None else None

    if VENDOR == "iluvatar":
        return _iluvatar_native_op if iluvatar_moe_reduce_sum is not None else None

    if VENDOR == "mthreads":
        return _mthreads_native_op if maybe_fast_moe_sum is not None else None

    return None


# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------


class MoeSumBenchmark(base.GenericBenchmarkExcluse1D):
    """Benchmark FlagGems-vllm moe_sum against the vendor-native baseline.

    Shapes are (num_tokens, topk, hidden_size) taken from mainstream MoE
    models, with num_tokens in {512, 2048, 8192, 16384, 32768}.

    NVIDIA:
        vLLM native moe_sum

    T-Head:
        vLLM native moe_sum

    Ascend:
        torch_npu MoeTokenUnpermute (identity indices + unit probs)

    Hygon:
        lightop op.moe_sum

    Iluvatar:
        ixformer moe_output_reduce_sum (neutral extra parameters)

    MThreads:
        vLLM-MUSA musa_fast_moe_sum

    Other vendors, or a listed vendor whose native library is not installed:
        no native baseline: only FlagGems latency is measured, and when
        --base-data is given the H800 latency scaled by _scaling_factors is
        reported as the 'vs Base' comparison instead.
    """

    # Metrics measured by running torch_op. Without a vendor-native baseline
    # they cannot be produced, so they are dropped instead of failing the run.
    BASELINE_METRICS = ("latency_base", "speedup")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.torch_op is None:
            self.metrics = [
                metric for metric in self.metrics if metric not in self.BASELINE_METRICS
            ]
            self.to_bench_metrics = self.metrics

    def set_shapes(self, shape_file_path=None):
        # Fixed mainstream-model shape set; ignore the generic shape file.
        _ = shape_file_path
        self.shapes = MOE_SUM_SHAPES

    def init_user_config(self):
        """Apply restrictions imposed by the selected baseline."""
        super().init_user_config()

        if VENDOR == "ascend":
            self.shapes = [
                shape for shape in self.shapes if _ascend_shape_supported(shape)
            ]

        elif VENDOR == "mthreads":
            self.shapes = [
                shape for shape in self.shapes if _mthreads_shape_supported(shape)
            ]

        elif VENDOR == "iluvatar":
            self.shapes = [
                shape for shape in self.shapes if _iluvatar_shape_supported(shape)
            ]

    def get_input_iter(self, dtype):
        # Also take the generic path when the MUSA library is missing: there
        # is no fast-path guard to probe and no native baseline to compare.
        if VENDOR != "mthreads" or maybe_fast_moe_sum is None:
            yield from super().get_input_iter(dtype)
            return

        # Probe the MUSA fast-path guard on the real buffers and skip shapes
        # it rejects instead of failing the whole benchmark. The probe also
        # warms up the JIT kernel, so the timed op can call the raw op.
        for shape in self.shapes:
            for args in self.input_fn(shape, dtype, self.device):
                input_tensor, output_tensor = args
                if not maybe_fast_moe_sum(input_tensor, output_tensor):
                    continue

                yield args


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------


@pytest.mark.moe_sum
def test_moe_sum():
    baseline_op = _get_baseline_op()

    if baseline_op is None:
        # Second comparison mode: this vendor has no native moe_sum on the
        # machine, either because the platform has no native op or because
        # its library is not installed. FlagGems latency is still measured,
        # and passing --base-data=<h800_baseline.json> additionally reports
        # the H800 latency scaled by the bandwidth ratio as 'compared_speedup'.
        if _BASELINE_IMPORT_ERROR is not None:
            reason = (
                f"native baseline library for vendor {VENDOR!r} is not "
                f"installed ({_BASELINE_IMPORT_ERROR})"
            )
        else:
            reason = f"vendor {VENDOR!r} has no native moe_sum baseline"
        print(
            f"[moe_sum] {reason}; measuring FlagGems latency only. Pass "
            "--base-data to also report the scaled H800 comparison ('vs Base')."
        )

    elif VENDOR == "nvidia":
        assert hasattr(
            torch.ops._moe_C,
            "moe_sum",
        ), "vLLM native moe_sum is not available"

    elif VENDOR == "ascend":
        assert hasattr(
            torch_npu,
            "npu_moe_token_unpermute",
        ), "torch_npu.npu_moe_token_unpermute is not available"

    elif VENDOR == "hygon":
        assert hasattr(
            hygon_op,
            "moe_sum",
        ), "lightop op.moe_sum is not available"

    elif VENDOR == "iluvatar":
        assert callable(
            iluvatar_moe_reduce_sum,
        ), "Iluvatar moe_output_reduce_sum is not available"

    elif VENDOR == "mthreads":
        assert hasattr(
            torch.ops.vllm,
            "musa_fast_moe_sum",
        ), "vLLM-MUSA musa_fast_moe_sum is not available"

    bench = MoeSumBenchmark(
        input_fn=_input_fn,
        op_name="moe_sum",
        # GenericBenchmark calls this field torch_op, but here it is always
        # the vendor-native baseline; there is no PyTorch fallback.
        torch_op=baseline_op,
        # Top-level FlagGems API; vendors that register a specialized
        # moe_sum backend are dispatched automatically.
        gems_op=flaggems_vllm.moe_sum,
        dtypes=_get_supported_dtypes(),
    )

    bench.run()
