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
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("vllm")
try:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
        selective_scan_fn,
        selective_state_update,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
except ImportError as exc:  # pragma: no cover - depends on installed vLLM build
    pytest.skip(
        f"Installed vLLM does not expose the Mamba-1 reference ops: {exc}",
        allow_module_level=True,
    )
try:
    import flaggems_vllm

    triton_mamba_mixer = flaggems_vllm.mamba_mixer
except (ImportError, AttributeError) as exc:  # pragma: no cover
    pytest.skip(
        f"Could not import flaggems_vllm.mamba_mixer: {exc}",
        allow_module_level=True,
    )

try:
    import vllm._custom_ops  # noqa: F401 - loads torch.ops._C

    _HAS_VLLM_SELECTIVE_SCAN = hasattr(torch.ops._C, "selective_scan_fwd")
except (ImportError, AttributeError, RuntimeError):
    _HAS_VLLM_SELECTIVE_SCAN = False


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MambaMixer Triton correctness tests require a CUDA/HIP device",
)


@pytest.fixture(autouse=True)
def _disable_reference_tf32():
    """Compare FP32 Triton tf32x3 GEMMs against IEEE FP32 torch F.linear.

    Restore the previous setting so this module does not affect other tests.
    """
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    scale: float = 1.0,
) -> torch.Tensor:
    return (torch.randn(shape, device=device, dtype=torch.float32) * scale).to(dtype)


def _make_weights(
    *,
    hidden_size: int,
    intermediate_size: int,
    state_size: int,
    time_step_rank: int,
    kernel_size: int,
    dtype: torch.dtype,
    device: torch.device,
    use_rms_norm: bool,
) -> dict[str, torch.Tensor | None]:
    """Create one shared set of Mamba-1 weights for both implementations."""
    weights: dict[str, torch.Tensor | None] = {
        "in_proj_weight": _randn(
            (2 * intermediate_size, hidden_size),
            device=device,
            dtype=dtype,
            scale=1.0 / math.sqrt(hidden_size),
        ),
        "in_proj_bias": _randn(
            (2 * intermediate_size,), device=device, dtype=dtype, scale=0.02
        ),
        "conv1d_weight": _randn(
            (intermediate_size, kernel_size),
            device=device,
            dtype=dtype,
            scale=0.15,
        ),
        "conv1d_bias": _randn(
            (intermediate_size,), device=device, dtype=dtype, scale=0.02
        ),
        "x_proj_weight": _randn(
            (time_step_rank + 2 * state_size, intermediate_size),
            device=device,
            dtype=dtype,
            scale=1.0 / math.sqrt(intermediate_size),
        ),
        "dt_proj_weight": _randn(
            (intermediate_size, time_step_rank),
            device=device,
            dtype=dtype,
            scale=1.0 / math.sqrt(time_step_rank),
        ),
        # vLLM's MambaMixer adds this bias inside selective_scan in FP32.
        "dt_proj_bias": _randn(
            (intermediate_size,),
            device=device,
            dtype=torch.float32,
            scale=0.05,
        ),
        # vLLM stores A and D in FP32 for the Mamba-1 SSM path.
        "A": -torch.exp(
            _randn(
                (intermediate_size, state_size),
                device=device,
                dtype=torch.float32,
                scale=0.1,
            )
        ),
        "D": _randn(
            (intermediate_size,),
            device=device,
            dtype=torch.float32,
            scale=0.1,
        ),
        "out_proj_weight": _randn(
            (hidden_size, intermediate_size),
            device=device,
            dtype=dtype,
            scale=1.0 / math.sqrt(intermediate_size),
        ),
        "out_proj_bias": _randn((hidden_size,), device=device, dtype=dtype, scale=0.02),
        "dt_norm_weight": None,
        "b_norm_weight": None,
        "c_norm_weight": None,
    }
    if use_rms_norm:
        weights["dt_norm_weight"] = 1.0 + _randn(
            (time_step_rank,), device=device, dtype=dtype, scale=0.02
        )
        weights["b_norm_weight"] = 1.0 + _randn(
            (state_size,), device=device, dtype=dtype, scale=0.02
        )
        weights["c_norm_weight"] = 1.0 + _randn(
            (state_size,), device=device, dtype=dtype, scale=0.02
        )

    return weights


def _clone_preserve_strides(tensor: torch.Tensor) -> torch.Tensor:
    """Clone a cache tensor without converting a transposed view to contiguous."""
    result = torch.empty_strided(
        tensor.shape,
        tensor.stride(),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    result.copy_(tensor)
    return result


def _make_cache_pair(
    *,
    cache_slots: int,
    intermediate_size: int,
    kernel_size: int,
    state_size: int,
    activation_dtype: torch.dtype,
    device: torch.device,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    """
    Build vLLM's default SD Conv State layout.
    The stored tensor is [slots, K-1, I]. The view passed to MambaMixer is
    [slots, I, K-1], which is intentionally non-contiguous.
    """
    conv_storage = _randn(
        (cache_slots, kernel_size - 1, intermediate_size),
        device=device,
        dtype=activation_dtype,
        scale=0.05,
    )
    conv_initial = conv_storage.transpose(1, 2)
    ssm_initial = _randn(
        (cache_slots, intermediate_size, state_size),
        device=device,
        dtype=torch.float32,
        scale=0.02,
    )
    reference_cache = (
        _clone_preserve_strides(conv_initial),
        _clone_preserve_strides(ssm_initial),
    )
    triton_cache = (
        _clone_preserve_strides(conv_initial),
        _clone_preserve_strides(ssm_initial),
    )
    assert not reference_cache[0].is_contiguous()
    assert not triton_cache[0].is_contiguous()
    return reference_cache, triton_cache


def _rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """FP32 accumulation, matching the vLLM RMSNorm contract."""
    x_fp32 = x.float()
    y = x_fp32 * torch.rsqrt(torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


def _ssm_transform_reference(
    conv_out: torch.Tensor,
    weights: dict[str, torch.Tensor | None],
    *,
    time_step_rank: int,
    state_size: int,
    use_rms_norm: bool,
    rms_norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The projection / optional three RMSNorms / dt projection in MambaMixer."""
    x_proj = F.linear(conv_out, weights["x_proj_weight"], bias=None)
    dt, b, c = torch.split(
        x_proj,
        [time_step_rank, state_size, state_size],
        dim=-1,
    )
    if use_rms_norm:
        dt = _rms_norm_reference(dt, weights["dt_norm_weight"], rms_norm_eps)
        b = _rms_norm_reference(b, weights["b_norm_weight"], rms_norm_eps)
        c = _rms_norm_reference(c, weights["c_norm_weight"], rms_norm_eps)

    dt = F.linear(dt, weights["dt_proj_weight"], bias=None)
    return dt, b, c


def _torch_selective_scan_reference(
    u: torch.Tensor,
    ssm_state: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor | None,
    delta_bias: torch.Tensor | None,
    *,
    query_start_loc: torch.Tensor | None,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    null_block_id: int,
) -> torch.Tensor:
    """Exact FP32 sequential SSM reference when vLLM._C.selective_scan_fwd is absent."""
    intermediate_size, total_tokens = u.shape
    state_size = A.shape[1]
    out = torch.empty_like(u)
    if query_start_loc is None:
        starts = [0]
        ends = [total_tokens]
    else:
        qsl = query_start_loc.tolist()
        starts = qsl[:-1]
        ends = qsl[1:]
    num_seqs = len(starts)
    c_idx = (
        cache_indices.tolist() if cache_indices is not None else list(range(num_seqs))
    )
    h_init = (
        has_initial_state.tolist()
        if has_initial_state is not None
        else [False] * num_seqs
    )
    A_f = A.float()
    D_f = D.float()
    db_f = delta_bias.float() if delta_bias is not None else None
    for s in range(num_seqs):
        s_start, s_end = starts[s], ends[s]
        slot = c_idx[s]
        slot_valid = slot != null_block_id and slot >= 0
        if slot_valid and h_init[s]:
            state = ssm_state[slot].float().clone()
        else:
            state = torch.zeros(
                (intermediate_size, state_size), device=u.device, dtype=torch.float32
            )
        for t in range(s_start, s_end):
            u_t = u[:, t].float()
            dt_t = delta[:, t].float()
            if db_f is not None:
                dt_t = dt_t + db_f
            dt_t = F.softplus(dt_t)
            b_t = B[:, t].float()
            c_t = C[:, t].float()
            d_a = torch.exp(dt_t[:, None] * A_f)
            d_b = dt_t[:, None] * b_t[None, :]
            state = state * d_a + d_b * u_t[:, None]
            y_t = (state * c_t[None, :]).sum(dim=-1) + D_f * u_t
            if z is not None:
                y_t = y_t * F.silu(z[:, t].float())
            out[:, t] = y_t.to(out.dtype)
        if slot_valid and s_end > s_start:
            ssm_state[slot].copy_(state.to(ssm_state.dtype))
    return out


def _vllm_prefill_reference(
    hidden_states: torch.Tensor,
    weights: dict[str, torch.Tensor | None],
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    use_rms_norm: bool,
    rms_norm_eps: float,
    activation: str,
) -> torch.Tensor:
    """Reference for the pure-prefill branch of MambaMixer.forward_impl."""
    intermediate_size = weights["conv1d_weight"].shape[0]
    state_size = (
        weights["x_proj_weight"].shape[0] - weights["dt_proj_weight"].shape[1]
    ) // 2
    time_step_rank = weights["dt_proj_weight"].shape[1]
    projected = F.linear(
        hidden_states,
        weights["in_proj_weight"],
        weights["in_proj_bias"],
    )
    x, gate = projected.split(intermediate_size, dim=-1)
    # vLLM causal_conv1d_fn expects [I, T], and mutates Conv State in-place.
    conv_out = causal_conv1d_fn(
        x.transpose(0, 1),
        weights["conv1d_weight"],
        weights["conv1d_bias"],
        conv_state,
        query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=activation,
        null_block_id=NULL_BLOCK_ID,
    )
    dt, b, c = _ssm_transform_reference(
        conv_out.transpose(0, 1).contiguous(),
        weights,
        time_step_rank=time_step_rank,
        state_size=state_size,
        use_rms_norm=use_rms_norm,
        rms_norm_eps=rms_norm_eps,
    )
    # vLLM selective_scan_fn writes the gated result into its z argument.
    scan_gate = gate.transpose(0, 1).contiguous()
    if _HAS_VLLM_SELECTIVE_SCAN:
        scan_out = selective_scan_fn(
            conv_out,
            ssm_state,
            dt.transpose(0, 1).contiguous(),
            weights["A"],
            b.transpose(0, 1).contiguous(),
            c.transpose(0, 1).contiguous(),
            weights["D"],
            scan_gate,
            weights["dt_proj_bias"],
            delta_softplus=True,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            null_block_id=NULL_BLOCK_ID,
        )
    else:
        scan_out = _torch_selective_scan_reference(
            conv_out,
            ssm_state,
            dt.transpose(0, 1).contiguous(),
            weights["A"],
            b.transpose(0, 1).contiguous(),
            c.transpose(0, 1).contiguous(),
            weights["D"],
            scan_gate,
            weights["dt_proj_bias"],
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            null_block_id=NULL_BLOCK_ID,
        )
    return F.linear(
        scan_out.transpose(0, 1).contiguous(),
        weights["out_proj_weight"],
        weights["out_proj_bias"],
    )


def _vllm_decode_reference(
    hidden_states: torch.Tensor,
    weights: dict[str, torch.Tensor | None],
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    cache_indices: torch.Tensor,
    *,
    use_rms_norm: bool,
    rms_norm_eps: float,
    activation: str,
) -> torch.Tensor:
    """Reference for the one-token-per-sequence decode branch of MambaMixer."""
    intermediate_size = weights["conv1d_weight"].shape[0]
    state_size = (
        weights["x_proj_weight"].shape[0] - weights["dt_proj_weight"].shape[1]
    ) // 2
    time_step_rank = weights["dt_proj_weight"].shape[1]
    projected = F.linear(
        hidden_states,
        weights["in_proj_weight"],
        weights["in_proj_bias"],
    )
    x, gate = projected.split(intermediate_size, dim=-1)
    # causal_conv1d_update expects [batch, I] for one-token decode.
    conv_out = causal_conv1d_update(
        x.contiguous(),
        conv_state,
        weights["conv1d_weight"],
        weights["conv1d_bias"],
        activation=activation,
        conv_state_indices=cache_indices,
        null_block_id=NULL_BLOCK_ID,
    )
    dt, b, c = _ssm_transform_reference(
        conv_out,
        weights,
        time_step_rank=time_step_rank,
        state_size=state_size,
        use_rms_norm=use_rms_norm,
        rms_norm_eps=rms_norm_eps,
    )
    scan_out = torch.empty_like(conv_out)
    selective_state_update(
        ssm_state,
        conv_out,
        dt,
        weights["A"],
        b,
        c,
        weights["D"],
        weights["dt_proj_bias"],
        z=gate,
        dt_softplus=True,
        state_batch_indices=cache_indices,
        dst_state_batch_indices=cache_indices,
        null_block_id=NULL_BLOCK_ID,
        out=scan_out,
    )
    return F.linear(
        scan_out,
        weights["out_proj_weight"],
        weights["out_proj_bias"],
    )


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
    *,
    name: str,
) -> None:
    # Keep thresholds consistent with benchmark/test_mamba_mixer_perf.py.
    # The SSM state itself is FP32 even for FP16/BF16 activations; using the
    # same end-to-end threshold is intentional for this integration test.
    if dtype == torch.bfloat16:
        atol, rtol = 3e-2, 3e-2
    elif dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    elif dtype == torch.float32:
        atol, rtol = 1e-3, 1e-3
    else:
        raise AssertionError(f"unexpected dtype in correctness test: {dtype}")
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        msg=f"{name} mismatch",
    )


@pytest.mark.mamba_mixer
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
@pytest.mark.parametrize("use_rms_norm", [False, True], ids=["no-rmsnorm", "rmsnorm"])
def test_mamba_mixer_prefill_matches_vllm(dtype, use_rms_norm):
    """Packed variable-length Prefill, including short sequences and cache reuse."""
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(2026)
    hidden_size = 64
    intermediate_size = 128
    state_size = 16
    time_step_rank = 8
    kernel_size = 4
    batch_size = 2
    seq_lens = (2, 5)  # first sequence is shorter than K-1
    num_actual_tokens = sum(seq_lens)
    padded_tokens = num_actual_tokens + 2
    cache_slots = batch_size + 1
    rms_norm_eps = 1e-5
    activation = "silu"
    weights = _make_weights(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        state_size=state_size,
        time_step_rank=time_step_rank,
        kernel_size=kernel_size,
        dtype=dtype,
        device=device,
        use_rms_norm=use_rms_norm,
    )
    reference_cache, triton_cache = _make_cache_pair(
        cache_slots=cache_slots,
        intermediate_size=intermediate_size,
        kernel_size=kernel_size,
        state_size=state_size,
        activation_dtype=dtype,
        device=device,
    )
    conv_state_ref, ssm_state_ref = reference_cache
    conv_state_tri, ssm_state_tri = triton_cache
    hidden_states = _randn(
        (padded_tokens, hidden_size),
        device=device,
        dtype=dtype,
        scale=0.2,
    )
    query_start_loc = torch.tensor(
        [0, seq_lens[0], num_actual_tokens],
        dtype=torch.int32,
        device=device,
    )
    # Cache slot zero is vLLM's NULL_BLOCK_ID.
    cache_indices = torch.tensor([1, 2], dtype=torch.int32, device=device)
    # Exercise both a fresh request and a request continuing from cache.
    has_initial_state = torch.tensor([False, True], dtype=torch.bool, device=device)
    with torch.inference_mode():
        expected = _vllm_prefill_reference(
            hidden_states[:num_actual_tokens],
            weights,
            conv_state_ref,
            ssm_state_ref,
            query_start_loc,
            cache_indices,
            has_initial_state,
            use_rms_norm=use_rms_norm,
            rms_norm_eps=rms_norm_eps,
            activation=activation,
        )
        sentinel = 123.0
        output = torch.full(
            (padded_tokens, hidden_size),
            sentinel,
            dtype=dtype,
            device=device,
        )
        actual = triton_mamba_mixer(
            hidden_states,
            weights["in_proj_weight"],
            weights["conv1d_weight"],
            weights["x_proj_weight"],
            weights["dt_proj_weight"],
            weights["A"],
            weights["D"],
            weights["out_proj_weight"],
            in_proj_bias=weights["in_proj_bias"],
            conv1d_bias=weights["conv1d_bias"],
            dt_proj_bias=weights["dt_proj_bias"],
            out_proj_bias=weights["out_proj_bias"],
            conv_state=conv_state_tri,
            ssm_state=ssm_state_tri,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            dt_norm_weight=weights["dt_norm_weight"],
            b_norm_weight=weights["b_norm_weight"],
            c_norm_weight=weights["c_norm_weight"],
            use_rms_norm=use_rms_norm,
            rms_norm_eps=rms_norm_eps,
            activation=activation,
            output=output,
            num_actual_tokens=num_actual_tokens,
            null_block_id=NULL_BLOCK_ID,
        )
        torch.cuda.synchronize()
    _assert_close(
        actual[:num_actual_tokens],
        expected,
        dtype,
        name="Prefill output",
    )

    # Padding rows must not be touched by the candidate.
    assert torch.equal(
        actual[num_actual_tokens:],
        torch.full_like(actual[num_actual_tokens:], sentinel),
    )
    _assert_close(
        conv_state_tri,
        conv_state_ref,
        dtype,
        name="Prefill Conv State",
    )
    _assert_close(
        ssm_state_tri,
        ssm_state_ref,
        dtype,
        name="Prefill SSM State",
    )


@pytest.mark.mamba_mixer
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
@pytest.mark.parametrize("use_rms_norm", [False, True], ids=["no-rmsnorm", "rmsnorm"])
def test_mamba_mixer_decode_matches_vllm(dtype, use_rms_norm):
    """One-token-per-sequence Decode with non-contiguous vLLM Conv State."""
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(2027)
    hidden_size = 64
    intermediate_size = 128
    state_size = 16
    time_step_rank = 8
    kernel_size = 4
    batch_size = 8
    cache_slots = batch_size + 1
    rms_norm_eps = 1e-5
    activation = "silu"

    weights = _make_weights(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        state_size=state_size,
        time_step_rank=time_step_rank,
        kernel_size=kernel_size,
        dtype=dtype,
        device=device,
        use_rms_norm=use_rms_norm,
    )
    reference_cache, triton_cache = _make_cache_pair(
        cache_slots=cache_slots,
        intermediate_size=intermediate_size,
        kernel_size=kernel_size,
        state_size=state_size,
        activation_dtype=dtype,
        device=device,
    )
    conv_state_ref, ssm_state_ref = reference_cache
    conv_state_tri, ssm_state_tri = triton_cache
    hidden_states = _randn(
        (batch_size, hidden_size),
        device=device,
        dtype=dtype,
        scale=0.2,
    )
    cache_indices = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)
    with torch.inference_mode():
        expected = _vllm_decode_reference(
            hidden_states,
            weights,
            conv_state_ref,
            ssm_state_ref,
            cache_indices,
            use_rms_norm=use_rms_norm,
            rms_norm_eps=rms_norm_eps,
            activation=activation,
        )
        actual = triton_mamba_mixer(
            hidden_states,
            weights["in_proj_weight"],
            weights["conv1d_weight"],
            weights["x_proj_weight"],
            weights["dt_proj_weight"],
            weights["A"],
            weights["D"],
            weights["out_proj_weight"],
            in_proj_bias=weights["in_proj_bias"],
            conv1d_bias=weights["conv1d_bias"],
            dt_proj_bias=weights["dt_proj_bias"],
            out_proj_bias=weights["out_proj_bias"],
            conv_state=conv_state_tri,
            ssm_state=ssm_state_tri,
            cache_indices=cache_indices,
            # is_decode=True and has_initial_state=None means read the
            # existing Conv/SSM state for each sequence.
            is_decode=True,
            dt_norm_weight=weights["dt_norm_weight"],
            b_norm_weight=weights["b_norm_weight"],
            c_norm_weight=weights["c_norm_weight"],
            use_rms_norm=use_rms_norm,
            rms_norm_eps=rms_norm_eps,
            activation=activation,
            null_block_id=NULL_BLOCK_ID,
        )
        torch.cuda.synchronize()
    _assert_close(actual, expected, dtype, name="Decode output")
    _assert_close(
        conv_state_tri,
        conv_state_ref,
        dtype,
        name="Decode Conv State",
    )
    _assert_close(
        ssm_state_tri,
        ssm_state_ref,
        dtype,
        name="Decode SSM State",
    )


@pytest.mark.mamba_mixer
def test_mamba_mixer_export_and_vendor_dispatch():
    assert "mamba_mixer" in flaggems_vllm.ops.__all__
    assert callable(flaggems_vllm.ops.mamba_mixer)
    assert callable(flaggems_vllm.mamba_mixer)
    if flaggems_vllm.vendor_name == "hygon":
        from flaggems_vllm.runtime.backend._hygon.ops.mamba_mixer import (
            mamba_mixer as hygon_mamba_mixer,
        )

        assert flaggems_vllm.mamba_mixer is hygon_mamba_mixer
    elif flaggems_vllm.vendor_name == "nvidia":
        assert flaggems_vllm.mamba_mixer is flaggems_vllm.ops.mamba_mixer


@pytest.mark.mamba_mixer
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
@pytest.mark.parametrize("use_rms_norm", [False, True], ids=["no-rmsnorm", "rmsnorm"])
def test_mamba_mixer_prefill_multi_chunk_matches_reference(dtype, use_rms_norm):
    """Multi-chunk variable-length prefill (>64 tokens) across chunk boundaries."""
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(2028)
    hidden_size = 64
    intermediate_size = 128
    state_size = 16
    time_step_rank = 8
    kernel_size = 4
    seq_lens = (3, 131, 64)
    batch_size = len(seq_lens)
    num_actual_tokens = sum(seq_lens)
    cache_slots = batch_size + 1
    rms_norm_eps = 1e-5
    activation = "silu"
    weights = _make_weights(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        state_size=state_size,
        time_step_rank=time_step_rank,
        kernel_size=kernel_size,
        dtype=dtype,
        device=device,
        use_rms_norm=use_rms_norm,
    )
    reference_cache, triton_cache = _make_cache_pair(
        cache_slots=cache_slots,
        intermediate_size=intermediate_size,
        kernel_size=kernel_size,
        state_size=state_size,
        activation_dtype=dtype,
        device=device,
    )
    conv_state_ref, ssm_state_ref = reference_cache
    conv_state_tri, ssm_state_tri = triton_cache
    hidden_states = _randn(
        (num_actual_tokens, hidden_size),
        device=device,
        dtype=dtype,
        scale=0.15,
    )
    query_start_loc = torch.tensor(
        [0, seq_lens[0], seq_lens[0] + seq_lens[1], num_actual_tokens],
        dtype=torch.int32,
        device=device,
    )
    # Slot 0 is NULL_BLOCK_ID and must remain untouched.
    cache_indices = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    has_initial_state = torch.tensor(
        [True, False, True], dtype=torch.bool, device=device
    )
    with torch.inference_mode():
        expected = _vllm_prefill_reference(
            hidden_states,
            weights,
            conv_state_ref,
            ssm_state_ref,
            query_start_loc,
            cache_indices,
            has_initial_state,
            use_rms_norm=use_rms_norm,
            rms_norm_eps=rms_norm_eps,
            activation=activation,
        )
        actual = triton_mamba_mixer(
            hidden_states,
            weights["in_proj_weight"],
            weights["conv1d_weight"],
            weights["x_proj_weight"],
            weights["dt_proj_weight"],
            weights["A"],
            weights["D"],
            weights["out_proj_weight"],
            in_proj_bias=weights["in_proj_bias"],
            conv1d_bias=weights["conv1d_bias"],
            dt_proj_bias=weights["dt_proj_bias"],
            out_proj_bias=weights["out_proj_bias"],
            conv_state=conv_state_tri,
            ssm_state=ssm_state_tri,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            dt_norm_weight=weights["dt_norm_weight"],
            b_norm_weight=weights["b_norm_weight"],
            c_norm_weight=weights["c_norm_weight"],
            use_rms_norm=use_rms_norm,
            rms_norm_eps=rms_norm_eps,
            activation=activation,
            null_block_id=NULL_BLOCK_ID,
        )
        torch.cuda.synchronize()
    _assert_close(actual, expected, dtype, name="Multi-chunk prefill output")
    _assert_close(
        conv_state_tri,
        conv_state_ref,
        dtype,
        name="Multi-chunk prefill Conv State",
    )
    _assert_close(
        ssm_state_tri,
        ssm_state_ref,
        dtype,
        name="Multi-chunk prefill SSM State",
    )
