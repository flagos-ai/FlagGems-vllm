import itertools

import pytest
import torch

import flaggems_vllm
from benchmark.base import Benchmark
from flaggems_vllm.ops.FLA import chunk_dplr

DEFAULT_H = 96
DEFAULT_D = 128
FIXED_CASES = [
    [8192],
]
VARLEN_CASES = [
    [1300, 547, 2048, 963, 271, 3063],
    [1024] * 8,
]
FLASHDPLR_CASES = FIXED_CASES + VARLEN_CASES


def _chunk_dplr_inference_op(*args, **kwargs):
    with torch.inference_mode():
        return chunk_dplr(*args, **kwargs)


def _naive_dplr_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    """Naive DPLR recurrence implementation for reference.

    Based on fla-9/fla/ops/generalized_delta_rule/dplr/naive.py::dplr_recurrence
    S_t = S_t @ (I + alpha_t beta_t^T) + v_t k_t^T
    """
    orig_dtype = q.dtype
    b, h, l, d_k = q.shape
    q, k, v, beta, gk = map(lambda x: x.float(), [q, k, v, beta, gk])
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * (d_k**-0.5)

    if initial_state is not None:
        S += initial_state

    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i]
        _alpha = alpha[:, :, i].clone()
        _beta = beta[:, :, i].clone()
        _kv = (
            _k[..., None] * _v[..., None, :]
            + (S.clone() * _alpha[..., None]).sum(-2, keepdim=True) * _beta[..., None]
        )
        S = S.clone() * gk[:, :, i].exp()[..., None] + _kv
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", _q, S)
    S = None if output_final_state is False else S
    return o.to(orig_dtype), S


def _chunk_dplr_torch_reference_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 16,
):
    """Reference implementation using naive DPLR recurrence.

    Input format: q, k, v, a, b, gk are all [B, T, H, D] or [1, T_total, H, D]
    cu_seqlens is used to split T_total into multiple sequences if varlen
    """
    B, T, H, K = q.shape
    _ = v.shape[-1]  # V unused but kept for clarity

    if cu_seqlens is not None:
        # For varlen case, process each sequence separately
        N = cu_seqlens.shape[0] - 1

        outputs = []
        final_states = []

        for i in range(N):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()

            q_i = q[:, start:end, :, :]  # [1, T_i, H, K]
            k_i = k[:, start:end, :, :]
            v_i = v[:, start:end, :, :]
            a_i = a[:, start:end, :, :]
            b_i = b[:, start:end, :, :]
            gk_i = gk[:, start:end, :, :]

            # Convert to [B, H, L, D] format for naive implementation
            q_i = q_i.transpose(1, 2)  # [1, H, T_i, K]
            k_i = k_i.transpose(1, 2)
            v_i = v_i.transpose(1, 2)
            a_i = a_i.transpose(1, 2)
            b_i = b_i.transpose(1, 2)
            gk_i = gk_i.transpose(1, 2)

            o_i, s_i = _naive_dplr_recurrence(
                q_i,
                k_i,
                v_i,
                a_i,
                b_i,
                gk_i,
                initial_state=None,
                output_final_state=output_final_state,
            )

            # Convert back to [1, T_i, H, V] format
            o_i = o_i.transpose(1, 2)  # [1, T_i, H, V]
            outputs.append(o_i)
            if output_final_state:
                final_states.append(s_i)

        o = torch.cat(outputs, dim=1)  # [1, T_total, H, V]
        final_state = torch.cat(final_states, dim=0) if output_final_state else None
    else:
        # Fixed length case
        # Convert from [B, T, H, D] to [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        a = a.transpose(1, 2)
        b = b.transpose(1, 2)
        gk = gk.transpose(1, 2)

        o, final_state = _naive_dplr_recurrence(
            q,
            k,
            v,
            a,
            b,
            gk,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

        # Convert back to [B, T, H, V]
        o = o.transpose(1, 2)

    return o, final_state


class ChunkDPLRBenchmark(Benchmark):
    """
    Benchmark for DPLR (Delta rule with Per-head Low-rank) attention kernel.

    Compares FlagGems optimized Triton implementation against naive PyTorch reference.
    """

    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPES = [
        tuple(itertools.chain(seq_lens, [DEFAULT_H, DEFAULT_D]))
        for seq_lens in FLASHDPLR_CASES
    ]
    DEFAULT_SHAPE_DESC = "seq_lens..., H, D"

    def init_user_config(self):
        super().init_user_config()
        if any(len(shape) < 3 for shape in self.shapes):
            self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            *seq_lens, H, D = shape
            yield self._build_inputs(seq_lens, H, D, cur_dtype)

    def set_more_dtypes(self):
        return [
            torch.bfloat16,
        ]

    def get_bwd_fn(self):
        return lambda o, final_state: o.sum() + (
            final_state.sum() if final_state is not None else 0
        )

    def _build_inputs(self, seq_lens, H, D, dtype):
        K, V = D, D
        device = self.device

        # Calculate total sequence length
        T_total = sum(seq_lens)
        N = len(seq_lens)

        # For varlen, use cu_seqlens but keep shape as [1, T_total, H, K]
        # For fixed length, shape is [B, T, H, K] where B=1
        cu_seqlens = None
        if N > 1:
            cu_seqlens = torch.tensor(
                [0] + list(itertools.accumulate(seq_lens)),
                dtype=torch.long,
                device=device,
            )

        # Always use [B=1, T, H, D] format
        q = torch.randn((1, T_total, H, K), dtype=dtype, device=device)
        k = torch.randn((1, T_total, H, K), dtype=dtype, device=device)
        v = torch.randn((1, T_total, H, V), dtype=dtype, device=device)
        a = torch.randn((1, T_total, H, K), dtype=dtype, device=device)
        b = torch.randn((1, T_total, H, K), dtype=dtype, device=device)
        gk = torch.randn((1, T_total, H, K), dtype=dtype, device=device)

        scale = K**-0.5
        initial_state = None
        chunk_size = 16

        return (
            q,
            k,
            v,
            a,
            b,
            gk,
            {
                "scale": scale,
                "initial_state": initial_state,
                "output_final_state": True,
                "cu_seqlens": cu_seqlens,
                "chunk_size": chunk_size,
            },
        )


@pytest.mark.skipif(
    flaggems_vllm.device != "cuda", reason="chunk_dplr benchmark requires CUDA"
)
@pytest.mark.chunk_dplr
def test_chunk_dplr():
    bench = ChunkDPLRBenchmark(
        op_name="chunk_dplr",
        torch_op=_chunk_dplr_torch_reference_op,
    )
    bench.set_gems(_chunk_dplr_inference_op)
    bench.run()
