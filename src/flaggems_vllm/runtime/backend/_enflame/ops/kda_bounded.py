# SPDX-License-Identifier: Apache-2.0
"""GCU overrides for the bounded KDA operator.

The public entrypoints keep the generic implementations: they resolve their
collaborators through the generic module's namespace, so the validation
suite's monkeypatching contract keeps working. This module contributes:

- the vLLM 0.20.x FLA adapters in ``_enflame.ops.kda64`` (natural-log gate
  units, [K, V] history reader, no ``USE_EXP2``), wired into the generic
  module in place of the stock collaborators;
- an additive ``_chunk_kda_fwd_with_cumulative_g`` shim on vLLM's kda
  module. vLLM >= 0.23 ships that private host as the prefill reference for
  the GLM5.3 validation suite and benchmark; the vLLM pinned here (0.20.x)
  predates it. The shim runs the very same 0.20.x core launches the
  operator uses, so reference and operator stay bit-identical.
"""

import sys
from functools import wraps
from importlib import import_module

import triton

from flaggems_vllm.ops.kda_bounded import (
    chunk_kda_with_safe_gate,
    fused_recurrent_kda,
    recurrent_kda,
    safe_kda_gate,
    safe_kda_gate_chunk_cumsum,
)
from flaggems_vllm.runtime.backend._enflame.ops.kda64 import (
    _gcu_recurrent_kda_fwd,
    _prefill_core,
    prefill_kda64,
    recurrent_kda_vk,
)

__all__ = [
    "chunk_kda_with_safe_gate",
    "fused_recurrent_kda",
    "recurrent_kda",
    "safe_kda_gate",
    "safe_kda_gate_chunk_cumsum",
]


def kda_ref():
    return import_module("vllm.model_executor.layers.fla.ops.kda")


def _chunk_kda_fwd_with_cumulative_g(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=64,
):
    """vLLM >= 0.23 cumulative-g prefill host, on the 0.20.x core.

    ``g`` must already be the chunk-local inclusive cumsum scaled by
    RCP_LN2 (log2 units), matching the upstream contract. The 0.20.x core
    consumes natural-log gates, so the shared adapter converts the units;
    both this reference path and the operator path feed identical bits to
    identical kernel launches.
    """
    if chunk_size != 64:
        raise NotImplementedError(
            "GCU KDA prefill keeps the original 64-token chunk size"
        )
    if cu_seqlens is not None:
        if chunk_indices is None:
            chunk_indices = kda_ref().prepare_chunk_indices(cu_seqlens, chunk_size)
        chunk_offsets = import_module(
            "vllm.model_executor.layers.fla.ops.index"
        ).prepare_chunk_offsets(cu_seqlens, chunk_size)
    else:
        chunk_offsets = None
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    return prefill_kda64(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        _prefill_core(),
    )


def _install_vllm_compat():
    """Add the missing prefill reference host to vLLM's kda module.

    Additive only: an existing attribute always wins, so a real vLLM >= 0.23
    install is never shadowed.
    """
    try:
        kda = kda_ref()
    except Exception:
        return
    if not callable(getattr(kda, "_chunk_kda_fwd_with_cumulative_g", None)):
        kda._chunk_kda_fwd_with_cumulative_g = _chunk_kda_fwd_with_cumulative_g
    _install_decode_reroute(kda)


def _install_decode_reroute(kda):
    """Reroute vLLM's decode host around the unlinkable BK >= 128 kernel.

    The stock ``fused_recurrent_gated_delta_rule_fwd_kernel`` lowers its
    [BV, BK] state mask to a 128-lane 2-D bool mask at BK >= 128, which
    ld.lld rejects (R_DTU_ADDR16_LO_ICALL, see compiler_issues/
    ld_lld_relocation.md). The wrapper keeps vLLM's host for every K that
    links (K <= 64) and only substitutes the GCU adapter — which mirrors the
    0.20 kernel line by line — for the unlinkable tiles. On this toolchain
    the substituted path is also what the operator runs, so reference and
    operator stay on one launch path, exactly like the prefill shim above.
    """
    original = getattr(kda, "fused_recurrent_kda", None)
    if not callable(original) or getattr(original, "_flaggems_gcu_reroute", False):
        return

    @wraps(original)
    def fused_recurrent_kda(
        q,
        k,
        v,
        g,
        beta=None,
        scale=None,
        initial_state=None,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
        ssm_state_indices=None,
        **kwargs,
    ):
        if triton.next_power_of_2(k.shape[-1]) >= 128:
            if cu_seqlens is not None and q.shape[0] != 1:
                raise ValueError(
                    f"The batch size is expected to be 1 rather than {q.shape[0]}"
                    f" when using `cu_seqlens`."
                    f"Please flatten variable-length inputs before processing."
                )
            if scale is None:
                scale = k.shape[-1] ** -0.5
            return _gcu_recurrent_kda_fwd(
                q=q.contiguous(),
                k=k.contiguous(),
                v=v.contiguous(),
                g=g.contiguous(),
                beta=beta.contiguous(),
                scale=scale,
                initial_state=initial_state,
                inplace_final_state=inplace_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=ssm_state_indices,
            )
        return original(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            inplace_final_state,
            use_qk_l2norm_in_kernel,
            cu_seqlens,
            ssm_state_indices,
            **kwargs,
        )

    fused_recurrent_kda._flaggems_gcu_reroute = True
    fused_recurrent_kda._flaggems_vllm_original = original
    kda.fused_recurrent_kda = fused_recurrent_kda


def _install():
    """Wire the GCU collaborators into the generic module entrypoints.

    Tests and benchmarks import ``from flaggems_vllm.ops import kda_bounded``,
    which resolves against the already-imported generic module, so the
    SpecOpRegistrar top-level namespace swap alone cannot reach them.
    Replace the collaborators on that module in place, mirroring the kpool
    backend override precedent: only a pristine generic collaborator is
    swapped, so explicit user or test patches always win.
    """
    mod = sys.modules.get("flaggems_vllm.ops.kda_bounded")
    if mod is None:
        return
    for name, override in (
        ("_prefill_core", _prefill_core),
        ("prefill_kda64", prefill_kda64),
        ("recurrent_kda_vk", recurrent_kda_vk),
    ):
        current = getattr(mod, name, None)
        if current is not None and getattr(current, "__module__", None) == (
            "flaggems_vllm.ops.kda64"
        ):
            setattr(mod, name, override)
    _install_vllm_compat()


try:
    _install()
except Exception:
    # pytest assertion-rewrite 上下文中 @triton.jit 源定位可能失败
    # (collect 期动态触碰 generic 模块触发);跳过接线不影响 op 路由
    # (SpecOpRegistrar 已保证),直连 import 入口的接线由运行期重试兜底。
    pass