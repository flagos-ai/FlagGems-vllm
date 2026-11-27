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

"""Accuracy tests for the Ascend A5 HcPre op.

Run on an idle 950DT, for example::

    ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=src pytest -q tests/test_hc_pre.py

What these assert against, and why it is not the vendor op: the baseline
``torch.ops._C_ascend.npu_hc_pre_v2`` is a vllm-ascend *vendor* op rather than a
CANN builtin, and reaching it requires
``vllm_ascend.utils.bootstrap_custom_op_env(include_vendor_lib=True)`` to run
before ``torch_npu`` is imported -- which a pytest session cannot guarantee,
since any earlier collected module may import ``torch_npu`` first.  The
reference here is therefore ``hc_pre_ref``, an fp32 torch transcription of
vllm's own ``model_executor/kernels/mhc/torch.py::mhc_pre_torch``; the vendor
comparison runs as one extra test, skipped when the namespace is unreachable.

Tolerance is 4e-2, which is the vendor op's own error level against the same
fp32 reference (it runs the mixes GEMM in HF32 and emits bf16 y).  Our vector
path, T < 16, lands around 2e-3; the cube path matches the vendor at 3.1e-2
because it enables the same HF32 switch.  Requiring better than the op being
replaced would not be meaningful -- see ``hc_pre.py`` for the full table.
"""

import sys

import pytest
import torch

if "torch_npu" not in sys.modules:
    # Best effort: if this module is imported before anything pulls in
    # torch_npu, put the vendor op within reach of test_matches_vendor_op.
    try:
        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env(include_vendor_lib=True)
    except Exception:  # noqa: BLE001 - optional, and never fatal here
        pass

torch_npu = pytest.importorskip("torch_npu")

from flaggems_vllm.runtime.backend._ascend.a5.ops.hc_pre import (  # noqa: E402
    HC_MULT_LIMIT,
    HC_POST_MULT,
    hc_pre,
)

from . import conftest as cfg  # noqa: E402

ATOL = RTOL = 4e-2


def hc_pre_ref(
    x,
    hc_fn,
    hc_scale,
    hc_base,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    norm_eps=1e-6,
    hc_eps=1e-6,
):
    """fp32 torch reference, line by line from vllm's ``mhc_pre_torch``.

    ``post_mult`` is hard-coded to 2.0 as the vendor op does.  Returns
    ``(y, post, comb, pre)``; ``pre`` is extra, for the tests' benefit.

    Test-only helper: production code must not import this (``workflow.md``
    NO_TORCH_COMPUTE_FALLBACK).
    """
    n = x.shape[-2]
    D = x.shape[-1]
    outer = x.shape[:-2]
    xf = x.reshape(-1, n * D).float()
    mixes = xf @ hc_fn.t().float()
    mixes = mixes * torch.rsqrt(xf.square().sum(-1, keepdim=True) / (n * D) + norm_eps)

    pre = torch.sigmoid(mixes[:, :n] * hc_scale[0] + hc_base[:n]) + hc_eps
    post = (
        torch.sigmoid(mixes[:, n : 2 * n] * hc_scale[1] + hc_base[n : 2 * n])
        * HC_POST_MULT
    )

    c = mixes[:, 2 * n :].view(-1, n, n) * hc_scale[2] + hc_base[2 * n :].view(1, n, n)
    c = torch.softmax(c, dim=-1) + hc_eps
    c = c / (c.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(hc_sinkhorn_iters - 1):
        c = c / (c.sum(dim=-1, keepdim=True) + hc_eps)
        c = c / (c.sum(dim=-2, keepdim=True) + hc_eps)

    y = (pre.unsqueeze(-1) * x.reshape(-1, n, D).float()).sum(1).to(x.dtype)
    return (
        y.view(*outer, D),
        post.view(*outer, n),
        c.view(*outer, n, n),
        pre.view(*outer, n),
    )


def _inputs(T, D, n=4, seed=0):
    """Build HcPre inputs on CPU and on the NPU, from the same draw.

    ``hc_fn`` is scaled down to 0.02 to match real weights.  At unit scale the
    mixes are large enough to saturate every sigmoid, pre/post degenerate into
    constants and numerical problems stop being visible.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    j = n * n + 2 * n
    x = torch.randn(T, n, D, generator=g, dtype=torch.bfloat16)
    fn = torch.randn(j, n * D, generator=g, dtype=torch.float32) * 0.02
    scale = torch.rand(3, generator=g, dtype=torch.float32) + 0.5
    base = torch.randn(j, generator=g, dtype=torch.float32) * 0.1
    cpu = (x, fn, scale, base)
    return tuple(t.npu() for t in cpu), cpu


def _assert_close(name, got, want):
    got = got.float().cpu()
    want = want.float().cpu().reshape(got.shape)
    delta = (got - want).abs()
    worst = float(delta.max()) if delta.numel() else 0.0
    assert bool(
        (delta <= ATOL + RTOL * want.abs()).all()
    ), f"{name}: max abs diff {worst:.3e} exceeds atol={ATOL} rtol={RTOL}"
    return worst


def _check(T, D, n=4, iters=20, seed=0, **kwargs):
    dev, cpu = _inputs(T, D, n, seed)
    ry, rpost, rcomb, _ = hc_pre_ref(*cpu, n, iters)
    y, post, comb = hc_pre(*dev, n, iters, **kwargs)
    assert y.shape == (T, D)
    assert post.shape == (T, n)
    assert comb.shape == (T, n, n)
    assert y.dtype == torch.bfloat16
    assert post.dtype == torch.float32 and comb.dtype == torch.float32
    _assert_close("y", y, ry)
    _assert_close("post", post, rpost)
    _assert_close("comb", comb, rcomb)
    # Structural invariant, independent of the reference: the sinkhorn loop
    # ends on a column normalisation, so comb's *columns* sum to 1 (to fp32
    # precision) for any iteration count.  Rows do not -- at T=16 the fp32
    # reference's own row sums are off by up to 3.8e-2 after 20 iterations,
    # because sinkhorn has not converged for those draws -- so row sums are
    # covered by the comparison against the reference above, not asserted here.
    cs = comb.float().cpu().sum(dim=-2)
    assert float((cs - 1.0).abs().max()) < 1e-4, float((cs - 1.0).abs().max())


# Shapes: T selects both the mix path (vector below 16, cube above) and the
# epilogue path (fused below 256, comb+y above), so the list straddles both
# thresholds.  T=1 is the only variant in the DSV4 profile.
if cfg.QUICK_MODE:
    T_CASES = [1, 16, 256]
    D_CASES = [4096]
    ITER_CASES = [20]
else:
    T_CASES = [1, 2, 8, 16, 64, 128, 256, 1000, 2048]
    D_CASES = [16, 2560, 4096, 7168]
    ITER_CASES = [1, 2, 5, 20]


@pytest.mark.hc_pre
@pytest.mark.parametrize("T", T_CASES)
def test_token_counts(T):
    """Both mix paths and both epilogue paths, including a non-power-of-two T.

    T=1000 is there because the cube path's BT rarely divides T and the row
    mask has to hold; T=2048 is a full prefill chunk.
    """
    _check(T, 4096)


@pytest.mark.hc_pre
@pytest.mark.parametrize("D", D_CASES)
def test_hidden_sizes(D):
    """D=16 is the smallest 16-aligned hidden size; D=2560 is the aclnn doc's
    example; D=7168 is wider than the DSV4 shape.

    At D=16, K = 4*16 = 64 is too small for ``_pick_sk``'s BK floor to leave a
    candidate, so it falls back to one program for the whole block.
    """
    _check(1, D)


@pytest.mark.hc_pre
@pytest.mark.parametrize("iters", ITER_CASES)
def test_sinkhorn_iteration_counts(iters):
    """iters=1 is the loop boundary: only the softmax and the first column
    normalization run, and the host drops the epilogue's task split because
    there is no sinkhorn left for y to hide behind.
    """
    _check(1, 4096, iters=iters)


@pytest.mark.hc_pre
def test_rank_4_input_keeps_outer_dims():
    dev, cpu = _inputs(4, 4096)
    x, fn, scale, base = dev
    ry, rpost, rcomb, _ = hc_pre_ref(cpu[0], *cpu[1:], 4, 20)
    y, post, comb = hc_pre(x.view(2, 2, 4, 4096), fn, scale, base, 4, 20)
    assert y.shape == (2, 2, 4096)
    assert post.shape == (2, 2, 4)
    assert comb.shape == (2, 2, 4, 4)
    _assert_close("y", y, ry)
    _assert_close("post", post, rpost)
    _assert_close("comb", comb, rcomb)


@pytest.mark.hc_pre
def test_non_contiguous_x():
    """The vendor op documents non-contiguous support; we get it from the
    ``.contiguous()`` in the host wrapper.
    """
    dev, cpu = _inputs(1, 4096)
    x, fn, scale, base = dev
    wide = torch.empty((1, 4, 8192), device=x.device, dtype=x.dtype)
    wide[..., ::2] = x
    strided = wide[..., ::2]
    assert not strided.is_contiguous()
    ry, _, _, _ = hc_pre_ref(cpu[0], *cpu[1:], 4, 20)
    y, _, _ = hc_pre(strided, fn, scale, base, 4, 20)
    _assert_close("y", y, ry)


@pytest.mark.hc_pre
@pytest.mark.parametrize("split_k", [1, 2, 4, 8])
def test_split_k_override_is_equivalent(split_k):
    """``split_k`` only changes how the k-loop is partitioned, so every value
    that divides K must give the same answer within tolerance.  Accuracy
    actually improves with SK (shorter accumulation chains per partial).
    """
    _check(64, 4096, split_k=split_k)


@pytest.mark.hc_pre
@pytest.mark.parametrize("block_d", [256, 1024, 4096])
def test_block_d_override_is_equivalent(block_d):
    _check(256, 4096, block_d=block_d)


@pytest.mark.hc_pre
def test_unsupported_inputs_raise():
    dev, _ = _inputs(1, 4096)
    x, fn, scale, base = dev

    with pytest.raises(ValueError, match="rank 3 or 4"):
        hc_pre(x[0, 0], fn, scale, base)
    with pytest.raises(ValueError, match="rank 3 or 4"):
        hc_pre(x.view(1, 1, 1, 4, 4096), fn, scale, base)
    with pytest.raises(ValueError, match="must equal"):
        hc_pre(x, fn, scale, base, hc_mult=2)
    with pytest.raises(ValueError, match="hc_fn must be"):
        hc_pre(x, fn[:, :-4], scale, base)
    with pytest.raises(ValueError, match="hc_base must hold"):
        hc_pre(x, fn, scale, base[:-1])
    with pytest.raises(ValueError, match="hc_scale must hold"):
        hc_pre(x, fn, scale[:2], base)
    with pytest.raises(ValueError, match="must be >= 1"):
        hc_pre(x, fn, scale, base, 4, 0)
    with pytest.raises(ValueError, match="must be bf16"):
        hc_pre(x.half(), fn, scale, base)
    with pytest.raises(ValueError, match="hc_fn must be fp32"):
        hc_pre(x, fn.bfloat16(), scale, base)
    with pytest.raises(ValueError, match="must divide K"):
        hc_pre(x, fn, scale, base, split_k=3)


@pytest.mark.hc_pre
@pytest.mark.parametrize("n", [1, 2, 8])
def test_hc_mult_other_than_four_is_unsupported(n):
    """The vendor op has the same limit (``HC_PRE_HC_LIMIT`` in
    ``torch_binding.cpp``), so this raises rather than falling back.
    """
    assert HC_MULT_LIMIT == 4
    dev, _ = _inputs(1, 1024, n=n)
    with pytest.raises(ValueError, match=f"only n={HC_MULT_LIMIT}"):
        hc_pre(*dev, n, 20)


def _vendor_hc_pre(x, fn, scale, base, n, iters, norm_eps=1e-6, hc_eps=1e-6):
    """The baseline as ``DeepseekV4DecoderLayer.hc_pre`` calls it
    (``vllm_ascend/models/deepseek_v4/model.py``).

    The op schema is registered as soon as ``vllm_ascend_C`` is imported, but
    the kernel itself only resolves if the vendor lib was preloaded before
    ``torch_npu``.  pytest cannot guarantee that ordering (conftest and plugins
    import torch_npu first), so a missing ``aclnnHcPre`` is an ordering
    condition, not a failure -- skip rather than report red.
    """
    try:
        return torch.ops._C_ascend.npu_hc_pre_v2(
            x, fn, scale, base, n, iters, norm_eps, hc_eps
        )
    except RuntimeError as exc:
        if "libopapi" in str(exc) or "aclnnHcPre" in str(exc):
            pytest.skip(
                "npu_hc_pre_v2 is registered but aclnnHcPre is unresolved: "
                "bootstrap_custom_op_env(include_vendor_lib=True) has to run "
                "before torch_npu is imported.  Use benchmark/bench_hc_pre.py "
                "for the vendor cross-check."
            )
        raise


def _vendor_available():
    try:
        import vllm_ascend.vllm_ascend_C  # noqa: F401

        # The _C_ascend namespace is lazy, so dir() lists only 7 names and
        # cannot be used to test for membership; fetch the attribute instead.
        getattr(torch.ops._C_ascend, "npu_hc_pre_v2")
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.mark.hc_pre
@pytest.mark.skipif(
    not _vendor_available(),
    reason="npu_hc_pre_v2 needs bootstrap_custom_op_env() before torch_npu",
)
@pytest.mark.parametrize("T", [1, 256])
def test_matches_vendor_op(T):
    """Cross-check against the op being replaced, at the same tolerance.

    Both sides are compared to the fp32 reference rather than to each other, so
    neither inherits the other's error.
    """
    dev, cpu = _inputs(T, 4096)
    ry, rpost, rcomb, _ = hc_pre_ref(*cpu, 4, 20)
    vy, vpost, vcomb = _vendor_hc_pre(*dev, 4, 20)
    y, post, comb = hc_pre(*dev, 4, 20)
    for name, mine, theirs, want in (
        ("y", y, vy, ry),
        ("post", post, vpost, rpost),
        ("comb", comb, vcomb, rcomb),
    ):
        assert mine.shape == theirs.shape, name
        assert mine.dtype == theirs.dtype, name
        _assert_close(f"vendor {name}", theirs, want)
        _assert_close(f"ours {name}", mine, want)


@pytest.mark.hc_pre
def test_production_path_has_no_torch_compute():
    """``workflow.md`` NO_TORCH_COMPUTE_FALLBACK audit for this op.

    The repo-wide CI check walks ``runtime/backend/_*/{ops,fused,FLA}`` and so
    does not descend into the ``a5/`` arch directory; this test covers the file
    directly.  Allocation, copies and metadata reads are allowed; anything that
    emits a compute or cast kernel is not.
    """
    import ast
    import importlib
    import inspect

    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.a5.ops.hc_pre"
    )
    assert inspect.ismodule(module)

    allowed = {"empty", "empty_like", "zeros", "zeros_like", "bfloat16", "float32"}
    tree = ast.parse(inspect.getsource(module))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts = []
        f = node.func
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if not isinstance(f, ast.Name):
            continue
        # torch.* by attribute chain, plus the usual aliases a fallback would
        # arrive under (``import torch.nn.functional as F``, ``aten``).
        if f.id in {"F", "aten"}:
            call = ".".join(reversed(parts))
            offenders.append(f"line {node.lineno}: {f.id}.{call}()")
            continue
        if f.id != "torch":
            continue
        name = ".".join(reversed(parts))
        if name not in allowed:
            offenders.append(f"line {node.lineno}: torch.{name}()")
    # Importing a compute namespace at all, even unused, is a fallback waiting
    # to happen.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if name.startswith(("torch.nn", "torch.ops", "torch_npu")):
                offenders.append(f"line {node.lineno}: import {name}")
    assert not offenders, "torch compute in the production path: " + "; ".join(
        offenders
    )
    # The fp32 torch reference belongs in this test file, not in the op module.
    defined = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not [n for n in defined if n.endswith("_ref")], sorted(defined)
