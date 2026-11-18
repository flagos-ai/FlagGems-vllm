# Copyright 2026- Xcoresigma Technology Co., Ltd
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

"""Correctness coverage for the Ascend causal-conv1d-update (decode) kernel.

The kernel under test is the Ascend TLE-DSA port living in
``flaggems_vllm/runtime/backend/_ascend/ops/causal_conv1d_update.py`` and
exported as ``flaggems_vllm.causal_conv1d_update`` via the vendor op registrar.

Torch references live in this test module only. The production wrapper keeps
its vendor-port torch compute calls (dtype cast / contiguous) and is
exercised as-is.
"""

import glob
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("torch_npu")
import torch.nn.functional as F  # noqa: E402

import flaggems_vllm  # noqa: E402

from .conftest import QUICK_MODE  # noqa: E402

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend" or not torch.npu.is_available(),
    reason="causal_conv1d_update requires an Ascend NPU",
)

device = "npu"
PAD_SLOT_ID = -1


def _ci_env_fingerprint():
    """Environment fingerprint for CI logs (stdout is archived by run_tests.py
    --dump-output into accuracy_stdout.log; compare against a local baseline)."""
    import importlib.metadata as im

    import torch_npu
    import triton

    parts = [
        f"torch={torch.__version__}",
        f"torch_npu={torch_npu.__version__}",
        f"triton={triton.__version__}",
    ]
    try:
        parts.append(f"flagtree={im.version('flagtree')}")
    except im.PackageNotFoundError:
        pass
    parts.append(f"device={torch.npu.get_device_name(0)}")
    parts.append(f"cann_dirs={sorted(glob.glob('/usr/local/Ascend/cann*'))}")
    try:
        with open("/usr/local/Ascend/driver/version.info") as f:
            parts.append(f"driver={f.read().strip().splitlines()[0]}")
    except OSError:
        pass
    return " ".join(parts)


def _kernel_ir_fingerprint(kernel_name):
    """One-line lowering fingerprint of the newest <kernel_name> dump (requires
    the dump enabled by the module fixture below). The dump is written once per
    specialization, so the fingerprint printed by the first case of each new
    specialization describes that specialization's lowering."""
    import triton

    pattern = os.path.join(triton.knobs.cache.dump_dir, "*", kernel_name + ".ttadapter")
    dumps = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not dumps:
        return "no-dump"
    with open(dumps[-1]) as f:
        text = f.read()
    ops = (
        "scf.for",
        "arith.select",
        "memref.copy",
        "memref.subview",
        "tensor.extract_slice",
        "linalg.fill",
        "pointer_cast",
    )
    counts = " ".join(f"{op}={text.count(op)}" for op in ops)
    return f"lines={len(text.splitlines())} dumps={len(dumps)} {counts}"


def _tiling_fingerprint(batch, dim):
    """Mirror of the wrapper tiling heuristic (causal_conv1d_update.py). batch
    here is padded_batch_size (the kernel-side batch, i.e. x.shape[0]). The
    B_TILE/BLOCK_N choice selects the compiled specialization; B_TILE > 1 turns
    on the per-program sequence loop (`for bi in tl.range(0, B_TILE)`), the
    structure unique to the batch_size=64 cases."""
    import triton

    block_n = 1024 if dim >= 1024 else 256
    g = triton.cdiv(dim, block_n)
    target = 2 * 48  # CORE_HINT = 48 in the wrapper
    b_tile_raw = max(1, (batch * g + target - 1) // target)
    if b_tile_raw <= 1:
        b_tile = 1
    elif b_tile_raw <= 2:
        b_tile = 2
    elif b_tile_raw <= 4:
        b_tile = 4
    else:
        b_tile = 8
    t_chunk = 1 if block_n == 1024 else 48
    grid = (triton.cdiv(batch, b_tile), triton.cdiv(dim, block_n))
    return f"B_TILE={b_tile} BLOCK_N={block_n} T_CHUNK={t_chunk} grid={grid}"


@pytest.fixture(scope="module", autouse=True)
def _ci_dump_ir():
    """Print the env fingerprint once and enable IR dump for this module only
    (runtime knob, lazy-read at compile time; restored on teardown)."""
    import triton

    print(f"\n[ci-diag] {_ci_env_fingerprint()}")
    prev = triton.knobs.compilation.dump_ir
    triton.knobs.compilation.dump_ir = True
    yield
    triton.knobs.compilation.dump_ir = prev


if QUICK_MODE:
    BATCH = [3]
    WITH_PADDING = [True]
    DIM = [2048 + 16]
    SEQS = [1]
    WIDTH = [3]
    HAS_BIAS = [True]
    SILU = [True]
    ITYPES = [torch.bfloat16]
else:
    BATCH = [3, 64]
    WITH_PADDING = [True, False]
    DIM = [2048 + 16, 4096]
    SEQS = [1, 3]
    WIDTH = [3, 4]
    HAS_BIAS = [False, True]
    SILU = [True]
    ITYPES = [torch.bfloat16]


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    """Per-sequence torch reference; test-only, never imported by production.

    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1; updated
    in place by taking the last state_len columns of concat(conv_state, x).
    weight: (dim, width); bias: (dim,)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(
            weight.dtype
        )  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


@pytest.mark.causal_conv1d_update
@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("silu_activation", SILU)
@pytest.mark.parametrize("has_bias", HAS_BIAS)
@pytest.mark.parametrize("seqlen", SEQS)
@pytest.mark.parametrize("width", WIDTH)
@pytest.mark.parametrize("dim", DIM)
@pytest.mark.parametrize("with_padding", WITH_PADDING)
@pytest.mark.parametrize("batch_size", BATCH)
def test_causal_conv1d_update_with_batch_gather(
    batch_size, with_padding, dim, width, seqlen, has_bias, silu_activation, itype
):
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2

    torch.manual_seed(0)

    padding = 5 if with_padding else 0
    padded_batch_size = batch_size + padding
    # total_entries = number of cache line
    total_entries = 10 * batch_size

    # x will be (batch, dim, seqlen) with contiguous along dim-axis
    x = torch.randn(
        padded_batch_size, seqlen, dim, device=device, dtype=itype
    ).transpose(1, 2)

    x_ref = x[:batch_size].clone()
    conv_state_indices = torch.randperm(total_entries)[:batch_size].to(
        dtype=torch.int32, device=device
    )
    padded_state_indices = torch.concat(
        [
            conv_state_indices,
            torch.as_tensor([PAD_SLOT_ID] * padding, dtype=torch.int32, device=device),
        ],
        dim=0,
    )

    # conv_state will be (cache_lines, dim, state_len)
    # with contiguous along dim-axis
    conv_state = torch.randn(
        total_entries, width - 1, dim, device=device, dtype=itype
    ).transpose(1, 2)

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    conv_state_ref = conv_state[conv_state_indices, :].detach().clone()
    activation = None if not silu_activation else "silu"
    npu_out = flaggems_vllm.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=padded_state_indices,
        pad_slot_id=PAD_SLOT_ID,
    )

    # CI diagnosability: dump the specialization-selecting inputs (padded
    # batch/dim drive B_TILE/BLOCK_N; state indices come from CPU randperm and
    # must match between CI and local) and the lowering fingerprint to stdout.
    print(
        f"[ci-diag] batch_kernel={padded_batch_size} dim={dim} "
        f"{_tiling_fingerprint(padded_batch_size, dim)}"
    )
    print(f"[ci-diag] conv_state_indices={padded_state_indices.tolist()}")
    print(
        "[ci-diag] ir-fingerprint: "
        f"{_kernel_ir_fingerprint('_causal_conv1d_update_kernel_npu_tiled')}"
    )

    out_ref = causal_conv1d_update_ref(
        x_ref, conv_state_ref, weight, bias, activation=activation
    )

    npu_out = npu_out.cpu()
    out_ref = out_ref.cpu()

    # 1. mainline output matches the reference
    assert torch.allclose(npu_out[:batch_size], out_ref, rtol=rtol, atol=atol)

    # 2. in-place conv-state update matches the reference
    assert torch.allclose(
        conv_state[conv_state_indices].cpu(),
        conv_state_ref.cpu(),
        rtol=rtol,
        atol=atol,
    )
