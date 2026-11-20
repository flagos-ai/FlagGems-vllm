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

"""Prepared KDA H oracle: explicit native BF16 boundaries, V-first state."""
import pytest
import torch

import flaggems_vllm as gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

pytestmark = pytest.mark.skipif(gems.vendor_name != "ascend", reason="Ascend only")


def make_inputs(lengths=(65, 127), heads=2):
    torch.manual_seed(815)
    t = sum(lengths)
    shape = (1, heads, t, 128)
    k = (torch.randn(shape) * 0.04).bfloat16().to(gems.device)
    w = (torch.randn(shape) * 0.01).bfloat16().to(gems.device)
    u = (torch.randn(shape) * 0.3).bfloat16().to(gems.device)
    g = (-torch.rand(shape) * 0.2).to(gems.device)
    initial = (torch.randn(len(lengths), heads, 128, 128) * 0.1).to(gems.device)
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return (k, w, u), dict(gk=g, initial_state=initial, cu_seqlens=cu)


def reference(k, w, u, *, gk, initial_state, cu_seqlens, value_tile=128):
    # Independent CPU oracle; do not use implementation helpers or tile code.
    k, w, u, g = [x.cpu().double() for x in (k, w, u, gk)]
    state = initial_state.cpu().double().clone()
    snapshots = []
    values = torch.empty_like(u, dtype=torch.bfloat16)
    for seq, (bos, eos) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for start in range(bos, eos, 64):
            end = min(start + 64, eos)
            snapshot = state[seq].bfloat16()
            snapshots.append(snapshot)
            prediction = w[0, :, start:end] @ snapshot.double().transpose(-1, -2)
            corrected = (u[0, :, start:end] - prediction).bfloat16()
            values[0, :, start:end] = corrected
            delta = corrected.double().transpose(-1, -2) @ k[0, :, start:end]
            state[seq] = (
                state[seq]
                * torch.exp(g[0, :, end - 1] * 0.6931471805599453)[:, None, :]
                + delta
            )
    h = (
        torch.stack(snapshots).unsqueeze(0)
        if snapshots
        else torch.empty(1, 0, k.shape[1], 128, 128, dtype=torch.bfloat16)
    )
    return h, values, state.float()


def assert_outputs(actual, expected):
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a.shape == e.shape and a.dtype == e.dtype
        assert torch.isfinite(a).all()
        torch.testing.assert_close(
            a.cpu(),
            e.cpu(),
            # Native H has the same BF16 rounding sensitivity against FP64.
            # Keep the original H-stage validation envelope, plus aggregate error.
            rtol=0.01,
            atol=0.001,
        )

        if a.numel():
            error = a.cpu().double() - e.cpu().double()
            denominator = e.cpu().double().square().mean().clamp_min(1e-20)
            assert (error.square().mean() / denominator).sqrt() <= 0.001


CASES = [(1,), (63,), (64,), (65,), (0,), (0, 65, 0, 127)]
if not QUICK_MODE:
    CASES += [(4096,), (8193,), (17, 256, 4097)]


@pytest.mark.chunk_kda_fwd_h
@pytest.mark.parametrize("lengths", CASES)
@pytest.mark.parametrize("tile", [32, 64, 128])
def test_h_recurrence(lengths, tile):
    args, kw = make_inputs(lengths)
    before = [x.clone() for x in (*args, kw["gk"], kw["initial_state"])]
    actual = gems.chunk_kda_fwd_h(*args, **kw, value_tile=tile)
    assert_outputs(actual, reference(*args, **kw))
    for tensor, original in zip((*args, kw["gk"], kw["initial_state"]), before):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


@pytest.mark.chunk_kda_fwd_h
@pytest.mark.parametrize(
    "invalid",
    [
        "dtype",
        "layout",
        "gate",
        "state",
        "cu",
        "device_cu",
        "tile",
        "grad",
        "tile_float",
        "tile_tensor",
        "cu_item_tensor",
    ],
)
def test_h_rejects_unsupported(invalid):
    args, kw = make_inputs()
    args = list(args)
    error = NotImplementedError
    if invalid == "dtype":
        args[0] = args[0].float()
    if invalid == "layout":
        args[0] = args[0].transpose(-1, -2).contiguous().transpose(-1, -2)
    if invalid == "gate":
        kw["gk"] = kw["gk"].bfloat16()
    if invalid == "state":
        kw["initial_state"] = kw["initial_state"][:1]
        error = ValueError
    if invalid == "cu":
        kw["cu_seqlens"] = [0, 193, 192]
        error = ValueError
    if invalid == "device_cu":
        kw["cu_seqlens"] = torch.tensor([0, 65, 192], device=gems.device)
    if invalid == "tile":
        kw["value_tile"] = 16
        error = ValueError
    if invalid == "grad":
        kw["initial_state"].requires_grad_()
    if invalid == "tile_float":
        kw["value_tile"] = 64.0
        error = ValueError
    if invalid == "tile_tensor":
        kw["value_tile"] = torch.tensor(64, device=gems.device)
        error = ValueError
    if invalid == "cu_item_tensor":
        kw["cu_seqlens"] = [0, torch.tensor(192, device=gems.device)]
    with pytest.raises(error):
        gems.chunk_kda_fwd_h(*args, **kw)


@pytest.mark.chunk_kda_fwd_h
def test_h_changed_input_graph_replay():
    args, kw = make_inputs((65, 127))
    gems.chunk_kda_fwd_h(*args, **kw)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = gems.chunk_kda_fwd_h(*args, **kw)
    for factor in (0.0, 0.5, 1.25):
        args[2].fill_(factor)
        kw["initial_state"].fill_(factor * 0.1)
        graph.replay()
        torch.npu.synchronize()
        assert_outputs(out, reference(*args, **kw))


@pytest.mark.chunk_kda_fwd_h
@pytest.mark.parametrize("lengths", [(65, 127), (4096,), (8193,)])
def test_h_native_parity(lengths):
    native = pytest.importorskip("fla_npu.ops.ascendc").npu_chunk_fwd_h
    args, kw = make_inputs(lengths, heads=16)
    cu = kw["cu_seqlens"]
    indices = [
        x
        for s, (a, b) in enumerate(zip(cu, cu[1:]))
        for c in range((b - a + 63) // 64)
        for x in (s, c)
    ]
    expected = native(
        *args,
        **kw,
        output_final_state=True,
        chunk_size=64,
        save_new_value=True,
        chunk_indices=indices,
        use_exp2=True,
        state_v_first=True,
    )
    actual = gems.chunk_kda_fwd_h(*args, **kw)
    for a, e in zip(actual, expected):
        utils.gems_assert_equal(a, e)


@pytest.mark.chunk_kda_fwd_h
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.0])
def test_h_nonfinite_and_zero(value):
    args, kw = make_inputs((65,), heads=1)
    args[2].fill_(value)
    actual = gems.chunk_kda_fwd_h(*args, **kw)
    expected = reference(*args, **kw)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(
            a.cpu(), e.cpu(), rtol=0.016, atol=0.002, equal_nan=True
        )


@pytest.mark.chunk_kda_fwd_h
def test_h_public_dispatch_log(caplog):
    args, kw = make_inputs((1,), heads=1)
    entry = gems.chunk_kda_fwd_h
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(entry)):
        actual = entry(*args, **kw)
    assert f"{utils.gems_log_prefix(entry)} CHUNK_KDA_FWD_H" in caplog.text
    expected = reference(*args, **kw)
    # The first snapshot is exactly the BF16 materialized initial state.
    utils.gems_assert_equal(actual[0].cpu(), expected[0])
    assert_outputs(actual, expected)
