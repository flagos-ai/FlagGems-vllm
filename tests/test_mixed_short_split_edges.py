# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm as gems
from tests.accuracy_utils import gems_assert_close
from tests.test_flash_attn_varlen_func_w8a8_int8 import _reference, _run_case

pytestmark = [
    pytest.mark.flash_attn_varlen_func_w8a8,
    pytest.mark.skipif(gems.vendor_name != "metax", reason="MetaX mixed scheduling"),
]


@pytest.mark.parametrize(
    "broadcast,strided", [(False, False), (False, True), (True, True)]
)
def test_mixed_scales_and_strides(broadcast, strided):
    _run_case(
        [12, 1025],
        [8193, 1057],
        dim=256,
        heads=6,
        kvheads=2,
        paged=True,
        causal=True,
        dtype=torch.float16,
        broadcast_scales=broadcast,
        strided=strided,
        return_lse=True,
    )


def test_mixed_mutable_lengths_graph():
    torch.manual_seed(9041)
    device = gems.device
    total = 1100
    heads = 4
    dim = 256
    page = 16
    pages = 513
    q = torch.randint(-16, 17, (total, heads, dim), device=device, dtype=torch.int8)
    k = torch.randint(
        -16, 17, (2 * pages, page, 1, dim), device=device, dtype=torch.int8
    )
    v = torch.randint(-16, 17, k.shape, device=device, dtype=torch.int8)
    cu = torch.tensor([0, 12, total], device=device, dtype=torch.int32)
    used = torch.tensor([8193, 1153], device=device, dtype=torch.int32)
    table = torch.arange(2 * pages, device=device, dtype=torch.int32).reshape(2, pages)
    qs = torch.full((2, heads, 1), 0.01, device=device).expand(2, heads, 9)
    ks = torch.full((2, 1, 1), 0.01, device=device).expand(2, 1, 65)
    vs = torch.full((2, 1, 1), 0.1, device=device).expand(2, 1, 65)

    def call():
        return gems.flash_attn_varlen_func(
            q,
            k,
            v,
            total,
            cu,
            8193,
            seqused_k=used,
            block_table=table,
            q_descale=qs,
            k_descale=ks,
            v_descale=vs,
            causal=True,
            return_softmax_lse=True,
        )

    for _ in range(2):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = call()
    qr = q.float() * 0.01
    kflat = k.reshape(-1, 1, dim).float() * 0.01
    vflat = v.reshape(-1, 1, dim).float() * 0.1
    for nq, nk in [
        (12, 8193),
        (32, 8192),
        (33, 8193),
        (1, 8191),
        (0, 8193),
        (12, 8193),
    ]:
        cu[1].fill_(nq)
        used[0].fill_(nk)
        graph.replay()
        torch.cuda.synchronize()
        kr = torch.cat([kflat[:nk], kflat[pages * page : pages * page + 1153]])
        vr = torch.cat([vflat[:nk], vflat[pages * page : pages * page + 1153]])
        expected, expected_lse = _reference(
            qr, kr, vr, [nq, total - nq], [nk, 1153], True
        )
        gems_assert_close(out.float(), expected, dtype=torch.float32, atol=0.025)
        gems_assert_close(lse, expected_lse, dtype=torch.float32, atol=0.0015)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_short_request_after_prefill(dtype):
    # Exercise a negative temporary output-base adjustment: the split request
    # begins after the large prefill in Q, but at slot 32 in its workspace.
    _run_case(
        [1025, 12],
        [1057, 8193],
        dim=256,
        heads=8,
        kvheads=2,
        paged=True,
        causal=True,
        dtype=dtype,
        broadcast_scales=True,
        strided=True,
        return_lse=True,
    )


def test_non_power_of_two_partitions(monkeypatch):
    import importlib

    launcher = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.flash_attention.launcher"
    )
    original = launcher.flash_varlen_int8_mixed_short
    calls = []

    class Audit:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                calls.append(kwargs["KV_SPLITS"])
                return original[grid](*args, **kwargs)

            return run

    monkeypatch.setattr(launcher, "flash_varlen_int8_mixed_short", Audit())
    # The 128 MiB workspace budget permits 15 partitions. Verify masked merge
    # lanes and empty batch entries without relying on a power-of-two split.
    _run_case(
        [12] + [0] * 14 + [1025],
        [8193] + [0] * 14 + [1057],
        dim=256,
        heads=16,
        kvheads=2,
        paged=True,
        causal=True,
        dtype=torch.float16,
        broadcast_scales=True,
        return_lse=True,
    )
    # _run_case checks the LSE-returning call and the out-only call.
    assert calls == [15, 15]
