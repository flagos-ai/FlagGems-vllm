# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit dense callback contract, exercised with CPU tensors and a launch spy."""
import importlib
from unittest.mock import Mock

import pytest
import torch

mla = importlib.import_module("flaggems_vllm.ops.flash_mla_with_kvcache")


def inputs(interleaved=False):
    q = torch.empty(2, 1, 12, 576)
    cache = (
        torch.empty(4, 2, 768, 1, 576)[:, 0]
        if interleaved
        else torch.empty(4, 768, 1, 576)
    )
    table = torch.empty(2, 4, dtype=torch.int32)
    lengths = torch.empty(2, dtype=torch.int32)
    out = torch.empty(2, 1, 12, 512)
    lse = torch.empty(2, 12, 1)
    return (q, cache, table, lengths, out, lse, 2, 1, 12, 576, 512, 768, 0.5, False)


def test_dense_callback_receives_interleaved_cache_without_copy():
    args = inputs(interleaved=True)
    observed = []

    def backend(q, cache, table, lengths, out, lse, *, softmax_scale, causal):
        observed.append((q, cache, table, lengths, out, lse, softmax_scale, causal))
        return True

    mla._dense_decode_dispatch(*args, dense_decode_backend=backend)
    assert len(observed) == 1
    assert all(a is b for a, b in zip(observed[0][:6], args[:6]))
    assert observed[0][6:] == (0.5, False)
    assert not args[1].is_contiguous()


@pytest.mark.parametrize("backend", [None, lambda *args, **kwargs: False])
def test_declined_or_missing_callback_uses_native_launch(monkeypatch, backend):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(mla, "_can_use_tle_dense_decode", lambda *args: False)
    monkeypatch.setattr(mla, "_dense_decode_kernel", Kernel())
    args = inputs()
    mla._dense_decode_dispatch(*args, dense_decode_backend=backend)
    assert len(calls) == 1
    grid, values, keywords = calls[0]
    assert grid == (1, 2)
    assert values[0] is args[0]
    assert values[4].data_ptr() == args[1].data_ptr()
    assert keywords == {}


def test_dense_callback_error_is_not_silently_retried():
    backend = Mock(side_effect=ValueError("invalid backend input"))
    with pytest.raises(ValueError, match="invalid backend input"):
        mla._dense_decode_dispatch(*inputs(), dense_decode_backend=backend)
    backend.assert_called_once()
