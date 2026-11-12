# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import subprocess
import sys
import textwrap

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.utils import has_triton_tle_attrs

pytestmark = pytest.mark.skipif(
    flaggems_vllm.runtime.device.vendor_name != "thead", reason="T-Head import contract"
)


@pytest.mark.skipif(
    has_triton_tle_attrs(("load",), 3, 6, 0), reason="requires a non-TLE compiler"
)
def test_non_tle_import_does_not_load_aiu_only_modules():
    code = """
        import builtins
        original = builtins.__import__
        optional = (
            'flaggems_vllm.runtime.backend._thead.ops.int8_einsum',
            'flaggems_vllm.runtime.backend._thead.ops.w8a8_block_int8_bmm',
            'flaggems_vllm.runtime.backend._thead.ops.bmm_w8a8_int8',
        )
        def guarded(name, *args, **kwargs):
            if name in optional:
                raise AssertionError('attempted unavailable import: ' + name)
            return original(name, *args, **kwargs)
        builtins.__import__ = guarded
        import flaggems_vllm as library
        from flaggems_vllm.runtime.backend._thead import ops
        assert callable(library.moe_align_block_size_no_tle)
        assert callable(library.moe_align_block_size_triton)
        assert callable(ops.w8a8_block_matmul)
        for name in ('int8_einsum', 'w8a8_block_int8_bmm', 'bmm_w8a8_int8'):
            assert name not in ops.__all__
            assert name not in library.__all__
    """
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], check=True, timeout=120
    )


@pytest.mark.parametrize("broken", [False, True])
def test_optional_exports_preserve_identity_and_import_errors(broken):
    # Isolate dependency availability; no TLE numerical execution is simulated.
    code = """
        import builtins
        import importlib
        import sys
        from types import ModuleType
        import flaggems_vllm
        import flaggems_vllm.utils as utils
        backend = importlib.import_module('flaggems_vllm.runtime.backend._thead.ops')
        public = importlib.import_module('flaggems_vllm.ops')
        utils.has_triton_tle_attrs = lambda *args: True
        functions = {}
        for name in ('int8_einsum', 'w8a8_block_int8_bmm', 'bmm_w8a8_int8'):
            module = ModuleType(backend.__name__ + '.' + name)
            def sentinel(*args, **kwargs):
                raise AssertionError('capability test must not execute kernels')
            functions[name] = sentinel
            setattr(module, name, sentinel)
            sys.modules[module.__name__] = module
        if BROKEN:
            original = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name == backend.__name__ + '.int8_einsum':
                    raise ImportError('optional-kernel-import-failure')
                return original(name, *args, **kwargs)
            builtins.__import__ = guarded
            try:
                importlib.reload(backend)
            except ImportError as error:
                assert str(error) == 'optional-kernel-import-failure'
            else:
                raise AssertionError('import failure was swallowed')
        else:
            importlib.reload(backend)
            importlib.reload(public)
            for name in ('int8_einsum', 'w8a8_block_int8_bmm'):
                assert name in backend.__all__ and name in public.__all__
                assert getattr(backend, name) is functions[name]
                assert getattr(public, name) is functions[name]
    """.replace(
        "BROKEN", repr(broken)
    )
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], check=True, timeout=120
    )


def _assert_alignment(routes, experts, block, result):
    # Independent CPU oracle: every route occurs once in its expert's padded run.
    ids, expert_ids, post_pad = [value.cpu().tolist() for value in result]
    flat = [expert for row in routes for expert in row]
    cursor = 0
    expected_experts = []
    for expert in range(experts):
        expected = [index for index, value in enumerate(flat) if value == expert]
        padded = (len(expected) + block - 1) // block * block
        segment = ids[cursor : cursor + padded]
        assert sorted(value for value in segment if value < len(flat)) == expected
        assert all(value == len(flat) for value in segment if value >= len(flat))
        expected_experts.extend([expert] * (padded // block))
        cursor += padded
    assert post_pad == [cursor]
    assert expert_ids[: cursor // block] == expected_experts


@pytest.mark.parametrize(
    "tokens,topk,experts,block,dtype",
    [
        (0, 2, 7, 16, torch.int32),
        (1, 1, 1, 4, torch.int64),
        (1, 2, 64, 16, torch.int32),
        (9, 2, 7, 4, torch.int64),
        (64, 8, 64, 16, torch.int32),
        (257, 2, 7, 64, torch.int64),
    ],
)
def test_public_non_tle_moe_alignment(tokens, topk, experts, block, dtype):
    routes = [
        [(row * 3 + col * 5) % experts for col in range(topk)] for row in range(tokens)
    ]
    cpu_ids = torch.tensor(routes, dtype=dtype).reshape(tokens, topk)
    storage = torch.empty(cpu_ids.numel() + 1, dtype=dtype, device=flaggems_vllm.device)
    ids = storage[1:].view(tokens, topk)
    ids.copy_(cpu_ids)
    result = flaggems_vllm.moe_align_block_size_no_tle(ids, block, experts)
    _assert_alignment(routes, experts, block, result)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires T-Head graph support"
)
def test_non_tle_moe_graph_replays_changed_routes():
    experts, block, tokens = 7, 16, 64
    routes = [[row % experts, (row + 1) % experts] for row in range(tokens)]
    ids = torch.tensor(routes, dtype=torch.int32, device=flaggems_vllm.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flaggems_vllm.moe_align_block_size_no_tle(ids, block, experts)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = flaggems_vllm.moe_align_block_size_no_tle(ids, block, experts)
    for changed in (True, False, True):
        current = [[0, 0] for _ in range(tokens)] if changed else routes
        ids.copy_(torch.tensor(current, dtype=ids.dtype))
        for output in result:
            output.fill_(-123)
        graph.replay()
        torch.cuda.synchronize()
        _assert_alignment(current, experts, block, result)
