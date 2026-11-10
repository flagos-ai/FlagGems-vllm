# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of actual MetaX wrappers, isolated before package imports."""

import ast
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/flaggems_vllm/runtime/backend/_metax/ops/sparse_mla_prefill.py"


class Tensor:
    def __init__(self, shape, dtype="bf16", device="cuda:0", contiguous=True):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device
        self.contiguous = contiguous
        self.source = None

    def is_contiguous(self):
        return self.contiguous

    def unsqueeze(self, dim):
        result = Tensor(
            self.shape[:dim] + (1,) + self.shape[dim:],
            self.dtype,
            self.device,
            self.contiguous,
        )
        result.source = self
        return result


def load_wrapper(calls):
    tree = ast.parse(MODULE.read_text())
    selected = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"flash_mla_sparse_fwd", "_validate_local_prefill"}
    ]
    for node in selected:
        node.returns = None
        for arg in node.args.args + node.args.kwonlyargs:
            arg.annotation = None
    torch = types.SimpleNamespace(bfloat16="bf16", float32="fp32", int32="i32")

    def local(*args):
        calls.append(args)
        return args[6:9]

    ns = {
        "torch": torch,
        "logger": types.SimpleNamespace(debug=lambda *a: None),
        "_sparse_prefill_local": local,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(MODULE), "exec"), ns)
    return ns["flash_mla_sparse_fwd"]


class LocalPrefillContract(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.op = load_wrapper(self.calls)
        self.q = Tensor((17, 16, 512))
        self.kv = Tensor((128, 1, 512))
        self.ids = Tensor((17, 64), "i32")
        self.kw = dict(
            d_v=512,
            attn_sink=Tensor((16,), "fp32"),
            topk_length=Tensor((17,), "i32"),
            out=Tensor((17, 16, 512)),
            max_logits_out=Tensor((17, 16), "fp32"),
            lse_out=Tensor((17, 16), "fp32"),
        )

    def invoke(self):
        return self.op(self.q, self.kv, self.ids, 0.125, **self.kw)

    def assert_rejected(self):
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertEqual(self.calls, [])

    def test_two_dimensional_view_and_output_alias(self):
        result = self.invoke()
        self.assertEqual(len(self.calls), 1)
        args = self.calls[0]
        self.assertIs(args[0], self.q)
        self.assertIs(args[1], self.kv)
        self.assertEqual(args[2].shape, (17, 1, 64))
        self.assertIs(args[2].source, self.ids)
        self.assertEqual(args[3], 0.125)
        self.assertIs(args[4], self.kw["attn_sink"])
        self.assertIs(args[5], self.kw["topk_length"])
        for i, name in enumerate(("out", "max_logits_out", "lse_out")):
            self.assertIs(result[i], self.kw[name])

    def test_576_head_dimension_and_optional_inputs(self):
        self.q = Tensor((17, 16, 576))
        self.kv = Tensor((128, 1, 576))
        self.kw["attn_sink"] = self.kw["topk_length"] = None
        self.invoke()
        self.assertEqual(len(self.calls), 1)

    def test_missing_output_buffers(self):
        for name in ("out", "max_logits_out", "lse_out"):
            with self.subTest(name=name):
                old = self.kw[name]
                self.kw[name] = None
                self.assert_rejected()
                self.kw[name] = old

    def test_each_tensor_metadata_before_launch(self):
        for name in (
            "q",
            "kv",
            "ids",
            "out",
            "max_logits_out",
            "lse_out",
            "attn_sink",
            "topk_length",
        ):
            tensor = (
                getattr(self, name) if name in {"q", "kv", "ids"} else self.kw[name]
            )
            for field, bad in (
                ("dtype", "wrong"),
                ("device", "cuda:1"),
                ("contiguous", False),
            ):
                # Mutating q.device moves the expected device and makes kv disagree.
                with self.subTest(tensor=name, field=field):
                    old = getattr(tensor, field)
                    setattr(tensor, field, bad)
                    self.assert_rejected()
                    setattr(tensor, field, old)

    def test_output_and_optional_shapes(self):
        for name in ("out", "max_logits_out", "lse_out", "attn_sink", "topk_length"):
            with self.subTest(name=name):
                tensor = self.kw[name]
                old = tensor.shape
                tensor.shape = (99,)
                self.assert_rejected()
                tensor.shape = old

    def test_input_shapes(self):
        for name, shape in (
            ("q", (17, 8, 512)),
            ("q", (17, 16, 128)),
            ("q", (0, 16, 512)),
            ("kv", (128, 2, 512)),
            ("kv", (0, 1, 512)),
            ("kv", (128, 1, 576)),
            ("ids", (18, 64)),
            ("ids", (17, 0)),
        ):
            with self.subTest(name=name, shape=shape):
                tensor = getattr(self, name)
                old = tensor.shape
                tensor.shape = shape
                self.assert_rejected()
                tensor.shape = old

    def test_incompatible_semantic_arguments_rejected(self):
        self.kw["d_v"] = 256
        self.assert_rejected()
        self.kw["d_v"] = 512
        self.kw["real_num_heads"] = 8
        self.assert_rejected()

    def test_legacy_public_local_name_is_not_defined(self):
        names = {
            n.name
            for n in ast.parse(MODULE.read_text()).body
            if isinstance(n, ast.FunctionDef)
        }
        self.assertNotIn("sparse_prefill_local", names)
        self.assertIn("_sparse_prefill_local", names)


if __name__ == "__main__":
    unittest.main()
