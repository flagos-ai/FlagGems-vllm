# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""CPU-only API contracts; standalone unittest avoids GPU conftest imports.

These tests execute isolated host-function ASTs with metadata/launch doubles.
They do not establish numerical correctness or real device/backend dispatch.
"""
import ast
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/flaggems_vllm"
VENDOR = SRC / "runtime/backend/_metax"


def function(path, name, env):
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env[name]


class Tensor:
    def __init__(self, shape, dtype="fp32", device="cuda:0", strides=None):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim = len(shape)
        self._strides = strides or (shape[1], 1)

    def is_contiguous(self):
        return self._strides == (self.shape[1], 1)

    def contiguous(self):
        return Tensor(self.shape, self.dtype, self.device)

    def stride(self, dim=None):
        return self._strides if dim is None else self._strides[dim]


class ApiContracts(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.env = {
            "torch": SimpleNamespace(float32="fp32", int32="i32", int64="i64"),
            "logger": SimpleNamespace(debug=lambda *a: None),
            "_SHARED_SELECTOR_AVAILABLE": False,
            "_stable_topk_shared": lambda *a, **k: self.calls.append(("shared", a, k)),
            "_decode_topk_tree": lambda *a, **k: self.calls.append(("tree", a, k)),
            "_general_top_k_per_row_decode": lambda *a: self.calls.append(
                ("general", a)
            ),
        }
        self.op = function(
            VENDOR / "ops/top_k_per_row_decode.py", "top_k_per_row_decode", self.env
        )
        self.logits = Tensor((3, 2048))
        self.lengths = Tensor((3,), "i32", strides=(1,))
        self.out = Tensor((3, 512), "i32")
        self.args = (self.logits, 1, self.lengths, self.out, 3, 2048, 1, 512)

    def test_topk_exact_common_signature(self):
        native = function(
            SRC / "ops/top_k_per_row_decode.py", "top_k_per_row_decode", {}
        )
        self.assertEqual(
            str(inspect.signature(native)), str(inspect.signature(self.op))
        )

    def test_tree_mutates_supplied_buffer_and_returns_none(self):
        self.assertIsNone(self.op(*self.args))
        self.assertEqual(
            self.calls, [("tree", (self.logits, self.lengths, self.out), {"k": 512})]
        )

    def test_next_n_general_arguments_unchanged(self):
        args = list(self.args)
        args[1] = 2
        self.assertIsNone(self.op(*args))
        self.assertEqual(self.calls, [("general", tuple(args))])

    def test_other_k_general_arguments_unchanged(self):
        args = list(self.args)
        args[-1] = 256
        self.op(*args)
        self.assertEqual(self.calls, [("general", tuple(args))])

    def test_incompatible_stride_not_sent_to_tree(self):
        args = list(self.args)
        args[-3] = 4096
        self.op(*args)
        self.assertEqual(self.calls, [("general", tuple(args))])

    def test_tree_failure_never_retried(self):
        def fail(*args, **kwargs):
            self.calls.append("failed")
            raise RuntimeError("launch failure")

        self.env["_decode_topk_tree"] = fail
        with self.assertRaisesRegex(RuntimeError, "launch failure"):
            self.op(*self.args)
        self.assertEqual(self.calls, ["failed"])

    def test_shared_selector_bounded_k512(self):
        self.env["_SHARED_SELECTOR_AVAILABLE"] = True
        self.assertIsNone(self.op(*self.args))
        self.assertEqual(
            self.calls,
            [
                (
                    "shared",
                    (self.logits, None, self.lengths, self.out, 3),
                    {"top_k": 512},
                )
            ],
        )

    def test_large_k512_retains_tree(self):
        self.env["_SHARED_SELECTOR_AVAILABLE"] = True
        args = (Tensor((3, 16641)), 1, self.lengths, self.out, 3, 16641, 1, 512)
        self.op(*args)
        self.assertEqual(self.calls[0][0], "tree")

    def test_stability_domain_and_noncontiguous_input(self):
        self.env["_SHARED_SELECTOR_AVAILABLE"] = True
        for k in (1, 63, 64, 511, 512, 513, 1023, 1024):
            with self.subTest(k=k):
                self.calls.clear()
                logits = Tensor((3, 8192), strides=(16384, 2))
                out = Tensor((3, k), "i32")
                self.op(logits, 1, self.lengths, out, 3, 16384, 2, k)
                kind, args, kwargs = self.calls[0]
                self.assertEqual(kind, "shared")
                self.assertEqual(args[0].stride(), (8192, 1))
                self.assertIsNot(args[0], logits)
                self.assertIs(args[3], out)
                self.assertEqual(kwargs, {"top_k": k})

    def test_wider_or_pitched_output_keeps_original_route(self):
        self.env["_SHARED_SELECTOR_AVAILABLE"] = True
        for output, k, route in (
            (Tensor((3, 514), "i32"), 512, "tree"),
            (Tensor((3, 1024), "i32", strides=(2048, 2)), 1024, "general"),
        ):
            with self.subTest(route=route):
                self.calls.clear()
                self.op(self.logits, 1, self.lengths, output, 3, 2048, 1, k)
                self.assertEqual(self.calls[0][0], route)

    def test_shared_failure_never_retried(self):
        self.env["_SHARED_SELECTOR_AVAILABLE"] = True

        def fail(*args, **kwargs):
            self.calls.append("shared_failed")
            raise RuntimeError("shared launch failure")

        self.env["_stable_topk_shared"] = fail
        with self.assertRaisesRegex(RuntimeError, "shared launch failure"):
            self.op(*self.args)
        self.assertEqual(self.calls, ["shared_failed"])

    def test_non_integer_k_keeps_original_route(self):
        self.env["_SHARED_SELECTOR_AVAILABLE"] = True
        for k in (True, 512.0, 0, 1025):
            self.calls.clear()
            self.op(*self.args[:-1], k)
            self.assertEqual(self.calls[0][0], "general")


if __name__ == "__main__":
    unittest.main(verbosity=2)
