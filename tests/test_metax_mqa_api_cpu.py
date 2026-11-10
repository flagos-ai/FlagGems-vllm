# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract checks of actual wrapper ASTs; no operator imports."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "src/flaggems_vllm/runtime/backend/_metax/ops"


class Tensor:
    def __init__(self, shape, dtype="bf16", device="gpu0", strides=None, owner=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.ndim = len(shape)
        self.owner = self if owner is None else owner
        current = 1
        default = []
        for dim in reversed(shape):
            default.append(current)
            current *= dim
        self.strides = tuple(reversed(default)) if strides is None else tuple(strides)

    def stride(self, index=None):
        return self.strides if index is None else self.strides[index]

    def is_contiguous(self):
        return self.strides == Tensor(self.shape).strides

    def view(self, *shape):
        return Tensor(shape, self.dtype, self.device, owner=self.owner)

    reshape = view

    def __getitem__(self, index):
        if index != (slice(None), 0):
            raise ValueError(index)
        return Tensor(
            (self.shape[0], *self.shape[2:]), self.dtype, self.device, owner=self.owner
        )


class WrapperContract(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.failure = None
        self.common_result = object()

        def shared(*args, **kwargs):
            self.calls.append(("shared", args, kwargs))
            if self.failure:
                raise self.failure

        def gathered(*args, **kwargs):
            self.calls.append(("gathered", args, kwargs))
            if self.failure:
                raise self.failure
            return args[5]

        def paged(*args, **kwargs):
            self.calls.append(("paged", args, kwargs))
            if self.failure:
                raise self.failure
            return kwargs["out"] if kwargs["out"] is not None else self.common_result

        def common(*args):
            self.calls.append(("common", args, {}))
            if self.failure:
                raise self.failure
            return self.common_result

        torch = SimpleNamespace(
            bfloat16="bf16",
            float8_e4m3fn="fp8",
            float32="fp32",
            uint8="u8",
            int32="i32",
            int64="i64",
            empty=lambda shape, dtype, device: Tensor(shape, dtype, device),
        )
        namespace = {
            "torch": torch,
            "logger": SimpleNamespace(debug=lambda *a: None),
            "_mqa_logits_into": shared,
            "_batched_mqa_logits": gathered,
            "_paged_indexer_mqa_logits_match": paged,
            "_common_paged_mqa_logits": common,
        }
        for filename, name in (
            ("mqa_logits.py", "mqa_logits"),
            ("paged_mqa_logits.py", "paged_mqa_logits"),
        ):
            parsed = ast.parse((OPS / filename).read_text())
            nodes = [
                n
                for n in parsed.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            ]
            self.assertEqual([n.name for n in nodes], [name])
            code = compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec")
            exec(code, namespace)
        self.mqa = namespace["mqa_logits"]
        self.paged = namespace["paged_mqa_logits"]
        self.q = Tensor((3, 64, 128))
        self.w = Tensor((3, 64), "fp32")
        self.starts = Tensor((3,), "i32")
        self.ends = Tensor((3,), "i32")

    def gathered_args(self):
        return (
            self.q,
            (Tensor((3, 256, 128), "fp8"), Tensor((3, 256), "fp32")),
            self.w,
            None,
            self.ends,
        )

    def paged_args(self):
        return (
            self.q,
            Tensor((32, 64, 1, 132), "u8"),
            self.w,
            self.ends,
            Tensor((3, 4), "i32"),
            object(),
            256,
            True,
        )

    def test_shared_padding_alias_and_row_shard(self):
        kv = Tensor((37, 128))
        out = Tensor((5, 37), "fp32", strides=(44, 1))
        result = self.mqa(
            self.q, kv, self.w, self.starts, self.ends, out=out, row_shard=True
        )
        self.assertIs(result, out)
        self.assertEqual(
            self.calls,
            [
                (
                    "shared",
                    (self.q, kv, self.w, self.starts, self.ends, out),
                    {"row_shard": True},
                )
            ],
        )

    def test_shared_default_output(self):
        result = self.mqa(self.q, Tensor((17, 128)), self.w, self.starts, self.ends)
        self.assertEqual((result.shape, result.dtype), ((3, 17), "fp32"))
        self.assertIs(self.calls[0][1][5], result)

    def test_shared_requires_explicit_bounds(self):
        with self.assertRaises(ValueError):
            self.mqa(self.q, Tensor((17, 128)), self.w, None, self.ends)
        self.assertEqual(self.calls, [])

    def test_gathered_views_preserve_storage_and_end_bounds(self):
        args = self.gathered_args()
        out = Tensor((3, 256), "fp32")
        self.assertIs(self.mqa(*args, out=out), out)
        kind, inner, kwargs = self.calls[0]
        self.assertEqual(kind, "gathered")
        self.assertEqual(kwargs, {})
        self.assertEqual(inner[1].shape, (768, 128))
        self.assertIs(inner[1].owner, args[1][0])
        self.assertIs(inner[2].owner, args[1][1])
        self.assertIs(inner[4], self.ends)
        self.assertEqual(inner[6], 256)

    def test_gathered_nonzero_start_not_silently_ignored(self):
        args = list(self.gathered_args())
        args[3] = self.starts
        with self.assertRaises(ValueError):
            self.mqa(*args)
        self.assertEqual(self.calls, [])

    def test_gathered_rejects_row_shard_before_launch(self):
        with self.assertRaises(ValueError):
            self.mqa(*self.gathered_args(), row_shard=True)
        self.assertEqual(self.calls, [])

    def test_gathered_rejects_non_fp8_keys(self):
        args = list(self.gathered_args())
        args[1][0].dtype = "bf16"
        with self.assertRaises(ValueError):
            self.mqa(*args)
        self.assertEqual(self.calls, [])

    def test_fp8_query_is_not_misrouted_to_bf16(self):
        args = list(self.gathered_args())
        args[0].dtype = "fp8"
        with self.assertRaises(ValueError):
            self.mqa(*args)
        self.assertEqual(self.calls, [])

    def test_paged_fp8_tuple_forwards_unchanged(self):
        args = list(self.paged_args())
        args[0] = (Tensor((3, 1, 64, 128), "fp8"), None)
        args[-1] = False
        self.assertIs(self.paged(*args), self.common_result)
        self.assertEqual(self.calls, [("common", tuple(args), {})])

    def test_paged_fp8_out_rejected_before_launch(self):
        args = list(self.paged_args())
        args[0] = (Tensor((3, 1, 64, 128), "fp8"), None)
        with self.assertRaises(ValueError):
            self.paged(*args, out=Tensor((3, 256), "fp32"))
        self.assertEqual(self.calls, [])

    def test_paged_bf16_alias_and_rank4_view(self):
        args = list(self.paged_args())
        args[0] = Tensor((3, 1, 64, 128))
        out = Tensor((3, 256), "fp32", strides=(260, 1))
        self.assertIs(self.paged(*args, out=out), out)
        kind, inner, kwargs = self.calls[0]
        self.assertEqual(kind, "paged")
        self.assertIs(inner[0].owner, args[0])
        self.assertIs(inner[1], args[1])
        self.assertEqual(kwargs, {"out": out, "num_slots": 256})

    def test_paged_clean_false_rejected_before_launch(self):
        args = list(self.paged_args())
        args[-1] = False
        with self.assertRaises(ValueError):
            self.paged(*args)
        self.assertEqual(self.calls, [])

    def test_paged_multi_query_rejected_before_launch(self):
        args = list(self.paged_args())
        args[0] = Tensor((3, 2, 64, 128))
        with self.assertRaises(ValueError):
            self.paged(*args)
        self.assertEqual(self.calls, [])

    def test_no_runtime_retry(self):
        self.failure = RuntimeError("device error sentinel")
        with self.assertRaisesRegex(RuntimeError, "device error sentinel"):
            self.paged(*self.paged_args())
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][0], "paged")


if __name__ == "__main__":
    unittest.main()
