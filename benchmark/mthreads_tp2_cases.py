# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Original TP2 trace and input construction, for tests/benchmarks only."""

import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path

SOURCE_SHA256 = "918e7d77ff9e86cf9ab117c28d872b5ed62991afe7e296dbde7b5c211e083cb6"
INPUT_POLICY = "tp2-interleaved-v1"
DATA_FILE = (
    Path(__file__).with_name("shape_data") / "qwen36_tp2_flash_attn_cases.json.gz"
)


@lru_cache(maxsize=1)
def _manifest():
    raw = gzip.decompress(DATA_FILE.read_bytes())
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("TP2 source manifest SHA256 mismatch")
    data = json.loads(raw)
    if len(data["cases"]) != 1040:
        raise ValueError("Incomplete TP2 source manifest")
    if len({row["id"] for row in data["cases"]}) != 1040:
        raise ValueError("Duplicate TP2 source case IDs")
    return data


def load_tp2_cases():
    data = _manifest()
    cases = []
    for row in data["cases"]:
        cases.append(
            {
                "case_id": row["id"],
                "tp": "TP2",
                "category": row["source_category"].title(),
                "strict_category": row["category"].title(),
                "q_shape": [row[f"q_dim{i}"] for i in range(3)],
                "k_shape": [row[f"k_dim{i}"] for i in range(4)],
                "v_shape": [row[f"v_dim{i}"] for i in range(4)],
                "cu_seqlens_q": list(row["cu_q"]),
                "seqused_k": list(row["klens"]),
                "block_table_shape": [
                    row["block_table_dim0"],
                    row["block_table_dim1"],
                ],
                "max_seqlen_q": row["max_seqlen_q"],
                "max_seqlen_k": row["max_seqlen_k"],
                "num_splits": row["num_splits"],
                "causal": bool(row["causal"]),
                "input_policy": INPUT_POLICY,
                "input_seed": data["seed"] + row["line"],
                "kv_seed": data["seed"],
                "kv_storage_shape": list(data["kv_storage_shape"]),
                "source_line": row["line"],
                "observed_count": row["count"],
                "source_manifest_sha256": SOURCE_SHA256,
            }
        )
    return cases


class TP2InputFactory:
    """Preserve the source's interleaved KV pool, seeds, table, and out guard.

    A factory reuses one initialized KV pool across TP2 cases, just as the
    source replay does. It does not cache operator outputs or GPU metadata.
    Tensor construction stays outside the timed operator call.
    """

    def __init__(self):
        self.storage = None
        self.output_guard = None
        self._pool_key = None

    def make_inputs(self, case, dtype, device):
        import torch

        if case.get("input_policy") != INPUT_POLICY:
            raise ValueError("Expected an original TP2 replay case")
        if dtype != torch.bfloat16:
            raise ValueError("The frozen TP2 workload is BF16 only")
        pool_key = (
            tuple(case["kv_storage_shape"]),
            case["kv_seed"],
            dtype,
            str(device),
        )
        if self.storage is None:
            torch.manual_seed(case["kv_seed"])
            self.storage = torch.randn(
                *case["kv_storage_shape"], device=device, dtype=dtype
            )
            self._pool_key = pool_key
        elif pool_key != self._pool_key:
            raise ValueError("Cannot change a TP2 factory's frozen KV pool")

        torch.manual_seed(case["input_seed"])
        k, v = self.storage.permute(1, 0, 2, 3, 4).unbind(0)
        q = torch.randn(*case["q_shape"], device=device, dtype=dtype)
        out_storage = torch.full(
            (case["q_shape"][0] + 1, *case["q_shape"][1:]),
            13.0,
            device=device,
            dtype=dtype,
        )
        self.output_guard = out_storage[-1]
        cu = torch.tensor(case["cu_seqlens_q"], device=device, dtype=torch.int32)
        used = torch.tensor(case["seqused_k"], device=device, dtype=torch.int32)
        permutation = torch.randperm(
            case["k_shape"][0],
            generator=torch.Generator(device="cpu").manual_seed(case["input_seed"]),
            device="cpu",
        )
        table = torch.zeros(*case["block_table_shape"], dtype=torch.int32, device="cpu")
        cursor = 0
        page_size = case["k_shape"][1]
        for index, length in enumerate(case["seqused_k"]):
            pages = (length + page_size - 1) // page_size
            table[index, :pages] = permutation[cursor : cursor + pages]
            cursor += pages
        return {
            "q": q,
            "k": k,
            "v": v,
            "out": out_storage[:-1],
            "cu_seqlens_q": cu,
            "seqused_k": used,
            "max_seqlen_q": case["max_seqlen_q"],
            "max_seqlen_k": case["max_seqlen_k"],
            "causal": case["causal"],
            "softmax_scale": case["q_shape"][2] ** -0.5,
            "block_table": table.to(device),
            "num_splits": case["num_splits"],
        }

    def packed_inputs(self, case, dtype, device):
        inputs = self.make_inputs(case, dtype, device)
        return (
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["max_seqlen_q"],
            inputs["cu_seqlens_q"],
            inputs["max_seqlen_k"],
            None,
            inputs["seqused_k"],
            None,
            0.0,
            inputs["softmax_scale"],
            inputs["causal"],
            (-1, -1),
            0.0,
            None,
            False,
            False,
            inputs["block_table"],
            False,
            inputs["out"],
            {"num_splits": inputs["num_splits"]},
        )
