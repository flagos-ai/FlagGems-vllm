# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded kernel tuning with KV-length reuse and fixed Split-KV plans."""

import hashlib
from functools import lru_cache
from pathlib import Path

from flaggems_vllm.utils.libentry import LibEntry

WORKSPACE_LIMIT = 64 * 1024 * 1024
KV_LENGTH_BUCKET = 1024


def prune_configs(configs, arguments, **kwargs):
    arguments = {**arguments, **kwargs}
    groups = arguments["HQ"] // arguments["HK"]
    packed = arguments["PACKED"]
    result = []
    for config in configs:
        meta = config.kwargs
        bm, bn = meta["BM"], meta["BN"]
        if packed and bm < groups:
            continue
        if packed and arguments.get("SPLITS") is not None and bm % groups:
            continue
        if arguments.get("SPLITS") is None and "ORDER" not in meta:
            if (packed and bm > 32) or (not packed and (bm < 32 or bn > 64)):
                continue
        if "ORDER" in meta and (bn > arguments["PAGE"] or arguments["PAGE"] % bn):
            continue
        result.append(config)
    if not result:
        raise RuntimeError("No legal Hygon attention tuning configurations")
    return result


def kv_bucket(length):
    # Sixteen 64-token tiles share a tuning result. Actual sequence lengths
    # remain kernel inputs; this bound only identifies the selected config.
    return (length + KV_LENGTH_BUCKET - 1) // KV_LENGTH_BUCKET * KV_LENGTH_BUCKET


def tuning_strategy(keys):
    return [kv_bucket if key == "MAX_K" else "default" for key in keys]


def select_splits(arguments, plan):
    """Use the validated plan, bounded by the FP32 partial-output workspace."""
    q, _, _, _, _, cuq = arguments[:6]
    total_q = 128 * (cuq.numel() - 1) if plan.family == "hybrid" else q.shape[0]
    available = WORKSPACE_LIMIT // (total_q * q.shape[1] * (q.shape[2] + 1) * 4)
    if available < 1:
        raise RuntimeError("Split-KV exceeds the workspace limit even with one split")
    return min(plan.splits, 1 << (available.bit_length() - 1))


class AttentionEntry(LibEntry):
    """Match LibEntry's compiled dispatch cache to LibTuner's length buckets."""

    def key(self, spec_args, dns_args, const_args):
        # The four kernels mark only MAX_Q/MAX_K as do_not_specialize. Keep
        # MAX_Q exact and group MAX_K without changing either runtime value.
        max_q, max_k = dns_args
        return super().key(spec_args, dns_args, const_args) + (
            max_q,
            kv_bucket(max_k),
        )


@lru_cache(maxsize=1)
def source_version():
    digest = hashlib.sha256()
    directory = Path(__file__).parent
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update((directory.parent.parent / "tune_configs.yaml").read_bytes())
    return digest.hexdigest()[:24]
