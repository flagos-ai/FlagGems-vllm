# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public contracts; backend implementations are selected by the runtime registry."""


def indexer_prefix_topk(logits, lengths, k, out=None):
    raise NotImplementedError("this API requires the registered NVIDIA backend")


def supports_indexer_prefix_topk(logits, lengths, k, out):
    return False


__all__ = ["indexer_prefix_topk", "supports_indexer_prefix_topk"]
