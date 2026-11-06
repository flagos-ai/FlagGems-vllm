# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public contracts; backend implementations are selected by the runtime registry."""


def int8_indexer_paged_logits_empty_tile(
    q,
    cache,
    weights,
    block_table,
    row_batch,
    lengths,
    max_model_len,
    page_size,
    *,
    output_buffer=None,
    pure_prefill=False,
):
    raise NotImplementedError("this API requires the registered NVIDIA backend")


__all__ = ["int8_indexer_paged_logits_empty_tile"]
