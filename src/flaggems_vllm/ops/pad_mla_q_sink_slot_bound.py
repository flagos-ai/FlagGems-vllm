# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public contracts; backend implementations are selected by the runtime registry."""


def pad_mla_q_sink_slot_bound(q, sink, indices, out_q, out_sink, out_bound):
    raise NotImplementedError("this API requires the registered NVIDIA backend")


__all__ = ["pad_mla_q_sink_slot_bound"]
