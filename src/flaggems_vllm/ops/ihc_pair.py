# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public contracts; backend implementations are selected by the runtime registry."""


def can_ihc_pair(
    x, weight, scale, base, magnitude, hc_eps, norm_eps, *, residual=None, post=None
):
    return False


def ihc_pre_pair(x, weight, scale, base, magnitude, hc_eps, norm_eps):
    raise NotImplementedError("this API requires the registered NVIDIA backend")


def ihc_post_pre_pair(
    x, residual, post, weight, scale, base, magnitude, hc_eps, norm_eps
):
    raise NotImplementedError("this API requires the registered NVIDIA backend")


__all__ = ["can_ihc_pair", "ihc_pre_pair", "ihc_post_pre_pair"]
