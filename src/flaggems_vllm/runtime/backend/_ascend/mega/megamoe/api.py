# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import logging

logger = logging.getLogger(__name__)


def prepare_megamoe(
    w1,
    w2,
    *,
    group,
    capacity_tokens,
    topk,
    rendezvous="tcp://127.0.0.1:29783",
):
    """Collective setup. Call workspace.compile(x, ids, probs) for each collective input profile.

    Recompile when any rank changes input shape, dtype or strides. Only the
    latest compiled profile is active for eager calls; captured graphs keep
    their own profile. Weights are immutable after preparation. Use one
    in-flight invocation per workspace and destroy its captured graphs before
    collectively closing it.
    """
    logger.debug("GEMS_ASCEND PREPARE_MEGAMOE")
    from .workspace import MegaMoEWorkspace

    return MegaMoEWorkspace(
        w1,
        w2,
        group=group,
        capacity_tokens=capacity_tokens,
        topk=topk,
        rendezvous=rendezvous,
    )


def megamoe(x, ids, probs, workspace):
    """Launch with a compiled workspace and return BF16 output and expert counts.

    All ranks in the workspace's group must call in the same order.
    """
    logger.debug("GEMS_ASCEND MEGAMOE")
    return workspace(x, ids, probs)
