# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from flaggems_vllm.runtime.backend._thead.ops.flash_attention.launcher import (
    launch_attention,
)

__all__ = ["launch_attention"]
