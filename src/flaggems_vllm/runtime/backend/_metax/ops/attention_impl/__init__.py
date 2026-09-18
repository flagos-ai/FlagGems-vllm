# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Private MetaX FlashAttention planning and launch package.

The launch configurations are carried over from the validated C550 TN10 source.
C550 uses 64 threads per warp, at most 512 threads and 65,536 bytes of shared
memory per CTA. TN MMA paths rely on the compiler's w4/s4 Async-TN pipeline;
changing warps or stages can remove that pipeline or exceed shared memory.
These paths deliberately keep their compiler/resource-constrained settings
instead of using the generic attention autotuner. This is not a claim that
every inherited specialization is spill-free.
"""
from .launcher import launch_metax_attention
from .scheduling import (
    MetaXAttentionPlan,
    MetaXAttentionScheduler,
    MetaXKernelFamily,
    MetaXTaskMapper,
)

__all__ = [
    "MetaXAttentionPlan",
    "MetaXAttentionScheduler",
    "MetaXKernelFamily",
    "MetaXTaskMapper",
    "launch_metax_attention",
]
