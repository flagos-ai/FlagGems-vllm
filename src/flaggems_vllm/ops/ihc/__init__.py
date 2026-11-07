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

"""iHC (independent Hyper-Connections) operators for HY V4.

Unlike mHC, iHC has no Sinkhorn and no cross-stream mixing matrix: each
residual channel has a single scalar sigmoid gate. See ihc_pre / ihc_post.
"""

from flaggems_vllm.ops.ihc.ihc_post import ihc_post, ihc_post_ref
from flaggems_vllm.ops.ihc.ihc_pre import ihc_pre, ihc_pre_ref

__all__ = [
    "ihc_post",
    "ihc_post_ref",
    "ihc_pre",
    "ihc_pre_ref",
]
