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

"""Top-k/top-p filtering APIs; vendor implementations are selected at runtime."""


def top_k_top_p(
    logits, p=None, k=None, *, topk_len_override=None, p_is_one_override=None
):
    """Return filtered logits without modifying the input.

    The shared argument order is ``(logits, p=None, k=None)``. Optional launch
    hints are backend-specific; unsupported backends raise before execution.
    """
    if p is None and k is None:
        raise ValueError("p and k cannot both be None")
    raise NotImplementedError("top_k_top_p is supported on MetaX and Ascend")
