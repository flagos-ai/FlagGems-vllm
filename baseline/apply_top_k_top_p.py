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

"""Independent vLLM ascending-sort reference with explicit stable ties."""


def apply_top_k_top_p(logits, k=None, p=None):
    if k is None and p is None:
        return logits
    values, indices = logits.sort(dim=-1, stable=True)
    if k is not None:
        threshold = values.gather(1, (values.shape[1] - k.long())[:, None])
        values.masked_fill_(values < threshold, -float("inf"))
    if p is not None:
        cumulative = values.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative <= 1 - p[:, None]
        remove[:, -1] = False
        values.masked_fill_(remove, -float("inf"))
    return logits.scatter_(1, indices, values)
