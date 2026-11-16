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

"""Backend specializations for extension operators.

Mirrors the layout of ``flaggems_vllm.runtime.backend`` but stays outside it on
purpose: ``runtime.backend._<vendor>/`` treats every non-underscore
subdirectory as an architecture, so extension overrides live here instead of
polluting architecture detection.

Adding a backend:

1. Create ``backends/_<vendor>/__init__.py``.
2. Add ``backends/_<vendor>/ops/<op>.py`` with
   ``@register_backend_impl("<op>", vendor="<vendor>")``.
3. For a single micro-architecture, add
   ``backends/_<vendor>/<arch>/ops/<op>.py`` and pass ``arch="<arch>"``.

A vendor directory is optional; without one, its devices use the generic
implementation in ``extensions/ops/``.
"""
