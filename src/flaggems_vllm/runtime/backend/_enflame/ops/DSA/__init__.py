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

"""DSA (DeepSeek Sparse Attention) operator package.

Explicit package marker: the generic ``flaggems_vllm.ops.DSA`` directory
relies on implicit namespace packaging (no ``__init__.py``), so the seed
copy had no marker to inherit. Declaring the package explicitly keeps this
backend copy consistent with the FLA/ihc/mhc/qwen4 subpackages and avoids
import-path surprises under vendor override loading. Operator wiring lives
in the top-level ``_enflame/ops/__init__.py``.
"""
