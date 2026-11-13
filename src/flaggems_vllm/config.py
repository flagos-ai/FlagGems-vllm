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

import os
import warnings
from pathlib import Path

has_c_extension = False
use_c_extension = False
aten_patch_list = []

# set FlagGems-vllm_SOURCE_DIR for cpp extension to find
os.environ["FlagGems-vllm_SOURCE_DIR"] = str(Path(__file__).parent.resolve())

try:
    from flaggems_vllm import c_operators  # type: ignore[attr-defined]

    has_c_extension = True
except ImportError:
    c_operators = None
    has_c_extension = False


use_env_c_extension = os.environ.get("USE_C_EXTENSION", "0") == "1"
if use_env_c_extension and not has_c_extension:
    warnings.warn(
        "[FlagGems-vllm] USE_C_EXTENSION is set, but C extension is not available. "
        "Falling back to pure Python implementation.",
        RuntimeWarning,
    )

if has_c_extension and use_env_c_extension:
    try:
        from flaggems_vllm import aten_patch  # type: ignore[attr-defined]

        aten_patch_list = aten_patch.get_registered_ops()
        use_c_extension = True
    except (ImportError, AttributeError):
        aten_patch_list = []
        use_c_extension = False


__all__ = [
    "aten_patch_list",
    "has_c_extension",
    "use_c_extension",
]
