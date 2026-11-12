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

"""Public Hopper exports for the TLE MegaGate operator."""

from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megagate.api import (
    PROFILES,
    MegaGateContext,
    MegaGateContextError,
    MegaGateProfile,
    UnpreparedShapeError,
    create_megagate_context,
    get_profile,
    megagate,
)

__all__ = [
    "MegaGateContext",
    "MegaGateContextError",
    "MegaGateProfile",
    "PROFILES",
    "UnpreparedShapeError",
    "create_megagate_context",
    "get_profile",
    "megagate",
]
