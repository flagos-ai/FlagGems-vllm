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

import triton
from triton.compiler.compiler import make_backend

try:
    import triton.experimental.tle as tle_api
except ImportError:
    tle_api = None


def supports_tle_primitives(features: tuple[str, ...]) -> bool:
    if tle_api is None:
        return False
    try:
        if hasattr(tle_api, "is_tle_supported"):
            return tle_api.is_tle_supported(features)
        if hasattr(tle_api, "get_supported_primitives"):
            target = triton.runtime.driver.active.get_current_target()
            backend = type(make_backend(target)).__module__.split(".")[-2]
            supported = tle_api.get_supported_primitives(backend)
            return all(feature in supported for feature in features)
    except ValueError:
        # Older FlagTree versions can reject an unrecognized primitive name.
        return False
    return False
