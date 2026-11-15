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

import torch


def get_device_core_counts(device):
    """Read physical Cube/Vector counts for an explicit NPU device.

    These are hardware properties, not per-stream runtime resource limits.
    Do not change the current device, cache a device-dependent default, or
    guess another model's core count when the runtime cannot report it.
    """
    try:
        properties = torch.npu.get_device_properties(device)
        num_cube_cores = int(properties.cube_core_num)
        num_vector_cores = int(properties.vector_core_num)
        if num_cube_cores <= 0 or num_vector_cores <= 0:
            raise ValueError("core counts must be positive")
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            f"Cannot query physical NPU core counts for {device}"
        ) from error
    return num_cube_cores, num_vector_cores
