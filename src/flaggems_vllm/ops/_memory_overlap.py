# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Conservative byte spans for metadata-only overlap validation."""


def tensor_ranges_overlap(left, right):
    if left.device != right.device or not left.numel() or not right.numel():
        return False

    def bounds(tensor):
        start = tensor.data_ptr()
        low = high = 0
        for size, stride in zip(tensor.shape, tensor.stride()):
            extent = (size - 1) * stride
            low += min(extent, 0)
            high += max(extent, 0)
        size = tensor.element_size()
        return start + low * size, start + (high + 1) * size

    left_start, left_end = bounds(left)
    right_start, right_end = bounds(right)
    return left_start < right_end and right_start < left_end
