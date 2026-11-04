# SPDX-License-Identifier: Apache-2.0
"""Independent CPU integer oracle for pool expansion (never a perf baseline)."""

import torch


def expand_pool_oracle(pool_ids, seq_lens, pool_size):
    """Construct every output element using Python integers and CPU int64 inputs."""
    pools = pool_ids.detach().to(device="cpu", dtype=torch.int64)
    lengths = seq_lens.detach().to(device="cpu", dtype=torch.int64)
    rows, groups = pools.shape
    out_cols = groups * pool_size + pool_size - 1
    result = []
    for row_pools, length in zip(pools.tolist(), lengths.tolist()):
        for pool in row_pools:
            for offset in range(pool_size):
                result.append(pool * pool_size + offset if pool >= 0 else -1)
        tail_start = (length // pool_size) * pool_size
        tail_count = length % pool_size
        for offset in range(pool_size - 1):
            result.append(tail_start + offset if offset < tail_count else -1)
    return torch.tensor(result, dtype=torch.int32, device="cpu").reshape(rows, out_cols)
