# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Collective preparation of peer-addressable storage for device-side MoE.

No method here participates in Graph replay. The kernel protocol, not a host
collective, owns per-invocation publication, consumption and slot reuse.
"""

import math

import torch
import torch.distributed as dist


class SymmetricArena:
    _active = None

    def __init__(self, *, group, rendezvous, heap_bytes):
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.device = torch.npu.current_device()
        import shmem

        self.group, self.runtime = group, shmem
        self.heap_bytes, self.used = heap_bytes, 16 << 20
        self.allocations = []
        self.closed = True
        dist.barrier(group=group)
        shmem.set_conf_store_tls(False, "")
        attr = shmem.InitAttr()
        attr.my_rank, attr.n_ranks = self.rank, self.world
        attr.local_mem_size = heap_bytes
        attr.ip_port = rendezvous
        attr.option_attr.data_op_engine_type = shmem.OpEngineType.MTE
        status = shmem.aclshmem_init(attr)
        if status != 0:
            raise RuntimeError(f"SHMEM initialization failed: {status}")
        self.closed = False
        SymmetricArena._active = self

    def allocate(self, shape, dtype):
        if self.closed:
            raise RuntimeError("SHMEM arena is closed")
        if any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("positive allocation dimensions required")
        width = {torch.bfloat16: 2, torch.float32: 4, torch.int32: 4, torch.int64: 8}[
            dtype
        ]
        size = math.prod(shape) * width
        padded = (size + 511) // 512 * 512
        if self.used + padded > self.heap_bytes:
            raise ValueError("SHMEM allocation exceeds prepared heap budget")
        local = self.runtime.aclshmem_create_tensor(
            list(shape), dtype=dtype, device_id=self.device
        )
        peers = []
        self.allocations.append((local, peers))
        self.used += padded
        for rank in range(self.world):
            address = self.runtime.aclshmem_ptr(local.data_ptr(), rank)
            if not address:
                raise RuntimeError(f"no SHMEM mapping for peer {rank}")
            peers.append(
                self.runtime.construct_tensor_from_ptr(
                    address, list(shape), dtype, local.device
                )
            )
        return local, tuple(peers)

    def close(self):
        if self.closed:
            return
        # Caller must destroy captured graphs before collectively closing.
        torch.npu.synchronize()
        dist.barrier(group=self.group)
        for local, peers in reversed(self.allocations):
            peers.clear()
            self.runtime.aclshmem_free_tensor(local)
        self.allocations.clear()
        self.runtime.aclshmem_finalize()
        self.closed = True
        SymmetricArena._active = None
