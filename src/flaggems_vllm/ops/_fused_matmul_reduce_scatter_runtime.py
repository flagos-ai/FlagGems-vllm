# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Process-group-scoped FlagCX resources, without TLE's global communicator.

No init_process_group/destroy_process_group calls are made here. Store traffic
bootstraps handles and validates metadata; it never computes tensor results.
"""

from __future__ import annotations

import json
import socket
from dataclasses import asdict

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

_SEQUENCES = {}


def _validate_topology(records):
    """Require equal, node-contiguous group membership and identical workloads."""
    signature = records[0]["signature"]
    if any(record["signature"] != signature for record in records):
        raise ValueError(
            "all ranks must use identical shapes, dtype and launch configuration"
        )
    hosts = [record["host"] for record in records]
    node_order = list(dict.fromkeys(hosts))
    local_size = hosts.count(hosts[0])
    if hosts != [host for host in node_order for _ in range(local_size)]:
        raise NotImplementedError(
            "group ranks must form equal contiguous blocks per node"
        )
    for host in node_order:
        devices = [record["device"] for record in records if record["host"] == host]
        if len(set(devices)) != len(devices):
            raise ValueError("each group rank must use a distinct GPU on its node")
    return local_size


class _FlagCXAllocation:
    """Own a FlagCX allocation exposed as a zero-copy CUDA byte tensor.

    CUDA array interface construction and dtype views only wrap existing memory;
    neither launches computation nor copies data. The explicit owner stays alive
    until all kernels and registered-window users have completed.
    """

    def __init__(self, library, nbytes, device):
        self.library = library
        self.device = device
        self.ptr = library.flagcxMemAlloc(nbytes)
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "strides": None,
            "typestr": "|u1",
            "data": (self.ptr.value, False),
            "version": 3,
        }

    def release(self):
        if self.ptr is not None:
            with torch.cuda.device(self.device):
                self.library.flagcxMemFree(self.ptr)
            self.ptr = None


class CollectiveContext:
    def __init__(self, group, device, dtype, m, n, k, config, kernels):
        from triton.experimental.tle.language import communication
        from triton.experimental.tle.language.flagcx_wrapper import (
            FLAGCX_WIN_COLL_SYMMETRIC,
            flagcxDevCommRequirements,
        )
        from triton.runtime import DistributedRtContext

        if not communication.enabled:
            raise NotImplementedError(
                "FlagTree requires a matching FlagCX device runtime"
            )
        self.library = communication.flagcx
        self.group = group
        self.device = device
        self.comm = self.window = self.dev_comm = self.dev_mem = None
        self.dev_comm_ptr = self.dev_mem_ptr = None
        self.allocation = self.rs = None
        self.used = False

        sequence = _SEQUENCES.get(group, 0)
        _SEQUENCES[group] = sequence + 1
        store = dist.PrefixStore(
            f"flaggems-gemm-rs/{sequence}/", c10d._get_process_group_store(group)
        )
        properties = torch.cuda.get_device_properties(device)
        driver = kernels.triton.runtime.driver.active
        target = driver.get_current_target()
        driver_properties = driver.utils.get_device_properties(device.index)
        shared_limit = driver_properties["max_shared_mem"]
        signature = [
            m,
            n,
            k,
            str(dtype),
            asdict(config),
            properties.multi_processor_count,
            str(target),
            shared_limit,
        ]
        record = {
            "signature": signature,
            "host": socket.gethostname(),
            "device": str(getattr(properties, "uuid", device.index)),
        }
        store.set(f"rank/{group.rank()}", json.dumps(record))
        records = [
            json.loads(store.get(f"rank/{rank}")) for rank in range(group.size())
        ]
        local_size = _validate_topology(records)
        nnodes = group.size() // local_size
        if (
            nnodes > 1
            and kernels.triton.next_power_of_2(m // group.size() * n)
            > kernels.tl.TRITON_MAX_TENSOR_NUMEL
        ):
            raise NotImplementedError(
                "node transfer exceeds the TLE tensor element limit"
            )
        reserved_sms = config.num_scatter_sms
        if nnodes > 1:
            reserved_sms += 1 + config.num_reduction_sms
        self.num_gemm_sms = properties.multi_processor_count - reserved_sms
        if self.num_gemm_sms < 1:
            raise ValueError("reduce-scatter SM reservation leaves no SM for GEMM")
        if config.estimated_shared_memory() > shared_limit:
            raise NotImplementedError(
                "configured GEMM exceeds the shared-memory budget"
            )

        if group.rank() == 0:
            unique_id = self.library.flagcxGetUniqueId()
            store.set("unique-id", bytes(unique_id.internal))
        unique_id = self.library.unique_id_from_bytes(store.get("unique-id"))
        try:
            self.comm = self.library.flagcxCommInitRank(
                group.size(), unique_id, group.rank()
            )
            # Only the communication window must use the FlagCX allocator.
            per_node_rows = m // local_size
            comm_elements = m * n + 2 * per_node_rows * n
            self.allocation = _FlagCXAllocation(self.library, comm_elements * 2, device)
            comm_buf = torch.as_tensor(self.allocation, device=device).view(dtype)
            if comm_buf.data_ptr() != self.allocation.ptr.value:
                raise RuntimeError(
                    "FlagCX communication buffer must be a zero-copy view"
                )
            self.window = self.library.flagcxCommWindowRegister(
                self.comm,
                comm_buf.data_ptr(),
                comm_buf.numel() * comm_buf.element_size(),
                flags=FLAGCX_WIN_COLL_SYMMETRIC,
            )
            requirements = flagcxDevCommRequirements()
            requirements.barrierCount = 8
            requirements.intraBarrierCount = 8
            requirements.interBarrierCount = 8
            requirements.interContextCount = 4
            requirements.interSignalCount = 8
            requirements.interCounterCount = 8
            self.dev_comm = self.library.flagcxDevCommCreate(self.comm, requirements)
            self.dev_comm_ptr = self.library.flagcxDevCommGetDevicePtr(self.dev_comm)
            self.dev_mem = self.library.flagcxDevMemCreate(
                self.comm,
                comm_buf.data_ptr(),
                comm_buf.numel() * comm_buf.element_size(),
                self.window,
            )
            self.dev_mem_ptr = self.library.flagcxDevMemGetDevicePtr(self.dev_mem)
            # Follow FlagTree communication.create_dist_tensor's handle convention.
            distributed = DistributedRtContext(
                self.dev_mem_ptr.value, self.dev_comm_ptr.value
            )
            distributed.register_buffer(comm_buf)
            self.gemm_out = torch.empty((m, n), dtype=dtype, device=device)
            signals = torch.zeros((group.size(),), dtype=torch.int32, device=device)
            self.counters = torch.zeros_like(signals)
            self.rs_stream = torch.cuda.Stream(device=device, priority=-1)
            reduction_stream = torch.cuda.Stream(device=device, priority=-1)
            self.rs = kernels.TleReduceScatter2DContext(
                max_M=m,
                N=n,
                rank=group.rank(),
                world_size=group.size(),
                local_world_size=local_size,
                dtype=dtype,
                comm_buf=comm_buf,
                signal_buf=signals,
                dist_ctx=distributed,
                reduction_stream=reduction_stream,
                num_p2p_sms=1,
                num_reduction_sms=config.num_reduction_sms,
                num_scatter_sms=config.num_scatter_sms,
            )
            self.done = torch.cuda.Event()
            # All handles and initialized flags must exist before any kernel starts.
            torch.cuda.synchronize(device)
            dist.barrier(group=group)
        except Exception:
            self._release()
            raise

    def _release(self):
        if self.dev_mem_ptr is not None:
            self.library.flagcxDevMemFreeDevicePtr(self.dev_mem)
            self.dev_mem_ptr = None
        if self.dev_comm_ptr is not None:
            self.library.flagcxDevCommFreeDevicePtr(self.dev_comm)
            self.dev_comm_ptr = None
        if self.dev_mem is not None:
            self.library.flagcxDevMemDestroy(self.comm, self.dev_mem)
            self.dev_mem = None
        if self.dev_comm is not None:
            self.library.flagcxDevCommDestroy(self.comm, self.dev_comm)
            self.dev_comm = None
        if self.window is not None:
            self.library.flagcxCommWindowDeregister(self.comm, self.window)
            self.window = None
        if self.comm is not None:
            self.library.flagcxCommDestroy(self.comm)
            self.comm = None
        self.rs = None
        if self.allocation is not None:
            self.allocation.release()
            self.allocation = None

    def close(self):
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "clear collective resources outside CUDA graph capture"
                )
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.group)
            self._release()
