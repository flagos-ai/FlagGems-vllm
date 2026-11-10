# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Current single-node BF16 MegaMoE workspace.

Preparation and compilation are collective; weights are immutable and a workspace
has one in-flight invocation. Logical dimensions are padded to internal Cube tiles.
Destroy captured graphs before close(). No host collective participates in replay.
"""
import torch
import torch.distributed as dist
import triton

from . import bindings
from .kernel import _queue_pack
from .prepare import _copy, _copy_matrix, _pack_gated_weights, _zero
from .symmetric import SymmetricArena

_RETURN_CHUNK_TOKENS = 1024


class MegaMoEWorkspace:
    def __init__(
        self,
        w1,
        w2,
        *,
        group,
        capacity_tokens,
        topk,
        rendezvous="tcp://127.0.0.1:29783",
    ):
        # Validate the complete preparation contract before allocation or launch.
        if not dist.is_initialized():
            raise RuntimeError("initialize the distributed process group first")
        if not isinstance(w1, (list, tuple)) or not isinstance(w2, (list, tuple)):
            raise TypeError("weights must be lists or tuples of expert tensors")
        if not w1 or len(w1) != len(w2):
            raise ValueError("matching nonempty expert lists required")
        if not all(
            isinstance(w, torch.Tensor) and w.ndim == 2 and w.layout == torch.strided
            for w in (*w1, *w2)
        ):
            raise NotImplementedError("MegaMoE requires rank-two strided weights")
        logical_h, doubled = w1[0].shape
        logical_i, e = doubled // 2, len(w1)
        if logical_h < 1 or logical_i < 1 or doubled % 2:
            raise NotImplementedError("weight dimensions must describe gate/up pairs")
        device = w1[0].device
        if device.type != "npu" or device.index != torch.npu.current_device():
            raise NotImplementedError("weights must be on the current NPU")
        for values, shape in ((w1, (logical_h, doubled)), (w2, (logical_i, logical_h))):
            if not all(
                tuple(w.shape) == shape
                and w.dtype == torch.bfloat16
                and w.device == device
                and not w.requires_grad
                for w in values
            ):
                raise NotImplementedError(
                    "MegaMoE requires matching BF16 inference weights on the current NPU"
                )
        self.world, self.rank = dist.get_world_size(group), dist.get_rank(group)
        if not 1 <= self.world <= 8:
            raise NotImplementedError("single-node 910B supports at most eight ranks")
        if type(capacity_tokens) is not int or capacity_tokens < 1:
            raise ValueError("capacity_tokens must be a positive integer")
        if type(topk) is not int or not 1 <= topk <= e * self.world:
            raise ValueError("topk must be between one and the global expert count")
        if not isinstance(rendezvous, str) or not rendezvous.startswith("tcp://"):
            raise ValueError("rendezvous must be a tcp:// address")
        if SymmetricArena._active is not None:
            raise RuntimeError("close the previous workspace first")
        h = triton.cdiv(logical_h, 256) * 256
        i = triton.cdiv(logical_i, 128) * 128
        c = triton.cdiv(capacity_tokens, 32) * 32
        # Route IDs and task counts remain int32; tensor offsets are int64.
        # A flattened buffer may therefore contain more than 2**31 elements.
        if self.world * (c * topk + e * 128) >= 2**31:
            raise NotImplementedError("routing and task counts must fit in int32")
        kp = max(8, triton.next_power_of_2(topk))
        dispatch_rows = 8 if h <= 4096 else 1
        prefix = 5 * (triton.cdiv(e, 8) * 8) + 32 + 64 * kp
        tables = 32 * e * (self.world + 1)
        payload = dispatch_rows * h // 2 + 8 + dispatch_rows * 12 * topk + e
        scratch_words = triton.next_power_of_2(prefix + max(tables, payload))
        if scratch_words * 4 > 128 * 1024:
            raise NotImplementedError("dispatch metadata exceeds UB budget")
        geometry = (logical_h, logical_i, e, topk, capacity_tokens)
        geometries = [None] * self.world
        dist.all_gather_object(geometries, geometry, group=group)
        if any(value != geometry for value in geometries):
            raise ValueError("rank geometry mismatch")
        self.logical_h = logical_h
        self.dispatch_aivs = {2: 8, 4: 24}.get(self.world, 2 * self.world)
        self.max_tokens = capacity_tokens
        self.compiled = None
        self.extent = None
        self.capacity = c
        self.h, self.i, self.e, self.k = h, i, e, topk
        self.kp = kp
        self.group = group
        # Limit UB input staging without reviving a second dispatch algorithm.
        self.dispatch_rows = dispatch_rows
        self.total_groups = 20
        self.compute = self.total_groups - 1
        self.aiv_count = 2 * self.total_groups
        self.w1 = torch.empty((e, h, 2 * i), dtype=torch.bfloat16, device=device)
        self.w2 = torch.empty((e, i, h), dtype=torch.bfloat16, device=device)
        for values, packed in ((w1, self.w1), (w2, self.w2)):
            for index, value in enumerate(values):
                if value.is_contiguous() and tuple(value.shape) == tuple(
                    packed[index].shape
                ):
                    _copy[(triton.cdiv(value.numel(), 4096),)](
                        value, packed[index], value.numel(), 4096
                    )
                else:
                    _copy_matrix[(triton.cdiv(packed[index].numel(), 4096),)](
                        value,
                        packed[index],
                        *value.shape,
                        *packed[index].shape,
                        *value.stride(),
                        GATED=values is w1,
                    )
        c, p, kp = self.capacity, self.world, self.kp
        count_stride = triton.cdiv(e, 8) * 8
        self.return_chunks = triton.cdiv(c, _RETURN_CHUNK_TOKENS)
        symm_bytes = (
            c * h * 2
            + p * c * (4 + kp * 8)
            + p * 16 * 4
            + c * self.world * h * 4
            + 3 * p * 16 * 8
            + p * (count_stride + self.return_chunks) * 4
        )
        heap = 1 << max(26, (symm_bytes + (64 << 20) - 1).bit_length())
        self.arena = SymmetricArena(group=group, rendezvous=rendezvous, heap_bytes=heap)
        try:
            # The arena owns the local tensors behind these per-rank views.
            _, self.peer_x = self.arena.allocate((c * h,), torch.bfloat16)
            _, self.peer_tokens = self.arena.allocate((p * c,), torch.int32)
            _, self.peer_experts = self.arena.allocate((p * c * kp,), torch.int32)
            _, self.peer_weights = self.arena.allocate((p * c * kp,), torch.float32)
            _, self.peer_counts = self.arena.allocate((p * 16,), torch.int32)
            self.returned, self.peer_returns = self.arena.allocate(
                (c * self.world * h,), torch.float32
            )
            self.flags, self.peer_flags = self.arena.allocate(
                (3 * p * 16,), torch.int64
            )
            _, self.peer_expert_counts = self.arena.allocate(
                (p * count_stride,), torch.int32
            )
            _, self.peer_chunk_counts = self.arena.allocate(
                (p * self.return_chunks,), torch.int32
            )
            rows = p * (c * topk + e * 128)
            self.merged_packed = torch.empty(
                (rows, h), dtype=torch.bfloat16, device=device
            )
            self.merged_headers = torch.empty(
                (rows, 8), dtype=torch.int32, device=device
            )
            self.ready = torch.empty((e * p * 16,), dtype=torch.int64, device=device)
            self.merged_ready = torch.empty((e * 16,), dtype=torch.int64, device=device)
            self.ring = torch.empty(
                (self.compute, 6, 16384), dtype=torch.float32, device=device
            )
            self.activation = torch.empty(
                (self.compute, 128, i), dtype=torch.bfloat16, device=device
            )
            self.local_slots = torch.empty(
                (p * c * topk * h,), dtype=torch.float32, device=device
            )
            control_shape = (self.aiv_count * 16,)
            self.send_control = torch.empty(
                control_shape, dtype=torch.int64, device=device
            )
            limit = p * c * topk // 32 + e
            self.queue_flags = torch.empty(
                (limit * 16,), dtype=torch.int64, device=device
            )
            self.task_meta = torch.empty((16,), dtype=torch.int64, device=device)
            self.meta_sources = torch.empty((p * 16,), dtype=torch.int64, device=device)
            self.progress = torch.empty(
                (p * self.return_chunks * self.compute * 2 * 16,),
                dtype=torch.int64,
                device=device,
            )
            self.output_requests = torch.empty(
                control_shape, dtype=torch.int64, device=device
            )
            self.output_assignments = torch.empty(
                control_shape, dtype=torch.int64, device=device
            )
            self.w1_fused = torch.empty_like(self.w1)
            _pack_gated_weights[(self.aiv_count,)](
                self.w1, self.w1_fused, self.w1.numel(), i, 128
            )
            self.scratch_words = scratch_words
            self.errors = torch.empty(
                ((self.compute + 2 * p) * 16,), dtype=torch.int64, device=device
            )
            self.control = torch.empty(
                (self.compute, 3, 16), dtype=torch.int64, device=device
            )
            self.epoch = torch.empty((1,), dtype=torch.int64, device=device)
            self.publish_status = torch.empty((p,), dtype=torch.int64, device=device)
            for value in (
                self.flags,
                self.ready,
                self.errors,
                self.epoch,
                self.task_meta,
                self.meta_sources,
                self.queue_flags,
            ):
                _zero[(triton.cdiv(value.numel(), 1024),)](value, value.numel())
            torch.npu.synchronize()
            dist.barrier(group=group)
        except Exception:
            self.arena.close()
            raise

    def _select_tiles(self, t):
        expected_rows = t * self.k / self.e
        self.stream_return = expected_rows >= 128
        if self.stream_return:
            self.block_m, self.block_n = (128, 128)
            self.block_k = 256 if self.i % 256 == 0 else 128
            self.fused_g1 = True
        elif expected_rows >= 64:
            self.block_m = 64
            self.block_n = 256 if self.i % 256 == 0 else 128
            self.block_k = 128
            self.fused_g1 = False
        else:
            self.block_m = 32
            self.block_n = (
                512
                if self.h % 512 == 0 and self.i % 512 == 0
                else 256 if self.i % 256 == 0 else 128
            )
            self.block_k = 64
            self.fused_g1 = False

        # Keep return geometry independent of the compute headroom policy.
        self.return_m, self.return_n = self.block_m, self.block_n
        self.atomic_partial = self.world == 4 and self.k > 1 and expected_rows >= 2
        self.native_atomic = self.atomic_partial and expected_rows >= 32
        self.no_input_padding = self.world == 4 and 2 <= expected_rows < 128
        if self.world == 4 and 32 <= expected_rows < 128:
            self.block_m = 64 if expected_rows < 64 else 128
            self.block_n = 128
            self.block_k = 256 if self.i % 256 == 0 and self.h % 256 == 0 else 128
            self.fused_g1 = True
            self.return_m = 16

    def arguments(self, t, output, counts, output_t):
        return dict(
            TaskMeta=self.task_meta,
            MetaSources=self.meta_sources,
            MergedPacked=self.merged_packed,
            MergedHeaders=self.merged_headers,
            MergedReady=self.merged_ready,
            LocalSlots=self.local_slots,
            SendControl=self.send_control,
            PeerX=self.peer_x,
            PeerTokens=self.peer_tokens,
            PeerExperts=self.peer_experts,
            PeerWeights=self.peer_weights,
            PeerCounts=self.peer_counts,
            PeerReturns=self.peer_returns,
            PeerFlags=self.peer_flags,
            LocalFlags=self.flags,
            Epoch=self.epoch,
            Ready=self.ready,
            Ring=self.ring,
            Activation=self.activation,
            Errors=self.errors,
            Control=self.control,
            W1=self.w1_fused if self.fused_g1 else self.w1,
            W2=self.w2,
            Return=self.returned,
            Output=output,
            Counts=counts,
            T=t,
            OUTPUT_T=output_t,
            H=self.h,
            I=self.i,
            K=self.k,
            KP=self.kp,
            E=self.e,
            C=self.capacity,
            WORLD=self.world,
            RANK=self.rank,
            COMPUTE=self.compute,
            DISPATCH_AIVS=self.dispatch_aivs,
            SCRATCH_WORDS=self.scratch_words,
            NATIVE_REV=bindings.REVISION,
            BATCH_ROWS=self.dispatch_rows,
            QueueFlags=self.queue_flags,
            Progress=self.progress,
            OutputRequests=self.output_requests,
            OutputAssignments=self.output_assignments,
            PeerExpertCounts=self.peer_expert_counts,
            BM=self.block_m,
            BN=self.block_n,
            BK=self.block_k,
            FUSED_G1=self.fused_g1,
            ATOMIC_PARTIAL=self.atomic_partial,
            NATIVE_ATOMIC=self.native_atomic,
            NO_INPUT_PADDING=self.no_input_padding,
            RM=self.return_m,
            RN=self.return_n,
        )

    def _validate_inputs(self, x, ids, probs):
        if self.arena.closed:
            raise RuntimeError("workspace has been closed")
        if not all(
            isinstance(value, torch.Tensor)
            and value.ndim == 2
            and value.layout == torch.strided
            for value in (x, ids, probs)
        ):
            raise NotImplementedError("MegaMoE requires rank-two strided inputs")
        t = x.shape[0]
        if t > self.max_tokens:
            raise ValueError("input exceeds prepared token capacity")
        for name, value, shape, dtypes in (
            ("x", x, (t, self.logical_h), (torch.bfloat16,)),
            ("ids", ids, (t, self.k), (torch.int32, torch.int64)),
            (
                "probs",
                probs,
                (t, self.k),
                (torch.float16, torch.bfloat16, torch.float32),
            ),
        ):
            if tuple(value.shape) != shape or value.dtype not in dtypes:
                raise NotImplementedError(f"unsupported {name} shape or dtype")
            if value.device != self.w1.device or value.requires_grad:
                raise NotImplementedError(
                    f"{name} must be an inference tensor on the workspace NPU"
                )
        return (t, tuple((value.dtype, value.stride()) for value in (x, ids, probs)))

    def _launch(self, x, ids, probs, output, counts, *, extent, warmup):
        """One launch definition for collective compilation and graph capture."""
        grid = (self.total_groups,)
        # Native packing reads FP16/BF16 as 16-bit words. The view preserves strides.
        probs_arg = probs.view(torch.int16) if probs.element_size() == 2 else probs
        launch_args = dict(
            self.arguments(extent, output, counts, x.shape[0]),
            LOGICAL_H=self.logical_h,
            X=x,
            Ids=ids,
            Probs=probs_arg,
            PublishStatus=self.publish_status,
            X0=x.stride(0),
            X1=x.stride(1),
            I0=ids.stride(0),
            I1=ids.stride(1),
            P0=probs.stride(0),
            P1=probs.stride(1),
            ERROR_WORDS=self.errors.numel(),
            PeerChunkCounts=self.peer_chunk_counts,
            RETURN_CHUNKS=self.return_chunks,
            RETURN_BLOCK=_RETURN_CHUNK_TOKENS if self.stream_return else 0,
            AIVS=self.aiv_count,
            PROBS_BF16=probs.dtype == torch.bfloat16,
            multibuffer=True,
            enable_preload=None,
            disable_auto_inject_block_sync=True,
            enable_auto_bind_sub_block=False,
            enable_legacy_insert_load_store_for_mix_cv=True,
        )
        if warmup:
            _queue_pack.warmup(grid=grid, **launch_args)
        else:
            _queue_pack[grid](**launch_args)

    def __call__(self, x, ids, probs):
        key = self._validate_inputs(x, ids, probs)
        if key != self.compiled:
            raise RuntimeError("call workspace.compile collectively first")
        extent = self.extent
        self._select_tiles(extent)
        output = torch.empty(tuple(x.shape), dtype=x.dtype, device=x.device)
        counts = torch.empty((self.e,), dtype=torch.int32, device=x.device)
        self._launch(x, ids, probs, output, counts, extent=extent, warmup=False)
        return output, counts

    def compile(self, x, ids, probs):
        """Activate one collective input profile; old captured graphs remain valid.

        Call collectively again before switching any rank's shape/dtype/strides.
        Keeping only the active profile avoids ambiguous local keys for ragged EP.
        """
        key = self._validate_inputs(x, ids, probs)
        tokens = [None] * self.world
        dist.all_gather_object(tokens, key[0], group=self.group)
        extent = max(1, *tokens)
        self._select_tiles(extent)
        output = torch.empty(tuple(x.shape), dtype=x.dtype, device=x.device)
        counts = torch.empty((self.e,), dtype=torch.int32, device=x.device)
        self._launch(x, ids, probs, output, counts, extent=extent, warmup=True)
        dist.barrier(group=self.group)
        self.compiled, self.extent = key, extent

    def close(self):
        self.arena.close()
