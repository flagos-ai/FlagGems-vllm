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

"""Shared test and benchmark support for NVIDIA Hopper and MetaX MegaMoE."""

from __future__ import annotations

import ctypes
import json
import os

import torch
import triton
import triton.language as tl


class StandaloneNvshmem:
    """Own process-level NVSHMEM resources for tests and benchmarks."""

    def __init__(self):
        from triton.experimental.tle.raw.nvshmem.utils import (
            init_nvshmem_by_torch_pg,
            init_torch_distributed,
            load_common_host,
        )

        self.group = init_torch_distributed()
        self.rank = self.group.rank()
        self.world_size = self.group.size()
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        self.common = load_common_host()
        init_nvshmem_by_torch_pg(self.common, self.group)
        self.library = self.common.library
        self.library.nvshmem_malloc.argtypes = [ctypes.c_size_t]
        self.library.nvshmem_malloc.restype = ctypes.c_void_p
        self.library.nvshmem_free.argtypes = [ctypes.c_void_p]
        self.library.nvshmem_free.restype = None
        self.library.nvshmem_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.library.nvshmem_ptr.restype = ctypes.c_void_p
        self.library.nvshmemx_barrier_all_on_stream.argtypes = [ctypes.c_void_p]
        self.library.nvshmemx_barrier_all_on_stream.restype = None

        self._allocations: list[int] = []
        self._closed = False
        self._timing_events = None
        self._timing_barriers = 0

    def malloc(self, nbytes: int) -> int:
        pointer = self.library.nvshmem_malloc(nbytes)
        if not pointer:
            raise MemoryError(f"NVSHMEM allocation failed for {nbytes} bytes")
        address = int(pointer)
        self._allocations.append(address)
        return address

    def peer_addresses(self, pointer: int) -> tuple[int, ...]:
        addresses = []
        for peer in range(self.world_size):
            address = (
                pointer
                if peer == self.rank
                else self.library.nvshmem_ptr(ctypes.c_void_p(pointer), peer)
            )
            if not address:
                raise RuntimeError(f"rank {peer} is not directly addressable")
            addresses.append(int(address))
        return tuple(addresses)

    def peer_table(self, pointer: int) -> torch.Tensor:
        return torch.tensor(
            self.peer_addresses(pointer), dtype=torch.int64, device=self.device
        )

    def barrier_all_on_stream(self, stream: torch.cuda.Stream) -> None:
        if self._timing_events is not None and self._timing_barriers == 1:
            self._timing_events[1].record(stream)
        self.library.nvshmemx_barrier_all_on_stream(ctypes.c_void_p(stream.cuda_stream))
        if self._timing_events is not None and self._timing_barriers == 0:
            self._timing_events[0].record(stream)
        if self._timing_events is not None:
            self._timing_barriers += 1

    def arm_kernel_timing(self) -> None:
        assert self._timing_events is None
        self._timing_events = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        self._timing_barriers = 0

    def finish_kernel_timing(self) -> float:
        assert self._timing_events is not None and self._timing_barriers == 2
        start, end = self._timing_events
        end.synchronize()
        elapsed_us = start.elapsed_time(end) * 1e3
        self._timing_events = None
        self._timing_barriers = 0
        return elapsed_us

    def close(self) -> None:
        if self._closed:
            return
        stream = torch.cuda.current_stream(self.device)
        self.barrier_all_on_stream(stream)
        stream.synchronize()
        for pointer in reversed(self._allocations):
            self.library.nvshmem_free(ctypes.c_void_p(pointer))
        self._allocations.clear()
        torch.distributed.barrier(group=self.group)
        result = self.common.nvshmem_finalize_from_torch_distributed()
        assert result == 0
        torch.distributed.destroy_process_group()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def interleave_l1_gate_up_rows(weight: torch.Tensor) -> torch.Tensor:
    """Pack logical ``[gate, up]`` rows into the kernel's 8-row interleave."""

    experts, rows, hidden = weight.shape
    half = rows // 2
    gate = weight[:, :half].reshape(experts, half // 8, 8, hidden)
    up = weight[:, half:].reshape(experts, half // 8, 8, hidden)
    return torch.stack((gate, up), dim=2).reshape(experts, rows, hidden).contiguous()


def pool_tokens(topk_all, config) -> int:
    """Compute the padded receive-pool capacity required by a route plan."""

    experts_per_rank = config.num_experts // config.num_ranks
    counts = [
        [[0] * config.num_ranks for _ in range(experts_per_rank)]
        for _ in range(config.num_ranks)
    ]
    for source_rank, experts in enumerate(
        topk_all.reshape(config.num_ranks, -1).tolist()
    ):
        for expert in experts:
            if expert < 0:
                continue
            counts[expert // experts_per_rank][expert % experts_per_rank][
                source_rank
            ] += 1
    peak_recv = max(
        count
        for rank_counts in counts
        for expert_counts in rank_counts
        for count in expert_counts
    )
    if peak_recv > config.max_recv:
        raise ValueError(
            f"max_recv={config.max_recv} is smaller than the observed "
            f"per-source expert load {peak_recv}"
        )
    blocks_per_rank = [
        sum((sum(expert) + 127) // 128 for expert in rank_counts)
        for rank_counts in counts
    ]
    return max(max(blocks_per_rank) * 128, 128)


def make_context(config, runtime: StandaloneNvshmem, pool_size: int):
    """Allocate the caller-owned context used by a standalone invocation."""

    from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamoe.api import (
        MegaMoEContext,
    )

    device = runtime.device
    world_size = runtime.world_size
    experts_per_rank = config.num_experts // world_size
    scale_columns = config.hidden // 128
    n_l1_tiles = config.intermediate // 64
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    pointers = (
        runtime.malloc(config.tokens * config.hidden),
        runtime.malloc(config.tokens * scale_columns * 4),
        runtime.malloc(config.tokens * config.topk * 4),
        runtime.malloc(experts_per_rank * world_size * config.max_recv * 4),
        runtime.malloc(world_size * experts_per_rank * 8),
        runtime.malloc(experts_per_rank * 8),
        runtime.malloc(config.topk * config.tokens * config.hidden * 2),
        runtime.malloc(16 * 4),
        runtime.malloc(pool_size * 3 * 4),
    )
    tables = tuple(runtime.peer_table(pointer) for pointer in pointers)

    def local(*shape, dtype):
        return torch.empty(shape, dtype=dtype, device=device)

    return MegaMoEContext(
        rank=runtime.rank,
        world_size=world_size,
        num_sms=num_sms,
        num_experts=config.num_experts,
        hidden_size=config.hidden,
        intermediate_size=config.intermediate,
        max_tokens=config.tokens,
        max_topk=config.topk,
        max_recv=config.max_recv,
        pool_tokens=pool_size,
        token_ptrs=tables[0],
        scale_ptrs=tables[1],
        route_weight_ptrs=tables[2],
        queue_ptrs=tables[3],
        recv_count_ptrs=tables[4],
        recv_sum_ptrs=tables[5],
        combine_ptrs=tables[6],
        signal_ptrs=tables[7],
        metadata_ptrs=tables[8],
        combine_local=pointers[6],
        token_peer_ptrs=runtime.peer_addresses(pointers[0]),
        expert_send_count=local(config.num_experts, dtype=torch.int64),
        l1_arrival_count=local(pool_size // 128, dtype=torch.int32),
        l2_arrival_mask=local(pool_size // 128, dtype=torch.int64),
        global_counters=local(8, dtype=torch.int32),
        metadata_ready=local(num_sms, dtype=torch.int32),
        l1_tokens=local(pool_size, config.hidden, dtype=torch.float8_e4m3fn),
        l1_scales=local(scale_columns * pool_size, dtype=torch.float32),
        l1_weights=local(pool_size, dtype=torch.float32),
        l2_activations=local(pool_size, config.intermediate, dtype=torch.float8_e4m3fn),
        l2_scales=local(n_l1_tiles * pool_size, dtype=torch.float32),
        barrier_all_on_stream=runtime.barrier_all_on_stream,
    )


def configure_descriptor_allocator(device: torch.device) -> None:
    """Install the descriptor workspace allocator used by standalone workers."""

    import triton

    def allocate(size: int, alignment: int, stream):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(allocate)


OUTPUT_RELATIVE_L2_TOLERANCE = 0.005


def _relative_l2_from_delta(delta, target):
    # Match the full53 oracle: CPU FP64 sum-of-squares and squared-norm floor.
    return (
        float(
            delta.double().square().sum()
            / target.double().square().sum().clamp_min(1e-30)
        )
        ** 0.5
    )


def assert_moe_output(actual, expected):
    """One final-output policy for every tensor case; no case-specific tolerances."""
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    actual, expected = actual.float().cpu(), expected.float().cpu()
    assert bool(torch.isfinite(actual).all()), "nonfinite MegaMoE output"
    assert bool(torch.isfinite(expected).all()), "nonfinite reference output"
    error = _relative_l2_from_delta(actual - expected, expected)
    assert (
        error <= OUTPUT_RELATIVE_L2_TOLERANCE
    ), f"relative L2 {error:.9g} exceeds {OUTPUT_RELATIVE_L2_TOLERANCE}"
    if not bool(torch.count_nonzero(expected)):
        assert not bool(
            torch.count_nonzero(actual)
        ), "zero reference requires zero output"


def reference(x, x_scale, w1, w2, s1, s2, routes, weights):
    """CPU integer GEMMs and BF16 rounding, with dynamic symmetric quantization."""
    x, x_scale, w1, w2, s1, s2, routes, weights = (
        t.cpu() for t in (x, x_scale, w1, w2, s1, s2, routes, weights)
    )
    i = w2.shape[-1]
    result = torch.zeros(x.shape, dtype=torch.float32)

    for token in range(x.shape[0]):
        for slot in range(routes.shape[1]):
            expert = int(routes[token, slot])
            if expert < 0:
                continue
            gu = (w1[expert].int() @ x[token].int()).float()
            gu = (gu * x_scale[token] * s1[expert]).bfloat16().float()
            gate, up = gu[:i], gu[i:]
            silu = (gate / (1 + torch.exp(-gate))).bfloat16().float()
            act = (silu * up).bfloat16().float()
            amax = act.abs().max()
            scale = (
                torch.div(amax, torch.tensor(127.0)) if amax > 0 else torch.tensor(1.0)
            )
            q = torch.div(act, scale).round().clamp(-127, 127).to(torch.int8)
            partial = (w2[expert].int() @ q.int()).float()
            partial = (partial * scale * s2[expert] * weights[token, slot]).bfloat16()
            result[token] += partial.float()
    return result.bfloat16()


def make_inputs(t, h, i, experts, top_k, device="cuda"):
    gen = torch.Generator().manual_seed(71)
    x = torch.randint(-32, 33, (t, h), dtype=torch.int8, generator=gen)
    w1 = torch.randint(-32, 33, (experts, 2 * i, h), dtype=torch.int8, generator=gen)
    w2 = torch.randint(-32, 33, (experts, h, i), dtype=torch.int8, generator=gen)
    xs = torch.rand((t,), generator=gen) * 0.02 + 0.005
    s1 = torch.rand((experts, 2 * i), generator=gen) * 0.02 + 0.005
    s2 = torch.rand((experts, h), generator=gen) * 0.02 + 0.005
    routes = torch.stack(
        [torch.randperm(experts, generator=gen)[:top_k] for _ in range(t)]
    )
    weights = torch.rand((t, top_k), generator=gen)
    return tuple(z.to(device) for z in (x, xs, w1, w2, s1, s2, routes, weights))


def formula_numerical_reference(
    l1_tok, l1_sf, l1_w, w1_q, w1_sf, w2_q, w2_sf, block_expert_tab, MAX_BLOCKS, BLOCK_M
):
    import torch
    import triton.language as tl

    EPR, _, K = w1_q.shape
    INTER = w2_q.shape[-1]
    POOL_TOKENS = MAX_BLOCKS * BLOCK_M
    dev = l1_tok.device
    from vllm.model_executor.layers.fused_moe import fused_moe as vf

    ref_weights_l1 = (
        w1_q.reshape(EPR, INTER // 8, 2, 8, K)
        .permute(0, 2, 1, 3, 4)
        .reshape(EPR, 2 * INTER, K)
        .contiguous()
    )
    ref_scales_l1 = (
        w1_sf[:, 0]
        .reshape(EPR, INTER // 8, 2, 8)
        .permute(0, 2, 1, 3)
        .reshape(EPR, 2 * INTER, 1)
        .contiguous()
    )
    order = torch.arange(POOL_TOKENS, device=dev, dtype=torch.int32)
    expert_ids = block_expert_tab
    padded = torch.tensor([MAX_BLOCKS * BLOCK_M], device=dev, dtype=torch.int32)
    cfg = dict(
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=16,
        num_warps=4,
        num_stages=1,
    )
    ref_gu = torch.zeros((POOL_TOKENS, 1, 2 * INTER), device=dev, dtype=torch.bfloat16)
    vf.invoke_fused_moe_triton_kernel(
        l1_tok[:POOL_TOKENS],
        ref_weights_l1,
        ref_gu,
        l1_sf[:POOL_TOKENS].view(-1, 1),
        ref_scales_l1,
        None,
        order,
        expert_ids,
        padded,
        False,
        1,
        cfg,
        tl.bfloat16,
        False,
        True,
        False,
        False,
        True,
    )
    ref_act = torch.empty((POOL_TOKENS, INTER), device=dev, dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul(ref_act, ref_gu.view(POOL_TOKENS, 2 * INTER))
    # Independent CPU IEEE FP32 formula; do not call the vendor quantizer.
    cpu = ref_act.float().cpu()
    amax = cpu.abs().amax(dim=-1, keepdim=True)
    cpu_sf = torch.where(amax > 0, torch.div(amax, torch.tensor(127.0)), 1.0)
    cpu_q = torch.div(cpu, cpu_sf).round().clamp(-127, 127).to(torch.int8)
    ref_q = cpu_q.to(dev)
    ref_sf = cpu_sf.to(dev)
    ref_out = torch.zeros((POOL_TOKENS, 1, K), device=dev, dtype=torch.bfloat16)
    vf.invoke_fused_moe_triton_kernel(
        ref_q,
        w2_q,
        ref_out,
        ref_sf,
        w2_sf[:, 0].unsqueeze(-1).contiguous(),
        l1_w.view(-1, 1),
        order,
        expert_ids,
        padded,
        True,
        1,
        cfg,
        tl.bfloat16,
        False,
        True,
        False,
        False,
        True,
    )
    return ref_q, ref_sf, ref_out[:, 0], ref_act, ref_gu[:, 0]


@triton.jit(do_not_specialize=["tokens"])
def read_local_partials(
    table,
    output,
    tokens,
    RANK: tl.constexpr,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    idx = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    row, col = idx // H, idx % H
    slot, token = row // tokens, row % tokens
    pointer = tl.load(table + RANK).to(tl.pointer_type(tl.bfloat16))
    value = tl.load(
        pointer + (slot * CAPACITY + token) * H + col,
        mask=idx < tokens * TOPK * H,
        other=0,
    )
    tl.store(output + idx, value, mask=idx < tokens * TOPK * H)


def snapshot_partials(ws, tokens):
    h = ws.model_shape[1]
    slots = ws.world_size if ws.rank_reduce else ws.top_k
    snapshot = torch.empty(
        (slots, tokens, h), device=ws.buffers[0].device, dtype=torch.bfloat16
    )
    if snapshot.numel():
        read_local_partials[(triton.cdiv(snapshot.numel(), 1024),)](
            ws.partials, snapshot, tokens, ws.rank, h, slots, ws.capacity
        )
    return snapshot.cpu()


def check_formula(ws, w1, w2, s1, s2, local, lib):
    diagnostics = {}
    try:

        state, _, _, _, _, _, plan, metadata, pool, sf, weights, _, q, qs = ws.buffers
        e, _, h = w1.shape
        i = w2.shape[-1]
        blocks = int(state[1].cpu())
        gate = w1[:, :i].reshape(e, i // 8, 8, h)
        up = w1[:, i:].reshape(e, i // 8, 8, h)
        packed_w1 = torch.stack((gate, up), 2).reshape(e, 2 * i, h)
        gate_s = s1[:, :i].reshape(e, i // 8, 8)
        up_s = s1[:, i:].reshape(e, i // 8, 8)
        packed_s1 = torch.stack((gate_s, up_s), 2).reshape(e, 1, 2 * i)
        rq, rs, rp, _, _ = formula_numerical_reference(
            pool,
            sf,
            weights,
            packed_w1,
            packed_s1,
            w2,
            s2[:, None, :],
            plan[:blocks, 0].contiguous(),
            blocks,
            ws.bm,
        )
        rows = torch.arange(blocks * ws.bm, device=w1.device)
        indices = rows[rows % ws.bm < plan[rows // ws.bm, 1]]
        diagnostics["formula_q_bad"] = int((q[indices] != rq[indices]).sum())
        diagnostics["formula_sf_bad"] = int((qs[indices] != rs[indices, 0]).sum())
        diagnostics["formula_sf_nonfinite"] = int((~torch.isfinite(qs[indices])).sum())
        meta, expected = metadata[indices].cpu().long(), rp[indices].cpu()
        diagnostics["formula_metadata_bad"] = int(
            (
                (meta[:, 0] < 0)
                | (meta[:, 0] >= ws.world_size)
                | (meta[:, 1] < 0)
                | (meta[:, 1] >= local.shape[1])
                | (meta[:, 2] < 0)
                | (meta[:, 2] >= ws.top_k)
            ).sum()
        )
    except Exception as error:
        diagnostics["formula_error"] = f"{type(error).__name__}: {error}"
    # A rank-local oracle failure must not strand peers inside the broadcasts.
    ready = "formula_error" not in diagnostics and not diagnostics.get(
        "formula_metadata_bad", 0
    )
    if not lib.mxshmem_all_success_wrapper(int(ready)):
        diagnostics["formula_exact"] = False
        return diagnostics
    if ws.rank_reduce:
        got = ws.expert_partials[indices].cpu()
        bad = int((got != expected).sum())
        nonfinite = int((~torch.isfinite(got)).sum())
        max_abs = (
            float((got.float() - expected.float()).abs().max()) if got.numel() else 0.0
        )
        reduction_bad, reduction_nonfinite = 0, 0
        for source in range(ws.world_size):
            received = local if source == ws.rank else torch.empty_like(local)
            lib.mxshmem_broadcast_bytes_wrapper(
                received.data_ptr(), received.numel() * received.element_size(), source
            )
            combined = torch.zeros((local.shape[1], h), dtype=torch.float32)
            for slot in range(ws.top_k):
                selected = (meta[:, 0] == source) & (meta[:, 2] == slot)
                combined[meta[selected, 1]] += expected[selected].float()
            target = combined.to(torch.bfloat16)
            reduction_bad += int((received[ws.rank] != target).sum())
            reduction_nonfinite += int((~torch.isfinite(received[ws.rank])).sum())
        diagnostics.update(
            formula_partial_bad=bad,
            formula_partial_nonfinite=nonfinite,
            formula_partial_max_abs=max_abs,
            rank_reduction_bad=reduction_bad,
            rank_reduction_nonfinite=reduction_nonfinite,
        )
        diagnostics["formula_exact"] = not any(
            diagnostics[key]
            for key in (
                "formula_q_bad",
                "formula_sf_bad",
                "formula_sf_nonfinite",
                "formula_metadata_bad",
                "formula_partial_bad",
                "formula_partial_nonfinite",
                "rank_reduction_bad",
                "rank_reduction_nonfinite",
            )
        )
        return diagnostics
    bad, nonfinite, max_abs = 0, 0, 0.0
    for source in range(ws.world_size):
        received = local if source == ws.rank else torch.empty_like(local)
        lib.mxshmem_broadcast_bytes_wrapper(
            received.data_ptr(), received.numel() * received.element_size(), source
        )
        selected = meta[:, 0] == source
        got = received[meta[selected, 2], meta[selected, 1]]
        target = expected[selected]
        bad += int((got != target).sum())
        nonfinite += int((~torch.isfinite(got)).sum())
        if got.numel():
            max_abs = max(max_abs, float((got.float() - target.float()).abs().max()))
    diagnostics.update(
        formula_partial_bad=bad,
        formula_partial_nonfinite=nonfinite,
        formula_partial_max_abs=max_abs,
    )
    diagnostics["formula_exact"] = not any(
        diagnostics[key]
        for key in (
            "formula_q_bad",
            "formula_sf_bad",
            "formula_sf_nonfinite",
            "formula_metadata_bad",
            "formula_partial_bad",
            "formula_partial_nonfinite",
        )
    )
    return diagnostics


def check_output(output, local, routes, fixture, rank_reduce=False):
    actual = output.cpu()
    routes = routes.cpu()
    combined = torch.zeros_like(actual, dtype=torch.float32)
    sum_abs = torch.zeros_like(combined)
    if rank_reduce:
        for rank in range(local.shape[0]):
            partial = local[rank].float()
            combined += partial
            sum_abs += partial.abs()
    else:
        for slot in range(local.shape[0]):
            selected = routes[:, slot] >= 0
            partial = local[slot, selected].float()
            combined[selected] += partial
            sum_abs[selected] += partial.abs()
    target = fixture["output"].float()
    delta = actual.float() - target
    rel_l2 = _relative_l2_from_delta(delta, target)
    diagnostics = dict(
        rel_l2=rel_l2,
        max_abs=float(delta.abs().max()) if delta.numel() else 0.0,
        combine_bad=int((actual != combined.to(torch.bfloat16)).sum()),
        # This bound covers BF16 rounding of this formula's slot sum.
        reduction_bound_bad=int(
            ((actual.float() - combined).abs() > sum_abs * (8.0 / 256.0) + 1e-8).sum()
        ),
        vllm_elementwise_bound_exceedances=int(
            (delta.abs() > sum_abs * (8.0 / 256.0) + 1e-8).sum()
        ),
        output_nonfinite=int((~torch.isfinite(delta)).sum()),
    )
    diagnostics["output_ok"] = (
        rel_l2 <= OUTPUT_RELATIVE_L2_TOLERANCE
        and not diagnostics["combine_bad"]
        and not diagnostics["reduction_bound_bad"]
        and not diagnostics["output_nonfinite"]
    )
    return diagnostics


def check_dispatch(ws, shared, tokens):
    state, _, _, _, _, _, plan, metadata, pool, sf, weights, *_ = ws.buffers
    blocks = int(state[1].cpu())
    if not 0 <= blocks <= plan.shape[0]:
        raise ValueError(f"invalid active block count: {blocks}")
    plan = plan[:blocks].cpu().long()
    rows = torch.arange(blocks * ws.bm)
    rows = rows[rows % ws.bm < plan[rows // ws.bm, 1]]
    meta = metadata[rows.to(metadata.device)].cpu().long()
    routes = shared["topk_idx"][: ws.world_size, :tokens]
    local_experts = ws.model_shape[0]
    first_expert = ws.rank * local_experts
    owned = (routes >= first_expert) & (routes < first_expert + local_experts)
    expected_keys = torch.nonzero(owned.reshape(-1)).flatten()
    metadata_bad = int(
        (
            (meta[:, 0] < 0)
            | (meta[:, 0] >= ws.world_size)
            | (meta[:, 1] < 0)
            | (meta[:, 1] >= tokens)
            | (meta[:, 2] < 0)
            | (meta[:, 2] >= ws.top_k)
        ).sum()
    )
    diagnostics = dict(
        dispatch_expected_routes=expected_keys.numel(),
        dispatch_actual_routes=meta.shape[0],
        dispatch_metadata_bad=metadata_bad,
        dispatch_ok=False,
    )
    if metadata_bad:
        return diagnostics
    source, token, slot = meta.unbind(1)
    actual_keys = (source * tokens + token) * ws.top_k + slot
    diagnostics.update(
        dispatch_route_set_bad=int(
            not torch.equal(actual_keys.sort().values, expected_keys)
        ),
        dispatch_expert_bad=int(
            (routes[source, token, slot] != first_expert + plan[rows // ws.bm, 0]).sum()
        ),
        dispatch_token_bad=int(
            (
                pool[rows.to(pool.device)].cpu() != shared["input_int8"][source, token]
            ).sum()
        ),
        dispatch_scale_bad=int(
            (
                sf[rows.to(sf.device)].cpu()
                != shared["input_scales"][source, token].reshape(-1)
            ).sum()
        ),
        dispatch_weight_bad=int(
            (
                weights[rows.to(weights.device)].cpu()
                != shared["topk_weights"][source, token, slot]
            ).sum()
        ),
    )
    diagnostics["dispatch_ok"] = not any(
        value for name, value in diagnostics.items() if name.endswith("_bad")
    )
    return diagnostics


def validate(ws, w1, w2, s1, s2, output, routes, fixture, shared, lib, formula, phase):
    diagnostics = dict(
        kind="correctness",
        correctness_phase=phase,
        rank=ws.rank,
        world=ws.world_size,
        tokens=output.shape[0],
        input_hashes_match=True,
        formula_checked=formula,
        quantization_contract="symmetric-int8-rne-v1",
        reduction_tree="rank-fp32-bf16-fp32" if ws.rank_reduce else "slot-fp32",
        output_ok=False,
        dispatch_ok=False,
    )
    # Inspect destination-owned storage before either oracle can affect GPU state.
    local = None
    try:
        local = snapshot_partials(ws, output.shape[0])
        diagnostics.update(
            check_output(output, local, routes, fixture, rank_reduce=ws.rank_reduce)
        )
        diagnostics.update(check_dispatch(ws, shared, output.shape[0]))
    except Exception as error:
        diagnostics["validation_error"] = f"{type(error).__name__}: {error}"
    snapshots_ready = lib.mxshmem_all_success_wrapper(
        int(
            local is not None
            and "validation_error" not in diagnostics
            and not diagnostics.get("dispatch_metadata_bad", 0)
        )
    )
    if formula and snapshots_ready:
        diagnostics.update(check_formula(ws, w1, w2, s1, s2, local, lib))
    elif formula:
        diagnostics["formula_exact"] = False
    passed = bool(
        lib.mxshmem_all_success_wrapper(
            int(
                diagnostics["output_ok"]
                and diagnostics["dispatch_ok"]
                and "validation_error" not in diagnostics
                and (not formula or diagnostics["formula_exact"])
            )
        )
    )
    diagnostics["status"] = "PASS" if passed else "FAIL"
    print(json.dumps(diagnostics), flush=True)
    return passed


__all__ = [
    "StandaloneNvshmem",
    "interleave_l1_gate_up_rows",
    "pool_tokens",
    "make_context",
    "configure_descriptor_allocator",
    "assert_moe_output",
    "reference",
    "make_inputs",
    "formula_numerical_reference",
    "read_local_partials",
    "snapshot_partials",
    "check_formula",
    "check_output",
    "check_dispatch",
    "validate",
]
