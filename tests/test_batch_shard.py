# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Exact pack/unpack tests, including padding and untouched output guards."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmark.test_batch_shard import (
    KERNEL_NAMES,
    check_case,
    make_case,
    run_candidate,
    run_reference,
)
from flaggems_vllm import device as DEVICE
from flaggems_vllm import runtime
from flaggems_vllm.ops import batch_shard as impl

pytestmark = [
    pytest.mark.gather_sampler_output,
    pytest.mark.skipif(
        DEVICE == "cpu" or not runtime.torch_device_fn.is_available(),
        reason=f"requires an available {DEVICE} accelerator",
    ),
]


@pytest.mark.gather_sampler_output
@pytest.mark.parametrize("name", KERNEL_NAMES)
@pytest.mark.parametrize(
    "n,k,c,scenario",
    [
        (0, 1, 1, "dense"),
        (1, 1, 1, "dense"),
        (7, 3, 7, "ragged"),
        (31, 7, 21, "pad"),
        (33, 8, 65, "truncate"),
        (129, 16, 257, "strided"),
        (513, 15, 6, "dense"),
        (129, 65, 7, "strided"),
        (3, 513, 7, "strided"),
    ],
)
def test_pack_unpack(name, n, k, c, scenario):
    case = make_case(name, n, k, c, scenario)
    run_reference(case)
    check_case(case)
    run_candidate(case)
    check_case(case)
    run_candidate(case)
    check_case(case)


@pytest.mark.parametrize("name", KERNEL_NAMES)
@pytest.mark.parametrize(
    "id_dtype,index_dtype",
    [
        (torch.int32, torch.int32),
        (torch.int64, torch.int64),
        (torch.int32, torch.int64),
    ],
)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_integer_and_float_types(name, id_dtype, index_dtype, dtype):
    case = make_case(
        name,
        17,
        4,
        21,
        "strided",
        dtype=dtype,
        id_dtype=id_dtype,
        index_dtype=index_dtype,
    )
    run_reference(case)
    check_case(case)
    run_candidate(case)
    check_case(case)


@pytest.mark.parametrize("name", KERNEL_NAMES[:2])
@pytest.mark.parametrize("with_nans", [False, True])
def test_optional_nans(name, with_nans):
    case = make_case(name, 37, 9, with_nans=with_nans, scenario="ragged")
    run_candidate(case)
    check_case(case)


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_plain_triton(name):
    case = make_case(name, 33, 4, 6, "ragged")
    run_candidate(case, use_layout=False)
    check_case(case)


@pytest.mark.skipif(
    runtime.device.vendor_name != "nvidia", reason="NVIDIA seed configurations"
)
@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_all_tuning_tiles(name):
    config_name = name.removeprefix("_").removesuffix("_kernel")
    for config in runtime.get_tuned_config(config_name):
        case = make_case(name, config.kwargs.get("BLOCK_ROWS", 1) + 1, 3, 7, "ragged")
        run_candidate(case, config=config)
        check_case(case)


@pytest.mark.parametrize("name", [KERNEL_NAMES[0], KERNEL_NAMES[2]])
def test_zero_source_columns(name):
    case = make_case(name, 9, 3, 7, source_cols=0)
    run_reference(case)
    check_case(case)
    run_candidate(case)
    check_case(case)


def test_logprob_special_values():
    case = make_case(KERNEL_NAMES[2], 1, 1, 6)
    values = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.0], device=DEVICE
    )
    case.args[6].copy_(values[None, :])
    # The second output is a guarded, contiguous float32 row.
    case.expected[1][:6] = values.cpu()
    run_reference(case)
    check_case(case)
    run_candidate(case)
    check_case(case)
    assert torch.equal(
        torch.signbit(case.outputs[1][3:5]), torch.tensor([False, True], device=DEVICE)
    )


@dataclass
class SamplerOutput:
    # Output-container contract from the pinned vLLM revision; older wheels
    # do not yet expose num_rejected. No compute is replaced by this fixture.
    sampled_token_ids: torch.Tensor
    logprobs_tensors: object
    num_nans: object
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    sampling_mask_tensors: object = None


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_large_integers_and_repeated_sources(name):
    case = make_case(name, 5, 3, 6, id_dtype=torch.int64, index_dtype=torch.int64)
    if name == KERNEL_NAMES[0]:
        for index in (2, 4, 5, 6):
            case.args[index].fill_(2**40 + 19)
    elif name == KERNEL_NAMES[1]:
        case.args[0].fill_(2**40 + 19)
        case.args[2].fill_(0)
    elif name == KERNEL_NAMES[2]:
        case.args[4].fill_(-(2**40) - 17)
        case.args[8].fill_(2**40 + 19)
    else:
        case.args[0].fill_(2**40 + 19)
        case.args[4].fill_(0)
    run_reference(case)
    expected = [value.clone() for value in case.outputs]
    for value in case.outputs:
        value.fill_(-123)
    run_candidate(case)
    for actual, ref in zip(case.outputs, expected):
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "with_nans,with_logprobs",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_gather_sampler_output_dataflow(monkeypatch, with_nans, with_logprobs, dtype):
    comm = pytest.importorskip("vllm.distributed.communication_op")
    output_module = pytest.importorskip("vllm.v1.worker.gpu.sample.output")
    monkeypatch.setattr(output_module, "SamplerOutput", SamplerOutput)
    # Four ranks, one empty shard, uneven counts, and original request order
    # interleaving the rank-major collective layout.
    owners = [2, 0, 1, 1, 2]
    counts = [1, 3, 2, 1, 3]
    tp, k, c, max_reqs = 4, 3, 7, 2
    requests = [
        [i for i, owner in enumerate(owners) if owner == rank] for rank in range(tp)
    ]
    src_indices = [
        owners[i] * max_reqs + requests[owners[i]].index(i) for i in range(len(owners))
    ]
    global_cu = np.r_[0, np.cumsum(counts)].astype(np.int32)
    global_batch = SimpleNamespace(
        num_reqs=len(owners),
        cu_num_logits_np=global_cu,
        cu_num_logits=torch.tensor(global_cu, device=DEVICE),
    )
    packed = torch.full(
        (tp * max_reqs, k + 2 + int(with_nans)), -991, dtype=torch.int64
    )
    ids = torch.full((tp * max_reqs * k, c + 1), -992, dtype=torch.int64)
    probs = torch.full((tp * max_reqs * k, c), -993.0, dtype=torch.float32)
    locals_ = []
    for rank, reqs in enumerate(requests):
        n = len(reqs)
        local_cu = np.r_[0, np.cumsum([counts[i] for i in reqs])].astype(np.int32)
        rows = int(local_cu[-1])
        # Both widening and truncation occur across rank-local logprob widths.
        width = (1, 5, 11, 7)[rank]
        token_width = 1 if rank == 0 else k
        token_ids = (
            torch.arange(n * token_width).reshape(n, token_width) + rank * 100
        ).long()
        sampled = torch.tensor([counts[i] for i in reqs], dtype=torch.int32)
        rejected = torch.arange(n, dtype=torch.int32)
        nans = torch.arange(rows, dtype=torch.int32) + 1 if with_nans else None
        lp_ids = (torch.arange(rows * width).reshape(rows, width) + rank * 1000).int()
        lp_probs = (-torch.arange(rows * width).reshape(rows, width) / 7).to(dtype)
        lp_ranks = torch.arange(rows, dtype=torch.int64) + rank * 10
        for j, req in enumerate(reqs):
            out = rank * max_reqs + j
            packed[out, :k] = 0
            packed[out, :token_width] = token_ids[j]
            packed[out, k] = sampled[j]
            packed[out, k + 1] = rejected[j]
            begin, end = local_cu[j : j + 2]
            if with_nans:
                packed[out, k + 2] = nans[begin:end].long().sum()
            for pos in range(end - begin):
                dst = out * k + pos
                ids[dst, :c] = 0
                probs[dst] = -float("inf")
                ids[dst, : min(c, width)] = lp_ids[begin + pos, : min(c, width)]
                ids[dst, c] = lp_ranks[begin + pos]
                probs[dst, : min(c, width)] = lp_probs[
                    begin + pos, : min(c, width)
                ].float()
        to_device = lambda t: t.to(DEVICE) if t is not None else None
        lp = (
            SimpleNamespace(
                logprob_token_ids=to_device(lp_ids),
                logprobs=to_device(lp_probs),
                selected_token_ranks=to_device(lp_ranks),
            )
            if with_logprobs
            else None
        )
        local_output = (
            SamplerOutput(
                to_device(token_ids),
                lp,
                to_device(nans),
                to_device(sampled),
                to_device(rejected),
            )
            if n
            else None
        )
        locals_.append(
            (
                local_output,
                SimpleNamespace(cu_num_logits=torch.tensor(local_cu, device=DEVICE)),
            )
        )

    gathered = [packed.to(DEVICE), ids.to(DEVICE), probs.to(DEVICE)]
    expected_tokens = packed[src_indices, :k]
    expected_ids = []
    expected_probs = []
    expected_ranks = []
    for req, count in enumerate(counts):
        start = src_indices[req] * k
        expected_ids.append(ids[start : start + count, :c])
        expected_probs.append(probs[start : start + count])
        expected_ranks.append(ids[start : start + count, c])
    for rank, (local_output, local_batch) in enumerate(locals_):
        call = [0]

        def all_gather(send, dim=0):
            assert dim == 0
            which = call[0]
            call[0] += 1
            expected = gathered[which][
                rank * send.shape[0] : (rank + 1) * send.shape[0]
            ]
            # Ignore only unowned request padding and inactive logits slots,
            # which upstream intentionally leaves uninitialized.
            if which == 0:
                torch.testing.assert_close(
                    send[: len(requests[rank])],
                    expected[: len(requests[rank])],
                    rtol=0,
                    atol=0,
                )
            else:
                for j, req in enumerate(requests[rank]):
                    span = slice(j * k, j * k + counts[req])
                    torch.testing.assert_close(
                        send[span], expected[span], rtol=0, atol=0
                    )
            return gathered[which]

        monkeypatch.setattr(comm, "tensor_model_parallel_all_gather", all_gather)
        metadata = SimpleNamespace(
            tp_size=tp,
            max_num_logits_per_req=k,
            max_num_reqs_per_rank=max_reqs,
            num_local_reqs=len(requests[rank]),
            gathered_src_indices=torch.tensor(
                src_indices, device=DEVICE, dtype=torch.int64
            ),
        )
        result = impl.gather_sampler_output(
            local_output,
            metadata,
            torch.device(DEVICE),
            global_batch,
            local_batch,
            gather_num_nans=with_nans,
            logprobs_dims=(c - 1, 0) if with_logprobs else None,
        )
        torch.testing.assert_close(
            result.sampled_token_ids.cpu(), expected_tokens, rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.num_sampled.cpu(), packed[src_indices, k].int(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.num_rejected.cpu(), packed[src_indices, k + 1].int(), rtol=0, atol=0
        )
        if with_nans:
            torch.testing.assert_close(
                result.num_nans.cpu(), packed[src_indices, k + 2].int(), rtol=0, atol=0
            )
        else:
            assert result.num_nans is None
        if with_logprobs:
            lp = result.logprobs_tensors
            torch.testing.assert_close(
                lp.logprob_token_ids.cpu(), torch.cat(expected_ids), rtol=0, atol=0
            )
            torch.testing.assert_close(
                lp.logprobs.cpu(), torch.cat(expected_probs), rtol=0, atol=0
            )
            torch.testing.assert_close(
                lp.selected_token_ranks.cpu(), torch.cat(expected_ranks), rtol=0, atol=0
            )
            assert lp.cu_num_generated_tokens == global_cu.tolist()
            assert call[0] == 3
        else:
            assert result.logprobs_tensors is None
            assert call[0] == 1
