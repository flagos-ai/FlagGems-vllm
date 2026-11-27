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

"""Contract tests for the ordinary chained rejection sampler.

The CPU FP64 oracle is independent of the candidate and imports no accelerator
package. Device tests lazily import the public candidate; import/export errors
fail the test instead of disguising a broken installation as a skip.
"""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace
from typing import Any

import pytest
import torch

SOURCE_COMMIT = "vllm-project/vllm@428c371b32c18a788def35abea1cb9f579278381"
SEED = 20261119


def _candidate():
    # Lazy import keeps the CPU oracle importable without an accelerator stack.
    # Real package import/export failures propagate; they are never a skip.
    import flaggems_vllm

    return flaggems_vllm.rejection_sample


def _stable_softmax(logits: torch.Tensor) -> torch.Tensor:
    """CPU FP64 softmax with -inf masking and no candidate code reuse."""
    values = logits.to(dtype=torch.float64)
    finite = torch.isfinite(values)
    max_value = torch.where(finite, values, torch.tensor(float("-inf"))).amax(
        -1, keepdim=True
    )
    shifted = values - max_value
    weights = torch.where(finite, torch.exp(shifted), torch.zeros_like(shifted))
    denominator = weights.sum(-1, keepdim=True)
    return torch.where(
        denominator > 0, weights / denominator, torch.zeros_like(weights)
    )


def _sample_from_uniform(probs: torch.Tensor, uniform: float) -> int:
    """Cumulative FP64 inverse-CDF adapter for deterministic oracle fixtures."""
    cumulative = probs.to(torch.float64).cumsum(-1)
    # The source stream excludes zero.  Keep the adapter's input contract
    # explicit rather than silently replacing an invalid zero with epsilon.
    if not 0.0 < uniform <= 1.0:
        raise ValueError("oracle uniforms must be in (0, 1]")
    return int(torch.searchsorted(cumulative, torch.tensor(uniform), right=False))


def oracle_rejection_sample(
    target_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    temperature: torch.Tensor,
    *,
    draft_logits: torch.Tensor | None = None,
    acceptance_uniforms: torch.Tensor | None = None,
    resample_uniforms: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(sampled, counts, defined_mask)`` for a CPU fixture.

    ``target_logits`` is shaped ``[B, K + 1, V]`` and is already processed by
    the sampling parameters.  ``draft_logits`` is pre-temperature and is
    shaped ``[B, K, V]``.  When it is absent, the draft distribution is the
    one-hot distribution at each proposed token.
    """
    if target_logits.device.type != "cpu" or draft_sampled.device.type != "cpu":
        raise ValueError("the independent oracle only accepts CPU tensors")
    target = _stable_softmax(target_logits)
    batch, rows, vocab = target.shape
    steps = rows - 1
    draft = draft_sampled.to(torch.int64)
    temps = temperature.to(torch.float64).flatten()
    if draft.shape != (batch, steps) or temps.numel() != batch:
        raise ValueError("fixture shapes do not match")
    proposal = (
        _stable_softmax(draft_logits.to(torch.float64) / temps[:, None, None])
        if draft_logits is not None
        else None
    )
    sampled = torch.full((batch, rows), -1, dtype=torch.int64)
    counts = torch.empty(batch, dtype=torch.int32)
    defined = torch.zeros((batch, rows), dtype=torch.bool)
    for request in range(batch):
        accepted = 0
        for step in range(steps):
            token = int(draft[request, step])
            if token < 0:
                break
            if temps[request] == 0:
                accept = token == int(target_logits[request, step].argmax())
            else:
                p = float(target[request, step, token])
                q = 1.0 if proposal is None else float(proposal[request, step, token])
                uniform = (
                    float(acceptance_uniforms[request, step])
                    if acceptance_uniforms is not None
                    else 0.5
                )
                accept = p > uniform * q
            if not accept:
                break
            sampled[request, accepted] = token
            defined[request, accepted] = True
            accepted += 1
        row = accepted
        if row < steps and int(draft[request, row]) >= 0:
            rejected = int(draft[request, row])
            if temps[request] == 0:
                sampled[request, row] = int(target_logits[request, row].argmax())
            else:
                q = (
                    torch.nn.functional.one_hot(
                        torch.tensor(rejected), num_classes=vocab
                    ).to(torch.float64)
                    if proposal is None
                    else proposal[request, row]
                )
                residual = (target[request, row] - q).clamp_min(0)
                mass = float(residual.sum())
                if mass <= 0:
                    raise ValueError("oracle residual has zero mass")
                residual /= mass
                uniform = (
                    float(resample_uniforms[request])
                    if resample_uniforms is not None
                    else 0.5
                )
                sampled[request, row] = _sample_from_uniform(residual, uniform)
            defined[request, row] = True
        else:
            uniform = (
                float(resample_uniforms[request])
                if resample_uniforms is not None
                else 0.5
            )
            sampled[request, row] = (
                int(target_logits[request, row].argmax())
                if temps[request] == 0
                else _sample_from_uniform(target[request, row], uniform)
            )
            defined[request, row] = True
        counts[request] = row + 1
    return sampled, counts, defined


def _device() -> torch.device:
    import flaggems_vllm

    return torch.device(flaggems_vllm.device)


def _cache(values: torch.Tensor | None, row_mapping: torch.Tensor | None = None):
    return (
        None
        if values is None
        else SimpleNamespace(values=values, row_mapping=row_mapping)
    )


def _call_candidate(inputs: dict[str, Any]):
    kwargs = dict(inputs)
    kwargs["draft_logits_cache"] = _cache(
        kwargs.pop("draft_logits", None), kwargs.pop("draft_row_mapping", None)
    )
    return _candidate()(**kwargs)


def _fixture(
    *,
    device: torch.device,
    accepted: tuple[int, ...],
    placeholder: tuple[int, ...] = (),
    variable_lengths: tuple[int, ...] | None = None,
    vocab: int = 17,
    noncontiguous: bool = False,
    temperature: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[dict[str, Any], list[list[int]]]:
    """Build a packed ordinary-chain fixture with unique target argmax rows."""
    batch = len(accepted)
    if variable_lengths is None:
        lengths = (max(accepted, default=0),) * batch
    else:
        lengths = variable_lengths
    steps = max(max(accepted, default=0), max(lengths, default=0))
    rows_per_request = [length + 1 for length in lengths]
    total_rows = sum(rows_per_request)
    target_cpu = torch.full((total_rows, vocab), -40.0, dtype=torch.float32)
    draft_cpu = torch.full((batch, steps, vocab), -40.0, dtype=torch.float32)
    draft_tokens = torch.zeros((batch, steps), dtype=torch.int64)
    expected = []
    row = 0
    for request, length in enumerate(lengths):
        request_expected = []
        for step in range(length + 1):
            target_token = (request * (steps + 1) + step + 1) % vocab
            target_cpu[row, target_token] = 12.0
            request_expected.append(target_token)
            if step < length:
                token = (
                    target_token
                    if step < accepted[request]
                    else (target_token + 1) % vocab
                )
                if step == 0 and request in placeholder:
                    token = -1
                draft_tokens[request, step] = token
                draft_cpu[request, step, token if token >= 0 else 0] = 12.0
            row += 1
        expected.append(request_expected)
    target = target_cpu.to(device=device, dtype=dtype)
    if noncontiguous:
        padded = torch.empty((total_rows, vocab + 1), dtype=dtype, device=device)
        padded[:, :vocab].copy_(target)
        target = padded[:, :vocab]
    draft_values = draft_cpu.to(device=device, dtype=dtype) if temperature > 0 else None
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length + 1)
    expanded_req = []
    expanded_pos = []
    for request, length in enumerate(lengths):
        expanded_req.extend([request] * (length + 1))
        expanded_pos.extend(range(length + 1))
    inputs = {
        "target_logits": target,
        "draft_logits": draft_values,
        "draft_sampled": torch.cat(
            [
                torch.cat((torch.zeros(1, dtype=torch.int64), draft_tokens[r, :length]))
                for r, length in enumerate(lengths)
            ]
        ).to(device),
        "cu_num_logits": torch.tensor(cu, dtype=torch.int32, device=device),
        "pos": torch.arange(total_rows, dtype=torch.int32, device=device) + 101,
        "idx_mapping": torch.arange(batch, dtype=torch.int32, device=device),
        "expanded_idx_mapping": torch.tensor(
            expanded_req, dtype=torch.int32, device=device
        ),
        "expanded_local_pos": torch.tensor(
            expanded_pos, dtype=torch.int32, device=device
        ),
        "temperature": torch.full(
            (batch,), temperature, dtype=torch.float32, device=device
        ),
        "seed": torch.arange(batch, dtype=torch.int64, device=device) + 17,
        "num_speculative_steps": steps,
    }
    return inputs, expected


def _assert_valid_prefix(sampled, counts, vocab: int, max_steps: int):
    assert tuple(sampled.shape) == (counts.numel(), max_steps + 1)
    assert sampled.dtype == torch.int64
    assert counts.dtype == torch.int32
    assert bool(((counts >= 1) & (counts <= max_steps + 1)).all())
    defined = (
        torch.arange(max_steps + 1, device=sampled.device)[None, :] < counts[:, None]
    )
    in_range = (sampled >= 0) & (sampled < vocab)
    assert bool(torch.all(~defined | in_range))


@pytest.mark.rejection_sample
@pytest.mark.parametrize("accepted", [(3, 3), (0, 3), (1, 3), (2, 3)])
def test_rejection_sample_greedy_prefix_and_bonus(accepted):
    device = _device()
    inputs, expected = _fixture(device=device, accepted=accepted)
    before = {
        key: value.clone() for key, value in inputs.items() if torch.is_tensor(value)
    }
    sampled, counts = _call_candidate(inputs)
    _assert_valid_prefix(sampled, counts, vocab=17, max_steps=3)
    for request, count in enumerate(counts.tolist()):
        assert count == accepted[request] + 1
        assert sampled[request, :count].tolist() == expected[request][:count]
    for key, value in before.items():
        assert torch.equal(inputs[key], value), f"candidate mutated input {key}"


@pytest.mark.rejection_sample
def test_rejection_sample_zero_k_bonus():
    device = _device()
    inputs, expected = _fixture(
        device=device, accepted=(0,), variable_lengths=(0,), temperature=0.0
    )
    sampled, counts = _call_candidate(inputs)
    _assert_valid_prefix(sampled, counts, vocab=17, max_steps=0)
    assert counts.tolist() == [1]
    assert sampled[0, 0].item() == expected[0][0]


@pytest.mark.rejection_sample
def test_rejection_sample_single_token_vocabulary():
    # The only possible token is zero; both verification and bonus sampling
    # have a literal independent result at the smallest direct vector width.
    device = _device()
    for steps in (0, 3):
        for temperature in (0.0, 1.0):
            rows = steps + 1
            inputs = {
                "target_logits": torch.zeros((rows, 1), device=device),
                "draft_logits": (
                    torch.zeros((1, steps, 1), device=device) if steps else None
                ),
                "draft_sampled": torch.zeros(rows, dtype=torch.int64, device=device),
                "cu_num_logits": torch.tensor(
                    [0, rows], dtype=torch.int32, device=device
                ),
                "pos": torch.arange(rows, dtype=torch.int64, device=device) + 100003,
                "idx_mapping": torch.zeros(1, dtype=torch.int32, device=device),
                "expanded_idx_mapping": torch.zeros(
                    rows, dtype=torch.int32, device=device
                ),
                "expanded_local_pos": torch.arange(
                    rows, dtype=torch.int32, device=device
                ),
                "temperature": torch.tensor([temperature], device=device),
                "seed": torch.tensor([2**40 + 3], dtype=torch.int64, device=device),
                "num_speculative_steps": steps,
            }
            snapshots = {
                name: value.clone()
                for name, value in inputs.items()
                if torch.is_tensor(value)
            }
            sampled, counts = _call_candidate(inputs)
            _assert_valid_prefix(sampled, counts, vocab=1, max_steps=steps)
            assert counts.tolist() == [rows]
            assert sampled.tolist() == [[0] * rows]
            for name, value in snapshots.items():
                assert torch.equal(
                    inputs[name], value
                ), f"candidate mutated input {name}"


@pytest.mark.rejection_sample
def test_rejection_sample_full_cache_equal_p_accepts_and_bonus():
    device = _device()
    inputs, expected = _fixture(
        device=device, accepted=(3, 3), temperature=1.0, vocab=17
    )
    sampled, counts = _call_candidate(inputs)
    _assert_valid_prefix(sampled, counts, vocab=17, max_steps=3)
    assert counts.tolist() == [4, 4]
    for request in range(2):
        assert sampled[request, :4].tolist() == expected[request][:4]


@pytest.mark.rejection_sample
def test_rejection_sample_zero_draft_and_placeholder():
    device = _device()
    inputs, expected = _fixture(
        device=device, accepted=(0, 1), placeholder=(0,), variable_lengths=(1, 2)
    )
    sampled, counts = _call_candidate(inputs)
    _assert_valid_prefix(sampled, counts, vocab=17, max_steps=2)
    assert counts.tolist() == [1, 2]
    assert sampled[0, 0].item() == expected[0][0]
    assert sampled[1, :2].tolist() == expected[1][:2]


@pytest.mark.rejection_sample
def test_rejection_sample_noncontiguous_outer_stride_and_vocab_tail():
    device = _device()
    inputs, expected = _fixture(
        device=device, accepted=(2,), vocab=257, noncontiguous=True
    )
    assert inputs["target_logits"].stride(-1) == 1
    assert inputs["target_logits"].stride(0) == 258
    sampled, counts = _call_candidate(inputs)
    _assert_valid_prefix(sampled, counts, vocab=257, max_steps=2)
    assert sampled[0, : int(counts[0])].tolist() == expected[0][: int(counts[0])]


@pytest.mark.rejection_sample
@pytest.mark.parametrize("vocab", [1023, 1024, 1025])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rejection_sample_stats_gate_independent_oracle(vocab, dtype):
    _assert_stats_gate_independent_oracle(vocab, dtype)


@pytest.mark.rejection_sample
@pytest.mark.parametrize("vocab", [2048, 2049])
def test_rejection_sample_two_block_stats_gate_independent_oracle(vocab):
    _assert_stats_gate_independent_oracle(vocab, torch.float32)


def _assert_stats_gate_independent_oracle(vocab, dtype):
    # Both sides of the single-block stats gate use independent CPU fixtures.
    # Mixed temperatures include exact dyadic scaling: equal p/q must accept
    # even a low-probability proposal, while disjoint supports force rejection.
    device = _device()
    lengths = (2, 2, 3, 0, 2)
    accepted = (2, 1, 1, 0, 0)
    request_slots = torch.tensor([4, 1, 5, 0, 3], dtype=torch.int32)
    cache_rows = torch.tensor([6, 2, 5, 0, 4, 1], dtype=torch.int32)
    temperatures = torch.tensor([0.25, 0.5, 1.0, 1.5, 0.0, 2.0])
    cu = [0]
    target_rows = []
    proposals = []
    oracle_outputs = []
    draft = torch.full((7, 3, vocab + 3), float("-inf"), dtype=dtype)[..., :vocab]
    for request, length in enumerate(lengths):
        state = int(request_slots[request])
        cache_row = int(cache_rows[state])
        temp = float(temperatures[state])
        target = torch.full((length + 1, vocab), float("-inf"), dtype=dtype)
        tokens = []
        for step in range(length + 1):
            token = (vocab - 4 + step) % vocab
            other = (token + 1) % vocab
            if step < accepted[request]:
                if temp == 0:
                    # Equal maxima select the lower vocabulary index.
                    target[step, token] = target[step, other] = 4.0
                else:
                    target[step, token], target[step, other] = -8.0, 0.0
                    draft[cache_row, step] = target[step] * temp
                tokens.append(token)
            else:
                target[step, other] = 0.0
                if step < length:
                    draft[cache_row, step, token] = 0.0
                    tokens.append(-1 if request == 4 and step == 0 else token)
        expected, counts, defined = oracle_rejection_sample(
            target[None],
            torch.tensor([tokens], dtype=torch.int64).reshape(1, length),
            torch.tensor([temp]),
            draft_logits=draft[cache_row, :length][None],
        )
        oracle_outputs.append((expected[defined].tolist(), int(counts[0])))
        target_rows.append(target)
        proposals.extend([0, *tokens])
        cu.append(cu[-1] + length + 1)
    target_cpu = torch.cat(target_rows)
    target = torch.empty((cu[-1], vocab + 2), dtype=dtype, device=device)[:, :vocab]
    target.copy_(target_cpu.to(device))
    # Preserve noncontiguous outer strides for both target and draft cache.
    draft_device = torch.empty((7, 3, vocab + 3), dtype=dtype, device=device)[
        ..., :vocab
    ]
    draft_device.copy_(draft.to(device))
    inputs = {
        "target_logits": target,
        "draft_logits": draft_device,
        "draft_row_mapping": cache_rows.to(device),
        "draft_sampled": torch.tensor(proposals, dtype=torch.int64, device=device),
        "cu_num_logits": torch.tensor(cu, dtype=torch.int32, device=device),
        "pos": torch.arange(cu[-1], dtype=torch.int64, device=device) + 100003,
        "idx_mapping": request_slots.to(device),
        "expanded_idx_mapping": torch.repeat_interleave(
            request_slots, torch.tensor(lengths) + 1
        ).to(device),
        "expanded_local_pos": torch.tensor(
            [step for length in lengths for step in range(length + 1)],
            dtype=torch.int32,
            device=device,
        ),
        "temperature": temperatures.to(device),
        "seed": (torch.arange(6, dtype=torch.int64) * 17 + 2**40 + 3).to(device),
        "num_speculative_steps": 3,
    }
    sampled, counts = _call_candidate(inputs)
    _assert_valid_prefix(sampled, counts, vocab=vocab, max_steps=3)
    assert counts.tolist() == [count for _, count in oracle_outputs]
    for request, (expected, count) in enumerate(oracle_outputs):
        assert sampled[request, :count].tolist() == expected


@pytest.mark.rejection_sample
@pytest.mark.parametrize("steps", [8, 9])
def test_rejection_sample_static_step_boundary(steps):
    # Ascend's short-chain unrolling ends at K=8. Literal point masses make
    # the all-accepted greedy chain and last-step stochastic rejection exact.
    device = _device()
    inputs, expected = _fixture(
        device=device,
        accepted=(steps, steps - 1, 0),
        variable_lengths=(steps, steps, 0),
        temperature=1.0,
        vocab=17,
    )
    for name in ("target_logits", "draft_logits"):
        values = inputs[name]
        inputs[name] = torch.where(values == 12.0, 0.0, float("-inf"))
    inputs["temperature"] = torch.tensor([0.0, 2.0, 0.5], device=device)
    sampled, counts = _call_candidate(inputs)
    _assert_valid_prefix(sampled, counts, vocab=17, max_steps=steps)
    assert counts.tolist() == [steps + 1, steps, 1]
    for request, count in enumerate(counts.tolist()):
        assert sampled[request, :count].tolist() == expected[request][:count]


@pytest.mark.rejection_sample
def test_rejection_sample_multiple_vocab_blocks():
    device = _device()
    inputs, expected = _fixture(
        device=device, accepted=(2,), vocab=1025, temperature=0.0
    )
    inputs["target_logits"][:, :] = -40.0
    inputs["target_logits"][:, 1024] = 12.0
    inputs["draft_sampled"][1:] = 1024
    sampled, counts = _call_candidate(inputs)
    _assert_valid_prefix(sampled, counts, vocab=1025, max_steps=2)
    assert counts.tolist() == [3]
    assert sampled[0, :3].tolist() == [1024, 1024, 1024]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.rejection_sample
def test_rejection_sample_greedy_floating_dtypes(dtype):
    device = _device()
    inputs, expected = _fixture(
        device=device, accepted=(2,), temperature=0.0, dtype=dtype
    )
    sampled, counts = _call_candidate(inputs)
    _assert_valid_prefix(sampled, counts, vocab=17, max_steps=2)
    assert sampled[0, : int(counts[0])].tolist() == expected[0][: int(counts[0])]


@pytest.mark.rejection_sample
def test_rejection_sample_zero_probability_recovery_stays_in_support():
    device = _device()
    vocab = 7
    target = torch.full((2, vocab), float("-inf"), device=device)
    target[0, 1:3] = torch.tensor([0.0, -0.2], device=device)
    target[1, 4:6] = torch.tensor([0.0, -0.2], device=device)
    inputs = {
        "target_logits": target,
        "draft_logits": None,
        "draft_sampled": torch.tensor([0, 0], dtype=torch.int64, device=device),
        "cu_num_logits": torch.tensor([0, 2], dtype=torch.int32, device=device),
        "pos": torch.tensor([31, 32], dtype=torch.int32, device=device),
        "idx_mapping": torch.tensor([0], dtype=torch.int32, device=device),
        "expanded_idx_mapping": torch.tensor([0, 0], dtype=torch.int32, device=device),
        "expanded_local_pos": torch.tensor([0, 1], dtype=torch.int32, device=device),
        "temperature": torch.tensor([1.0], dtype=torch.float32, device=device),
        "seed": torch.tensor([41], dtype=torch.int64, device=device),
        "num_speculative_steps": 1,
    }
    sampled, counts = _call_candidate(inputs)
    _assert_valid_prefix(sampled, counts, vocab=vocab, max_steps=1)
    assert sampled[0, 0].item() in (1, 2)


@pytest.mark.rejection_sample
def test_rejection_sample_full_cache_p_zero_q_positive_rejects():
    device = _device()
    target = torch.tensor(
        [
            [float("-inf"), 0.0, -0.2, float("-inf")],
            [float("-inf"), 0.0, -0.2, float("-inf")],
        ],
        dtype=torch.float32,
        device=device,
    )
    draft = torch.tensor(
        [[[5.0, float("-inf"), float("-inf"), float("-inf")]]],
        dtype=torch.float32,
        device=device,
    )
    inputs = {
        "target_logits": target,
        "draft_logits": draft,
        "draft_sampled": torch.tensor([0, 0], dtype=torch.int64, device=device),
        "cu_num_logits": torch.tensor([0, 2], dtype=torch.int32, device=device),
        "pos": torch.tensor([31, 32], dtype=torch.int32, device=device),
        "idx_mapping": torch.tensor([0], dtype=torch.int32, device=device),
        "expanded_idx_mapping": torch.tensor([0, 0], dtype=torch.int32, device=device),
        "expanded_local_pos": torch.tensor([0, 1], dtype=torch.int32, device=device),
        "temperature": torch.tensor([1.0], dtype=torch.float32, device=device),
        "seed": torch.tensor([41], dtype=torch.int64, device=device),
        "num_speculative_steps": 1,
    }
    sampled, counts = _call_candidate(inputs)
    _assert_valid_prefix(sampled, counts, vocab=4, max_steps=1)
    assert counts.tolist() == [1]
    assert sampled[0, 0].item() in (1, 2)


@pytest.mark.rejection_sample
def test_rejection_sample_full_cache_q_zero_p_positive_accepts():
    device = _device()
    target = torch.tensor(
        [[0.0, -0.2, float("-inf")], [0.0, -0.2, float("-inf")]],
        dtype=torch.float32,
        device=device,
    )
    draft = torch.tensor(
        [[[float("-inf"), 0.0, float("-inf")]]],
        dtype=torch.float32,
        device=device,
    )
    inputs = {
        "target_logits": target,
        "draft_logits": draft,
        "draft_sampled": torch.tensor([0, 0], dtype=torch.int64, device=device),
        "cu_num_logits": torch.tensor([0, 2], dtype=torch.int32, device=device),
        "pos": torch.tensor([31, 32], dtype=torch.int32, device=device),
        "idx_mapping": torch.tensor([0], dtype=torch.int32, device=device),
        "expanded_idx_mapping": torch.tensor([0, 0], dtype=torch.int32, device=device),
        "expanded_local_pos": torch.tensor([0, 1], dtype=torch.int32, device=device),
        "temperature": torch.tensor([1.0], dtype=torch.float32, device=device),
        "seed": torch.tensor([41], dtype=torch.int64, device=device),
        "num_speculative_steps": 1,
    }
    sampled, counts = _call_candidate(inputs)
    _assert_valid_prefix(sampled, counts, vocab=3, max_steps=1)
    assert counts.tolist() == [2]
    assert sampled[0, 0].item() == 0


@pytest.mark.rejection_sample
def test_rejection_sample_does_not_advance_default_rng_or_mutate_state():
    device = _device()
    inputs, _ = _fixture(device=device, accepted=(1, 1), temperature=1.0)
    snapshots = {
        key: value.clone() for key, value in inputs.items() if torch.is_tensor(value)
    }
    from flaggems_vllm.runtime import torch_device_fn

    cpu_rng = torch.random.get_rng_state()
    device_rng = torch_device_fn.get_rng_state().clone()
    first = _call_candidate(inputs)
    second_inputs = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    second = _call_candidate(second_inputs)
    assert torch.equal(first[1], second[1])
    for first_row, second_row, count in zip(first[0], second[0], first[1].tolist()):
        assert torch.equal(first_row[:count], second_row[:count])
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    assert torch.equal(torch_device_fn.get_rng_state(), device_rng)
    for key, value in snapshots.items():
        assert torch.equal(inputs[key], value), f"candidate mutated input {key}"


@pytest.mark.rejection_sample
def test_rejection_sample_batch_order_and_cache_mapping():
    device = _device()
    inputs, expected = _fixture(
        device=device, accepted=(2, 2), temperature=1.0, vocab=17
    )
    # Persistent request slots differ from compact batch rows. Cache rows are
    # addressed through row_mapping rather than idx_mapping.
    inputs["idx_mapping"] = torch.tensor([2, 0], dtype=torch.int32, device=device)
    inputs["expanded_idx_mapping"] = torch.tensor(
        [2, 2, 2, 0, 0, 0], dtype=torch.int32, device=device
    )
    inputs["temperature"] = torch.ones(4, dtype=torch.float32, device=device)
    inputs["seed"] = torch.tensor([71, 73, 79, 83], dtype=torch.int64, device=device)
    inputs["draft_row_mapping"] = torch.tensor(
        [1, 1, 0, 0], dtype=torch.int32, device=device
    )
    sampled, counts = _call_candidate(inputs)
    _assert_valid_prefix(sampled, counts, vocab=17, max_steps=2)
    assert counts.tolist() == [3, 3]
    for request in range(2):
        assert sampled[request, :3].tolist() == expected[request][:3]
    reverse = dict(inputs)
    row_order = torch.tensor([3, 4, 5, 0, 1, 2], device=device)
    for name in (
        "target_logits",
        "draft_sampled",
        "pos",
        "expanded_idx_mapping",
        "expanded_local_pos",
    ):
        reverse[name] = inputs[name][row_order]
    reverse["idx_mapping"] = inputs["idx_mapping"].flip(0)
    permuted, permuted_counts = _call_candidate(reverse)
    assert torch.equal(permuted_counts.flip(0), counts)
    for request, count in enumerate(counts.tolist()):
        assert torch.equal(permuted[1 - request, :count], sampled[request, :count])


@pytest.mark.rejection_sample
@pytest.mark.parametrize(
    "grid,limit,expected",
    [
        ((65535,), 65535, [((65535,), 0)]),
        ((65536,), 65535, [((65535,), 0), ((1,), 65535)]),
        ((131071,), 65535, [((65535,), 0), ((65535,), 65535), ((1,), 131070)]),
        ((32767, 2), 65535, [((32767, 2), 0)]),
        ((32768, 2), 65535, [((65535,), 0), ((1,), 65535)]),
        ((1, 65536), 65535, [((65535,), 0), ((1,), 65535)]),
        ((32768, 2), None, [((32768, 2), 0)]),
    ],
)
def test_rejection_sample_launch_program_limit(grid, limit, expected):
    module = importlib.import_module("flaggems_vllm.ops.rejection_sampler")
    calls = []
    sentinel = object()

    class Kernel:
        def __getitem__(self, launch_grid):
            def launch(*args, **kwargs):
                assert args == (sentinel,)
                assert kwargs.pop("unchanged") is sentinel
                offset = kwargs.pop("program_offset")
                if len(grid) == 2:
                    assert kwargs.pop("FLAT_GRID") is (len(expected) > 1)
                assert not kwargs
                calls.append((launch_grid, offset))

            return launch

    module._launch_kernel(
        Kernel(), grid, sentinel, max_programs=limit, unchanged=sentinel
    )
    assert calls == expected


@pytest.mark.rejection_sample
@pytest.mark.parametrize("full_cache", [False, True])
def test_rejection_sample_launch_chunks_preserve_random_trace(monkeypatch, full_cache):
    _assert_launch_chunks_preserve_random_trace(monkeypatch, full_cache, vocab=1025)


@pytest.mark.rejection_sample
@pytest.mark.parametrize("full_cache", [False, True])
def test_rejection_sample_single_block_launch_chunks_preserve_random_trace(
    monkeypatch, full_cache
):
    _assert_launch_chunks_preserve_random_trace(monkeypatch, full_cache, vocab=257)


@pytest.mark.rejection_sample
@pytest.mark.parametrize("full_cache", [False, True])
def test_rejection_sample_general_launch_chunks_preserve_random_trace(
    monkeypatch, full_cache
):
    _assert_launch_chunks_preserve_random_trace(monkeypatch, full_cache, vocab=2049)


def _assert_launch_chunks_preserve_random_trace(monkeypatch, full_cache, vocab):
    module = importlib.import_module(_candidate().__module__)
    shared = importlib.import_module("flaggems_vllm.ops.rejection_sampler")
    launch = shared._launch_kernel
    device = _device()
    lengths = (3, 2, 1, 3, 0, 2, 3)
    inputs, _ = _fixture(
        device=device,
        accepted=(0, 1, 1, 2, 0, 2, 1),
        variable_lengths=lengths,
        temperature=1.0,
        vocab=vocab,
    )
    generator = torch.Generator().manual_seed(SEED)
    target = torch.randn(
        (sum(length + 1 for length in lengths), vocab), generator=generator
    )
    request_slots = torch.tensor([4, 1, 6, 0, 5, 2, 3], dtype=torch.int32)
    cache_rows = torch.tensor([3, 5, 0, 6, 2, 1, 4], dtype=torch.int32)
    inputs["idx_mapping"] = request_slots.to(device)
    inputs["expanded_idx_mapping"] = torch.repeat_interleave(
        request_slots, torch.tensor(lengths) + 1
    ).to(device)
    inputs["draft_row_mapping"] = cache_rows.to(device)
    inputs["seed"] = (torch.arange(7, dtype=torch.int64) * 17 + 2**40 + 3).to(device)
    inputs["pos"] = (torch.arange(target.shape[0], dtype=torch.int64) * 37 + 100003).to(
        device
    )
    if full_cache:
        draft = torch.zeros((7, 3, vocab))
        start = 0
        for request, length in enumerate(lengths):
            cache_row = int(cache_rows[request_slots[request]])
            draft[cache_row, :length] = target[start : start + length]
            if request in (1, 3):
                # Force a first/middle rejection after an equal-p/q prefix.
                step = request // 2
                proposal = int(inputs["draft_sampled"][start + step + 1])
                target[start + step, proposal] = float("-inf")
                draft[cache_row, step].fill_(-40.0)
                draft[cache_row, step, proposal] = 12.0
            start += length + 1
        inputs["draft_logits"] = draft.to(device)
    else:
        inputs["draft_logits"] = None
    inputs["target_logits"] = target.to(device)
    snapshots = {
        name: value.clone() for name, value in inputs.items() if torch.is_tensor(value)
    }
    # With multiple blocks cap=5 cuts within requests and vocabulary rows.
    # A single block exercises fused stats on Ascend. Both paths require a
    # rejection-kernel chunk and remainder for the seven requests.
    monkeypatch.setattr(
        module,
        "_launch_kernel",
        lambda kernel, grid, *args, **kwargs: launch(
            kernel, grid, *args, max_programs=None, **kwargs
        ),
    )
    unsplit, unsplit_counts = _call_candidate(inputs)
    monkeypatch.setattr(
        module,
        "_launch_kernel",
        lambda kernel, grid, *args, **kwargs: launch(
            kernel, grid, *args, max_programs=5, **kwargs
        ),
    )
    split, split_counts = _call_candidate(inputs)
    _assert_valid_prefix(split, split_counts, vocab=vocab, max_steps=3)
    assert torch.equal(split_counts, unsplit_counts)
    for request, count in enumerate(split_counts.tolist()):
        assert torch.equal(split[request, :count], unsplit[request, :count])
    if full_cache:
        assert split_counts.tolist() == [4, 1, 2, 2, 1, 3, 4]
    for name, value in snapshots.items():
        assert torch.equal(inputs[name], value), f"candidate mutated input {name}"


@pytest.mark.rejection_sample
def test_rejection_sample_source_murmur_zero_hash_tail():
    # Frozen vLLM Murmur3(8, 54108339, token=0) == 0. Its midpoint uniform is
    # 2**-33; adding the tl.rand lower clamp changes the winner for this case.
    device = _device()
    inputs, _ = _fixture(device=device, accepted=(0,), variable_lengths=(0,))
    inputs["target_logits"] = torch.tensor([[-22.7, 0.0]], device=device)
    inputs["temperature"] = torch.tensor([1.0], device=device)
    inputs["seed"] = torch.tensor([8], dtype=torch.int64, device=device)
    inputs["pos"] = torch.tensor([54108339], dtype=torch.int64, device=device)
    sampled, counts = _call_candidate(inputs)
    assert counts.tolist() == [1]
    assert sampled[0, 0].item() == 0


def test_rejection_sample_oracle_greedy_and_residual_cpu():
    target = torch.tensor(
        [[[8.0, 0.0, -3.0], [0.0, 8.0, -3.0]], [[0.0, 8.0, -3.0], [8.0, 0.0, -3.0]]]
    )
    draft = torch.tensor([[0], [0]])
    sampled, counts, defined = oracle_rejection_sample(
        target, draft, torch.tensor([0.0, 0.0])
    )
    assert counts.tolist() == [2, 1]
    assert sampled[0, :2].tolist() == [0, 1]
    assert sampled[1, 0].item() == 1
    assert bool(defined[0, :2].all()) and bool(defined[1, 0])

    stochastic_target = torch.tensor([[[0.0, -1.0, -4.0], [0.0, -1.0, -4.0]]])
    stochastic_draft = torch.tensor([[0]])
    residual_sampled, residual_counts, _ = oracle_rejection_sample(
        stochastic_target,
        stochastic_draft,
        torch.tensor([1.0]),
        acceptance_uniforms=torch.tensor([[1.0]]),
        resample_uniforms=torch.tensor([0.95]),
    )
    assert residual_counts.tolist() == [1]
    assert residual_sampled[0, 0].item() in (1, 2)


# Frozen before candidate execution, as specified by validator-plan-20261119.
STATISTICS_N = 131072
STATISTICS_SEEDS = (20261119, 20261123)
STATISTICS_ALPHA = 0.01
STATISTICS_FAMILY_SIZE = 24


def _distribution_inputs(p_rows, q_rows, proposals, fixed_seed, device):
    """Pack immutable CPU probability fixtures; proposals come from PCG64."""
    n, steps = proposals.shape
    rows, vocab = p_rows.shape
    assert rows == steps + 1
    target = p_rows.log().float().repeat(n, 1).to(device)
    draft = None if q_rows is None else q_rows.log().float().repeat(n, 1, 1).to(device)
    packed = torch.zeros((n, rows), dtype=torch.int64)
    packed[:, 1:] = torch.as_tensor(proposals, dtype=torch.int64)
    return {
        "target_logits": target,
        "draft_logits": draft,
        "draft_sampled": packed.flatten().to(device),
        "cu_num_logits": (torch.arange(n + 1, dtype=torch.int32) * rows).to(device),
        "pos": (torch.arange(n * rows, dtype=torch.int64) + 101).to(device),
        "idx_mapping": torch.arange(n, dtype=torch.int32, device=device),
        "expanded_idx_mapping": torch.arange(
            n, dtype=torch.int32, device=device
        ).repeat_interleave(rows),
        "expanded_local_pos": torch.arange(
            rows, dtype=torch.int32, device=device
        ).repeat(n),
        "temperature": torch.ones(n, dtype=torch.float32, device=device),
        "seed": (torch.arange(n, dtype=torch.int64) + fixed_seed).to(device),
        "num_speculative_steps": steps,
    }


def _categorical_record(name, seed, values, probabilities, scipy_stats):
    """FP64 bins, exact zero-support checks, frozen expected-count merge rule."""
    probabilities = probabilities.double().cpu()
    probabilities /= probabilities.sum()
    observed = torch.bincount(values.cpu(), minlength=probabilities.numel()).double()
    assert values.numel() > 0, f"empty statistical population: {name}"
    assert bool(torch.all(observed[probabilities == 0] == 0)), name
    expected = probabilities * values.numel()
    large = expected >= 20
    small = (expected > 0) & ~large
    obs_bins, exp_bins = observed[large], expected[large]
    if bool(small.any()):
        obs_bins = torch.cat((obs_bins, observed[small].sum().reshape(1)))
        exp_bins = torch.cat((exp_bins, expected[small].sum().reshape(1)))
    assert obs_bins.numel() >= 2
    statistic = float(((obs_bins - exp_bins).square() / exp_bins).sum())
    pvalue = float(scipy_stats.chi2.sf(statistic, obs_bins.numel() - 1))
    return dict(
        case=name,
        seed=seed,
        N=values.numel(),
        observed=observed.tolist(),
        expected=expected.tolist(),
        statistic=statistic,
        pvalue=pvalue,
    )


def _holm_decisions(
    records, alpha=STATISTICS_ALPHA, family_size=STATISTICS_FAMILY_SIZE
):
    assert 0 < len(records) <= family_size
    rejecting = True
    for rank, record in enumerate(sorted(records, key=lambda row: row["pvalue"])):
        record["holm_threshold"] = alpha / (family_size - rank)
        record["rejected"] = rejecting and record["pvalue"] <= record["holm_threshold"]
        if not record["rejected"]:
            rejecting = False
    return [record for record in records if record["rejected"]]


def test_rejection_sample_holm_decision_cpu():
    rows = [dict(pvalue=0.5) for _ in range(24)]
    rows[0]["pvalue"] = 0.00001
    assert _holm_decisions(rows) == [rows[0]]


@pytest.mark.rejection_sample
@pytest.mark.skipif(
    os.environ.get("FLAGGEMS_REJECTION_STATISTICS") != "1",
    reason="set FLAGGEMS_REJECTION_STATISTICS=1 for frozen N=131072, M=24 suite",
)
def test_rejection_sample_statistical_target_and_residual_laws(record_property):
    """Frozen S1--S6 suite: two PCG64 seeds and one Holm family (M=24).

    S6 has exact zero-support assertions attached to the S2/S3 runs.  They
    deliberately do not consume additional Holm slots: the validator plan
    treats masked support as a hard contract in addition to the S2/S3 laws.
    """
    import json

    import numpy as np
    from scipy import stats

    device = _device()
    n = STATISTICS_N
    p0 = torch.tensor([0.30, 0.22, 0.18, 0.12, 0.10, 0.05, 0.03], dtype=torch.float64)
    p1 = torch.tensor([0.07, 0.11, 0.15, 0.21, 0.19, 0.17, 0.10], dtype=torch.float64)
    q0 = torch.tensor([0.12, 0.18, 0.24, 0.16, 0.14, 0.10, 0.06], dtype=torch.float64)
    q1 = torch.tensor([0.25, 0.20, 0.15, 0.13, 0.11, 0.09, 0.07], dtype=torch.float64)
    masked_p = torch.tensor(
        [0.0, 0.28, 0.23, 0.18, 0.14, 0.10, 0.07], dtype=torch.float64
    )
    masked_q = torch.tensor(
        [0.25, 0.18, 0.14, 0.13, 0.10, 0.11, 0.09], dtype=torch.float64
    )
    records = []
    for fixed_seed in STATISTICS_SEEDS:
        rng = np.random.Generator(np.random.PCG64(fixed_seed))
        # S1: four full-acceptance bonus assertions (two distributions, K=1/3).
        for steps in (1, 3):
            for row_id, p in enumerate((p0, p1)):
                proposals = rng.choice(p.numel(), size=(n, steps), p=p.numpy())
                inputs = _distribution_inputs(
                    p.repeat(steps + 1, 1),
                    p.repeat(steps, 1),
                    proposals,
                    fixed_seed,
                    device,
                )
                sampled, counts = _call_candidate(inputs)
                assert bool(torch.all(counts == steps + 1))
                assert torch.equal(sampled[:, :steps].cpu(), torch.as_tensor(proposals))
                records.append(
                    _categorical_record(
                        f"S1_bonus_K{steps}_p{row_id}",
                        fixed_seed,
                        sampled[:, steps].cpu(),
                        p,
                        stats,
                    )
                )
        # S2/S6: full-cache forced rejection, with exact masked support checks.
        proposals = np.zeros((n, 1), dtype=np.int64)
        inputs = _distribution_inputs(
            masked_p.repeat(2, 1), masked_q[None, :], proposals, fixed_seed, device
        )
        sampled, counts = _call_candidate(inputs)
        assert bool(torch.all(counts == 1))
        residual = (masked_p - masked_q).clamp_min(0)
        # S6 full-logit masked support: p[0] == 0 is an exact zero bin.
        assert bool(torch.all(sampled[:, 0] != 0))
        records.append(
            _categorical_record(
                "S2_full_residual", fixed_seed, sampled[:, 0].cpu(), residual, stats
            )
        )
        # S3: one-hot proposal has positive p; condition only on rejected rows.
        inputs = _distribution_inputs(
            p0.repeat(2, 1), None, proposals, fixed_seed, device
        )
        sampled, counts = _call_candidate(inputs)
        residual = p0.clone()
        residual[0] = 0
        records.append(
            _categorical_record(
                "S3_onehot_residual",
                fixed_seed,
                sampled[counts == 1, 0].cpu(),
                residual,
                stats,
            )
        )

        # S6 one-hot masked support.  The proposal is token 0, so p[0] == 0
        # forces rejection and the recovery distribution must still exclude
        # token 0.  This hard assertion is intentionally outside the Holm
        # family; the S3 residual record above covers its statistical slot.
        masked_onehot = _distribution_inputs(
            masked_p.repeat(2, 1), None, proposals, fixed_seed, device
        )
        masked_sampled, masked_counts = _call_candidate(masked_onehot)
        assert bool(torch.all(masked_counts == 1))
        assert bool(torch.all(masked_sampled[:, 0] != 0))

        # S4: three K=1 acceptance-rate cases.  The analytic acceptance
        # probability is A=sum(min(p,q)); exact binomial p-values join the
        # same fixed M=24 Holm family as categorical records.
        for pair_id, (target_p, draft_q) in enumerate(
            ((p0, q0), (p1, q1), (masked_p, masked_q))
        ):
            pair_proposals = rng.choice(draft_q.numel(), size=(n, 1), p=draft_q.numpy())
            pair_inputs = _distribution_inputs(
                target_p.repeat(2, 1),
                draft_q[None, :],
                pair_proposals,
                fixed_seed,
                device,
            )
            _, pair_counts = _call_candidate(pair_inputs)
            successes = int((pair_counts == 2).sum().item())
            acceptance = float(torch.minimum(target_p, draft_q).sum())
            pvalue = float(stats.binomtest(successes, n=n, p=acceptance).pvalue)
            records.append(
                dict(
                    case=f"S4_acceptance_pair{pair_id}",
                    seed=fixed_seed,
                    N=n,
                    successes=successes,
                    observed_rate=successes / n,
                    target_rate=acceptance,
                    absolute_error=abs(successes / n - acceptance),
                    pvalue=pvalue,
                )
            )

        # S5: independently generated per-step (p_j,q_j) pairs.  For every
        # emitted position j, condition on num_sampled >= j+1 and check the
        # target marginal p_j.  This includes accepted proposals and recovery
        # tokens after a partial survival, as required by validator-plan.
        s5_target = torch.stack((p0, p1, masked_p, p0))
        s5_draft = torch.stack((q0, q1, masked_q))
        s5_proposals = np.column_stack(
            [
                rng.choice(row.numel(), size=n, p=row.numpy())
                for row in (q0, q1, masked_q)
            ]
        )
        s5_inputs = _distribution_inputs(
            s5_target, s5_draft, s5_proposals, fixed_seed, device
        )
        s5_sampled, s5_counts = _call_candidate(s5_inputs)
        for position in range(3):
            eligible = s5_counts >= position + 1
            assert bool(torch.any(eligible))
            records.append(
                _categorical_record(
                    f"S5_position_{position + 1}",
                    fixed_seed,
                    s5_sampled[eligible, position].cpu(),
                    s5_target[position],
                    stats,
                )
            )
    # The complete pre-registered S1--S5 statistical family is exactly 24
    # records across the two fixed seeds; fail closed if a case disappears.
    assert len(records) == STATISTICS_FAMILY_SIZE
    rejected = _holm_decisions(records)
    result = dict(
        source=SOURCE_COMMIT,
        seeds=list(STATISTICS_SEEDS),
        N=n,
        alpha=STATISTICS_ALPHA,
        family_size=STATISTICS_FAMILY_SIZE,
        records=records,
    )
    # Always publish the complete fixed-seed result before raising a failure.
    record_property("rejection_statistics", json.dumps(result))
    from tests import conftest as test_config

    if test_config.RECORD_JSON:
        test_config.TEST_RESULTS.setdefault("rejection_sample", {}).setdefault(
            "statistics", []
        ).append(result)
    print(json.dumps(result, sort_keys=True))
    assert not rejected, f"Holm-rejected fixed-seed assertions: {rejected}"


@pytest.mark.rejection_sample
def test_rejection_sample_backend_dispatch():
    import inspect

    import flaggems_vllm

    shared = importlib.import_module("flaggems_vllm.ops.rejection_sampler")
    expected = (
        "flaggems_vllm.runtime.backend._ascend.ops.rejection_sampler"
        if flaggems_vllm.vendor_name == "ascend"
        else "flaggems_vllm.ops.rejection_sampler"
    )
    assert flaggems_vllm.rejection_sample.__module__ == expected
    assert inspect.signature(flaggems_vllm.rejection_sample) == inspect.signature(
        shared.rejection_sample
    )
