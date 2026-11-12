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

"""Public mHC post capture/replay with independent CPU attention-free math."""

import importlib

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.mhc.mhc_post import MixLayout

pytestmark = [
    pytest.mark.mhc_post,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA-compatible graph device required"
    ),
]


def _host_inputs(n, h, seed):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, h, generator=generator).bfloat16(),
        torch.randn(n, 4, h, generator=generator).bfloat16(),
        torch.randn(n, 4, 1, generator=generator),
        torch.randn(n, 4, 4, generator=generator),
    )


def _reference(host):
    x, residual, post, mix = host
    return (post * x.float().unsqueeze(1) + torch.bmm(mix, residual.float())).bfloat16()


@pytest.mark.parametrize("n,h", [(1, 129), (8, 3584)])
@pytest.mark.parametrize("mix_layout", [MixLayout.NATURAL, MixLayout.TRANSPOSED])
@pytest.mark.parametrize("strided_mix", [False, True])
@torch.inference_mode()
def test_mhc_post_changed_input_graph(monkeypatch, n, h, mix_layout, strided_mix):
    host = _host_inputs(n, h, 1134)
    x, residual, post = [t.to(flaggems_vllm.device) for t in host[:3]]
    wide = torch.empty((n, 4, 5 if strided_mix else 4), device=x.device)
    mix = wide[..., 1:] if strided_mix else wide
    if mix_layout is MixLayout.TRANSPOSED:
        mix = mix.transpose(-1, -2)
        if not strided_mix:
            mix = mix.contiguous()
    assert mix.is_contiguous() is (not strided_mix)
    hits = []
    if flaggems_vllm.runtime.device.vendor_name == "thead":
        module = importlib.import_module(
            "flaggems_vllm.runtime.backend._thead.ops.mhc_post"
        )
        actual = module.mhc_post_hc4_tokloop

        def observed(*args, **kwargs):
            hits.append(1)
            return actual(*args, **kwargs)

        monkeypatch.setattr(module, "mhc_post_hc4_tokloop", observed)

    def update(values):
        for target, source in zip((x, residual, post), values[:3]):
            target.copy_(source)
        logical = values[3]
        mix.copy_(
            logical if mix_layout is MixLayout.NATURAL else logical.transpose(-1, -2)
        )

    def invoke():
        return flaggems_vllm.mhc_post(x, residual, post, mix, mix_layout=mix_layout)

    update(host)
    eager = invoke()
    torch.testing.assert_close(eager.cpu(), _reference(host), rtol=1e-2, atol=0.03125)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = invoke()
    if flaggems_vllm.runtime.device.vendor_name == "thead":
        assert hits, "public API did not enter the real PPU launcher"
    for seed in (1135, 1136):
        changed = _host_inputs(n, h, seed)
        update(changed)
        output.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.cpu(), _reference(changed), rtol=1e-2, atol=0.03125
        )
