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

import pytest
import torch

import flaggems_vllm

try:
    from vllm.models.deepseek_v4.common.ops import (
        dequantize_and_gather_k_cache as vllm_dequantize_and_gather_k_cache,
    )

    # The vllm baseline kernel does not compile on every backend (pointer bitcast on Ascend).
    try:
        vllm_dequantize_and_gather_k_cache(
            torch.zeros((1, 1, 64), dtype=torch.bfloat16, device=flaggems_vllm.device),
            torch.zeros((1, 1024), dtype=torch.uint8, device=flaggems_vllm.device),
            torch.full((1,), 1, dtype=torch.int32, device=flaggems_vllm.device),
            torch.full((1,), 1, dtype=torch.int32, device=flaggems_vllm.device),
            torch.zeros((1, 1), dtype=torch.int32, device=flaggems_vllm.device),
            64,
            0,
        )
        flaggems_vllm.runtime.torch_device_fn.synchronize()
        _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE = True
    except Exception:
        _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE = False
except Exception:
    vllm_dequantize_and_gather_k_cache = None
    _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE = False


def _op_runs():
    """Probe once that this backend can actually run the operator."""
    try:
        _nope, _rope = 448, 64
        _ss = (_nope + 63) // 64 + 1
        _bs = 64
        _stride = _bs * (_nope + _rope * 2) + _bs * _ss
        out = torch.zeros(
            (1, 1, _nope + _rope), dtype=torch.bfloat16, device=flaggems_vllm.device
        )
        k_cache = torch.zeros(
            (1, _stride), dtype=torch.uint8, device=flaggems_vllm.device
        )
        flaggems_vllm.dequantize_and_gather_k_cache(
            out,
            k_cache,
            torch.full((1,), 1, dtype=torch.int32, device=flaggems_vllm.device),
            torch.full((1,), 1, dtype=torch.int32, device=flaggems_vllm.device),
            torch.zeros((1, 1), dtype=torch.int32, device=flaggems_vllm.device),
            _bs,
            rope_dim=_rope,
            nope_dim=_nope,
            scale_slots=_ss,
        )
        flaggems_vllm.runtime.torch_device_fn.synchronize()
        return True
    except Exception:
        return False


_OP_RUNS = _op_runs()

from . import base  # noqa: E402

_E4M3_LUT = None
_UE8M0_LUT = None


def _e4m3_lut(device):
    global _E4M3_LUT
    if _E4M3_LUT is None or _E4M3_LUT.device != device:
        i = torch.arange(256, dtype=torch.int32)
        e = (i >> 3) & 0x0F
        man = i & 0x07
        mag = torch.where(
            e == 0,
            man.to(torch.float32) * (2.0**-9),
            torch.pow(2.0, (e - 7).to(torch.float32))
            * (1.0 + man.to(torch.float32) / 8.0),
        )
        _E4M3_LUT = torch.where((i & 0x80) != 0, -mag, mag).to(device)
    return _E4M3_LUT


def _ue8m0_lut(device):
    global _UE8M0_LUT
    if _UE8M0_LUT is None or _UE8M0_LUT.device != device:
        code = (
            ((torch.arange(256, dtype=torch.int32) - 127) + 127)
            .clamp(0, 255)
            .to(torch.int32)
        )
        _UE8M0_LUT = (code << 23).contiguous().view(torch.float32).to(device)
    return _UE8M0_LUT


def _e4m3_bytes_to_f32(u8):
    return torch.index_select(_e4m3_lut(u8.device), 0, u8.to(torch.int64))


def _exact_pow2(exponent):
    return torch.index_select(
        _ue8m0_lut(exponent.device), 0, exponent.to(torch.int64).clamp(0, 255)
    )


def torch_dequantize_and_gather(
    k_cache,
    seq_lens,
    gather_lens,
    block_table,
    block_size,
    offset=0,
    rope_dim=64,
    nope_dim=448,
    scale_slots=8,
):
    token_data_size = nope_dim + rope_dim * 2
    batch = seq_lens.shape[0]
    gc = gather_lens if gather_lens is not None else seq_lens
    n_slots = (nope_dim + 63) // 64
    dev = k_cache.device

    gl = gc.to(torch.int64).detach().cpu()
    sl_ = seq_lens.to(torch.int64).detach().cpu()
    max_gl = int(gl.max().item())
    req = torch.arange(batch, dtype=torch.int64).repeat_interleave(gl)
    li = torch.cat([torch.arange(int(g), dtype=torch.int64) for g in gl])
    pos = sl_[req] - gl[req] + li
    blk = pos // block_size
    pib = pos % block_size
    phys = block_table.to(torch.int64).detach().cpu()[req, blk]

    data_off = phys * token_data_size + pib * token_data_size
    scale_off = (
        phys * token_data_size + block_size * token_data_size + pib * scale_slots
    )
    n_tok = int(req.numel())

    kc = k_cache
    d_idx = data_off.to(dev)
    s_idx = scale_off.to(dev)
    ar = torch.arange(nope_dim, device=dev).unsqueeze(0)
    flat = kc.reshape(-1)
    xb = flat[(d_idx.unsqueeze(1) + ar).reshape(-1)].reshape(n_tok, nope_dim)
    x = (
        torch.index_select(_e4m3_lut(dev), 0, xb.reshape(-1).to(torch.int64))
        .reshape(n_tok, nope_dim)
        .to(torch.float32)
    )

    sar = torch.arange(n_slots, device=dev).unsqueeze(0)
    sb = flat[(s_idx.unsqueeze(1) + sar).reshape(-1)].reshape(n_tok, n_slots)
    scale = (
        torch.index_select(_ue8m0_lut(dev), 0, sb.reshape(-1).to(torch.int64))
        .reshape(n_tok, n_slots)
        .repeat_interleave(64, dim=1)[:, :nope_dim]
    )

    rar = torch.arange(rope_dim * 2, device=dev).unsqueeze(0)
    rb = flat[(d_idx.unsqueeze(1) + nope_dim + rar).reshape(-1)].reshape(
        n_tok, rope_dim * 2
    )
    rope = rb.contiguous().view(torch.bfloat16)  # [n_tok, rope_dim] bf16, by bytes

    out = torch.zeros(
        (batch, max_gl, nope_dim + rope_dim), dtype=torch.bfloat16, device=dev
    )
    out.view(batch * max_gl, nope_dim + rope_dim)[
        req * max_gl + li + offset, :nope_dim
    ] = (x * scale).to(torch.bfloat16)
    out.view(batch * max_gl, nope_dim + rope_dim)[
        req * max_gl + li + offset, nope_dim:
    ] = rope
    return out


class DequantizeAndGatherKCacheBenchmark(base.Benchmark):
    def __init__(self):
        def baseline_adapter(
            out,
            k_cache,
            seq_lens,
            gather_lens,
            block_table,
            block_size,
            offset=0,
            rope_dim=64,
            nope_dim=None,
            scale_slots=None,
        ):
            if _HAS_VLLM_DEQUANTIZE_AND_GATHER_K_CACHE:
                return vllm_dequantize_and_gather_k_cache(
                    out,
                    k_cache,
                    seq_lens,
                    gather_lens,
                    block_table,
                    block_size,
                    offset,
                )
            return torch_dequantize_and_gather(
                k_cache,
                seq_lens,
                gather_lens,
                block_table,
                block_size,
                offset,
                rope_dim,
                nope_dim if nope_dim is not None else 448,
                scale_slots if scale_slots is not None else 8,
            )

        super().__init__(
            "dequantize_and_gather_k_cache",
            baseline_adapter,
            [torch.bfloat16],
            gems_op=flaggems_vllm.dequantize_and_gather_k_cache,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [
            (1, 512, 128, 512, 448, 64),
            (2, 1024, 256, 512, 448, 64),
            (4, 2048, 512, 512, 448, 64),
            (4, 2048, 2048, 512, 448, 64),
            (8, 4096, 1024, 512, 448, 64),
        ]

    def get_input_iter(self, dtype):
        _ = dtype
        for batch, seq_len, gather_len, dim, nope_dim, rope_dim in self.shapes:
            scale_slots = (nope_dim + 63) // 64 + (1 if nope_dim % 64 == 0 else 0)
            block_size = 64
            token_data_size = nope_dim + rope_dim * 2
            block_stride = block_size * token_data_size + block_size * scale_slots
            num_blocks = batch * ((seq_len + block_size - 1) // block_size)
            out = torch.empty(
                (batch, gather_len, dim),
                device=flaggems_vllm.device,
                dtype=torch.bfloat16,
            )
            k_cache = torch.zeros(
                (num_blocks, block_stride),
                device=flaggems_vllm.device,
                dtype=torch.uint8,
            )
            seq_lens = torch.full(
                (batch,), seq_len, device=flaggems_vllm.device, dtype=torch.int32
            )
            gather_lens = torch.full(
                (batch,), gather_len, device=flaggems_vllm.device, dtype=torch.int32
            )
            block_table = torch.arange(
                num_blocks, device=flaggems_vllm.device, dtype=torch.int32
            ).view(batch, -1)
            yield (
                out,
                k_cache,
                seq_lens,
                gather_lens,
                block_table,
                block_size,
                0,
                rope_dim,
                nope_dim,
                scale_slots,
            )


@pytest.mark.dequantize_and_gather_k_cache
@pytest.mark.skipif(
    not _OP_RUNS,
    reason="requires an E4M3-casting backend",
)
def test_dequantize_and_gather_k_cache_benchmark():
    DequantizeAndGatherKCacheBenchmark().run()
