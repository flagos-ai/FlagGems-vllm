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

import math

import pytest
import torch

import flaggems_vllm
from flaggems_vllm import fp8_einsum

from . import conftest as cfg

DEFAULT_BLOCK_SHAPE = [128, 128]


def is_cuda_available():
    if flaggems_vllm.vendor_name != "nvidia" or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()
pytestmark = [
    pytest.mark.fp8_einsum,
    pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires NVIDIA Hopper GPU"),
]


# (h, r, d) groups -- r and d must be divisible by the 128 block grid.
_HRD_GROUPS = {
    "flash": (8, 4096, 1024),
    "pro": (16, 7168, 1024),
}
_BATCH_SIZES = (1, 4, 8, 16, 32, 64, 128)
if not cfg.QUICK_MODE:
    _BATCH_SIZES += (4096, 8192, 16384, 32768)

# (b, h, r, d)
FP8_EINSUM_CONFIGS = [
    (b, h, r, d) for (h, r, d) in _HRD_GROUPS.values() for b in _BATCH_SIZES
]


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round FP32 scales up to the nearest power-of-two (UE8M0 grid)."""
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def per_token_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool = True, gran_k: int = 128):
    assert x.dim() == 2
    m, n = x.shape
    padded_n = math.ceil(n / gran_k) * gran_k
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (
        (x_view * (1.0 / sf.unsqueeze(2)))
        .to(torch.float8_e4m3fn)
        .view(m, padded_n)[:, :n]
        .contiguous()
    )
    return x_fp8, sf


def per_block_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool = True, gran_k: int = 128):
    assert x.dim() == 2
    m, n = x.shape
    padded_m = math.ceil(m / gran_k) * gran_k
    padded_n = math.ceil(n / gran_k) * gran_k
    x_padded = torch.zeros((padded_m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, gran_k, x_padded.size(1) // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return (
        x_scaled.view_as(x_padded)[:m, :n].contiguous(),
        sf.view(x_view.size(0), x_view.size(2)),
    )


def _make_fp8_einsum_inputs(b, h, r, d, block_shape, device, seed=0):
    """Build block-wise FP8 ``bhr,hdr->bhd`` inputs (per-token x, per-block y)."""
    block_n, block_k = block_shape
    torch.manual_seed(seed)
    x = torch.randn((b, h, r), device=device, dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device=device, dtype=torch.bfloat16)

    x_fp8 = per_token_cast_to_fp8(x.view(-1, r), use_ue8m0=True, gran_k=block_k)
    x_data = x_fp8[0].view(b, h, r)
    x_scale = x_fp8[1].view(b, h, math.ceil(r / block_k))

    y_data = torch.empty_like(y, dtype=torch.float8_e4m3fn)
    y_scale = torch.empty(
        (h, math.ceil(d / block_n), math.ceil(r / block_k)),
        device=device,
        dtype=torch.float32,
    )
    for i in range(h):
        y_data[i], y_scale[i] = per_block_cast_to_fp8(y[i], use_ue8m0=True)

    return x_data, x_scale, y_data, y_scale


def torch_fp8_block_einsum_reference(x_data, x_scale, y_data, y_scale, block_shape):
    """Pure-PyTorch reference: dequantize the block-wise FP8 inputs, then einsum.

    Mirrors the kernel math (FP8 data scaled by the block grid, FP32 accumulation):
        out[b,h,d] = sum_r x[b,h,r] * y[h,d,r]
    """
    block_n, block_k = block_shape
    b, h, r = x_data.shape
    d = y_data.shape[1]

    x_f = x_data.to(torch.float32)
    y_f = y_data.to(torch.float32)

    k_tiles = x_scale.shape[-1]
    n_tiles = y_scale.shape[1]

    x_deq = torch.empty_like(x_f)
    for kt in range(k_tiles):
        ks, ke = kt * block_k, min((kt + 1) * block_k, r)
        x_deq[:, :, ks:ke] = x_f[:, :, ks:ke] * x_scale[:, :, kt : kt + 1]

    y_deq = torch.empty_like(y_f)
    for nt in range(n_tiles):
        ns, ne = nt * block_n, min((nt + 1) * block_n, d)
        for kt in range(k_tiles):
            ks, ke = kt * block_k, min((kt + 1) * block_k, r)
            y_deq[:, ns:ne, ks:ke] = y_f[:, ns:ne, ks:ke] * y_scale[:, nt, kt].view(
                h, 1, 1
            )

    return torch.einsum("bhr,hdr->bhd", x_deq, y_deq)


@pytest.mark.parametrize("config", FP8_EINSUM_CONFIGS)
@pytest.mark.parametrize("block_shape", [[128, 128]])
def test_accuracy_fp8_einsum(config, block_shape):
    """Validate FlagGems fp8_einsum against a dequantized PyTorch reference."""
    b, h, r, d = config
    device = flaggems_vllm.device

    x_data, x_scale, y_data, y_scale = _make_fp8_einsum_inputs(
        b, h, r, d, block_shape, device
    )

    result = fp8_einsum(
        "bhr,hdr->bhd",
        x_data,
        x_scale,
        y_data,
        y_scale,
        block_size=block_shape,
    )

    ref = torch_fp8_block_einsum_reference(
        x_data, x_scale, y_data, y_scale, block_shape
    )

    torch.cuda.synchronize()

    assert result.shape == (b, h, d)
    # FP8 block-wise quantization + bf16 output accumulate rounding error.
    rtol = 2e-1
    atol = max(5e-2, ref.abs().max().item() * 5e-2)
    torch.testing.assert_close(result, ref.to(result.dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["contiguous", "vllm", "strided", "unaligned"])
def test_fp8_einsum_out_and_layout(dtype, layout):
    b, h, r, d = 3, 2, 256, 128
    x, xs, y, ys = _make_fp8_einsum_inputs(b, h, r, d, [128, 128], "cuda")
    if layout == "vllm":
        x = x.permute(1, 0, 2).contiguous().permute(1, 0, 2)
        scales = torch.empty_strided(xs.shape, (1, 8, 4), device="cuda")
        scales.copy_(xs)
        xs = scales
    elif layout == "strided":
        y = y.transpose(1, 2).contiguous().transpose(1, 2)
        # Keep K contiguous but make the head and scale grids non-contiguous.
        y_buf = torch.empty((h, d * 2, r), device="cuda", dtype=y.dtype)
        y_buf[:, ::2, :] = y
        y = y_buf[:, ::2, :]
        ys_buf = torch.empty((*ys.shape[:-1], ys.shape[-1] * 2), device="cuda")
        ys_buf[..., ::2] = ys
        ys = ys_buf[..., ::2]
    elif layout == "unaligned":
        x_buf = torch.empty((b, h, r + 1), device="cuda", dtype=x.dtype)
        x_buf[..., 1:] = x
        x = x_buf[..., 1:]
    out_buf = torch.full((b, h, d + 16), float("nan"), dtype=dtype, device="cuda")
    out = out_buf[..., :d]
    result = fp8_einsum("bhr,hdr->bhd", x, xs, y, ys, output_dtype=dtype, out=out)
    assert result is out
    ref = torch_fp8_block_einsum_reference(x, xs, y, ys, [128, 128])
    torch.testing.assert_close(result, ref.to(dtype), rtol=2e-2, atol=0.25)
    assert torch.isnan(out_buf[..., d:]).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fp8_einsum_splitk_repeated(dtype):
    inputs = _make_fp8_einsum_inputs(3, 1, 4096, 128, [128, 128], "cuda")
    out = torch.full((3, 1, 128), 123.0, device="cuda", dtype=dtype)
    ref = torch_fp8_block_einsum_reference(*inputs, [128, 128])
    for _ in range(3):
        fp8_einsum("bhr,hdr->bhd", *inputs, output_dtype=dtype, out=out)
        torch.testing.assert_close(out, ref.to(dtype), rtol=2e-1, atol=2.0)


def test_fp8_einsum_without_tle(monkeypatch):
    import importlib

    bmm = importlib.import_module(
        "flaggems_vllm.runtime.backend._nvidia.hopper.ops.w8a8_block_fp8_bmm"
    )
    monkeypatch.setattr(bmm, "HAS_TLE_W8A8_BLOCK_FP8_BMM", False)
    inputs = _make_fp8_einsum_inputs(3, 2, 256, 128, [128, 128], "cuda")
    result = fp8_einsum("bhr,hdr->bhd", *inputs)
    ref = torch_fp8_block_einsum_reference(*inputs, [128, 128])
    torch.testing.assert_close(result, ref.to(result.dtype), rtol=2e-2, atol=0.25)


@pytest.mark.parametrize(
    "shape", [(0, 2, 128, 128), (2, 0, 128, 128), (2, 2, 128, 0), (2, 2, 0, 128)]
)
def test_fp8_einsum_empty(shape):
    b, h, r, d = shape
    x = torch.empty((b, h, r), device="cuda", dtype=torch.float8_e4m3fn)
    xs = torch.empty((b, h, r // 128), device="cuda")
    y = torch.empty((h, d, r), device="cuda", dtype=torch.float8_e4m3fn)
    ys = torch.empty((h, d // 128, r // 128), device="cuda")
    out = torch.full((b, h, d), float("nan"), device="cuda", dtype=torch.bfloat16)
    result = fp8_einsum("bhr,hdr->bhd", x, xs, y, ys, out=out)
    assert result is out
    torch.testing.assert_close(result, torch.zeros_like(result))


@pytest.mark.parametrize(
    "case",
    [
        "equation",
        "block_size",
        "dtype",
        "scales",
        "scale_shape",
        "shape",
        "stride",
        "out_shape",
        "out_dtype",
        "out_overlap",
        "out_alias",
        "grad",
    ],
)
def test_fp8_einsum_invalid_inputs(case):
    x, xs, y, ys = _make_fp8_einsum_inputs(2, 2, 256, 128, [128, 128], "cuda")
    equation, kwargs = "bhr,hdr->bhd", {}
    error = ValueError
    if case == "equation":
        equation, error = "abc,cde->abe", NotImplementedError
    elif case == "block_size":
        kwargs, error = {"block_size": [64, 128]}, NotImplementedError
    elif case == "dtype":
        x, error = x.to(torch.bfloat16), NotImplementedError
    elif case == "scales":
        xs, error = xs.to(torch.int32), NotImplementedError
    elif case == "scale_shape":
        xs = xs[..., :1]
    elif case == "shape":
        y = y[:1]
    elif case == "stride":
        x, error = x.transpose(0, 2).contiguous().transpose(0, 2), NotImplementedError
    elif case == "out_shape":
        kwargs["out"] = torch.empty((2, 2, 64), device="cuda", dtype=torch.bfloat16)
    elif case == "out_dtype":
        kwargs["out"] = torch.empty((2, 2, 128), device="cuda", dtype=torch.float32)
    elif case == "out_overlap":
        kwargs["out"] = torch.empty(
            (1, 2, 128), device="cuda", dtype=torch.bfloat16
        ).expand(2, -1, -1)
    elif case == "out_alias":
        kwargs["out"] = x.view(torch.bfloat16)
    elif case == "grad":
        xs, error = xs.requires_grad_(), NotImplementedError
    with pytest.raises(error):
        fp8_einsum(equation, x, xs, y, ys, **kwargs)


def test_fp8_einsum_vllm_reference():
    deep_gemm = pytest.importorskip("vllm.utils.deep_gemm")
    if not deep_gemm.is_deep_gemm_supported():
        pytest.skip("vLLM DeepGEMM is unavailable")
    inputs = _make_fp8_einsum_inputs(3, 8, 4096, 1024, [128, 128], "cuda")
    x, xs, y, ys = inputs
    expected = torch.empty((3, 8, 1024), device="cuda", dtype=torch.bfloat16)
    deep_gemm.fp8_einsum(
        "bhr,hdr->bhd", (x, xs), (y, ys), expected, recipe=(1, 128, 128)
    )
    result = fp8_einsum("bhr,hdr->bhd", *inputs)
    torch.testing.assert_close(result, expected, rtol=2e-2, atol=1.0)


def test_fp8_einsum_exports_and_configs():
    from flaggems_vllm import runtime

    assert flaggems_vllm.fp8_einsum is fp8_einsum
    assert ("fp8_einsum", fp8_einsum) in flaggems_vllm._FULL_CONFIG
    for name in (
        "w8a8_block_fp8_bmm",
        "w8a8_block_fp8_bmm_general",
        "w8a8_block_fp8_bmm_splitk",
    ):
        assert runtime.get_tuned_config(name)
        assert runtime.get_expand_config(name) != -1
        assert runtime.ops_get_configs(name)


@pytest.mark.parametrize("config", [(3, 2, 256, 128), (3, 1, 4096, 128)])
def test_fp8_einsum_cudagraph_replay(config):
    b, h, r, d = config
    inputs = _make_fp8_einsum_inputs(b, h, r, d, [128, 128], "cuda")
    out = torch.empty((b, h, d), device="cuda", dtype=torch.bfloat16)
    fp8_einsum("bhr,hdr->bhd", *inputs, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fp8_einsum("bhr,hdr->bhd", *inputs, out=out)

    for seed in (1, 2):
        updated = _make_fp8_einsum_inputs(b, h, r, d, [128, 128], "cuda", seed=seed)
        for dst, src in zip(inputs, updated):
            dst.copy_(src)
        ref = torch_fp8_block_einsum_reference(*inputs, [128, 128])
        for _ in range(2):
            graph.replay()
            torch.testing.assert_close(out, ref.to(out.dtype), rtol=2e-1, atol=2.0)
