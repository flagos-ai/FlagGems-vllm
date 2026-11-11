# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Project public-API tests: original 123 cases plus 21 optimized workloads.

The fixture always uses flaggems_vllm.fused_q. It cannot be redirected to a
standalone file or accidentally validate only baseline/fused_q.py.
"""

import hashlib
import importlib
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

if os.environ.get("FUSED_Q_REQUIRE_CUDA") == "1":
    import torch
else:
    torch = pytest.importorskip("torch")

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


pytestmark = pytest.mark.fused_q


@pytest.fixture(scope="module")
def operator(record_testsuite_property):
    def unavailable(message):
        if os.environ.get("FUSED_Q_REQUIRE_CUDA") == "1":
            pytest.fail(message)
        pytest.skip(message)

    if not torch.cuda.is_available() or torch.version.hip is not None:
        unavailable("fused_q requires NVIDIA CUDA")
    if os.environ.get("FUSED_Q_DEVICE"):
        torch.cuda.set_device(os.environ["FUSED_Q_DEVICE"])
    if torch.cuda.get_device_capability() < (8, 9):
        unavailable("fused_q requires SM89+ OCP FP8 support")
    import flaggems_vllm as gems

    if gems.vendor_name != "nvidia":
        unavailable("This implementation targets NVIDIA")
    module = importlib.import_module("flaggems_vllm.ops.fused_q")
    assert (
        gems.fused_q is module.fused_q
    ), "Public export is missing or routes elsewhere"
    path = Path(module.__file__).resolve()
    record_testsuite_property("fused_q_operator_path", str(path))
    record_testsuite_property(
        "fused_q_operator_sha256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    return SimpleNamespace(fused_q=gems.fused_q, module=module)


@pytest.fixture(autouse=True)
def quick_mode(request):
    if not request.config.getoption("quick", default=False):
        return
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    tokens = params.get("num_tokens", 1)
    if (
        tokens > 96
        or request.node.name == "test_fused_q_triton_supports_large_token_count"
    ):
        pytest.skip("--quick excludes large workloads; run without it for acceptance")


class ModelConfig(NamedTuple):
    """Per-model attention dims (num_heads is the TP8 local count)."""

    q_lora: int
    kv_lora: int
    rope_dim: int
    num_heads: int
    index_heads: int
    index_head_dim: int
    hidden: int


MODEL_CONFIGS = [
    ModelConfig(2048, 512, 64, 8, 32, 128, 6144),  # GLM-5.2
    ModelConfig(1536, 512, 64, 16, 64, 128, 7168),  # DeepSeek-V3.2
]
MODEL_IDS = ["glm5.2", "dsv3.2"]


def make_cos_sin(max_pos: int, rot_dim: int, device) -> torch.Tensor:
    """cos||sin cache: row[pos] = [cos(theta)(rot/2), sin(theta)(rot/2)]."""
    half = rot_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def rope(
    x: torch.Tensor, pos: torch.Tensor, cos_sin: torch.Tensor, interleave: bool
) -> torch.Tensor:
    """Apply RoPE to the first ``rot_dim`` elements of x's last dim.

    x: [..., head_dim] fp32. ``cos_sin`` is [max_pos, rot_dim]. ``interleave``
    selects adjacent-pair (GLM) vs split-half NeoX (DeepSeek-V3.2) layout.
    """
    rot = cos_sin.shape[-1]
    half = rot // 2
    cs = cos_sin[pos.long()]
    cos, sin = cs[..., :half], cs[..., half:]
    out = x.float().clone()
    r = out[..., :rot]
    if interleave:
        x1, x2 = r[..., 0::2].clone(), r[..., 1::2].clone()
        r[..., 0::2] = x1 * cos - x2 * sin
        r[..., 1::2] = x2 * cos + x1 * sin
    else:
        x1, x2 = r[..., :half].clone(), r[..., half:].clone()
        r[..., :half] = x1 * cos - x2 * sin
        r[..., half:] = x2 * cos + x1 * sin
    return out


def ue8m0_quant(vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (last dim) UE8M0 fp8 quant matching kernels._fp8_ue8m0_quantize."""
    amax = vals.float().abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax, min=1e-4) / FP8_MAX
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (vals.float() / scale).to(FP8)
    return q, scale.squeeze(-1)


def _fp8_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    def key(t):
        u = t.contiguous().view(torch.uint8).to(torch.int64)
        return torch.where(u >= 0x80, 0xFF - u, u + 0x80)

    return int((key(a) - key(b)).abs().max().item())


def assert_bf16(got: torch.Tensor, ref_fp32: torch.Tensor, msg: str):
    # Kernel keeps RMSNorm/RoPE in fp32 and rounds to bf16 once; the fp32
    # reduction/FMA order differs from torch, so a few elements land on the
    # opposite side of a round-to-nearest tie. Use the same tolerance the
    # sibling deepseek_v4 fused-kernel test uses for this bf16 norm+rope class.
    torch.testing.assert_close(
        got.float(), ref_fp32.float(), rtol=1e-2, atol=1e-2, msg=lambda m: f"{msg}: {m}"
    )


def assert_fp8(got: torch.Tensor, ref: torch.Tensor, msg: str):
    assert _fp8_ulp(got, ref) <= 1, f"{msg}: >1 fp8 ULP"


# ── fused_q ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("num_tokens", [1, 4, 17, 512, 4096])
@pytest.mark.parametrize("index_interleave", [True, False])
def test_fused_q(num_tokens: int, index_interleave: bool, cfg: ModelConfig, operator):
    K = operator
    torch.manual_seed(2)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64) % max_pos

    q_pe = torch.randn(
        num_tokens, cfg.num_heads, cfg.rope_dim, device=dev, dtype=torch.bfloat16
    )
    ql_nope = torch.randn(
        num_tokens, cfg.num_heads, cfg.kv_lora, device=dev, dtype=torch.bfloat16
    )
    index_q = torch.randn(
        num_tokens,
        cfg.index_heads,
        cfg.index_head_dim,
        device=dev,
        dtype=torch.bfloat16,
    )
    index_w = torch.randn(num_tokens, cfg.index_heads, device=dev, dtype=torch.float32)
    q_scale = torch.tensor([0.37], device=dev, dtype=torch.float32)
    softmax_scale = cfg.index_head_dim**-0.5
    head_scale = cfg.index_heads**-0.5
    q_cos_sin = make_cos_sin(max_pos, cfg.rope_dim, dev)  # q_pe: interleaved
    idx_cos_sin = make_cos_sin(max_pos, cfg.rope_dim, dev)

    iq_fp8, iw_out, mqa = K.fused_q(
        pos,
        q_pe,
        q_cos_sin,
        index_q,
        idx_cos_sin,
        ql_nope,
        q_scale,
        index_w,
        softmax_scale,
        head_scale,
        has_indexer=True,
        index_rope_interleave=index_interleave,
    )

    s = q_scale.item()
    # MQA query: [ql_nope | q_pe RoPE'd (interleaved)], per-tensor fp8.
    mqa_nope_ref = (ql_nope.float() / s).to(FP8)
    qpe_ref = rope(
        q_pe.float(),
        pos.unsqueeze(-1).expand(num_tokens, cfg.num_heads),
        q_cos_sin,
        interleave=True,
    )
    mqa_pe_ref = (qpe_ref / s).to(FP8)
    assert_fp8(mqa[:, :, : cfg.kv_lora], mqa_nope_ref, "mqa ql_nope")
    assert_fp8(mqa[:, :, cfg.kv_lora :], mqa_pe_ref, "mqa q_pe")

    # Indexer-Q: RoPE + UE8M0 fp8 quant; index weights fold in q-scale.
    iq_ref = rope(
        index_q.float(),
        pos.unsqueeze(-1).expand(num_tokens, cfg.index_heads),
        idx_cos_sin,
        interleave=index_interleave,
    )
    q_ref, scale_ref = ue8m0_quant(iq_ref)
    assert_fp8(iq_fp8, q_ref, "indexer-Q fp8")
    iw_ref = index_w * scale_ref * softmax_scale * head_scale
    torch.testing.assert_close(iw_out, iw_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("num_tokens", [1, 17, 512])
def test_fused_q_no_indexer(num_tokens: int, cfg: ModelConfig, operator):
    K = operator
    torch.manual_seed(3)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    q_pe = torch.randn(
        num_tokens, cfg.num_heads, cfg.rope_dim, device=dev, dtype=torch.bfloat16
    )
    ql_nope = torch.randn(
        num_tokens, cfg.num_heads, cfg.kv_lora, device=dev, dtype=torch.bfloat16
    )
    q_scale = torch.tensor([0.5], device=dev, dtype=torch.float32)
    q_cos_sin = make_cos_sin(max_pos, cfg.rope_dim, dev)
    _, _, mqa = K.fused_q(
        pos,
        q_pe,
        q_cos_sin,
        None,
        None,
        ql_nope,
        q_scale,
        None,
        0.0,
        0.0,
        has_indexer=False,
        index_rope_interleave=False,
    )
    s = q_scale.item()
    assert_fp8(mqa[:, :, : cfg.kv_lora], (ql_nope.float() / s).to(FP8), "mqa ql_nope")
    qpe_ref = rope(
        q_pe.float(),
        pos.unsqueeze(-1).expand(num_tokens, cfg.num_heads),
        q_cos_sin,
        interleave=True,
    )
    assert_fp8(mqa[:, :, cfg.kv_lora :], (qpe_ref / s).to(FP8), "mqa q_pe")


@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=MODEL_IDS)
@pytest.mark.parametrize("num_tokens", [1, 17, 512])
@pytest.mark.parametrize("has_indexer", [True, False])
@pytest.mark.parametrize("use_mha", [True, False])
@pytest.mark.parametrize(
    "index_interleave,cache_dtype", [(False, torch.float32), (True, torch.bfloat16)]
)
@pytest.mark.parametrize("capture", [False, True])
def test_fused_q_bf16_query(
    num_tokens: int,
    has_indexer: bool,
    use_mha: bool,
    index_interleave: bool,
    cache_dtype: torch.dtype,
    capture: bool,
    cfg: ModelConfig,
    operator,
):
    """bf16-query path (FlashMLA sparse, SM90/SM100): only the RoPE'd q_pe is
    produced; MHA does not need an absorbed ql_nope projection."""
    K = operator
    torch.manual_seed(6)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64) % max_pos

    # Attention takes q_pe as a strided slice of the full Q projection.
    q_pe = torch.randn(
        num_tokens, cfg.num_heads, 4 * cfg.rope_dim, device=dev, dtype=torch.bfloat16
    )[..., -cfg.rope_dim :]
    ql_nope = torch.randn(
        num_tokens, cfg.num_heads, cfg.kv_lora, device=dev, dtype=torch.bfloat16
    )
    q_scale = torch.tensor([0.37], device=dev, dtype=torch.float32)
    q_cos_sin = make_cos_sin(max_pos, cfg.rope_dim, dev).to(cache_dtype)

    index_q = index_w = idx_cos_sin = None
    if has_indexer:
        index_q = torch.randn(
            num_tokens,
            cfg.index_heads,
            cfg.index_head_dim,
            device=dev,
            dtype=torch.bfloat16,
        )
        index_w = torch.randn(
            num_tokens, cfg.index_heads, device=dev, dtype=torch.float32
        )
        idx_cos_sin = make_cos_sin(max_pos, cfg.rope_dim, dev).to(cache_dtype)

    run = partial(
        K.fused_q,
        pos,
        q_pe,
        q_cos_sin,
        index_q,
        idx_cos_sin,
        q_pe if use_mha else ql_nope,
        q_scale,
        index_w,
        cfg.index_head_dim**-0.5,
        cfg.index_heads**-0.5,
        has_indexer=has_indexer,
        index_rope_interleave=index_interleave,
        quantize_mqa=False,
    )

    iq_fp8, iw_out, q_pe_out = run()
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            iq_fp8, iw_out, q_pe_out = run()
        q_pe.mul_(0.5)
        if index_q is not None:
            index_q.mul_(1.5)
        graph.replay()

    # MQA query: only the RoPE'd q_pe, bf16, unquantized.
    assert q_pe_out.dtype == torch.bfloat16
    assert q_pe_out.shape == (num_tokens, cfg.num_heads, cfg.rope_dim)
    qpe_ref = rope(
        q_pe.float(),
        pos.unsqueeze(-1).expand(num_tokens, cfg.num_heads),
        q_cos_sin,
        interleave=True,
    )
    assert_bf16(q_pe_out, qpe_ref, "bf16 q_pe RoPE")

    # Indexer-Q is unchanged on this path (still UE8M0 fp8 + folded weights).
    if has_indexer:
        assert index_q is not None
        iq_ref = rope(
            index_q.float(),
            pos.unsqueeze(-1).expand(num_tokens, cfg.index_heads),
            idx_cos_sin,
            interleave=index_interleave,
        )
        q_ref, scale_ref = ue8m0_quant(iq_ref)
        assert_fp8(iq_fp8, q_ref, "indexer-Q fp8 (bf16-query path)")
        iw_ref = (
            index_w * scale_ref * (cfg.index_head_dim**-0.5) * (cfg.index_heads**-0.5)
        )
        torch.testing.assert_close(iw_out, iw_ref, rtol=1e-3, atol=1e-3)


def test_fused_q_triton_supports_large_token_count(operator):
    """Keep the token count off CUDA grid-y in the Triton fallback.

    The minimal dimensions also bypass CuTeDSL on SM100.
    """
    K = operator
    num_tokens = 65536
    dev = "cuda"
    dtype = torch.bfloat16
    positions = torch.zeros(num_tokens, device=dev, dtype=torch.int64)
    q_pe = torch.ones((num_tokens, 1, 2), device=dev, dtype=dtype)
    ql_nope = torch.ones((num_tokens, 1, 1), device=dev, dtype=dtype)
    cos_sin = torch.tensor([[1.0, 0.0]], device=dev, dtype=torch.float32)
    q_scale = torch.ones(1, device=dev, dtype=torch.float32)

    _, _, mqa_q = K.fused_q(
        positions,
        q_pe,
        cos_sin,
        None,
        None,
        ql_nope,
        q_scale,
        None,
        0.0,
        0.0,
        has_indexer=False,
    )

    rows = torch.tensor([0, num_tokens - 1], device=dev)
    ref = torch.cat([ql_nope, q_pe], dim=-1).to(FP8)
    assert_fp8(mqa_q[rows], ref[rows], "large-token fused Q")


def assert_benchmark_case(values, actual):
    """Independent full reference for the optimized default contract."""
    n, h, r = values["q_pe"].shape
    ih, idim = values["index_q"].shape[1:]
    d = values["ql_nope"].shape[-1]
    assert isinstance(actual, tuple) and len(actual) == 3
    for got, shape, dtype in zip(
        actual, ((n, ih, idim), (n, ih), (n, h, d + r)), (FP8, torch.float32, FP8)
    ):
        assert (
            got.shape == shape
            and got.dtype == dtype
            and got.device == values["q_pe"].device
        )
    pos = values["positions"]
    pe = rope(
        values["q_pe"], pos[:, None].expand(n, h), values["q_pe_cos_sin_cache"], True
    )
    index = rope(
        values["index_q"],
        pos[:, None].expand(n, ih),
        values["index_q_cos_sin_cache"],
        False,
    )
    iq, scale = ue8m0_quant(index)
    assert_fp8(actual[0], iq, "index FP8")
    expected = torch.cat(
        (values["ql_nope"].float() / values["q_scale"], pe / values["q_scale"]), dim=-1
    ).to(FP8)
    assert_fp8(actual[2], expected, "MQA FP8")
    weights = (
        values["index_weights"]
        * scale
        * values["index_weights_softmax_scale"]
        * values["index_weights_head_scale"]
    )
    torch.testing.assert_close(actual[1], weights, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("num_tokens", [1, 6, 24, 96, 384, 1536, 4096])
@pytest.mark.parametrize("num_heads", [16, 32, 64])
def test_fused_q_optimized_contract(num_tokens, num_heads, operator, monkeypatch):
    from baseline import fused_q as workload

    binding = workload.binding_from_shape([num_tokens, num_heads, 512, 64, 32, 128])
    _, values = workload.build_inputs(binding, torch.bfloat16, "cuda")
    before = {k: v.clone() for k, v in values.items() if isinstance(v, torch.Tensor)}
    module = operator.module
    original, launches = module._fused_q_kernel, []

    class Capture:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, kwargs))
                return original[grid](*args, **kwargs)

            return launch

    with monkeypatch.context() as patch:
        patch.setattr(module, "_fused_q_kernel", Capture())
        output = operator.fused_q(**values)
    assert_benchmark_case(values, output)
    assert len(launches) == 1
    full_h20, pdl = module._device_policy(values["q_pe"].device.index)
    compact = full_h20 and num_heads in (16, 32)
    grid, flags = launches[0]
    assert grid == ((num_tokens, num_heads + 32) if compact else (num_tokens, 3, 32))
    assert flags["COMPACT_GRID"] == compact
    assert flags["USE_PDL"] == (pdl and not full_h20)
    assert flags.get("launch_pdl", False) == flags["USE_PDL"]
    assert flags["num_warps"] == 1
    control = workload.native()(**values)
    again = operator.fused_q(**values)
    for a, b, c in zip(output, control, again):
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )
        assert torch.equal(
            a.contiguous().view(torch.uint8), c.contiguous().view(torch.uint8)
        )
    for key, value in before.items():
        assert torch.equal(values[key], value), "Input modified: " + key
