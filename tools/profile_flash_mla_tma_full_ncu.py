import time

import torch
import triton

import flaggems_vllm


def main():
    torch.manual_seed(0)
    device = "cuda"
    b = 128
    s_q = 1
    seqlen = 8192
    h_q = 128
    h_kv = 1
    d = 576
    dv = 512
    block_size = 64
    dtype = torch.bfloat16

    cache_seqlens = torch.full((b,), seqlen, dtype=torch.int32, device=device)
    max_seqlen_pad = triton.cdiv(seqlen, 256) * 256
    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32, device=device
    ).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(
        (b * max_seqlen_pad // block_size, block_size, h_kv, d),
        device=device,
        dtype=dtype,
    )
    q = torch.randn((b, s_q, h_q, d), device=device, dtype=dtype)

    for _ in range(3):
        out = flaggems_vllm.flash_mla(
            q,
            block_table,
            blocked_k,
            max_seqlen_pad,
            block_size,
            b,
            s_q,
            cache_seqlens,
            h_q,
            h_kv,
            d,
            dv,
            True,
        )
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    start = time.perf_counter()
    out = flaggems_vllm.flash_mla(
        q,
        block_table,
        blocked_k,
        max_seqlen_pad,
        block_size,
        b,
        s_q,
        cache_seqlens,
        h_q,
        h_kv,
        d,
        dv,
        True,
    )
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    elapsed_ms = (time.perf_counter() - start) * 1000
    print(out.shape, out.dtype, torch.isfinite(out).all().item(), f"{elapsed_ms:.3f} ms")


if __name__ == "__main__":
    main()
