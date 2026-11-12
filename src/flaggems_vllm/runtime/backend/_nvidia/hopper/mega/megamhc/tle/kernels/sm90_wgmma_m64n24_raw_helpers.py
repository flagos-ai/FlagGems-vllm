"""Raw M64N24K8 WGMMA helpers used by the production T=400 kernel."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _raw_m64n24k64_accumulate(
    q_shared_address,
    fn_shared_address,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    c8,
    c9,
    c10,
    c11,
):
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .u32 tid, mtid, lane, warp, row0, elem, row_base;
          .reg .u32 half, half4, load_idx, bank, byte_offset, address;
          .reg .u32 encoded_fn, half_desc32;
          .reg .u64 desc_base, half_desc64, desc0, desc1, desc2, desc3;
          .reg .b16 bh0, bh1, bh2, bh3;
          .reg .b32 a<16>;
          .reg .pred more_half;

          mov.f32 $0,$14; mov.f32 $1,$15; mov.f32 $2,$16;
          mov.f32 $3,$17; mov.f32 $4,$18; mov.f32 $5,$19;
          mov.f32 $6,$20; mov.f32 $7,$21; mov.f32 $8,$22;
          mov.f32 $9,$23; mov.f32 $10,$24; mov.f32 $11,$25;
          mov.u32 tid, %tid.x;
          sub.u32 mtid, tid, 512;
          and.b32 lane, mtid, 31;
          shr.u32 warp, mtid, 5;
          shr.u32 row0, lane, 2;
          mad.lo.u32 row0, warp, 16, row0;
          and.b32 elem, lane, 3;
          mad.lo.u32 row_base, row0, 128, $12;
          shl.b32 byte_offset, elem, 1;
          add.u32 row_base, row_base, byte_offset;
          bfe.u32 encoded_fn, $13, 4, 14;
          cvt.u64.u32 desc_base, encoded_fn;
          add.u64 desc_base, desc_base, 0x4000004000000000;
          mov.u32 half, 0;
        DG_TLE_MHC_RAW_N24_HALF:
          shl.b32 half4, half, 2;
          add.u32 load_idx, half4, 0;
          xor.b32 bank, row0, load_idx;
          and.b32 bank, bank, 7;
          shl.b32 byte_offset, bank, 4;
          add.u32 address, row_base, byte_offset;
          ld.shared.b16 bh0,[address]; ld.shared.b16 bh2,[address+8];
          ld.shared.b16 bh1,[address+1024]; ld.shared.b16 bh3,[address+1032];
          cvt.f32.bf16 a0,bh0; cvt.f32.bf16 a1,bh1;
          cvt.f32.bf16 a2,bh2; cvt.f32.bf16 a3,bh3;
          add.u32 load_idx, half4, 1;
          xor.b32 bank, row0, load_idx;
          and.b32 bank, bank, 7;
          shl.b32 byte_offset, bank, 4;
          add.u32 address, row_base, byte_offset;
          ld.shared.b16 bh0,[address]; ld.shared.b16 bh2,[address+8];
          ld.shared.b16 bh1,[address+1024]; ld.shared.b16 bh3,[address+1032];
          cvt.f32.bf16 a4,bh0; cvt.f32.bf16 a5,bh1;
          cvt.f32.bf16 a6,bh2; cvt.f32.bf16 a7,bh3;
          add.u32 load_idx, half4, 2;
          xor.b32 bank, row0, load_idx;
          and.b32 bank, bank, 7;
          shl.b32 byte_offset, bank, 4;
          add.u32 address, row_base, byte_offset;
          ld.shared.b16 bh0,[address]; ld.shared.b16 bh2,[address+8];
          ld.shared.b16 bh1,[address+1024]; ld.shared.b16 bh3,[address+1032];
          cvt.f32.bf16 a8,bh0; cvt.f32.bf16 a9,bh1;
          cvt.f32.bf16 a10,bh2; cvt.f32.bf16 a11,bh3;
          add.u32 load_idx, half4, 3;
          xor.b32 bank, row0, load_idx;
          and.b32 bank, bank, 7;
          shl.b32 byte_offset, bank, 4;
          add.u32 address, row_base, byte_offset;
          ld.shared.b16 bh0,[address]; ld.shared.b16 bh2,[address+8];
          ld.shared.b16 bh1,[address+1024]; ld.shared.b16 bh3,[address+1032];
          cvt.f32.bf16 a12,bh0; cvt.f32.bf16 a13,bh1;
          cvt.f32.bf16 a14,bh2; cvt.f32.bf16 a15,bh3;
          shl.b32 half_desc32, half, 8;
          cvt.u64.u32 half_desc64, half_desc32;
          add.u64 desc0, desc_base, half_desc64;
          add.u64 desc1, desc0, 2; add.u64 desc2, desc0, 4;
          add.u64 desc3, desc0, 6;
          wgmma.fence.sync.aligned;
          wgmma.mma_async.sync.aligned.m64n24k8.f32.tf32.tf32
            {$0,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11},
            {a0,a1,a2,a3}, desc0, 1, 1, 1;
          wgmma.mma_async.sync.aligned.m64n24k8.f32.tf32.tf32
            {$0,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11},
            {a4,a5,a6,a7}, desc1, 1, 1, 1;
          wgmma.mma_async.sync.aligned.m64n24k8.f32.tf32.tf32
            {$0,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11},
            {a8,a9,a10,a11}, desc2, 1, 1, 1;
          wgmma.mma_async.sync.aligned.m64n24k8.f32.tf32.tf32
            {$0,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11},
            {a12,a13,a14,a15}, desc3, 1, 1, 1;
          add.u32 half, half, 1;
          setp.lt.u32 more_half, half, 2;
          @more_half bra.uni DG_TLE_MHC_RAW_N24_HALF;
          wgmma.commit_group.sync.aligned;
          wgmma.wait_group.sync.aligned 0;
        }
        """,
        constraints=(
            "=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,"
            "r,r,f,f,f,f,f,f,f,f,f,f,f,f"
        ),
        args=[
            q_shared_address,
            fn_shared_address,
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            c9,
            c10,
            c11,
        ],
        dtype=(tl.float32,) * 12,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_raw_n24_partials(partials_address, tile, token_limit, values):
    """Store c<12> into a global [T,32] split plane with tail masks."""
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = values
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred valid0,valid1;
          .reg .u32 tid,mtid,lane,warp,row0,row1,odd,pair,col,off,base_row;
          .reg .u64 a0,a1,cb;
          mov.u32 tid,%tid.x; sub.u32 mtid,tid,512;
          and.b32 lane,mtid,31; shr.u32 warp,mtid,5;
          shr.u32 row0,lane,2; mad.lo.u32 row0,warp,16,row0;
          add.u32 row1,row0,8; shl.b32 base_row,$2,6;
          add.u32 row0,row0,base_row; add.u32 row1,row1,base_row;
          setp.lt.u32 valid0,row0,$3; setp.lt.u32 valid1,row1,$3;
          shr.u32 odd,lane,1; and.b32 odd,odd,1; and.b32 pair,lane,1;
          shl.b32 col,odd,2; shl.b32 off,pair,1; add.u32 col,col,off;
          mad.wide.u32 a0,row0,128,$1; mad.wide.u32 a1,row1,128,$1;
          cvt.u64.u32 cb,col; shl.b64 cb,cb,2;
          add.u64 a0,a0,cb; add.u64 a1,a1,cb;
          @valid0 st.global.v2.f32 [a0],{$4,$5};
          @valid1 st.global.v2.f32 [a1],{$6,$7};
          @valid0 st.global.v2.f32 [a0+32],{$8,$9};
          @valid1 st.global.v2.f32 [a1+32],{$10,$11};
          @valid0 st.global.v2.f32 [a0+64],{$12,$13};
          @valid1 st.global.v2.f32 [a1+64],{$14,$15};
          mov.u32 $0,0;
        }
        """,
        constraints="=r,l,r,r,f,f,f,f,f,f,f,f,f,f,f,f",
        args=[
            partials_address,
            tile,
            token_limit,
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            c9,
            c10,
            c11,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


__all__ = [
    "_raw_m64n24k64_accumulate",
    "_store_raw_n24_partials",
]
