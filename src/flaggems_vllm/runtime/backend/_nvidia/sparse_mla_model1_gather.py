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

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flaggems_vllm.ops.flash_mla_with_kvcache_model1 import tle_subslice


@triton.jit
def split_block_range(split, NUM_BLOCKS, SPLITS: tl.constexpr):
    if SPLITS == 1:
        return 0, NUM_BLOCKS
    base = NUM_BLOCKS // SPLITS
    rem = NUM_BLOCKS - base * SPLITS
    first = split * base + tl.minimum(split, rem)
    count = base + tl.where(split < rem, 1, 0)
    return first, count


@triton.jit
def scaled_bytes(quantized, ratios, R: tl.constexpr):
    factors = tl.broadcast_to(ratios[:, :, None], (R, 8, 64)).reshape(R, 512)
    return (quantized.to(tl.float32) * factors).to(tl.float8e4nv)


@triton.jit
def normalize_query(Q, QS, Norm, Magnitude, H: tl.constexpr):
    b, tile = tl.program_id(0), tl.program_id(1)
    rows = b * H + tile * 4 + tl.arange(0, 4)
    g, d = tl.arange(0, 8), tl.arange(0, 512)
    scales = tl.load(QS + rows[:, None] * 7 + g[None, :], g[None, :] < 7, 0.0)
    mx = tl.max(scales, 1)
    mn = tl.min(tl.where(scales > 0.0, scales, float("inf")), 1)
    q = tl.load(Q + rows[:, None] * 448 + d[None, :], d[None, :] < 448, 0.0)
    qnorm = scaled_bytes(q, scales / tl.where(mx > 0.0, mx, 1.0)[:, None], 4)
    tl.store(Norm + rows[:, None] * 448 + d[None, :], qnorm, d[None, :] < 448)
    tl.store(Magnitude + rows, tl.where(mx > 4.0 * mn, -mx, mx))


@triton.jit
def normalize_cache(
    K,
    E,
    Normalized,
    Scale,
    PackedRope,
    N,
    STRIDE: tl.constexpr,
):
    # Each block processes consecutive physical slots, including a partial final page.
    ids = tl.program_id(0) * 32 + tl.arange(0, 32)
    page = (ids >> 6).to(tl.int64) * STRIDE
    slot = ids & 63
    g = tl.arange(0, 8)
    word = tl.load(
        E.to(tl.pointer_type(tl.uint64)) + page // 8 + 4608 + slot, ids < N, 0
    )
    exps = ((word[:, None] >> (g * 8)[None, :]) & 255).to(tl.int32)
    hi = tl.max(tl.where(g[None, :] < 7, exps, 0), 1)
    lo = tl.min(tl.where(g[None, :] < 7, exps, 255), 1)
    d = tl.arange(0, 512)
    content = tl.load(
        K + (page + slot * 576)[:, None] + d[None, :],
        (ids[:, None] < N) & (d[None, :] < 448),
        0.0,
    )
    ratio = (tl.maximum(127 + tl.minimum(exps - hi[:, None], 0), 0) << 23).to(
        tl.float32, bitcast=True
    )
    norm = scaled_bytes(content, ratio, 32)
    tl.store(
        Normalized + ids.to(tl.int64)[:, None] * 448 + d[None, :],
        norm,
        (ids[:, None] < N) & (d[None, :] < 448),
    )
    r = tl.arange(0, 64)
    src = (page + slot * 576 + 448) // 2
    rope = tl.load(
        E.to(tl.pointer_type(tl.bfloat16)) + src[:, None] + r[None, :],
        ids[:, None] < N,
        0.0,
    )
    tl.store(PackedRope + ids[:, None] * 64 + r[None, :], rope, ids[:, None] < N)
    scale = tl.where(hi == 0, 0x00400000, hi << 23).to(tl.float32, bitcast=True)
    tl.store(Scale + ids, tl.where(hi - lo > 2, -scale, scale), ids < N)


@triton.constexpr_function
def gather_asm(block, page_stride, panel_rows, warps):
    # A subgroup of two or four lanes owns one selected row. Every lane keeps
    # that row's address and exponent word through both copying and rescaling.
    rows = block // warps
    lanes = 32 // rows
    asm = f"""
    {{
    .reg .pred ok, first, diff, any_diff, more, lower, wide, skip;
    .reg .b32 tid,lane,warp,row,chunk,pos,id,page,slot,e0,e1,hi,lo,ex,sz,flag;
    .reg .b32 off,sw,kbase,rbase,kdst,rdst,sdst,tmp,u,group,word,shift,delta,fac;
    .reg .b32 a,b,c,d;
    .reg .b16 xlo,xhi;
    .reg .b64 address,ptr,src,pg;
    mov.u32 tid,%tid.x;
    and.b32 tid,tid,{warps*32-1};
    shr.u32 warp,tid,5;
    and.b32 lane,tid,31;
    shr.u32 row,lane,{lanes.bit_length()-1};
    mad.lo.u32 row,warp,{rows},row;
    and.b32 chunk,lane,{lanes-1};
    add.u32 pos,$8,row;
    setp.lt.u32 ok,pos,$9;
    mov.u32 id,-1;
    mul.wide.u32 address,pos,4;
    add.u64 ptr,$3,address;
    @ok ld.global.L1::evict_first.s32 id,[ptr];
    setp.lt.u32 ok,id,$7;
    @!ok mov.u32 id,0;
    shr.u32 page,id,6;
    and.b32 slot,id,63;
    mul.wide.u32 address,page,{page_stride};
    add.u64 pg,$2,address;
    mul.wide.u32 address,slot,8;
    add.u64 ptr,pg,address;
    ld.global.v2.b32 {{e0,e1}},[ptr+36864];
    mul.wide.u32 address,slot,576;
    add.u64 src,pg,address;
    shl.b32 off,chunk,4;
    cvt.u64.u32 address,off;
    add.u64 src,src,address;
    selp.u32 sz,16,0,ok;
    and.b32 sw,row,7;
    shl.b32 sw,sw,4;
    xor.b32 off,off,sw;
    mad.lo.u32 tmp,row,128,$4;
    add.u32 kbase,tmp,off;
    mad.lo.u32 tmp,row,128,$5;
    add.u32 rbase,tmp,off;
    """
    for g in range(9):
        for seg in range(4 // lanes):
            column_offset = (g % 2) * 64 + seg * lanes * 16
            adr = g * 64 + seg * lanes * 16
            if g < 7:
                asm += f"xor.b32 kdst,kbase,{column_offset};\n"
                asm += f"cp.async.cg.shared.global [kdst+{(g//2)*panel_rows*128}],[src+{adr}],16,sz;\n"
            else:
                column_offset = (g - 7) * 64 + seg * lanes * 16
                asm += f"xor.b32 rdst,rbase,{column_offset};\ncp.async.cg.shared.global [rdst],[src+{adr}],16,sz;\n"
    asm += """
    cp.async.commit_group;
    and.b32 hi,e0,255;
    mov.b32 lo,hi;
    """
    for g in range(1, 7):
        asm += f"bfe.u32 ex,{'e0' if g<4 else 'e1'},{(g%4)*8},8; min.u32 lo,lo,ex; max.u32 hi,hi,ex;\n"
    asm += """
    shl.b32 tmp,hi,23;
    setp.eq.u32 first,hi,0;
    @first mov.b32 tmp,0x00400000;
    @!ok mov.b32 tmp,0xbf800000;
    setp.eq.u32 first,chunk,0;
    mad.lo.u32 sdst,row,4,$6;
    @first st.shared.b32 [sdst],tmp;
    sub.u32 delta,hi,lo;
    setp.gt.and.u32 wide,delta,2,ok;
    selp.u32 flag,1,0,wide;
    setp.ne.and.u32 diff,delta,0,ok;
    vote.sync.any.pred any_diff,diff,0xffffffff;
    cp.async.wait_group 0;
    @!any_diff bra DONE;
    mov.u32 group,0;
    RESCALE:
    setp.lt.u32 lower,group,4;
    selp.u32 word,e0,e1,lower;
    and.b32 shift,group,3;
    shl.b32 shift,shift,3;
    bfe.u32 ex,word,shift,8;
    setp.eq.u32 skip,hi,ex;
    @skip bra NEXT;
    sub.u32 delta,hi,ex;
    min.u32 delta,delta,15;
    sub.u32 fac,15,delta;
    shl.b32 fac,fac,10;
    prmt.b32 fac,fac,fac,0x1010;
    and.b32 tmp,group,1;
    shl.b32 tmp,tmp,6;
    xor.b32 kdst,kbase,tmp;
    shr.u32 tmp,group,1;
    """
    asm += f"mad.lo.u32 kdst,tmp,{panel_rows*128},kdst;\n"
    for seg in range(4 // lanes):
        asm += (
            f"xor.b32 rdst,kdst,{seg*lanes*16};\nld.shared.v4.b32 {{a,b,c,d}},[rdst];\n"
        )
        for reg in ("a", "b", "c", "d"):
            asm += f"""
            mov.b32 {{xlo,xhi}},{reg};
            cvt.rn.f16x2.e4m3x2 tmp,xlo;
            cvt.rn.f16x2.e4m3x2 u,xhi;
            mul.f16x2 tmp,tmp,fac;
            mul.f16x2 u,u,fac;
            cvt.rn.satfinite.e4m3x2.f16x2 xlo,tmp;
            cvt.rn.satfinite.e4m3x2.f16x2 xhi,u;
            mov.b32 {reg},{{xlo,xhi}};
            """
        asm += "st.shared.v4.b32 [rdst],{a,b,c,d};\n"
    return (
        asm
        + """
    NEXT:
    add.u32 group,group,1;
    setp.lt.u32 more,group,7;
    @more bra RESCALE;
    DONE:
    mov.b32 $0,flag;
    }
    """
    )


@triton.constexpr_function
def warp_gather_asm(block, page_stride, panel_rows, warps):
    # Each warp resolves its selected rows before issuing 16-byte copies.
    # The 8 (or 16) metadata lanes broadcast one physical slot at a time.
    asm = """
    {
    .reg .pred ok,active,first,rope,loop,real,fromlo,wide,rescale,row_diff,any_diff,fast;
    .reg .b32 tid,lane,warp,row,pos,id,page,slot,e0,e1,ex,emin,emax,flag,sz;
    .reg .b32 src0,src1,peer0,peer1,peer_sz,peer_hi,peer_e0,peer_e1,which;
    .reg .b32 panel,off,sw,kdst,rdst,sdst,fac,a,b,c,d,t,u,delta,iter;
    .reg .b16 xlo,xhi;
    .reg .b64 address,pgaddr,src,peer,ptr;
    mov.u32 tid,%tid.x;
    and.b32 tid,tid,TIDMASK;
    shr.u32 warp,tid,5;
    and.b32 lane,tid,31;
    and.b32 row,lane,ROWMASK;
    mad.lo.u32 row,row,WARPS,warp;
    add.u32 pos,$8,row;
    setp.lt.u32 ok,pos,$9;
    mul.wide.u32 address,pos,4;
    add.u64 ptr,$3,address;
    mov.b32 id,-1;
    @ok ld.global.L1::evict_first.s32 id,[ptr];
    setp.lt.u32 ok,id,$7;
    @!ok mov.b32 id,0;
    shr.u32 page,id,6;
    and.b32 slot,id,63;
    mul.wide.u32 address,page,PAGESTRIDE;
    add.u64 pgaddr,$2,address;
    mul.wide.u32 address,slot,8;
    add.u64 ptr,pgaddr,address;
    ld.global.v2.b32 {e0,e1},[ptr+36864];
    mul.wide.u32 address,slot,576;
    add.u64 src,pgaddr,address;
    mov.b64 {src0,src1},src;
    selp.u32 sz,16,0,ok;
    shr.u32 panel,lane,3;
    mul.lo.u32 panel,panel,PANEL;
    setp.lt.u32 real,lane,28;
    setp.lt.u32 rope,lane,8;
    mov.u32 iter,0;
    COPY:
    shfl.sync.idx.b32 peer0,src0,iter,31,0xffffffff;
    shfl.sync.idx.b32 peer1,src1,iter,31,0xffffffff;
    shfl.sync.idx.b32 peer_sz,sz,iter,31,0xffffffff;
    mov.b64 peer,{peer0,peer1};
    mad.lo.u32 row,iter,WARPS,warp;
    and.b32 sw,row,7;
    shl.b32 sw,sw,4;
    and.b32 off,lane,7;
    shl.b32 off,off,4;
    cvt.u64.u32 address,off;
    add.u64 address,peer,address;
    add.u64 address,address,448;
    mad.lo.u32 off,row,128,off;
    xor.b32 off,off,sw;
    add.u32 rdst,$5,off;
    @rope cp.async.cg.shared.global [rdst],[address],16,peer_sz;
    add.u32 kdst,$4,off;
    add.u32 kdst,kdst,panel;
    shl.b32 t,lane,4;
    cvt.u64.u32 address,t;
    add.u64 address,peer,address;
    @real cp.async.cg.shared.global [kdst],[address],16,peer_sz;
    add.u32 iter,iter,1;
    setp.le.u32 loop,iter,ROWMASK;
    @loop bra COPY;
    cp.async.commit_group;
    and.b32 emin,e0,255;
    mov.b32 emax,emin;
    """
    for group in range(1, 7):
        asm += f"bfe.u32 ex,{'e0' if group < 4 else 'e1'},{(group % 4) * 8},8; "
        asm += "min.u32 emin,emin,ex; max.u32 emax,emax,ex; "
    asm += """
    sub.u32 delta,emax,emin;
    setp.gt.u32 wide,delta,2;
    and.pred wide,wide,ok;
    selp.u32 flag,1,0,wide;
    // One metadata lane owns the SMEM scale for this row.
    shl.b32 t,emax,23;
    setp.eq.u32 first,emax,0;
    @first mov.b32 t,0x00400000;
    @!ok mov.b32 t,0xbf800000;
    and.b32 row,lane,ROWMASK;
    mad.lo.u32 row,row,WARPS,warp;
    mad.lo.u32 sdst,row,4,$6;
    setp.le.u32 first,lane,ROWMASK;
    @first st.shared.b32 [sdst],t;
    cp.async.wait_group 0;
    mov.u32 iter,0;
    setp.ne.u32 row_diff,emax,emin;
    and.pred row_diff,row_diff,first;
    and.pred row_diff,row_diff,ok;
    vote.sync.any.pred any_diff,row_diff,0xffffffff;
    @!any_diff bra DONE_RESCALE;
    RESCALE:
    shfl.sync.idx.b32 peer_e0,e0,iter,31,0xffffffff;
    shfl.sync.idx.b32 peer_e1,e1,iter,31,0xffffffff;
    shfl.sync.idx.b32 peer_hi,emax,iter,31,0xffffffff;
    setp.lt.u32 fromlo,lane,16;
    selp.u32 ex,peer_e0,peer_e1,fromlo;
    shr.u32 which,lane,2;
    and.b32 which,which,3;
    shl.b32 which,which,3;
    bfe.u32 ex,ex,which,8;
    setp.ne.u32 rescale,peer_hi,ex;
    and.pred rescale,rescale,real;
    @!rescale bra NEXT;
    mad.lo.u32 row,iter,WARPS,warp;
    and.b32 sw,row,7;
    shl.b32 sw,sw,4;
    and.b32 off,lane,7;
    shl.b32 off,off,4;
    mad.lo.u32 off,row,128,off;
    xor.b32 off,off,sw;
    add.u32 kdst,$4,off;
    add.u32 kdst,kdst,panel;
    ld.shared.v4.b32 {a,b,c,d},[kdst];
    sub.s32 delta,peer_hi,ex;
    // Subtracting from each exponent is exact while every merged_value stays normal.
    // The half2 conversion below handles zeros, subnormals and wide spreads.
    min.u32 fac,delta,14;
    shl.b32 fac,fac,3;
    mul.lo.u32 fac,fac,0x01010101;
    add.u32 which,fac,0x08080808;
    or.b32 t,a,0x80808080;
    sub.u32 t,t,which;
    or.b32 u,b,0x80808080;
    sub.u32 u,u,which;
    and.b32 t,t,u;
    or.b32 u,c,0x80808080;
    sub.u32 u,u,which;
    and.b32 t,t,u;
    or.b32 u,d,0x80808080;
    sub.u32 u,u,which;
    and.b32 t,t,u;
    and.b32 t,t,0x80808080;
    and.b32 u,a,0x7f7f7f7f;
    add.u32 u,u,0x01010101;
    and.b32 which,b,0x7f7f7f7f;
    add.u32 which,which,0x01010101;
    or.b32 u,u,which;
    and.b32 which,c,0x7f7f7f7f;
    add.u32 which,which,0x01010101;
    or.b32 u,u,which;
    and.b32 which,d,0x7f7f7f7f;
    add.u32 which,which,0x01010101;
    or.b32 u,u,which;
    not.b32 u,u;
    and.b32 t,t,u;
    and.b32 t,t,0x80808080;
    setp.eq.u32 fast,t,0x80808080;
    setp.lt.u32.and fast,delta,15,fast;
    @!fast bra SLOW_RESCALE;
    sub.u32 a,a,fac;
    sub.u32 b,b,fac;
    sub.u32 c,c,fac;
    sub.u32 d,d,fac;
    bra WRITE_RESCALE;
    SLOW_RESCALE:
    min.u32 delta,delta,15;
    sub.s32 fac,15,delta;
    shl.b32 fac,fac,10;
    prmt.b32 fac,fac,fac,0x1010;
    """
    for reg in ("a", "b", "c", "d"):
        asm += """
        mov.b32 {xlo,xhi},WORD;
        cvt.rn.f16x2.e4m3x2 t,xlo;
        cvt.rn.f16x2.e4m3x2 u,xhi;
        mul.f16x2 t,t,fac;
        mul.f16x2 u,u,fac;
        cvt.rn.satfinite.e4m3x2.f16x2 xlo,t;
        cvt.rn.satfinite.e4m3x2.f16x2 xhi,u;
        mov.b32 WORD,{xlo,xhi};
        """.replace(
            "WORD", reg
        )
    asm += """
    WRITE_RESCALE:
    st.shared.v4.b32 [kdst],{a,b,c,d};
    NEXT:
    add.u32 iter,iter,1;
    setp.le.u32 loop,iter,ROWMASK;
    @loop bra RESCALE;
    DONE_RESCALE:
    mov.b32 $0,flag;
    }
    """
    return (
        asm.replace("PAGESTRIDE", str(page_stride))
        .replace("PANEL", str(panel_rows * 128))
        .replace("TIDMASK", str(warps * 32 - 1))
        .replace("WARPS", str(warps))
        .replace("ROWMASK", str(block // warps - 1))
    )


@triton.jit
def gather_tile(
    K,
    Idx,
    sk,
    skr,
    ks,
    b,
    first_pos,
    N,
    T: tl.constexpr,
    BK: tl.constexpr,
    STRIDE: tl.constexpr,
):
    return tl.inline_asm_elementwise(
        (
            warp_gather_asm(BK, STRIDE, BK, 8)
            if BK == 128
            else gather_asm(BK, STRIDE, BK, 8)
        ),
        "=r,r,l,l,r,r,r,r,r,r",
        [
            tl.arange(0, 256),
            K.to(tl.uint64),
            (Idx + b.to(tl.int64) * T).to(tl.uint64),
            tle.gpu.local_ptr(sk, (0, 0)).to(tl.uint64).to(tl.uint32),
            tle.gpu.local_ptr(skr, (0, 0)).to(tl.uint64).to(tl.uint32),
            tle.gpu.local_ptr(ks, (0,)).to(tl.uint64).to(tl.uint32),
            N,
            first_pos,
            tl.full((), T, tl.uint32),
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def producer(
    K,
    KR,
    Idx,
    Flags,
    RepairState,
    Normalized,
    Scale,
    sk,
    skr,
    ks,
    kfull,
    kfree,
    N,
    T: tl.constexpr,
    STRIDE: tl.constexpr,
    S: tl.constexpr,
    BK: tl.constexpr,
    BUFS: tl.constexpr,
    PREPACK: tl.constexpr,
):
    b = tl.program_id(0)
    first, count = split_block_range(tl.program_id(2), tl.cdiv(T, BK), S)
    repair = tl.full((), 0, tl.int32)
    for step in range(count):
        buf, cycle = step % BUFS, step // BUFS
        if BUFS == 1:
            if step > 0:
                tle.gpu.barrier_wait(kfree[0])
            else:
                pass
        else:
            tle.gpu.barrier_wait(kfree[buf], phaseIdx=cycle)
        if PREPACK:
            k = tl.arange(0, BK)
            g = tl.arange(0, 8)
            pos = (first + step) * BK + k
            ids = tl.load(
                Idx + b.to(tl.int64) * T + pos,
                pos < T,
                -1,
                eviction_policy="evict_first",
            )
            valid = (ids >= 0) & (ids < N)
            safe = tl.where(valid, ids, 0)
            signed_scale = tl.load(Scale + safe)
            repair |= tl.max((valid & (signed_scale < 0.0)).to(tl.int32), 0)
            tl.store(
                tle.gpu.local_ptr(ks.slot(buf)),
                tl.where(valid, tl.abs(signed_scale), -1.0),
            )
            rows = tl.broadcast_to(k[:, None], (BK, 8))
            cols = tl.broadcast_to((g * 8)[None, :], (BK, 8))
            dst = tle.gpu.local_ptr(skr.slot(buf), (rows, cols))
            src = KR + safe.to(tl.int64)[:, None] * 64 + cols
            tl.inline_asm_elementwise(
                "{ cp.async.cg.shared.global [$1], [$2], 16, $3; mov.u32 $0, 0; }",
                "=r,r,l,r",
                [
                    dst.to(tl.uint64).to(tl.uint32),
                    src.to(tl.uint64),
                    tl.where(valid[:, None], 16, 0),
                ],
                dtype=tl.uint32,
                is_pure=False,
                pack=1,
            )
            byte_cols = cols * 2
            for tile in tl.static_range(4):
                dst_k = tle.gpu.local_ptr(
                    tle_subslice(sk.slot(buf), [0, tile * 128], [BK, 128]),
                    (rows, byte_cols),
                )
                src_k = (
                    Normalized
                    + safe.to(tl.int64)[:, None] * 448
                    + tile * 128
                    + byte_cols
                )
                tl.inline_asm_elementwise(
                    "{ .reg .pred p; setp.ne.b32 p,$4,0; "
                    "@p cp.async.cg.shared.global [$1], [$2], 16, $3; mov.u32 $0, 0; }",
                    "=r,r,l,r,r",
                    [
                        dst_k.to(tl.uint64).to(tl.uint32),
                        src_k.to(tl.uint64),
                        tl.where(valid[:, None], 16, 0),
                        ((tile < 3) | (byte_cols < 64)).to(tl.int32),
                    ],
                    dtype=tl.uint32,
                    is_pure=False,
                    pack=1,
                )
        else:
            flag = gather_tile(
                K,
                Idx,
                sk.slot(buf),
                skr.slot(step % 4 if BUFS == 2 else 0),
                ks.slot(buf),
                b,
                (first + step) * BK,
                N,
                T,
                BK,
                STRIDE,
            )
            repair |= tl.max(flag, 0).to(tl.int32)
        tl.inline_asm_elementwise(
            (
                "{ cp.async.commit_group; cp.async.wait_group 0; fence.proxy.async.shared::cta; mov.u32 $0, $1; }"
                if PREPACK
                else "{ fence.proxy.async.shared::cta; mov.u32 $0, $1; }"
            ),
            "=r,r",
            [tl.arange(0, 256)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        tl.debug_barrier()
        if BUFS == 1:
            tle.gpu.barrier_arrive(kfull[0])
        else:
            tle.gpu.barrier_arrive(kfull[buf], phaseIdx=cycle)
    tl.atomic_or(Flags + b, 1, mask=repair != 0, sem="relaxed")
    tl.atomic_or(RepairState, 1, mask=repair != 0, sem="relaxed")
