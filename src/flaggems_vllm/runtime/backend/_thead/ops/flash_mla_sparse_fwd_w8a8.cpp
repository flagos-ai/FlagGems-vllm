// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Each warp computes 16x16 score fragments. Queries stay in registers while
// asynchronous copies alternate between two KV tiles in shared memory.
constexpr int BH = 64, BN = 128, W = 16, CUBE = 256;
constexpr int GROUP = (ROPE || PARTS > 1) ? 32 : 16;
constexpr bool LOCAL = SHORT_LOCAL || (PARTS > 1 && ROPE);
constexpr float FACTOR = 224.0f;
// Used for one 640-token selection. Even 640 * 255 * 128 * 16 fits in INT32.
constexpr int residual_radix = 16;
static_assert(!REFINE_LAST || (PARTS == 1 && ROPE == 0),
              "last-tile refinement");
using u32 = unsigned;
using u16 = unsigned short;
using u64 = unsigned long long;
using i8 = signed char;
template <class T> using G = T __attribute__((address_space(1)));
template <class T> using S = T __attribute__((address_space(3)));
extern "C" u32 tid_x() asm("llvm.nvvm.read.ptx.sreg.tid.x");
extern "C" u32 cta_count() asm("llvm.nvvm.read.ptx.sreg.nctaid.y");
extern "C" void sync() asm("llvm.nvvm.barrier0");
extern "C" float ex2(float) asm("llvm.nvvm.ex2.approx.ftz.f");
extern "C" float lg2(float) asm("llvm.nvvm.lg2.approx.ftz.f");
extern "C" u32 trans8(u32) asm("__flagtree_ppu_intrinsic__mat.trans.b8");
extern "C" void commit() asm("llvm.nvvm.cp.async.commit.group");
extern "C" void wait0(int) asm("llvm.nvvm.cp.async.wait.group");
extern "C" int shfl(int, int, int, int) asm("llvm.nvvm.shfl.sync.idx.i32");
#define INLINE inline __attribute__((always_inline))
#define UNROLL _Pragma("clang loop unroll(full)")
struct F {
  u32 f[4];
};
struct C {
  int c[8];
};
struct FC {
  float c[8];
};
INLINE void mma(C &c, F a, F b) {
  asm("ppu.mma.sync.aligned.m16n16k32.row.col.satfinite.s32.s8.s8.s32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+r"(c.c[0]), "+r"(c.c[1]), "+r"(c.c[2]), "+r"(c.c[3]), "+r"(c.c[4]),
        "+r"(c.c[5]), "+r"(c.c[6]), "+r"(c.c[7])
      : "r"(a.f[0]), "r"(a.f[1]), "r"(a.f[2]), "r"(a.f[3]), "r"(b.f[0]),
        "r"(b.f[1]), "r"(b.f[2]), "r"(b.f[3]));
}
INLINE void umma(C &c, F a, F b) {
  asm("ppu.tc01.mma.sync.aligned.m16n16k32.row.col.s32.u8.s8.s32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+r"(c.c[0]), "+r"(c.c[1]), "+r"(c.c[2]), "+r"(c.c[3]), "+r"(c.c[4]),
        "+r"(c.c[5]), "+r"(c.c[6]), "+r"(c.c[7])
      : "r"(a.f[0]), "r"(a.f[1]), "r"(a.f[2]), "r"(a.f[3]), "r"(b.f[0]),
        "r"(b.f[1]), "r"(b.f[2]), "r"(b.f[3]));
}

INLINE void fmma(FC &c, F a, F b) {
  asm("ppu.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};"
      : "+f"(c.c[0]), "+f"(c.c[1]), "+f"(c.c[2]), "+f"(c.c[3]), "+f"(c.c[4]),
        "+f"(c.c[5]), "+f"(c.c[6]), "+f"(c.c[7])
      : "r"(a.f[0]), "r"(a.f[1]), "r"(a.f[2]), "r"(a.f[3]), "r"(b.f[0]),
        "r"(b.f[1]), "r"(b.f[2]), "r"(b.f[3]));
}
INLINE u32 perm(u32 x, u32 y, u32 code) {
  u32 z;
  asm("ppu.prmt.b32 %0,%1,%2,%3;" : "=r"(z) : "r"(x), "r"(y), "r"(code));
  return z;
}
INLINE int round_nearest(float x) {
  int v;
  asm("ppu.cvt.rni.s32.f32 %0,%1;" : "=r"(v) : "f"(x));
  return v;
}
INLINE u32 pack_probabilities(const u32 (&z)[4]) {
  u32 ab = perm(z[0], z[1], 0x5410), cd = perm(z[2], z[3], 0x5410);
  return perm(ab, cd, 0x6420);
}
INLINE u16 bf(float x) {
  u32 a = __builtin_bit_cast(u32, x);
  return (a + 0x7fff + ((a >> 16) & 1)) >> 16;
}
INLINE u32 sw(int t, int col) {
  return ((col / CUBE) * BN + t) * CUBE +
         ((col % (CUBE)) ^ ((t & ((CUBE / 16) - 1)) * 16));
}
INLINE u32 word(S<i8> *s, int t, int col) {
  return *(S<u32> *)(s + sw(t, col));
}
extern "C" u32 ppu_sparse_mla(u64 qp, u64 qrp, u64 vp, u64 krp, u64 qsp,
                              u64 ksp, u64 idp, u64 outp, u64 lsep, u64 errp,
                              u64 anchors, u32 shared, int row, int headblock,
                              int K, int tokens, float scale) {
  constexpr int WM = BH / 16, WN = W / WM, NT = BN / 16 / WN,
                NF = 512 / 16 / WN;
  static_assert(BN >= 16 * WN && NF > 0, "tile");
  unsigned tid = tid_x(), lane = tid & 31, warp = tid >> 5, hm = warp / WN,
           wn = warp % WN, lr = lane >> 2, quad = lane & 3;
  int head = hm * 16 + lr;
  int requests = cta_count() / PARTS, split = row % PARTS;
  row /= PARTS;
  int tiles_total = (K + BN - 1) / BN;
  int first = (tiles_total * split / PARTS) * BN,
      last = (tiles_total * (split + 1) / PARTS) * BN;
  G<int> *ids = (G<int> *)idp + (u64)row * K + first;
  K = (last < K ? last : K) - first;
  int dstrow = split * requests + row;
  G<float> *ks = (G<float> *)ksp;
  G<u32> *q = (G<u32> *)qp + ((row * 64 + headblock * BH + head) * 128);
  G<i8> *v = (G<i8> *)vp;
  float qs[2] = {((G<float> *)qsp)[row * 64 + headblock * BH + head],
                 ((G<float> *)qsp)[row * 64 + headblock * BH + head + 8]};
  S<i8> *tiles = (S<i8> *)(u64)shared;
  S<i8> *ropes = tiles + 2 * BN * 512;
  S<u32> *packed = (S<u32> *)(ropes + 2 * BN * ROPE * 2);
  S<float> *allscales = (S<float> *)(packed + BH * BN / 4);
  S<float> *sums = allscales + 2 * BN;
  S<u32> *low_packed = (S<u32> *)(sums + 2 * WN * BH);
  float anchor;
  // Short selections and RoPE partitions scan their scales in the same CTA.
  if constexpr (LOCAL) {
    float z = 1.e-30f, scale_sum = 0.f;
    int count = 0;
    for (int i = tid; i < K; i += W * 32) {
      int idx = ids[i];
      if ((unsigned)idx < (unsigned)tokens) {
        float x = ks[idx];
        z = z > x ? z : x;
        if constexpr (REFINE_LAST) {
          scale_sum += x;
          ++count;
        }
      }
    }
    UNROLL for (int a = 1; a < 32; a *= 2) {
      float x = __builtin_bit_cast(
          float, shfl(-1, __builtin_bit_cast(int, z), lane ^ a, 31));
      z = z > x ? z : x;
    }
    if constexpr (REFINE_LAST) {
      UNROLL for (int a = 1; a < 32; a *= 2) {
        scale_sum += __builtin_bit_cast(
            float, shfl(-1, __builtin_bit_cast(int, scale_sum), lane ^ a, 31));
        count += shfl(-1, count, lane ^ a, 31);
      }
    }
    if (lane == 0) {
      sums[warp] = z;
      if constexpr (REFINE_LAST) {
        sums[W + warp] = scale_sum;
        sums[2 * W + warp] = (float)count;
      }
    }
    sync();
    anchor = 1.e-30f;
    scale_sum = 0.f;
    float valid = 0.f;
    UNROLL for (int j = 0; j < W; ++j) {
      float x = sums[j];
      anchor = anchor > x ? anchor : x;
      if constexpr (REFINE_LAST) {
        scale_sum += sums[W + j];
        valid += sums[2 * W + j];
      }
    }
    if constexpr (REFINE_LAST) {
      float typical = scale_sum / (124.f * (valid > 0 ? valid : 1.f));
      float peak = anchor / 152.f;
      anchor = FACTOR * (peak > typical ? peak : typical);
    }
  } else
    anchor = ((G<float> *)anchors)[row];
  float factor = FACTOR / anchor;
  F qrfrag[4];
  if constexpr (ROPE) {
    G<u32> *qr = (G<u32> *)qrp + ((row * 64 + headblock * BH + head) * 32);
    UNROLL for (int k = 0; k < 4; ++k) {
      int off = k * 8 + quad;
      qrfrag[k] = {
          {qr[off], qr[off + 4], qr[off + 8 * 32], qr[off + 8 * 32 + 4]}};
    }
  }
  F qfrag[16];
  UNROLL for (int k = 0; k < 16; ++k) {
    int off = k * 8 + quad;
    qfrag[k] = {{q[off], q[off + 4], q[off + 8 * 128], q[off + 8 * 128 + 4]}};
  }
  // Reuse each value fragment across head tiles owned by the same warp.
  constexpr int pv_head_warps = ROPE ? (PARTS > 1 ? WM : 2) : 1;
  constexpr int pv_feature_warps = W / pv_head_warps;
  constexpr int head_tiles = WM / pv_head_warps;
  constexpr int feature_tiles = 512 / (16 * pv_feature_warps);
  const int pv_head = warp / pv_feature_warps,
            pv_feature = warp % pv_feature_warps;
  C value_acc[head_tiles][feature_tiles] = {};
  float den[2] = {}, err[2] = {};
  constexpr int PER = BN / W;
  auto index_for = [&](int start) {
    int pos = start + warp * PER + lane;
    return lane < PER && pos < K ? ids[pos] : -1;
  };
  auto copy = [&](int src, int slot, int ix) {
    if (lane < PER)
      allscales[slot * BN + warp * PER + lane] =
          (unsigned)ix < (unsigned)tokens ? ks[ix] : 0.f;
    UNROLL for (int j = 0; j < PER / (32 / GROUP); ++j) {
      int ii = j * (32 / GROUP) + lane / GROUP;
      int id = shfl(-1, ix, ii, 31), t = warp * PER + ii;
      bool good = (unsigned)id < (unsigned)tokens;
      u64 gp = (u64)(v + (u64)(good ? id : 0) * 512) + (lane % GROUP) * 16;
      UNROLL for (int f = 0; f < 32 / GROUP; ++f) {
        int cc = (lane % GROUP + f * GROUP) * 16;
        u32 dst = shared + slot * BN * 512 + sw(t, cc);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst),
                     "l"(gp + f * GROUP * 16), "r"(good ? 16 : 0)
                     : "memory");
      }
    }
    if constexpr (ROPE) {
      UNROLL for (int j = 0; j < PER / 4; ++j) {
        int ii = j * 4 + lane / 8, id = shfl(-1, ix, ii, 31),
            tt = warp * PER + ii;
        bool good = (unsigned)id < (unsigned)tokens;
        u32 dst = shared + 2 * BN * 512 + slot * BN * 128 + tt * 128 +
                  (((lane & 7) * 16) ^ ((tt & 7) * 16));
        u64 gp = krp + (u64)(good ? id : 0) * 128 + (lane & 7) * 16;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst),
                     "l"(gp), "r"(good ? 16 : 0)
                     : "memory");
      }
    }
    commit();
  };
  copy(0, 0, index_for(0));
  int ahead = index_for(BN);
  for (int src = 0; src < K; src += BN) {
    int slot = (src / BN) % 2;
    wait0(0);
    sync();
    S<i8> *sv = tiles + slot * BN * 512;
    u32 smbase = shared + slot * BN * 512;
    S<float> *scales = allscales + slot * BN;
    if (src + BN < K)
      copy(src + BN, 1 - slot, ahead);
    ahead = index_for(src + 2 * BN);
    C scores[NT] = {};
    UNROLL for (int k = 0; k < 16; ++k) {
      UNROLL for (int j = 0; j < NT; ++j) {
        int t = (wn + j * WN) * 16 + (lane & 7) + (lane & 16) / 2;
        int col = k * 32 + (lane & 8) * 2;
        u32 addr = smbase + sw(t, col);
        F b;
        asm("ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(b.f[0]), "=r"(b.f[1]), "=r"(b.f[2]), "=r"(b.f[3])
            : "r"(addr)
            : "memory");
        mma(scores[j], qfrag[k], b);
      }
    }
    UNROLL for (int j = 0; j < NT; ++j) {
      int nt = wn + j * WN;
      FC rrscore = {};
      if constexpr (ROPE) {
        UNROLL for (int k = 0; k < 4; ++k) {
          int tt = nt * 16 + (lane & 7) + (lane & 16) / 2;
          int col = k * 32 + (lane & 8) * 2;
          u32 addr = shared + 2 * BN * 512 + slot * BN * 128 + tt * 128 +
                     (col ^ ((tt & 7) * 16));
          F f;
          asm("ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, "
              "[%4];"
              : "=r"(f.f[0]), "=r"(f.f[1]), "=r"(f.f[2]), "=r"(f.f[3])
              : "r"(addr)
              : "memory");
          fmma(rrscore, qrfrag[k], f);
        }
      }

      UNROLL for (int rr = 0; rr < 2; ++rr) {
        u32 z[4], low[4];
        UNROLL for (int i = 0; i < 4; ++i) {
          int t = nt * 16 + quad + i * 4;
          float ksc = scales[t];
          float score = (float)scores[j].c[rr * 4 + i] + rrscore.c[rr * 4 + i];
          float log2_weight = score * qs[rr] * ksc * scale;
          float p = ksc > 0.f ? ex2(log2_weight) : 0.f;
          den[rr] += p;
          float norm = p * (ksc * factor);
          z[i] = round_nearest(norm < 255.f ? norm : 255.f);
          float delta = norm - (float)z[i];
          if constexpr (REFINE_LAST) {
            if (src + BN >= K) {
              // Reuse the resident V tile for a second, signed INT8 product.
              float extra = delta * residual_radix;
              float clipped =
                  extra < 127.f ? (extra > -128.f ? extra : -128.f) : 127.f;
              int code = round_nearest(clipped);
              low[i] = (u32)code;
              delta -= (float)code / residual_radix;
            }
          }
          float residual = __builtin_fabsf(delta);
          bool unsafe = ksc > 0.f && !(p > 1.e-16f && p < 1.e16f);
          err[rr] += unsafe ? __builtin_inff() : residual;
        }
        u32 high_word = pack_probabilities(z);
        int peer = (lane & 16) | (quad << 2) | (lr & 3);
        int offset =
            (hm * (BN / 32) + nt / 2) * 128 + (rr * 2 + (nt & 1)) * 32 + peer;
        packed[offset] = trans8(shfl(-1, high_word, peer, 31));
        if constexpr (REFINE_LAST) {
          if (src + BN >= K)
            low_packed[offset] =
                trans8(shfl(-1, pack_probabilities(low), peer, 31));
        }
      }
    }
    sync();
    const int passes = REFINE_LAST && src + BN >= K ? 2 : 1;
    for (int pass = 0; pass < passes; ++pass) {
      if (pass == 1) {
        UNROLL for (int m = 0; m < head_tiles; ++m) {
          UNROLL for (int j = 0; j < feature_tiles; ++j) {
            UNROLL for (int e = 0; e < 8; ++e) value_acc[m][j].c[e] *=
                residual_radix;
          }
        }
      }
      UNROLL for (int k = 0; k < BN / 32; ++k) {
        F ah[head_tiles];
        UNROLL for (int m = 0; m < head_tiles; ++m)
            UNROLL for (int i = 0; i < 4; ++i) {
          int off = ((pv_head + m * pv_head_warps) * (BN / 32) + k) * 128 +
                    i * 32 + lane;
          ah[m].f[i] = (pass == 0 ? packed : low_packed)[off];
        }
        UNROLL for (int j = 0; j < feature_tiles; ++j) {
          int t = k * 32 + quad * 4 + (lr & 3);
          int col = 16 * (pv_feature + j * pv_feature_warps) + (lane >> 4) * 4;
          F b = {{trans8(word(sv, t, col)), trans8(word(sv, t + 16, col)),
                  trans8(word(sv, t, col + 8)),
                  trans8(word(sv, t + 16, col + 8))}};
          UNROLL for (int m = 0; m < head_tiles; ++m) {
            if (pass == 0)
              umma(value_acc[m][j], ah[m], b);
            else
              mma(value_acc[m][j], ah[m], b);
          }
        }
      }
    }
  }
  UNROLL for (int r = 0; r < 2; ++r) {
    float z = den[r];
    z += __builtin_bit_cast(float,
                            shfl(-1, __builtin_bit_cast(int, z), lane ^ 1, 31));
    z += __builtin_bit_cast(float,
                            shfl(-1, __builtin_bit_cast(int, z), lane ^ 2, 31));
    float a = err[r];
    a += __builtin_bit_cast(float,
                            shfl(-1, __builtin_bit_cast(int, a), lane ^ 1, 31));
    a += __builtin_bit_cast(float,
                            shfl(-1, __builtin_bit_cast(int, a), lane ^ 2, 31));
    if (quad == 0) {
      sums[wn * BH + head + 8 * r] = z;
      sums[WN * BH + wn * BH + head + 8 * r] = a;
    }
  }
  sync();
  UNROLL for (int m = 0; m < head_tiles; ++m)
      UNROLL for (int r = 0; r < 2; ++r) {
    int hr = (pv_head + m * pv_head_warps) * 16 + lr + 8 * r;
    float total = 0;
    UNROLL for (int n = 0; n < WN; ++n) total += sums[n * BH + hr];
    float unit = anchor / FACTOR;
    float delta = 0;
    UNROLL for (int nn = 0; nn < WN; ++nn) delta += sums[(nn + WN) * BH + hr];
    int ix = dstrow * 64 + headblock * BH + hr;
    if constexpr (PARTS == 1)
      unit /= total > 0 ? total : 1.f;
    if (quad == 0 && pv_feature == 0) {
      // Multiplication by 128 bounds the error for every signed value byte.
      float bound = delta * unit * 128.f;
      if (!(anchor <= 1.e16f && total < 1.e30f))
        bound = __builtin_inff();
      ((G<float> *)errp)[ix] = bound;
      if constexpr (PARTS == 1)
        ((G<float> *)lsep)[ix] =
            total > 0 ? lg2(total) * .6931471805599453f : __builtin_inff();
      else
        ((G<float> *)lsep)[ix] = total;
    }
    UNROLL for (int j = 0; j < feature_tiles; ++j)
        UNROLL for (int a = 0; a < 4; ++a) {
      int e = r * 4 + a;
      float out = ((float)value_acc[m][j].c[e]) *
                  (REFINE_LAST ? unit / residual_radix : unit);
      auto dest = (u64)ix * 512 + (pv_feature + j * pv_feature_warps) * 16 +
                  quad + a * 4;
      if constexpr (PARTS == 1)
        ((G<u16> *)outp)[dest] = bf(out);
      else
        ((G<float> *)outp)[dest] = out;
    }
  }
  return tid;
}
