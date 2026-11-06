// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <maca.h>
#include <math.h>

#ifndef FMLA_NO_ROPE
#define FMLA_NO_ROPE 0
#endif

typedef unsigned uint2_ __attribute__((ext_vector_type(2)));
typedef unsigned uint4_ __attribute__((ext_vector_type(4)));
typedef int int4_ __attribute__((ext_vector_type(4)));
typedef float float2_ __attribute__((ext_vector_type(2)));
typedef float float4_ __attribute__((ext_vector_type(4)));
#define GLOBAL __attribute__((address_space(1)))
#define SHARED __attribute__((address_space(3)))
namespace {
template <class T>
__device__ __forceinline__ const GLOBAL T *input(long long p) {
  return (const GLOBAL T *)(__UINTPTR_TYPE__)p;
}
template <class T> __device__ __forceinline__ GLOBAL T *output(long long p) {
  return (GLOBAL T *)(__UINTPTR_TYPE__)p;
}
__device__ __forceinline__ int uniform(int x) {
  return __builtin_mxc_readfirstlane(x);
}
__device__ __forceinline__ long long uniform(long long x) {
  return (
      long long)((unsigned long long)(unsigned)uniform((int)x) |
                 ((unsigned long long)(unsigned)uniform((int)(x >> 32)) << 32));
}
__device__ __forceinline__ void sync() {
  __builtin_mxc_arrive_bsmcnt(0);
  __builtin_mxc_barrier_ex(2);
}
__device__ __forceinline__ unsigned perm(unsigned hi, unsigned lo,
                                         unsigned selector) {
  return __builtin_mxc_byte_perm(hi, lo, selector);
}
__device__ __forceinline__ int4_ mma(unsigned a, unsigned b, int4_ c) {
  return __builtin_mxc_mma_16x16x16i8((int)a, (int)b, c);
}
__device__ __forceinline__ float4_ rope_mma(uint2_ a, uint2_ b, float4_ c,
                                            int bf16) {
  if (bf16)
    return __builtin_mxc_mma_16x16x16bf16(a, b, c);
  return __builtin_mxc_mma_16x16x16f16(a, b, c);
}
// Sparse tables contain compact physical token IDs rather than page IDs.
__device__ __forceinline__ long long cache_page(long long table, int batch,
                                                long long stride, int page) {
#if defined(FMLA_SPARSE)
  return page;
#else
  return uniform(input<int>(table)[batch * stride + page]);
#endif
}
__device__ __forceinline__ long long cache_token(long long table, int batch,
                                                 long long page, int token) {
#if defined(FMLA_SPARSE)
  const int index = input<int>(table)[(long long)batch * FMLA_CAP + page * 64 + token];
#if defined(FMLA_SPARSE_RAW)
  return index >= 0 && index < FMLA_CACHE_TOKENS ? index : 0;
#else
  return index;
#endif
#else
  return page * 64 + token;
#endif
}
__device__ __forceinline__ bool cache_valid(long long table, int batch,
                                             long long page, int token) {
#if defined(FMLA_SPARSE_RAW)
  const int index = input<int>(table)[(long long)batch * FMLA_CAP + page * 64 + token];
  return index >= 0 && index < FMLA_CACHE_TOKENS;
#else
  return true;
#endif
}
__device__ __forceinline__ unsigned pack_bf16(float a, float b) {
  auto pair = __builtin_mxc_cvt_pk_f32tobf16((float2_){a, b});
  return *(unsigned *)&pair;
}
} // namespace

extern "C" __device__ __attribute__((const)) float
reciprocal(float) __asm("llvm.mxc.rcp.f32");

#if defined(FMLA_STAGE_SMALL)

#ifndef HEADS
#define HEADS 16
#endif
#if defined(FMLA_SPARSE) && HEADS == 16 && !FMLA_NO_ROPE
#define FMLA_SEPARATE_ROPE_STATS 1
#else
#define FMLA_SEPARATE_ROPE_STATS 0
#endif
#if defined(FMLA_SPARSE) && !FMLA_NO_ROPE
#define FMLA_PIPELINE_ROPE 1
#else
#define FMLA_PIPELINE_ROPE 0
#endif
namespace {
constexpr int NW = 64 / HEADS, NR = 4 / NW, QV = 8 / NW;

__device__ __forceinline__ int koff(int token, int dim) {
  return token * 128 + (dim ^ (((token >> 1) & 7) * 16));
}
__device__ __forceinline__ int roff(int token, int dim) {
  return token * 64 + (dim ^ (((token >> 1) & 3) * 16));
}

} // namespace

extern "C" __device__ __attribute__((used, always_inline)) float
mla_int8_small(long long qp, long long qrp, long long kp, long long krp,
               long long qsp, long long ksp, long long tp, long long lp,
               long long pp, long long sp, long long op, long long lsep,
               long long ts, long long ls, long long orderp, int batches,
               int hq, int sq, int splits, int causal, int bf16,
               float scale_log2, int head_tile, int batch, int split) {
  qp = uniform(qp);
  qrp = uniform(qrp);
  kp = uniform(kp);
  krp = uniform(krp);
  qsp = uniform(qsp);
  ksp = uniform(ksp);
  tp = uniform(tp);
  lp = uniform(lp);
  pp = uniform(pp);
  sp = uniform(sp);
  op = uniform(op);
  lsep = uniform(lsep);
  ts = uniform(ts);
  ls = uniform(ls);
  orderp = uniform(orderp);
  batches = uniform(batches);
  hq = uniform(hq);
  sq = uniform(sq);
  splits = uniform(splits);
  causal = uniform(causal);
  bf16 = uniform(bf16);
  head_tile = uniform(head_tile);
  batch = uniform(batch);
  split = uniform(split);
  batches = FMLA_B;
  hq = FMLA_HQ;
  sq = FMLA_SQ;
  splits = FMLA_SPLITS;
  causal = FMLA_CAUSAL;
  bf16 = FMLA_BF16;
  ts = FMLA_CAP / 64;
  ls = 1;
  scale_log2 = FMLA_SCALE;
  if (orderp)
    batch = uniform(input<int>(orderp)[batch]);
  const int t = (threadIdx.x & 255), lane = t & 63, wave = t >> 6,
            row = lane & 15, group = lane >> 4;
  const int wn = wave % NW, wm = wave / NW,
            head = head_tile * HEADS + wm * 16 + row;
  const int allheads = hq * sq, length = uniform(input<int>(lp)[batch * ls]);
  const int pages = (length + 63) / 64, begin = pages * split / splits,
            end = pages * (split + 1) / splits;
  const int bound = causal ? max(length - sq + head / hq + 1, 0) : length;
  const long long qrow = (long long)batch * allheads + head;
  const float qs = input<float>(qsp)[qrow];
#if HEADS == 32
  unsigned query[32];
  uint2_ query_rope[4];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const uint4_ words = *input<uint4_>(qp + qrow * 512 + k * 64 + group * 16);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      query[4 * k + j] = words[j];
  }
#pragma unroll
  for (int k = 0; k < (FMLA_NO_ROPE ? 0 : 2); ++k) {
    const uint4_ words =
        *input<uint4_>(qrp + (qrow * 64 + group * 16 + k * 8) * 2);
    query_rope[2 * k] = (uint2_){words[0], words[1]};
    query_rope[2 * k + 1] = (uint2_){words[2], words[3]};
  }

#define QNOPE(chunk, k)                                                        \
  ((uint4_){query[(chunk) * 8 + (k) * 4], query[(chunk) * 8 + (k) * 4 + 1],    \
            query[(chunk) * 8 + (k) * 4 + 2],                                  \
            query[(chunk) * 8 + (k) * 4 + 3]})
#define QROPE(k)                                                               \
  ((uint4_){query_rope[2 * (k)][0], query_rope[2 * (k)][1],                    \
            query_rope[2 * (k) + 1][0], query_rope[2 * (k) + 1][1]})
#else
  __shared__ unsigned char qbuffer[8192];
  __shared__ unsigned char qrbuffer[2048];
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const int i = t + c * 256, r = i >> 5, dim = ((i & 31) ^ (r & 7)) * 16;
    void *dest = qbuffer + ((t & ~63) + c * 256) * 16;
    void *src = (void *)input<unsigned char>(
        qp + ((long long)batch * allheads + head_tile * 16 + r) * 512 + dim);
    __builtin_mxc_ldg_b128_bsm_predicator(dest, src, 0, true, true, false, true,
                                          1, 2, 40);
  }
  if (!FMLA_NO_ROPE && t < 128) {
    const int r = t >> 3, dim = ((t & 7) ^ ((r & 3) * 2)) * 8;
    void *dest = qrbuffer + (t & ~63) * 16;
    void *src = (void *)input<unsigned char>(
        qrp +
        (((long long)batch * allheads + head_tile * 16 + r) * 64 + dim) * 2);
    __builtin_mxc_ldg_b128_bsm_predicator(dest, src, 0, true, true, false, true,
                                          1, 2, 40);
  }
  __builtin_mxc_arrive_gvmcnt(0);
  sync();
#define QNOPE(chunk, k)                                                        \
  (*(const uint4_ *)(qbuffer + row * 512 +                                     \
                     (((chunk) * 128 + (k) * 64 + group * 16) ^                \
                      ((row & 7) * 16))))
#define QROPE(k)                                                               \
  (*(const uint4_ *)(qrbuffer + (row * 64 + ((group * 16 + (k) * 8) ^          \
                                             ((row & 3) * 16))) *              \
                                    2))
#endif
  // Two K planes overlap H16 copies with QK; H32 retains four planes for PV.
  __shared__ unsigned char planes[HEADS == 32 ? 4 : 2][8192];
  auto copy_wait = [&]() {
    __builtin_mxc_arrive_gvmcnt(0);
    __builtin_mxc_barrier_ex(2);
  };
  auto load_plane = [&](long long page, int page_index, int chunk) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int i = t + c * 256, token = i >> 3;
      const int dim = ((i & 7) ^ ((token >> 1) & 7)) * 16;
      void *dest = planes[HEADS == 32 ? chunk : (chunk & 1)] +
                   ((t & ~63) + c * 256) * 16;
      void *src = (void *)input<unsigned char>(kp + cache_token(tp, batch, page, token) * 512 +
                                               chunk * 128 + dim);
      __builtin_mxc_ldg_b128_bsm_predicator(dest, src, 0, true, true, false,
                                            true, 1, 2, 40);
    }
  };
  auto load_rope = [&](long long page, int page_index) {
#pragma unroll
    for (int c = 0; c < (FMLA_NO_ROPE ? 0 : 2); ++c) {
      const int i = t + c * 256, token = i >> 3,
                dim = ((i & 7) ^ (((token >> 1) & 3) * 2)) * 8;
      void *dest = planes[0] + ((t & ~63) + c * 256) * 16;
      void *src = (void *)input<unsigned char>(
          krp + (cache_token(tp, batch, page, token) * 64 + dim) * 2);
      __builtin_mxc_ldg_b128_bsm_predicator(dest, src, 0, true, true, false,
                                            true, page_index * 64 + token,
                                            length, 40);
    }
  };
  auto fragments = [&](const unsigned char *base, int token, int col,
                       unsigned (&words)[QV]) {
    const int aligned = col & ~3;
    const unsigned a = *(const unsigned *)(base + koff(token, aligned));
    const unsigned b = *(const unsigned *)(base + koff(token + 1, aligned));
    const unsigned c = *(const unsigned *)(base + koff(token + 2, aligned));
    const unsigned d = *(const unsigned *)(base + koff(token + 3, aligned));
    if (QV == 2) {
      const unsigned selector = 0x05040100 + (col & 2) * 0x01010101;
      const unsigned u = perm(b, a, selector), v = perm(d, c, selector);
      words[0] = perm(v, u, 0x06040200);
      words[1] = perm(v, u, 0x07050301);
    } else {
      const unsigned u = perm(b, a, 0x05040100), v = perm(d, c, 0x05040100);
      const unsigned w = perm(b, a, 0x07060302), z = perm(d, c, 0x07060302);
      words[0] = perm(v, u, 0x06040200);
      words[1] = perm(v, u, 0x07050301);
      words[2] = perm(z, w, 0x06040200);
      words[3] = perm(z, w, 0x07050301);
    }
  };
  float4_ acc[QV * 4];
#pragma unroll
  for (int a = 0; a < QV * 4; ++a)
    acc[a] = (float4_){0, 0, 0, 0};
  float maximum = -INFINITY, denominator = 0.f;
  for (int page_index = begin; page_index < end; ++page_index) {
    const long long page = cache_page(tp, batch, ts, page_index);
    int4_ content[NR];
#pragma unroll
    for (int n = 0; n < NR; ++n)
      content[n] = (int4_){0, 0, 0, 0};
    unsigned retained[HEADS == 16 ? 4 : 1][4][QV];
    if (HEADS == 32) {
#pragma unroll
      for (int chunk = 0; chunk < 4; ++chunk)
        load_plane(page, page_index, chunk);
      copy_wait();
    } else {
      load_plane(page, page_index, 0);
      load_plane(page, page_index, 1);
      copy_wait();
    }
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      if (HEADS == 16 && chunk == 2)
        copy_wait();
      const auto tile = planes[HEADS == 32 ? chunk : (chunk & 1)];
#pragma unroll
      for (int n = 0; n < NR; ++n) {
        const int token = wn * (64 / NW) + row * NR + n;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const uint4_ words =
              *(const uint4_ *)(tile + koff(token, k * 64 + group * 16));
          const uint4_ qwords = QNOPE(chunk, k);
#pragma unroll
          for (int j = 0; j < 4; ++j)
            content[n] = mma(words[j], qwords[j], content[n]);
        }
      }
      if (HEADS == 16 || chunk == 0) {
#pragma unroll
        for (int k = 0; k < 4; ++k)
          fragments(tile, group * 16 + k * 4, wn * (128 / NW) + row * QV,
                    retained[HEADS == 16 ? chunk : 0][k]);
      }
      if (HEADS == 16) {
#if defined(FMLA_SPARSE)
        // Plane 1 is not reused until the following stats/copy barrier.
        if (chunk < 3 || FMLA_NO_ROPE)
          sync();
#else
        sync();
#endif
        if (chunk < 2)
          load_plane(page, page_index, chunk + 2);
        if (FMLA_PIPELINE_ROPE && chunk == 2)
          load_rope(page, page_index);
      } else if (FMLA_PIPELINE_ROPE && chunk == 0) {
        // Plane 0 is retained in registers before the RoPE gather overwrites it.
        sync();
        load_rope(page, page_index);
      }
    }
    if (HEADS == 32 && !FMLA_PIPELINE_ROPE)
      sync();
    if (!FMLA_PIPELINE_ROPE)
      load_rope(page, page_index);
    // The zero-width RoPE specialization issued no copy to wait for.
    if (!FMLA_NO_ROPE)
      copy_wait();
    float4_ rope[NR], scales[NR], probability[NR];
    float local_max = -INFINITY;
#pragma unroll
    for (int n = 0; n < NR; ++n) {
      rope[n] = (float4_){0, 0, 0, 0};
      const int token = wn * (64 / NW) + row * NR + n;
#pragma unroll
      for (int k = 0; k < (FMLA_NO_ROPE ? 0 : 2); ++k) {
        const uint4_ words =
            *(const uint4_ *)(planes[0] + roff(token, group * 16 + k * 8) * 2);
        const uint4_ qwords = QROPE(k);
        rope[n] = rope_mma((uint2_){words[0], words[1]},
                           (uint2_){qwords[0], qwords[1]}, rope[n], bf16);
        rope[n] = rope_mma((uint2_){words[2], words[3]},
                           (uint2_){qwords[2], qwords[3]}, rope[n], bf16);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
        scales[n][j] = input<float>(
            ksp)[cache_token(tp, batch, page,
                              wn * (64 / NW) + group * (NR * 4) + j * NR + n)];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int token_index =
            page_index * 64 + wn * (64 / NW) + group * (NR * 4) + j * NR + n;
        if (token_index >= length)
          scales[n][j] = 0.f;
        probability[n][j] = token_index < bound && cache_valid(tp, batch, page, token_index - page_index * 64)
                                ? ((float)content[n][j] + rope[n][j]) * qs *
                                      scales[n][j] * scale_log2
                                : -INFINITY;
        local_max = fmaxf(local_max, probability[n][j]);
      }
    }
    // K readers already synchronized; only RoPE readers still need this barrier.
    if (!FMLA_NO_ROPE && !FMLA_SEPARATE_ROPE_STATS)
      sync();
    local_max = fmaxf(local_max, __shfl_xor(local_max, 16, 64));
    local_max = fmaxf(local_max, __shfl_xor(local_max, 32, 64));
    // H16 retains V in registers, so plane 1 can hold stats beside RoPE in plane 0.
    float *stats = FMLA_SEPARATE_ROPE_STATS
                       ? (float *)planes[1]
                       : (float *)(planes[0] + (HEADS == 16 ? 4096 : 0));
    if (group == 0)
      stats[wn * HEADS + wm * 16 + row] = local_max;
    sync();
    float page_max = -INFINITY;
#pragma unroll
    for (int n = 0; n < NW; ++n)
      page_max = fmaxf(page_max, stats[n * HEADS + wm * 16 + row]);
    const float next_max = fmaxf(maximum, page_max),
                safe = next_max == -INFINITY ? 0.f : next_max;
    float local_sum = 0.f, local_peak = 0.f;
#pragma unroll
    for (int n = 0; n < NR; ++n) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        probability[n][j] = exp2f(probability[n][j] - safe);
        local_sum += probability[n][j];
        probability[n][j] *= scales[n][j];
        local_peak = fmaxf(local_peak, probability[n][j]);
      }
    }
    local_sum += __shfl_xor(local_sum, 16, 64);
    local_sum += __shfl_xor(local_sum, 32, 64);
    local_peak = fmaxf(local_peak, __shfl_xor(local_peak, 16, 64));
    local_peak = fmaxf(local_peak, __shfl_xor(local_peak, 32, 64));
    if (HEADS == 32)
      sync();
    if (group == 0) {
      stats[((HEADS == 16 ? 2 * NW : 0) + wn) * HEADS + wm * 16 + row] =
          local_sum;
      stats[((HEADS == 16 ? 3 * NW : NW) + wn) * HEADS + wm * 16 + row] =
          local_peak;
    }
    sync();
    float page_sum = 0.f, peak = 0.f;
#pragma unroll
    for (int n = 0; n < NW; ++n) {
      page_sum +=
          stats[((HEADS == 16 ? 2 * NW : 0) + n) * HEADS + wm * 16 + row];
      peak = fmaxf(
          peak,
          stats[((HEADS == 16 ? 3 * NW : NW) + n) * HEADS + wm * 16 + row]);
    }
    const float pscale =
        fmaxf(peak / (HEADS == 32 ? 32512.f : 16129.f), 1.e-30f);
    const float next_den = denominator * exp2f(maximum - safe) + page_sum;
    const float inverse_den = reciprocal(next_den > 0.f ? next_den : 1.f);
    const float old_weight = denominator * exp2f(maximum - safe) * inverse_den,
                outscale = pscale * inverse_den;
    const float inv = reciprocal(pscale);
    if (HEADS == 32)
      sync();
#pragma unroll
    for (int word = 0; word < NR; ++word) {
      unsigned integer[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int index = word * 4 + j;
        const float p = probability[index % NR][index / NR];
        integer[j] = HEADS == 32 ? __float_as_uint(p * inv + 8388736.f)
                                 : (unsigned)(int)(p * inv + 0.5f);
      }
      const unsigned p01 = perm(integer[1], integer[0], 0x05040100);
      const unsigned p23 = perm(integer[3], integer[2], 0x05040100);
      unsigned high, low;
      if (HEADS == 32) {
        high = perm(p23, p01, 0x07050301);
        low = perm(p23, p01, 0x06040200) ^ 0x80808080u;
      } else {
        high = perm((p23 + 0x00400040u) >> 7, (p01 + 0x00400040u) >> 7,
                    0x06040200);
        low = perm(p23, p01, 0x06040200) ^ ((high & 0x01010101u) << 7);
      }
      const int token = wn * (64 / NW) + group * (NR * 4) + word * 4;
      const int offset =
          (wm * 16 + row) * 64 + (token ^ ((((wm * 16 + row) >> 1) & 3) * 16));
      *(unsigned *)(planes[0] + offset) = high;
      *(unsigned *)(planes[0] + HEADS * 64 + offset) = low;
    }
    sync();
    const int pbase = (wm * 16 + row) * 64 +
                      ((group * 16) ^ ((((wm * 16 + row) >> 1) & 3) * 16));
    const uint4_ ph = *(const uint4_ *)(planes[0] + pbase),
                 pl = *(const uint4_ *)(planes[0] + HEADS * 64 + pbase);
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      unsigned values[4][QV];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        if (HEADS == 16 || chunk == 0) {
#pragma unroll
          for (int m = 0; m < QV; ++m)
            values[k][m] = retained[HEADS == 16 ? chunk : 0][k][m];
        } else
          fragments(planes[chunk], group * 16 + k * 4,
                    wn * (128 / NW) + row * QV, values[k]);
      }
#pragma unroll
      for (int m = 0; m < QV; ++m) {
        int4_ high = {0, 0, 0, 0};
#pragma unroll
        for (int k = 0; k < 4; ++k)
          high = mma(values[k][m], ph[k], high);
        int4_ combined = high * (HEADS == 32 ? 256 : 128);
#pragma unroll
        for (int k = 0; k < 4; ++k)
          combined = mma(values[k][m], pl[k], combined);
        float4_ converted;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          converted[j] = (float)combined[j];
        acc[chunk * QV + m] =
            acc[chunk * QV + m] * old_weight + converted * outscale;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          unsigned bits = __float_as_uint(acc[chunk * QV + m][j]);
          asm volatile("" : "+r"(bits));
          acc[chunk * QV + m][j] = __uint_as_float(bits);
        }
      }
    }
    maximum = next_max;
    denominator = next_den;
    sync();
  }
  const long long outrow = ((long long)batch * allheads + head) * 512;
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    const int dim = chunk * 128 + wn * (128 / NW) + group * (4 * QV);
    if (splits == 1 && bf16) {
#pragma unroll
      for (int v = 0; v < QV / 2; ++v) {
        uint4_ packed;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const int i = v * 8 + w * 2;
          packed[w] = pack_bf16(acc[chunk * QV + i % QV][i / QV],
                                acc[chunk * QV + (i + 1) % QV][(i + 1) / QV]);
        }
        *output<uint4_>(op + (outrow + dim + v * 8) * 2) = packed;
      }
    } else if (splits == 1) {
#pragma unroll
      for (int i = 0; i < 4 * QV; ++i)
        output<_Float16>(op)[outrow + dim + i] =
            (_Float16)acc[chunk * QV + i % QV][i / QV];
    } else {
#pragma unroll
      for (int v = 0; v < QV; ++v) {
        float4_ packed;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int i = v * 4 + j;
          packed[j] = acc[chunk * QV + i % QV][i / QV];
        }
        const long long index =
            (long long)split * batches * allheads * 512 + outrow + dim + v * 4;
        *output<float4_>(pp + index * 4) = packed;
      }
    }
  }
  if (group == 0 && wn == 0) {
    const long long r = (long long)batch * allheads + head;
    if (splits == 1)
      output<float>(
          lsep)[((long long)batch * hq + head % hq) * sq + head / hq] =
          denominator > 0.f ? maximum * 0.6931471805599453f + logf(denominator)
                            : INFINITY;
    else {
      output<float>(sp)[(long long)(split * 2) * batches * allheads + r] =
          maximum;
      output<float>(sp)[(long long)(split * 2 + 1) * batches * allheads + r] =
          denominator;
    }
  }
  return 0.f;
}

#elif defined(FMLA_STAGE_SCORES)

namespace {
__device__ __forceinline__ float max4(float4_ x) {
  return fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3]));
}
__device__ __forceinline__ float sum4(float4_ x) {
  return (x[0] + x[1]) + (x[2] + x[3]);
}

__device__ __forceinline__ int koff(int token, int dim) {
  return token * 256 + (dim ^ (((token >> 2) & 7) * 16));
}
__device__ __forceinline__ int roff(int token, int dim) {
  return token * 64 + (dim ^ (((token >> 2) & 3) * 16));
}

} // namespace
extern "C" __device__ __attribute__((used, always_inline)) float
mla_int8_scores(long long qp, long long qrp, long long kp, long long krp,
                long long qsp, long long ksp, long long tp, long long lp,
                long long probability_ptr, long long page_stats_ptr,
                long long split_stats_ptr, long long orderp, long long ts,
                long long ls, int hq, int sq, int capacity, int splits,
                int causal, int bf16, float scale_log2, int split,
                int head_tile, int batch) {
  qp = uniform(qp);
  qrp = uniform(qrp);
  kp = uniform(kp);
  krp = uniform(krp);
  qsp = uniform(qsp);
  ksp = uniform(ksp);
  tp = uniform(tp);
  lp = uniform(lp);
  probability_ptr = uniform(probability_ptr);
  page_stats_ptr = uniform(page_stats_ptr);
  split_stats_ptr = uniform(split_stats_ptr);
  orderp = uniform(orderp);
  ts = uniform(ts);
  ls = uniform(ls);
  hq = uniform(hq);
  sq = uniform(sq);
  capacity = uniform(capacity);
  splits = uniform(splits);
  causal = uniform(causal);
  bf16 = uniform(bf16);
  split = uniform(split);
  head_tile = uniform(head_tile);
  batch = uniform(batch);
  hq = FMLA_HQ;
  sq = FMLA_SQ;
  capacity = FMLA_CAP;
  splits = FMLA_SPLITS;
  causal = FMLA_CAUSAL;
  bf16 = FMLA_BF16;
  ts = FMLA_CAP / 64;
  ls = 1;
  scale_log2 = FMLA_SCALE;
  if (orderp)
    batch = uniform(input<int>(orderp)[batch]);
  const int t = (threadIdx.x & 255), lane = t & 63, wave = t >> 6,
            row = lane & 15, group = lane >> 4;
  const int head = head_tile * 64 + wave * 16 + row, heads = hq * sq;
  const int length = uniform(input<int>(lp)[batch * ls]),
            pages = (length + 63) / 64;
  const int begin = pages * split / splits, end = pages * (split + 1) / splits;
  const int bound = causal ? max(length - sq + head / hq + 1, 0) : length;
  const long long qrow = (long long)batch * heads + head;
  const float qs = input<float>(qsp)[qrow];
  // A lane owns one complete 16-token quantization group.
  unsigned query[32];
  uint2_ query_rope[4];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const uint4_ words = *input<uint4_>(qp + qrow * 512 + k * 64 + group * 16);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      query[4 * k + j] = words[j];
  }
#pragma unroll
  for (int k = 0; k < (FMLA_NO_ROPE ? 0 : 2); ++k) {
    const uint4_ words =
        *input<uint4_>(qrp + (qrow * 64 + group * 16 + k * 8) * 2);
    query_rope[2 * k] = (uint2_){words[0], words[1]};
    query_rope[2 * k + 1] = (uint2_){words[2], words[3]};
  }
  __shared__ unsigned char key[16384];
  __shared__ unsigned short rope[4096];
  float maximum = -INFINITY, denominator = 0.f, pscale = 0.f;
  for (int page_index = begin; page_index < end; ++page_index) {
    const long long page = cache_page(tp, batch, ts, page_index);
    int4_ content[4];
#pragma unroll
    for (int n = 0; n < 4; ++n)
      content[n] = (int4_){0, 0, 0, 0};
#pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int i = t + c * 256, token = i >> 4,
                  dim = ((i & 15) ^ ((token >> 2) & 7)) * 16;
        void *dest = key + ((t & ~63) + c * 256) * 16;
        void *src = (void *)input<unsigned char>(
            kp + cache_token(tp, batch, page, token) * 512 + chunk * 256 + dim);
        __builtin_mxc_ldg_b128_bsm_predicator(dest, src, 0, true, true, false,
                                              true, 1, 2, 40);
      }
      if (!FMLA_NO_ROPE && chunk == 1) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const int i = t + c * 256, token = i >> 3,
                    dim = ((i & 7) ^ (((token >> 2) & 3) * 2)) * 8;
          void *dest = (unsigned char *)rope + ((t & ~63) + c * 256) * 16;
          void *src = (void *)input<unsigned char>(
              krp + (cache_token(tp, batch, page, token) * 64 + dim) * 2);
          __builtin_mxc_ldg_b128_bsm_predicator(dest, src, 0, true, true, false,
                                                true, page_index * 64 + token,
                                                length, 40);
        }
      }
      __builtin_mxc_arrive_gvmcnt(0);
      __builtin_mxc_barrier_ex(2);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const uint4_ words =
              *(const uint4_ *)(key + koff(row * 4 + n, k * 64 + group * 16));
#pragma unroll
          for (int j = 0; j < 4; ++j)
            content[n] = __builtin_mxc_mma_16x16x16i8(
                (int)words[j], (int)query[chunk * 16 + k * 4 + j], content[n]);
        }
      }
      sync();
    }
    float4_ scores[4], weighted[4], scales[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#if defined(FMLA_SPARSE)
#pragma unroll
      for (int n = 0; n < 4; ++n)
        scales[j][n] = input<float>(ksp)[
            cache_token(tp, batch, page, group * 16 + j * 4 + n)];
#else
      scales[j] = *input<float4_>(ksp + (page * 64 + group * 16 + j * 4) * 4);
#endif
    }
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      float4_ r = {0, 0, 0, 0};
#pragma unroll
      for (int k = 0; k < (FMLA_NO_ROPE ? 0 : 2); ++k) {
        const uint4_ words =
            *(const uint4_ *)(rope + roff(row * 4 + n, group * 16 + k * 8));
        r = rope_mma((uint2_){words[0], words[1]}, query_rope[2 * k], r, bf16);
        r = rope_mma((uint2_){words[2], words[3]}, query_rope[2 * k + 1], r,
                     bf16);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int token = page_index * 64 + group * 16 + j * 4 + n;
        if (token >= length)
          scales[j][n] = 0.f;
        scores[n][j] = token < bound && cache_valid(tp, batch, page, token - page_index * 64) ? ((float)content[n][j] + r[j]) * qs *
                                           scales[j][n] * scale_log2
                                     : -INFINITY;
      }
    }
    const float local_max = fmaxf(fmaxf(max4(scores[0]), max4(scores[1])),
                                  fmaxf(max4(scores[2]), max4(scores[3])));
    const float safe = local_max == -INFINITY ? 0.f : local_max;
    float4_ sums, peaks;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      float4_ p;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        p[j] = exp2f(scores[n][j] - safe);
        weighted[n][j] = p[j] * scales[j][n];
      }
      sums[n] = sum4(p);
      peaks[n] = max4(weighted[n]);
    }
    const float local_sum = sum4(sums), peak = max4(peaks);
    const float local_scale = fmaxf(peak * (1.f / 32512.f), 1.e-30f),
                inv = reciprocal(local_scale);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      uint4_ packed;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const int i = half * 8 + w * 2;
        const unsigned lo =
            __float_as_uint(weighted[i % 4][i / 4] * inv + 8388608.f);
        const unsigned hi = __float_as_uint(
            weighted[(i + 1) % 4][(i + 1) / 4] * inv + 8388608.f);
        packed[w] = __builtin_mxc_byte_perm(hi, lo, 0x05040100);
      }
      const long long dest =
          ((long long)batch * capacity + page_index * 64) * heads + head * 64 +
          group * 16 + half * 8;
      *output<uint4_>(probability_ptr + dest * 2) = packed;
    }
    const long long offset =
        (((long long)batch * (capacity / 64) * 2 + page_index * 2) * heads +
         head) *
            4 +
        group;
    output<float>(page_stats_ptr)[offset] = local_max;
    output<float>(page_stats_ptr)[offset + heads * 4] = local_scale;
    const float raised = fmaxf(maximum, local_max),
                safe_max = raised == -INFINITY ? 0.f : raised;
    const float alpha = exp2f(maximum - safe_max),
                beta = exp2f(local_max - safe_max);
    denominator = denominator * alpha + local_sum * beta;
    pscale = fmaxf(pscale * alpha, local_scale * beta);
    maximum = raised;
    sync();
  }
  const long long offset =
      (((long long)batch * splits * 3 + split * 3) * 4 + group) * heads + head;
  output<float>(split_stats_ptr)[offset] = maximum;
  output<float>(split_stats_ptr)[offset + heads * 4] = pscale;
  output<float>(split_stats_ptr)[offset + heads * 8] = denominator;
  return 0.f;
}

#elif defined(FMLA_STAGE_VALUES)

// Coarse device-function organization follows FlagGems-vllm PR #972.
#ifndef VDIM
#define VDIM 512
#endif
constexpr int THREADS = VDIM;
constexpr int PCOPIES = 512 / THREADS;

namespace {

__device__ __forceinline__ int voffset(int token, int dim) {
  return token * VDIM + (dim ^ (((token >> 4) & 1) * 64));
}
__device__ __forceinline__ int poffset(int head, int token) {
  return head * 64 + (token ^ (((head >> 1) & 3) * 16));
}
__device__ __forceinline__ unsigned quantize(unsigned value, float factor) {
  const float biased = __uint_as_float(value | 0x4b000000u);
  const float normalized = fmaf(biased, factor, -8388608.f * factor);
  return __float_as_uint(normalized + 8388736.f);
}

} // namespace

extern "C" __device__ __attribute__((used, always_inline)) float
mla_int8_pv(long long kv_ptr, long long table_ptr, long long lengths_ptr,
            long long scores_ptr, long long cache_metadata_ptr,
            long long metadata_ptr, long long scale_ptr, long long output_ptr,
            long long partial_ptr, long long flag_ptr, long long kv_page_stride,
            long long kv_token_stride, long long table_stride,
            long long length_stride, int batches, int heads, int capacity,
            int splits, int adaptive, int sort_batches, int uniform_splits,
            int output_bf16, int value_tile, int head_tile, int job) {
  kv_ptr = uniform(kv_ptr);
  table_ptr = uniform(table_ptr);
  lengths_ptr = uniform(lengths_ptr);
  scores_ptr = uniform(scores_ptr);
  cache_metadata_ptr = uniform(cache_metadata_ptr);
  metadata_ptr = uniform(metadata_ptr);
  scale_ptr = uniform(scale_ptr);
  output_ptr = uniform(output_ptr);
  partial_ptr = uniform(partial_ptr);
  flag_ptr = uniform(flag_ptr);
  kv_page_stride = uniform(kv_page_stride);
  kv_token_stride = uniform(kv_token_stride);
  table_stride = uniform(table_stride);
  length_stride = uniform(length_stride);
  batches = uniform(batches);
  heads = uniform(heads);
  capacity = uniform(capacity);
  splits = uniform(splits);
  adaptive = uniform(adaptive);
  sort_batches = uniform(sort_batches);
  uniform_splits = uniform(uniform_splits);
  output_bf16 = uniform(output_bf16);
  value_tile = uniform(value_tile);
  head_tile = uniform(head_tile);
  job = uniform(job);
  batches = FMLA_B;
  heads = FMLA_HEADS;
  capacity = FMLA_CAP;
  splits = FMLA_SPLITS;
  adaptive = FMLA_ADAPTIVE;
  sort_batches = FMLA_SORT;
  uniform_splits = FMLA_UNIFORM;
  output_bf16 = FMLA_BF16;
  kv_page_stride = 32768;
  kv_token_stride = 512;
  table_stride = FMLA_CAP / 64;
  length_stride = 1;
  int batch = adaptive ? job % batches : job;
  const int split = adaptive ? job / batches : 0;
  const auto flags = input<int>(flag_ptr);
  const int split_count =
      adaptive ? (flags[0] != 0 ? splits : uniform_splits) : 1;
  if (split >= split_count)
    return 0.f;
  if (sort_batches)
    batch = uniform(flags[1 + batch]);
  const int length = uniform(input<int>(lengths_ptr)[batch * length_stride]);
  const int pages = (length + 63) / 64, begin = pages * split / split_count,
            end = pages * (split + 1) / split_count;
  const int t = threadIdx.x & (THREADS - 1), lane = t & 63, warp = t >> 6,
            row = lane & 15, group = lane >> 4;
  __shared__ unsigned char values[64 * VDIM];
  __shared__ unsigned char probability[2][2][64 * 64];
  int4_ high[4][4], low[4][4];
#pragma unroll
  for (int m = 0; m < 4; ++m) {
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      high[m][n] = (int4_){0, 0, 0, 0};
      low[m][n] = (int4_){0, 0, 0, 0};
    }
  }
  float maximum[PCOPIES], inverse[PCOPIES];
#pragma unroll
  for (int c = 0; c < PCOPIES; ++c) {
    const int worker = t + c * THREADS, h = worker >> 3,
              head = head_tile * 64 + h;
    const long long offset = ((long long)batch * heads + head) * 2;
    maximum[c] = head < heads ? input<float>(metadata_ptr)[offset] : 0.f;
    inverse[c] = head < heads ? input<float>(metadata_ptr)[offset + 1] : 0.f;
  }
  struct ProbabilityData {
    uint4_ score[PCOPIES];
    float factor[PCOPIES];
  };
  auto load_probability = [&](int page_index) {
    ProbabilityData data;
#pragma unroll
    for (int c = 0; c < PCOPIES; ++c) {
      const int worker = t + c * THREADS, h = worker >> 3,
                tokens = (worker & 7) * 8, head = head_tile * 64 + h;
      if (head < heads) {
        const long long offset =
            ((long long)batch * capacity + page_index * 64) * heads +
            head * 64 + tokens;
        data.score[c] = *input<uint4_>(scores_ptr + offset * 2);
        const long long meta =
            (((long long)batch * (capacity / 64) * 2 + page_index * 2) * heads +
             head) *
                4 +
            tokens / 16;
        const float local_max = input<float>(cache_metadata_ptr)[meta];
        const float local_scale =
            input<float>(cache_metadata_ptr)[meta + heads * 4];
        data.factor[c] =
            local_scale * exp2f(local_max - maximum[c]) * inverse[c];
      } else {
        data.score[c] = (uint4_){0, 0, 0, 0};
        data.factor[c] = 0.f;
      }
    }
    return data;
  };
  auto store_probability = [&](const ProbabilityData &data, int slot) {
#pragma unroll
    for (int c = 0; c < PCOPIES; ++c) {
      const int worker = t + c * THREADS, h = worker >> 3,
                tokens = (worker & 7) * 8;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const unsigned a = data.score[c][2 * i], b = data.score[c][2 * i + 1];
        const unsigned x0 = quantize(a & 65535u, data.factor[c]),
                       x1 = quantize(a >> 16, data.factor[c]);
        const unsigned x2 = quantize(b & 65535u, data.factor[c]),
                       x3 = quantize(b >> 16, data.factor[c]);
        const unsigned p01 = perm(x1, x0, 0x05040100),
                       p23 = perm(x3, x2, 0x05040100);
        const int dest = poffset(h, tokens + i * 4);
        *(unsigned *)(probability[slot][0] + dest) = perm(p23, p01, 0x07050301);
        *(unsigned *)(probability[slot][1] + dest) =
            perm(p23, p01, 0x06040200) ^ 0x80808080u;
      }
    }
  };
  auto issue_values = [&](int page_index) {
    const long long page =
        cache_page(table_ptr, batch, table_stride, page_index);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int i = t + c * THREADS, token = i / (VDIM / 16),
                dim = ((i % (VDIM / 16)) ^ (((token >> 4) & 1) * 4)) * 16;
      void *dest = values + ((t & ~63) + c * THREADS) * 16;
      void *src = (void *)input<unsigned char>(kv_ptr + cache_token(table_ptr, batch, page, token) *
                                               kv_token_stride +
                                               value_tile * VDIM + dim);
      __builtin_mxc_ldg_b128_bsm_predicator(dest, src, 0, true, true, false,
                                            true, 1, 2, 40);
    }
  };
  int slot = 0;
  if (begin < end) {
    issue_values(begin);
    const auto first = load_probability(begin);
    store_probability(first, 0);
    __builtin_mxc_arrive_gvmcnt(0);
    sync();
  }
  for (int page_index = begin; page_index < end; ++page_index) {
    const bool has_next = page_index + 1 < end;
    ProbabilityData next;
    if (has_next)
      next = load_probability(page_index + 1);
    unsigned fragment[4][4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int token = group * 16 + k * 4, col = warp * 64 + row * 4;
      const unsigned a = *(const unsigned *)(values + voffset(token, col));
      const unsigned b = *(const unsigned *)(values + voffset(token + 1, col));
      const unsigned c = *(const unsigned *)(values + voffset(token + 2, col));
      const unsigned d = *(const unsigned *)(values + voffset(token + 3, col));
      const unsigned p01 = perm(b, a, 0x05040100), p23 = perm(d, c, 0x05040100);
      const unsigned q01 = perm(b, a, 0x07060302), q23 = perm(d, c, 0x07060302);
      fragment[k][0] = perm(p23, p01, 0x06040200);
      fragment[k][1] = perm(p23, p01, 0x07050301);
      fragment[k][2] = perm(q23, q01, 0x06040200);
      fragment[k][3] = perm(q23, q01, 0x07050301);
    }
    sync();
    if (has_next) {
      store_probability(next, slot ^ 1);
      issue_values(page_index + 1);
    }
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int p = poffset(n * 16 + row, group * 16);
      const uint4_ ph = *(const uint4_ *)(probability[slot][0] + p),
                   pl = *(const uint4_ *)(probability[slot][1] + p);
#pragma unroll
      for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          high[m][n] = mma(fragment[k][m], ph[k], high[m][n]);
          low[m][n] = mma(fragment[k][m], pl[k], low[m][n]);
        }
      }
    }
    if (has_next) {
      __builtin_mxc_arrive_gvmcnt(0);
      sync();
      slot ^= 1;
    }
  }
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    const int head = head_tile * 64 + n * 16 + row;
    if (head < heads) {
      const float scale =
          split_count == 1
              ? input<float>(scale_ptr)[(long long)batch * heads + head]
              : 1.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float4_ raw;
#pragma unroll
        for (int m = 0; m < 4; ++m)
          raw[m] = (float)high[m][n][j] * 256.f + (float)low[m][n][j];
        const int dim = value_tile * VDIM + warp * 64 + group * 16 + j * 4;
        const long long index = ((long long)batch * heads + head) * 512 + dim;
        if (split_count == 1) {
          const float4_ result = raw * scale;
          if (output_bf16) {
            const uint2_ packed = {pack_bf16(result[0], result[1]),
                                   pack_bf16(result[2], result[3])};
            *output<uint2_>(output_ptr + index * 2) = packed;
          } else {
#pragma unroll
            for (int m = 0; m < 4; ++m)
              output<_Float16>(output_ptr)[index + m] = (_Float16)result[m];
          }
        } else {
          const long long partial_index =
              (long long)split * batches * heads * 512 + index;
          *output<float4_>(partial_ptr + partial_index * 4) = raw;
        }
      }
    }
  }
  return 0.f;
}

#else
#error "Select FMLA_STAGE_SMALL, FMLA_STAGE_SCORES or FMLA_STAGE_VALUES"
#endif
