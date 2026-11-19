// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <type_traits>
#include <hggc_runtime.h>

extern "C" __device__ int megamhc_wait(uint64_t address, uint64_t expected) {
  if (threadIdx.x == 0) {
    uint64_t observed;
    do {
      asm volatile("ppu.ld.global.gpu.acquire.b64 %0, [ %1 + 0 ];"
                   : "=l"(observed) : "l"(address) : "memory");
    } while (observed < expected);
  }
  __syncthreads();
  return 0;
}

namespace {
constexpr unsigned kColumns = 256;
constexpr unsigned kLanes = 32;
constexpr unsigned kValues = kColumns / kLanes;
constexpr unsigned kRows = 2;

__device__ __forceinline__ void copy_tile(unsigned addr, uint64_t src) {
  asm volatile("cp.async.cg.shared.global [ %0 + 0 ], [ %1 + 0 ], 0x10, 0x10;"
               : : "r"(addr), "l"(src) : "memory");
}

struct Bf16x8 {
  uint32_t word[4];
};

__device__ __forceinline__ Bf16x8 read_tile(unsigned addr) {
  Bf16x8 v;
  asm volatile("ld.shared.v4.b32 { %0, %1, %2, %3 }, [ %4 + 0 ];"
               : "=r"(v.word[0]), "=r"(v.word[1]), "=r"(v.word[2]), "=r"(v.word[3])
               : "r"(addr) : "memory");
  return v;
}

__device__ __forceinline__ void write_tile(uint64_t addr, const Bf16x8 &v) {
  asm volatile("ppu.st.global.wt.v4.b32 [ %0 + 0 ], { %1, %2, %3, %4 };"
               : : "l"(addr), "r"(v.word[0]), "r"(v.word[1]), "r"(v.word[2]), "r"(v.word[3])
               : "memory");
}

__device__ __forceinline__ float as_float(uint32_t bits, unsigned half) {
  return __uint_as_float((bits >> (16 * half)) << 16);
}

__device__ __forceinline__ uint32_t to_bf16(float lo, float hi) {
  unsigned packed;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(hi), "f"(lo));
  return packed;
}

template <unsigned R>
__device__ __forceinline__ int
post(uint64_t x, uint64_t residual, uint64_t post_weights,
     uint64_t comb_weights, uint64_t output, uint64_t squares,
     uint64_t row_start, uint64_t comb_t, uint64_t comb_s, uint64_t comb_d,
     unsigned shared_slot, unsigned hidden, unsigned rows) {
  constexpr unsigned slot_plane = kRows * kColumns * sizeof(uint16_t);
  constexpr unsigned slot_size = slot_plane * (R + 1);
  const unsigned local_row = threadIdx.x / kLanes;
  const unsigned lane = threadIdx.x % kLanes;
  const uint64_t row = row_start + local_row;
  const unsigned smem_lane = shared_slot + threadIdx.x * 16;
  const uint64_t x_offset = (row * hidden + kValues * lane) * sizeof(uint16_t);
  const uint64_t residual_offset =
      (row * R * hidden + kValues * lane) * sizeof(uint16_t);
  const unsigned tiles_per_pair = hidden / kColumns;
  const unsigned tiles = (rows / kRows) * tiles_per_pair;
  float post[R];
  float comb[R][R];

  // Keep the two shared slots in flight across row pairs as well as H tiles.
  auto prefetch = [&](unsigned tile,
                      unsigned slot) __attribute__((always_inline)) {
    const unsigned pair = tile / tiles_per_pair;
    const unsigned column = tile % tiles_per_pair * kColumns;
    const unsigned base = smem_lane + slot * slot_size;
    copy_tile(base, x + x_offset + 2 * (pair * kRows * hidden + column));
#pragma unroll
    for (unsigned src = 0; src < R; ++src) {
      copy_tile(base + (src + 1) * slot_plane,
                residual + residual_offset +
                    2 * ((pair * kRows * R + src) * hidden + column));
    }
    asm volatile("cp.async.commit_group;" : : : "memory");
  };

  float sum[kValues] = {};
  unsigned current = 0;
  prefetch(0, current);
  if (tiles > 1)
    prefetch(1, 1);
  for (unsigned tile = 0; tile < tiles; ++tile) {
    const unsigned pair = tile / tiles_per_pair;
    const unsigned column = tile % tiles_per_pair * kColumns;
    if (column == 0) {
#pragma unroll
      for (unsigned dst = 0; dst < R; ++dst) {
        post[dst] = reinterpret_cast<const float *>(
            post_weights)[(row + pair * kRows) * R + dst];
#pragma unroll
        for (unsigned src = 0; src < R; ++src) {
          comb[src][dst] = reinterpret_cast<const float *>(
              comb_weights)[(row + pair * kRows) * comb_t + src * comb_s +
                            dst * comb_d];
        }
      }
    }
    // Only the final tile drains the queue; a row boundary keeps it active.
    if (tiles - tile > 1)
      asm volatile("cp.async.wait_group 1;" : : : "memory");
    else
      asm volatile("cp.async.wait_group 0;" : : : "memory");
    const unsigned base = smem_lane + current * slot_size;
    const Bf16x8 xv = read_tile(base);
    Bf16x8 residuals[R];
#pragma unroll
    for (unsigned src = 0; src < R; ++src) {
      residuals[src] = read_tile(base + (src + 1) * slot_plane);
    }
    // Each lane has read its own slot into registers. Refill it two tiles
    // ahead.
    if (tiles - tile > 2)
      prefetch(tile + 2, current);
#pragma unroll
    for (unsigned dst = 0; dst < R; ++dst) {
      Bf16x8 rounded;
#pragma unroll
      for (unsigned word = 0; word < 4; ++word) {
        float lo = post[dst] * as_float(xv.word[word], 0);
        float hi = post[dst] * as_float(xv.word[word], 1);
#pragma unroll
        for (unsigned src = 0; src < R; ++src) {
          lo = __fmaf_rn(comb[src][dst], as_float(residuals[src].word[word], 0),
                         lo);
          hi = __fmaf_rn(comb[src][dst], as_float(residuals[src].word[word], 1),
                         hi);
        }
        rounded.word[word] = to_bf16(lo, hi);
        const float rounded_lo = as_float(rounded.word[word], 0);
        const float rounded_hi = as_float(rounded.word[word], 1);
        sum[2 * word] = __fmaf_rn(rounded_lo, rounded_lo, sum[2 * word]);
        sum[2 * word + 1] =
            __fmaf_rn(rounded_hi, rounded_hi, sum[2 * word + 1]);
      }
      write_tile(output + residual_offset +
                     2 * ((pair * kRows * R + dst) * hidden + column),
                 rounded);
    }
    current ^= 1;
    if (column + kColumns == hidden) {
#pragma unroll
      for (unsigned distance = kValues / 2; distance > 0; distance /= 2) {
#pragma unroll
        for (unsigned i = 0; i < distance; ++i) {
          sum[i] += sum[i + distance];
        }
      }
#pragma unroll
      for (unsigned mask = kLanes / 2; mask > 0; mask /= 2) {
        sum[0] += __shfl_xor_sync(0xffffffff, sum[0], mask);
      }
      if (lane == 0)
        reinterpret_cast<float *>(squares)[row + pair * kRows] = sum[0];
#pragma unroll
      for (unsigned i = 0; i < kValues; ++i)
        sum[i] = 0.f;
    }
  }
  return 0;
}
} // namespace

#define DEFINE_MHC_POST(R)                                                     \
  extern "C" __device__ __attribute__((always_inline)) int megamhc_post_##R(   \
      uint64_t x, uint64_t residual, uint64_t weights, uint64_t comb,          \
      uint64_t output, uint64_t squares, uint64_t row_start, uint64_t comb_t,  \
      uint64_t comb_s, uint64_t comb_d, unsigned smem, unsigned hidden,        \
      unsigned rows) {                                                         \
    return post<R>(x, residual, weights, comb, output, squares, row_start,     \
                   comb_t, comb_s, comb_d, smem, hidden, rows);                \
  }

DEFINE_MHC_POST(2)
DEFINE_MHC_POST(4)

// Multi-row SIMD projection reuses the FP32 weights without MMA mode.
namespace megamhc_simt {
// Eight packed vectors cap the normalization cache at 32 registers per lane.
constexpr unsigned kCachedVectors = 8;
// Wider head groups use the CTA's vector loads for exceptional dot products.
template <unsigned Warps, unsigned R, unsigned Rows>
constexpr bool kCtaRetry = Rows >= 4 || (Rows >= 2 && R * (R + 2) > 2 * Warps);
// Vector transactions move eight BF16 values (16 bytes). Each token reserves
// one warp-sized projection row: the supported 1/2/4-stream mixes fit below
// lane 32, leaving a lane for the row norm used by compensated retries.
__device__ __forceinline__ float4 load_float4(uint64_t address) {
  float4 v;
  asm volatile("ppu.ld.global.v4.f32 { %0, %1, %2, %3 }, [ %4 + 0 ];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(address)
               : "memory");
  return v;
}
struct PackedBf16 {
  uint32_t values[4];
};

__device__ __forceinline__ float bf16_at(const PackedBf16 &data, unsigned i) {
  return __uint_as_float((data.values[i / 2] >> ((i % 2) * 16)) << 16);
}
__device__ __forceinline__ PackedBf16 load_bf16(uint64_t address) {
  PackedBf16 v;
  asm volatile("ppu.ld.global.v4.b32 { %0, %1, %2, %3 }, [ %4 + 0 ];"
               : "=r"(v.values[0]), "=r"(v.values[1]), "=r"(v.values[2]),
                 "=r"(v.values[3])
               : "l"(address)
               : "memory");
  return v;
}
__device__ __forceinline__ void store_bf16(uint64_t address,
                                           const PackedBf16 &v) {
  asm volatile("ppu.st.global.v4.b32 [ %0 + 0 ], { %1, %2, %3, %4 };"
               :
               : "l"(address), "r"(v.values[0]), "r"(v.values[1]),
                 "r"(v.values[2]), "r"(v.values[3])
               : "memory");
}
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (unsigned mask = 16; mask > 0; mask /= 2) {
    value += __shfl_xor_sync(0xffffffff, value, mask);
  }
  return value;
}

__device__ __forceinline__ void shared_store(unsigned base, unsigned index,
                                             float value) {
  asm volatile("st.shared.b32 [ %0 + 0 ], %1;"
               :
               : "r"(base + index * 4), "f"(value)
               : "memory");
}
__device__ __forceinline__ float shared_load(unsigned base, unsigned index) {
  float value;
  asm volatile("ld.shared.b32 %0, [ %1 + 0 ];"
               : "=f"(value)
               : "r"(base + index * 4)
               : "memory");
  return value;
}

template <unsigned Warps, unsigned R, unsigned Rows>
__device__ __forceinline__ int
project(uint64_t xp, uint64_t rp, uint64_t postp, uint64_t combp, uint64_t fnp,
        uint64_t outp, uint64_t token, uint64_t cs0, uint64_t cs1, uint64_t cs2,
        unsigned hidden, unsigned shared, float eps) {
  // The host bounds dense byte offsets before selecting the SIMD path.
  using Index = typename std::conditional<(Warps >= R * (R + 2)), unsigned,
                                          uint64_t>::type;
  constexpr unsigned P = R * (R + 2);
  constexpr unsigned Heads = (P + Warps - 1) / Warps;
  const unsigned lane = threadIdx.x & 31;
  const unsigned warp = threadIdx.x / 32;
  constexpr unsigned RowWarps = Warps / Rows;
  const unsigned row = warp / RowWarps;
  const unsigned inrow = warp % RowWarps;

  const Index row_hidden = (Index(token) + row) * hidden;
  const Index row_residual = (Index(token) + row) * R * hidden;
  float sq[8] = {};
  {
    float post[R], comb[R][R];
#pragma unroll
    for (unsigned dst = 0; dst < R; ++dst) {
      post[dst] = reinterpret_cast<const float *>(
          postp)[(Index(token) + row) * R + dst];
#pragma unroll
      for (unsigned src = 0; src < R; ++src) {
        comb[dst][src] = reinterpret_cast<const float *>(
            combp)[(Index(token) + row) * cs0 + src * cs1 + dst * cs2];
      }
    }
    for (unsigned start = (inrow * 32 + lane) * 8; start < hidden;
         start += RowWarps * 32 * 8) {
      PackedBf16 x = load_bf16(xp + 2 * (row_hidden + start));
      PackedBf16 sources[R];
#pragma unroll
      for (unsigned src = 0; src < R; ++src) {
        sources[src] =
            load_bf16(rp + 2 * (row_residual + src * hidden + start));
      }
#pragma unroll
      for (unsigned dst = 0; dst < R; ++dst) {
        PackedBf16 rounded;
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
          float lo = 0.0f, hi = 0.0f;
#pragma unroll
          for (unsigned src = 0; src < R; ++src) {
            lo = __fmaf_rn(comb[dst][src], bf16_at(sources[src], word * 2), lo);
            hi = __fmaf_rn(comb[dst][src], bf16_at(sources[src], word * 2 + 1),
                           hi);
          }
          lo += post[dst] * bf16_at(x, word * 2);
          hi += post[dst] * bf16_at(x, word * 2 + 1);
          asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
              : "=r"(rounded.values[word])
              : "f"(hi), "f"(lo));
          lo = bf16_at(rounded, word * 2);
          hi = bf16_at(rounded, word * 2 + 1);
          sq[2 * word] = __fmaf_rn(lo, lo, sq[2 * word]);
          sq[2 * word + 1] = __fmaf_rn(hi, hi, sq[2 * word + 1]);
        }
        store_bf16(outp + 2 * (row_residual + dst * hidden + start), rounded);
      }
    }
  }
#pragma unroll
  for (unsigned width = 4; width > 0; width /= 2) {
#pragma unroll
    for (unsigned j = 0; j < width; ++j)
      sq[j] += sq[j + width];
  }
  float sum = warp_sum(sq[0]);
  if (lane == 0)
    shared_store(shared, Rows * 32 + warp, sum);
  __syncthreads();
  if (inrow == 0) {
    float total = lane < RowWarps
                      ? shared_load(shared, Rows * 32 + row * RowWarps + lane)
                      : 0.f;
    total = warp_sum(total);
    if (lane == 0)
      shared_store(shared, row * 32 + P, rsqrtf(total / (R * hidden) + eps));
  }
  __syncthreads();
  float rstd[Rows];
#pragma unroll
  for (unsigned rr = 0; rr < Rows; ++rr) {
    rstd[rr] = shared_load(shared, rr * 32 + P);
  }

  float acc[Rows][Heads][8] = {};
  // Uneven multi-head groups split once before K. Keep the established
  // schedule for the full and one-head groups.
  if constexpr (Heads > 1 && P % Warps != 0) {
    auto accumulate_heads = [&](auto count) {
      constexpr unsigned ActiveHeads = decltype(count)::value;
      constexpr unsigned Unroll = ActiveHeads == 1 ? 2 : 1;
#pragma unroll Unroll
      for (unsigned start = lane * 8; start < R * hidden; start += 32 * 8) {
        PackedBf16 av[Rows];
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row)
          av[row] =
              load_bf16(outp + 2 * ((Index(token) + row) * R * hidden + start));
#pragma unroll
        for (unsigned j = 0; j < ActiveHeads; ++j) {
          const unsigned head = warp + j * Warps;
          const float *weights =
              reinterpret_cast<const float *>(fnp) + head * R * hidden + start;
          float4 lo = load_float4(reinterpret_cast<uint64_t>(weights));
          float4 hi = load_float4(reinterpret_cast<uint64_t>(weights + 4));
          const float w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
          for (unsigned row = 0; row < Rows; ++row) {
#pragma unroll
            for (unsigned i = 0; i < 8; ++i)
              acc[row][j][i] =
                  __fmaf_rn(w[i], bf16_at(av[row], i), acc[row][j][i]);
          }
        }
      }
    };
    if (warp < P % Warps) {
      accumulate_heads(std::integral_constant<unsigned, Heads>{});
    } else {
      accumulate_heads(std::integral_constant<unsigned, Heads - 1>{});
    }
  } else {
    // A single head per warp can overlap two iterations without growing the
    // multi-head accumulator's register footprint.
    constexpr unsigned Unroll = Heads == 1 ? 2 : 1;
#pragma unroll Unroll
    for (unsigned start = lane * 8; start < R * hidden; start += 32 * 8) {
      PackedBf16 av[Rows];
#pragma unroll
      for (unsigned row = 0; row < Rows; ++row)
        av[row] =
            load_bf16(outp + 2 * ((Index(token) + row) * R * hidden + start));
#pragma unroll
      for (unsigned j = 0; j < Heads; ++j) {
        const unsigned head = warp + j * Warps;
        if (head < P) {
          const float *weights =
              reinterpret_cast<const float *>(fnp) + head * R * hidden + start;
          float4 lo = load_float4(reinterpret_cast<uint64_t>(weights));
          float4 hi = load_float4(reinterpret_cast<uint64_t>(weights + 4));
          const float w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
          for (unsigned row = 0; row < Rows; ++row) {
#pragma unroll
            for (unsigned i = 0; i < 8; ++i)
              acc[row][j][i] =
                  __fmaf_rn(w[i], bf16_at(av[row], i), acc[row][j][i]);
          }
        }
      }
    }
  }

#pragma unroll
  for (unsigned row = 0; row < Rows; ++row) {
#pragma unroll
    for (unsigned j = 0; j < Heads; ++j) {
      float magnitude = 0.f;
#pragma unroll
      for (unsigned i = 0; i < 8; ++i)
        magnitude += fabsf(acc[row][j][i]);
      magnitude = warp_sum(magnitude);
      if (lane == 0 && warp + j * Warps < P)
        shared_store(shared, Rows * 32 + row * 32 + warp + j * Warps,
                     magnitude * rstd[row]);
#pragma unroll
      for (unsigned width = 4; width > 0; width /= 2) {
#pragma unroll
        for (unsigned i = 0; i < width; ++i)
          acc[row][j][i] += acc[row][j][i + width];
      }
      const float sum = warp_sum(acc[row][j][0]);
      if (lane == 0 && warp + j * Warps < P)
        shared_store(shared, row * 32 + warp + j * Warps, sum * rstd[row]);
    }
    if (threadIdx.x < 32 && threadIdx.x >= P) {
      shared_store(shared, Rows * 32 + row * 32 + threadIdx.x, 0.f);
      if (threadIdx.x > P)
        shared_store(shared, row * 32 + threadIdx.x, 0.f);
    }
  }
  __syncthreads();
  return 0;
}

__device__ __forceinline__ void store_mask(unsigned base, unsigned index,
                                           unsigned value) {
  asm volatile("st.shared.b32 [ %0 + 0 ], %1;" ::"r"(base + 4 * index),
               "r"(value)
               : "memory");
}
__device__ __forceinline__ unsigned load_mask(unsigned base, unsigned index) {
  unsigned value;
  asm volatile("ld.shared.b32 %0, [ %1 + 0 ];"
               : "=r"(value)
               : "r"(base + 4 * index)
               : "memory");
  return value;
}

template <unsigned Warps, unsigned R, unsigned Rows>
__device__ __forceinline__ int retry_projection(uint64_t residual, uint64_t fnp,
                                                uint64_t token, unsigned hidden,
                                                unsigned shared) {
  constexpr unsigned P = R * (R + 2);
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  if (warp < Rows) {
    const float value = shared_load(shared, warp * 32 + lane);
    const float magnitude = shared_load(shared, Rows * 32 + warp * 32 + lane);
    const unsigned mask =
        __ballot_sync(0xffffffff, lane < P && magnitude > 0.f &&
                                      magnitude <= 0x1.fffffep127f &&
                                      fabsf(value) <= 0x1.fffffep127f &&
                                      fabsf(value) <= 0x1p-20f * magnitude);
    if (lane == 0)
      store_mask(shared, warp * 32 + P + 1, mask);
  }
  __syncthreads();
  // A cancelled head uses the whole CTA, rather than leaving a single
  // coefficient warp to serially reread the full hidden dimension.
  for (unsigned row = 0; row < Rows; ++row) {
    unsigned redo = load_mask(shared, row * 32 + P + 1);
    while (redo) {
      const unsigned head = __ffs(redo) - 1;
      float sum = 0.f, error = 0.f;
      for (unsigned k = threadIdx.x * 8; k < R * hidden; k += Warps * 32 * 8) {
        const uint64_t position = (token + row) * R * hidden + k;
        const PackedBf16 values = load_bf16(residual + 2 * position);
        const uint64_t address = fnp + 4 * (uint64_t(head) * R * hidden + k);
        const float4 lo = load_float4(address), hi = load_float4(address + 16);
        const float weights[8] = {lo.x, lo.y, lo.z, lo.w,
                                  hi.x, hi.y, hi.z, hi.w};
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
          const float corrected = bf16_at(values, i) * weights[i] - error;
          const float updated = sum + corrected;
          error = (updated - sum) - corrected;
          sum = updated;
        }
      }
      const float partial = warp_sum(sum - error);
      if (lane == 0)
        shared_store(shared, Rows * 32 + warp, partial);
      __syncthreads();
      if (warp == 0) {
        float total =
            lane < Warps ? shared_load(shared, Rows * 32 + lane) : 0.f;
        total = warp_sum(total) * shared_load(shared, row * 32 + P);
        if (lane == 0 && fabsf(total) <= 0x1.fffffep127f)
          shared_store(shared, row * 32 + head, total);
      }
      __syncthreads();
      redo &= redo - 1;
    }
  }
  return 0;
}

template <unsigned Warps, unsigned R, unsigned Rows, bool Cached>
__device__ __forceinline__ int
finish(uint64_t residual, uint64_t yp, uint64_t norm_p, uint64_t token,
       unsigned hidden, unsigned shared, float norm_eps, bool norm) {
  constexpr unsigned RowWarps = Warps / Rows;
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const unsigned row = warp / RowWarps, inrow = warp % RowWarps;

  float pre[R];
#pragma unroll
  for (unsigned src = 0; src < R; ++src)
    pre[src] = shared_load(shared, 2 * Rows * 32 + row * R + src);
  float sq[8] = {};
  constexpr unsigned VectorStride = RowWarps * 32 * 8;
  constexpr unsigned Unroll = Cached ? kCachedVectors : 1;
  const unsigned chunks =
      Cached ? kCachedVectors : (hidden + VectorStride - 1) / VectorStride;
  PackedBf16 tiles[Cached ? kCachedVectors : 1];
#pragma unroll Unroll
  for (unsigned chunk = 0; chunk < chunks; ++chunk) {
    const unsigned start = (inrow * 32 + lane) * 8 + chunk * VectorStride;
    if (start < hidden) {
      PackedBf16 sources[R], rounded;
#pragma unroll
      for (unsigned src = 0; src < R; ++src)
        sources[src] = load_bf16(
            residual + 2 * ((token + row) * R * hidden + src * hidden + start));
#pragma unroll
      for (unsigned word = 0; word < 4; ++word) {
        float lo = 0.f, hi = 0.f;
#pragma unroll
        for (unsigned src = 0; src < R; ++src) {
          lo = __fmaf_rn(pre[src], bf16_at(sources[src], word * 2), lo);
          hi = __fmaf_rn(pre[src], bf16_at(sources[src], word * 2 + 1), hi);
        }
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
            : "=r"(rounded.values[word])
            : "f"(hi), "f"(lo));
        lo = bf16_at(rounded, word * 2);
        hi = bf16_at(rounded, word * 2 + 1);
        sq[2 * word] += lo * lo;
        sq[2 * word + 1] += hi * hi;
      }
      // Cache BF16 bits, preserving the collapse rounding before RMSNorm.
      if constexpr (Cached)
        tiles[chunk] = rounded;
      else
        store_bf16(yp + 2 * ((token + row) * hidden + start), rounded);
    }
  }
#pragma unroll
  for (unsigned width = 4; width > 0; width /= 2) {
#pragma unroll
    for (unsigned j = 0; j < width; ++j)
      sq[j] += sq[j + width];
  }
  float sum = warp_sum(sq[0]);
  const unsigned scratch = 2 * Rows * 32 + Rows * R;
  if (lane == 0)
    shared_store(shared, scratch + warp, sum);
  __syncthreads();
  float total = lane < RowWarps
                    ? shared_load(shared, scratch + row * RowWarps + lane)
                    : 0.f;
  total = warp_sum(total);
  float rsqrt = rsqrtf(total / hidden + norm_eps);
  if (norm) {
#pragma unroll Unroll
    for (unsigned chunk = 0; chunk < chunks; ++chunk) {
      const unsigned start = (inrow * 32 + lane) * 8 + chunk * VectorStride;
      if (start < hidden) {
        PackedBf16 y;
        if constexpr (Cached)
          y = tiles[chunk];
        else
          y = load_bf16(yp + 2 * ((token + row) * hidden + start));
        PackedBf16 w = load_bf16(norm_p + 2 * start), out;
#pragma unroll
        for (unsigned word = 0; word < 4; ++word) {
          float lo = bf16_at(y, 2 * word) * rsqrt * bf16_at(w, 2 * word);
          float hi =
              bf16_at(y, 2 * word + 1) * rsqrt * bf16_at(w, 2 * word + 1);
          asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
              : "=r"(out.values[word])
              : "f"(hi), "f"(lo));
        }
        store_bf16(yp + 2 * ((token + row) * hidden + start), out);
      }
    }
  }
  __syncthreads();

  return 0;
}

template <unsigned Warps, unsigned R, unsigned Rows>
__device__ __forceinline__ int
epilogue(uint64_t residual, uint64_t fnp, uint64_t np, uint64_t nc, uint64_t yp,
         uint64_t scale_p, uint64_t base_p, uint64_t norm_p, uint64_t token,
         unsigned hidden, unsigned shared, unsigned repeat, float pre_eps,
         float sink_eps, float mult, float norm_eps, bool norm) {
  unsigned lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const float *scale = reinterpret_cast<const float *>(scale_p);
  const float *base = reinterpret_cast<const float *>(base_p);
  if (warp < Rows) {
    const unsigned row = warp;
    constexpr unsigned P = R * (R + 2);
    float logits = shared_load(shared, row * 32 + lane);
    if constexpr (!kCtaRetry<Warps, R, Rows>) {
      const float magnitude = shared_load(shared, Rows * 32 + row * 32 + lane);
      const float inverse = __shfl_sync(0xffffffff, logits, P);
      unsigned redo =
          __ballot_sync(0xffffffff, lane < P && magnitude > 0.f &&
                                        magnitude <= 0x1.fffffep127f &&
                                        fabsf(logits) <= 0x1.fffffep127f &&
                                        fabsf(logits) <= 0x1p-20f * magnitude);
      // Narrow tiles avoid an extra CTA handoff on the common path.
      while (redo) {
        const unsigned head = __ffs(redo) - 1;
        float sum = 0.f, error = 0.f;
        for (unsigned k = lane; k < R * hidden; k += 32) {
          const uint64_t position = (token + row) * R * hidden + k;
          const uint16_t bits =
              reinterpret_cast<const uint16_t *>(residual)[position];
          const float value = __uint_as_float(uint32_t(bits) << 16);
          const float product =
              value * reinterpret_cast<const float *>(
                          fnp)[uint64_t(head) * R * hidden + k];
          const float corrected = product - error;
          const float updated = sum + corrected;
          error = (updated - sum) - corrected;
          sum = updated;
        }
        const float projected = warp_sum(sum - error) * inverse;
        if (lane == head && fabsf(projected) <= 0x1.fffffep127f)
          logits = projected;
        redo &= redo - 1;
      }
    }
    float post_logit = __shfl_sync(0xffffffff, logits, R + lane % R);
    float combination_logit =
        __shfl_sync(0xffffffff, logits, 2 * R + lane % (R * R));
    if (lane < R) {
      const float pre =
          __fdividef(1.f, 1.f + __expf(-(logits * scale[0] + base[lane]))) +
          pre_eps;
      const float post =
          __fdividef(1.f,
                     1.f + __expf(-(post_logit * scale[1] + base[R + lane]))) *
          mult;
      shared_store(shared, 2 * Rows * 32 + row * R + lane, pre);
      reinterpret_cast<float *>(np)[(token + row) * R + lane] = post;
    }
    float c =
        lane < R * R ? combination_logit * scale[2] + base[2 * R + lane] : 0.f;
    float maximum = c;
#pragma unroll
    for (unsigned delta = 1; delta < R; delta *= 2)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
    c = __expf(c - maximum);
    float total = c;
#pragma unroll
    for (unsigned delta = 1; delta < R; delta *= 2)
      total += __shfl_xor_sync(0xffffffff, total, delta);
    c = c * __fdividef(1.f, total) + sink_eps;
    total = c;
#pragma unroll
    for (unsigned delta = R; delta < R * R; delta *= 2)
      total += __shfl_xor_sync(0xffffffff, total, delta);
    c = c * __fdividef(1.f, total + sink_eps);
    for (unsigned iter = 1; iter < repeat; ++iter) {
      total = c;
#pragma unroll
      for (unsigned delta = 1; delta < R; delta *= 2)
        total += __shfl_xor_sync(0xffffffff, total, delta);
      c = c * __fdividef(1.f, total + sink_eps);
      total = c;
#pragma unroll
      for (unsigned delta = R; delta < R * R; delta *= 2)
        total += __shfl_xor_sync(0xffffffff, total, delta);
      c = c * __fdividef(1.f, total + sink_eps);
    }
    if (lane < R * R)
      reinterpret_cast<float *>(nc)[(token + row) * R * R + lane] = c;
  }
  __syncthreads();
  // Reuse registers released by a multi-head projection. Narrow projections
  // keep the streaming finish to avoid raising their register requirement.
  if constexpr (Warps < R * (R + 2)) {
    if (norm && hidden <= kCachedVectors * (Warps / Rows) * 32 * 8)
      return finish<Warps, R, Rows, true>(residual, yp, norm_p, token, hidden,
                                          shared, norm_eps, norm);
  }
  return finish<Warps, R, Rows, false>(residual, yp, norm_p, token, hidden,
                                       shared, norm_eps, norm);
}

} // namespace megamhc_simt

#define MHC_SIMT(W, R, N)                                                      \
  extern "C" __device__ __attribute__((always_inline)) int                     \
  megamhc_simt_##W##_##R##_##N(                                                \
      uint64_t x, uint64_t residual, uint64_t post, uint64_t comb,             \
      uint64_t fn, uint64_t scale, uint64_t base, uint64_t norm,               \
      uint64_t output, uint64_t new_post, uint64_t new_comb, uint64_t y,       \
      uint64_t token, uint64_t s0, uint64_t s1, uint64_t s2, unsigned hidden,  \
      unsigned shared, unsigned repeat, float rms_eps, float pre_eps,          \
      float sink_eps, float mult, float norm_eps, unsigned has_norm) {         \
    megamhc_simt::project<W, R, N>(x, residual, post, comb, fn, output, token, \
                                   s0, s1, s2, hidden, shared, rms_eps);       \
    if constexpr (megamhc_simt::kCtaRetry<W, R, N>)                            \
      megamhc_simt::retry_projection<W, R, N>(output, fn, token, hidden,       \
                                              shared);                         \
    return megamhc_simt::epilogue<W, R, N>(                                    \
        output, fn, new_post, new_comb, y, scale, base, norm, token, hidden,   \
        shared, repeat, pre_eps, sink_eps, mult, norm_eps, has_norm);          \
  }
#define MHC_SIMT_STREAMS(R)                                                    \
  MHC_SIMT(32, R, 1)                                                           \
  MHC_SIMT(16, R, 1)                                                           \
  MHC_SIMT(8, R, 1)                                                            \
  MHC_SIMT(8, R, 2)                                                            \
  MHC_SIMT(32, R, 2)                                                           \
  MHC_SIMT(16, R, 2)                                                           \
  MHC_SIMT(8, R, 4)
MHC_SIMT_STREAMS(1)
MHC_SIMT_STREAMS(2)
MHC_SIMT_STREAMS(4)
