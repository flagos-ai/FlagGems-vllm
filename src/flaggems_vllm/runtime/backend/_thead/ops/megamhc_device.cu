// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
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
__device__ __forceinline__ int post(
    uint64_t x, uint64_t residual, uint64_t post_weights, uint64_t comb_weights,
    uint64_t output, uint64_t squares, uint64_t row_start, uint64_t comb_t,
    uint64_t comb_s, uint64_t comb_d, unsigned shared_slot, unsigned hidden) {
  constexpr unsigned slot_plane = kRows * kColumns * sizeof(uint16_t);
  constexpr unsigned slot_size = slot_plane * (R + 1);
  const unsigned local_row = threadIdx.x / kLanes;
  const unsigned lane = threadIdx.x % kLanes;
  const uint64_t row = row_start + local_row;
  const unsigned smem_lane = shared_slot + threadIdx.x * 16;
  const uint64_t x_offset = (row * hidden + kValues * lane) * sizeof(uint16_t);
  const uint64_t residual_offset =
      (row * R * hidden + kValues * lane) * sizeof(uint16_t);
  float post[R];
  float comb[R][R];
#pragma unroll
  for (unsigned dst = 0; dst < R; ++dst) {
    post[dst] = reinterpret_cast<const float *>(post_weights)[row * R + dst];
#pragma unroll
    for (unsigned src = 0; src < R; ++src) {
      comb[src][dst] = reinterpret_cast<const float *>(comb_weights)
          [row * comb_t + src * comb_s + dst * comb_d];
    }
  }

  auto prefetch = [&](unsigned column, unsigned slot) {
    const unsigned base = smem_lane + slot * slot_size;
    copy_tile(base, x + x_offset + 2 * column);
#pragma unroll
    for (unsigned src = 0; src < R; ++src) {
      copy_tile(base + (src + 1) * slot_plane,
                residual + residual_offset + 2 * (src * hidden + column));
    }
    asm volatile("cp.async.commit_group;" : : : "memory");
  };

  float sum[kValues] = {};
  unsigned current = 0;
  prefetch(0, current);
  for (unsigned column = 0; column < hidden; column += kColumns) {
    asm volatile("cp.async.wait_group 0;" : : : "memory");
    // The next tile occupies the other slot and can load while this tile
    // computes and publishes its residual. Each lane reads the bytes it copied.
    if (column + kColumns < hidden) {
      prefetch(column + kColumns, current ^ 1);
    }
    const unsigned base = smem_lane + current * slot_size;
    const Bf16x8 xv = read_tile(base);
    Bf16x8 residuals[R];
#pragma unroll
    for (unsigned src = 0; src < R; ++src) {
      residuals[src] = read_tile(base + (src + 1) * slot_plane);
    }
#pragma unroll
    for (unsigned dst = 0; dst < R; ++dst) {
      Bf16x8 rounded;
#pragma unroll
      for (unsigned word = 0; word < 4; ++word) {
        float lo = post[dst] * as_float(xv.word[word], 0);
        float hi = post[dst] * as_float(xv.word[word], 1);
#pragma unroll
        for (unsigned src = 0; src < R; ++src) {
          lo = __fmaf_rn(comb[src][dst], as_float(residuals[src].word[word], 0), lo);
          hi = __fmaf_rn(comb[src][dst], as_float(residuals[src].word[word], 1), hi);
        }
        rounded.word[word] = to_bf16(lo, hi);
        const float rounded_lo = as_float(rounded.word[word], 0);
        const float rounded_hi = as_float(rounded.word[word], 1);
        sum[2 * word] = __fmaf_rn(rounded_lo, rounded_lo, sum[2 * word]);
        sum[2 * word + 1] =
            __fmaf_rn(rounded_hi, rounded_hi, sum[2 * word + 1]);
      }
      write_tile(output + residual_offset + 2 * (dst * hidden + column), rounded);
    }
    current ^= 1;
  }
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
  if (lane == 0) {
    reinterpret_cast<float *>(squares)[row] = sum[0];
  }
  return 0;
}
} // namespace

#define DEFINE_MHC_POST(R)                                                     \
  extern "C" __device__ __attribute__((always_inline)) int megamhc_post_##R(    \
      uint64_t x, uint64_t residual, uint64_t weights, uint64_t comb,            \
      uint64_t output, uint64_t squares, uint64_t row_start, uint64_t comb_t,    \
      uint64_t comb_s, uint64_t comb_d, unsigned smem, unsigned hidden) {         \
    return post<R>(x, residual, weights, comb, output, squares, row_start,        \
                   comb_t, comb_s, comb_d, smem, hidden);                       \
  }

DEFINE_MHC_POST(2)
DEFINE_MHC_POST(4)
