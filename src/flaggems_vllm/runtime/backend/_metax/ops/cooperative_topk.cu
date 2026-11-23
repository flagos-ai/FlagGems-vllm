// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

#include <math.h>
#include <mc_runtime.h>
#include <stdint.h>

namespace c550_row_detail {

__device__ __forceinline__ unsigned ordered_key(float x) {
  unsigned a = __float_as_uint(x), mag = a & 0x7fffffff;
  return mag > 0x7f800000u ? 0xffffffffu
         : mag == 0        ? 0x80000000u
                           : (a & 0x80000000u ? ~a : a ^ 0x80000000u);
}
template <int BINS>
__device__ __forceinline__ unsigned coarse_bucket(float x, unsigned u) {
  return u == 0xffffffffu
             ? BINS - 1
             : (unsigned)fminf(
                   fmaxf(x * (float(BINS) / 8.0f) + float(BINS / 2), 0.f),
                   float(BINS - 1));
}
// Each lane owns adjacent buckets. A wave suffix scan finds the interval
// containing the target rank; only its owner visits the individual bins.
template <int BINS>
__device__ void select_bin(const unsigned *histogram, unsigned *digit,
                           unsigned *above, unsigned *count,
                           unsigned remaining) {
  if (threadIdx.x < 64) {
    constexpr int ITEMS = BINS / 64;
    unsigned local[ITEMS], total = 0;
#pragma unroll
    for (int j = 0; j < ITEMS; j++) {
      local[j] = histogram[threadIdx.x * ITEMS + j];
      total += local[j];
    }
    unsigned suffix = total;
#pragma unroll
    for (int off = 1; off < 64; off *= 2) {
      auto adj = __shfl_down_sync(~0ull, suffix, off, 64);
      if (threadIdx.x + off < 64)
        suffix += adj;
    }
    unsigned higher = suffix - total;
    if (higher < remaining && remaining <= suffix) {
#pragma unroll
      for (int j = ITEMS - 1; j >= 0; j--) {
        if (higher < remaining && remaining <= higher + local[j]) {
          *digit = threadIdx.x * ITEMS + j;
          *above = higher;
          *count = local[j];
        }
        higher += local[j];
      }
    }
  }
  __syncthreads();
}
template <int MAIN, int BINS, bool TAIL, bool FULL>
__device__ __forceinline__ void select_tiles(const float *logits, int *out,
                                             int length, int k) {
  constexpr int NT = 512;
  int tid = threadIdx.x;
  constexpr int ITEMS = MAIN / NT;
  __shared__ unsigned histogram[BINS], digit, above, count, greater_cursor,
      boundary_cursor;
  __shared__ unsigned boundary_keys[128], boundary_indices[128];
  for (int b = tid; b < BINS; b += NT)
    histogram[b] = 0;
  if (!tid)
    greater_cursor = boundary_cursor = 0;
  __syncthreads();
  unsigned key[ITEMS];
  unsigned short buckets[ITEMS];
#pragma unroll
  for (int z = 0; z < ITEMS / 4; ++z) {
    int base = 4 * (tid + z * NT);
    float4 v;
    if (FULL || (base + 3 < length && ((uintptr_t)(logits + base) % 16) == 0))
      v = *(const float4 *)(logits + base);
    else {
      v.x = base < length ? logits[base] : 0;
      v.y = base + 1 < length ? logits[base + 1] : 0;
      v.z = base + 2 < length ? logits[base + 2] : 0;
      v.w = base + 3 < length ? logits[base + 3] : 0;
    }
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
      int j = z * 4 + sub, i = base + sub;
      float f = sub == 0 ? v.x : sub == 1 ? v.y : sub == 2 ? v.z : v.w;
      key[j] = ordered_key(f);
      buckets[j] = coarse_bucket<BINS>(f, key[j]);
      if ((FULL || i < length))
        atomicAdd(histogram + buckets[j], 1u);
    }
  }
  unsigned tail_key = 0, tail_bucket = 0;
  if (TAIL && tid == 0 && length > MAIN) {
    float v = logits[MAIN];
    tail_key = ordered_key(v);
    tail_bucket = coarse_bucket<BINS>(v, tail_key);
    atomicAdd(histogram + tail_bucket, 1u);
  }
  __syncthreads();
  select_bin<BINS>(histogram, &digit, &above, &count, k);
  unsigned boundary = digit, boundary_count = count, local_greater = 0,
           local_equal = 0;
  // Local ranges come from one reservation per contributing thread.
  if (boundary_count <= 128) {
#pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
      int i = 4 * (tid + (j / 4) * NT) + (j % 4);
      local_greater += (FULL || i < length) && buckets[j] > boundary;
      local_equal += (FULL || i < length) && buckets[j] == boundary;
    }
    bool owns_tail = TAIL && length > MAIN && !tid;
    local_greater += owns_tail && tail_bucket > boundary;
    local_equal += owns_tail && tail_bucket == boundary;
    unsigned greater_slot =
                 local_greater ? atomicAdd(&greater_cursor, local_greater) : 0,
             equal_slot =
                 local_equal ? atomicAdd(&boundary_cursor, local_equal) : 0;
#pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
      int i = 4 * (tid + (j / 4) * NT) + (j % 4);
      if ((FULL || i < length) && buckets[j] > boundary)
        out[greater_slot++] = i;
      if ((FULL || i < length) && buckets[j] == boundary) {
        boundary_keys[equal_slot] = key[j];
        boundary_indices[equal_slot++] = i;
      }
    }
    if (owns_tail && tail_bucket > boundary)
      out[greater_slot] = MAIN;
    if (owns_tail && tail_bucket == boundary) {
      boundary_keys[equal_slot] = tail_key;
      boundary_indices[equal_slot] = MAIN;
    }
    __syncthreads();
    // Each wave ranks one boundary key at a time. Shared-slot order breaks
    // ties.
    int lane = tid & 63, wave_id = tid >> 6;
    unsigned lower_key = lane < boundary_count ? boundary_keys[lane] : 0,
             upper_key =
                 lane + 64 < boundary_count ? boundary_keys[lane + 64] : 0;
    for (unsigned z = wave_id; z < boundary_count; z += NT / 64) {
      unsigned pivot = boundary_keys[z];
      unsigned rank = __popcll(__ballot_sync(
          ~0ull, lane < boundary_count &&
                     (lower_key > pivot || (lower_key == pivot && lane < z))));
      if (boundary_count > 64)
        rank += __popcll(__ballot_sync(
            ~0ull,
            lane + 64 < boundary_count &&
                (upper_key > pivot || (upper_key == pivot && lane + 64 < z))));
      if (!lane && above + rank < k)
        out[above + rank] = boundary_indices[z];
    }
  } else {
    // Refine all four bytes when the boundary does not fit in shared candidate
    // storage.
    unsigned prefix = 0, remaining = k - above;
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (int b = tid; b < 256; b += NT)
        histogram[b] = 0;
      __syncthreads();
#pragma unroll
      for (int j = 0; j < ITEMS; ++j) {
        int i = 4 * (tid + (j / 4) * NT) + (j % 4);
        unsigned u = key[j];
        if ((FULL || i < length) && buckets[j] == boundary &&
            (shift == 24 || (u >> (shift + 8)) == (prefix >> (shift + 8))))
          atomicAdd(histogram + ((u >> shift) & 255), 1u);
      }
      if (TAIL && tid == 0 && length > MAIN && tail_bucket == boundary &&
          (shift == 24 || (tail_key >> (shift + 8)) == (prefix >> (shift + 8))))
        atomicAdd(histogram + ((tail_key >> shift) & 255), 1u);
      __syncthreads();
      select_bin<256>(histogram, &digit, &above, &count, remaining);
      prefix |= digit << shift;
      remaining -= above;
      __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
      int i = 4 * (tid + (j / 4) * NT) + (j % 4);
      local_greater += (FULL || i < length) && key[j] > prefix;
      local_equal += (FULL || i < length) && key[j] == prefix;
    }
    bool owns_tail = TAIL && length > MAIN && !tid;
    local_greater += owns_tail && tail_key > prefix;
    local_equal += owns_tail && tail_key == prefix;
    unsigned greater_slot =
                 local_greater ? atomicAdd(&greater_cursor, local_greater) : 0,
             equal_slot =
                 local_equal ? atomicAdd(&boundary_cursor, local_equal) : 0;
#pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
      int i = 4 * (tid + (j / 4) * NT) + (j % 4);
      if ((FULL || i < length) && key[j] > prefix)
        out[greater_slot++] = i;
      if ((FULL || i < length) && key[j] == prefix) {
        if (equal_slot < remaining)
          out[k - remaining + equal_slot] = i;
        equal_slot++;
      }
    }
    if (owns_tail && tail_key > prefix)
      out[greater_slot] = MAIN;
    if (owns_tail && tail_key == prefix && equal_slot < remaining)
      out[k - remaining + equal_slot] = MAIN;
  }
}
template <int MAIN, int BINS, bool TAIL>
__global__ void select_row(const float *logits, const int *lengths, int *out,
                           int width, int64_t stride, int k) {
  constexpr int NT = 512;
  int tid = threadIdx.x, row = blockIdx.x,
      length = max(0, min(lengths[row], width));
  logits += row * (size_t)stride;
  out += row * k;
  if (length <= k) {
    for (int i = tid; i < k; i += NT)
      out[i] = i < length ? i : -1;
    return;
  }
  if (length >= MAIN && ((uintptr_t)logits % 16) == 0)
    select_tiles<MAIN, BINS, TAIL, true>(logits, out, length, k);
  else
    select_tiles<MAIN, BINS, TAIL, false>(logits, out, length, k);
}
} // namespace c550_row_detail

// The caller validates dtype, buffer layout and the supported width before
// launch. A single peeled element avoids doubling the full tile for
// power-of-two + 1 widths.
inline void c550_row(const float *logits, const int *lengths, int *output,
                     int rows, int width, int64_t stride, int k,
                     mcStream_t stream) {
  using c550_row_detail::select_row;
  if (width <= 4097) {
    select_row<4096, 256, true>
        <<<rows, 512, 0, stream>>>(logits, lengths, output, width, stride, k);
  } else if (width <= 8193) {
    select_row<8192, 512, true>
        <<<rows, 512, 0, stream>>>(logits, lengths, output, width, stride, k);
  } else if (width <= 16384) {
    select_row<16384, 1024, false>
        <<<rows, 512, 0, stream>>>(logits, lengths, output, width, stride, k);
  } else {
    select_row<16384, 1024, true>
        <<<rows, 512, 0, stream>>>(logits, lengths, output, width, stride, k);
  }
}

namespace c550_wide_detail {

// Canonical keys preserve subnormals and signed-zero equality. Both NaN
// signs carry the largest key, so only exact keys decide boundary ties.
__device__ __forceinline__ unsigned ordered_key(float f) {
  unsigned u = __float_as_uint(f);
  unsigned mag = u & 0x7fffffff;
  if (mag > 0x7f800000)
    return ~0u;
  if (mag == 0)
    return 0x80000000u;
  return (u >> 31) ? ~u : u ^ 0x80000000u;
}
// Coarse bins only prune candidates. Saturated bins can contain arbitrary
// input distributions and go through the same exact refinement.
template <int Bins> __device__ __forceinline__ int coarse_bucket(float v) {
  if (__float_as_uint(v) << 1 > 0xff000000u)
    return Bins - 1;
  return (int)fminf(Bins - 1.f, fmaxf(0.f, fmaf(v, Bins / 8.f, Bins / 2.f)));
}
__device__ __forceinline__ int reserve_lanes(bool pred, int *counter) {
  int lane = threadIdx.x & 63;
  unsigned long long votes = __ballot(pred);
  int base = 0;
  if (votes) {
    if (lane == 0)
      base = atomicAdd(counter, __popcll(votes));
    base = __shfl(base, 0, 64);
  }
  return base + __popcll(votes & ((1ull << lane) - 1ull));
}
template <int Parts, int Bins, int Threads>
__global__ void count_coarse_buckets(const float *logits, const int *lengths,
                                     int *workspace, int width,
                                     int64_t stride) {
  int part = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
  int length = min(width, max(0, lengths[row]));
  auto histogram = (unsigned short *)(workspace + 4096 * row);
  __shared__ int hist[Bins];
  for (int i = tid; i < Bins; i += Threads)
    hist[i] = 0;
  __syncthreads();
  const int tile = (width + Parts - 1) / Parts;
  int first = part * tile, end = min(length, first + tile);
  for (int i = first + tid; i < end; i += Threads)
    atomicAdd(hist + coarse_bucket<Bins>(logits[(long long)row * stride + i]),
              1);
  __syncthreads();
  for (int i = tid; i < Bins; i += Threads)
    histogram[part * Bins + i] = hist[i];
}

__device__ __forceinline__ int wave_scan(int v) {
  int lane = threadIdx.x & 63;
  for (int delta = 1; delta < 64; delta *= 2) {
    int u = __shfl_up(v, delta, 64);
    if (lane >= delta)
      v += u;
  }
  return v;
}
// First wave chooses a digit from the cumulative population.
template <int Bins>
__device__ __forceinline__ void
select_digit(const int *hist, int need, int &digit, int &above, int &equal) {
  int lane = threadIdx.x & 63;
  int end = Bins - 1 - lane * (Bins / 64), total = 0;
  for (int i = 0; i < Bins / 64; i++)
    total += hist[end - i];
  int greater = wave_scan(total) - total;
  int win = need > greater && need <= greater + total;
  int src = __ffsll(__ballot(win)) - 1;
  src = max(0, src);
  int d = end, left = need - greater;
  for (int i = 0; i < Bins / 64 - 1; i++) {
    int z = hist[d];
    if (left > z) {
      left -= z;
      d--;
    }
  }
  digit = __shfl(d, src, 64);
  above = __shfl(need - left, src, 64);
  equal = __shfl(hist[d], src, 64);
}
template <int Parts, int Bins, int Threads, int Capacity>
__global__ void select_coarse_bucket(int *workspace, const int *lengths,
                                     int k) {
  int row = blockIdx.x, tid = threadIdx.x, lane = tid & 63, wave = tid / 64;
  if (lengths[row] <= k) {
    if (tid == 0)
      workspace[row * 4096] = 0;
    return;
  }
  workspace += 4096 * row;
  auto histogram = (unsigned short *)workspace;
  __shared__ int hist[Bins], partial[Threads], cut, above_count, boundary_count,
      above_wave_sums[2], equal_wave_sums[2];
  int bin = tid % Bins, group = tid / Bins, sum = 0;
  for (int part = group; part < Parts; part += Threads / Bins)
    sum += histogram[part * Bins + bin];
  partial[tid] = sum;
  __syncthreads();
  if (tid < Bins) {
    int total = 0;
    for (int g = 0; g < Threads / Bins; g++)
      total += partial[g * Bins + tid];
    hist[tid] = total;
  }
  __syncthreads();
  if (tid < 64) {
    int d, a, e;
    select_digit<Bins>(hist, k, d, a, e);
    if (!tid) {
      cut = d;
      above_count = a;
      boundary_count = e;
    }
  }
  __syncthreads();
  int part = tid % Parts, g = tid / Parts;
  sum = 0;
  for (int b = cut + 1 + g; b < Bins; b += Threads / Parts)
    sum += histogram[part * Bins + b];
  partial[tid] = sum;
  __syncthreads();
  int v = 0, e = 0;
  if (tid < Parts) {
    for (int j = 0; j < Threads / Parts; j++)
      v += partial[j * Parts + tid];
    e = histogram[tid * Bins + cut];
  }
  int above_prefix = wave_scan(v), equal_prefix = wave_scan(e);
  if (lane == 63 && tid < Parts) {
    above_wave_sums[wave] = above_prefix;
    equal_wave_sums[wave] = equal_prefix;
  }
  __syncthreads();
  if (wave && tid < Parts) {
    above_prefix += above_wave_sums[0];
    equal_prefix += equal_wave_sums[0];
  }
  if (tid < Parts) {
    workspace[4 + tid] = above_prefix - v;
    workspace[4 + Parts + tid] = equal_prefix - e;
  }
  bool repair = lengths[row] > k && boundary_count > Capacity &&
                boundary_count != k - above_count;
  if (!tid) {
    workspace[0] = repair ? -1 : cut;
    workspace[1] = above_count;
    workspace[2] = boundary_count;
    workspace[3] = 0;
  }
  if (repair)
    for (int j = tid; j < 16 + 3072; j += Threads)
      workspace[4 + 2 * Parts + j] = 0;
}
// Sort only the active bits of the compacted boundary keys. A complete
// boundary bin fitting the remaining quota allows inclusive emission before
// all low bits are known; otherwise emit exactly the required number of ties.

template <int Threads, int Capacity>
__device__ unsigned select_compacted(const volatile unsigned *candidate_keys,
                                     int boundary_count, int need, int *hist,
                                     unsigned &remaining) {
  __shared__ unsigned prefix, rowMin;
  __shared__ unsigned lows[Threads / 64], highs[Threads / 64];
  __shared__ int remaining_rank, done, startShift;
  unsigned keys[Capacity / Threads], low = 0xffffffffu, high = 0;
  int tid = threadIdx.x, lane = tid & 63;
  for (int i = 0; i < Capacity / Threads; i++) {
    unsigned k = (i * Threads + tid < boundary_count)
                     ? candidate_keys[i * Threads + tid]
                     : 0;
    keys[i] = k;
    low = min(low, k ? k : 0xffffffffu);
    high = max(high, k);
  }
  for (int delta = 32; delta; delta >>= 1) {
    low = min(low, __shfl_xor(low, delta, 64));
    high = max(high, __shfl_xor(high, delta, 64));
  }
  if (!lane) {
    lows[tid / 64] = low;
    highs[tid / 64] = high;
  }
  __syncthreads();
  if (!tid) {
    for (int wave = 0; wave < Threads / 64; wave++) {
      low = min(low, lows[wave]);
      high = max(high, highs[wave]);
    }
    rowMin = low;
    startShift = high == low ? 0 : max(0, 24 - __clz(high - low));
    prefix = 0;
    remaining_rank = need;
    done = (high == low) ? 2 : 0;
  }
  __syncthreads();
  for (int i = 0; i < Capacity / Threads; i++)
    keys[i] -= rowMin;
  int shift = startShift, previous_shift = 32;
  while (!done) {
    for (int j = tid; j < 256; j += Threads)
      hist[j] = 0;
    __syncthreads();
    for (int i = 0; i < Capacity / Threads; i++)
      if (i * Threads + tid < boundary_count &&
          (previous_shift == 32 ||
           (keys[i] >> previous_shift) == (prefix >> previous_shift)))
        atomicAdd(hist + ((keys[i] >> shift) & 255), 1);
    __syncthreads();
    if (tid < 64) {
      int d, a, e;
      int left = remaining_rank;
      select_digit<256>(hist, left, d, a, e);
      if (!tid) {
        prefix |= (unsigned)d << shift;
        remaining_rank = left - a;
        if (shift == 0)
          done = 2;
        else if (remaining_rank == e)
          done = 1;
      }
    }
    __syncthreads();
    previous_shift = shift;
    shift = max(0, shift - 8);
  }
  remaining = done == 1 ? 0x80000000u : remaining_rank;
  return rowMin + prefix;
}

// Local histograms are published via device-scope atomics. The last CTA derives
// the next 11/10/11-bit digit. The first and third histograms share space.
// Every repair pass has its own completion counter and immutable prefix.
template <int Parts, int Threads, int Phase>
__device__ void radix_phase(const float *logits, int length, int width, int k,
                            int *workspace, int *output) {
  int part = blockIdx.x, tid = threadIdx.x;
  int *state = workspace + 4 + 2 * Parts;
  int chunk = (width + Parts - 1) / Parts;
  int begin = part * chunk, end = min(begin + chunk, length);
  if constexpr (Phase < 3) {
    constexpr int bins = Phase == 1 ? 1024 : 2048;
    constexpr int shift = Phase == 0 ? 21 : (Phase == 1 ? 11 : 0);
    constexpr int previous_shift = Phase == 2 ? 11 : 21;
    __shared__ int histogram[bins], last;
    for (int i = tid; i < bins; i += Threads)
      histogram[i] = 0;
    __syncthreads();
    unsigned prefix = Phase ? (unsigned)state[4 + Phase - 1] : 0;
    for (int j = begin + tid; j < end; j += Threads) {
      unsigned bits = ordered_key(logits[j]);
      bool active = true;
      if constexpr (Phase > 0)
        active = (bits >> previous_shift) == (prefix >> previous_shift);
      if (active)
        atomicAdd(histogram + ((bits >> shift) & (bins - 1)), 1);
    }
    __syncthreads();
    int *global = state + 16 + (Phase == 1 ? 2048 : 0);
    for (int j = tid; j < bins; j += Threads)
      if (histogram[j])
        atomicAdd(global + j, histogram[j]);
    __threadfence();
    __syncthreads();
    if (tid == 0)
      last =
          __atomic_fetch_add(state + Phase, 1, __ATOMIC_ACQ_REL) == Parts - 1;
    __syncthreads();
    if (last) {
      __threadfence();
      for (int j = tid; j < bins; j += Threads)
        histogram[j] = ((volatile int *)global)[j];
      __syncthreads();
      if constexpr (Phase == 0)
        for (int j = tid; j < bins; j += Threads)
          global[j] = 0;
      if (tid < 64) {
        int d, a, e, remaining_rank = Phase ? state[8 + Phase - 1] : k;
        select_digit<bins>(histogram, remaining_rank, d, a, e);
        if (tid == 0) {
          state[4 + Phase] = (int)(prefix | ((unsigned)d << shift));
          state[8 + Phase] = remaining_rank - a;
        }
      }
    }
  } else {
    unsigned cutoff_key = (unsigned)state[6];
    int ties = state[10], lane = tid & 63;
    for (int start = begin; start < end; start += Threads) {
      int j = start + tid;
      unsigned b = j < end ? ordered_key(logits[j]) : 0;
      auto greater_mask = __ballot(j < end && b > cutoff_key),
           equal = __ballot(j < end && b == cutoff_key);
      int g = 0, e = 0;
      if (!lane) {
        if (greater_mask)
          g = atomicAdd(state + 12, __popcll(greater_mask));
        if (equal)
          e = atomicAdd(state + 13, __popcll(equal));
      }
      g = __shfl(g, 0, 64) + __popcll(greater_mask & ((1ull << lane) - 1ull));
      e = __shfl(e, 0, 64) + __popcll(equal & ((1ull << lane) - 1ull));
      if (j < end && b > cutoff_key)
        output[g] = j;
      if (j < end && b == cutoff_key && e < ties)
        output[k - ties + e] = j;
    }
  }
}
template <int Parts, int Threads, int Phase>
__global__ void repair_overflow(const float *logits, const int *lengths,
                                int *output, int *workspace, int width,
                                int64_t stride, int k) {
  int row = blockIdx.y;
  workspace += row * 4096;
  if (workspace[0] != -1)
    return;
  radix_phase<Parts, Threads, Phase>(logits + (long long)row * stride,
                                     min(width, max(0, lengths[row])), width, k,
                                     workspace, output + row * k);
}
template <int Parts, int Bins, int Threads, int Capacity>
__global__ void emit_candidates(const float *logits, const int *lengths,
                                int *output, int *workspace, int width,
                                int64_t stride, int k) {
  int row = blockIdx.y, part = blockIdx.x, tid = threadIdx.x;
  workspace += 4096 * row;
  logits += (long long)row * stride;
  output += row * k;
  int length = min(width, max(0, lengths[row]));
  if (length <= k) {
    for (int i = part * Threads + tid; i < k; i += Parts * Threads)
      output[i] = (i < length ? i : -1);
    return;
  }
  int d = workspace[0], above_count = workspace[1],
      boundary_count = workspace[2], need = k - above_count;
  if (d < 0) {
    radix_phase<Parts, Threads, 0>(logits, length, width, k, workspace, output);
    return;
  }
  volatile int *candidates = workspace + 4 + 2 * Parts;
  __shared__ int a, e, last, hist[256];
  if (tid == 0) {
    a = workspace[4 + part];
    e = workspace[4 + Parts + part];
  }
  __syncthreads();
  int tile = (width + Parts - 1) / Parts, first = tile * part,
      end = min(length, first + tile);
  // All lanes participate in the ballot, including the partial final wave.
  for (int start = first; start < end; start += Threads) {
    int i = start + tid;
    int b = i < end ? coarse_bucket<Bins>(logits[i]) : -1;
    int ga = reserve_lanes(b > d, &a);
    if (b > d)
      output[ga] = i;
    int ge = reserve_lanes(b == d, &e);
    if (b == d) {
      if (boundary_count == need)
        output[above_count + ge] = i;
      else {
        candidates[ge] = i;
        ((volatile unsigned *)candidates)[Capacity + ge] =
            ordered_key(logits[i]);
      }
    }
  }
  if (boundary_count == need)
    return;
  __threadfence();
  __syncthreads();
  if (tid == 0)
    last = __atomic_fetch_add(workspace + 3, 1, __ATOMIC_ACQ_REL) == Parts - 1;
  __syncthreads();
  if (!last)
    return;
  // Acquire the other CTAs' volatile writes before consuming their keys.
  __threadfence();
  __shared__ unsigned threshold;
  __shared__ int ties;
  unsigned remaining_rank = 0;
  unsigned cutoff_key = select_compacted<Threads, Capacity>(
      (volatile unsigned *)candidates + Capacity, boundary_count, need, hist,
      remaining_rank);
  if (tid == 0) {
    threshold = cutoff_key;
    ties = remaining_rank;
  }
  __syncthreads();
  if (tid == 0) {
    a = above_count;
    e = 0;
  }
  __syncthreads();
  for (int start = 0; start < boundary_count; start += Threads) {
    int j = start + tid, i = (j < boundary_count) ? candidates[j] : 0;
    unsigned u = j < boundary_count
                     ? ((volatile unsigned *)candidates)[Capacity + j]
                     : 0;
    if (ties < 0) {
      int z = reserve_lanes(j < boundary_count && u >= threshold, &a);
      if (j < boundary_count && u >= threshold)
        output[z] = i;
      continue;
    }
    int ga = reserve_lanes(j < boundary_count && u > threshold, &a);
    if (j < boundary_count && u > threshold)
      output[ga] = i;
    int ge = reserve_lanes(j < boundary_count && u == threshold, &e);
    if (j < boundary_count && u == threshold && ge < ties)
      output[k - ties + ge] = i;
  }
}
template <int Parts, int Bins, int Threads, int Capacity = 1024>
inline void launch(const float *logits, const int *lengths, int *output,
                   int *workspace, int rows, int width, int64_t stride, int k,
                   mcStream_t stream) {
  // The caller supplies 4096 int32 words per row. The coarse histogram
  // shares them with the prefix tables and either the candidate buffer or
  // the overflow histograms. Different launches order these scratch uses.
  static_assert(Parts * Bins <= 8192 && 4 + 2 * Parts + 2 * Capacity <= 4096 &&
                4 + 2 * Parts + 16 + 3072 <= 4096);
  static_assert(Parts <= 128 && Threads % 64 == 0 && Capacity % Threads == 0);
  count_coarse_buckets<Parts, Bins, Threads>
      <<<dim3(Parts, rows), Threads, 0, stream>>>(logits, lengths, workspace,
                                                  width, stride);
  select_coarse_bucket<Parts, Bins, 512, Capacity>
      <<<rows, 512, 0, stream>>>(workspace, lengths, k);
  emit_candidates<Parts, Bins, Threads, Capacity>
      <<<dim3(Parts, rows), Threads, 0, stream>>>(logits, lengths, output,
                                                  workspace, width, stride, k);
  repair_overflow<Parts, Threads, 1><<<dim3(Parts, rows), Threads, 0, stream>>>(
      logits, lengths, output, workspace, width, stride, k);
  repair_overflow<Parts, Threads, 2><<<dim3(Parts, rows), Threads, 0, stream>>>(
      logits, lengths, output, workspace, width, stride, k);
  repair_overflow<Parts, Threads, 3><<<dim3(Parts, rows), Threads, 0, stream>>>(
      logits, lengths, output, workspace, width, stride, k);
}
} // namespace c550_wide_detail

// Supported caller range is 16385 < width <= 1048576. Both schedules keep
// the per-partition maximum below the 16-bit coarse histogram limit.
inline void c550_wide(const float *logits, const int *lengths, int *output,
                      int *scratch, int rows, int width, int64_t stride, int k,
                      mcStream_t stream) {
  if (width <= 131072)
    c550_wide_detail::launch<32, 256, 512, 512>(
        logits, lengths, output, scratch, rows, width, stride, k, stream);
  else
    c550_wide_detail::launch<128, 64, 512, 1024>(
        logits, lengths, output, scratch, rows, width, stride, k, stream);
}

extern "C" const char *
flaggems_c550_cooperative_topk(const float *logits, const int *lengths,
                               int *output, int *workspace, int rows, int width,
                               int64_t stride, int k, void *stream_handle) {
  auto stream = static_cast<mcStream_t>(stream_handle);
  if (width <= 16385) {
    c550_row(logits, lengths, output, rows, width, stride, k, stream);
  } else {
    c550_wide(logits, lengths, output, workspace, rows, width, stride, k,
              stream);
  }
  const auto error = mcGetLastError();
  return error == mcSuccess ? nullptr : mcGetErrorString(error);
}
