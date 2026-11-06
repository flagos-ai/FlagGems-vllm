// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>

// UserHopper-shaped D6-D9 tail for a two-warp TLE dispatch role.
//
// Both physical warps enter this function after jointly completing D1-D6 in
// TLE.  They then reuse themselves as independent pull streams.  No CTA-wide
// barrier is legal here: loader/math partitions execute different programs.

static constexpr unsigned long long kEvictFirst = 0x12f0000000000000ull;
static constexpr unsigned long long kEvictNormal = 0x1000000000000000ull;

__device__ __forceinline__ void mbarrier_init(unsigned barrier) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
               :
               : "r"(barrier)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(unsigned barrier,
                                                          int num_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :
               : "r"(barrier), "r"(num_bytes)
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait(unsigned barrier,
                                              unsigned &phase) {
  asm volatile(
      "{\n\t"
      ".reg .pred done;\n\t"
      "D8_UNIFIED_WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n\t"
      "@done bra D8_UNIFIED_DONE_%=;\n\t"
      "bra D8_UNIFIED_WAIT_%=;\n\t"
      "D8_UNIFIED_DONE_%=:\n\t"
      "}"
      :
      : "r"(barrier), "r"(phase), "r"(0x989680)
      : "memory");
  phase ^= 1;
}

__device__ __forceinline__ void tma_load_1d(unsigned stage,
                                            const void *global_src,
                                            unsigned barrier, int num_bytes) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes."
      "L2::cache_hint [%0], [%1], %2, [%3], %4;"
      :
      : "r"(stage), "l"(global_src), "r"(num_bytes), "r"(barrier),
        "l"(kEvictFirst)
      : "memory");
}

__device__ __forceinline__ void
tma_store_1d_issue(void *global_dst, unsigned stage, int num_bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint "
               "[%0], [%1], %2, %3;"
               :
               : "l"(global_dst), "r"(stage), "r"(num_bytes), "l"(kEvictNormal)
               : "memory");
}

__device__ __forceinline__ void tma_store_commit_wait() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void release_add_one(unsigned int *ptr) {
  unsigned old;
  asm volatile("atom.global.add.release.gpu.u32 %0, [%1], 1;"
               : "=r"(old)
               : "l"(ptr)
               : "memory");
  (void)old;
}

__device__ __forceinline__ void d8_unified_dispatch_pull_impl(
    const uint64_t rsum_addr, const uint64_t recv_addr,
    const uint64_t queue_addr, const uint64_t metadata_addr,
    const uint64_t sf_table_addr, const uint64_t weight_table_addr,
    const uint64_t arrival_addr, const uint64_t l1_token_addr,
    const uint64_t l1_sf_addr, const uint64_t l1_weight_addr,
    const uint64_t stage0_addr, const uint64_t state0_addr,
    const uint64_t stage1_addr, const uint64_t state1_addr,
    const uint64_t token0, const uint64_t token1, const uint64_t token2,
    const uint64_t token3, const uint64_t token4, const uint64_t token5,
    const uint64_t token6, const uint64_t token7, const int num_ranks,
    const int num_sms, const int experts_per_rank, const int max_recv,
    const int block_m, const int hidden, const int num_sf,
    const int pool_tokens, const int topk) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned physical_warp = threadIdx.x >> 5;
  const unsigned warp_in_partition = physical_warp & 3u;
  if (warp_in_partition >= 2u)
    return;
  const unsigned stream = warp_in_partition;

  // Each stream owns one SMEM token buffer and one mbarrier.
  const uint64_t stage_addr = stream == 0 ? stage0_addr : stage1_addr;
  const uint64_t state_addr = stream == 0 ? state0_addr : state1_addr;
  const unsigned stage = static_cast<unsigned>(__cvta_generic_to_shared(
      reinterpret_cast<void *>(static_cast<uintptr_t>(stage_addr))));
  const unsigned barrier = static_cast<unsigned>(__cvta_generic_to_shared(
      reinterpret_cast<void *>(static_cast<uintptr_t>(state_addr))));

  const auto *rsum =
      reinterpret_cast<const uint64_t *>(static_cast<uintptr_t>(rsum_addr));
  const auto *recv =
      reinterpret_cast<const uint64_t *>(static_cast<uintptr_t>(recv_addr));
  const auto *queue =
      reinterpret_cast<const int *>(static_cast<uintptr_t>(queue_addr));
  auto *metadata =
      reinterpret_cast<int *>(static_cast<uintptr_t>(metadata_addr));
  const auto *sf_table =
      reinterpret_cast<const uint64_t *>(static_cast<uintptr_t>(sf_table_addr));
  const auto *weight_table = reinterpret_cast<const uint64_t *>(
      static_cast<uintptr_t>(weight_table_addr));
  auto *arrival =
      reinterpret_cast<unsigned int *>(static_cast<uintptr_t>(arrival_addr));
  auto *l1_token =
      reinterpret_cast<unsigned char *>(static_cast<uintptr_t>(l1_token_addr));
  auto *l1_sf = reinterpret_cast<float *>(static_cast<uintptr_t>(l1_sf_addr));
  auto *l1_weight =
      reinterpret_cast<float *>(static_cast<uintptr_t>(l1_weight_addr));

  if (lane == 0)
    mbarrier_init(barrier);
  __syncwarp();
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncwarp();

  unsigned phase = 0;
  const unsigned sm_idx = blockIdx.x;
  const unsigned stream_count = 2u;
  // Distribute expert-sorted routes across two streams on every SM.
  unsigned token_idx = sm_idx * stream_count + stream;
  int current_expert = -1;
  unsigned expert_start = 0;
  unsigned expert_end = 0;
  unsigned pool_block = 0;

  while (true) {
    // Locate the local expert and the route's offset within that expert.
    while (token_idx >= expert_end) {
      if (++current_expert >= experts_per_rank)
        break;
      pool_block +=
          (expert_end - expert_start + static_cast<unsigned>(block_m) - 1u) /
          static_cast<unsigned>(block_m);
      expert_start = expert_end;
      expert_end += static_cast<unsigned>(rsum[current_expert]);
    }
    if (current_expert >= experts_per_rank)
      break;

    // Raw regions lose convergent attributes when inlined. Avoid warp vote and
    // shuffle intrinsics here because branch-scoped warp specialization can
    // assign them an invalid convergence token. With at most eight ranks, a
    // redundant scalar scan in every lane remains inexpensive.
    unsigned remaining[8];
    // Select the source rank and queue slot with the dispatch round-robin rule.
#pragma unroll
    for (unsigned rank = 0; rank < 8; ++rank) {
      remaining[rank] =
          rank < static_cast<unsigned>(num_ranks)
              ? static_cast<unsigned>(
                    recv[rank * experts_per_rank + current_expert])
              : 0u;
    }
    unsigned slot = token_idx - expert_start;
    unsigned offset = 0;
    unsigned src_rank = 0;
    unsigned token_in_rank = 0;

    while (true) {
      unsigned num_active = 0;
      unsigned length = 0xffffffffu;
#pragma unroll
      for (unsigned rank = 0; rank < 8; ++rank) {
        const bool active =
            rank < static_cast<unsigned>(num_ranks) && remaining[rank] > 0;
        num_active += active ? 1u : 0u;
        if (active && remaining[rank] < length)
          length = remaining[rank];
      }
      if (num_active == 0)
        break;
      const unsigned round_tokens = length * num_active;
      // Consume an equal prefix from every active rank in round-robin order.
      if (slot < round_tokens) {
        const unsigned source_order = slot % num_active;
        unsigned active_order = 0;
#pragma unroll
        for (unsigned rank = 0; rank < 8; ++rank) {
          if (rank < static_cast<unsigned>(num_ranks) && remaining[rank] > 0) {
            if (active_order == source_order)
              src_rank = rank;
            ++active_order;
          }
        }
        token_in_rank = offset + slot / num_active;
        break;
      }
      // Advance past the completed round and continue with remaining routes.
      slot -= round_tokens;
      offset += length;
#pragma unroll
      for (unsigned rank = 0; rank < 8; ++rank)
        remaining[rank] -= remaining[rank] < length ? remaining[rank] : length;
    }

    // Read the original route ID from the source-rank queue.
    const unsigned q_offset = (static_cast<unsigned>(current_expert) *
                                   static_cast<unsigned>(num_ranks) +
                               src_rank) *
                                  static_cast<unsigned>(max_recv) +
                              token_in_rank;
    const int source_token_topk = queue[q_offset];
    const unsigned source_token =
        static_cast<unsigned>(source_token_topk / topk);
    const unsigned source_topk =
        static_cast<unsigned>(source_token_topk % topk);
    const unsigned token_in_expert = token_idx - expert_start;
    const unsigned pool_token =
        pool_block * static_cast<unsigned>(block_m) + token_in_expert;

    uint64_t token_base = token0;
    token_base = src_rank == 1 ? token1 : token_base;
    token_base = src_rank == 2 ? token2 : token_base;
    token_base = src_rank == 3 ? token3 : token_base;
    token_base = src_rank == 4 ? token4 : token_base;
    token_base = src_rank == 5 ? token5 : token_base;
    token_base = src_rank == 6 ? token6 : token_base;
    token_base = src_rank == 7 ? token7 : token_base;

    const auto *source = reinterpret_cast<const unsigned char *>(
                             static_cast<uintptr_t>(token_base)) +
                         static_cast<uint64_t>(source_token) * hidden;
    // Lane 0 issues the remote token TMA load.
    if (lane == 0) {
      tma_load_1d(stage, source, barrier, hidden);
    }
    __syncwarp();

    const uint64_t sf_base = sf_table[src_rank];
    const auto *remote_sf =
        reinterpret_cast<const float *>(static_cast<uintptr_t>(sf_base));
    // All lanes copy activation scales with regular loads and stores.
    for (unsigned col = lane; col < static_cast<unsigned>(num_sf); col += 32)
      l1_sf[col * static_cast<unsigned>(pool_tokens) + pool_token] =
          remote_sf[source_token * static_cast<unsigned>(num_sf) + col];
    __syncwarp();

    // Lane 0 copies the route's top-k weight.
    if (lane == 0) {
      const uint64_t weight_base = weight_table[src_rank];
      const auto *remote_weight =
          reinterpret_cast<const float *>(static_cast<uintptr_t>(weight_base));
      l1_weight[pool_token] = remote_weight[source_token_topk];
    }
    __syncwarp();

    auto *destination = l1_token + static_cast<uint64_t>(pool_token) * hidden;
    // Wait for the load, then store the token into the local L1 pool.
    if (lane == 0) {
      mbarrier_arrive_expect_tx(barrier, hidden);
      mbarrier_wait(barrier, phase);
      tma_store_1d_issue(destination, stage, hidden);
    }
    __syncwarp();

    // Publish route metadata and notify the loader after the TMA store completes.
    if (lane == 0) {
      metadata[pool_token * 3u + 0u] = static_cast<int>(src_rank);
      metadata[pool_token * 3u + 1u] = static_cast<int>(source_token);
      metadata[pool_token * 3u + 2u] = static_cast<int>(source_topk);

      tma_store_commit_wait();
      release_add_one(arrival + pool_block +
                      token_in_expert / static_cast<unsigned>(block_m));
    }
    __syncwarp();
    token_idx += static_cast<unsigned>(num_sms) * stream_count;
  }
}

extern "C" __device__ void TleD8UnifiedDispatchPull(
    const uint64_t rsum_addr, const uint64_t recv_addr,
    const uint64_t queue_addr, const uint64_t metadata_addr,
    const uint64_t sf_table_addr, const uint64_t weight_table_addr,
    const uint64_t arrival_addr, const uint64_t l1_token_addr,
    const uint64_t l1_sf_addr, const uint64_t l1_weight_addr,
    const uint64_t stage0_addr, const uint64_t state0_addr,
    const uint64_t stage1_addr, const uint64_t state1_addr,
    const uint64_t token0, const uint64_t token1, const uint64_t token2,
    const uint64_t token3, const uint64_t token4, const uint64_t token5,
    const uint64_t token6, const uint64_t token7, const int num_ranks,
    const int num_sms, const int experts_per_rank, const int max_recv,
    const int block_m, const int hidden, const int num_sf,
    const int pool_tokens, const int topk) {
  d8_unified_dispatch_pull_impl(
      rsum_addr, recv_addr, queue_addr, metadata_addr, sf_table_addr,
      weight_table_addr, arrival_addr, l1_token_addr, l1_sf_addr,
      l1_weight_addr, stage0_addr, state0_addr, stage1_addr, state1_addr,
      token0, token1, token2, token3, token4, token5, token6, token7, num_ranks,
      num_sms, experts_per_rank, max_recv, block_m, hidden, num_sf, pool_tokens,
      topk);
}
