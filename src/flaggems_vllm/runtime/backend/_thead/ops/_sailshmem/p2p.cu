// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Build only the on-node path. The vendor transfer library also contains
// off-node code that requires printf lowering unavailable to this JIT.
#ifndef __CUDA_ARCH__
// The sailSHMEM headers use this guard for PPU bulk-copy support.
#define __CUDA_ARCH__ 100
#endif
#include <device/nvshmem_defines.h>

extern __constant__ nvshmemi_version_t nvshmemi_device_lib_version_d;

__device__ static void set_peer_signal(uint64_t signal, uint64_t epoch, int peer) {
    __threadfence_system();
    auto *remote_signal = static_cast<volatile uint64_t *>(
        nvshmem_ptr(reinterpret_cast<void *>(signal), peer));
    *remote_signal = epoch;
}

extern "C" __device__ int mega_put_signal(uint64_t dest, uint64_t source,
                                          uint64_t bytes, uint64_t ready,
                                          uint64_t epoch, int peer) {
    auto *remote = nvshmem_ptr(reinterpret_cast<void *>(dest), peer);
    nvshmemi_memcpy_threadgroup_bulk<nvshmemi_threadgroup_block, false, true>(
        remote, reinterpret_cast<const void *>(source), bytes);
    __syncthreads();
    if (threadIdx.x == 0) {
        set_peer_signal(ready, epoch, peer);
    }
    __syncthreads();
    return 0;
}

extern "C" __device__ int mega_wait(uint64_t signal, uint64_t epoch) {
    if (threadIdx.x == 0) {
        nvshmem_signal_wait_until(reinterpret_cast<uint64_t *>(signal),
                                  NVSHMEM_CMP_GE, epoch);
    }
    __syncthreads();
    return 0;
}

extern "C" __device__ int mega_signal(uint64_t signal, uint64_t epoch, int peer) {
    // Registration queries this symbol. Keep it in the linked module.
    if (reinterpret_cast<volatile nvshmemi_version_t *>(
            &nvshmemi_device_lib_version_d)->major == 0) {
        return -1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        set_peer_signal(signal, epoch, peer);
    }
    return 0;
}

template <int World>
__device__ static int broadcast_tile(uint64_t dest, uint64_t source, uint64_t bytes,
                                     uint64_t ready, uint64_t epoch, int rank,
                                     int tile) {
    if (bytes % RMT_BULK_V4_SIZE != 0 || ((dest | source) & 3) != 0) {
        for (int step = 1; step < World; ++step) {
            mega_put_signal(dest, source, bytes, ready, epoch,
                            (rank + step) % World);
        }
        return 0;
    }

    uint8_t *targets[World - 1];
    int peers[World - 1];
#pragma unroll
    for (int i = 0; i < World - 1; ++i) {
        peers[i] = (rank + 1 + (i + tile) % (World - 1)) % World;
        targets[i] = static_cast<uint8_t *>(
            nvshmem_ptr(reinterpret_cast<void *>(dest), peers[i]));
    }
    const auto *input = reinterpret_cast<const uint8_t *>(source);
    const int warp = threadIdx.x / warpSize;
    const int warps = blockDim.x / warpSize;
    // Each warp forwards one local bulk load to all peers.
    for (size_t chunk = warp; chunk < bytes / RMT_BULK_V4_SIZE; chunk += warps) {
        const size_t offset = chunk * RMT_BULK_V4_SIZE;
        const uint4 value = __ppu_global_ldcv_bulk_b32x4(input + offset);
#pragma unroll
        for (int i = 0; i < World - 1; ++i) {
            __ppu_remote_store_bulk_ncp_b32x4(targets[i] + offset, value);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
#pragma unroll
        for (int i = 0; i < World - 1; ++i) {
            auto *flag = static_cast<volatile uint64_t *>(
                nvshmem_ptr(reinterpret_cast<void *>(ready), peers[i]));
            *flag = epoch;
        }
    }
    __syncthreads();
    return 0;
}

extern "C" __device__ int mega_broadcast(uint64_t dest, uint64_t source,
                                         uint64_t bytes, uint64_t ready,
                                         uint64_t epoch, int rank, int tile,
                                         int world) {
    if (world == 4) {
        return broadcast_tile<4>(dest, source, bytes, ready, epoch, rank, tile);
    }
    return broadcast_tile<8>(dest, source, bytes, ready, epoch, rank, tile);
}
