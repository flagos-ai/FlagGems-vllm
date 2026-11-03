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

// MODEL1 sparse decode core for MetaX C550 (xcore1000), linked into the
// Triton kernel _attention (flash_mla_with_kvcache_model1.py) and inlined
// there.
//
// One call per thread of a 512-thread program (8 waves of 64 lanes) handles
// one partition of the tile schedule: every request in it, all 64 heads.
// - A 16-token tile is loaded by rows of 32 threads, each taking two values of
//   every 64-wide group (two e4m3 bytes, dequantized to BF16, or two BF16 RoPE
//   values), and stored to shared memory.
// - Waves w and w + 4 compute S = Q K^T for heads 16 (w % 4) .. + 15 over
//   their halves of the head dim (Q stays in registers as MMA A fragments)
//   and add each other's partial S through shared memory.
// - They compute O for those heads and columns 256 (w / 4) .. + 255, taking P
//   straight from the S accumulator (the MMA C and A fragments share one
//   layout: lane % 16 is the row, lane / 16 the group of four columns) and V
//   from the same K tile.
// Tiles run as one software pipeline across the partition's requests (see
// there). Parameters and the tile cursor are wave-uniform and kept in scalar
// registers (readfirstlane, uniform branches).
#include <maca.h>
#include <math.h>

typedef float float4_ __attribute__((ext_vector_type(4)));
typedef float float2_ __attribute__((ext_vector_type(2)));
typedef unsigned int uint2_ __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_ __attribute__((ext_vector_type(4)));
typedef int int4_ __attribute__((ext_vector_type(4)));

namespace {

constexpr int HEADS = 64;
constexpr int DIM = 512;
constexpr int NOPE = 448;
constexpr int GROUPS = NOPE / 64;
constexpr int TOKEN_BYTES = 576;
constexpr int BLOCK = 16;
constexpr int Q_STEPS = DIM / 16 / 2;  // k-steps of a wave's half of Q K^T
constexpr int ACC_TILES = 16;
// Between the MMAs of Q K^T: the dequantization of groups 0 .. DEQUANT_QK - 1
// after k-steps DEQUANT_FROM, + 2, ..., and the scores after k-step SCORES_AT;
// the other groups are dequantized between the MMAs of P V.
constexpr int DEQUANT_FROM = 2, DEQUANT_QK = 5, SCORES_AT = 8;
static_assert(DEQUANT_FROM + 2 * (DEQUANT_QK - 1) < Q_STEPS && GROUPS - DEQUANT_QK <= 4, "");
constexpr float LN2 = 0.6931471805599453f;
constexpr float LOG2E = 1.4426950408889634f;
// Tile info: request << 3 | flags; -1 past the partition's last tile.
constexpr int INFO_LAST = 1, INFO_WHOLE = 2, INFO_MAIN = 4;

__device__ __forceinline__ float4_ mma(uint2_ a, uint2_ b, float4_ c) {
    return __builtin_mxc_mma_16x16x16bf16(b, a, c);
}

// Barrier on shared-memory accesses only (outstanding global loads go on).
__device__ __forceinline__ void block_sync() {
    __builtin_mxc_arrive_bsmcnt(0);
    __builtin_mxc_barrier_ex(2);
}

// Bytes of {hi, lo} picked by the selector's bytes, lowest result byte first:
// 0-3 are lo's bytes, 4-7 hi's.
__device__ __forceinline__ unsigned perm(unsigned hi, unsigned lo, unsigned selector) {
    return __builtin_mxc_byte_perm(hi, lo, selector);
}

// Two FP32 values as packed BF16 (lo, hi), round to nearest even.
__device__ __forceinline__ unsigned bf16_pack(float lo, float hi) {
    auto v = __builtin_mxc_cvt_pk_f32tobf16((float2_){lo, hi});
    return *(unsigned *)&v;
}

// The same, rounding half up (for softmax weights).
__device__ __forceinline__ unsigned bf16_pack_fast(float lo, float hi) {
    return perm(__float_as_uint(hi) + 0x8000u, __float_as_uint(lo) + 0x8000u, 0x07060302);
}

// 2^x by the hardware instruction; results below 2^-126 are flushed to zero.
__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("exp_f32 %0, %1" : "=r"(y) : "r"(x));
    return y;
}

// Two e4m3 bytes (low 16 bits of w) times 2^(e - 127) as packed BF16, exactly.
// A byte's low 7 bits at bits 20..26 of an FP32 word are its magnitude times
// 2^-120 (FP32 subnormals line up with e4m3 ones); the products by powers of
// two are exact BF16 values, so their high halves are the result, and the
// signs go to bits 15 and 31.
__device__ __forceinline__ unsigned dequant_pair(unsigned w, unsigned e) {
    float2_ x = {__uint_as_float((w << 20) & 0x07F00000u), __uint_as_float((w << 12) & 0x07F00000u)};
    x = __builtin_mxc_pk_mul_f32(x, (float2_){0x1p120f, 0x1p120f});
    const float scale = __uint_as_float(e << 23);
    x = __builtin_mxc_pk_mul_f32(x, (float2_){scale, scale});
    const unsigned magnitude = perm(__float_as_uint(x[1]), __float_as_uint(x[0]), 0x07060302);
    return (perm(w, w, 0x01000000) & 0x80008000u) | magnitude;
}

// Global-address-space pointers (their loads are not waited for by the
// shared-memory waits).
#define GLOBAL __attribute__((address_space(1)))
template <typename T> __device__ __forceinline__ const GLOBAL T *global(const void *p) {
    return (const GLOBAL T *)(__UINTPTR_TYPE__)p;
}

// The same value in every lane, in a scalar register (Triton passes the
// arguments of the extern call per lane).
__device__ __forceinline__ int uniform(int x) { return __builtin_mxc_readfirstlane(x); }
__device__ __forceinline__ float uniform(float x) { return __int_as_float(uniform(__float_as_int(x))); }
__device__ __forceinline__ long long uniform(long long x) {
    const unsigned lo = (unsigned)uniform((int)x), hi = (unsigned)uniform((int)(x >> 32));
    return (long long)((unsigned long long)hi << 32 | lo);
}

struct Caches {
    const unsigned char *cache;
    long long page_stride;
    int page_shift;
    const unsigned char *extra_cache;
    long long extra_page_stride;
    int extra_page_shift;
};

// What a thread holds of one token: two e4m3 bytes of each NoPE group, two
// RoPE values and the token's scale bytes.
struct Loaded {
    unsigned nope[GROUPS];
    unsigned rope;
    unsigned scale_lo, scale_hi;
    int valid;
};

// Thread (n, sub) loads values 64 k + 2 sub, + 1 of token n of the tile. An
// invalid token (-1) reads token 0 of the tile's cache; masked() zeroes it
// where it is used (a select on a loaded value would wait for the load).
__device__ __forceinline__ Loaded load_token(const Caches &c, int token, int info, int sub) {
    Loaded v;
    v.valid = token >= 0;
    const unsigned safe = v.valid ? (unsigned)token : 0u;
    auto from = [&](const unsigned char *cache, long long page_stride, int page_shift) {
        const unsigned shift = (unsigned)page_shift & 31u;
        const unsigned page = safe >> shift, slot = safe & ((1u << shift) - 1u);
        const unsigned char *page_base = cache + (unsigned long long)page * (unsigned)page_stride;
        const unsigned char *data = page_base + (slot * TOKEN_BYTES + 2 * sub);
#pragma unroll
        for (int k = 0; k < GROUPS; ++k)
            v.nope[k] = *global<unsigned short>(data + 64 * k);
        v.rope = *global<unsigned>(data + NOPE + 2 * sub);
        const uint2_ scales = *global<uint2_>(page_base + ((TOKEN_BYTES << shift) + slot * 8));
        v.scale_lo = scales.x;
        v.scale_hi = scales.y;
    };
    // The cache is the same for the whole tile: a uniform branch keeps its
    // parameters scalar (the empty asm keeps the arms from being merged into
    // selects).
    if (uniform(info) & INFO_MAIN) {
        asm volatile("");
        from(c.cache, c.page_stride, c.page_shift);
    } else {
        asm volatile("");
        from(c.extra_cache, c.extra_page_stride, c.extra_page_shift);
    }
    return v;
}

// An invalid token as zeros (zero scales dequantize its NoPE values to zero);
// it is masked in S, and P = 0 times zero keeps O finite.
__device__ __forceinline__ void masked(Loaded &v) {
    v.rope = v.valid ? v.rope : 0u;
    v.scale_lo = v.valid ? v.scale_lo : 0u;
    v.scale_hi = v.valid ? v.scale_hi : 0u;
}

__device__ __forceinline__ unsigned dequant_group(const Loaded &v, int k) {
    const unsigned e = ((k < 4 ? v.scale_lo : v.scale_hi) >> (8 * (k & 3))) & 255u;
    return dequant_pair(v.nope[k], e);
}

// A thread's part of a tile as BF16 words: values 64 k + 2 sub, + 1 of token n.
struct Words {
    unsigned w[GROUPS + 1];
};

}  // namespace

extern "C" __device__ __attribute__((used, always_inline)) float mla_partition(
    long long q_ptr, long long cache_ptr, long long page_stride, int page_shift,
    long long extra_cache_ptr, long long extra_page_stride, int extra_page_shift,
    long long indices_ptr, int topk, long long extra_indices_ptr, int extra_topk,
    long long lengths_ptr, long long extra_lengths_ptr, int has_extra,
    long long metadata_ptr, long long partial_ptr, long long partial_lse_ptr,
    long long out_ptr, long long lse_ptr, long long sink_ptr, float scale_log2, int batch, int part) {
    q_ptr = uniform(q_ptr);
    cache_ptr = uniform(cache_ptr);
    page_stride = uniform(page_stride);
    page_shift = uniform(page_shift);
    extra_cache_ptr = uniform(extra_cache_ptr);
    extra_page_stride = uniform(extra_page_stride);
    extra_page_shift = uniform(extra_page_shift);
    indices_ptr = uniform(indices_ptr);
    topk = uniform(topk);
    extra_indices_ptr = uniform(extra_indices_ptr);
    extra_topk = uniform(extra_topk);
    lengths_ptr = uniform(lengths_ptr);
    extra_lengths_ptr = uniform(extra_lengths_ptr);
    has_extra = uniform(has_extra);
    metadata_ptr = uniform(metadata_ptr);
    partial_ptr = uniform(partial_ptr);
    partial_lse_ptr = uniform(partial_lse_ptr);
    out_ptr = uniform(out_ptr);
    lse_ptr = uniform(lse_ptr);
    sink_ptr = uniform(sink_ptr);
    scale_log2 = uniform(scale_log2);
    batch = uniform(batch);
    part = uniform(part);

    // Three K tiles (see the pipeline below); 16-byte blocks of row n are XOR
    // swizzled by (n & 7) << 3 elements within each 64-element group, so the
    // fragment reads and the stores are free of bank conflicts. Partial S,
    // double-buffered: lane t's half is read by lane t ^ 256.
    __shared__ unsigned short k_tiles[3][BLOCK * DIM];
    __shared__ float4_ s_exchange[2][512];

    const int t = threadIdx.x;
    const int lane = t & 63, wave = t >> 6;
    const int row = lane & 15, group = lane >> 4;
    const int head = 16 * (wave & 3) + row;  // this lane's row of S and O
    const int column0 = 256 * (wave >> 2);   // this wave's half of the head dim
    const int n = t >> 5, sub = t & 31;      // loader role

    // See _schedule.
    const int *metadata = (const int *)metadata_ptr + part * 8;
    const int first = metadata[0];
    if (first >= batch)
        return 0.f;
    const int first_tile = metadata[1], last = metadata[2], last_tile = metadata[3];
    const int first_split = metadata[4], last_split = metadata[7];
    if (last < first)
        return 0.f;
    const int *lengths = (const int *)lengths_ptr;
    const int *extra_lengths = (const int *)extra_lengths_ptr;
    const int *main_ids = (const int *)indices_ptr;
    const int *extra_ids = (const int *)extra_indices_ptr;
    const float *sink = (const float *)sink_ptr;
    const Caches caches = {(const unsigned char *)cache_ptr, page_stride, page_shift,
                           (const unsigned char *)extra_cache_ptr, extra_page_stride, extra_page_shift};

    // Token-id stage: a cursor over the partition's tiles, request by request:
    // the request, its tile, its end tile (exclusive), its number of main
    // tiles, its lengths, and whether the partition holds all of it.
    int request, tile, end, main_tiles, length, extra_length, whole;
    int next_length = 0, next_extra_length = 0;
    auto enter = [&](int b, int from) {
        request = b;
        tile = from;
        main_tiles = (max(length, 1) + BLOCK - 1) / BLOCK;
        const int tiles = main_tiles + (has_extra ? (max(extra_length, 0) + BLOCK - 1) / BLOCK : 0);
        end = b == last ? last_tile : tiles;
        whole = from == 0 && end == tiles;
        if (b < last) {
            next_length = lengths[b + 1];
            next_extra_length = has_extra ? extra_lengths[b + 1] : 0;
        }
    };
    // Info of the cursor's tile, this thread's token id (read from position 0
    // when out of range; combined with in_range where used), this lane's four
    // tokens 4 group .. + 3 (its columns of S: their ids and how many are in
    // range); then advance.
    auto step = [&](int &token, bool &in_range, int4_ &lane_ids, unsigned &lane_in) {
        request = uniform(request);
        tile = uniform(tile);
        end = uniform(end);
        main_tiles = uniform(main_tiles);
        length = uniform(length);
        extra_length = uniform(extra_length);
        whole = uniform(whole);
        if (request < 0) {
            token = -1;
            in_range = false;
            lane_ids = (int4_){-1, -1, -1, -1};
            lane_in = 0u;
            return -1;
        }
        const bool is_main = tile < main_tiles;
        int tile_base, limit;  // the tile's first position; how many of its tokens are in range
        const int *ids;
        if (is_main) {
            asm volatile("");
            tile_base = tile * BLOCK;
            limit = min(length, topk) - tile_base;
            ids = main_ids + (long long)request * topk + tile_base;
        } else {
            asm volatile("");
            tile_base = (tile - main_tiles) * BLOCK;
            limit = min(extra_length, extra_topk) - tile_base;
            ids = extra_ids + (long long)request * extra_topk + tile_base;
        }
        in_range = n < limit;
        lane_in = (unsigned)(limit - 4 * group);
        token = *global<int>(ids + (in_range ? n : 0));
        lane_ids = *global<int4_>(ids + (4 * group < limit ? 4 * group : 0));
        const int info = request << 3 | (is_main ? INFO_MAIN : 0) | (whole ? INFO_WHOLE : 0) |
                         (tile == end - 1 ? INFO_LAST : 0);
        if (++tile == end) {
            if (request == last) {
                request = -1;
            } else {
                length = next_length;
                extra_length = next_extra_length;
                enter(request + 1, 0);
            }
        }
        return info;
    };
    // Which of this lane's four tokens are valid: in range and id >= 0.
    auto valid_bits = [](int4_ ids, unsigned in) {
        const int count = (int)in;
        return (unsigned)((ids.x >= 0) & (count > 0)) | (unsigned)((ids.y >= 0) & (count > 1)) << 1 |
               (unsigned)((ids.z >= 0) & (count > 2)) << 2 | (unsigned)((ids.w >= 0) & (count > 3)) << 3;
    };

    // Q as A fragments, this wave's half of the head dim. The order of the
    // head dim within Q K^T is free as long as K follows it: k-step s = 2 j + h
    // gives lane group g dims column0 + 32 j + 8 g + 4 h .. + 3, so a lane
    // loads its Q with 16-byte loads. Kept as packed pairs, joined into the
    // operand vector at each use.
    unsigned qx[Q_STEPS], qy[Q_STEPS];
    auto load_q = [&](int b) {
        const GLOBAL uint4_ *q_row =
            global<uint4_>((const unsigned short *)q_ptr + ((long long)b * HEADS + head) * DIM + column0 + group * 8);
#pragma unroll
        for (int j = 0; j < Q_STEPS / 2; ++j) {
            const uint4_ v = q_row[j * 4];
            qx[2 * j] = v.x;
            qy[2 * j] = v.y;
            qx[2 * j + 1] = v.z;
            qy[2 * j + 1] = v.w;
        }
    };
    load_q(first);
    length = metadata[5];
    extra_length = has_extra ? metadata[6] : 0;
    enter(first, first_tile);
    const float sink_value = sink_ptr ? sink[head] : -INFINITY;

    float m = -INFINITY, l = 0.f;
    float4_ acc[ACC_TILES];
#pragma unroll
    for (int j = 0; j < ACC_TILES; ++j)
        acc[j] = (float4_){0.f, 0.f, 0.f, 0.f};

    // Fixed parts of this thread's shared-memory addresses (the swizzle only
    // touches bits within a 64-element group).
    const int store_base = n * DIM + ((2 * sub) ^ ((n & 7) << 3));
    const int swizzle = (row & 7) << 3;
    // K of k-steps 2 j, 2 j + 1 at dims column0 + 64 (j >> 1) + qk_col[j & 1].
    const int qk_col[2] = {(group * 8) ^ swizzle, (group * 8 + 32) ^ swizzle};
    // V of P V (see there).
    int v_base[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        v_base[i] = (4 * group + i) * DIM + ((16 * (row & 3) + 4 * (row >> 2)) ^ (((4 * group + i) & 7) << 3)) + column0;

    auto store_words = [&](const Words &w, int buffer) {
#pragma unroll
        for (int k = 0; k <= GROUPS; ++k)
            *(unsigned *)(k_tiles[buffer] + store_base + 64 * k) = w.w[k];
    };
    auto dequant_all = [&](Loaded v) {
        masked(v);
        Words w;
#pragma unroll
        for (int k = 0; k < GROUPS; ++k)
            w.w[k] = dequant_group(v, k);
        w.w[GROUPS] = v.rope;
        return w;
    };
    // This wave's half of S = Q K^T (16 heads x 16 tokens) of the tile in
    // buffer; between(ks) runs after the MMAs of k-steps ks, ks + 1.
    auto qk = [&](int buffer, auto between) {
        const unsigned short *k_row = k_tiles[buffer] + row * DIM + column0;
        float4_ s0 = {0.f, 0.f, 0.f, 0.f}, s1 = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ks = 0; ks < Q_STEPS; ks += 2) {
            const uint4_ kf = *(const uint4_ *)(k_row + qk_col[(ks >> 1) & 1] + 64 * (ks >> 2));
            s0 = mma((uint2_){qx[ks], qy[ks]}, (uint2_){kf.x, kf.y}, s0);
            s1 = mma((uint2_){qx[ks + 1], qy[ks + 1]}, (uint2_){kf.z, kf.w}, s1);
            between(ks);
        }
        return s0 + s1;
    };

    // Software pipeline over the partition's tiles, across requests. Iteration
    // i stores tile i + 1, then after the barrier: Q K^T of tile i + 1 with the
    // dequantization of tile i + 2 and the scores of tile i between its MMAs,
    // the load of tile i + 3, the ids of tile i + 4, then the softmax and P V
    // of tile i and, if it ends a request, the output. Three K buffers: a
    // buffer is rewritten two barriers after its last reader. The next
    // request's Q is loaded right after the last Q K^T of the current one.
    int token0, token1, token2, token3;
    bool in0, in1, in2, in3;
    int4_ ids0, ids1, ids2, ids3;
    unsigned lane_in0, lane_in1, lane_in2, lane_in3;
    int info = step(token0, in0, ids0, lane_in0);
    int info1 = step(token1, in1, ids1, lane_in1);
    int info2 = step(token2, in2, ids2, lane_in2);
    int info3 = step(token3, in3, ids3, lane_in3);

    int buffer = 0, slot = 0;
    const Loaded first_data = load_token(caches, in0 ? token0 : -1, info, sub);
    Loaded next_data = load_token(caches, in1 ? token1 : -1, info1, sub);
    Words words = dequant_all(first_data);
    unsigned valid0 = valid_bits(ids0, lane_in0), valid1 = valid_bits(ids1, lane_in1);
    store_words(words, 0);
    block_sync();
    float4_ s = qk(0, [](int) {});
    s_exchange[0][t] = s;
    words = dequant_all(next_data);
    if ((info & INFO_LAST) && info1 >= 0)
        load_q(info1 >> 3);
    next_data = load_token(caches, in2 ? token2 : -1, info2, sub);
    unsigned valid2 = valid_bits(ids2, lane_in2);

    while (info >= 0) {
        const int next_buffer = buffer == 2 ? 0 : buffer + 1;
        store_words(words, next_buffer);  // tile i + 1 (unused past the last tile)
        block_sync();

        s += s_exchange[slot][t ^ 256];  // S of tile i: both halves
        masked(next_data);
        Words next_words;
        next_words.w[GROUPS] = next_data.rope;
        float scores[4];
        float lane_max = -INFINITY;
        const float4_ s_next = qk(next_buffer, [&](int ks) {
            if (ks >= DEQUANT_FROM && ks < DEQUANT_FROM + 2 * DEQUANT_QK)
                next_words.w[(ks - DEQUANT_FROM) / 2] = dequant_group(next_data, (ks - DEQUANT_FROM) / 2);
            if (ks == SCORES_AT) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    scores[i] = (valid0 >> i) & 1 ? s[i] * scale_log2 : -INFINITY;
                    lane_max = fmaxf(lane_max, scores[i]);
                }
            }
        });
        if ((info1 & INFO_LAST) && info2 >= 0)
            load_q(info2 >> 3);
        const Loaded tail = next_data;  // the groups dequantized during P V
        next_data = load_token(caches, in3 ? token3 : -1, info3, sub);
        const unsigned valid3 = valid_bits(ids3, lane_in3);
        int token4;
        bool in4;
        int4_ ids4;
        unsigned lane_in4;
        const int info4 = step(token4, in4, ids4, lane_in4);

        // Softmax of tile i. The running max m (the same for a row's four lanes)
        // is raised only when a score exceeds it by more than 8 (so P < 2^8);
        // only then are the row maxima reduced across lanes and O rescaled. The
        // row sum l is per lane, reduced at the end.
        if (__any(lane_max > m + 8.f)) {
            float tile_max = fmaxf(lane_max, __shfl_xor(lane_max, 16, 64));
            tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 32, 64));
            const bool raise = tile_max > m + 8.f;
            const float alpha = raise ? exp2f(m - tile_max) : 1.f;
            m = raise ? tile_max : m;
            l *= alpha;
#pragma unroll
            for (int j = 0; j < ACC_TILES; ++j)
                acc[j] *= alpha;
        }
        const float m_safe = m == -INFINITY ? 0.f : m;
        float p[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            p[i] = fast_exp2(scores[i] - m_safe);
            l += p[i];
        }

        // O += P V for 16 heads x 256 columns. Per 64-column block, lane
        // (c = row, g = group) reads tokens 4 g .. 4 g + 3 at dims 4 c' .. 4 c' + 3,
        // c' = 4 (c % 4) + c / 4, and transposes them into four fragments;
        // fragment f feeds the MMA for dims 4 c' + f, so acc[4 q + f][i] holds
        // dim column0 + 64 q + 16 i + 4 g + f.
        const unsigned short *k_tile = k_tiles[buffer];
        const uint2_ p_frag = {bf16_pack_fast(p[0], p[1]), bf16_pack_fast(p[2], p[3])};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint2_ x0 = *(const uint2_ *)(k_tile + v_base[0] + 64 * q);
            const uint2_ x1 = *(const uint2_ *)(k_tile + v_base[1] + 64 * q);
            const uint2_ x2 = *(const uint2_ *)(k_tile + v_base[2] + 64 * q);
            const uint2_ x3 = *(const uint2_ *)(k_tile + v_base[3] + 64 * q);
            const uint2_ f0 = {perm(x1.x, x0.x, 0x05040100), perm(x3.x, x2.x, 0x05040100)};
            const uint2_ f1 = {perm(x1.x, x0.x, 0x07060302), perm(x3.x, x2.x, 0x07060302)};
            const uint2_ f2 = {perm(x1.y, x0.y, 0x05040100), perm(x3.y, x2.y, 0x05040100)};
            const uint2_ f3 = {perm(x1.y, x0.y, 0x07060302), perm(x3.y, x2.y, 0x07060302)};
            acc[4 * q] = mma(p_frag, f0, acc[4 * q]);
            acc[4 * q + 1] = mma(p_frag, f1, acc[4 * q + 1]);
            acc[4 * q + 2] = mma(p_frag, f2, acc[4 * q + 2]);
            acc[4 * q + 3] = mma(p_frag, f3, acc[4 * q + 3]);
            if (DEQUANT_QK + q < GROUPS)
                next_words.w[DEQUANT_QK + q] = dequant_group(tail, DEQUANT_QK + q);
        }

        // Output of the request that tile i ends: a whole request to out (with
        // the sink) and lse (natural log); a piece of a split one to partial
        // and partial_lse (log2), merged by _combine. Both outputs are BF16.
        if (info & INFO_LAST) {
            const int b = info >> 3;
            const bool is_whole = info & INFO_WHOLE;
            l += __shfl_xor(l, 16, 64);
            l += __shfl_xor(l, 32, 64);
            const bool nonempty = l > 0.f;
            float factor = nonempty ? 1.f / l : 0.f;
            const float log2_sum = nonempty ? m + log2f(l) : -INFINITY;
            float lse_value = log2_sum;
            long long out_index;
            if (is_whole) {
                const float natural = nonempty ? log2_sum * LN2 : INFINITY;
                if (sink_ptr)
                    factor *= sink_value == INFINITY ? 0.f : 1.f / (1.f + exp2f((sink_value - natural) * LOG2E));
                lse_value = natural;
                out_index = (long long)b * HEADS + head;
            } else {
                out_index = (long long)(b == first ? first_split : last_split) * HEADS + head;
            }
            if (group == 0 && wave < 4)
                ((float *)(is_whole ? lse_ptr : partial_lse_ptr))[out_index] = lse_value;
            unsigned short *out_row = (unsigned short *)(is_whole ? out_ptr : partial_ptr) + out_index * DIM + column0;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                // Lanes g and g ^ 1 swap halves, so each stores 8 contiguous
                // values: for i = 2 p, 2 p + 1, an even lane dims 16 (2 p) +
                // 8 (g / 2) .. + 7, an odd lane 16 (2 p + 1) + 8 (g / 2) .. + 7.
                float4_ v[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    v[i] = (float4_){acc[4 * q][i], acc[4 * q + 1][i], acc[4 * q + 2][i], acc[4 * q + 3][i]} * factor;
#pragma unroll
                for (int i = 0; i < 4; i += 2) {
                    const bool odd = group & 1;
                    const uint2_ w0 = {bf16_pack(v[i][0], v[i][1]), bf16_pack(v[i][2], v[i][3])};
                    const uint2_ w1 = {bf16_pack(v[i + 1][0], v[i + 1][1]), bf16_pack(v[i + 1][2], v[i + 1][3])};
                    const uint2_ send = odd ? w0 : w1, mine = odd ? w1 : w0;
                    const uint2_ got = {(unsigned)__shfl_xor((int)send.x, 16, 64),
                                        (unsigned)__shfl_xor((int)send.y, 16, 64)};
                    *(uint4_ *)(out_row + 64 * q + 16 * (i + odd) + 8 * (group >> 1)) =
                        odd ? (uint4_){got.x, got.y, mine.x, mine.y} : (uint4_){mine.x, mine.y, got.x, got.y};
                }
                asm volatile("" ::: "memory");
            }
            m = -INFINITY;
            l = 0.f;
#pragma unroll
            for (int j = 0; j < ACC_TILES; ++j)
                acc[j] = (float4_){0.f, 0.f, 0.f, 0.f};
        }

        // The partial S of tile i + 1 for the partner wave, stored last: the
        // store waits for the MMAs of Q K^T.
        s_exchange[slot ^ 1][t] = s_next;
        s = s_next;
        words = next_words;
        info = info1;
        info1 = info2;
        info2 = info3;
        info3 = info4;
        token3 = token4;
        in3 = in4;
        ids3 = ids4;
        lane_in3 = lane_in4;
        valid0 = valid1;
        valid1 = valid2;
        valid2 = valid3;
        slot ^= 1;
        buffer = next_buffer;
    }
    return 0.f;
}
