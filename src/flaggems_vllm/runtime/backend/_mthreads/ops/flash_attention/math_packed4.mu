// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <musa_runtime.h>
#include <musa_burst.h>

union FloatQuad {
    struct { unsigned long long lo, hi; } bits;
    float4 values;
};

#define DEVICE_INLINE extern "C" __device__ __attribute__((used, always_inline))

DEVICE_INLINE unsigned long long fa_exp2_packed4_lo(unsigned long long lo, unsigned long long hi) {
    FloatQuad x, y;
    x.bits = {lo, hi};
    y.values = __exp2f(x.values);
    return y.bits.lo;
}

DEVICE_INLINE unsigned long long fa_exp2_packed4_hi(unsigned long long lo, unsigned long long hi) {
    FloatQuad x, y;
    x.bits = {lo, hi};
    y.values = __exp2f(x.values);
    return y.bits.hi;
}

DEVICE_INLINE unsigned long long fa_mul_packed4_lo(unsigned long long lo, unsigned long long hi, float alpha) {
    FloatQuad x, y;
    x.bits = {lo, hi};
    y.values = mul(x.values, make_float4(alpha, alpha, alpha, alpha));
    return y.bits.lo;
}

DEVICE_INLINE unsigned long long fa_mul_packed4_hi(unsigned long long lo, unsigned long long hi, float alpha) {
    FloatQuad x, y;
    x.bits = {lo, hi};
    y.values = mul(x.values, make_float4(alpha, alpha, alpha, alpha));
    return y.bits.hi;
}
