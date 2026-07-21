// simd.hpp — NEON (arm64) accelerations for the decode-dominant kernels, with
// scalar fallbacks. M5: correctness is fixed by the scalar reference; these are
// selected only where __ARM_NEON is defined and are covered by the quant kernel
// consistency test (dot_block == dot(dequant, x)).
#pragma once
#include "core.hpp"

#if defined(__ARM_NEON)
#include <arm_neon.h>

// fp32 dot over n contiguous elements.
inline Scalar dot_f32_neon(const Scalar* a, const Scalar* b, size_t n) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a+i),    vld1q_f32(b+i));
        s1 = vfmaq_f32(s1, vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
        s2 = vfmaq_f32(s2, vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
        s3 = vfmaq_f32(s3, vld1q_f32(a+i+12), vld1q_f32(b+i+12));
    }
    for (; i + 4 <= n; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a+i), vld1q_f32(b+i));
    Scalar acc = vaddvq_f32(vaddq_f32(vaddq_f32(s0,s1), vaddq_f32(s2,s3)));
    for (; i < n; ++i) acc += a[i]*b[i];
    return acc;
}

// Accumulate 16 quant values (uint8 0..15 for Q4, 0..63 for Q6 pre-offset)
// as w = scale*val - min, times x[0..15], into `acc`. `vals` holds 16 uint8.
inline void acc16_scaled(float32x4_t& acc, uint8x16_t vals, float scale, float minus,
                         const float* x) {
    uint16x8_t lo = vmovl_u8(vget_low_u8(vals));
    uint16x8_t hi = vmovl_u8(vget_high_u8(vals));
    float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo)));
    float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo)));
    float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi)));
    float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi)));
    const float32x4_t vm = vdupq_n_f32(minus);
    float32x4_t w0 = vsubq_f32(vmulq_n_f32(f0, scale), vm);
    float32x4_t w1 = vsubq_f32(vmulq_n_f32(f1, scale), vm);
    float32x4_t w2 = vsubq_f32(vmulq_n_f32(f2, scale), vm);
    float32x4_t w3 = vsubq_f32(vmulq_n_f32(f3, scale), vm);
    acc = vfmaq_f32(acc, w0, vld1q_f32(x));
    acc = vfmaq_f32(acc, w1, vld1q_f32(x+4));
    acc = vfmaq_f32(acc, w2, vld1q_f32(x+8));
    acc = vfmaq_f32(acc, w3, vld1q_f32(x+12));
}

// signed variant (Q6/Q8: values are int8 after offset), w = scale*val, times x.
inline void acc16_signed(float32x4_t& acc, int8x16_t vals, float scale, const float* x) {
    int16x8_t lo = vmovl_s8(vget_low_s8(vals));
    int16x8_t hi = vmovl_s8(vget_high_s8(vals));
    float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
    float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
    float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
    float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
    acc = vfmaq_f32(acc, vmulq_n_f32(f0, scale), vld1q_f32(x));
    acc = vfmaq_f32(acc, vmulq_n_f32(f1, scale), vld1q_f32(x+4));
    acc = vfmaq_f32(acc, vmulq_n_f32(f2, scale), vld1q_f32(x+8));
    acc = vfmaq_f32(acc, vmulq_n_f32(f3, scale), vld1q_f32(x+12));
}
#endif // __ARM_NEON
