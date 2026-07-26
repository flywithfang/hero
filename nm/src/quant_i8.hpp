// quant_i8.hpp — int8 activation path (M5 "integer-dot"): quantize the
// activation vector once per matvec (Q8_0 for 32-block weights, Q8_K for
// 256-superblock weights), then do integer dot products per output row —
// exactly llama.cpp's CPU strategy.
//
// The scalar kernels are line-faithful clones of ggml's *_generic reference
// implementations (ggml/src/ggml-cpu/quants.c) including accumulation
// structure, so they are bit-comparable against the exported
// ggml_vec_dot_*_generic symbols (verified in the scratchpad harness).
// The NEON (dotprod) variants change only float summation order — the integer
// arithmetic is exact in both.
#pragma once
#include "quant.hpp"
#include <cstdlib>

#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#endif

// ---- activation block formats ---------------------------------------------
#pragma pack(push, 1)
struct block_q8_K {                    // ggml: fp32 delta + 256 quants + bsums
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};
#pragma pack(pop)
static_assert(sizeof(block_q8_K) == 4 + 256 + 32);

// ggml's nearest_int: round-to-nearest-even via the 2^23 magic number.
inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i; std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// quantize_row_q8_K_ref clone (note the NEGATIVE iscale = -127/max trick).
inline void quantize_row_q8_K(const float* x, block_q8_K* y, size_t k) {
    const size_t nb = k / QK_K;
    for (size_t i = 0; i < nb; ++i) {
        float max = 0, amax = 0;
        for (size_t j = 0; j < QK_K; ++j) {
            float ax = std::fabs(x[j]);
            if (ax > amax) { amax = ax; max = x[j]; }
        }
        if (amax == 0.f) {
            y[i].d = 0; std::memset(y[i].qs, 0, QK_K);
            std::memset(y[i].bsums, 0, sizeof y[i].bsums);
            x += QK_K; continue;
        }
        const float iscale = -127.f / max;
        for (size_t j = 0; j < QK_K; ++j)
            y[i].qs[j] = int8_t(std::min(127, nearest_int(iscale * x[j])));
        for (size_t j = 0; j < QK_K / 16; ++j) {
            int sum = 0;
            for (size_t l = 0; l < 16; ++l) sum += y[i].qs[j * 16 + l];
            y[i].bsums[j] = int16_t(sum);
        }
        y[i].d = 1 / iscale;
        x += QK_K;
    }
}

// quantize_row_q8_0_ref clone (roundf, d = amax/127 as fp16).
inline void quantize_row_q8_0(const float* x, block_q8_0* y, size_t k) {
    const size_t nb = k / QK8_0;
    for (size_t i = 0; i < nb; ++i) {
        float amax = 0;
        for (size_t j = 0; j < QK8_0; ++j) amax = std::max(amax, std::fabs(x[i*QK8_0+j]));
        const float d = amax / 127.f;
        const float id = d ? 1.f / d : 0.f;
        y[i].d = fp32_to_fp16(d);
        for (size_t j = 0; j < QK8_0; ++j)
            y[i].qs[j] = int8_t(std::roundf(x[i*QK8_0+j] * id));
    }
}

// ---- integer dot kernels: one weight row (nb blocks) · quantized activation -

// ggml_vec_dot_q8_0_q8_0_generic clone.
inline float vec_dot_q8_0_q8_0(size_t n, const block_q8_0* x, const block_q8_0* y) {
    const size_t nb = n / QK8_0;
    float sumf = 0;
    for (size_t ib = 0; ib < nb; ++ib) {
#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t acc = vdotq_s32(vdupq_n_s32(0), vld1q_s8(x[ib].qs), vld1q_s8(y[ib].qs));
        acc = vdotq_s32(acc, vld1q_s8(x[ib].qs + 16), vld1q_s8(y[ib].qs + 16));
        int sumi = vaddvq_s32(acc);
#else
        int sumi = 0;
        for (size_t j = 0; j < QK8_0; ++j) sumi += x[ib].qs[j] * y[ib].qs[j];
#endif
        sumf += sumi * (fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
    }
    return sumf;
}

// ggml_vec_dot_q4_0_q8_0_generic clone.
inline float vec_dot_q4_0_q8_0(size_t n, const block_q4_0* x, const block_q8_0* y) {
    const size_t nb = n / QK8_0;
    float sumf = 0;
    for (size_t ib = 0; ib < nb; ++ib) {
#if defined(__ARM_FEATURE_DOTPROD)
        const uint8x16_t q = vld1q_u8(x[ib].qs);
        const int8x16_t v0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0xF))), vdupq_n_s8(8));
        const int8x16_t v1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)),             vdupq_n_s8(8));
        int32x4_t acc = vdotq_s32(vdupq_n_s32(0), v0, vld1q_s8(y[ib].qs));
        acc = vdotq_s32(acc, v1, vld1q_s8(y[ib].qs + 16));
        int sumi = vaddvq_s32(acc);
#else
        int sumi0 = 0, sumi1 = 0;
        for (size_t j = 0; j < QK8_0 / 2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;
            sumi0 += v0 * y[ib].qs[j];
            sumi1 += v1 * y[ib].qs[j + QK8_0/2];
        }
        int sumi = sumi0 + sumi1;
#endif
        sumf += sumi * fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d);
    }
    return sumf;
}

// Q4_K 6-bit scale/min unpack, ggml's kmask bit-shuffle form (identical values
// to get_scale_min_k4; this form matches the generic kernel line-for-line).
inline void q4k_unpack_scales(const uint8_t* packed, uint8_t* scales8, uint8_t* mins8) {
    uint32_t utmp[4];
    std::memcpy(utmp, packed, 12);
    const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    std::memcpy(scales8, &utmp[0], 8);
    std::memcpy(mins8,   &utmp[2], 8);
}

// ggml_vec_dot_q4_K_q8_K_generic clone (scalar); NEON dotprod fast path.
inline float vec_dot_q4_K_q8_K_scalar(size_t n, const block_q4_K* x, const block_q8_K* y) {
    const size_t nb = n / QK_K;
    float sumf = 0;
    int8_t aux8[QK_K]; int16_t aux16[8]; float sums[8]; int32_t aux32[8];
    std::memset(sums, 0, sizeof sums);
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t* q4 = x[i].qs; const int8_t* q8 = y[i].qs;
        std::memset(aux32, 0, sizeof aux32);
        int8_t* a = aux8;
        for (int j = 0; j < (int)QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = int8_t(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = int8_t(q4[l] >> 4);
            a += 32; q4 += 32;
        }
        uint8_t scales[8], mins[8];
        q4k_unpack_scales(x[i].scales, scales, mins);
        int sumi = 0;
        for (int j = 0; j < (int)QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8; int is = 0;
        for (int j = 0; j < (int)QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int rep = 0; rep < 4; ++rep) {
                for (int l = 0; l < 8; ++l) aux16[l] = int16_t(q8[l] * a[l]);
                for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
                q8 += 8; a += 8;
            }
        }
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = fp16_to_fp32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}
#if defined(__ARM_FEATURE_DOTPROD)
inline float vec_dot_q4_K_q8_K_neon(size_t n, const block_q4_K* x, const block_q8_K* y) {
    const size_t nb = n / QK_K;
    float sumf = 0;
    for (size_t i = 0; i < nb; ++i) {
        uint8_t scales[8], mins[8];
        q4k_unpack_scales(x[i].scales, scales, mins);
        const float d    = fp16_to_fp32(x[i].d)    * y[i].d;
        const float dmin = fp16_to_fp32(x[i].dmin) * y[i].d;
        int sumi_mins = 0;
        for (int j = 0; j < (int)QK_K/16; ++j) sumi_mins += y[i].bsums[j] * mins[j/2];
        const uint8_t* q4 = x[i].qs; const int8_t* q8 = y[i].qs;
        const uint8x16_t m4 = vdupq_n_u8(0xF);
        int32_t total = 0;
        for (int j = 0; j < (int)QK_K/64; ++j) {          // 64 elems: 2 scales
            uint8x16_t r0 = vld1q_u8(q4), r1 = vld1q_u8(q4 + 16); q4 += 32;
            int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(r0, m4));
            int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(r1, m4));
            int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(r0, 4));
            int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(r1, 4));
            int32x4_t a0 = vdotq_s32(vdupq_n_s32(0), lo0, vld1q_s8(q8));
            a0 = vdotq_s32(a0, lo1, vld1q_s8(q8 + 16));   q8 += 32;
            int32x4_t a1 = vdotq_s32(vdupq_n_s32(0), hi0, vld1q_s8(q8));
            a1 = vdotq_s32(a1, hi1, vld1q_s8(q8 + 16));   q8 += 32;
            total += int32_t(scales[2*j])   * vaddvq_s32(a0);
            total += int32_t(scales[2*j+1]) * vaddvq_s32(a1);
        }
        sumf += d * float(total) - dmin * float(sumi_mins);
    }
    return sumf;
}
#endif
inline float vec_dot_q4_K_q8_K(size_t n, const block_q4_K* x, const block_q8_K* y) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vec_dot_q4_K_q8_K_neon(n, x, y);
#else
    return vec_dot_q4_K_q8_K_scalar(n, x, y);
#endif
}

// ggml_vec_dot_q6_K_q8_K_generic clone (scalar); NEON dotprod fast path.
inline float vec_dot_q6_K_q8_K_scalar(size_t n, const block_q6_K* x, const block_q8_K* y) {
    const size_t nb = n / QK_K;
    float sumf = 0;
    int8_t aux8[QK_K]; int16_t aux16[8]; float sums[8]; int32_t aux32[8];
    std::memset(sums, 0, sizeof sums);
    for (size_t i = 0; i < nb; ++i) {
        const uint8_t* q4 = x[i].ql; const uint8_t* qh = x[i].qh;
        const int8_t* q8 = y[i].qs;
        std::memset(aux32, 0, sizeof aux32);
        int8_t* a = aux8;
        for (int j = 0; j < (int)QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = int8_t((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = int8_t((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = int8_t((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = int8_t((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a += 128; q4 += 64; qh += 32;
        }
        a = aux8; int is = 0;
        for (int j = 0; j < (int)QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int rep = 0; rep < 2; ++rep) {
                for (int l = 0; l < 8; ++l) aux16[l] = int16_t(q8[l] * a[l]);
                for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
                q8 += 8; a += 8;
            }
        }
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    return sumf;
}
#if defined(__ARM_FEATURE_DOTPROD)
inline float vec_dot_q6_K_q8_K_neon(size_t n, const block_q6_K* x, const block_q8_K* y) {
    const size_t nb = n / QK_K;
    float sumf = 0;
    for (size_t i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        const uint8_t* ql = x[i].ql; const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales; const int8_t* q8 = y[i].qs;
        const uint8x16_t m4 = vdupq_n_u8(0xF), m3 = vdupq_n_u8(3);
        const int8x16_t off = vdupq_n_s8(32);
        int32_t total = 0;
        for (int half = 0; half < 2; ++half) {            // two 128-elem groups
            for (int seg16 = 0; seg16 < 2; ++seg16) {     // 16-elem columns
                const int L = seg16 * 16;
                uint8x16_t h  = vld1q_u8(qh + L);
                uint8x16_t a0 = vld1q_u8(ql + L), a1 = vld1q_u8(ql + 32 + L);
                int8x16_t v0 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(a0, m4), vshlq_n_u8(vandq_u8(h,            m3), 4))), off);
                int8x16_t v1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(a1, m4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h,2), m3), 4))), off);
                int8x16_t v2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(a0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h,4), m3), 4))), off);
                int8x16_t v3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(a1, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h,6), m3), 4))), off);
                const int is = seg16;                     // scale idx within 8
                total += int32_t(sc[is+0]) * vaddvq_s32(vdotq_s32(vdupq_n_s32(0), v0, vld1q_s8(q8 + L)));
                total += int32_t(sc[is+2]) * vaddvq_s32(vdotq_s32(vdupq_n_s32(0), v1, vld1q_s8(q8 + 32 + L)));
                total += int32_t(sc[is+4]) * vaddvq_s32(vdotq_s32(vdupq_n_s32(0), v2, vld1q_s8(q8 + 64 + L)));
                total += int32_t(sc[is+6]) * vaddvq_s32(vdotq_s32(vdupq_n_s32(0), v3, vld1q_s8(q8 + 96 + L)));
            }
            (void)half;
            ql += 64; qh += 32; sc += 8; q8 += 128;
        }
        sumf += d * float(total);
    }
    return sumf;
}
#endif
inline float vec_dot_q6_K_q8_K(size_t n, const block_q6_K* x, const block_q8_K* y) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vec_dot_q6_K_q8_K_neon(n, x, y);
#else
    return vec_dot_q6_K_q8_K_scalar(n, x, y);
#endif
}

// Which weight types have an int8-activation kernel. Q5_K currently does not:
// it falls back to the fp32-activation dequant-in-dot path, which is this
// engine's numerics reference, so the result is correct and only slower.
inline bool has_i8_kernel(GT t) {
    return t == GT::Q8_0 || t == GT::Q4_0 || t == GT::Q4_K || t == GT::Q6_K;
}

// ---- quantized activation holder for one matvec ---------------------------
// Quantize x ONCE, reuse across all Out rows (the llama.cpp strategy).
struct ActQ {
    std::vector<block_q8_0> q8;    // for Q8_0/Q4_0 weights
    std::vector<block_q8_K> qk;    // for Q4_K/Q6_K weights
    void from(const float* x, size_t n, GT wtype) {
        if (wtype == GT::Q4_K || wtype == GT::Q6_K) {
            qk.resize(n / QK_K);
            quantize_row_q8_K(x, qk.data(), n);
        } else {
            q8.resize(n / QK8_0);
            quantize_row_q8_0(x, q8.data(), n);
        }
    }
};

inline bool nm_fp32_act() {
    static const bool v = [] {
        const char* e = std::getenv("NM_FP32_ACT");
        return e && e[0] && e[0] != '0';
    }();
    return v;
}

// ---- Weight::matvec (declared in quant.hpp) --------------------------------
template <size_t In, size_t Out>
Vec<Out> Weight<In, Out>::matvec(VecView<In> x) const {
    Vec<Out> y;
    const GT t = type_;
    if (t == GT::F32) {
        par_for(Out, [&](size_t o){ y[o] = dot(x, f32_.row(o)); });
        return y;
    }
    const size_t rb = type_size_bytes(t, In);
    if (!nm_fp32_act() && has_i8_kernel(t)) {      // int8 activations (default)
        // Plain local, captured by reference: workers must read THIS act.
        // (A thread_local here would hand each worker its own empty instance.)
        ActQ act;                                  // quantized once, pre-par_for
        act.from(&x[0], In, t);
        par_for(Out, [&](size_t o) {
            const uint8_t* row = qp_ + o * rb;
            switch (t) {
                case GT::Q8_0: y[o] = vec_dot_q8_0_q8_0(In, (const block_q8_0*)row, act.q8.data()); break;
                case GT::Q4_0: y[o] = vec_dot_q4_0_q8_0(In, (const block_q4_0*)row, act.q8.data()); break;
                case GT::Q4_K: y[o] = vec_dot_q4_K_q8_K(In, (const block_q4_K*)row, act.qk.data()); break;
                case GT::Q6_K: y[o] = vec_dot_q6_K_q8_K(In, (const block_q6_K*)row, act.qk.data()); break;
                default: break;
            }
        });
        return y;
    }
    par_for(Out, [&](size_t o) {                   // fp32 activations (reference)
        const uint8_t* row = qp_ + o * rb;
        Scalar acc = 0;
        switch (t) {
            case GT::Q8_0: {
                const block_q8_0* b = (const block_q8_0*)row;
                for (size_t i = 0; i < In / QK8_0; ++i) acc += dot_block_q8_0(b[i], &x[i*QK8_0]);
                break; }
            case GT::Q4_0: {
                const block_q4_0* b = (const block_q4_0*)row;
                for (size_t i = 0; i < In / QK4_0; ++i) acc += dot_block_q4_0(b[i], &x[i*QK4_0]);
                break; }
            case GT::Q4_K: {
                const block_q4_K* b = (const block_q4_K*)row;
                for (size_t i = 0; i < In / QK_K; ++i) acc += dot_block_q4_K(b[i], &x[i*QK_K]);
                break; }
            case GT::Q5_K: {
                const block_q5_K* b = (const block_q5_K*)row;
                for (size_t i = 0; i < In / QK_K; ++i) acc += dot_block_q5_K(b[i], &x[i*QK_K]);
                break; }
            case GT::Q6_K: {
                const block_q6_K* b = (const block_q6_K*)row;
                for (size_t i = 0; i < In / QK_K; ++i) acc += dot_block_q6_K(b[i], &x[i*QK_K]);
                break; }
            default: break;
        }
        y[o] = acc;
    });
    return y;
}

// Mathematical interface for X[T,In] @ W[In,Out]. Activations are prepared
// once, then each worker holds one output/weight row stationary while applying
// it to all T activation rows. This streams W once per batch instead of once
// per token. A tiled SME/GPU backend can replace this reference kernel without
// changing components or architectures.
template <size_t In, size_t Out>
Matrix<Out> Weight<In, Out>::matmul(MatrixView<In> x) const {
    Matrix<Out> y(x.rows());
    if (x.rows() == 0) return y;

    const GT type = type_;
    if (type == GT::F32) {
        par_for(Out, [&](size_t output) {
            const VecView<In> weight = f32_.row(output);
            for (size_t row = 0; row < x.rows(); ++row)
                y.data()[row * Out + output] =
                    dot(x.row(row), weight);
        });
        return y;
    }

    const size_t row_bytes = type_size_bytes(type, In);
    if (type != GT::Q8_0 && type != GT::Q4_0 && type != GT::Q4_K &&
        type != GT::Q5_K && type != GT::Q6_K)
        throw std::runtime_error(
            "Weight::matmul: unsupported quantized type");
    if (!nm_fp32_act() && has_i8_kernel(type)) {
        std::vector<ActQ> activations(x.rows());
        par_for(x.rows(), [&](size_t row) {
            activations[row].from(x.row(row).begin(), In, type);
        });
        par_for(Out, [&](size_t output) {
            const uint8_t* weight_row = qp_ + output * row_bytes;
            for (size_t row = 0; row < x.rows(); ++row) {
                const ActQ& activation = activations[row];
                Scalar result = 0;
                switch (type) {
                    case GT::Q8_0:
                        result = vec_dot_q8_0_q8_0(
                            In,
                            reinterpret_cast<const block_q8_0*>(weight_row),
                            activation.q8.data());
                        break;
                    case GT::Q4_0:
                        result = vec_dot_q4_0_q8_0(
                            In,
                            reinterpret_cast<const block_q4_0*>(weight_row),
                            activation.q8.data());
                        break;
                    case GT::Q4_K:
                        result = vec_dot_q4_K_q8_K(
                            In,
                            reinterpret_cast<const block_q4_K*>(weight_row),
                            activation.qk.data());
                        break;
                    case GT::Q6_K:
                        result = vec_dot_q6_K_q8_K(
                            In,
                            reinterpret_cast<const block_q6_K*>(weight_row),
                            activation.qk.data());
                        break;
                    default:
                        break;
                }
                y.data()[row * Out + output] = result;
            }
        });
        return y;
    }

    par_for(Out, [&](size_t output) {
        const uint8_t* weight_row = qp_ + output * row_bytes;
        for (size_t row = 0; row < x.rows(); ++row) {
            const VecView<In> activation = x.row(row);
            Scalar result = 0;
            switch (type) {
                case GT::Q8_0: {
                    const auto* blocks =
                        reinterpret_cast<const block_q8_0*>(weight_row);
                    for (size_t block = 0; block < In / QK8_0; ++block)
                        result += dot_block_q8_0(
                            blocks[block],
                            &activation[block * QK8_0]);
                    break;
                }
                case GT::Q4_0: {
                    const auto* blocks =
                        reinterpret_cast<const block_q4_0*>(weight_row);
                    for (size_t block = 0; block < In / QK4_0; ++block)
                        result += dot_block_q4_0(
                            blocks[block],
                            &activation[block * QK4_0]);
                    break;
                }
                case GT::Q4_K: {
                    const auto* blocks =
                        reinterpret_cast<const block_q4_K*>(weight_row);
                    for (size_t block = 0; block < In / QK_K; ++block)
                        result += dot_block_q4_K(
                            blocks[block],
                            &activation[block * QK_K]);
                    break;
                }
                case GT::Q5_K: {
                    const auto* blocks =
                        reinterpret_cast<const block_q5_K*>(weight_row);
                    for (size_t block = 0; block < In / QK_K; ++block)
                        result += dot_block_q5_K(
                            blocks[block],
                            &activation[block * QK_K]);
                    break;
                }
                case GT::Q6_K: {
                    const auto* blocks =
                        reinterpret_cast<const block_q6_K*>(weight_row);
                    for (size_t block = 0; block < In / QK_K; ++block)
                        result += dot_block_q6_K(
                            blocks[block],
                            &activation[block * QK_K]);
                    break;
                }
                default:
                    break;
            }
            y.data()[row * Out + output] = result;
        }
    });
    return y;
}
