// quant.hpp — GGUF tensor element types, block formats, dequant kernels, and
// the unified Weight<In,Out> that dispatches matvec across storage formats.
//
// Block layouts and dequant math are copied VERBATIM from ggml (ggml-common.h,
// ggml-quants.c) — the bit-unpacking is fiddly and any deviation is silent
// garbage (trap: quant is the one place we cannot "roughly" match). Kernels
// dequantise block-by-block INSIDE the fp32 dot product (T6). A dequant-at-load
// debug path also exists (dequant_to_f32) and is used by the loader for F16 and
// as the M1 fidelity crutch.
#pragma once
#include "core.hpp"
#include "simd.hpp"
#include <cassert>
#include <stdexcept>
#include <string>

// ggml_type values (ggml.h) — only the subset this engine reads.
enum class GT : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    BF16 = 30,
};

inline const char* gt_name(GT t) {
    switch (t) {
        case GT::F32:
            return "F32";
        case GT::F16:
            return "F16";
        case GT::BF16:
            return "BF16";
        case GT::Q4_0:
            return "Q4_0";
        case GT::Q8_0:
            return "Q8_0";
        case GT::Q4_K:
            return "Q4_K";
        case GT::Q5_K:
            return "Q5_K";
        case GT::Q6_K:
            return "Q6_K";
    }
    return "UNKNOWN";
}

// block sizes (elements) and byte sizes, matching ggml static_asserts.
constexpr size_t QK4_0 = 32, QK8_0 = 32, QK_K = 256, K_SCALE_SIZE = 12;

#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;
    uint8_t qs[QK4_0 / 2];
};  // 18
struct block_q8_0 {
    uint16_t d;
    int8_t qs[QK8_0];
};  // 34
struct block_q4_K {
    uint16_t d, dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};  // 144
// Q5_K is Q4_K plus one high bit per weight, carried in a separate plane.
struct block_q5_K {
    uint16_t d, dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
};  // 176
struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
};  // 210
#pragma pack(pop)
static_assert(sizeof(block_q4_0) == 18);
static_assert(sizeof(block_q8_0) == 34);
static_assert(sizeof(block_q4_K) == 144);
static_assert(sizeof(block_q5_K) == 176);
static_assert(sizeof(block_q6_K) == 210);

// bytes for a tensor of `nelem` elements stored as type t (nelem % blocklen==0).
inline size_t type_size_bytes(GT t, size_t nelem) {
    switch (t) {
        case GT::F32:
            return nelem * 4;
        case GT::F16:
        case GT::BF16:
            return nelem * 2;
        case GT::Q4_0:
            return nelem / QK4_0 * sizeof(block_q4_0);
        case GT::Q8_0:
            return nelem / QK8_0 * sizeof(block_q8_0);
        case GT::Q4_K:
            return nelem / QK_K * sizeof(block_q4_K);
        case GT::Q5_K:
            return nelem / QK_K * sizeof(block_q5_K);
        case GT::Q6_K:
            return nelem / QK_K * sizeof(block_q6_K);
    }
    throw std::runtime_error("type_size_bytes: unknown type");
}
inline size_t block_len(GT t) {
    switch (t) {
        case GT::F32:
        case GT::F16:
        case GT::BF16:
            return 1;
        case GT::Q4_0:
            return QK4_0;
        case GT::Q8_0:
            return QK8_0;
        case GT::Q4_K:
        case GT::Q5_K:
        case GT::Q6_K:
            return QK_K;
    }
    return 1;
}

// ---- dequant-in-dot kernels: dot(x[base..base+len), row block) -------------
// Each returns the block's contribution to a dot product with x starting at xb.

inline Scalar dot_block_q8_0(const block_q8_0& b, const Scalar* xb) {
    const Scalar d = fp16_to_fp32(b.d);
#if defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0);
    acc16_signed(acc, vld1q_s8(b.qs), 1.0f, xb);
    acc16_signed(acc, vld1q_s8(b.qs + 16), 1.0f, xb + 16);
    return d * vaddvq_f32(acc);
#else
    Scalar s = 0;
    for (size_t i = 0; i < QK8_0; ++i) s += Scalar(b.qs[i]) * xb[i];
    return d * s;
#endif
}
inline Scalar dot_block_q4_0(const block_q4_0& b, const Scalar* xb) {
    const Scalar d = fp16_to_fp32(b.d);
    Scalar s = 0;
    for (size_t j = 0; j < QK4_0 / 2; ++j) {
        const int lo = (b.qs[j] & 0x0F) - 8;  // element j
        const int hi = (b.qs[j] >> 4) - 8;    // element j+16
        s += Scalar(lo) * xb[j] + Scalar(hi) * xb[j + QK4_0 / 2];
    }
    return d * s;
}
// get_scale_min_k4 (ggml-quants.c) — 6-bit scale/min unpack for Q4_K.
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
inline Scalar dot_block_q4_K(const block_q4_K& b, const Scalar* xb) {
    const Scalar d = fp16_to_fp32(b.d), mn = fp16_to_fp32(b.dmin);
    const uint8_t* q = b.qs;
    int is = 0;
    const Scalar* x = xb;
#if defined(__ARM_NEON)
    float32x4_t vacc = vdupq_n_f32(0);
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    for (int j = 0; j < (int)QK_K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, b.scales, sc, m);
        const float d1 = d * sc, m1 = mn * m;
        get_scale_min_k4(is + 1, b.scales, sc, m);
        const float d2 = d * sc, m2 = mn * m;
        uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
        acc16_scaled(vacc, vandq_u8(q0, mask), d1, m1, x);       // low  0..15
        acc16_scaled(vacc, vandq_u8(q1, mask), d1, m1, x + 16);  // low 16..31
        acc16_scaled(vacc, vshrq_n_u8(q0, 4), d2, m2, x + 32);   // high 0..15
        acc16_scaled(vacc, vshrq_n_u8(q1, 4), d2, m2, x + 48);   // high 16..31
        x += 64;
        q += 32;
        is += 2;
    }
    return vaddvq_f32(vacc);
#else
    Scalar acc = 0;
    for (int j = 0; j < (int)QK_K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, b.scales, sc, m);
        const Scalar d1 = d * sc, m1 = mn * m;
        get_scale_min_k4(is + 1, b.scales, sc, m);
        const Scalar d2 = d * sc, m2 = mn * m;
        for (int l = 0; l < 32; ++l) acc += (d1 * Scalar(q[l] & 0xF) - m1) * *x++;
        for (int l = 0; l < 32; ++l) acc += (d2 * Scalar(q[l] >> 4) - m2) * *x++;
        q += 32;
        is += 2;
    }
    return acc;
#endif
}
// Q5_K: the same 6-bit scale/min structure as Q4_K, with a fifth bit per
// weight taken from the qh plane. Bit (2*sub) of qh[l] belongs to the low
// nibble of sub-block pair `sub`, bit (2*sub+1) to the high nibble — which is
// what the u1/u2 shifting below tracks.
inline Scalar dot_block_q5_K(const block_q5_K& b, const Scalar* xb) {
    const Scalar d = fp16_to_fp32(b.d), mn = fp16_to_fp32(b.dmin);
    const uint8_t* ql = b.qs;
    const uint8_t* qh = b.qh;
    const Scalar* x = xb;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    Scalar acc = 0;
    for (int j = 0; j < (int)QK_K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, b.scales, sc, m);
        const Scalar d1 = d * sc, m1 = mn * m;
        get_scale_min_k4(is + 1, b.scales, sc, m);
        const Scalar d2 = d * sc, m2 = mn * m;
        for (int l = 0; l < 32; ++l) acc += (d1 * Scalar((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1) * *x++;
        for (int l = 0; l < 32; ++l) acc += (d2 * Scalar((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2) * *x++;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
    return acc;
}

#if defined(__ARM_NEON)
// build 16 int8 Q6 values: (nib | (((qh>>Shift)&3)<<4)) - 32.
template <int Shift>
inline int8x16_t q6_vals(uint8x16_t nib, uint8x16_t qh16) {
    uint8x16_t hb = Shift == 0 ? vandq_u8(qh16, vdupq_n_u8(3)) : vandq_u8(vshrq_n_u8(qh16, Shift ? Shift : 1), vdupq_n_u8(3));
    uint8x16_t v = vorrq_u8(nib, vshlq_n_u8(hb, 4));
    return vsubq_s8(vreinterpretq_s8_u8(v), vdupq_n_s8(32));
}
#endif

inline Scalar dot_block_q6_K(const block_q6_K& b, const Scalar* xb) {
    const Scalar d = fp16_to_fp32(b.d);
    const uint8_t* ql = b.ql;
    const uint8_t* qh = b.qh;
    const int8_t* sc = b.scales;
    const Scalar* y = xb;
#if defined(__ARM_NEON)
    float32x4_t vacc = vdupq_n_f32(0);
    const uint8x16_t m0f = vdupq_n_u8(0x0F);
    for (int n = 0; n < (int)QK_K; n += 128) {
        for (int half = 0; half < 2; ++half) {
            const int L = half * 16, is = half;
            uint8x16_t qh16 = vld1q_u8(qh + L);
            uint8x16_t qlA = vld1q_u8(ql + L);       // low/high -> segments 0,2
            uint8x16_t qlB = vld1q_u8(ql + 32 + L);  // low/high -> segments 1,3
            acc16_signed(vacc, q6_vals<0>(vandq_u8(qlA, m0f), qh16), d * sc[is + 0], y + L);
            acc16_signed(vacc, q6_vals<2>(vandq_u8(qlB, m0f), qh16), d * sc[is + 2], y + 32 + L);
            acc16_signed(vacc, q6_vals<4>(vshrq_n_u8(qlA, 4), qh16), d * sc[is + 4], y + 64 + L);
            acc16_signed(vacc, q6_vals<6>(vshrq_n_u8(qlB, 4), qh16), d * sc[is + 6], y + 96 + L);
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return vaddvq_f32(vacc);
#else
    Scalar acc = 0;
    for (int n = 0; n < (int)QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = int((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = int((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            acc += d * sc[is + 0] * q1 * y[l + 0];
            acc += d * sc[is + 2] * q2 * y[l + 32];
            acc += d * sc[is + 4] * q3 * y[l + 64];
            acc += d * sc[is + 6] * q4 * y[l + 96];
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
    return acc;
#endif
}

// ---- dequant a whole row (In elements) to fp32 (debug / F16 load path) ------
inline void dequant_to_f32(GT t, const void* src, Scalar* dst, size_t n) {
    switch (t) {
        case GT::F32:
            std::memcpy(dst, src, n * 4);
            return;
        case GT::F16: {
            const uint16_t* h = (const uint16_t*)src;
            for (size_t i = 0; i < n; ++i) dst[i] = fp16_to_fp32(h[i]);
            return;
        }
        case GT::BF16: {
            const uint16_t* b = (const uint16_t*)src;
            for (size_t i = 0; i < n; ++i) dst[i] = bf16_to_fp32(b[i]);
            return;
        }
        case GT::Q8_0: {
            const block_q8_0* b = (const block_q8_0*)src;
            for (size_t i = 0; i < n / QK8_0; ++i) {
                const Scalar d = fp16_to_fp32(b[i].d);
                for (size_t j = 0; j < QK8_0; ++j) dst[i * QK8_0 + j] = d * b[i].qs[j];
            }
            return;
        }
        case GT::Q4_0: {
            const block_q4_0* b = (const block_q4_0*)src;
            for (size_t i = 0; i < n / QK4_0; ++i) {
                const Scalar d = fp16_to_fp32(b[i].d);
                for (size_t j = 0; j < QK4_0 / 2; ++j) {
                    dst[i * QK4_0 + j] = d * ((b[i].qs[j] & 0xF) - 8);
                    dst[i * QK4_0 + j + QK4_0 / 2] = d * ((b[i].qs[j] >> 4) - 8);
                }
            }
            return;
        }
        case GT::Q4_K: {
            const block_q4_K* b = (const block_q4_K*)src;
            Scalar* y = dst;
            for (size_t i = 0; i < n / QK_K; ++i) {
                const Scalar d = fp16_to_fp32(b[i].d), mn = fp16_to_fp32(b[i].dmin);
                const uint8_t* q = b[i].qs;
                int is = 0;
                for (int j = 0; j < (int)QK_K; j += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, b[i].scales, sc, m);
                    const Scalar d1 = d * sc, m1 = mn * m;
                    get_scale_min_k4(is + 1, b[i].scales, sc, m);
                    const Scalar d2 = d * sc, m2 = mn * m;
                    for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
                    for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
                    q += 32;
                    is += 2;
                }
            }
            return;
        }
        case GT::Q5_K: {
            const block_q5_K* b = (const block_q5_K*)src;
            Scalar* y = dst;
            for (size_t i = 0; i < n / QK_K; ++i) {
                const Scalar d = fp16_to_fp32(b[i].d), mn = fp16_to_fp32(b[i].dmin);
                const uint8_t* ql = b[i].qs;
                const uint8_t* qh = b[i].qh;
                int is = 0;
                uint8_t u1 = 1, u2 = 2;
                for (int j = 0; j < (int)QK_K; j += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, b[i].scales, sc, m);
                    const Scalar d1 = d * sc, m1 = mn * m;
                    get_scale_min_k4(is + 1, b[i].scales, sc, m);
                    const Scalar d2 = d * sc, m2 = mn * m;
                    for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
                    for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                    ql += 32;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
            return;
        }
        case GT::Q6_K: {
            const block_q6_K* b = (const block_q6_K*)src;
            Scalar* y = dst;
            for (size_t i = 0; i < n / QK_K; ++i) {
                const Scalar d = fp16_to_fp32(b[i].d);
                const uint8_t* ql = b[i].ql;
                const uint8_t* qh = b[i].qh;
                const int8_t* sc = b[i].scales;
                for (int nn = 0; nn < (int)QK_K; nn += 128) {
                    for (int l = 0; l < 32; ++l) {
                        const int is = l / 16;
                        const int q1 = int((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int q3 = int((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l + 0] = d * sc[is + 0] * q1;
                        y[l + 32] = d * sc[is + 2] * q2;
                        y[l + 64] = d * sc[is + 4] * q3;
                        y[l + 96] = d * sc[is + 6] * q4;
                    }
                    y += 128;
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
            }
            return;
        }
    }
    throw std::runtime_error("dequant_to_f32: unsupported type");
}

// ---- reference quantisers (for round-trip tests and synthetic fixtures) ----
inline void quantize_q8_0(const Scalar* x, block_q8_0* b, size_t n) {
    for (size_t i = 0; i < n / QK8_0; ++i) {
        Scalar amax = 0;
        for (size_t j = 0; j < QK8_0; ++j) amax = std::max(amax, std::fabs(x[i * QK8_0 + j]));
        const Scalar d = amax / 127.f;
        const Scalar id = d ? 1.f / d : 0.f;
        b[i].d = fp32_to_fp16(d);
        for (size_t j = 0; j < QK8_0; ++j) b[i].qs[j] = int8_t(std::lround(x[i * QK8_0 + j] * id));
    }
}
inline void quantize_q4_0(const Scalar* x, block_q4_0* b, size_t n) {
    for (size_t i = 0; i < n / QK4_0; ++i) {
        Scalar amax = 0, max = 0;
        for (size_t j = 0; j < QK4_0; ++j) {
            const Scalar v = x[i * QK4_0 + j];
            if (std::fabs(v) > amax) {
                amax = std::fabs(v);
                max = v;
            }
        }
        const Scalar d = max / -8.f;
        const Scalar id = d ? 1.f / d : 0.f;
        b[i].d = fp32_to_fp16(d);
        for (size_t j = 0; j < QK4_0 / 2; ++j) {
            const int x0 = std::min(15, int(x[i * QK4_0 + j] * id + 8.5f));
            const int x1 = std::min(15, int(x[i * QK4_0 + j + QK4_0 / 2] * id + 8.5f));
            b[i].qs[j] = uint8_t(x0 | (x1 << 4));
        }
    }
}

// ============ Weight<In,Out>: layout owner across storage formats ===========
// The concept's Linear held a Mat that owned fp32. The real weight is a MatT
// view over the mmap when fp32/quant, or an owned fp32 buffer when dequantised.
// One type keeps every layout decision (trap T1) in one place.
template <size_t In, size_t Out>
class Weight {
public:
    // F32 storage may be an mmap view or an owned, materialised matrix.
    explicit Weight(MatT<In, Out> f32, std::shared_ptr<const void> keepalive = {}) : type_(GT::F32), f32_(std::move(f32)), keepalive_(std::move(keepalive)) {
        if (!f32_.p) throw std::invalid_argument("Weight: null F32 data");
    }

    // Quantised storage is always a zero-copy view into a kept-alive owner.
    Weight(GT type, const void* data, std::shared_ptr<const void> keepalive) : type_(type), qp_(static_cast<const uint8_t*>(data)), keepalive_(std::move(keepalive)) {
        if (type_ == GT::F32 || type_ == GT::F16 || type_ == GT::BF16) throw std::invalid_argument("Weight: expected a quantised type");
        if (!qp_) throw std::invalid_argument("Weight: null quantised data");
    }

    GT type() const { return type_; }
    size_t row_bytes() const { return type_size_bytes(type_, In); }

    // Defined in quant_i8.hpp (included via components.hpp): default path
    // quantizes the activation to int8 once and does integer dot products per
    // row, exactly like llama.cpp's CPU path; NM_FP32_ACT=1 selects the fp32-
    // activation dequant-in-dot path (our numerics reference). Block
    // divisibility (In % QK == 0) for quantised tensors is checked at load
    // time, not statically — F32 tensors like FF=172 are legal.
    Vec<Out> matvec(VecView<In> x) const;
    Matrix<Out> matmul(MatrixView<In> x) const;
    Matrix<In> gather_rows(std::span<const TokenId> rows) const {
        Matrix<In> output;
        output.reserve(rows.size());
        for (TokenId row : rows) output.append(dequant_row(size_t(row)));
        return output;
    }

    // materialise one output row as fp32 (used for tied unembedding rows etc.)
    Vec<In> dequant_row(size_t o) const {
        Vec<In> r;
        if (type_ == GT::F32) {
            for (size_t i = 0; i < In; ++i) r[i] = f32_.row(o)[i];
            return r;
        }
        dequant_to_f32(type_, qp_ + o * type_size_bytes(type_, In), r.begin(), In);
        return r;
    }

private:
    GT type_ = GT::F32;
    MatT<In, Out> f32_;                      // valid iff type_==F32
    const uint8_t* qp_ = nullptr;            // quant blocks, row-major (row=output)
    std::shared_ptr<const void> keepalive_;  // holds mmap/owner alive
};
