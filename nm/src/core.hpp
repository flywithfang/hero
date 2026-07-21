// core.hpp — strata 0 (typed storage) and 1 (algebra).
//
// Split out of gpt_pipeline_v10.cpp, preserving its design. This is the ONLY
// place layout knowledge lives. Two matrix layouts coexist here and their
// difference is the whole of trap T1 (see INFERENCE_PLAN.md §M1):
//
//   Mat<R,C>   row-major, y = x·W  (In dimension outer).      concept default.
//   MatT<In,Out> row-major with row = one OUTPUT's weights (length In),
//                contiguous per output. y = matvec_T(x,W). This is EXACTLY
//                how ggml/llama.cpp store a 2-D weight (ne0 = In contiguous),
//                so a GGUF fp32 tensor loads zero-copy into a MatT view over
//                the mmap, and each output is a single contiguous dot product.
//
// Decision (documented once, here): the engine's runtime weights are MatT.
// The concept Mat<R,C> stays for the property tests and any owned scratch.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

using Scalar = float;

// ---- fp16 <-> fp32 (IEEE half; ggml stores deltas/F16 tensors as these) ----
inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {                              // subnormal / zero
        if (mant == 0) { bits = sign; }
        else {
            int e = -1;
            uint32_t m = mant;
            do { m <<= 1; ++e; } while ((m & 0x400) == 0);
            m &= 0x3FF;
            bits = sign | ((uint32_t(112 - e)) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {                     // inf / nan
        bits = sign | 0x7F800000 | (mant << 13);
    } else {                                      // normal
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &bits, 4); return f;
}
inline uint16_t fp32_to_fp16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = int32_t((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) {                               // flush tiny to subnormal/0
        if (exp < -10) return uint16_t(sign);
        mant |= 0x800000;
        uint32_t shift = uint32_t(14 - exp);
        uint32_t r = mant >> shift;
        if ((mant >> (shift - 1)) & 1) ++r;       // round to nearest
        return uint16_t(sign | r);
    }
    if (exp >= 0x1F) return uint16_t(sign | 0x7C00); // overflow -> inf
    uint16_t h = uint16_t(sign | (uint32_t(exp) << 10) | (mant >> 13));
    if (mant & 0x1000) ++h;                        // round to nearest
    return h;
}

// ---- bfloat16 <-> fp32 ----------------------------------------------------
// BF16 keeps fp32's eight exponent bits and truncates the significand to seven
// bits. Checkpoints use it because its range matches fp32; arithmetic in this
// engine still widens to Scalar/fp32 before accumulation.
inline float bf16_to_fp32(uint16_t b) {
    const uint32_t bits = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t fp32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    // Preserve NaNs as NaNs even when their payload exists only in the low half.
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
        return uint16_t((bits >> 16) | 0x0040u);
    // Round to nearest, ties to even.
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return uint16_t(bits >> 16);
}

// ================= stratum 0: typed storage ================================

template <size_t N>
struct Vec {
    static constexpr size_t extent = N;
    std::unique_ptr<Scalar[]> p = std::make_unique<Scalar[]>(N); // zeroed
    Scalar&       operator[](size_t i)       { return p[i]; }
    const Scalar& operator[](size_t i) const { return p[i]; }
    Scalar* begin() { return p.get(); }  Scalar* end() { return p.get() + N; }
};

template <size_t N>
struct VecView {
    const Scalar* p;
    VecView(const Vec<N>& v) : p(v.p.get()) {}
    explicit VecView(const Scalar* raw) : p(raw) {}
    const Scalar& operator[](size_t i) const { return p[i]; }
    const Scalar* begin() const { return p; } const Scalar* end() const { return p + N; }
};

template <size_t N>
struct MutVecView {
    Scalar* p;
    Scalar& operator[](size_t i) const { return p[i]; }
    Scalar* begin() const { return p; }
    Scalar* end() const { return p + N; }
};

// Runtime token count x compile-time channel width. Unlike model parameters,
// activations are intentionally writable scratch; shape and ownership remain
// private and construction establishes the complete storage invariant.
template <size_t C>
class TokenMatrix {
public:
    explicit TokenMatrix(size_t rows)
        : rows_(rows), data_(element_count(rows), Scalar(0)) {}

    size_t rows() const { return rows_; }
    static constexpr size_t cols() { return C; }

    VecView<C> row(size_t i) const {
        check_row(i);
        return VecView<C>(data_.data() + i * C);
    }
    MutVecView<C> row_mut(size_t i) {
        check_row(i);
        return MutVecView<C>{data_.data() + i * C};
    }
    void set_row(size_t i, VecView<C> value) {
        MutVecView<C> dst = row_mut(i);
        std::copy(value.begin(), value.end(), dst.begin());
    }

private:
    static size_t element_count(size_t rows) {
        if constexpr (C != 0) {
            if (rows > std::numeric_limits<size_t>::max() / C)
                throw std::length_error("TokenMatrix: shape overflows size_t");
        }
        return rows * C;
    }
    void check_row(size_t i) const {
        if (i >= rows_) throw std::out_of_range("TokenMatrix: row out of range");
    }

    size_t rows_;
    std::vector<Scalar> data_;
};

template <size_t R, size_t C>
struct Mat {                               // concept layout: y = x·W
    static constexpr size_t rows = R, cols = C;
    std::unique_ptr<Scalar[]> p = std::make_unique<Scalar[]>(R * C);
    Scalar&       operator()(size_t r, size_t c)       { return p[r * C + c]; }
    const Scalar& operator()(size_t r, size_t c) const { return p[r * C + c]; }
    VecView<C> row(size_t i) const { return VecView<C>(&p[i * C]); }
    Scalar* raw() { return p.get(); }
};

template <size_t C>
struct RowStore {                          // runtime rows x compile-time cols
    std::vector<Scalar> data;
    size_t rows = 0;
    VecView<C> row(size_t i) const { return VecView<C>(&data[i * C]); }
    void append(VecView<C> v) { data.insert(data.end(), v.begin(), v.end()); ++rows; }
};

enum class TokenId : int32_t {};
struct Position { size_t i; };

// Two head-index types that must never be confused (GQA). group() in Attention
// is the sole constructor of a KvHead from a QHead.
struct QHead  { size_t i; };
struct KvHead { size_t i; };

template <size_t Hkv, size_t Dqk, size_t Dv>
struct PastView {
    const RowStore<Hkv * Dqk>& K;
    const RowStore<Hkv * Dv>&  V;
    VecView<Dqk> key  (size_t j, KvHead g) const;
    VecView<Dv>  value(size_t j, KvHead g) const;
};

// ================= stratum 1: algebra ======================================

template <size_t N> Vec<N> operator+(VecView<N> a, VecView<N> b) {
    Vec<N> y; for (size_t i = 0; i < N; ++i) y[i] = a[i] + b[i]; return y;
}
template <size_t N> void operator+=(Vec<N>& y, VecView<N> x) {
    for (size_t i = 0; i < N; ++i) y[i] += x[i];
}
template <size_t N> void operator+=(Vec<N>& y, const Vec<N>& x) {
    for (size_t i = 0; i < N; ++i) y[i] += x[i];
}
template <size_t N> Vec<N> copy(VecView<N> v) {
    Vec<N> y; for (size_t i = 0; i < N; ++i) y[i] = v[i]; return y;
}
template <size_t N> Scalar dot(VecView<N> a, VecView<N> b) {
    Scalar s = 0; for (size_t i = 0; i < N; ++i) s += a[i] * b[i]; return s; // fp32 accum (T6)
}
template <size_t N> void axpy(Scalar a, VecView<N> x, Vec<N>& y) {
    for (size_t i = 0; i < N; ++i) y[i] += a * x[i];
}
template <size_t W_, size_t N>
VecView<W_> slice(VecView<N> v, size_t off) {
    static_assert(W_ <= N);
    return VecView<W_>(&v[off]);
}
template <size_t W_, size_t N>
MutVecView<W_> slice_mut(Vec<N>& v, size_t off) {
    static_assert(W_ <= N);
    return MutVecView<W_>{&v[off]};
}

template <size_t Hkv, size_t Dqk, size_t Dv>
VecView<Dqk> PastView<Hkv, Dqk, Dv>::key(size_t j, KvHead g) const {
    return slice<Dqk>(K.row(j), g.i * Dqk);
}
template <size_t Hkv, size_t Dqk, size_t Dv>
VecView<Dv> PastView<Hkv, Dqk, Dv>::value(size_t j, KvHead g) const {
    return slice<Dv>(V.row(j), g.i * Dv);
}

inline void softmax(std::span<Scalar> s) {
    Scalar mx = *std::max_element(s.begin(), s.end()), sum = 0;
    for (auto& v : s) { v = std::exp(v - mx); sum += v; }
    for (auto& v : s) v /= sum;
}
template <size_t N> void gelu(Vec<N>& v) {
    for (size_t i = 0; i < N; ++i) {
        Scalar x = v[i];
        v[i] = 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}
template <size_t N> void silu(Vec<N>& v) {            // z·sigmoid(z)
    for (size_t i = 0; i < N; ++i) v[i] = v[i] / (1.f + std::exp(-v[i]));
}
template <size_t N> void operator*=(Vec<N>& y, const Vec<N>& x) {
    for (size_t i = 0; i < N; ++i) y[i] *= x[i];       // Hadamard
}

// ---- the parallel seams (M5 replaces the bodies with a thread pool) --------
void   par_for(size_t n, const std::function<void(size_t)>& f);   // core.cpp
size_t nm_num_threads();                                          // core.cpp

// Parallel MAP: f(h) is a pure function of its index and by-value captures and
// RETURNS its partition; placement owned here. The thread/GPU backend seam.
template <size_t H, size_t Dim, class F>
Vec<H * Dim> par_map(F&& f) {
    Vec<H * Dim> out;
    par_for(H, [&](size_t h) {
        Vec<Dim> r = f(h);
        std::copy(r.begin(), r.end(), &out[h * Dim]);
    });
    return out;
}

// ---- MatT<In,Out>: the ggml/llama.cpp weight layout (row = output) ---------
// Owns nothing by default: `p` may point into an mmap'd GGUF file. When a
// tensor must be materialised as fp32 (F16 dequant, or the debug dequant path
// for quantised tensors), `own` holds the storage and `p` points into it.
template <size_t In, size_t Out>
struct MatT {
    const Scalar*             p = nullptr;   // [Out*In], row o at p + o*In
    std::shared_ptr<Scalar[]> own;           // optional backing; null => view
    VecView<In> row(size_t o) const { return VecView<In>(p + o * In); }

    static MatT owning() {                   // allocate [Out*In], zeroed
        MatT m; m.own = std::shared_ptr<Scalar[]>(new Scalar[In * Out]());
        m.p = m.own.get(); return m;
    }
    Scalar* raw() { return const_cast<Scalar*>(p); }
};

// matvec_T: y[o] = dot(x, W.row(o)), Out independent dot products of length In.
// [BANDWIDTH] streams the whole weight once; [COMPUTE] In*Out MACs.
template <size_t In, size_t Out>
Vec<Out> matvec_T(VecView<In> x, const MatT<In, Out>& W) {
    Vec<Out> y;
    par_for(Out, [&](size_t o) { y[o] = dot(x, W.row(o)); });
    return y;
}
