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
    const uint32_t exp = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {  // subnormal / zero
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -1;
            uint32_t m = mant;
            do {
                m <<= 1;
                ++e;
            } while ((m & 0x400) == 0);
            m &= 0x3FF;
            bits = sign | ((uint32_t(112 - e)) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {  // inf / nan
        bits = sign | 0x7F800000 | (mant << 13);
    } else {  // normal
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = int32_t((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) {  // flush tiny to subnormal/0
        if (exp < -10) return uint16_t(sign);
        mant |= 0x800000;
        uint32_t shift = uint32_t(14 - exp);
        uint32_t r = mant >> shift;
        if ((mant >> (shift - 1)) & 1) ++r;  // round to nearest
        return uint16_t(sign | r);
    }
    if (exp >= 0x1F) return uint16_t(sign | 0x7C00);  // overflow -> inf
    uint16_t h = uint16_t(sign | (uint32_t(exp) << 10) | (mant >> 13));
    if (mant & 0x1000) ++h;  // round to nearest
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
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return uint16_t((bits >> 16) | 0x0040u);
    // Round to nearest, ties to even.
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return uint16_t(bits >> 16);
}

// ================= stratum 0: typed storage ================================

template <size_t N>
struct VecView;

template <size_t N>
struct Vec {
    static constexpr size_t extent = N;
    std::unique_ptr<Scalar[]> p = std::make_unique<Scalar[]>(N);  // zeroed
    Scalar& operator[](size_t i) { return p[i]; }
    const Scalar& operator[](size_t i) const { return p[i]; }
    Scalar* begin() { return p.get(); }
    Scalar* end() { return p.get() + N; }

    void scaled_add(VecView<N> addend, Scalar scale);
};

template <size_t N>
struct VecView {
    const Scalar* p;
    VecView(const Vec<N>& v) : p(v.p.get()) {}
    explicit VecView(const Scalar* raw) : p(raw) {}
    const Scalar& operator[](size_t i) const { return p[i]; }
    const Scalar* begin() const { return p; }
    const Scalar* end() const { return p + N; }
};

template <size_t N>
void Vec<N>::scaled_add(VecView<N> addend, Scalar scale) {
    for (size_t i = 0; i < N; ++i) (*this)[i] += scale * addend[i];
}

template <size_t N>
struct MutVecView {
    Scalar* p;
    Scalar& operator[](size_t i) const { return p[i]; }
    Scalar* begin() const { return p; }
    Scalar* end() const { return p + N; }
};

// Runtime rows x compile-time columns. This is the activation tensor carried
// through the model: [tokens, channels] for decoder activations and
// [queries, projected channels] for attention intermediates.
template <size_t C>
class Matrix;

template <size_t C>
class MatrixView {
public:
    MatrixView(const Scalar* data, size_t rows) : data_(data), rows_(rows) {
        if (rows_ != 0 && data_ == nullptr) throw std::invalid_argument("MatrixView: null data");
    }
    MatrixView(const Matrix<C>& matrix) : MatrixView(matrix.data(), matrix.rows()) {}

    size_t rows() const { return rows_; }
    static constexpr size_t cols() { return C; }
    VecView<C> row(size_t i) const {
        check_row(i);
        return VecView<C>(data_ + i * C);
    }
    const Scalar* data() const { return data_; }

private:
    void check_row(size_t i) const {
        if (i >= rows_) throw std::out_of_range("MatrixView: row out of range");
    }

    const Scalar* data_;
    size_t rows_;
};

template <size_t C>
class MutableMatrixView {
public:
    MutableMatrixView(Scalar* data, size_t rows) : data_(data), rows_(rows) {
        if (rows_ != 0 && data_ == nullptr) throw std::invalid_argument("MutableMatrixView: null data");
    }

    size_t rows() const { return rows_; }
    static constexpr size_t cols() { return C; }
    VecView<C> row(size_t i) const {
        check_row(i);
        return VecView<C>(data_ + i * C);
    }
    MutVecView<C> row_mut(size_t i) const {
        check_row(i);
        return MutVecView<C>{data_ + i * C};
    }
    Scalar* data() const { return data_; }

private:
    void check_row(size_t i) const {
        if (i >= rows_) throw std::out_of_range("MutableMatrixView: row out of range");
    }

    Scalar* data_;
    size_t rows_;
};

template <size_t C>
class Matrix {
public:
    explicit Matrix(size_t rows) : rows_(rows), data_(element_count(rows), Scalar(0)) {}

    size_t rows() const { return rows_; }
    static constexpr size_t cols() { return C; }
    MatrixView<C> view() const { return MatrixView<C>(data_.data(), rows_); }
    MutableMatrixView<C> mutable_view() { return MutableMatrixView<C>(data_.data(), rows_); }

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
    const Scalar* data() const { return data_.data(); }
    Scalar* data() { return data_.data(); }

private:
    static size_t element_count(size_t rows) {
        if constexpr (C != 0) {
            if (rows > std::numeric_limits<size_t>::max() / C) throw std::length_error("Matrix: shape overflows size_t");
        }
        return rows * C;
    }
    void check_row(size_t i) const {
        if (i >= rows_) throw std::out_of_range("Matrix: row out of range");
    }

    size_t rows_;
    std::vector<Scalar> data_;
};

// Transitional source compatibility for modality/vision code. New algebra and
// model code should name the mathematical object directly as Matrix<C>.
template <size_t C>
using TokenMatrix = Matrix<C>;

template <size_t R, size_t C>
struct Mat {  // concept layout: y = x·W
    static constexpr size_t rows = R, cols = C;
    std::unique_ptr<Scalar[]> p = std::make_unique<Scalar[]>(R * C);
    Scalar& operator()(size_t r, size_t c) { return p[r * C + c]; }
    const Scalar& operator()(size_t r, size_t c) const { return p[r * C + c]; }
    VecView<C> row(size_t i) const { return VecView<C>(&p[i * C]); }
    Scalar* raw() { return p.get(); }
};

enum class TokenId : int32_t {};
struct Position {
    size_t i;
};

// Two head-index types that must never be confused (GQA). A KvHead is only
// ever constructed by dividing a query-head index by the group size, which
// attention.hpp does in exactly one place.
struct QHead {
    size_t i;
};
struct KvHead {
    size_t i;
};

// ================= stratum 1: algebra ======================================

template <size_t N>
Vec<N> operator+(VecView<N> a, VecView<N> b) {
    Vec<N> y;
    for (size_t i = 0; i < N; ++i) y[i] = a[i] + b[i];
    return y;
}
template <size_t N>
void operator+=(Vec<N>& y, VecView<N> x) {
    for (size_t i = 0; i < N; ++i) y[i] += x[i];
}
template <size_t N>
void operator+=(Vec<N>& y, const Vec<N>& x) {
    for (size_t i = 0; i < N; ++i) y[i] += x[i];
}
template <size_t N>
Vec<N> copy(VecView<N> v) {
    Vec<N> y;
    for (size_t i = 0; i < N; ++i) y[i] = v[i];
    return y;
}
template <size_t N>
Scalar dot(VecView<N> a, VecView<N> b) {
    Scalar s = 0;
    for (size_t i = 0; i < N; ++i) s += a[i] * b[i];
    return s;  // fp32 accum (T6)
}
template <size_t N>
Vec<N> hadamard(VecView<N> left, VecView<N> right) {
    Vec<N> output;
    for (size_t i = 0; i < N; ++i) output[i] = left[i] * right[i];
    return output;
}

template <size_t N>
void scale_in_place(Vec<N>& vector, Scalar scale) {
    for (size_t i = 0; i < N; ++i) vector[i] *= scale;
}

template <size_t N>
Vec<N> scaled_sum(VecView<N> left, VecView<N> right, Scalar scale) {
    Vec<N> output = left + right;
    scale_in_place(output, scale);
    return output;
}

template <size_t N>
Matrix<N> copy(MatrixView<N> input) {
    Matrix<N> output(input.rows());
    std::copy(input.data(), input.data() + input.rows() * N, output.data());
    return output;
}

template <size_t N>
Matrix<N> add(MatrixView<N> left, MatrixView<N> right) {
    if (left.rows() != right.rows()) throw std::invalid_argument("add: matrix row mismatch");
    Matrix<N> output(left.rows());
    const size_t elements = left.rows() * N;
    for (size_t i = 0; i < elements; ++i) output.data()[i] = left.data()[i] + right.data()[i];
    return output;
}

template <size_t N>
void add_in_place(MutableMatrixView<N> output, MatrixView<N> addend) {
    if (output.rows() != addend.rows()) throw std::invalid_argument("add_in_place: matrix row mismatch");
    const size_t elements = output.rows() * N;
    for (size_t i = 0; i < elements; ++i) output.data()[i] += addend.data()[i];
}

template <size_t N>
void add_bias_in_place(MutableMatrixView<N> output, VecView<N> bias) {
    for (size_t row = 0; row < output.rows(); ++row)
        for (size_t channel = 0; channel < N; ++channel) output.row_mut(row)[channel] += bias[channel];
}

template <size_t N>
Matrix<N> hadamard(MatrixView<N> left, MatrixView<N> right) {
    if (left.rows() != right.rows()) throw std::invalid_argument("hadamard: matrix row mismatch");
    Matrix<N> output(left.rows());
    const size_t elements = left.rows() * N;
    for (size_t i = 0; i < elements; ++i) output.data()[i] = left.data()[i] * right.data()[i];
    return output;
}

template <size_t N>
Vec<N> clamp(VecView<N> input, Scalar minimum, Scalar maximum) {
    if (minimum > maximum) throw std::invalid_argument("clamp: reversed interval");
    Vec<N> output;
    for (size_t channel = 0; channel < N; ++channel) output[channel] = std::clamp(input[channel], minimum, maximum);
    return output;
}

template <size_t N>
Matrix<N> clamp(MatrixView<N> input, Scalar minimum, Scalar maximum) {
    if (minimum > maximum) throw std::invalid_argument("clamp: reversed interval");
    Matrix<N> output(input.rows());
    const size_t elements = input.rows() * N;
    for (size_t i = 0; i < elements; ++i) output.data()[i] = std::clamp(input.data()[i], minimum, maximum);
    return output;
}

template <size_t Width, size_t Total>
Matrix<Width> slice_columns(MatrixView<Total> input, size_t first_column) {
    if (first_column > Total || Width > Total - first_column) throw std::out_of_range("slice_columns: column range out of bounds");
    Matrix<Width> output(input.rows());
    for (size_t row = 0; row < input.rows(); ++row) std::copy(input.row(row).begin() + first_column, input.row(row).begin() + first_column + Width, output.row_mut(row).begin());
    return output;
}

template <size_t Heads, size_t HeadDim, class Transform>
Vec<Heads * HeadDim> transform_heads(VecView<Heads * HeadDim> input, Transform&& transform) {
    Vec<Heads * HeadDim> output;
    for (size_t head = 0; head < Heads; ++head) {
        Vec<HeadDim> transformed = std::invoke(transform, VecView<HeadDim>(input.begin() + head * HeadDim));
        std::copy(transformed.begin(), transformed.end(), output.begin() + head * HeadDim);
    }
    return output;
}

template <size_t Heads, size_t HeadDim, class Transform>
Matrix<Heads * HeadDim> transform_heads(MatrixView<Heads * HeadDim> input, Transform&& transform) {
    Matrix<Heads * HeadDim> output(input.rows());
    for (size_t row = 0; row < input.rows(); ++row) {
        Vec<Heads * HeadDim> transformed = transform_heads<Heads, HeadDim>(input.row(row), transform);
        output.set_row(row, transformed);
    }
    return output;
}

template <size_t N>
void scale_in_place(MutableMatrixView<N> matrix, Scalar scale) {
    const size_t elements = matrix.rows() * N;
    for (size_t i = 0; i < elements; ++i) matrix.data()[i] *= scale;
}

template <size_t N>
Matrix<N> scaled_sum(MatrixView<N> left, MatrixView<N> right, Scalar scale) {
    Matrix<N> output = add(left, right);
    scale_in_place(output.mutable_view(), scale);
    return output;
}

template <size_t N>
Matrix<N> rms_norm(MatrixView<N> input, VecView<N> gamma, Scalar eps) {
    Matrix<N> output(input.rows());
    for (size_t row = 0; row < input.rows(); ++row) {
        Scalar mean_square = 0;
        for (size_t channel = 0; channel < N; ++channel) mean_square += input.row(row)[channel] * input.row(row)[channel];
        const Scalar inverse_rms = 1.f / std::sqrt(mean_square / Scalar(N) + eps);
        for (size_t channel = 0; channel < N; ++channel) output.row_mut(row)[channel] = gamma[channel] * input.row(row)[channel] * inverse_rms;
    }
    return output;
}

template <size_t N>
Vec<N> rms_norm(VecView<N> input, VecView<N> gamma, Scalar eps) {
    Scalar mean_square = 0;
    for (size_t channel = 0; channel < N; ++channel) mean_square += input[channel] * input[channel];
    const Scalar inverse_rms = 1.f / std::sqrt(mean_square / Scalar(N) + eps);
    Vec<N> output;
    for (size_t channel = 0; channel < N; ++channel) output[channel] = gamma[channel] * input[channel] * inverse_rms;
    return output;
}

template <size_t N>
Matrix<N> rms_norm(MatrixView<N> input, Scalar eps) {
    Matrix<N> output(input.rows());
    for (size_t row = 0; row < input.rows(); ++row) {
        Scalar mean_square = 0;
        for (size_t channel = 0; channel < N; ++channel) mean_square += input.row(row)[channel] * input.row(row)[channel];
        const Scalar inverse_rms = 1.f / std::sqrt(mean_square / Scalar(N) + eps);
        for (size_t channel = 0; channel < N; ++channel) output.row_mut(row)[channel] = input.row(row)[channel] * inverse_rms;
    }
    return output;
}

template <size_t N>
Vec<N> rms_norm(VecView<N> input, Scalar eps) {
    Scalar mean_square = 0;
    for (size_t channel = 0; channel < N; ++channel) mean_square += input[channel] * input[channel];
    const Scalar inverse_rms = 1.f / std::sqrt(mean_square / Scalar(N) + eps);
    Vec<N> output;
    for (size_t channel = 0; channel < N; ++channel) output[channel] = input[channel] * inverse_rms;
    return output;
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

inline void softmax(std::span<Scalar> s) {
    Scalar mx = *std::max_element(s.begin(), s.end()), sum = 0;
    for (auto& v : s) {
        v = std::exp(v - mx);
        sum += v;
    }
    for (auto& v : s) v /= sum;
}
template <size_t N>
void gelu(Vec<N>& v) {
    for (size_t i = 0; i < N; ++i) {
        Scalar x = v[i];
        v[i] = 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}
inline Scalar sigmoid(Scalar x) { return 1.f / (1.f + std::exp(-x)); }

template <size_t N>
void silu(Vec<N>& v) {  // z·sigmoid(z)
    for (size_t i = 0; i < N; ++i) v[i] = v[i] / (1.f + std::exp(-v[i]));
}
template <size_t N>
void operator*=(Vec<N>& y, const Vec<N>& x) {
    for (size_t i = 0; i < N; ++i) y[i] *= x[i];  // Hadamard
}

template <size_t N>
void gelu_in_place(MutableMatrixView<N> matrix) {
    const size_t elements = matrix.rows() * N;
    for (size_t i = 0; i < elements; ++i) {
        const Scalar x = matrix.data()[i];
        matrix.data()[i] = 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

template <size_t N>
void silu_in_place(MutableMatrixView<N> matrix) {
    const size_t elements = matrix.rows() * N;
    for (size_t i = 0; i < elements; ++i) {
        const Scalar x = matrix.data()[i];
        matrix.data()[i] = x / (1.f + std::exp(-x));
    }
}

// ---- the parallel seams (M5 replaces the bodies with a thread pool) --------
void par_for(size_t n, const std::function<void(size_t)>& f);  // core.cpp
size_t nm_num_threads();                                       // core.cpp

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

// Dense multi-head attention:
//   softmax_rows(scale * Q K^T) V
// Q/K/V are [tokens, heads * head_dimension]. Some architectures normalize
// Q/K and use a scale of one (Gemma vision); conventional attention passes
// 1/sqrt(head_dimension). This reference kernel preserves the mathematical
// interface while a backend may fuse and tile the operation.
template <size_t Heads, size_t QueryKeyDim, size_t ValueDim>
Matrix<Heads * ValueDim> scaled_dot_product_attention(MatrixView<Heads * QueryKeyDim> queries, MatrixView<Heads * QueryKeyDim> keys, MatrixView<Heads * ValueDim> values, Scalar scale) {
    if (keys.rows() != values.rows()) throw std::invalid_argument("scaled_dot_product_attention: K/V row mismatch");

    Matrix<Heads * ValueDim> output(queries.rows());
    par_for(queries.rows() * Heads, [&](size_t task) {
        const size_t query_row = task / Heads;
        const size_t head = task % Heads;
        const VecView<QueryKeyDim> query = slice<QueryKeyDim>(queries.row(query_row), head * QueryKeyDim);

        std::vector<Scalar> scores(keys.rows());
        for (size_t key_row = 0; key_row < keys.rows(); ++key_row) scores[key_row] = scale * dot(query, slice<QueryKeyDim>(keys.row(key_row), head * QueryKeyDim));
        softmax(scores);

        Vec<ValueDim> attended;
        for (size_t key_row = 0; key_row < keys.rows(); ++key_row) attended.scaled_add(slice<ValueDim>(values.row(key_row), head * ValueDim), scores[key_row]);
        std::copy(attended.begin(), attended.end(), output.data() + (query_row * Heads + head) * ValueDim);
    });
    return output;
}

// ---- MatT<In,Out>: the ggml/llama.cpp weight layout (row = output) ---------
// Owns nothing by default: `p` may point into an mmap'd GGUF file. When a
// tensor must be materialised as fp32 (F16 dequant, or the debug dequant path
// for quantised tensors), `own` holds the storage and `p` points into it.
template <size_t In, size_t Out>
struct MatT {
    const Scalar* p = nullptr;      // [Out*In], row o at p + o*In
    std::shared_ptr<Scalar[]> own;  // optional backing; null => view
    VecView<In> row(size_t o) const { return VecView<In>(p + o * In); }

    static MatT owning() {  // allocate [Out*In], zeroed
        MatT m;
        m.own = std::shared_ptr<Scalar[]>(new Scalar[In * Out]());
        m.p = m.own.get();
        return m;
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
