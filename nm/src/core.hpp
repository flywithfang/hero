// core.hpp — strata 0 (typed storage) and 1 (algebra).
//
// Split out of gpt_pipeline_v10.cpp, preserving its design. This is the ONLY
// place layout knowledge lives. Two matrix layouts coexist here and their
// difference is the whole of trap T1 (see INFERENCE_PLAN.md §M1):
//
//   Mat<R,C>   row-major, y = x·W  (In dimension outer).      concept default.
//   MatT<In,Out> row-major with row = one OUTPUT's weights (length In),
//                contiguous per output. y = W.matvec(x). This is EXACTLY
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
#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
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

    void scale(Scalar factor) {
        for (size_t i = 0; i < N; ++i) p[i] *= factor;
    }
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
    MutVecView(Vec<N>& v) : p(v.begin()) {}
    explicit MutVecView(Scalar* raw) : p(raw) {}
    Scalar& operator[](size_t i) const { return p[i]; }
    Scalar* begin() const { return p; }
    Scalar* end() const { return p + N; }

    Scalar* p;
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
    MatrixView(const Matrix<C>& matrix);

    size_t rows() const { return rows_; }
    static constexpr size_t cols() { return C; }
    VecView<C> row(size_t i) const {
        check_row(i);
        return VecView<C>(data_ + i * C);
    }
    // The tail of the sequence, still a view. A range of rows is the only
    // thing callers ever wanted the base pointer for.
    MatrixView from_row(size_t first) const {
        if (first > rows_) throw std::out_of_range("MatrixView: row out of range");
        return MatrixView(data_ + first * C, rows_ - first);
    }

private:
    void check_row(size_t i) const {
        if (i >= rows_) throw std::out_of_range("MatrixView: row out of range");
    }

    const Scalar* data_;
    size_t rows_;
};

// A matrix is a SEQUENCE OF ROWS, and a row is always a complete vector. There
// is no "allocate T empty rows and fill them in later": a row arrives whole
// through append(), or the entire buffer arrives already written through
// from_row_major(). A Matrix is therefore never observable half-built, and
// row-major layout is knowledge that lives in this class and nowhere else —
// nothing outside it ever holds the base pointer.
template <size_t C>
class Matrix {
    static_assert(C != 0, "a row must have at least one column");

public:
    Matrix() = default;

    // Rows written out literally, for fixtures and tests. Still "a row is a
    // whole vector", just spelled inline.
    Matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
        reserve(rows.size());
        for (const std::initializer_list<Scalar>& row : rows) {
            if (row.size() != C) throw std::invalid_argument("Matrix: literal row has the wrong width");
            values_.insert(values_.end(), row.begin(), row.end());
        }
    }

    // ---- building -----------------------------------------------------------
    void append(VecView<C> row) { values_.insert(values_.end(), row.begin(), row.end()); }
    void append(const Vec<C>& row) { append(VecView<C>(row)); }
    void append(MatrixView<C> rows) {
        reserve(this->rows() + rows.rows());
        for (size_t i = 0; i < rows.rows(); ++i) append(rows.row(i));
    }
    void reserve(size_t rows) { values_.reserve(element_count(rows)); }

    // A complete, already-laid-out buffer. This exists for the kernels that do
    // not produce whole rows in order — an output-stationary matmul holds one
    // weight row and sweeps a COLUMN of the result — and it is the only way in
    // besides append(). Every element is written before the matrix exists, so
    // the invariant above still holds.
    static Matrix from_row_major(std::vector<Scalar> values) {
        if (values.size() % C != 0) throw std::invalid_argument("Matrix: buffer is not a whole number of rows");
        Matrix matrix;
        matrix.values_ = std::move(values);
        return matrix;
    }

    // ---- reading ------------------------------------------------------------
    size_t rows() const { return values_.size() / C; }
    static constexpr size_t cols() { return C; }
    VecView<C> row(size_t i) const {
        check_row(i);
        return VecView<C>(values_.data() + i * C);
    }
    MatrixView<C> view() const { return MatrixView<C>(values_.data(), rows()); }

    // ---- whole-matrix maps --------------------------------------------------
    // Rewriting every row is the only way a row changes once it is in, and the
    // shape cannot change while it happens. Activations, gates and RoPE are
    // exactly this. The transform receives one whole row, and its index too if
    // it asks for one.
    template <class F>
    void transform_rows(F&& transform) {
        for (size_t i = 0; i < rows(); ++i) {
            MutVecView<C> row{values_.data() + i * C};
            if constexpr (std::is_invocable_v<F&, size_t, MutVecView<C>>)
                transform(i, row);
            else
                transform(row);
        }
    }
    void scale(Scalar factor) {
        for (Scalar& element : values_) element *= factor;
    }

    static size_t element_count(size_t rows) {
        if (rows > std::numeric_limits<size_t>::max() / C) throw std::length_error("Matrix: shape overflows size_t");
        return rows * C;
    }

private:
    void check_row(size_t i) const {
        if (i >= rows()) throw std::out_of_range("Matrix: row out of range");
    }

    std::vector<Scalar> values_;
};

template <size_t C>
MatrixView<C>::MatrixView(const Matrix<C>& matrix) : MatrixView(matrix.view()) {}

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
Vec<N> scaled_sum(VecView<N> left, VecView<N> right, Scalar scale) {
    Vec<N> output = left + right;
    output.scale(scale);
    return output;
}

template <size_t N>
Vec<N> clamp(VecView<N> input, Scalar minimum, Scalar maximum) {
    if (minimum > maximum) throw std::invalid_argument("clamp: reversed interval");
    Vec<N> output;
    for (size_t channel = 0; channel < N; ++channel) output[channel] = std::clamp(input[channel], minimum, maximum);
    return output;
}

template <size_t W_, size_t N>
VecView<W_> slice(VecView<N> v, size_t off) {
    static_assert(W_ <= N);
    return VecView<W_>(&v[off]);
}
template <size_t W_, size_t N>
MutVecView<W_> slice_mut(MutVecView<N> v, size_t off) {
    static_assert(W_ <= N);
    return MutVecView<W_>(&v[off]);
}
template <size_t W_, size_t N>
MutVecView<W_> slice_mut(Vec<N>& v, size_t off) {
    return slice_mut<W_>(MutVecView<N>(v), off);
}

// RMSNorm is a row operation.
template <size_t N, class Scale>
void rms_norm_row(VecView<N> input, Scalar eps, MutVecView<N> output, Scale&& scale) {
    Scalar mean_square = 0;
    for (size_t channel = 0; channel < N; ++channel) mean_square += input[channel] * input[channel];
    const Scalar inverse_rms = 1.f / std::sqrt(mean_square / Scalar(N) + eps);
    for (size_t channel = 0; channel < N; ++channel) output[channel] = scale(channel) * input[channel] * inverse_rms;
}

template <size_t N>
Vec<N> rms_norm(VecView<N> input, VecView<N> gamma, Scalar eps) {
    Vec<N> output;
    rms_norm_row(input, eps, MutVecView<N>(output), [&](size_t channel) { return gamma[channel]; });
    return output;
}

template <size_t N>
Vec<N> rms_norm(VecView<N> input, Scalar eps) {
    Vec<N> output;
    rms_norm_row(input, eps, MutVecView<N>(output), [](size_t) { return 1.f; });
    return output;
}

// ---- the same algebra over a sequence of rows ------------------------------
//
// Every one of these is its row operation applied to each row in turn, and
// says so in one line. That is why the matrix and vector forms cannot drift
// apart, and why no kernel here indexes a flat buffer.

template <size_t N>
Matrix<N> copy(MatrixView<N> input) {
    Matrix<N> output;
    output.append(input);
    return output;
}

template <size_t N>
Matrix<N> add(MatrixView<N> left, MatrixView<N> right) {
    if (left.rows() != right.rows()) throw std::invalid_argument("add: matrix row mismatch");
    Matrix<N> output;
    output.reserve(left.rows());
    for (size_t row = 0; row < left.rows(); ++row) output.append(left.row(row) + right.row(row));
    return output;
}

template <size_t N>
void add_bias_in_place(Matrix<N>& output, VecView<N> bias) {
    output.transform_rows([&](MutVecView<N> row) {
        for (size_t channel = 0; channel < N; ++channel) row[channel] += bias[channel];
    });
}

template <size_t N>
Matrix<N> hadamard(MatrixView<N> left, MatrixView<N> right) {
    if (left.rows() != right.rows()) throw std::invalid_argument("hadamard: matrix row mismatch");
    Matrix<N> output;
    output.reserve(left.rows());
    for (size_t row = 0; row < left.rows(); ++row) output.append(hadamard(left.row(row), right.row(row)));
    return output;
}

template <size_t N>
Matrix<N> clamp(MatrixView<N> input, Scalar minimum, Scalar maximum) {
    if (minimum > maximum) throw std::invalid_argument("clamp: reversed interval");
    Matrix<N> output;
    output.reserve(input.rows());
    for (size_t row = 0; row < input.rows(); ++row) output.append(clamp(input.row(row), minimum, maximum));
    return output;
}

template <size_t Width, size_t Total>
Matrix<Width> slice_columns(MatrixView<Total> input, size_t first_column) {
    if (first_column > Total || Width > Total - first_column) throw std::out_of_range("slice_columns: column range out of bounds");
    Matrix<Width> output;
    output.reserve(input.rows());
    for (size_t row = 0; row < input.rows(); ++row) output.append(slice<Width>(input.row(row), first_column));
    return output;
}

template <size_t N>
Matrix<N> scaled_sum(MatrixView<N> left, MatrixView<N> right, Scalar scale) {
    Matrix<N> output = add(left, right);
    output.scale(scale);
    return output;
}

template <size_t N>
Matrix<N> rms_norm(MatrixView<N> input, VecView<N> gamma, Scalar eps) {
    Matrix<N> output;
    output.reserve(input.rows());
    for (size_t row = 0; row < input.rows(); ++row) output.append(rms_norm(input.row(row), gamma, eps));
    return output;
}

template <size_t N>
Matrix<N> rms_norm(MatrixView<N> input, Scalar eps) {
    Matrix<N> output;
    output.reserve(input.rows());
    for (size_t row = 0; row < input.rows(); ++row) output.append(rms_norm(input.row(row), eps));
    return output;
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
void gelu(MutVecView<N> v) {
    for (size_t i = 0; i < N; ++i) {
        Scalar x = v[i];
        v[i] = 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}
inline Scalar sigmoid(Scalar x) { return 1.f / (1.f + std::exp(-x)); }

template <size_t N>
void silu(MutVecView<N> v) {  // z·sigmoid(z)
    for (size_t i = 0; i < N; ++i) v[i] = v[i] / (1.f + std::exp(-v[i]));
}
// An owned vector converts to a mutable view, but deduction cannot see that,
// so both activations get the one-line forwarding overload.
template <size_t N>
void gelu(Vec<N>& v) {
    gelu(MutVecView<N>(v));
}
template <size_t N>
void silu(Vec<N>& v) {
    silu(MutVecView<N>(v));
}
template <size_t N>
void operator*=(Vec<N>& y, const Vec<N>& x) {
    for (size_t i = 0; i < N; ++i) y[i] *= x[i];  // Hadamard
}

template <size_t N>
void gelu_in_place(Matrix<N>& matrix) {
    matrix.transform_rows([](MutVecView<N> row) { gelu(row); });
}

template <size_t N>
void silu_in_place(Matrix<N>& matrix) {
    matrix.transform_rows([](MutVecView<N> row) { silu(row); });
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

// The same seam at matrix granularity: f(row) returns that whole row. Use this
// wherever rows are independent and a plain append loop would serialize them.
template <size_t C, class F>
Matrix<C> par_map_rows(size_t rows, F&& f) {
    std::vector<Scalar> values(Matrix<C>::element_count(rows));
    par_for(rows, [&](size_t row) {
        Vec<C> produced = f(row);
        std::copy(produced.begin(), produced.end(), values.data() + row * C);
    });
    return Matrix<C>::from_row_major(std::move(values));
}

// And over (row, head): f(row, head) returns that head's partition of that
// row. Rows and heads are independent, so this is ONE flat task space rather
// than a map over rows containing a map over heads — which matters because a
// decode step has a single row and all of its parallelism across heads.
//
// These two are the only places that place anything into a matrix buffer by
// index; every other producer in the engine hands over whole rows.
template <size_t Heads, size_t Dim, class F>
Matrix<Heads * Dim> par_map_heads(size_t rows, F&& f) {
    std::vector<Scalar> values(Matrix<Heads * Dim>::element_count(rows));
    par_for(rows * Heads, [&](size_t task) {
        const size_t row = task / Heads;
        const size_t head = task % Heads;
        Vec<Dim> partition = f(row, head);
        std::copy(partition.begin(), partition.end(), values.data() + (row * Heads + head) * Dim);
    });
    return Matrix<Heads * Dim>::from_row_major(std::move(values));
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

    return par_map_heads<Heads, ValueDim>(queries.rows(), [&](size_t query_row, size_t head) {
        const VecView<QueryKeyDim> query = slice<QueryKeyDim>(queries.row(query_row), head * QueryKeyDim);

        std::vector<Scalar> scores(keys.rows());
        for (size_t key_row = 0; key_row < keys.rows(); ++key_row) scores[key_row] = scale * dot(query, slice<QueryKeyDim>(keys.row(key_row), head * QueryKeyDim));
        softmax(scores);

        Vec<ValueDim> attended;
        for (size_t key_row = 0; key_row < keys.rows(); ++key_row) attended.scaled_add(slice<ValueDim>(values.row(key_row), head * ValueDim), scores[key_row]);
        return attended;
    });
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

    // y = W x: Out independent dot products of length In, y[o] = dot(x, row(o)).
    // [BANDWIDTH] streams the whole weight once; [COMPUTE] In*Out MACs.
    Vec<Out> matvec(VecView<In> x) const {
        Vec<Out> y;
        par_for(Out, [&](size_t o) { y[o] = dot(x, row(o)); });
        return y;
    }

    static MatT owning() {  // allocate [Out*In], zeroed
        MatT m;
        m.own = std::shared_ptr<Scalar[]>(new Scalar[In * Out]());
        m.p = m.own.get();
        return m;
    }
    Scalar* raw() { return const_cast<Scalar*>(p); }
};

