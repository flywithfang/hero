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
#include <compare>
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

// The storage types, declared up front because each hands out views over its
// own buffer and must therefore be a friend of the view types. That friend list
// IS the rule "layout knowledge lives only in storage types", written where the
// compiler can enforce it: a view over a raw address can be built by these and
// by nothing else.
template <size_t N>
struct VecView;
template <size_t N>
struct MutVecView;
template <size_t C>
class Matrix;
template <size_t C>
class MatrixView;
template <size_t R, size_t C>
struct Mat;
template <size_t In, size_t Out>
struct MatT;

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
    void clamp(Scalar minimum, Scalar maximum) {
        if (minimum > maximum) throw std::invalid_argument("clamp: reversed interval");
        for (size_t i = 0; i < N; ++i) p[i] = std::clamp(p[i], minimum, maximum);
    }
    template <size_t Width>
    void replace_partition(size_t partition, VecView<Width> value) {
        static_assert(N % Width == 0);
        if (partition >= N / Width) throw std::out_of_range("Vec: partition out of range");
        for (size_t i = 0; i < Width; ++i) p[partition * Width + i] = value[i];
    }
    void scaled_add(VecView<N> addend, Scalar scale);
    void operator+=(VecView<N> addend);
    void operator*=(VecView<N> factors);  // Hadamard, in place
};

// An operation on ONE object is that object's method; an owner mutates itself
// and a view, which cannot, derives a new value. `Vec::scale` and
// `VecView::clamp` are the two shapes of that rule. Operations over two peers
// (`dot`, `hadamard`), and any parameterised by a compile-time Width (`slice`),
// stay free functions in stratum 1 below.
//
// Members also spare every call site a conversion: `N` cannot be deduced from
// `Vec<N>` through the user-defined conversion to `VecView<N>`, so a free
// function must be handed `VecView<N>(v)` by hand, while a member already knows
// its own `N` and lets the argument convert.
template <size_t N>
struct VecView {
    VecView(const Vec<N>& v) : p(v.p.get()) {}
    const Scalar& operator[](size_t i) const { return p[i]; }
    const Scalar* begin() const { return p; }
    const Scalar* end() const { return p + N; }

    Vec<N> copy() const {
        Vec<N> y;
        for (size_t i = 0; i < N; ++i) y[i] = p[i];
        return y;
    }
    Vec<N> clamp(Scalar minimum, Scalar maximum) const {
        Vec<N> y = copy();
        y.clamp(minimum, maximum);
        return y;
    }
    // One Width-wide partition, counted in partitions. This is the ONE place a
    // partition index becomes an address, which is why it is a method: the free
    // `slice` below is a wrapper over it, so `.template` is written once here
    // rather than at each of its two dozen dependent call sites.
    template <size_t Width>
    VecView<Width> partition(size_t base, size_t index) const {
        if (base + Width > N || index >= (N - base) / Width) throw std::out_of_range("VecView: partition index out of range");
        return VecView<Width>(p + base + index * Width);
    }

private:
    explicit VecView(const Scalar* raw) : p(raw) {}

    template <size_t>
    friend struct VecView;
    template <size_t>
    friend struct MutVecView;
    template <size_t>
    friend class Matrix;
    template <size_t>
    friend class MatrixView;
    template <size_t, size_t>
    friend struct Mat;
    template <size_t, size_t>
    friend struct MatT;

    const Scalar* p;
};

template <size_t N>
void Vec<N>::scaled_add(VecView<N> addend, Scalar scale) {
    for (size_t i = 0; i < N; ++i) (*this)[i] += scale * addend[i];
}
template <size_t N>
void Vec<N>::operator+=(VecView<N> addend) {
    for (size_t i = 0; i < N; ++i) p[i] += addend[i];
}
template <size_t N>
void Vec<N>::operator*=(VecView<N> factors) {
    for (size_t i = 0; i < N; ++i) p[i] *= factors[i];
}

template <size_t N>
struct MutVecView {
    MutVecView(Vec<N>& v) : p(v.begin()) {}
    Scalar& operator[](size_t i) const { return p[i]; }
    Scalar* begin() const { return p; }
    Scalar* end() const { return p + N; }
    // Writable implies readable, so an in-place kernel can name its own input
    // without reconstructing a view from a raw pointer.
    operator VecView<N>() const { return VecView<N>(p); }

    template <size_t Width>
    MutVecView<Width> partition(size_t base, size_t index) const {
        if (base + Width > N || index >= (N - base) / Width) throw std::out_of_range("MutVecView: partition index out of range");
        return MutVecView<Width>(p + base + index * Width);
    }

private:
    explicit MutVecView(Scalar* raw) : p(raw) {}

    template <size_t>
    friend struct MutVecView;
    template <size_t>
    friend class Matrix;

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
    // One row, still shaped as a sequence — for the kernels written against a
    // [T,C] input that a decode step calls with T == 1.
    MatrixView single_row(size_t i) const {
        check_row(i);
        return MatrixView(data_ + i * C, 1);
    }

    // Each of these is its VecView operation applied to every row, and says so
    // in one line — which is why the vector and matrix forms cannot drift apart
    // and share a spelling at the call site. Defined below, once Matrix<C> is
    // complete.
    Matrix<C> copy() const;
    Matrix<C> clamp(Scalar minimum, Scalar maximum) const;
    Matrix<C> operator+(MatrixView right) const;

private:
    MatrixView(const Scalar* data, size_t rows) : data_(data), rows_(rows) {
        if (rows_ != 0 && data_ == nullptr) throw std::invalid_argument("MatrixView: null data");
    }
    void check_row(size_t i) const {
        if (i >= rows_) throw std::out_of_range("MatrixView: row out of range");
    }

    template <size_t>
    friend class Matrix;

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
    // Replace one complete row without exposing the scalar buffer. Shape does
    // not change, and no partially-written row is ever observable.
    void replace(size_t i, VecView<C> row) {
        check_row(i);
        for (size_t column = 0; column < C; ++column) values_[i * C + column] = row[column];
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
    static Matrix zero_rows(size_t rows) {
        Matrix matrix;
        matrix.values_.resize(element_count(rows));
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
    void clamp(Scalar minimum, Scalar maximum) {
        if (minimum > maximum) throw std::invalid_argument("clamp: reversed interval");
        for (Scalar& element : values_) element = std::clamp(element, minimum, maximum);
    }
    // One vector added to every row. The old free `add_bias_in_place` named its
    // own mutation because it could not be a method; now it is one.
    void add_bias(VecView<C> bias) {
        transform_rows([&](MutVecView<C> row) {
            for (size_t channel = 0; channel < C; ++channel) row[channel] += bias[channel];
        });
    }
    void fill(Scalar value) { std::fill(values_.begin(), values_.end(), value); }
    // Both operands are peers, so this would be a free function by the rule
    // above — but a free operator+ cannot deduce C through Matrix -> MatrixView,
    // and `H + F` must work whether either side is owned or a view. Two members
    // covering the four combinations is the smaller price.
    Matrix operator+(MatrixView<C> right) const { return view() + right; }
    template <size_t Width>
    void replace_partition(size_t row, size_t partition, VecView<Width> value) {
        static_assert(C % Width == 0);
        check_row(row);
        if (partition >= C / Width) throw std::out_of_range("Matrix: partition out of range");
        for (size_t i = 0; i < Width; ++i) values_[row * C + partition * Width + i] = value[i];
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

template <size_t C>
Matrix<C> MatrixView<C>::copy() const {
    Matrix<C> output;
    output.append(*this);
    return output;
}
template <size_t C>
Matrix<C> MatrixView<C>::clamp(Scalar minimum, Scalar maximum) const {
    Matrix<C> output = copy();
    output.clamp(minimum, maximum);
    return output;
}
template <size_t C>
Matrix<C> MatrixView<C>::operator+(MatrixView<C> right) const {
    if (rows() != right.rows()) throw std::invalid_argument("Matrix: row count mismatch");
    Matrix<C> output;
    output.reserve(rows());
    for (size_t i = 0; i < rows(); ++i) output.append(row(i) + right.row(i));
    return output;
}

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

// A coordinate in the decoder sequence. It may be ordered, advanced by a row
// count, subtracted from a later position to obtain a distance, and scaled by a
// RoPE frequency. The representation is deliberately not part of its API.
class Position {
public:
    explicit constexpr Position(size_t value) : value_(value) {}

    friend constexpr auto operator<=>(Position, Position) = default;

    friend constexpr Position operator+(Position position, size_t offset) {
        if (offset > std::numeric_limits<size_t>::max() - position.value_) throw std::length_error("Position: advance overflows");
        return Position{position.value_ + offset};
    }

    friend constexpr size_t operator-(Position later, Position earlier) {
        if (later < earlier) throw std::invalid_argument("Position: negative distance");
        return later.value_ - earlier.value_;
    }

    friend constexpr Scalar operator*(Position position, Scalar scale) { return Scalar(position.value_) * scale; }

private:
    size_t value_;
};

// ================= stratum 1: algebra ======================================
//
// What is left here after the one-object operations moved onto the types: the
// operations over two peers with no owner between them, and the ones
// parameterised by a compile-time Width, which as methods would need
// `v.template slice<HeadDim>(head)` at every dependent call site.

// Free because a member operator+ on VecView is never found for `Vec + Vec`:
// operator lookup does not apply a user conversion to the implicit object
// argument.
template <size_t N>
Vec<N> operator+(VecView<N> a, VecView<N> b) {
    Vec<N> y;
    for (size_t i = 0; i < N; ++i) y[i] = a[i] + b[i];
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

// ---- a wide vector read as a sequence of Width-wide partitions --------------
//
// Heads inside a projection, layers inside the packed PLE table, one head
// inside one region of a packed [q|k|v] stream. The runtime argument is a
// PARTITION INDEX, never an element offset: `slice<HeadDim>(q, head)`, not
// `head * HeadDim`. Multiplying is this function's job, so a caller cannot get
// the stride wrong.
//
// Base is where the partitioning starts, and it is a compile-time constant
// because it is always a region boundary in a fixed packed layout — the only
// place an element offset is legitimate, and it is checked at compile time.
// Each of these is `partition<Width>` under a name that puts Width first and
// hides the `.template` the call site would otherwise have to write. The
// address arithmetic is the view's, not theirs; all that is left here is the
// compile-time check that Base is a region boundary that fits.
template <size_t Width, size_t Base = 0, size_t N>
VecView<Width> slice(VecView<N> v, size_t index) {
    static_assert(Base + Width <= N, "slice: the first partition does not fit");
    return v.template partition<Width>(Base, index);
}
template <size_t Width, size_t Base = 0, size_t N>
MutVecView<Width> slice_mut(MutVecView<N> v, size_t index) {
    static_assert(Base + Width <= N, "slice_mut: the first partition does not fit");
    return v.template partition<Width>(Base, index);
}
template <size_t Width, size_t Base = 0, size_t N>
MutVecView<Width> slice_mut(Vec<N>& v, size_t index) {
    return slice_mut<Width, Base>(MutVecView<N>(v), index);
}

// ---- the same algebra over a sequence of rows ------------------------------
//
// Every one of these is its row operation applied to each row in turn, and
// says so in one line. That is why the matrix and vector forms cannot drift
// apart, and why no kernel here indexes a flat buffer.

template <size_t N>
Matrix<N> hadamard(MatrixView<N> left, MatrixView<N> right) {
    if (left.rows() != right.rows()) throw std::invalid_argument("hadamard: matrix row mismatch");
    Matrix<N> output;
    output.reserve(left.rows());
    for (size_t row = 0; row < left.rows(); ++row) output.append(hadamard(left.row(row), right.row(row)));
    return output;
}

// The same partitioning applied to every row: `index` is a partition index, as
// it is for slice().
template <size_t Width, size_t Total>
Matrix<Width> slice_columns(MatrixView<Total> input, size_t index) {
    Matrix<Width> output;
    output.reserve(input.rows());
    for (size_t row = 0; row < input.rows(); ++row) output.append(slice<Width>(input.row(row), index));
    return output;
}

template <size_t N>
Matrix<N> scaled_sum(MatrixView<N> left, MatrixView<N> right, Scalar scale) {
    Matrix<N> output = left + right;
    output.scale(scale);
    return output;
}

// sigmoid is a scalar function of a scalar, like std::exp: no storage to be a
// method of, no state to be a type. Gelu/Silu/Softmax live in components.hpp.
inline Scalar sigmoid(Scalar x) { return 1.f / (1.f + std::exp(-x)); }

// ---- the parallel seams (M5 replaces the bodies with a thread pool) --------
void par_for(size_t n, const std::function<void(size_t)>& f);  // core.cpp
size_t nm_num_threads();                                       // core.cpp

// Parallel MAP: f(h) is a pure function of its index and by-value captures and
// RETURNS its partition; placement owned here. The thread/GPU backend seam.
template <size_t H, size_t Dim, class F>
Vec<H * Dim> par_map(F&& f) {
    Vec<H * Dim> out;
    par_for(H, [&](size_t h) { out.template replace_partition<Dim>(h, f(h)); });
    return out;
}

// The same seam at matrix granularity: f(row) returns that whole row. Use this
// wherever rows are independent and a plain append loop would serialize them.
template <size_t C, class F>
Matrix<C> par_map_rows(size_t rows, F&& f) {
    Matrix<C> output = Matrix<C>::zero_rows(rows);
    par_for(rows, [&](size_t row) { output.replace(row, f(row)); });
    return output;
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
    Matrix<Heads * Dim> output = Matrix<Heads * Dim>::zero_rows(rows);
    par_for(rows * Heads, [&](size_t task) {
        const size_t row = task / Heads;
        const size_t head = task % Heads;
        output.template replace_partition<Dim>(row, head, f(row, head));
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

