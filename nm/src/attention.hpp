// attention.hpp — the token-mixing core, model-neutral.
//
// Everything here is architecture-independent math: a position-indexed K/V
// cache, the visibility predicate that defines causal and sliding-window
// masking, grouped-query attention as a gather-and-reduce, rotary position
// embedding, and per-head normalization. A model family supplies dimensions,
// a RoPE table, and a window; it does not re-derive the reduction.
//
// The four attention LAWS hold throughout: q-dim == k-dim; v-dim is free;
// Wo in = Hq*Dv; Wo out = D. Hq*Dqk == D is a convention, not a law — Gemma 4
// breaks it, so nothing below assumes it.
#pragma once
#include "components.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

// Q/K normalization is `PerHeadNorm` from components.hpp: one HeadDim-wide
// scale vector walked across the heads. Modern families (Gemma 4, Qwen 3.x) all
// do this; it is what lets a model fold the 1/sqrt(HeadDim) score scale into a
// learned weight instead of a constant.

// ---- gated attention --------------------------------------------------------
//
// Some modern families emit a per-head gate from the SAME projection as the
// query, packed head-major as [q_0 | gate_0 | q_1 | gate_1 | ...]. Splitting it
// is pure layout, so it belongs here rather than in a family file.
template <size_t Heads, size_t HeadDim>
struct HeadPair {
    Matrix<Heads * HeadDim> first;
    Matrix<Heads * HeadDim> second;
};

template <size_t Heads, size_t HeadDim>
HeadPair<Heads, HeadDim> split_head_pairs(MatrixView<2 * Heads * HeadDim> packed) {
    HeadPair<Heads, HeadDim> split;
    split.first.reserve(packed.rows());
    split.second.reserve(packed.rows());
    for (size_t row = 0; row < packed.rows(); ++row) {
        const VecView<2 * Heads * HeadDim> source = packed.row(row);
        Vec<Heads * HeadDim> first, second;
        for (size_t head = 0; head < Heads; ++head) {
            for (size_t i = 0; i < HeadDim; ++i) {
                first[head * HeadDim + i] = source[head * 2 * HeadDim + i];
                second[head * HeadDim + i] = source[head * 2 * HeadDim + HeadDim + i];
            }
        }
        split.first.append(first);
        split.second.append(second);
    }
    return split;
}

// ---- rotary position embedding ----------------------------------------------
//
// Half-split (HF / "NEOX") pairing: plane i couples channels (i, i + HeadDim/2).
// `RotaryNum/RotaryDen` is the fraction of planes that rotate at all; the rest
// receive inv_freq = 0, hence cos=1/sin=0, and pass through unchanged. Exponents
// keep HeadDim as the denominator even when only part of the head rotates.
//
// Pairing is a storage convention, not a model property: a checkpoint whose Q/K
// weights were permuted at conversion time needs interleaved pairing instead,
// and using the wrong one produces plausible-looking but wrong attention.
template <size_t HeadDim, size_t Base, size_t RotaryNum = 1, size_t RotaryDen = 1>
class RotaryEmbedding {
    static_assert(HeadDim % 2 == 0);
    static_assert(RotaryDen != 0 && RotaryNum <= RotaryDen);
    static constexpr size_t Planes = HeadDim / 2;
    static constexpr size_t RotaryPlanes = RotaryNum * HeadDim / RotaryDen / 2;

public:
    RotaryEmbedding() {
        for (size_t i = 0; i < Planes; ++i){
                 inv_freq_[i] = i < RotaryPlanes ? Scalar(std::pow(double(Base), -double(2 * i) / double(HeadDim))) : 0.f;
        }
    }

    void apply(MutVecView<HeadDim> value, Position position) const {
        for (size_t i = 0; i < Planes; ++i) {
            const Scalar angle = position * inv_freq_[i];
            const Scalar c = std::cos(angle), s = std::sin(angle);
            const Scalar x = value[i], y = value[i + Planes];
            value[i] = x * c - y * s;
            value[i + Planes] = y * c + x * s;
        }
    }

    static constexpr size_t rotary_planes() { return RotaryPlanes; }

private:
    std::array<Scalar, Planes> inv_freq_{};
};

// A rotation is, to rotate_heads, exactly what a norm is to PerHeadNorm: an
// in-place rewrite of one head-wide partition, here parameterized by position.
// Any table with that operation works — the two Gemma widths, the two bases,
// full or partial rotation — and nothing else does.
template <class R, size_t HeadDim>
concept HeadRotation = requires(const R& rope, MutVecView<HeadDim> head, Position position) { rope.apply(head, position); };

template <size_t Heads, size_t HeadDim, HeadRotation<HeadDim> Rope>
void rotate_heads(MutVecView<Heads * HeadDim> value, const Rope& rope, Position position) {
    for (size_t h = 0; h < Heads; ++h) {
        rope.apply(slice_mut<HeadDim>(value, h), position);
    }
}

template <size_t Heads, size_t HeadDim, HeadRotation<HeadDim> Rope>
void rotate_heads(Matrix<Heads * HeadDim>& values, const Rope& rope, Position first_position) {
    values.transform_rows([&](size_t row, MutVecView<Heads * HeadDim> value) { rotate_heads<Heads, HeadDim>(value, rope, first_position + row); });
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
        const VecView<QueryKeyDim> query = slice<QueryKeyDim>(queries.row(query_row), head);

        std::vector<Scalar> scores(keys.rows());
        for (size_t key_row = 0; key_row < keys.rows(); ++key_row) scores[key_row] = scale * dot(query, slice<QueryKeyDim>(keys.row(key_row), head));
        Softmax::apply(scores);

        Vec<ValueDim> attended;
        for (size_t key_row = 0; key_row < keys.rows(); ++key_row) attended.scaled_add(slice<ValueDim>(values.row(key_row), head), scores[key_row]);
        return attended;
    });
}

// A causal attention mask, optionally restricted to a sliding window. Width
// zero is full attention. Position arithmetic stays inside this domain type.
class CausalWindow {
public:
    explicit constexpr CausalWindow(size_t width) : width_(width) {}
    static constexpr CausalWindow full() { return CausalWindow{0}; }

    constexpr bool contains(Position key, Position query) const {
        if (key > query) return false;
        return width_ == 0 || query - key < width_;
    }

private:
    size_t width_;
};

// ---- multi-head attention and its K/V state --------------------------------
//
// The cache owns both the typed row storage and the complete multi-head
// reduction. Callers provide packed Q heads and, for an owning layer, packed K/V
// heads. No scalar buffer, cache-row index, visibility check, or one-head helper
// escapes this type. Rows x query heads form one flat parallel task space.
template <size_t Hq, size_t Hkv, size_t HeadDim>
class MultiHeadAttention {
    static_assert(Hkv != 0);
    static_assert(Hq % Hkv == 0, "GQA groups must divide evenly");
    static constexpr size_t Width = Hkv * HeadDim;

    struct VisibleRows {
        size_t first;
        size_t end;
    };
    struct KVHead {
        size_t index;
    };

public:
    explicit MultiHeadAttention(size_t capacity = 0) : capacity_(capacity) {}

    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    Position position(size_t logical) const { return positions_[physical(logical)]; }

    // Append one complete packed K/V row. This remains public for cache fixtures;
    // model code normally appends a batch through append_and_attend().
    void append(Position position, VecView<Width> key, VecView<Width> value) {
        if (size_ != 0 && position <= this->position(size_ - 1)) throw std::invalid_argument("MultiHeadAttention: positions must increase");

        if (capacity_ == 0) {
            positions_.push_back(position);
            keys_.append(key);
            values_.append(value);
            ++size_;
            return;
        }

        ensure_bounded_storage();
        size_t destination;
        if (size_ < capacity_) {
            destination = (start_ + size_) % capacity_;
            ++size_;
        } else {
            destination = start_;
            start_ = (start_ + 1) % capacity_;
        }
        positions_[destination] = position;
        keys_.replace(destination, key);
        values_.replace(destination, value);
    }

    // Multi-head attention against K/V already in this cache. For every query
    // row and query head in parallel:
    //   score_j = scale * dot(q_head, k_j,group)
    //   alpha   = softmax(score)
    //   output  = sum_j alpha_j * v_j,group
    // GQA is explicit here: adjacent groups of Hq/Hkv query heads share one
    // cached K/V head.
    Matrix<Hq * HeadDim> attend(MatrixView<Hq * HeadDim> queries, Position first_query_position, CausalWindow window, Scalar score_scale = 1.f) const {
        std::vector<VisibleRows> visible;
        visible.reserve(queries.rows());
        for (size_t row = 0; row < queries.rows(); ++row) visible.push_back(visible_rows(first_query_position + row, window));

        Matrix<Hq * HeadDim> result = Matrix<Hq * HeadDim>::zero_rows(queries.rows());
        par_for(queries.rows() * Hq, [&](size_t task) {
            const size_t row = task / Hq;
            const size_t head = task % Hq;
            const VecView<HeadDim> query = slice<HeadDim>(queries.row(row), head);
            const KVHead group{head / (Hq / Hkv)};
            const VisibleRows rows = visible[row];

            std::vector<Scalar> alpha;
            alpha.reserve(rows.end - rows.first);
            for (size_t j = rows.first; j < rows.end; ++j) alpha.push_back(score_scale * dot(query, key(j, group)));
            Softmax::apply(alpha);

            Vec<HeadDim> output;
            for (size_t n = 0; n < alpha.size(); ++n) output.scaled_add(value(rows.first + n, group), alpha[n]);
            result.template replace_partition<HeadDim>(row, head, output);
        });
        return result;
    }

    // Append this batch's K/V and perform the complete multi-head reduction.
    // Prefill appends the batch once and attends in parallel. A bounded ring
    // that would evict during the batch advances row by row so early queries do
    // not lose keys they are still allowed to see.
    Matrix<Hq * HeadDim> append_and_attend(MatrixView<Hq * HeadDim> queries, MatrixView<Width> keys, MatrixView<Width> values, Position first_query_position, CausalWindow window, Scalar score_scale = 1.f) {
        if (queries.rows() != keys.rows() || keys.rows() != values.rows()) throw std::invalid_argument("MultiHeadAttention::append_and_attend: Q/K/V row mismatch");

        if (can_append_without_eviction(queries.rows())) {
            for (size_t row = 0; row < queries.rows(); ++row) append(first_query_position + row, keys.row(row), values.row(row));
            return attend(queries, first_query_position, window, score_scale);
        }

        Matrix<Hq * HeadDim> output;
        output.reserve(queries.rows());
        for (size_t row = 0; row < queries.rows(); ++row) {
            const Position position = first_query_position + row;
            append(position, keys.row(row), values.row(row));
            const MatrixView<Hq * HeadDim> one_query(queries.row(row).begin(), 1);
            output.append(attend(one_query, position, window, score_scale).row(0));
        }
        return output;
    }

private:
    bool can_append_without_eviction(size_t rows) const { return capacity_ == 0 || rows <= capacity_ - size_; }

    VisibleRows visible_rows(Position query, CausalWindow window) const {
        size_t first = size_;
        size_t end = size_;
        for (size_t row = 0; row < size_; ++row) {
            if (!window.contains(position(row), query)) continue;
            if (first == size_) first = row;
            end = row + 1;
        }
        if (first == size_) throw std::runtime_error("MultiHeadAttention::attend: query has no visible keys");
        return VisibleRows{first, end};
    }

    VecView<HeadDim> key(size_t logical, KVHead head) const {
        if (head.index >= Hkv) throw std::out_of_range("MultiHeadAttention: K/V head out of range");
        return slice<HeadDim>(keys_.row(physical(logical)), head.index);
    }
    VecView<HeadDim> value(size_t logical, KVHead head) const {
        if (head.index >= Hkv) throw std::out_of_range("MultiHeadAttention: K/V head out of range");
        return slice<HeadDim>(values_.row(physical(logical)), head.index);
    }

    void ensure_bounded_storage() {
        if (!positions_.empty()) return;
        positions_.assign(capacity_, Position{0});
        keys_.reserve(capacity_);
        values_.reserve(capacity_);
        const Vec<Width> empty;
        for (size_t row = 0; row < capacity_; ++row) {
            keys_.append(empty);
            values_.append(empty);
        }
    }

    size_t physical(size_t logical) const {
        if (logical >= size_) throw std::out_of_range("MultiHeadAttention: row out of range");
        return capacity_ == 0 ? logical : (start_ + logical) % capacity_;
    }

    size_t capacity_;
    size_t start_ = 0;
    size_t size_ = 0;
    std::vector<Position> positions_;
    Matrix<Width> keys_;
    Matrix<Width> values_;
};
