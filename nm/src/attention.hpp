// attention.hpp — the token-mixing core, model-neutral.
//
// Everything here is architecture-independent: a position-indexed K/V cache, the
// visibility predicate that defines causal and sliding-window masking, rotary
// position embedding, and per-head normalization. What this file deliberately
// does NOT own is the softmax reduction over cached keys. That is one short
// equation per family, written out in the layer whose weights feed it, because
// families differ inside it — capped scores, sink logits, a gate on the output —
// and a shared kernel would grow a parameter for each variation.
//
// The four attention LAWS hold throughout: q-dim == k-dim; v-dim is free;
// Wo in = Hq*Dv; Wo out = D. Hq*Dqk == D is a convention, not a law — Gemma 4
// breaks it, so nothing below assumes it.
#pragma once
#include "components.hpp"
#include <algorithm>
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
void rotate_heads(Matrix<Heads * HeadDim>& values, const Rope& rope, Position conversation_position) {
    values.transform_rows([&](size_t row, MutVecView<Heads * HeadDim> value) { rotate_heads<Heads, HeadDim>(value, rope, conversation_position + row); });
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

// The rows of a position-ordered cache that one query may attend to: a half-open
// interval, because causal-plus-window is a single interval in POSITION space and
// consecutive positions carry it into ROW space unchanged.
struct VisibleRows {
    size_t first;
    size_t end;

    bool empty() const { return first == end; }
    size_t count() const { return end - first; }
};

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

    // The same predicate in closed form, over `rows` cached keys whose positions
    // run consecutively from `oldest`. Causal ends the interval at the query; the
    // width decides how far back it begins. Testing `contains` row by row instead
    // would sweep the entire cache per query to produce two integers, and in
    // decode that is a full re-read of a structure nothing else touches.
    // [BANDWIDTH]
    constexpr VisibleRows visible_rows(Position query, Position oldest, size_t rows) const {
        if (query < oldest) return VisibleRows{0, 0};
        const size_t distance = query - oldest;  // the row the query itself would occupy
        const size_t end = std::min(rows, distance + 1);
        const size_t begin = width_ == 0 || distance < width_ ? 0 : distance + 1 - width_;
        return VisibleRows{std::min(begin, end), end};  // a window past the newest row sees nothing
    }

private:
    size_t width_;
};

// ---- the K/V cache ----------------------------------------------------------
//
// Which cached K/V head a query head reads. Under GQA several query heads share
// one, so a query-head index and a K/V-head index are different quantities, and
// substituting one for the other yields plausible-looking wrong attention rather
// than a crash.
struct KVHead {
    size_t index;
};

// One layer's K/V rows: storage, the eviction ring, and the position anchor.
// Nothing else. The reduction that reads these rows belongs to the layer whose
// weights produced them — what happens between the scores and the softmax is
// family anatomy, not cache business, and a shared kernel would have to grow a
// parameter for every family that does it differently.
//
// Read the template parameters: a cache is sized by the K/V heads it HOLDS. How
// many QUERY heads later read it is no concern of this type, which is precisely
// what grouped-query attention means.
template <size_t Hkv, size_t HeadDim>
class KVCache {
    static_assert(Hkv != 0);

public:
    static constexpr size_t Width = Hkv * HeadDim;

    explicit KVCache(size_t capacity = 0) : capacity_(capacity) {}

    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    // Row 0 holds the oldest key still cached, and row i the one i positions
    // after it. A ring's evictions move that anchor forward; nothing else does.
    Position oldest_position() const { return oldest_position_; }
    Position position(size_t logical) const {
        if (logical >= size_) throw std::out_of_range("KVCache: row out of range");
        return oldest_position_ + logical;
    }

    // Whether `rows` more rows fit before anything is overwritten. A prefill
    // batch that does NOT fit cannot be appended and then attended in one go:
    // the earliest queries would lose keys they are still allowed to see.
    bool can_append_without_eviction(size_t rows) const { return capacity_ == 0 || rows <= capacity_ - size_; }

    // Append one complete packed K/V row. Positions are CONSECUTIVE, not merely
    // increasing: a decoder is handed every position in order, so a row's
    // position is its distance from the anchor and there is nothing per-row to
    // store.
    void append(Position position, VecView<Width> key, VecView<Width> value) {
        if (size_ == 0) oldest_position_ = position;
        else if (position != oldest_position_ + size_) throw std::invalid_argument("KVCache: positions must be consecutive");

        if (capacity_ == 0) {
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
            oldest_position_ = oldest_position_ + 1;  // the row just overwritten was the oldest
        }
        keys_.replace(destination, key);
        values_.replace(destination, value);
    }

    // The rows a query at `query` may attend to, as a half-open interval.
    // Empty is never a legitimate answer — a softmax over no keys has no value —
    // so this throws rather than handing back a range a caller might average.
    VisibleRows visible_rows(Position query, CausalWindow window) const {
        const VisibleRows rows = window.visible_rows(query, oldest_position_, size_);
        if (rows.empty()) throw std::runtime_error("KVCache: query has no visible keys");
        return rows;
    }

    // One head of one cached row. `logical` counts up from the oldest retained
    // row, so a ring's wrap-around never leaves this type.
    VecView<HeadDim> key(size_t logical, KVHead head) const {
        if (head.index >= Hkv) throw std::out_of_range("KVCache: K/V head out of range");
        return slice<HeadDim>(keys_.row(physical(logical)), head.index);
    }
    VecView<HeadDim> value(size_t logical, KVHead head) const {
        if (head.index >= Hkv) throw std::out_of_range("KVCache: K/V head out of range");
        return slice<HeadDim>(values_.row(physical(logical)), head.index);
    }

private:
    void ensure_bounded_storage() {
        if (keys_.rows() == capacity_) return;
        keys_.reserve(capacity_);
        values_.reserve(capacity_);
        const Vec<Width> empty;
        for (size_t row = 0; row < capacity_; ++row) {
            keys_.append(empty);
            values_.append(empty);
        }
    }

    size_t physical(size_t logical) const {
        if (logical >= size_) throw std::out_of_range("KVCache: row out of range");
        return capacity_ == 0 ? logical : (start_ + logical) % capacity_;
    }

    size_t capacity_;
    size_t start_ = 0;
    size_t size_ = 0;
    Position oldest_position_{0};
    Matrix<Width> keys_;
    Matrix<Width> values_;
};
