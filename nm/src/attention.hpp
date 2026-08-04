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

    MutVecView<HeadDim> operator()(MutVecView<HeadDim> value, Position position) const {
        for (size_t i = 0; i < Planes; ++i) {
            const Scalar angle = position * inv_freq_[i];
            const Scalar c = std::cos(angle), s = std::sin(angle);
            const Scalar x = value[i], y = value[i + Planes];
            value[i] = x * c - y * s;
            value[i + Planes] = y * c + x * s;
        }
        return value;
    }

    template <size_t Total>
    Matrix<Total> operator()(const Matrix<Total>& values, Position conversation_position) const { return (*this)(values.view().copy(), conversation_position); }

    template <size_t Total>
    Matrix<Total> operator()(Matrix<Total>&& values, Position conversation_position) const {
        static_assert(Total % HeadDim == 0, "RotaryEmbedding: tensor is not a whole number of heads");
        values.transform_rows([&](size_t row, MutVecView<Total> value) {
            for (size_t head = 0; head < Total / HeadDim; ++head) (*this)(slice_mut<HeadDim>(value, head), conversation_position + row);
        });
        return std::move(values);
    }

    static constexpr size_t rotary_planes() { return RotaryPlanes; }

private:
    std::array<Scalar, Planes> inv_freq_{};
};

// A head rotation is a composable operation over an owning matrix. Any table
// with that operation works — full or partial RoPE, at either Gemma width.
template <class R, size_t HeadDim>
concept HeadRotation = requires(const R& rope, Matrix<HeadDim> heads, Position position) {
    { rope(std::move(heads), position) } -> std::same_as<Matrix<HeadDim>>;
};

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
        scores = Softmax{}(std::move(scores));

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

// Full and local attention have different storage invariants, so they have
// different cache types. A full cache is append-only. A local cache is a fixed
// window ring, except while a prefill batch is deliberately retained for a
// later layer that shares its K/V. Both expose rows in chronological order:
// row zero is always the oldest retained position. Ring slots never escape the
// local cache type.
//
// The template parameters describe the K/V heads the cache HOLDS. How many
// query heads read them is a separate GQA concern.
template <size_t Hkv, size_t HeadDim>
class FullAttentionCache {
    static_assert(Hkv != 0);
    static constexpr size_t Width = Hkv * HeadDim;

public:
    size_t size() const { return keys_.rows(); }
    Position oldest_position() const { return oldest_position_; }
    Position position(size_t chronological_row) const {
        require_row(chronological_row);
        return oldest_position_ + chronological_row;
    }

    bool can_append_without_eviction(size_t) const { return true; }

    void append(Position position, VecView<Width> key, VecView<Width> value) {
        if (size() == 0) oldest_position_ = position;
        else if (position != oldest_position_ + size()) throw std::invalid_argument("FullAttentionCache: positions must be consecutive");
        keys_.append(key);
        values_.append(value);
    }

    VisibleRows visible_rows(Position query) const {
        const VisibleRows rows = CausalWindow::full().visible_rows(query, oldest_position_, size());
        if (rows.empty()) throw std::runtime_error("FullAttentionCache: query has no visible keys");
        return rows;
    }

    VecView<HeadDim> key(size_t chronological_row, KVHead head) const {
        require_head(head);
        require_row(chronological_row);
        return slice<HeadDim>(keys_.row(chronological_row), head.index);
    }
    VecView<HeadDim> value(size_t chronological_row, KVHead head) const {
        require_head(head);
        require_row(chronological_row);
        return slice<HeadDim>(values_.row(chronological_row), head.index);
    }

private:
    void require_row(size_t row) const {
        if (row >= size()) throw std::out_of_range("FullAttentionCache: row out of range");
    }
    static void require_head(KVHead head) {
        if (head.index >= Hkv) throw std::out_of_range("FullAttentionCache: K/V head out of range");
    }

    Position oldest_position_{0};
    Matrix<Width> keys_;
    Matrix<Width> values_;
};

template <size_t Hkv, size_t HeadDim>
class LocalAttentionCache {
    static_assert(Hkv != 0);
    static constexpr size_t Width = Hkv * HeadDim;

public:
    explicit LocalAttentionCache(size_t window) : window_(window) {
        if (window == 0) throw std::invalid_argument("LocalAttentionCache: window must be positive");
    }

    size_t size() const { return size_; }
    size_t window() const { return window_; }
    Position oldest_position() const { return oldest_position_; }
    Position position(size_t chronological_row) const {
        require_row(chronological_row);
        return oldest_position_ + chronological_row;
    }

    // Ordinarily the ring evicts as soon as it exceeds `window()`. A K/V source
    // whose later layers process the same prefill batch must defer that eviction:
    // each query sees only its local window, but the batch contains queries at
    // many positions. This operation retains exactly the old window plus the
    // rows about to be appended; end_batch_retention() restores the ring.
    void begin_batch_retention(size_t appended_rows) {
        if (retaining_batch_) throw std::logic_error("LocalAttentionCache: batch retention already active");
        if (appended_rows <= 1) return;

        Matrix<Width> ordered_keys;
        Matrix<Width> ordered_values;
        ordered_keys.reserve(size_ + appended_rows);
        ordered_values.reserve(size_ + appended_rows);
        for (size_t row = 0; row < size_; ++row) {
            ordered_keys.append(key_row(row));
            ordered_values.append(value_row(row));
        }
        keys_ = std::move(ordered_keys);
        values_ = std::move(ordered_values);
        oldest_slot_ = 0;
        retaining_batch_ = true;
        batch_rows_remaining_ = appended_rows;
    }

    void end_batch_retention() {
        if (!retaining_batch_) return;
        if (batch_rows_remaining_ != 0) throw std::logic_error("LocalAttentionCache: retained batch was not fully appended");

        const size_t kept = std::min(size_, window_);
        const size_t dropped = size_ - kept;
        Matrix<Width> compact_keys;
        Matrix<Width> compact_values;
        compact_keys.reserve(window_);
        compact_values.reserve(window_);
        for (size_t row = dropped; row < size_; ++row) {
            compact_keys.append(key_row(row));
            compact_values.append(value_row(row));
        }
        keys_ = std::move(compact_keys);
        values_ = std::move(compact_values);
        oldest_position_ = oldest_position_ + dropped;
        oldest_slot_ = 0;
        size_ = kept;
        retaining_batch_ = false;
    }

    bool can_append_without_eviction(size_t rows) const {
        if (retaining_batch_) return rows <= batch_rows_remaining_;
        return rows <= window_ - size_;
    }

    void append(Position position, VecView<Width> key, VecView<Width> value) {
        if (size_ == 0) oldest_position_ = position;
        else if (position != oldest_position_ + size_) throw std::invalid_argument("LocalAttentionCache: positions must be consecutive");

        if (retaining_batch_) {
            if (batch_rows_remaining_ == 0) throw std::length_error("LocalAttentionCache: retained batch received too many rows");
            keys_.append(key);
            values_.append(value);
            ++size_;
            --batch_rows_remaining_;
            return;
        }

        ensure_ring_storage();
        size_t destination;
        if (size_ < window_) {
            destination = (oldest_slot_ + size_) % window_;
            ++size_;
        } else {
            destination = oldest_slot_;
            oldest_slot_ = (oldest_slot_ + 1) % window_;
            oldest_position_ = oldest_position_ + 1;
        }
        keys_.replace(destination, key);
        values_.replace(destination, value);
    }

    VisibleRows visible_rows(Position query) const {
        const VisibleRows rows = CausalWindow{window_}.visible_rows(query, oldest_position_, size_);
        if (rows.empty()) throw std::runtime_error("LocalAttentionCache: query has no visible keys");
        return rows;
    }

    VecView<HeadDim> key(size_t chronological_row, KVHead head) const {
        require_head(head);
        return slice<HeadDim>(key_row(chronological_row), head.index);
    }
    VecView<HeadDim> value(size_t chronological_row, KVHead head) const {
        require_head(head);
        return slice<HeadDim>(value_row(chronological_row), head.index);
    }

private:
    void ensure_ring_storage() {
        if (keys_.rows() == window_) return;
        keys_.reserve(window_);
        values_.reserve(window_);
        const Vec<Width> empty;
        while (keys_.rows() < window_) {
            keys_.append(empty);
            values_.append(empty);
        }
    }

    size_t storage_row_for(size_t chronological_row) const {
        require_row(chronological_row);
        return retaining_batch_ ? chronological_row : (oldest_slot_ + chronological_row) % window_;
    }
    VecView<Width> key_row(size_t chronological_row) const { return keys_.row(storage_row_for(chronological_row)); }
    VecView<Width> value_row(size_t chronological_row) const { return values_.row(storage_row_for(chronological_row)); }
    void require_row(size_t row) const {
        if (row >= size_) throw std::out_of_range("LocalAttentionCache: row out of range");
    }
    static void require_head(KVHead head) {
        if (head.index >= Hkv) throw std::out_of_range("LocalAttentionCache: K/V head out of range");
    }

    size_t window_;
    size_t oldest_slot_ = 0;
    size_t size_ = 0;
    Position oldest_position_{0};
    Matrix<Width> keys_;
    Matrix<Width> values_;
    bool retaining_batch_ = false;
    size_t batch_rows_remaining_ = 0;
};
