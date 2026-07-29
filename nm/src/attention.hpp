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

// ---- per-head normalization -------------------------------------------------
//
// Q/K normalization is applied inside each head, over HeadDim, with one scale
// vector shared by every head. Modern families (Gemma 4, Qwen 3.x) all do this;
// it is what lets a model fold the 1/sqrt(HeadDim) score scale into a learned
// weight instead of a constant.
template <size_t Heads, size_t HeadDim, class Norm>
class PerHeadNorm {
public:
    explicit PerHeadNorm(Norm norm) : norm_(std::move(norm)) {}

    Vec<Heads * HeadDim> operator()(VecView<Heads * HeadDim> x) const {
        return transform_heads<Heads, HeadDim>(x, [&](VecView<HeadDim> head) { return norm_(head); });
    }
    Matrix<Heads * HeadDim> operator()(MatrixView<Heads * HeadDim> x) const {
        return transform_heads<Heads, HeadDim>(x, [&](VecView<HeadDim> head) { return norm_(head); });
    }

private:
    Norm norm_;
};

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
    HeadPair<Heads, HeadDim> split{Matrix<Heads * HeadDim>(packed.rows()), Matrix<Heads * HeadDim>(packed.rows())};
    for (size_t row = 0; row < packed.rows(); ++row) {
        const Scalar* source = packed.row(row).begin();
        Scalar* first = split.first.data() + row * Heads * HeadDim;
        Scalar* second = split.second.data() + row * Heads * HeadDim;
        for (size_t head = 0; head < Heads; ++head) {
            const Scalar* pair = source + head * 2 * HeadDim;
            std::copy(pair, pair + HeadDim, first + head * HeadDim);
            std::copy(pair + HeadDim, pair + 2 * HeadDim, second + head * HeadDim);
        }
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

    void apply(MutVecView<HeadDim> value, size_t position) const {
        for (size_t i = 0; i < Planes; ++i) {
            const Scalar angle = Scalar(position) * inv_freq_[i];
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

template <size_t Heads, size_t HeadDim, class Rope>
void rotate_heads(Vec<Heads * HeadDim>& value, const Rope& rope, size_t position) {
    for (size_t h = 0; h < Heads; ++h) {
        rope.apply(slice_mut<HeadDim>(value, h * HeadDim), position);
    }
}

template <size_t Heads, size_t HeadDim, class Rope>
void rotate_heads(Matrix<Heads * HeadDim>& values, const Rope& rope, size_t first_position) {
    for (size_t row = 0; row < values.rows(); ++row) {
        for (size_t head = 0; head < Heads; ++head) {
            rope.apply(MutVecView<HeadDim>{values.data() + row * Heads * HeadDim + head * HeadDim}, first_position + row);
        }
    }
}

// ---- the K/V cache ----------------------------------------------------------
//
// Position-aware K/V rows with either full retention (capacity == 0) or a ring
// of fixed width. Logical row zero is always the oldest retained position, so a
// sliding layer and a global layer are read through the same interface and only
// the storage cost differs. [BANDWIDTH][GROWS-T]
template <size_t Hkv, size_t HeadDim>
class KVCache {
    static constexpr size_t Width = Hkv * HeadDim;

public:
    explicit KVCache(size_t capacity = 0) : capacity_(capacity) {}

    size_t size() const { return size_; }
    bool can_append_without_eviction(size_t rows) const { return capacity_ == 0 || rows <= capacity_ - size_; }
    size_t capacity() const { return capacity_; }

    void append(Position position, VecView<Width> key, VecView<Width> value) {
        if (size_ != 0 && position.i <= this->position(size_ - 1).i) throw std::invalid_argument("KVCache: positions must increase");

        if (capacity_ == 0) {
            positions_.push_back(position.i);
            keys_.insert(keys_.end(), key.begin(), key.end());
            values_.insert(values_.end(), value.begin(), value.end());
            ++size_;
            return;
        }

        ensure_bounded_storage();

        size_t physical;
        if (size_ < capacity_) {
            physical = (start_ + size_) % capacity_;
            ++size_;
        } else {
            physical = start_;
            start_ = (start_ + 1) % capacity_;
        }
        positions_[physical] = position.i;
        std::copy(key.begin(), key.end(), keys_.begin() + physical * Width);
        std::copy(value.begin(), value.end(), values_.begin() + physical * Width);
    }

    Position position(size_t logical) const { return Position{positions_[physical(logical)]}; }
    VecView<HeadDim> key(size_t logical, KvHead head) const {
        if (head.i >= Hkv) throw std::out_of_range("KVCache: KV head out of range");
        return VecView<HeadDim>(keys_.data() + physical(logical) * Width + head.i * HeadDim);
    }
    VecView<HeadDim> value(size_t logical, KvHead head) const {
        if (head.i >= Hkv) throw std::out_of_range("KVCache: KV head out of range");
        return VecView<HeadDim>(values_.data() + physical(logical) * Width + head.i * HeadDim);
    }

private:
    void ensure_bounded_storage() {
        if (!positions_.empty()) return;
        positions_.resize(capacity_);
        keys_.resize(capacity_ * Width);
        values_.resize(capacity_ * Width);
    }

    size_t physical(size_t logical) const {
        if (logical >= size_) throw std::out_of_range("KVCache: row out of range");
        return capacity_ == 0 ? logical : (start_ + logical) % capacity_;
    }

    size_t capacity_;
    size_t start_ = 0;
    size_t size_ = 0;
    std::vector<size_t> positions_;
    std::vector<Scalar> keys_;
    std::vector<Scalar> values_;
};

// ---- the attention mask, defined exactly once -------------------------------
//
// A cached key at position k is visible to a query at position q when it is
// causally in the past and, on a sliding layer of width W, still inside the
// window:
//
//     visible(q, k)  <=>  k <= q  and  (W == 0  or  q - k < W)
//
// W == 0 means "no window", i.e. a full/global layer. Writing `k + W > q`
// rather than `q - k < W` keeps the unsigned arithmetic safe for k > q.
inline bool key_is_visible(Position key, Position query, size_t sliding_window) {
    if (key.i > query.i) return false;
    return sliding_window == 0 || key.i + sliding_window > query.i;
}

// softmax over an empty set is undefined, so a query that can see nothing is a
// bug in the caller (a ring that evicted too early, or a stale cache), never a
// silently-zero output.
template <size_t Hkv, size_t HeadDim>
void require_visible_key(const KVCache<Hkv, HeadDim>& cache, Position query_position, size_t sliding_window) {
    for (size_t j = 0; j < cache.size(); ++j)
        if (key_is_visible(cache.position(j), query_position, sliding_window)) return;
    throw std::runtime_error("attend: query has no visible keys");
}

// ---- one head of attention --------------------------------------------------
//
// For query head h at position q, over the cache rows j:
//
//     J     = { j : visible(q, position(j)) }
//     s_j   = Scale * (q_h . k_j)   for j in J
//     alpha = softmax(s)             over J only
//     out   = sum_{j in J} alpha_j * v_j
//
// Scale is a template parameter because it is model anatomy: conventional
// attention passes 1/sqrt(HeadDim), while a family that normalizes Q per head
// (Gemma 4) has already folded that constant into a learned weight and passes 1.
//
// GQA: the Hq query heads are partitioned into Hkv groups of Hq/Hkv, and every
// head of a group reads the same cached k/v rows. Only visible rows are
// gathered, so the score matrix is never materialized and masked-out entries
// cost nothing. [COMPUTE][GROWS-T]
template <size_t Hq, size_t Hkv, size_t HeadDim>
Vec<HeadDim> attend_head(VecView<Hq * HeadDim> query, const KVCache<Hkv, HeadDim>& cache, Position query_position, size_t query_head, size_t sliding_window, Scalar score_scale) {
    static_assert(Hq % Hkv == 0, "GQA groups must divide evenly");
    const KvHead group{query_head / (Hq / Hkv)};
    const VecView<HeadDim> q = slice<HeadDim>(query, query_head * HeadDim);

    std::vector<size_t> J;      // the visible cache rows
    std::vector<Scalar> alpha;  // scores s_j, then softmax weights
    J.reserve(cache.size());
    alpha.reserve(cache.size());
    for (size_t j = 0; j < cache.size(); ++j) {
        if (!key_is_visible(cache.position(j), query_position, sliding_window)) continue;
        J.push_back(j);
        alpha.push_back(score_scale * dot(q, cache.key(j, group)));
    }
    softmax(alpha);

    Vec<HeadDim> attended;  // zero-initialized; this is the sum
    ///score*V
    for (size_t n = 0; n < J.size(); ++n) attended.scaled_add(cache.value(J[n], group), alpha[n]);
    return attended;
}

// ---- attention over a whole cache, for one query and for a batch ------------
//
// Both forms are the same reduction; they differ only in how the independent
// (row, head) tasks are laid out for the parallel seam. Neither writes to the
// cache: Q is already normalized and rotated, and any K/V for these queries
// must already be present.

template <size_t Hq, size_t Hkv, size_t HeadDim>
Vec<Hq * HeadDim> attend(VecView<Hq * HeadDim> query, const KVCache<Hkv, HeadDim>& cache, Position query_position, size_t sliding_window = 0, Scalar score_scale = 1.f) {
    require_visible_key(cache, query_position, sliding_window);
    return par_map<Hq, HeadDim>([&](size_t head) { return attend_head<Hq>(query, cache, query_position, head, sliding_window, score_scale); });
}

template <size_t Hq, size_t Hkv, size_t HeadDim>
Matrix<Hq * HeadDim> attend(MatrixView<Hq * HeadDim> queries, const KVCache<Hkv, HeadDim>& cache, size_t first_query_position, size_t sliding_window = 0, Scalar score_scale = 1.f) {
    for (size_t row = 0; row < queries.rows(); ++row) require_visible_key(cache, Position{first_query_position + row}, sliding_window);

    // Rows x heads are mutually independent, so the batch is one flat task
    // space rather than a map over heads alone. par_for, not par_map, because
    // each task writes its own slice of a shared output rather than returning
    // a partition of a single vector.
    Matrix<Hq * HeadDim> output(queries.rows());
    par_for(queries.rows() * Hq, [&](size_t task) {
        const size_t row = task / Hq;
        const size_t head = task % Hq;
        Vec<HeadDim> attended = attend_head<Hq>(queries.row(row), cache, Position{first_query_position + row}, head, sliding_window, score_scale);
        std::copy(attended.begin(), attended.end(), output.data() + (row * Hq + head) * HeadDim);
    });
    return output;
}

// Append this batch's K/V, then attend. Splitting append from attend is what
// makes prefill cheap: every query in the batch sees the whole batch's keys
// (masked back to causal), so the cache is written once and read once.
template <size_t Hq, size_t Hkv, size_t HeadDim>
Matrix<Hq * HeadDim> attend_and_cache(MatrixView<Hq * HeadDim> queries, MatrixView<Hkv * HeadDim> keys, MatrixView<Hkv * HeadDim> values, KVCache<Hkv, HeadDim>& cache, size_t first_query_position, size_t sliding_window = 0, Scalar score_scale = 1.f) {
    if (queries.rows() != keys.rows() || keys.rows() != values.rows()) throw std::invalid_argument("attend_and_cache: Q/K/V row mismatch");

    if (cache.can_append_without_eviction(queries.rows())) {
        for (size_t row = 0; row < queries.rows(); ++row) {
            cache.append(Position{first_query_position + row}, keys.row(row), values.row(row));
        }
        return attend<Hq, Hkv, HeadDim>(queries, cache, first_query_position, sliding_window, score_scale);
    }

    // Eviction case only: a bounded sliding ring smaller than this batch would
    // drop a key that an early query still needs, so the batch shortcut would
    // change the result rather than just its cost. Fall back to advancing the
    // ring one position at a time, which is the definition of the semantics.
    Matrix<Hq * HeadDim> output(queries.rows());
    for (size_t row = 0; row < queries.rows(); ++row) {
        const Position position{first_query_position + row};
        cache.append(position, keys.row(row), values.row(row));
        output.set_row(row, attend<Hq, Hkv, HeadDim>(queries.row(row), cache, position, sliding_window, score_scale));
    }
    return output;
}
