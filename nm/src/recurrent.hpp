// recurrent.hpp — the OTHER token mixer, model-neutral.
//
// Attention mixes tokens by looking back over a cache that grows with T.
// A gated delta network mixes tokens by carrying a fixed-size state matrix
// forward, one token at a time. Both are "communication across tokens"; they
// differ in what they remember and what that memory costs:
//
//   attention      memory = L * Hkv * (Dqk + Dv) * T floats      [GROWS-T]
//   delta network  memory = Hv * Dk * Dv floats                  constant
//
// This file owns the math and the state; a family supplies dimensions and
// weights. Nothing here knows about Qwen.
//
// The recurrence (gated delta rule), for one value head, one token, with
// state S in R^{Dk x Dv}:
//
//     S <- S * exp(g)              gated decay, g <= 0
//     d  = (v - S^T k) * beta      how wrong the current memory is about v
//     S <- S + k (x) d             rank-1 correction toward v
//     o  = S^T q                   read
//
// With g = 0 and beta = 1 this is exactly the classic delta rule; the gate is
// what lets the state forget. [COMPUTE] per token is O(Dk*Dv) per head and
// does NOT grow with T.
#pragma once
#include "components.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

// ---- small pieces the recurrence needs --------------------------------------

// softplus(x) = log(1 + e^x), computed through the stable branch so that a
// large positive x does not overflow before the log takes it back down.
inline Scalar softplus(Scalar x) {
    return x > Scalar(20) ? x : std::log1p(std::exp(x));
}

// x / max(||x||_2, eps). Note this is L2 normalization, NOT RMS: there is no
// 1/sqrt(N) and no learned scale. Delta networks normalize q and k this way so
// that the rank-1 update has a bounded effect on the state.
template <size_t N>
Vec<N> l2_normalize(VecView<N> x, Scalar eps = 1e-6f) {
    Scalar sum_of_squares = 0;
    for (size_t i = 0; i < N; ++i) sum_of_squares += x[i] * x[i];
    const Scalar scale = 1.f / std::max(std::sqrt(sum_of_squares), eps);
    Vec<N> normalized;
    for (size_t i = 0; i < N; ++i) normalized[i] = x[i] * scale;
    return normalized;
}

// ---- the short causal convolution -------------------------------------------
//
// A depthwise causal conv1d of width W over Channels independent channels:
//
//     y[c, t] = sum_{i=0..W-1} weight[c, i] * x[c, t - (W-1) + i]
//
// Positions before the start of the sequence read zero. "Depthwise" is what
// makes this cheap and what makes its carried state small: the only history a
// channel needs is its own last W-1 values, so the state is Channels*(W-1)
// floats regardless of T. That is the same bargain the delta state makes.
//
// Weights come as Matrix<Width> with one row per channel — the checkpoint
// stores the taps contiguously per channel, so this is the checkpoint's own
// layout with no transpose.
template <size_t Channels, size_t Width>
class CausalConv1dState {
    static_assert(Width >= 1, "a causal conv needs at least one tap");
    static constexpr size_t History = Width - 1;

public:
    CausalConv1dState() : history_(History * Channels, 0.f) {}

    Vec<Channels> step(VecView<Channels> input, const Matrix<Width>& weights) {
        if (weights.rows() != Channels)
            throw std::invalid_argument("CausalConv1dState: wrong channel count");

        Vec<Channels> output;
        for (size_t c = 0; c < Channels; ++c) {
            const VecView<Width> taps = weights.row(c);
            Scalar sum = taps[History] * input[c];   // the current sample
            for (size_t tap = 0; tap < History; ++tap)
                sum += taps[tap] * past(tap)[c];     // tap 0 is the oldest
            output[c] = sum;
        }
        advance(input);
        return output;
    }

    void clear() {
        std::fill(history_.begin(), history_.end(), 0.f);
        start_ = 0;
    }

private:
    // Logical tap t (0 = oldest) maps into a ring of the last History samples.
    // Slots that no token has written yet are zero, which is the causal pad.
    const Scalar* past(size_t tap) const {
        return history_.data() + ((start_ + tap) % History) * Channels;
    }

    void advance(VecView<Channels> input) {
        if constexpr (History != 0) {
            // The newest sample overwrites the oldest slot, which start_ names.
            std::copy(input.begin(), input.end(),
                      history_.begin() + start_ * Channels);
            start_ = (start_ + 1) % History;
        }
    }

    std::vector<Scalar> history_;
    size_t start_ = 0;
};

// ---- the delta-network state ------------------------------------------------
//
// One [Dk x Dv] matrix per value head. Row i is indexed by the key dimension
// and column j by the value dimension, so S^T k and S^T q are both reductions
// over the key axis and k (x) d is an outer product into the same layout.
template <size_t Heads, size_t Dk, size_t Dv>
class DeltaNetState {
public:
    static constexpr size_t floats_per_head = Dk * Dv;
    static constexpr size_t floats = Heads * floats_per_head;

    DeltaNetState() : values_(floats, 0.f) {}

    void clear() { std::fill(values_.begin(), values_.end(), 0.f); }

    Scalar* head(size_t h) {
        if (h >= Heads) throw std::out_of_range("DeltaNetState: head out of range");
        return values_.data() + h * floats_per_head;
    }
    const Scalar* head(size_t h) const {
        if (h >= Heads) throw std::out_of_range("DeltaNetState: head out of range");
        return values_.data() + h * floats_per_head;
    }

private:
    std::vector<Scalar> values_;
};

// One head, one token. `state` is the [Dk x Dv] block for this head and is
// updated in place: the mixer is inherently sequential in T, which is exactly
// why its cost is constant in T.
//
// q is expected pre-scaled by 1/sqrt(Dk) and q/k pre-L2-normalized; keeping
// those out of here means the caller can hoist them across the head group.
template <size_t Dk, size_t Dv>
Vec<Dv> gated_delta_step(Scalar* state, VecView<Dk> q, VecView<Dk> k,
                         VecView<Dv> v, Scalar gate, Scalar beta) {
    const Scalar decay = std::exp(gate);

    // Decay, then read what the state currently predicts for this key.
    Vec<Dv> predicted;
    for (size_t i = 0; i < Dk; ++i) {
        Scalar* row = state + i * Dv;
        const Scalar ki = k[i];
        for (size_t j = 0; j < Dv; ++j) {
            row[j] *= decay;
            predicted[j] += row[j] * ki;
        }
    }

    // The correction is how far that prediction is from the actual value,
    // scaled by the per-head write strength beta.
    Vec<Dv> delta;
    for (size_t j = 0; j < Dv; ++j)
        delta[j] = (v[j] - predicted[j]) * beta;

    // Rank-1 write, then read with the query.
    Vec<Dv> output;
    for (size_t i = 0; i < Dk; ++i) {
        Scalar* row = state + i * Dv;
        const Scalar ki = k[i], qi = q[i];
        for (size_t j = 0; j < Dv; ++j) {
            row[j] += ki * delta[j];
            output[j] += row[j] * qi;
        }
    }
    return output;
}
