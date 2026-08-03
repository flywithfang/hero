// logits.hpp — the vocabulary axis and the operations that live on it.
//
// A decoder's output is not an anonymous Vec<V>. It is one score per vocabulary
// entry, and every operation meaningful on it — argmax, repetition penalty,
// temperature, top-k/top-p, log-softmax — is meaningful ONLY on that axis.
// Giving the axis a type collects that math in one place, the way
// MultiHeadAttention owns its cache layout and reduction. This is a COHESION argument, not a safety one: V is unique
// among the engine's widths, so the dimension already stops a Vec<D> from
// arriving here.
//
// The boundary with runtime.hpp's Sampler: Logits owns the MATH, Sampler owns
// the POLICY and the STATE (config, RNG, conversation history). Sampler decides
// whether and with what; Logits performs. Sampling stays outside the model
// either way — a Logits is a value the caller has been handed, not a stage in
// anyone's pipeline.
#pragma once
#include "core.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

// No repetition penalty. It is a 2019-era patch for degeneration in base
// models; instruction-tuned checkpoints rarely loop, top-p keeps the tail alive
// when they might, and llama.cpp itself now ships it disabled. Penalizing every
// token the conversation has ever seen also punishes the repetition real
// language depends on.
struct SamplerCfg {
    float temp = 0.0f;   // 0 => greedy
    int top_k = 0;       // 0 => disabled
    float top_p = 1.0f;  // 1 => disabled
    uint64_t seed = 0;
};

// One entry of a top-k report: the raw score and its normalised log-probability
// (log-sum-exp over the WHOLE vocabulary, not over the k that were kept).
struct ScoredToken {
    TokenId id;
    Scalar logit;
    double logprob;
};

template <size_t V>
class Logits {
public:
    static constexpr size_t VOCAB = V;

    Logits() = default;  // zeroed, like the Vec it wraps
    explicit Logits(Vec<V> values) : values_(std::move(values)) {}

    static constexpr size_t size() { return V; }
    Scalar operator[](size_t v) const { return values_[v]; }
    Scalar& operator[](size_t v) { return values_[v]; }
    VecView<V> view() const { return VecView<V>(values_); }
    MutVecView<V> mutable_view() { return MutVecView<V>{values_.begin()}; }
    Logits copy() const { return Logits(::copy(view())); }

    // The greedy token. Ties resolve to the lowest id, which is what makes
    // greedy decoding bit-for-bit reproducible against the oracle.
    TokenId argmax() const {
        size_t best = 0;
        for (size_t v = 1; v < V; ++v)
            if (values_[v] > values_[best]) best = v;
        return TokenId{int32_t(best)};
    }

    // One fused draw: sort down to k candidates ONCE, then temperature, top-p
    // and the draw all run over those k rather than over V. [BANDWIDTH] at
    // V=262144 the partial_sort is the only full-width pass.
    TokenId sample(const SamplerCfg& cfg, std::mt19937_64& rng) const {
        if (cfg.temp <= 0.0f) return argmax();

        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        const size_t k = cfg.top_k > 0 ? std::min<size_t>(size_t(cfg.top_k), V) : V;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return values_[a] > values_[b]; });
        idx.resize(k);

        const float mx = values_[idx[0]];
        std::vector<float> pr(k);
        float sum = 0;
        for (size_t i = 0; i < k; ++i) {
            pr[i] = std::exp((values_[idx[i]] - mx) / cfg.temp);
            sum += pr[i];
        }
        float cum = 0;
        size_t keep = k;
        for (size_t i = 0; i < k; ++i) {
            cum += pr[i] / sum;
            if (cum >= cfg.top_p) {
                keep = i + 1;
                break;
            }
        }
        float s2 = 0;
        for (size_t i = 0; i < keep; ++i) s2 += pr[i];
        std::uniform_real_distribution<float> U(0, s2);
        float r = U(rng), acc = 0;
        for (size_t i = 0; i < keep; ++i) {
            acc += pr[i];
            if (r <= acc) return TokenId{idx[i]};
        }
        return TokenId{idx[keep - 1]};
    }

    // The k highest-scoring entries with normalised logprobs — what a parity
    // dump compares against the oracle. [COMPUTE] one full pass for log Z.
    std::vector<ScoredToken> top(size_t k) const {
        k = std::min(k, V);
        if (k == 0) return {};
        std::vector<size_t> idx(V);
        std::iota(idx.begin(), idx.end(), size_t{0});
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](size_t a, size_t b) { return values_[a] > values_[b]; });

        const Scalar max_logit = values_[idx[0]];
        double exp_sum = 0;
        for (size_t v = 0; v < V; ++v) exp_sum += std::exp(double(values_[v] - max_logit));
        const double log_z = double(max_logit) + std::log(exp_sum);

        std::vector<ScoredToken> out;
        out.reserve(k);
        for (size_t i = 0; i < k; ++i) out.push_back(ScoredToken{TokenId{int32_t(idx[i])}, values_[idx[i]], double(values_[idx[i]]) - log_z});
        return out;
    }

private:
    Vec<V> values_;
};
