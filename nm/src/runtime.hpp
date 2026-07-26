// runtime.hpp - model-neutral sampling. The immutable Transformer is the
// inference function; prefix memoization lives in PrefixCache.
#pragma once
#include "transformer.hpp"
#include <random>

template <size_t V>
TokenId sample_greedy(const Vec<V>& lg) {
    size_t best = 0;
    for (size_t v = 1; v < V; ++v)
        if (lg[v] > lg[best]) best = v;
    return TokenId{int32_t(best)};
}

struct SamplerCfg {
    float temp = 0.0f;         // 0 => greedy
    int top_k = 0;             // 0 => disabled
    float top_p = 1.0f;        // 1 => disabled
    float rep_penalty = 1.0f;  // 1 => disabled
    uint64_t seed = 0;
};

// Host-side sampler over logits. Greedy remains bit-for-bit the argmax path.
template <size_t V>
class Sampler {
    SamplerCfg cfg_;
    std::mt19937_64 rng_;

public:
    explicit Sampler(SamplerCfg c = {}) : cfg_(c), rng_(c.seed) {}
    const SamplerCfg& cfg() const { return cfg_; }

    TokenId operator()(Vec<V>& lg, std::span<const TokenId> history) {
        if (cfg_.rep_penalty != 1.0f)
            for (TokenId t : history) {
                float& z = lg[size_t(t)];
                z = z > 0 ? z / cfg_.rep_penalty : z * cfg_.rep_penalty;
            }
        if (cfg_.temp <= 0.0f) return sample_greedy(lg);

        // candidate set: sort by logit, apply top-k then top-p on softmax probs
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        size_t k = cfg_.top_k > 0 ? std::min<size_t>(cfg_.top_k, V) : V;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        idx.resize(k);
        const float mx = lg[idx[0]];
        std::vector<float> pr(k);
        float sum = 0;
        for (size_t i = 0; i < k; ++i) {
            pr[i] = std::exp((lg[idx[i]] - mx) / cfg_.temp);
            sum += pr[i];
        }
        float cum = 0;
        size_t keep = k;
        for (size_t i = 0; i < k; ++i) {
            cum += pr[i] / sum;
            if (cum >= cfg_.top_p) {
                keep = i + 1;
                break;
            }
        }
        float s2 = 0;
        for (size_t i = 0; i < keep; ++i) s2 += pr[i];
        std::uniform_real_distribution<float> U(0, s2);
        float r = U(rng_), acc = 0;
        for (size_t i = 0; i < keep; ++i) {
            acc += pr[i];
            if (r <= acc) return TokenId{idx[i]};
        }
        return TokenId{idx[keep - 1]};
    }
};
