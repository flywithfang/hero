// runtime.hpp - model-neutral sampling and autoregressive runtime.
#pragma once
#include "transformer.hpp"
#include <chrono>
#include <random>

template <size_t V>
TokenId sample_greedy(const Vec<V>& lg) {
    size_t best = 0;
    for (size_t v = 1; v < V; ++v) if (lg[v] > lg[best]) best = v;
    return TokenId{int32_t(best)};
}

struct SamplerCfg {
    float  temp   = 0.0f;      // 0 => greedy
    int    top_k  = 0;         // 0 => disabled
    float  top_p  = 1.0f;      // 1 => disabled
    float  rep_penalty = 1.0f; // 1 => disabled
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
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b){ return lg[a] > lg[b]; });
        idx.resize(k);
        const float mx = lg[idx[0]];
        std::vector<float> pr(k);
        float sum = 0;
        for (size_t i = 0; i < k; ++i) { pr[i] = std::exp((lg[idx[i]] - mx) / cfg_.temp); sum += pr[i]; }
        float cum = 0; size_t keep = k;
        for (size_t i = 0; i < k; ++i) { cum += pr[i] / sum; if (cum >= cfg_.top_p) { keep = i + 1; break; } }
        float s2 = 0; for (size_t i = 0; i < keep; ++i) s2 += pr[i];
        std::uniform_real_distribution<float> U(0, s2);
        float r = U(rng_), acc = 0;
        for (size_t i = 0; i < keep; ++i) { acc += pr[i]; if (r <= acc) return TokenId{idx[i]}; }
        return TokenId{idx[keep - 1]};
    }
};

struct Stats {
    size_t prefill_tokens = 0, decode_tokens = 0;
    double prefill_s = 0, decode_s = 0;
    double prefill_tps() const { return prefill_tokens ? prefill_tokens / prefill_s : 0; }
    double decode_tps()  const { return decode_tokens  ? decode_tokens  / decode_s  : 0; }
};

template <class Architecture>
    requires TransformerArchitecture<Architecture>
class AutoregressiveRuntime {
    TransformerSession<Architecture> session_;
    Stats           stats_;
public:
    using Model = typename Architecture::Model;
    static constexpr size_t V = Architecture::V;

    explicit AutoregressiveRuntime(const Model& model) : session_(model) {}
    size_t context_used() const { return session_.context_used(); }
    size_t context_left() const { return session_.context_left(); }
    void   reset()               { session_.reset(); stats_ = {}; }
    const Stats& stats() const { return stats_; }

    Vec<V> prefill(std::span<const TokenId> prompt) {
        auto t0 = std::chrono::steady_clock::now();
        Vec<V> lg = session_.prefill(prompt);
        std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0;
        stats_.prefill_tokens += prompt.size(); stats_.prefill_s += dt.count();
        return lg;
    }
    Vec<V> step(TokenId id) {
        auto t0 = std::chrono::steady_clock::now();
        Vec<V> lg = session_.step(id);
        std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0;
        stats_.decode_tokens += 1; stats_.decode_s += dt.count();
        return lg;
    }

    Vec<V> forward(const EmbeddedSequence<Architecture::D>& input) {
        auto t0 = std::chrono::steady_clock::now();
        Vec<V> logits = session_.forward(input);
        std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0;
        stats_.prefill_tokens += input.tokens();
        stats_.prefill_s += dt.count();
        return logits;
    }

    // Greedy generate (the parity/test mode). eos_set: any of these stops.
    std::vector<TokenId> generate(std::span<const TokenId> prompt, size_t max_new,
                                  std::span<const TokenId> eos_set) {
        std::vector<TokenId> out;
        auto is_eos = [&](TokenId id){ for (TokenId e : eos_set) if (e == id) return true; return false; };
        Vec<V> logits = prefill(prompt);
        for (size_t i = 0; i < max_new; ++i) {
            TokenId id = sample_greedy(logits);
            if (is_eos(id)) break;
            out.push_back(id);
            if (context_left() == 0) break;
            logits = step(id);
        }
        return out;
    }
};
