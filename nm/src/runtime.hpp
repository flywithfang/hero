// runtime.hpp - model-neutral sampling. The immutable Transformer is the
// inference function; prefix memoization lives in PrefixCache.
//
// Sampler is the POLICY and the STATE: it owns the configuration, the RNG, and
// the decision of what to apply. The math over the vocabulary axis lives on
// Logits (logits.hpp), so this file has no loops over V.
#pragma once
#include "transformer.hpp"
#include <random>

// Host-side sampler over logits. Greedy remains bit-for-bit the argmax path.
template <size_t V>
class Sampler {
    SamplerCfg cfg_;
    std::mt19937_64 rng_;

public:
    explicit Sampler(SamplerCfg c = {}) : cfg_(c), rng_(c.seed) {}
    const SamplerCfg& cfg() const { return cfg_; }

    TokenId operator()(const Logits<V>& lg) { return lg.sample(cfg_, rng_); }
};
