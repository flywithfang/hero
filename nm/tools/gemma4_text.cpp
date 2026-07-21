#include "../src/gemma4_loader.hpp"
#include "../src/runtime.hpp"
#include "../src/tokenizer.hpp"
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "usage: %s gemma4-e4b-text.gguf prompt [generate-count]\n", argv[0]);
        return 2;
    }
    try {
        GGUF gguf(argv[1]);
        Tokenizer tokenizer(gguf);
        auto ids = tokenizer.encode(argv[2], /*add_bos=*/true, /*parse_special=*/true);
        std::vector<TokenId> tokens;
        tokens.reserve(ids.size());
        for (int32_t id : ids) tokens.push_back(TokenId{id});

        auto model = gemma4_loader::load_e4b_text(gguf);
        AutoregressiveRuntime<Gemma4E4BArchitecture> runtime(model);
        const auto start = std::chrono::steady_clock::now();
        auto logits = runtime.prefill(tokens);
        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();

        size_t best = 0;
        for (size_t i = 1; i < Gemma4E4BTextConfig::V; ++i)
            if (logits[i] > logits[best]) best = i;
        std::printf("prompt_tokens:");
        for (int32_t id : ids) std::printf(" %d", id);
        std::printf("\nnext_token: %zu\npiece: %s\nelapsed: %.3fs\n",
                    best, tokenizer.decode1(int32_t(best)).c_str(), elapsed);
        std::vector<size_t> top(Gemma4E4BTextConfig::V);
        std::iota(top.begin(), top.end(), 0);
        std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                          [&](size_t a, size_t b) { return logits[a] > logits[b]; });
        const Scalar max_logit = logits[top[0]];
        double exp_sum = 0;
        for (size_t i = 0; i < Gemma4E4BTextConfig::V; ++i)
            exp_sum += std::exp(double(logits[i] - max_logit));
        const double log_z = double(max_logit) + std::log(exp_sum);
        for (size_t i = 0; i < 5; ++i)
            std::printf("top%zu: id=%zu logprob=%.9f piece=%s\n", i + 1, top[i],
                        double(logits[top[i]]) - log_z,
                        tokenizer.decode1(int32_t(top[i])).c_str());

        const size_t generate_count = argc == 4 ? std::strtoull(argv[3], nullptr, 10) : 1;
        if (generate_count > 1) {
            std::printf("generated:");
            for (size_t i = 0; i < generate_count; ++i) {
                size_t token = 0;
                for (size_t v = 1; v < Gemma4E4BTextConfig::V; ++v)
                    if (logits[v] > logits[token]) token = v;
                std::printf(" %zu", token);
                if (tokenizer.is_eog(int32_t(token)) || i + 1 == generate_count) break;
                logits = runtime.step(TokenId{int32_t(token)});
            }
            std::printf("\n");
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
