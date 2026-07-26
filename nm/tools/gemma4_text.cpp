// gemma4_text — the logit-parity harness. Prints the prompt token ids, the
// top-5 next-token logprobs, and an optional greedy continuation. Rerun this
// after any numerics-touching change.
//
//   gemma4_text model.gguf "prompt" [generate-count]
//
// The size (E4B or 12B) is detected from the checkpoint's block_count.
#include "../src/gemma4_loader.hpp"
#include "../src/tokenizer.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>

template <class C, class Architecture>
static int run(Transformer<Architecture> transformer, const Tokenizer& tokenizer, const std::vector<int32_t>& ids, size_t generate_count) {
    std::vector<TokenId> tokens;
    tokens.reserve(ids.size());
    for (int32_t id : ids) tokens.push_back(TokenId{id});

    PrefixCache<Architecture> memo;
    const auto start = std::chrono::steady_clock::now();
    auto logits = transformer(tokens, memo);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    size_t best = 0;
    for (size_t i = 1; i < C::V; ++i)
        if (logits[i] > logits[best]) best = i;
    std::printf("prompt_tokens:");
    for (int32_t id : ids) std::printf(" %d", id);
    std::printf("\nnext_token: %zu\npiece: %s\nprefill: %.3fs (%zu tokens, %.1f tok/s)\n", best, tokenizer.decode1(int32_t(best)).c_str(), elapsed, ids.size(), double(ids.size()) / elapsed);

    std::vector<size_t> top(C::V);
    std::iota(top.begin(), top.end(), 0);
    std::partial_sort(top.begin(), top.begin() + 5, top.end(), [&](size_t a, size_t b) { return logits[a] > logits[b]; });
    const Scalar max_logit = logits[top[0]];
    double exp_sum = 0;
    for (size_t i = 0; i < C::V; ++i) exp_sum += std::exp(double(logits[i] - max_logit));
    const double log_z = double(max_logit) + std::log(exp_sum);
    for (size_t i = 0; i < 5; ++i) std::printf("top%zu: id=%zu logprob=%.9f piece=%s\n", i + 1, top[i], double(logits[top[i]]) - log_z, tokenizer.decode1(int32_t(top[i])).c_str());

    if (generate_count > 1) {
        const auto decode_start = std::chrono::steady_clock::now();
        size_t produced = 0;
        std::printf("generated:");
        std::string text;
        for (size_t i = 0; i < generate_count; ++i) {
            size_t token = 0;
            for (size_t v = 1; v < C::V; ++v)
                if (logits[v] > logits[token]) token = v;
            std::printf(" %zu", token);
            ++produced;
            text += tokenizer.decode1(int32_t(token));
            if (tokenizer.is_eog(int32_t(token)) || i + 1 == generate_count) break;
            tokens.push_back(TokenId{int32_t(token)});
            logits = transformer(tokens, memo);
        }
        const double decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
        std::printf("\ntext: %s\ndecode: %.3fs (%zu tokens, %.2f tok/s)\n", text.c_str(), decode_seconds, produced, double(produced) / decode_seconds);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "usage: %s gemma4-text.gguf prompt [generate-count]\n", argv[0]);
        return 2;
    }
    try {
        GGUF gguf(argv[1]);
        Tokenizer tokenizer(gguf);
        const auto ids = tokenizer.encode(argv[2], /*add_bos=*/true, /*parse_special=*/true);
        const size_t generate_count = argc == 4 ? std::strtoull(argv[3], nullptr, 10) : 1;

        const size_t blocks = size_t(gguf.get_int("gemma4.block_count", 0));
        if (blocks == Gemma4_12BTextConfig::L) {
            std::fprintf(stderr, "Gemma 4 12B (%zu layers)\n", blocks);
            return run<Gemma4_12BTextConfig, Gemma4_12BArchitecture>(gemma4_loader::load_12b_text(gguf), tokenizer, ids, generate_count);
        }
        if (blocks == Gemma4E4BTextConfig::L) {
            std::fprintf(stderr, "Gemma 4 E4B (%zu layers)\n", blocks);
            return run<Gemma4E4BTextConfig, Gemma4E4BArchitecture>(gemma4_loader::load_e4b_text(gguf), tokenizer, ids, generate_count);
        }
        std::fprintf(stderr, "no compiled Gemma 4 config has %zu blocks\n", blocks);
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
