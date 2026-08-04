// gemma4_text — the logit-parity harness. Prints the prompt token ids, the
// top-5 next-token logprobs, and an optional greedy continuation. Rerun this
// after any numerics-touching change.
//
//   gemma4_text model.gguf "prompt" [generate-count]
//
// The size (E4B or 12B) is detected from the checkpoint's block_count.
#include "../src/gemma4_loader.hpp"
#include "../src/tokenizer.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

template <class Model>
static int run(Model model, const Tokenizer& tokenizer, const std::vector<int32_t>& ids, size_t generate_count) {
    std::vector<TokenId> tokens;
    tokens.reserve(ids.size());
    for (const int32_t id : ids) tokens.push_back(TokenId{id});

    // One sequence, appended to as tokens are produced: the prefix cache keys
    // on its identity, so growing it in place is what makes decode incremental.
    EmbeddedSequence<Model::D> sequence;
    sequence.append(model.tokens(tokens), tokens);
    PrefixCache<Model> memo(model);
    const auto start = std::chrono::steady_clock::now();
    auto logits = memo.evaluate(sequence);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    const int32_t best = int32_t(logits.argmax());
    std::printf("prompt_tokens:");
    for (const int32_t id : ids) std::printf(" %d", id);
    std::printf("\nnext_token: %d\npiece: %s\nprefill: %.3fs (%zu tokens, %.1f tok/s)\n", best, tokenizer.decode1(best).c_str(), elapsed, ids.size(), double(ids.size()) / elapsed);

    const std::vector<ScoredToken> top = logits.top(5);
    for (size_t i = 0; i < top.size(); ++i) std::printf("top%zu: id=%d logprob=%.9f piece=%s\n", i + 1, int32_t(top[i].id), top[i].logprob, tokenizer.decode1(int32_t(top[i].id)).c_str());

    if (generate_count > 1) {
        const auto decode_start = std::chrono::steady_clock::now();
        size_t produced = 0;
        std::printf("generated:");
        std::string text;
        for (size_t i = 0; i < generate_count; ++i) {
            const TokenId token = logits.argmax();
            std::printf(" %d", int32_t(token));
            ++produced;
            text += tokenizer.decode1(int32_t(token));
            if (tokenizer.is_eog(int32_t(token)) || i + 1 == generate_count) break;
            const std::span<const TokenId> next(&token, 1);
            sequence.append(model.tokens(next), next);
            logits = memo.evaluate(sequence);
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
        const GGUF gguf(argv[1]);
        const Tokenizer tokenizer(gguf);
        const auto ids = tokenizer.encode(argv[2], /*add_bos=*/true, /*parse_special=*/true);
        const size_t generate_count = argc == 4 ? std::strtoull(argv[3], nullptr, 10) : 1;

        const size_t blocks = size_t(gguf.get_int("gemma4.block_count", 0));
        if (blocks == Gemma4_12BTextConfig::L) {
            std::fprintf(stderr, "Gemma 4 12B (%zu layers)\n", blocks);
            return run<Gemma4_12BModel>(gemma4_loader::load_12b_text(gguf), tokenizer, ids, generate_count);
        }
        if (blocks == Gemma4E4BTextConfig::L) {
            std::fprintf(stderr, "Gemma 4 E4B (%zu layers)\n", blocks);
            return run<Gemma4E4BModel>(gemma4_loader::load_e4b_text(gguf), tokenizer, ids, generate_count);
        }
        std::fprintf(stderr, "no compiled Gemma 4 config has %zu blocks\n", blocks);
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
