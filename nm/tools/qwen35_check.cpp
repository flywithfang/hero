// qwen35_check — validate a Qwen 3.5 GGUF against a compiled config, build the
// immutable assembly, and optionally run the forward pass.
//
//   qwen35_check model.gguf {4b|9b} [prompt [generate-count]]
//
// With a prompt it is the Qwen logit-parity harness: prompt ids, top-5
// logprobs, and a greedy continuation.
#include "../src/qwen35_loader.hpp"
#include "../src/tokenizer.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>

template <class C>
static int check(const GGUF& gguf, const char* name, const char* prompt, size_t generate_count) {
    qwen35_loader::validate_metadata<C>(gguf);
    std::printf("Qwen 3.5 %s metadata valid: %zu tensors\n", name, gguf.tensors().size());
    auto model = qwen35_loader::load<C>(gguf);
    (void)model;
    constexpr size_t full = C::L / C::FULL_ATTENTION_INTERVAL;
    std::printf(
        "Qwen 3.5 %s assembly loaded: %zu layers "
        "(%zu gated-deltanet, %zu full attention)\n",
        name, C::L, C::L - full, full);
    std::printf(
        "carried state: %zu floats/layer recurrent (constant in T), "
        "%zu floats/token/layer attention\n",
        Qwen35State<C>::recurrent_floats_per_layer(), Qwen35State<C>::attention_floats_per_token_per_layer());
    if (!prompt) return 0;

    const Tokenizer tokenizer(gguf);
    const auto ids = tokenizer.encode(prompt, /*add_bos=*/false, /*parse_special=*/true);
    std::vector<TokenId> tokens;
    for (const int32_t id : ids) tokens.push_back(TokenId{id});
    std::printf("prompt_tokens:");
    for (const int32_t id : ids) std::printf(" %d", id);
    std::printf("\n");

    // One sequence, appended to as tokens are produced: the prefix cache keys
    // on its identity, so growing it in place is what makes decode incremental.
    EmbeddedSequence<C::D> sequence;
    sequence.append(model.tokens(tokens), tokens);
    PrefixCache<Qwen35Model<C>> memo(model);
    const auto prefill_start = std::chrono::steady_clock::now();
    auto logits = memo.evaluate(sequence);
    const double prefill = std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill_start).count();
    std::printf("prefill: %.3fs (%zu tokens, %.2f tok/s)\n", prefill, ids.size(), double(ids.size()) / prefill);

    const std::vector<ScoredToken> top = logits.top(5);
    for (size_t i = 0; i < top.size(); ++i) std::printf("top%zu: id=%d logprob=%.9f piece=%s\n", i + 1, int32_t(top[i].id), top[i].logprob, tokenizer.decode1(int32_t(top[i].id)).c_str());

    if (generate_count > 1) {
        const auto decode_start = std::chrono::steady_clock::now();
        std::string text;
        size_t produced = 0;
        std::printf("generated:");
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
        const double decode = std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();
        std::printf("\ntext: %s\ndecode: %.3fs (%zu tokens, %.2f tok/s)\n", text.c_str(), decode, produced, double(produced) / decode);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5 || (std::strcmp(argv[2], "4b") && std::strcmp(argv[2], "9b"))) {
        std::fprintf(stderr, "usage: %s qwen35.gguf {4b|9b} [prompt [generate-count]]\n", argv[0]);
        return 2;
    }
    try {
        const GGUF gguf(argv[1]);
        const char* prompt = argc > 3 ? argv[3] : nullptr;
        const size_t generate_count = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 1;
        return std::strcmp(argv[2], "4b") == 0 ? check<Qwen35_4BConfig>(gguf, "4B", prompt, generate_count) : check<Qwen35_9BConfig>(gguf, "9B", prompt, generate_count);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
