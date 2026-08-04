#include "../src/gemma4_loader.hpp"
#include "../src/gemma4_vision_loader.hpp"
#include "../src/image_io.hpp"
#include "../src/runtime.hpp"
#include "../src/tokenizer.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

std::vector<TokenId> to_tokens(const std::vector<int32_t>& ids) {
    std::vector<TokenId> tokens;
    tokens.reserve(ids.size());
    for (const int32_t id : ids) tokens.push_back(TokenId{id});
    return tokens;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::fprintf(stderr, "usage: %s text.gguf mmproj.gguf image.ppm prompt [generate-count]\n", argv[0]);
        return 2;
    }
    try {
        const GGUF text_gguf(argv[1]);
        const GGUF vision_gguf(argv[2]);
        const Tokenizer tokenizer(text_gguf);
        auto model = gemma4_loader::load_e4b_text(text_gguf);
        auto vision_model = gemma4_vision_loader::load_e4b_vision(vision_gguf);
        const RGBImage image = load_rgb_image(argv[3]);

        const auto vision_start = std::chrono::steady_clock::now();
        Matrix<Gemma4E4BTextConfig::D> image_rows = vision_model.encode(image);
        const double vision_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - vision_start).count();

        const std::string prefix = "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n<|image>";
        const std::string suffix = std::string("<image|>") + argv[4] + "<turn|>\n<|turn>model\n";
        auto prefix_ids = tokenizer.encode(prefix, /*add_bos=*/true, /*parse_special=*/true);
        auto suffix_ids = tokenizer.encode(suffix, /*add_bos=*/false, /*parse_special=*/true);
        const std::vector<TokenId> prefix_tokens = to_tokens(prefix_ids);
        const std::vector<TokenId> suffix_tokens = to_tokens(suffix_ids);
        const size_t image_tokens = image_rows.rows();

        EmbeddedSequence<Gemma4E4BTextConfig::D> sequence;
        sequence.append(model.embed(prefix_tokens), prefix_tokens);
        sequence.append_soft_tokens(std::move(image_rows));
        sequence.append(model.embed(suffix_tokens), suffix_tokens);

        PrefixCache<Gemma4E4BModel> memo(model);
        const auto text_start = std::chrono::steady_clock::now();
        auto logits = memo.evaluate(sequence);
        const double text_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - text_start).count();

        const size_t count = argc == 6 ? std::strtoull(argv[5], nullptr, 10) : 32;
        std::string answer;
        std::printf("image_tokens: %zu\nvision_elapsed: %.3fs\nprefill_elapsed: %.3fs\n", image_tokens, vision_seconds, text_seconds);
        std::printf("generated:");
        for (size_t i = 0; i < count; ++i) {
            const int32_t token = int32_t(logits.argmax());
            std::printf(" %d", token);
            answer += tokenizer.decode1(int32_t(token));
            if (tokenizer.is_eog(int32_t(token))) break;
            const TokenId next{token};
            sequence.append(model.embed(std::span<const TokenId>(&next, 1)), std::span<const TokenId>(&next, 1));
            logits = memo.evaluate(sequence);
        }
        std::printf("\nanswer: %s\n", answer.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
