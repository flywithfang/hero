#include "../src/gemma4_loader.hpp"
#include "../src/gemma4_vision_loader.hpp"
#include "../src/image_io.hpp"
#include "../src/runtime.hpp"
#include "../src/tokenizer.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

EmbeddingSegment<Gemma4E4BTextConfig::D> text_segment(const Gemma4E4BModel& model, const std::vector<int32_t>& ids) {
    if (ids.empty()) throw std::invalid_argument("empty text segment");
    TokenMatrix<Gemma4E4BTextConfig::D> embeddings(ids.size());
    std::vector<TokenId> identities;
    identities.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        identities.push_back(TokenId{ids[i]});
        embeddings.set_row(i, model.token(TokenId{ids[i]}));
    }
    return EmbeddingSegment<Gemma4E4BTextConfig::D>(std::move(embeddings), std::move(identities));
}

size_t argmax(const Vec<Gemma4E4BTextConfig::V>& logits) {
    size_t best = 0;
    for (size_t i = 1; i < Gemma4E4BTextConfig::V; ++i)
        if (logits[i] > logits[best]) best = i;
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::fprintf(stderr, "usage: %s text.gguf mmproj.gguf image.ppm prompt [generate-count]\n", argv[0]);
        return 2;
    }
    try {
        GGUF text_gguf(argv[1]);
        GGUF vision_gguf(argv[2]);
        Tokenizer tokenizer(text_gguf);
        auto model = gemma4_loader::load_e4b_text(text_gguf);
        auto vision_model = gemma4_vision_loader::load_e4b_vision(vision_gguf);
        RGBImage image = load_rgb_image(argv[3]);

        const auto vision_start = std::chrono::steady_clock::now();
        auto image_segment = vision_model.encode(image);
        const double vision_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - vision_start).count();

        const std::string prefix = "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n<|image>";
        const std::string suffix = std::string("<image|>") + argv[4] + "<turn|>\n<|turn>model\n";
        auto prefix_ids = tokenizer.encode(prefix, /*add_bos=*/true, /*parse_special=*/true);
        auto suffix_ids = tokenizer.encode(suffix, /*add_bos=*/false, /*parse_special=*/true);
        std::vector<EmbeddingSegment<Gemma4E4BTextConfig::D>> segments;
        segments.push_back(text_segment(model, prefix_ids));
        const size_t image_tokens = image_segment.embeddings().rows();
        segments.push_back(std::move(image_segment));
        segments.push_back(text_segment(model, suffix_ids));
        auto sequence = compose_embeddings(std::move(segments));

        PrefixCache<Gemma4E4BModel> memo;
        const auto text_start = std::chrono::steady_clock::now();
        auto logits = evaluate(model, sequence, memo);
        const double text_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - text_start).count();

        const size_t count = argc == 6 ? std::strtoull(argv[5], nullptr, 10) : 32;
        std::string answer;
        std::printf("image_tokens: %zu\nvision_elapsed: %.3fs\nprefill_elapsed: %.3fs\n", image_tokens, vision_seconds, text_seconds);
        std::printf("generated:");
        for (size_t i = 0; i < count; ++i) {
            const size_t token = argmax(logits);
            std::printf(" %zu", token);
            answer += tokenizer.decode1(int32_t(token));
            if (tokenizer.is_eog(int32_t(token))) break;
            std::vector<int32_t> generated{int32_t(token)};
            sequence.append(text_segment(model, generated));
            logits = evaluate(model, sequence, memo);
        }
        std::printf("\nanswer: %s\n", answer.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
