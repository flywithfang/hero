#include "../src/gemma4_loader.hpp"
#include <cstdio>
#include <cstring>

// Validate a Gemma 4 text GGUF against a compiled config and build the
// immutable assembly. Size is auto-detected from block_count unless given.
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s gemma4-text.gguf [e4b|12b]\n", argv[0]);
        return 2;
    }
    try {
        GGUF gguf(argv[1]);
        std::string size = argc == 3 ? argv[2] : "";
        if (size.empty()) {
            const size_t blocks = size_t(gguf.get_int("gemma4.block_count", 0));
            size = blocks == Gemma4_12BTextConfig::L ? "12b" : blocks == Gemma4E4BTextConfig::L ? "e4b" : "";
            if (size.empty()) {
                std::fprintf(stderr, "no compiled Gemma 4 config has %zu blocks\n", blocks);
                return 1;
            }
        }
        if (size == "12b") {
            gemma4_loader::validate_12b_text_metadata(gguf);
            std::printf("Gemma 4 12B metadata valid: %zu tensors\n", gguf.tensors().size());
            auto model = gemma4_loader::load_12b_text(gguf);
            (void)model;
            std::printf(
                "Gemma 4 12B assembly loaded (%zu layers: 40 sliding + 8 global"
                " unified-K/V, no PLE, no shared KV)\n",
                Gemma4_12BTextConfig::L);
            return 0;
        }
        gemma4_loader::validate_e4b_text_metadata(gguf);
        std::printf("Gemma 4 E4B metadata valid: %zu tensors\n", gguf.tensors().size());
        auto model = gemma4_loader::load_e4b_text(gguf);
        std::printf("Gemma 4 E4B immutable text assembly loaded (%zu layers)\n", Gemma4E4BTextConfig::L);
        (void)model;
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
