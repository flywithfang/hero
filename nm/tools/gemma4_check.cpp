#include "../src/gemma4_loader.hpp"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s gemma4-e4b-text.gguf\n", argv[0]);
        return 2;
    }
    try {
        GGUF gguf(argv[1]);
        gemma4_loader::validate_e4b_text_metadata(gguf);
        std::printf("Gemma 4 E4B metadata valid: %zu tensors\n", gguf.tensors().size());
        auto model = gemma4_loader::load_e4b_text(gguf);
        std::printf("Gemma 4 E4B immutable text assembly loaded (%zu layers)\n",
                    Gemma4E4BTextConfig::L);
        (void)model;
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
