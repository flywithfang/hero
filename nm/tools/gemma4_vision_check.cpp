#include "../src/gemma4_vision_loader.hpp"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s mmproj.gguf\n", argv[0]);
        return 2;
    }
    try {
        GGUF gguf(argv[1]);
        auto model = gemma4_vision_loader::load_e4b_vision(gguf);
        (void)model;
        std::printf("Gemma 4 E4B immutable vision assembly loaded (16 layers)\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
