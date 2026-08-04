// tokenizer_test — reads lines from stdin, prints "id id ..." per line using
// our Tokenizer (encode with add_bos=false, parse_special=true). --roundtrip
// checks decode(encode(x)) == x instead; tools/tokenizer_parity.py diffs the
// printed ids against llama-tokenize on the same vocab, which is the real gate.
//   tokenizer_test vocab.gguf [--roundtrip]
#include "../src/tokenizer.hpp"
#include <cstdio>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s vocab.gguf [--roundtrip]\n", argv[0]);
        return 2;
    }
    const bool roundtrip = argc > 2 && std::string(argv[2]) == "--roundtrip";
    try {
        const GGUF g(argv[1]);
        const Tokenizer tok(g);
        std::string line;
        int bad = 0, total = 0;
        while (std::getline(std::cin, line)) {
            auto ids = tok.encode(line, /*add_bos=*/false, /*parse_special=*/true);
            if (roundtrip) {
                const std::string back = tok.decode(ids);
                ++total;
                if (back != line) {
                    ++bad;
                    std::fprintf(stderr, "ROUNDTRIP FAIL: [%s] -> [%s]\n", line.c_str(), back.c_str());
                }
            } else {
                for (size_t i = 0; i < ids.size(); ++i) std::printf("%d%s", ids[i], i + 1 < ids.size() ? " " : "");
                std::printf("\n");
            }
        }
        if (roundtrip) std::fprintf(stderr, "roundtrip: %d/%d ok\n", total - bad, total);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
