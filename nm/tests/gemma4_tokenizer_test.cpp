#include "../src/tokenizer.hpp"
#include <cstdio>

static int failures = 0;

static void check_case(const Tokenizer& tokenizer, const char* text, std::initializer_list<int32_t> expected, bool expect_roundtrip = true) {
    const auto actual = tokenizer.encode(text, /*add_bos=*/true, /*parse_special=*/true);
    const std::vector<int32_t> want(expected);
    const bool ids_match = actual == want;
    std::printf("  [%s] %s\n", ids_match ? "PASS" : "FAIL", text);
    if (!ids_match) {
        ++failures;
        std::printf("    got:");
        for (int32_t id : actual) std::printf(" %d", id);
        std::printf("\n    want:");
        for (int32_t id : want) std::printf(" %d", id);
        std::printf("\n");
        return;
    }
    std::vector<int32_t> without_bos(actual.begin() + 1, actual.end());
    if (expect_roundtrip && tokenizer.decode(without_bos) != text) {
        ++failures;
        std::printf("    round-trip failed\n");
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s gemma4.gguf\n", argv[0]);
        return 2;
    }
    try {
        GGUF gguf(argv[1]);
        Tokenizer tokenizer(gguf);
        check_case(tokenizer, "Hello world", {2, 9259, 1902});
        check_case(tokenizer, "caf\xC3\xA9 \xE4\xB8\xAD\xE6\x96\x87 \xF0\x9F\x91\x8B", {2, 123125, 236859, 17346, 237364, 155818});
        check_case(tokenizer, "  spaced   text", {2, 138, 169862, 139, 1005});
        check_case(tokenizer, "x += 42; // code", {2, 236781, 3323, 236743, 236812, 236778, 236793, 973, 3393});
        check_case(tokenizer, "<|turn>user", {2, 105, 2364}, false);
        const auto turn_end = tokenizer.encode("<turn|>", false, true);
        if (turn_end.size() != 1 || !tokenizer.is_eog(turn_end[0])) {
            ++failures;
            std::printf("  [FAIL] <turn|> is an EOG token\n");
        } else {
            std::printf("  [PASS] <turn|> is an EOG token\n");
        }
        std::printf("%s (%d failures)\n", failures ? "GEMMA4 TOKENIZER TESTS FAILED" : "ALL GEMMA4 TOKENIZER TESTS PASSED", failures);
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
