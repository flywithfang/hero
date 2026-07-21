// logit_diff — M1 fidelity harness. Loads a GGUF into the compiled Model and,
// for a FIXED token id sequence (no tokenizer needed), either:
//   --last   : print the last-position argmax + top-20 logits (compare to a
//              reference dump, e.g. tools/ref_logits against llama.cpp).
//   --argmax : print the greedy argmax at EVERY position (teacher-forced on the
//              given ids) — feed a generated sequence here and check that every
//              position's argmax matches, i.e. greedy parity.
//   --gen N  : greedily generate N tokens from the given ids and print them.
//
// Config is chosen by --cfg {stories|1b|3b}. Bitwise equality vs llama.cpp is
// NOT expected (summation order); the gate is max|delta| ~1e-2 and argmax match.
#include "../src/llama_loader.hpp"
#include "../src/runtime.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

template <class C>
int run(const char* path, const std::vector<int>& ids, const std::string& mode, int genN) {
    GGUF g(path);
    const auto model = llama_loader::load<C>(g);

    std::vector<TokenId> toks;
    for (int id : ids) toks.push_back(TokenId{id});

    if (mode == "--gen") {
        AutoregressiveRuntime<LlamaArchitecture<C>> rt(model);
        std::vector<TokenId> eos{TokenId{int32_t(g.get_int("tokenizer.ggml.eos_token_id", 2))}};
        auto out = rt.generate(toks, genN, eos);
        std::printf("gen %zu:", out.size());
        for (TokenId t : out) std::printf(" %d", int(t));
        std::printf("\n");
        return 0;
    }

    if (mode == "--argmax") {
        // per-position argmax by teacher forcing: decode prefix [0..i], record
        // argmax; independent KV each step keeps it simple and exact.
        std::printf("pos token argmax\n");
        for (size_t i = 0; i < toks.size(); ++i) {
            LlamaKVCache<C> cache;
            const auto prefix = std::span<const TokenId>(toks.data(), i + 1);
            auto input = LlamaArchitecture<C>::embed(model, prefix, 0);
            Vec<C::V> lg = transformer_forward<LlamaArchitecture<C>>(
                model, cache, input);
            TokenId am = sample_greedy(lg);
            std::printf("%zu %d %d\n", i, int(toks[i]), int(am));
        }
        return 0;
    }

    // --last (default)
    LlamaKVCache<C> cache;
    auto input = LlamaArchitecture<C>::embed(model, toks, 0);
    Vec<C::V> lg = transformer_forward<LlamaArchitecture<C>>(
        model, cache, input);
    std::vector<int> idx(C::V);
    for (size_t i = 0; i < C::V; ++i) idx[i] = int(i);
    std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(),
                      [&](int a, int b){ return lg[a] > lg[b]; });
    std::printf("n_vocab %zu  argmax %d  logit %.6f\n", C::V, idx[0], lg[idx[0]]);
    for (int i = 0; i < 20; ++i) std::printf("%d %.6f\n", idx[i], lg[idx[i]]);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s model.gguf --cfg {stories|1b|3b} [--last|--argmax|--gen N] id id ...\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    std::string cfg = "stories", mode = "--last";
    int genN = 20;
    std::vector<int> ids;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cfg") { cfg = argv[++i]; }
        else if (a == "--last" || a == "--argmax") { mode = a; }
        else if (a == "--gen") { mode = a; genN = atoi(argv[++i]); }
        else ids.push_back(atoi(a.c_str()));
    }
    try {
        if (cfg == "stories") return run<Stories260K>(path, ids, mode, genN);
        if (cfg == "1b")      return run<Llama32_1B>(path, ids, mode, genN);
        if (cfg == "3b")      return run<Llama32_3B>(path, ids, mode, genN);
        std::fprintf(stderr, "unknown --cfg %s\n", cfg.c_str());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
