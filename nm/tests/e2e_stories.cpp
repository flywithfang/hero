// e2e_stories.cpp — load the real (tiny) stories260K Llama checkpoint through
// the full loader+forward+generate path and check numerical sanity: logits are
// finite, softmax is a valid distribution, greedy generation runs. This
// exercises every M1 seam (ggml layout, RoPE, RMSNorm, SwiGLU, tied/untied)
// at 260K params in milliseconds. Parity vs a reference oracle is a separate
// harness (tools/logit_diff) — this is the "does it run and stay finite" gate.
#include "../src/llama_loader.hpp"
#include "../src/runtime.hpp"
#include <cstdio>

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/Users/yongli/.unsloth/.cache/stories260K.gguf";
    using C = Stories260K;
    try {
        GGUF g(path);
        const auto model = llama_loader::load<C>(g);
        std::printf("loaded %s: %zu params config, %zu tensors\n",
                    path, llama_param_count<C>(), g.tensors().size());

        AutoregressiveRuntime<LlamaArchitecture<C>> rt(model);
        // fixed token ids (BOS=1 then a few) — no tokenizer needed here.
        std::vector<TokenId> prompt{TokenId{1}, TokenId{306}, TokenId{100}};
        Vec<C::V> lg = rt.prefill(prompt);

        // sanity: all finite, argmax valid, softmax sums to 1.
        bool finite = true; double mx = -1e30; size_t arg = 0;
        for (size_t v = 0; v < C::V; ++v) {
            if (!std::isfinite(lg[v])) finite = false;
            if (lg[v] > mx) { mx = lg[v]; arg = v; }
        }
        std::vector<Scalar> probs(C::V);
        for (size_t v = 0; v < C::V; ++v) probs[v] = lg[v];
        softmax(std::span<Scalar>(probs.data(), C::V));
        double psum = 0; for (double p : probs) psum += p;

        std::printf("prefill: finite=%s  argmax=%zu (logit %.4f)  softmax_sum=%.6f\n",
                    finite ? "yes" : "NO", arg, mx, psum);

        // greedy generate 20 tokens
        std::vector<TokenId> eos{TokenId{2}};
        auto out = rt.generate(prompt, 20, eos);
        std::printf("greedy 20:");
        for (TokenId t : out) std::printf(" %d", int(t));
        std::printf("\n");
        std::printf("stats: prefill %.1f tok/s, decode %.1f tok/s, ctx %zu\n",
                    rt.stats().prefill_tps(), rt.stats().decode_tps(), rt.context_used());

        bool ok = finite && std::fabs(psum - 1.0) < 1e-3;
        std::printf("E2E %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
