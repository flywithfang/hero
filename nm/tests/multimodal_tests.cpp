#include "../src/multimodal.hpp"
#include <cstdio>
#include <type_traits>

static int g_fail = 0;
static void check(bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg);
    if (!ok) ++g_fail;
}

template <size_t D>
static TokenMatrix<D> rows(std::initializer_list<std::array<Scalar, D>> values) {
    TokenMatrix<D> out(values.size());
    size_t r = 0;
    for (const auto& value : values) {
        auto dst = out.row_mut(r++);
        std::copy(value.begin(), value.end(), dst.begin());
    }
    return out;
}

int main() {
    static_assert(!std::is_default_constructible_v<TokenMatrix<3>>);
    static_assert(!std::is_default_constructible_v<RGBImage>);
    static_assert(!std::is_default_constructible_v<MultimodalPrompt>);

    std::printf("== runtime token matrix ==\n");
    {
        auto m = rows<3>({{1, 2, 3}, {4, 5, 6}});
        check(m.rows() == 2 && m.cols() == 3, "runtime rows, static channels");
        check(m.row(1)[0] == 4 && m.row(1)[2] == 6, "rows preserve values");
        bool threw = false;
        try { (void)m.row(2); } catch (const std::out_of_range&) { threw = true; }
        check(threw, "row bounds are enforced");
    }

    std::printf("== multimodal embedding composition ==\n");
    {
        std::vector<EmbeddingSegment<2>> segments;
        segments.emplace_back(rows<2>({{1, 10}, {2, 20}}),
                              std::vector<TokenId>{TokenId{11}, TokenId{12}});
        segments.emplace_back(Modality::Image, rows<2>({{3, 30}, {4, 40}, {5, 50}}));
        segments.emplace_back(rows<2>({{6, 60}}), std::vector<TokenId>{TokenId{13}});
        auto seq = compose_embeddings(std::move(segments), 100);
        check(seq.tokens() == 6, "segments concatenate to one decoder sequence");
        check(seq.embedding(0)[0] == 1 && seq.embedding(5)[1] == 60,
              "composition preserves segment order and values");
        check(seq.position(0).i == 100 && seq.position(5).i == 105,
              "absolute positions remain continuous");
        check(seq.spans().size() == 3 && seq.spans()[1].modality() == Modality::Image &&
              seq.spans()[1].begin() == 2 && seq.spans()[1].end() == 5,
              "modality spans track soft-token placement");
        check(seq.ple_token_identity(0) == TokenId{11} &&
              seq.ple_token_identity(2) == TokenId{0} &&
              seq.ple_token_identity(5) == TokenId{13},
              "PLE preserves text identities and substitutes PAD for soft tokens");

        DecoderVisibility causal(seq.tokens());
        check(causal.allows(4, 1) && !causal.allows(1, 4), "causal visibility is lower triangular");
        DecoderVisibility image_bidir(seq.tokens(), {seq.spans()[1]});
        check(image_bidir.allows(2, 4), "image soft tokens can be bidirectional");
        check(!image_bidir.allows(1, 4), "bidirectionality does not leak into text");
    }

    std::printf("== owned prompt inputs ==\n");
    {
        MultimodalPrompt prompt({
            TextPrompt({TokenId{1}, TokenId{2}}),
            RGBImage(2, 1, {255, 0, 0, 0, 255, 0}),
        });
        check(prompt.parts().size() == 2, "prompt owns ordered text and image parts");
        bool threw = false;
        try { (void)RGBImage(2, 2, {0, 0, 0}); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "RGB image shape invariant is enforced");
    }

    std::printf("\n%s (%d failures)\n", g_fail ? "MULTIMODAL TESTS FAILED" : "ALL MULTIMODAL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
