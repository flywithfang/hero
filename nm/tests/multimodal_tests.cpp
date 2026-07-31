#include "../src/multimodal.hpp"
#include <cstdio>
#include <type_traits>

static int g_fail = 0;
static void check(bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg);
    if (!ok) ++g_fail;
}

int main() {
    static_assert(!std::is_default_constructible_v<RGBImage>);

    std::printf("== runtime token matrix ==\n");
    {
        Matrix<3> m{{1, 2, 3}, {4, 5, 6}};
        check(m.rows() == 2 && m.cols() == 3, "runtime rows, static channels");
        check(m.row(1)[0] == 4 && m.row(1)[2] == 6, "rows preserve values");
        bool threw = false;
        try {
            (void)m.row(2);
        } catch (const std::out_of_range&) {
            threw = true;
        }
        check(threw, "row bounds are enforced");
    }

    std::printf("== a decoder input is rows, wherever they came from ==\n");
    {
        // Two rows the vocabulary produced, three an encoder invented, one more
        // from the vocabulary. The sequence cannot tell them apart afterwards,
        // and that is the point.
        const std::vector<TokenId> opening{TokenId{11}, TokenId{12}};
        const std::vector<TokenId> closing{TokenId{13}};

        EmbeddedSequence<2> seq;
        seq.append(Matrix<2>{{1, 10}, {2, 20}}, opening);
        seq.append_soft_tokens(Matrix<2>{{3, 30}, {4, 40}, {5, 50}});
        seq.append(Matrix<2>{{6, 60}}, closing);

        const EmbeddedRows<2> all = seq.view();
        check(seq.tokens() == 6, "appends concatenate into one decoder sequence");
        check(all.embedding(0)[0] == 1 && all.embedding(5)[1] == 60, "appending preserves order and values");
        check(all.position(0).i == 0 && all.position(5).i == 5, "row i is at conversation position i");
        check(all.token_id(0) == TokenId{11} && all.token_id(2) == SOFT_TOKEN_ID && all.token_id(5) == TokenId{13}, "rows keep the vocabulary id they came from, and soft tokens carry the placeholder");

        // What a warm cache is handed: the tail, as a view, knowing where it
        // was cut from. No copy, and nothing called a suffix.
        const EmbeddedRows<2> uncached = seq.from_row(2);
        check(uncached.tokens() == 4 && uncached.position(0).i == 2, "the uncached tail knows where it starts");
        check(uncached.embedding(0)[0] == 3 && uncached.token_id(3) == TokenId{13}, "the tail views the same rows and ids, not copies of them");

        // Rows and their ids are one append or none: they can never drift.
        bool mismatch_rejected = false;
        try {
            seq.append(Matrix<2>{{7, 70}, {8, 80}}, closing);
        } catch (const std::invalid_argument&) {
            mismatch_rejected = true;
        }
        check(mismatch_rejected && seq.tokens() == 6, "an id count that does not match the rows is refused, and changes nothing");

        bool empty_rejected = false;
        try {
            seq.append_soft_tokens(Matrix<2>{});
        } catch (const std::invalid_argument&) {
            empty_rejected = true;
        }
        check(empty_rejected, "appending nothing is a caller bug, not a no-op");
    }

    std::printf("== owned image input ==\n");
    {
        bool threw = false;
        try {
            (void)RGBImage(2, 2, {0, 0, 0});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "RGB image shape invariant is enforced");
    }

    std::printf("\n%s (%d failures)\n", g_fail ? "MULTIMODAL TESTS FAILED" : "ALL MULTIMODAL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
