// property_tests — in-binary invariants that stay green every build (plan §5):
// RoPE offset identity, RMSNorm scale invariance, causal/sliding attention
// against its own equation, softmax sums to 1, and PrefixCache purity.
// Everything here is model-neutral: it tests the core, not a checkpoint.
#include "../src/attention.hpp"
#include "../src/transformer.hpp"
#include <cstdio>
#include <random>
#include <type_traits>

// Weight-carrying components are constructed from data, never default-built
// and filled in later; a rotary table and an empty cache are pure derived
// state, so those DO start from nothing.
static_assert(!std::is_default_constructible_v<RMSNorm<4>>);
static_assert(!std::is_default_constructible_v<Linear<2, 2>>);
static_assert(!std::is_default_constructible_v<PostNormalize<RMSNorm<4>>>);
static_assert(std::is_default_constructible_v<RotaryEmbedding<8, 10000>>);
static_assert(std::is_default_constructible_v<KVCache<1, 2>>);

static int g_fail = 0;
static void check(bool ok, const char* msg) { std::printf("  [%s] %s\n", ok?"PASS":"FAIL", msg); if(!ok) ++g_fail; }
static std::mt19937 rng(42);
static void fill(Scalar* p, size_t n, float sd=1.0f){ std::normal_distribution<Scalar> N(0,sd); for(size_t i=0;i<n;++i)p[i]=N(rng); }

struct ToyWeights {
    static constexpr size_t D = 2;
    static constexpr size_t V = 3;
    static constexpr size_t L = 1;

    explicit ToyWeights(Scalar scale) : scale_(scale) {}
    Scalar scale() const { return scale_; }

private:
    Scalar scale_;
};

struct ToyCache {
    size_t tokens = 0;
    Vec<2> prefix_sum;
};

struct ToyArchitecture {
    static constexpr size_t D = ToyWeights::D;
    static constexpr size_t V = ToyWeights::V;
    static constexpr size_t L = ToyWeights::L;
    static constexpr size_t CTX = 16;
    using Weights = ToyWeights;
    using PrefixState = ToyCache;
    using PreparedInput = NoPreparedInput;

    static PrefixState make_prefix_state() { return {}; }
    static size_t prefix_tokens(
        const PrefixState& prefix_state) {
        return prefix_state.tokens;
    }
    static void advance_prefix(
        PrefixState& prefix_state, size_t count) {
        prefix_state.tokens += count;
    }
    static EmbeddedSequence<D> embed(
        const Weights& weights,
        std::span<const TokenId> ids,
        size_t first_position) {
        return embed_text_tokens<D>(
            ids, first_position,
            [&](std::span<const TokenId> batch) {
                Matrix<D> X(batch.size());
                for (size_t row = 0; row < batch.size(); ++row) {
                    X.row_mut(row)[0] =
                        weights.scale() * Scalar(int32_t(batch[row]));
                    X.row_mut(row)[1] =
                        weights.scale() *
                        Scalar(int32_t(batch[row]) + 1);
                }
                return X;
            });
    }
    static PreparedInput prepare(
        const Weights&, const EmbeddedSequence<D>&) {
        return {};
    }
    static void forward_layer(
        const Weights&, PrefixState& prefix_state,
        const EmbeddedSequence<D>&,
        ResidualStream<D>& residual,
        PreparedInput&, size_t) {
        Vec<D> sum =
            copy(VecView<D>(prefix_state.prefix_sum));
        for (size_t row = 0; row < residual.tokens(); ++row) {
            sum += residual.token(row);
            residual.set_token(row, sum);
        }
        prefix_state.prefix_sum = std::move(sum);
    }
    static Vec<V> output(
        const Weights&, const ResidualStream<D>& residual) {
        const VecView<D> last =
            residual.token(residual.tokens() - 1);
        Vec<V> logits;
        logits[0] = last[0];
        logits[1] = last[1];
        logits[2] = last[0] + last[1];
        return logits;
    }
};

static_assert(TransformerArchitecture<ToyArchitecture>);

int main() {
    std::printf("== RoPE offset identity (half-split pairing) ==\n");
    {
        // (R(p)q)·(R(m)k) depends only on m-p. HeadDim=8, base 10000.
        RotaryEmbedding<8, 10000> r;
        Vec<8> q0, k0; fill(q0.begin(),8); fill(k0.begin(),8);
        auto score = [&](size_t p, size_t m){
            Vec<8> q=copy(VecView<8>(q0)), k=copy(VecView<8>(k0));
            r.apply(slice_mut<8>(q,0), p); r.apply(slice_mut<8>(k,0), m);
            return dot(VecView<8>(q), VecView<8>(k));
        };
        Scalar a=score(3,1), b=score(10,8), c=score(60,58);
        check(std::fabs(a-b)<1e-4 && std::fabs(b-c)<1e-4, "same offset => same score at 3 abs positions");
    }
    std::printf("== RMSNorm scale invariance ==\n");
    {
        Vec<4> gamma; for(size_t i=0;i<4;++i) gamma[i]=1.f;
        RMSNorm<4> rn(std::move(gamma));
        Vec<4> a,b; for(size_t i=0;i<4;++i){ a[i]=Scalar(2*(i+1)); b[i]=10*a[i]; }
        Vec<4> na=rn(a), nb=rn(b);
        Scalar d=0; for(size_t i=0;i<4;++i) d=std::max(d,std::fabs(na[i]-nb[i]));
        check(d<1e-5f, "n(x) == n(10x)");
    }
    std::printf("== batched algebra agrees with row algebra ==\n");
    {
        Vec<4> gamma;
        for (size_t i = 0; i < 4; ++i) gamma[i] = Scalar(i + 1);
        RMSNorm<4> norm(std::move(gamma));
        Matrix<4> input(2);
        for (size_t row = 0; row < input.rows(); ++row)
            for (size_t channel = 0; channel < 4; ++channel)
                input.row_mut(row)[channel] =
                    Scalar(1 + row * 4 + channel);
        Matrix<4> batch = norm(input.view());
        Scalar difference = 0;
        for (size_t row = 0; row < input.rows(); ++row) {
            Vec<4> scalar = norm(input.row(row));
            for (size_t channel = 0; channel < 4; ++channel)
                difference = std::max(
                    difference,
                    std::fabs(
                        batch.row(row)[channel] - scalar[channel]));
        }
        check(
            difference == 0.f,
            "matrix RMSNorm is exactly the row operation");
    }
    std::printf("== causal attention is softmax(scale*QK^T)V over visible keys ==\n");
    {
        // One KV head, two query rows at positions 0 and 1. Row 0 may see only
        // key 0; row 1 sees both. Scores use the conventional 1/sqrt(HeadDim).
        KVCache<1, 2> cache;
        Vec<2> key0, key1, value0, value1;
        key0[0] = 1.f;
        key1[1] = 1.f;
        value0[0] = 10.f; value0[1] = 0.f;
        value1[0] = 30.f; value1[1] = 0.f;
        Matrix<2> queries(2);
        queries.row_mut(0)[0] = 1.f;
        queries.row_mut(1)[1] = 1.f;
        Matrix<2> keys(2), values(2);
        keys.set_row(0, key0);   keys.set_row(1, key1);
        values.set_row(0, value0); values.set_row(1, value1);

        const Scalar scale = 1.f / std::sqrt(2.f);
        Matrix<2> attended = attend_and_cache<1, 1, 2>(
            queries.view(), keys.view(), values.view(), cache, 0, 0, scale);
        const Scalar weight1 =
            std::exp(scale) / (1.f + std::exp(scale));
        const Scalar expected1 = (1.f - weight1) * 10.f + weight1 * 30.f;
        check(
            attended.row(0)[0] == 10.f &&
                std::fabs(attended.row(1)[0] - expected1) < 1e-5f,
            "the causal mask and row-wise softmax match the equation");

        // The same reduction under a width-1 window sees only the newest key.
        Vec<2> query1 = copy(queries.row(1));
        Vec<2> windowed = attend<1, 1, 2>(query1, cache, Position{1}, 1, scale);
        check(windowed[0] == 30.f,
              "a sliding window of one keeps only the current position");
    }
    std::printf("== softmax sums to 1 ==\n");
    {
        std::vector<Scalar> s(50); fill(s.data(),50,3.f);
        softmax(std::span<Scalar>(s.data(),50));
        Scalar sum=0; for(Scalar v:s) sum+=v;
        check(std::fabs(sum-1.f)<1e-5f, "sum == 1");
    }
    std::printf("== SiLU value ==\n");
    { Vec<1> z; z[0]=2.f; silu(z); check(std::fabs(z[0]-1.7616f)<1e-3f, "silu(2)=1.7616"); }

    std::printf("== PrefixCache changes cost, never the result ==\n");
    {
        const Transformer<ToyArchitecture> T(ToyWeights{0.5f});
        PrefixCache<ToyArchitecture> memo;
        std::vector<TokenId> X{TokenId{1}, TokenId{2}};
        const auto from_scratch = [&](std::span<const TokenId> input) {
            PrefixCache<ToyArchitecture> empty;
            return T(input, empty);
        };
        const auto embedded_from_scratch =
            [&](const EmbeddedSequence<2>& input) {
                PrefixCache<ToyArchitecture> empty;
                return T(input, empty);
            };
        const auto same = [](const Vec<3>& a, const Vec<3>& b) {
            for (size_t i = 0; i < 3; ++i)
                if (a[i] != b[i]) return false;
            return true;
        };

        const Vec<3> pure_prefix = from_scratch(X);
        const Vec<3> cached_prefix = T(X, memo);
        bool equal = same(pure_prefix, cached_prefix);
        equal = equal && memo.reused_tokens() == 0 &&
                memo.computed_tokens() == 2;

        X.push_back(TokenId{3});
        const Vec<3> pure_extension = from_scratch(X);
        const Vec<3> cached_extension = T(X, memo);
        equal = equal && same(pure_extension, cached_extension) &&
                memo.reused_tokens() == 2 &&
                memo.computed_tokens() == 1;

        const Vec<3> repeated = T(X, memo);
        equal = equal && same(repeated, pure_extension) &&
                memo.reused_tokens() == 3 &&
                memo.computed_tokens() == 0;

        const std::vector<TokenId> divergent{
            TokenId{1}, TokenId{9}};
        const Vec<3> pure_divergent =
            from_scratch(divergent);
        const Vec<3> cached_divergent =
            T(divergent, memo);
        equal = equal && same(cached_divergent, pure_divergent) &&
                memo.reused_tokens() == 0 &&
                memo.computed_tokens() == 2;

        auto embedded = ToyArchitecture::embed(
            T.weights(),
            std::span<const TokenId>(divergent), 0);
        PrefixCache<ToyArchitecture> embedded_memo;
        const Vec<3> embedded_prefix =
            T(embedded, embedded_memo);
        Matrix<2> next_embedding(1);
        next_embedding.row_mut(0)[0] = 2.5f;
        next_embedding.row_mut(0)[1] = 3.f;
        embedded.append(EmbeddingSegment<2>(
            std::move(next_embedding),
            std::vector<TokenId>{TokenId{5}}));
        const Vec<3> pure_embedded_extension =
            embedded_from_scratch(embedded);
        const Vec<3> cached_embedded_extension =
            T(embedded, embedded_memo);
        equal = equal &&
                same(embedded_prefix, pure_divergent) &&
                same(
                    cached_embedded_extension,
                    pure_embedded_extension) &&
                embedded_memo.reused_tokens() == 2 &&
                embedded_memo.computed_tokens() == 1;

        check(
            equal,
            "pure, extended, repeated, and divergent evaluations agree");
    }

    std::printf("== a query with no visible key is an error, never zero ==\n");
    {
        KVCache<1, 2> cache;
        Vec<2> key0, value0;
        key0[0] = 1.f;
        value0[0] = 5.f;
        cache.append(Position{0}, key0, value0);
        Vec<2> query;
        query[0] = 1.f;
        bool threw = false;
        try {
            // Position 9 with a window of 2 can see nothing at position 0.
            (void)attend<1, 1, 2>(query, cache, Position{9}, 2);
        } catch (const std::runtime_error&) { threw = true; }
        check(threw, "an empty visible set throws instead of softmaxing nothing");
    }

    std::printf("\n%s (%d failures)\n", g_fail?"PROPERTY TESTS FAILED":"ALL PROPERTY TESTS PASSED", g_fail);
    return g_fail?1:0;
}
