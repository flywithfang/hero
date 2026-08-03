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
static_assert(!std::is_default_constructible_v<GeluGatedMLP<2, 2>>);
static_assert(std::is_default_constructible_v<RotaryEmbedding<8, 10000>>);
static_assert(std::is_default_constructible_v<MultiHeadAttention<1, 1, 2>>);
static_assert(!std::is_convertible_v<Position, size_t>);
static_assert(!std::is_convertible_v<size_t, Position>);

static int g_fail = 0;
static void check(bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg);
    if (!ok) ++g_fail;
}
static std::mt19937 rng(42);
static void fill(Scalar* p, size_t n, float sd = 1.0f) {
    std::normal_distribution<Scalar> N(0, sd);
    for (size_t i = 0; i < n; ++i) p[i] = N(rng);
}

struct ToyCache {
    size_t tokens() const { return tokens_; }
    void advance(size_t count) { tokens_ += count; }

    size_t tokens_ = 0;
    Vec<2> prefix_sum;
};

// A model small enough to reason about exactly: one layer that replaces each
// token by the running prefix sum, so the PrefixCache purity property has a
// closed form. Same shape of entity as a real model — tensors (here, one
// scale) and math together, immutable and non-copyable.
class ToyModel {
public:
    static constexpr size_t D = 2;
    static constexpr size_t V = 3;
    static constexpr size_t L = 1;
    static constexpr size_t CTX = 16;
    using PrefixState = ToyCache;
    using PreparedInput = NoPreparedInput;

    explicit ToyModel(Scalar scale) : scale_(scale) {}
    ToyModel(const ToyModel&) = delete;
    ToyModel& operator=(const ToyModel&) = delete;
    ToyModel(ToyModel&&) = default;

    Matrix<D> tokens(std::span<const TokenId> ids) const {
        Matrix<D> X;
        X.reserve(ids.size());
        for (TokenId id : ids) {
            Vec<D> embedding;
            embedding[0] = scale_ * Scalar(int32_t(id));
            embedding[1] = scale_ * Scalar(int32_t(id) + 1);
            X.append(embedding);
        }
        return X;
    }
    Logits<V> forward(PrefixState& prefix_state, EmbeddedRows<D> input) const {
        ResidualStream<D> residual(input);
        Vec<D> sum = copy(VecView<D>(prefix_state.prefix_sum));
        Matrix<D> running;
        running.reserve(residual.tokens());
        for (size_t row = 0; row < residual.tokens(); ++row) {
            sum += residual.token(row);
            running.append(sum);
        }
        residual.set_matrix(std::move(running));
        prefix_state.prefix_sum = std::move(sum);
        prefix_state.advance(input.tokens());

        const VecView<D> last = residual.token(residual.tokens() - 1);
        Logits<V> logits;
        logits[0] = last[0];
        logits[1] = last[1];
        logits[2] = last[0] + last[1];
        return logits;
    }

private:
    Scalar scale_;
};

static_assert(TransformerModel<ToyModel>);

// Rows plus the ids they came from, which is all "embed this text" ever was.
template <class Model>
static EmbeddedSequence<Model::D> sequence_of(const Model& model, std::span<const TokenId> ids) {
    EmbeddedSequence<Model::D> sequence;
    sequence.append(model.tokens(ids), ids);
    return sequence;
}

int main() {
    std::printf("== RoPE offset identity (half-split pairing) ==\n");
    {
        // (R(p)q)·(R(m)k) depends only on m-p. HeadDim=8, base 10000.
        RotaryEmbedding<8, 10000> r;
        Vec<8> q0, k0;
        fill(q0.begin(), 8);
        fill(k0.begin(), 8);
        auto score = [&](Position p, Position m) {
            Vec<8> q = copy(VecView<8>(q0)), k = copy(VecView<8>(k0));
            r.apply(slice_mut<8>(q, 0), p);
            r.apply(slice_mut<8>(k, 0), m);
            return dot(VecView<8>(q), VecView<8>(k));
        };
        Scalar a = score(Position{3}, Position{1}), b = score(Position{10}, Position{8}), c = score(Position{60}, Position{58});
        check(std::fabs(a - b) < 1e-4 && std::fabs(b - c) < 1e-4, "same offset => same score at 3 abs positions");
    }
    std::printf("== RMSNorm scale invariance ==\n");
    {
        Vec<4> gamma;
        for (size_t i = 0; i < 4; ++i) gamma[i] = 1.f;
        RMSNorm<4> rn(std::move(gamma));
        Vec<4> a, b;
        for (size_t i = 0; i < 4; ++i) {
            a[i] = Scalar(2 * (i + 1));
            b[i] = 10 * a[i];
        }
        Vec<4> na = rn(a), nb = rn(b);
        Scalar d = 0;
        for (size_t i = 0; i < 4; ++i) d = std::max(d, std::fabs(na[i] - nb[i]));
        check(d < 1e-5f, "n(x) == n(10x)");
    }
    std::printf("== batched algebra agrees with row algebra ==\n");
    {
        Vec<4> gamma;
        for (size_t i = 0; i < 4; ++i) gamma[i] = Scalar(i + 1);
        RMSNorm<4> norm(std::move(gamma));
        Matrix<4> input;
        for (size_t row = 0; row < 2; ++row) {
            Vec<4> values;
            for (size_t channel = 0; channel < 4; ++channel) values[channel] = Scalar(1 + row * 4 + channel);
            input.append(values);
        }
        Matrix<4> batch = norm(input.view());
        Scalar difference = 0;
        for (size_t row = 0; row < input.rows(); ++row) {
            Vec<4> scalar = norm(input.row(row));
            for (size_t channel = 0; channel < 4; ++channel) difference = std::max(difference, std::fabs(batch.row(row)[channel] - scalar[channel]));
        }
        check(difference == 0.f, "matrix RMSNorm is exactly the row operation");
    }
    std::printf("== causal attention is softmax(scale*QK^T)V over visible keys ==\n");
    {
        // One KV head, two query rows at positions 0 and 1. Row 0 may see only
        // key 0; row 1 sees both. Scores use the conventional 1/sqrt(HeadDim).
        MultiHeadAttention<1, 1, 2> cache;
        Vec<2> key0, key1, value0, value1;
        key0[0] = 1.f;
        key1[1] = 1.f;
        value0[0] = 10.f;
        value0[1] = 0.f;
        value1[0] = 30.f;
        value1[1] = 0.f;
        Vec<2> query0, query1;
        query0[0] = 1.f;
        query1[1] = 1.f;
        Matrix<2> queries, keys, values;
        queries.append(query0);
        queries.append(query1);
        keys.append(key0);
        keys.append(key1);
        values.append(value0);
        values.append(value1);

        const Scalar scale = 1.f / std::sqrt(2.f);
        Matrix<2> attended = cache.append_and_attend(queries.view(), keys.view(), values.view(), Position{0}, CausalWindow::full(), scale);
        const Scalar weight1 = std::exp(scale) / (1.f + std::exp(scale));
        const Scalar expected1 = (1.f - weight1) * 10.f + weight1 * 30.f;
        check(attended.row(0)[0] == 10.f && std::fabs(attended.row(1)[0] - expected1) < 1e-5f, "the causal mask and row-wise softmax match the equation");

        // The same reduction under a width-1 window sees only the newest key.
        const MatrixView<2> second_query(queries.row(1).begin(), 1);
        Matrix<2> windowed = cache.attend(second_query, Position{1}, CausalWindow{1}, scale);
        check(windowed.row(0)[0] == 30.f, "a sliding window of one keeps only the current position");
    }
    std::printf("== softmax sums to 1 ==\n");
    {
        std::vector<Scalar> s(50);
        fill(s.data(), 50, 3.f);
        Softmax::apply(std::span<Scalar>(s.data(), 50));
        Scalar sum = 0;
        for (Scalar v : s) sum += v;
        check(std::fabs(sum - 1.f) < 1e-5f, "sum == 1");
    }
    std::printf("== SiLU value ==\n");
    {
        Vec<1> z;
        z[0] = 2.f;
        Silu::apply(z);
        check(std::fabs(z[0] - 1.7616f) < 1e-3f, "silu(2)=1.7616");
    }

    std::printf("== PrefixCache changes cost, never the result ==\n");
    {
        const ToyModel model(0.5f);
        PrefixCache<ToyModel> memo(model);
        const auto from_scratch = [&](std::span<const TokenId> ids) {
            PrefixCache<ToyModel> empty(model);
            return empty.evaluate(sequence_of(model, ids));
        };
        const auto embedded_from_scratch = [&](const EmbeddedSequence<2>& input) {
            PrefixCache<ToyModel> empty(model);
            return empty.evaluate(input);
        };
        const auto grow = [&](EmbeddedSequence<2>& sequence, TokenId id) { sequence.append(model.tokens(std::span<const TokenId>(&id, 1)), std::span<const TokenId>(&id, 1)); };

        const std::vector<TokenId> X2{TokenId{1}, TokenId{2}};
        EmbeddedSequence<2> X = sequence_of(model, X2);
        const auto same = [](const Logits<3>& a, const Logits<3>& b) {
            for (size_t i = 0; i < 3; ++i)
                if (a[i] != b[i]) return false;
            return true;
        };

        const Logits<3> pure_prefix = from_scratch(X2);
        const Logits<3> cached_prefix = memo.evaluate(X);
        bool equal = same(pure_prefix, cached_prefix);
        equal = equal && memo.reused_tokens() == 0 && memo.computed_tokens() == 2;

        grow(X, TokenId{3});
        const Logits<3> pure_extension = from_scratch(std::vector<TokenId>{TokenId{1}, TokenId{2}, TokenId{3}});
        const Logits<3> cached_extension = memo.evaluate(X);
        equal = equal && same(pure_extension, cached_extension) && memo.reused_tokens() == 2 && memo.computed_tokens() == 1;

        const Logits<3> repeated = memo.evaluate(X);
        equal = equal && same(repeated, pure_extension) && memo.reused_tokens() == 3 && memo.computed_tokens() == 0;

        // A different sequence OBJECT shares no identity with the cached one,
        // so the memo starts over even though the first row is the same.
        const std::vector<TokenId> divergent{TokenId{1}, TokenId{9}};
        const Logits<3> pure_divergent = from_scratch(divergent);
        EmbeddedSequence<2> divergent_input = sequence_of(model, divergent);
        const Logits<3> cached_divergent = memo.evaluate(divergent_input);
        equal = equal && same(cached_divergent, pure_divergent) && memo.reused_tokens() == 0 && memo.computed_tokens() == 2;

        EmbeddedSequence<2> embedded = sequence_of(model, divergent);
        PrefixCache<ToyModel> embedded_memo(model);
        const Logits<3> embedded_prefix = embedded_memo.evaluate(embedded);
        Vec<2> next_row;
        next_row[0] = 2.5f;
        next_row[1] = 3.f;
        Matrix<2> next_embedding;
        next_embedding.append(next_row);
        const TokenId next_id{5};
        embedded.append(std::move(next_embedding), std::span<const TokenId>(&next_id, 1));
        const Logits<3> pure_embedded_extension = embedded_from_scratch(embedded);
        const Logits<3> cached_embedded_extension = embedded_memo.evaluate(embedded);
        equal = equal && same(embedded_prefix, pure_divergent) && same(cached_embedded_extension, pure_embedded_extension) && embedded_memo.reused_tokens() == 2 && embedded_memo.computed_tokens() == 1;

        check(equal, "pure, extended, repeated, and divergent evaluations agree");
    }

    std::printf("== a query with no visible key is an error, never zero ==\n");
    {
        MultiHeadAttention<1, 1, 2> cache;
        Vec<2> key0, value0;
        key0[0] = 1.f;
        value0[0] = 5.f;
        cache.append(Position{0}, key0, value0);
        Vec<2> query;
        query[0] = 1.f;
        Matrix<2> queries;
        queries.append(query);
        bool threw = false;
        try {
            // Position 9 with a window of 2 can see nothing at position 0.
            (void)cache.attend(queries.view(), Position{9}, CausalWindow{2});
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "an empty visible set throws instead of softmaxing nothing");
    }

    std::printf("\n%s (%d failures)\n", g_fail ? "PROPERTY TESTS FAILED" : "ALL PROPERTY TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
