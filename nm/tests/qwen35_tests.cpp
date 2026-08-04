// qwen35_tests — the hybrid stack, checked on numbers small enough to verify
// by hand. The real checkpoints are gigabytes; every claim below is about the
// shape of the computation, not about a particular set of weights.
#include "../src/qwen35.hpp"
#include <cstdio>

static int failures = 0;
static void check(bool condition, const char* message) {
    std::printf("  [%s] %s\n", condition ? "PASS" : "FAIL", message);
    failures += !condition;
}

// A toy config with the same TOPOLOGY as Qwen 3.5 and dimensions a person can
// hold in their head: 8 layers, so 6 recurrent and 2 full-attention.
struct ToyQwen {
    static constexpr size_t V = 11;
    static constexpr size_t D = 4;
    static constexpr size_t L = 8;
    static constexpr size_t CTX = 64;

    static constexpr size_t Hq = 2;
    static constexpr size_t Hkv = 1;
    static constexpr size_t HEAD_DIM = 4;
    static constexpr size_t ROPE_BASE = 10000;
    static constexpr size_t ROTARY_NUM = 1;
    static constexpr size_t ROTARY_DEN = 2;

    static constexpr size_t KEY_HEAD_DIM = 2;
    static constexpr size_t VALUE_HEAD_DIM = 2;
    static constexpr size_t KEY_HEADS = 1;
    static constexpr size_t VALUE_HEADS = 2;
    static constexpr size_t CONV_WIDTH = 3;

    static constexpr size_t FF = 6;
    static constexpr Scalar RMS_EPS = 1e-6f;
    static constexpr size_t FULL_ATTENTION_INTERVAL = 4;

    using OutputWeight = std::monostate;

    static constexpr bool is_recurrent(size_t layer) { return (layer + 1) % FULL_ATTENTION_INTERVAL != 0; }
};

static_assert(TransformerModel<Qwen35Model<ToyQwen>>);
static_assert(TransformerModel<Qwen35Model<Qwen35_4BConfig>>);
static_assert(TransformerModel<Qwen35Model<Qwen35_9BConfig>>);

// ---- helpers to build a deterministic toy weight ----------------------------

template <size_t In, size_t Out>
static Weight<In, Out> weight_from(Scalar (*f)(size_t, size_t)) {
    MatT<In, Out> matrix = MatT<In, Out>::owning();
    for (size_t out = 0; out < Out; ++out)
        for (size_t in = 0; in < In; ++in) matrix.raw()[out * In + in] = f(in, out);
    return Weight<In, Out>(std::move(matrix));
}
template <size_t In, size_t Out>
static Linear<In, Out> linear_from(Scalar (*f)(size_t, size_t)) {
    return Linear<In, Out>(weight_from<In, Out>(f));
}
static Scalar small(size_t in, size_t out) { return Scalar(((in * 7 + out * 3) % 5) - 2) * 0.1f; }
template <size_t N>
static Vec<N> filled(Scalar value) {
    Vec<N> v;
    for (size_t i = 0; i < N; ++i) v[i] = value;
    return v;
}
template <size_t N>
static Vec<N> ones() {
    return filled<N>(1.f);
}
template <size_t N>
static RMSNorm<N> unit_norm() {
    return RMSNorm<N>(ones<N>(), ToyQwen::RMS_EPS);
}

static GatedMLP<ToyQwen::D, ToyQwen::FF> make_toy_mlp() {
    using C = ToyQwen;
    return GatedMLP<C::D, C::FF>(linear_from<C::D, C::FF>(small), linear_from<C::D, C::FF>(small), linear_from<C::FF, C::D>(small));
}

static Qwen35Block<ToyQwen, QwenRecurrentMixer<ToyQwen>> make_recurrent_layer() {
    using C = ToyQwen;
    using Mixer = QwenRecurrentMixer<C>;
    Vec<C::CONV_WIDTH> taps;  // the same taps on every channel
    for (size_t tap = 0; tap < C::CONV_WIDTH; ++tap) taps[tap] = tap == C::CONV_WIDTH - 1 ? 1.f : 0.25f;
    Matrix<C::CONV_WIDTH> conv;
    conv.reserve(Mixer::CONV_CHANNELS);
    for (size_t c = 0; c < Mixer::CONV_CHANNELS; ++c) conv.append(taps);
    // decay_scale must be <= 0: the checkpoint stores -exp(A_log), which is
    // what makes exp(gate) a contraction. A positive value here makes the
    // state grow every token; the loader rejects that.
    Mixer mixer{.WQKV = linear_from<C::D, Mixer::CONV_CHANNELS>(small), .WZ = linear_from<C::D, Mixer::VALUE_WIDTH>(small), .conv = std::move(conv), .Wbeta = linear_from<C::D, C::VALUE_HEADS>(small), .Walpha = linear_from<C::D, C::VALUE_HEADS>(small), .dt_bias = Vec<C::VALUE_HEADS>{}, .decay_scale = filled<C::VALUE_HEADS>(-1.f), .head_norm = PerHeadNorm<RMSNorm<C::VALUE_HEAD_DIM>>(unit_norm<C::VALUE_HEAD_DIM>()), .WO = linear_from<Mixer::VALUE_WIDTH, C::D>(small)};
    return Qwen35Block<C, Mixer>{unit_norm<C::D>(), std::move(mixer), unit_norm<C::D>(), make_toy_mlp()};
}

static Qwen35Block<ToyQwen, QwenAttentionMixer<ToyQwen>> make_attention_layer() {
    using C = ToyQwen;
    using Mixer = QwenAttentionMixer<C>;
    Mixer mixer{.WQG = linear_from<C::D, Mixer::GATED_QW>(small), .q_norm = PerHeadNorm<RMSNorm<C::HEAD_DIM>>(unit_norm<C::HEAD_DIM>()), .WK = linear_from<C::D, Mixer::KW>(small), .k_norm = PerHeadNorm<RMSNorm<C::HEAD_DIM>>(unit_norm<C::HEAD_DIM>()), .WV = linear_from<C::D, Mixer::KW>(small), .WO = linear_from<Mixer::QW, C::D>(small)};
    return Qwen35Block<C, Mixer>{unit_norm<C::D>(), std::move(mixer), unit_norm<C::D>(), make_toy_mlp()};
}

static Qwen35Model<ToyQwen> make_toy_model(size_t drop_attention_layers = 0) {
    std::vector<Qwen35Model<ToyQwen>::RecurrentLayer> recurrent;
    std::vector<Qwen35Model<ToyQwen>::AttentionLayer> attention;
    for (size_t i = 0; i < ToyQwen::L; ++i) {
        if (ToyQwen::is_recurrent(i))
            recurrent.push_back(make_recurrent_layer());
        else
            attention.push_back(make_attention_layer());
    }
    for (size_t i = 0; i < drop_attention_layers; ++i) attention.pop_back();
    return Qwen35Model<ToyQwen>(weight_from<ToyQwen::D, ToyQwen::V>(small), std::monostate{}, std::move(recurrent), std::move(attention), unit_norm<ToyQwen::D>());
}

int main() {
    std::printf("== the hybrid layer schedule ==\n");
    {
        size_t recurrent = 0, full = 0;
        for (size_t l = 0; l < Qwen35_4BConfig::L; ++l) (Qwen35_4BConfig::is_recurrent(l) ? recurrent : full)++;
        check(recurrent == 24 && full == 8, "4B is 24 gated-deltanet layers and 8 full-attention layers");
        check(!Qwen35_4BConfig::is_recurrent(Qwen35_4BConfig::L - 1), "the last layer is full attention");
    }

    std::printf("== state cost: only attention layers grow with T ==\n");
    {
        using State = Qwen35State<Qwen35_4BConfig>;
        check(State::recurrent_floats_per_layer() == 32 * 128 * 128, "a delta layer carries Hv*Dk*Dv floats, independent of T");
        check(State::attention_floats_per_token_per_layer() == 2 * 4 * 256, "an attention layer carries 2*Hkv*HeadDim floats PER TOKEN");
    }

    std::printf("== causal conv1d ==\n");
    {
        // Two channels, width 3, taps [1, 10, 100] on both channels: the
        // output at token t is x[t-2] + 10*x[t-1] + 100*x[t].
        const Matrix<3> taps{{1.f, 10.f, 100.f}, {1.f, 10.f, 100.f}};
        CausalConv1dState<2, 3> conv;
        Vec<2> x1, x2, x3;
        x1[0] = 1.f;
        x1[1] = 0.f;
        x2[0] = 2.f;
        x2[1] = 0.f;
        x3[0] = 3.f;
        x3[1] = 0.f;
        Vec<2> y1 = conv.step(x1, taps);
        Vec<2> y2 = conv.step(x2, taps);
        Vec<2> y3 = conv.step(x3, taps);
        check(y1[0] == 100.f, "the first token sees only itself (causal zero pad)");
        check(y2[0] == 210.f, "the second token sees one step of history");
        check(y3[0] == 321.f, "the third token sees the full window");

        CausalConv1dState<2, 3> restarted;
        Vec<2> again = restarted.step(x1, taps);
        check(again[0] == y1[0], "a fresh state reproduces the first token");
    }

    std::printf("== the gated delta rule ==\n");
    {
        // Dk = Dv = 2, state starts at zero. With beta = 1 and gate = 0, one
        // step writes v into the state along k, so reading back with q = k
        // returns v exactly. That is the defining property of the delta rule:
        // it stores an association and can retrieve it.
        DeltaNetState<1, 2, 2> state;
        Vec<2> k, v;
        k[0] = 1.f;
        k[1] = 0.f;  // unit key
        v[0] = 3.f;
        v[1] = -5.f;
        Vec<2> out = gated_delta_step<2, 2>(state.head(0), VecView<2>(k), VecView<2>(k), VecView<2>(v), 0.f, 1.f);
        check(std::fabs(out[0] - 3.f) < 1e-6f && std::fabs(out[1] + 5.f) < 1e-6f, "writing v at key k and reading at k returns v");

        // An orthogonal key reads nothing back: the state is an associative
        // memory, not a bag of values.
        Vec<2> orthogonal;
        orthogonal[0] = 0.f;
        orthogonal[1] = 1.f;
        Vec<2> miss = gated_delta_step<2, 2>(state.head(0), VecView<2>(orthogonal), VecView<2>(orthogonal), Vec<2>{}, 0.f, 0.f);
        check(std::fabs(miss[0]) < 1e-6f && std::fabs(miss[1]) < 1e-6f, "an orthogonal query retrieves nothing");

        // The gate is what lets the state forget: a strongly negative gate
        // decays the stored association toward zero.
        DeltaNetState<1, 2, 2> decaying;
        (void)gated_delta_step<2, 2>(decaying.head(0), VecView<2>(k), VecView<2>(k), VecView<2>(v), 0.f, 1.f);
        Vec<2> faded = gated_delta_step<2, 2>(decaying.head(0), VecView<2>(k), VecView<2>(k), Vec<2>{}, -4.f, 0.f);
        check(std::fabs(faded[0] - 3.f * std::exp(-4.f)) < 1e-5f, "exp(gate) decays what the state remembers");
    }

    std::printf("== softplus and l2 normalization ==\n");
    {
        check(std::fabs(softplus(0.f) - std::log(2.f)) < 1e-6f, "softplus(0)=ln2");
        check(softplus(60.f) == 60.f, "softplus saturates to identity, never inf");
        Vec<2> x;
        x[0] = 3.f;
        x[1] = 4.f;
        Vec<2> unit = l2_normalize<2>(VecView<2>(x));
        check(std::fabs(unit[0] - 0.6f) < 1e-6f && std::fabs(unit[1] - 0.8f) < 1e-6f, "l2_normalize divides by the norm, not by sqrt(N)");
    }

    std::printf("== gated attention splits its query projection ==\n");
    {
        // Two heads of width 2, packed [q0 | g0 | q1 | g1].
        const Matrix<8> packed{{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}};
        const HeadPair<2, 2> split = split_head_pairs<2, 2>(packed.view());
        check(split.first.row(0)[0] == 0.f && split.first.row(0)[1] == 1.f && split.first.row(0)[2] == 4.f && split.first.row(0)[3] == 5.f, "queries are the first half of each head pair");
        check(split.second.row(0)[0] == 2.f && split.second.row(0)[1] == 3.f && split.second.row(0)[2] == 6.f && split.second.row(0)[3] == 7.f, "gates are the second half of each head pair");
    }

    std::printf("== the schedule cannot be silently violated ==\n");
    {
        // A layer of the wrong KIND is a compile error: the two kinds are
        // different types in different vectors. A missing layer is rejected by
        // the model's constructor, so a mis-assembled model cannot exist at all
        // — the check happens once at load, not once per evaluation.
        bool threw = false;
        try {
            make_toy_model(/*drop_attention_layers=*/1);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "a model whose layer counts disagree with the schedule cannot be constructed");
        check(!std::is_copy_constructible_v<Qwen35Model<ToyQwen>>, "a loaded model's weights are never copied");
    }

    std::printf("== the hybrid stack runs, and PrefixCache stays pure ==\n");
    {
        const Qwen35Model<ToyQwen> model = make_toy_model();
        std::vector<TokenId> input{TokenId{1}, TokenId{4}, TokenId{2}, TokenId{7}, TokenId{0}};

        PrefixCache<Qwen35Model<ToyQwen>> scratch(model);
        EmbeddedSequence<ToyQwen::D> one_shot;
        one_shot.append(model.embed(input), input);
        const Logits<ToyQwen::V> whole = scratch.evaluate(one_shot);

        // Incremental decode must equal one-shot prefill. This is the property
        // the recurrent mixer could most easily break: its state is sequential,
        // so an off-by-one in the conv history or a missed decay shows up here
        // and nowhere else.
        PrefixCache<Qwen35Model<ToyQwen>> incremental(model);
        Logits<ToyQwen::V> stepwise;
        EmbeddedSequence<ToyQwen::D> growing;
        for (size_t n = 0; n < input.size(); ++n) {
            const std::span<const TokenId> next(input.data() + n, 1);
            growing.append(model.embed(next), next);
            stepwise = incremental.evaluate(growing);
        }

        Scalar worst = 0;
        for (size_t v = 0; v < ToyQwen::V; ++v) worst = std::max(worst, std::fabs(whole[v] - stepwise[v]));
        check(worst < 1e-4f, "token-by-token decode matches one-shot prefill");
        check(incremental.reused_tokens() == input.size() - 1, "the last step reused the whole prefix");

        bool finite = true;
        for (size_t v = 0; v < ToyQwen::V; ++v) finite = finite && std::isfinite(whole[v]);
        check(finite, "logits are finite through 6 recurrent and 2 attention layers");
    }

    std::printf("\n%s (%d failures)\n", failures ? "QWEN35 TESTS FAILED" : "ALL QWEN35 TESTS PASSED", failures);
    return failures ? 1 : 0;
}
