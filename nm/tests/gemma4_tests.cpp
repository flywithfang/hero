#include "../src/gemma4.hpp"
#include <cstdio>
#include <type_traits>

static_assert(TransformerArchitecture<Gemma4E4BArchitecture>);
static_assert(!std::is_default_constructible_v<Gemma4E4BTextWeights>);
static_assert(!std::is_default_constructible_v<Gemma4E4BTransformer>);

static int g_fail = 0;
static void check(bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg);
    if (!ok) ++g_fail;
}

template <size_t In, size_t Out>
static Linear<In, Out> zero_linear() {
    MatT<In, Out> matrix = MatT<In, Out>::owning();
    return Linear<In, Out>(Weight<In, Out>(std::move(matrix)));
}

template <size_t N>
static RMSNorm<N> unit_norm(Scalar eps = 1e-6f) {
    Vec<N> gamma;
    for (size_t i = 0; i < N; ++i) gamma[i] = 1.f;
    return RMSNorm<N>(std::move(gamma), eps);
}

int main() {
    std::printf("== Gemma E4B layer schedule ==\n");
    check(Gemma4E4BTextConfig::attention_kind(0) == GemmaAttentionKind::Sliding && Gemma4E4BTextConfig::attention_kind(5) == GemmaAttentionKind::Full && Gemma4E4BTextConfig::attention_kind(41) == GemmaAttentionKind::Full, "five sliding layers followed by one full layer");
    check(Gemma4E4BTextConfig::stores_shared_kv(22) && Gemma4E4BTextConfig::stores_shared_kv(23) && Gemma4E4BTextConfig::shares_kv(24), "last local/full producers feed the final 18 KV-sharing layers");

    std::printf("== Gemma normalization ==\n");
    {
        RMSNormNoScale<4> norm(1e-6f);
        Vec<4> a, b;
        for (size_t i = 0; i < 4; ++i) {
            a[i] = Scalar(i + 1);
            b[i] = 7 * a[i];
        }
        Vec<4> na = norm(a), nb = norm(b);
        Scalar delta = 0;
        for (size_t i = 0; i < 4; ++i) delta = std::max(delta, std::fabs(na[i] - nb[i]));
        check(delta < 1e-5f, "scale-free RMSNorm is scale invariant");
    }

    std::printf("== partial RoPE ==\n");
    {
        RotaryEmbedding<8, 10000, 1, 2> rope;
        static_assert(decltype(rope)::rotary_planes() == 2);
        Vec<8> value;
        for (size_t i = 0; i < 8; ++i) value[i] = Scalar(i + 1);
        Vec<8> original = copy(VecView<8>(value));
        rope.apply(slice_mut<8>(value, 0), 7);
        check(value[2] == original[2] && value[3] == original[3] && value[6] == original[6] && value[7] == original[7], "non-RoPE planes are exactly unchanged");
        check(value[0] != original[0] && value[4] != original[4], "the configured rotary prefix rotates");
    }

    std::printf("== heterogeneous K/V cache and Gemma attention ==\n");
    {
        KVCache<1, 2> full;
        Vec<2> k0, v0, k1, v1;
        k0[0] = 1;
        k0[1] = 0;
        v0[0] = 10;
        v0[1] = 20;
        k1[0] = 0;
        k1[1] = 1;
        v1[0] = 30;
        v1[1] = 40;
        full.append(Position{0}, k0, v0);
        full.append(Position{1}, k1, v1);
        Vec<4> q;
        q[0] = 1;
        q[1] = 0;
        q[2] = 1;
        q[3] = 0;
        Vec<4> global = attend<2, 1, 2>(q, full, Position{1});
        const Scalar p0 = std::exp(1.f) / (std::exp(1.f) + 1.f);
        check(std::fabs(global[0] - (p0 * 10 + (1 - p0) * 30)) < 1e-5f && global[0] == global[2], "GQA heads share KV and attention uses scale 1.0");
        Vec<4> local = attend<2, 1, 2>(q, full, Position{1}, 1);
        check(local[0] == 30 && local[1] == 40, "sliding attention excludes positions outside its window");

        KVCache<1, 2> ring(2);
        Vec<2> k2, v2;
        k2[0] = 2;
        k2[1] = 2;
        v2[0] = 50;
        v2[1] = 60;
        ring.append(Position{0}, k0, v0);
        ring.append(Position{1}, k1, v1);
        ring.append(Position{2}, k2, v2);
        check(ring.size() == 2 && ring.position(0).i == 1 && ring.position(1).i == 2, "bounded cache is an ordered sliding ring");
    }

    std::printf("== E4B cache topology ==\n");
    {
        Gemma4E4BCache cache;
        check(cache.local(0).capacity() == Gemma4E4BTextConfig::SLIDING_WINDOW, "ordinary local layers own bounded caches");
        check(cache.local(22).capacity() == 0 && &cache.local(24) == &cache.local(22), "shared local layers reuse the full-retention layer-22 cache");
        check(cache.global(23).capacity() == 0 && &cache.global(29) == &cache.global(23), "shared global layers reuse the layer-23 cache");
    }

    std::printf("== embedding scale and logit cap ==\n");
    {
        const Scalar scale = gemma_embedding_scale<2560>();
        const Scalar reference = bf16_to_fp32(fp32_to_bf16(std::sqrt(2560.f)));
        check(scale == reference, "embedding sqrt(D) includes BF16 rounding");
        check(std::fabs(gemma_softcap(1000.f, 30.f) - 30.f) < 1e-4f && gemma_softcap(-4.f, 30.f) < 0.f, "logit softcap is cap*tanh(x/cap)");
    }

    std::printf("== dense decoder tail anatomy ==\n");
    {
        // The layer equation now reads straight out of forward_layer() in each
        // architecture, so it has no mockable seam and is pinned by the parity
        // harness instead. What is still a component with numbers of its own is
        // the tail. Zero PLE projections make the residual contribute nothing,
        // leaving exactly the layer_output_scale on the FF residual.
        struct MockAttention {};
        GemmaPerLayerResidual<2, 1> ple(zero_linear<2, 1>(), zero_linear<1, 2>(), unit_norm<2>());
        GemmaPerLayerTail<2, 1> tail(std::move(ple), 2.f);
        auto x = [] {
            Vec<2> value;
            value[0] = 3;
            value[1] = -4;
            return value;
        };
        Vec<1> per_layer;
        per_layer[0] = 7;
        Vec<2> y = tail(x(), VecView<1>(per_layer));
        check(y[0] == 6 && y[1] == -8, "the PLE residual precedes layer scaling");

        Matrix<2> batch(1);
        batch.set_row(0, x());
        Matrix<1> batch_per_layer(1);
        batch_per_layer.set_row(0, per_layer);
        Matrix<2> batch_output = tail(std::move(batch), batch_per_layer.view());
        check(batch_output.row(0)[0] == 6 && batch_output.row(0)[1] == -8, "the matrix tail states the same equation as the scalar one");

        // 12B's tail types: a scalar-only tail, and no tail at all.
        Vec<2> scaled = GemmaLayerScaleTail<2>(2.f)(x());
        check(scaled[0] == 6 && scaled[1] == -8, "a PLE-free size still applies layer_output_scale");
        Vec<2> plain = GemmaNoTail<2>{}(x());
        check(plain[0] == 3 && plain[1] == -4, "a layer with no tail returns the FF residual unscaled");
        check(sizeof(GemmaDenseDecoderLayer<2, 2, MockAttention, GemmaNoTail<2>>) < sizeof(GemmaDenseDecoderLayer<2, 2, MockAttention, GemmaPerLayerTail<2, 1>>), "the absent tail costs no space");
    }

    std::printf("== Gemma 4 12B anatomy ==\n");
    {
        using C = Gemma4_12BTextConfig;
        size_t sliding = 0, global = 0;
        for (size_t l = 0; l < C::L; ++l) (C::attention_kind(l) == GemmaAttentionKind::Full ? global : sliding)++;
        check(sliding == 40 && global == 8, "12B is 40 sliding + 8 global layers");
        check(C::attention_kind(C::L - 1) == GemmaAttentionKind::Full, "the final layer is global, as the model card requires");
        check(C::kv_heads(0) == 8 && C::kv_heads(5) == 1, "KV head count differs per attention kind (8 sliding, 1 global)");
        check(C::kv_kind(0) == GemmaKVKind::Owned && C::kv_kind(5) == GemmaKVKind::Unified, "global layers use unified K/V, sliding layers own both");
        const double params = double(gemma_dense_param_count<C>());
        check(params > 11.7e9 && params < 12.1e9, "parameter count lands on the advertised ~11.95B");

        // Every Gemma 4 layer carries a scalar output scale; 12B has no PLE
        // but still has that, so its tail is scale-only rather than absent.
        GemmaLayerScaleTail<2> scale_tail(0.5f);
        Vec<2> h;
        h[0] = 6.f;
        h[1] = -8.f;
        Vec<2> scaled = scale_tail(std::move(h));
        check(scaled[0] == 3.f && scaled[1] == -4.f, "the layer output scale is applied last");

        Gemma4_12BCache cache;
        check(cache.local(0).capacity() == C::SLIDING_WINDOW, "sliding layers own a bounded ring, so their cost stops growing");
        check(cache.global(5).capacity() == 0, "global layers retain everything");
    }

    std::printf("== unified K/V takes one projection two ways ==\n");
    {
        // One head of width 2. W_k maps x -> (x0, x1). The key gets the learned
        // per-head RMSNorm; the value gets the scale-free one on the SAME raw
        // projection output, so the two differ only by that learned scale.
        MatT<2, 2> w = MatT<2, 2>::owning();
        w.raw()[0] = 1.f;
        w.raw()[1] = 0.f;  // out 0 = x0
        w.raw()[2] = 0.f;
        w.raw()[3] = 1.f;  // out 1 = x1
        Vec<2> gamma;
        gamma[0] = 3.f;
        gamma[1] = 3.f;
        GemmaUnifiedKeyValue<2, 1, 2> kv(Linear<2, 2>(Weight<2, 2>(std::move(w))), PerHeadNorm<1, 2, RMSNorm<2>>(RMSNorm<2>(std::move(gamma), 1e-6f)), PerHeadNorm<1, 2, RMSNormNoScale<2>>(RMSNormNoScale<2>(1e-6f)));

        Matrix<2> x(1);
        x.row_mut(0)[0] = 3.f;
        x.row_mut(0)[1] = 4.f;
        GemmaKeyValuePair<2> pair = kv.key_and_value(x.view());
        // rms([3,4]) = sqrt(12.5); normalized = [0.8485, 1.1314]
        const Scalar inv = 1.f / std::sqrt(12.5f);
        check(std::fabs(pair.value.row(0)[0] - 3.f * inv) < 1e-5f && std::fabs(pair.value.row(0)[1] - 4.f * inv) < 1e-5f, "the value is the raw projection with the scale-free norm");
        check(std::fabs(pair.key.row(0)[0] - 3.f * 3.f * inv) < 1e-5f, "the key is the same projection with the learned scale");
        Matrix<2> separate_key = kv.key(x.view());
        check(separate_key.row(0)[0] == pair.key.row(0)[0], "key() and key_and_value() agree");
    }

    std::printf("\n%s (%d failures)\n", g_fail ? "GEMMA4 TESTS FAILED" : "ALL GEMMA4 TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
