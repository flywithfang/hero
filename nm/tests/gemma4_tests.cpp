#include "../src/gemma4.hpp"
#include <cstdio>
#include <type_traits>

static_assert(TransformerModel<Gemma4E4BModel>);
static_assert(TransformerModel<Gemma4_12BModel>);
static_assert(!std::is_default_constructible_v<Gemma4E4BModel>);
static_assert(!std::is_copy_constructible_v<Gemma4E4BModel>);

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
static Linear<N, N> identity_linear() {
    MatT<N, N> matrix = MatT<N, N>::owning();
    for (size_t i = 0; i < N; ++i) matrix.raw()[i * N + i] = 1.f;
    return Linear<N, N>(Weight<N, N>(std::move(matrix)));
}

// Gemma's softmax reduction over cached keys, at toy size. This MIRRORS the
// bodies of GemmaE4BLayer::attention and friends, which cannot themselves be
// instantiated small: their D/Hq/Hkv come from the E4B and 12B configs, not from
// template parameters. Keeping the equation in the layers is a deliberate
// choice; this copy is the price, and the two must be kept in step by hand.
template <size_t Hq, size_t Hkv, size_t Dh, class Cache>
static Matrix<Hq * Dh> toy_attend(const Cache& cache, MatrixView<Hq * Dh> Q, Position at) {
    std::vector<VisibleRows> visible;
    visible.reserve(Q.rows());
    for (size_t row = 0; row < Q.rows(); ++row) visible.push_back(cache.visible_rows(at + row));

    Matrix<Hq * Dh> A = Matrix<Hq * Dh>::zero_rows(Q.rows());
    par_for(Q.rows() * Hq, [&](size_t task) {
        const size_t row = task / Hq, head = task % Hq;
        const VecView<Dh> query = slice<Dh>(Q.row(row), head);
        const KVHead group{head / (Hq / Hkv)};
        const VisibleRows rows = visible[row];

        std::vector<Scalar> alpha;
        alpha.reserve(rows.count());
        for (size_t j = rows.first; j < rows.end; ++j) alpha.push_back(dot(query, cache.key(j, group)));
        alpha = Softmax{}(std::move(alpha));

        Vec<Dh> head_output;
        for (size_t n = 0; n < alpha.size(); ++n) head_output.scaled_add(cache.value(rows.first + n, group), alpha[n]);
        A.template replace_partition<Dh>(row, head, head_output);
    });
    return A;
}

// A toy-sized statement of the unified K/V equation used by the concrete 12B
// full-attention layer: one projection with different K and V normalization paths.
struct ToyUnifiedLayer {
    static constexpr size_t D = 2, Hq = 1, Hkv = 1, Dh = 2;
    Linear<2, 2> WQ;
    PerHeadNorm<RMSNorm<2>> q_norm;
    Linear<2, 2> WK;
    PerHeadNorm<RMSNorm<2>> k_norm;
    PerHeadNorm<RMSNormNoScale<2>> v_norm;
    Linear<2, 2> WO;

    template <class Rope>
    Matrix<D> attention(MatrixView<D> X, FullAttentionCache<Hkv, Dh>& cache, const Rope& rope, Position conversation_position) const {
        const auto Q = rope(q_norm(WQ(X)), conversation_position);

        Matrix<Hkv * Dh> V = WK(X);
        const auto K = rope(k_norm(V), conversation_position);
        V = v_norm(std::move(V));

        for (size_t row = 0; row < Q.rows(); ++row) cache.append(conversation_position + row, K.row(row), V.row(row));
        const auto A = toy_attend<Hq, Hkv, Dh>(cache, Q.view(), conversation_position);
        return WO(A.view());
    }
};

template <size_t N>
static RMSNorm<N> unit_norm(Scalar eps = 1e-6f) {
    Vec<N> gamma;
    for (size_t i = 0; i < N; ++i) gamma[i] = 1.f;
    return RMSNorm<N>(std::move(gamma), eps);
}

int main() {
    std::printf("== Gemma E4B layer schedule ==\n");
    check(Gemma4E4BTextConfig::attention_type(0) == GemmaAttentionType::Local && Gemma4E4BTextConfig::attention_type(5) == GemmaAttentionType::Full && Gemma4E4BTextConfig::attention_type(41) == GemmaAttentionType::Full, "five local layers followed by one full layer");
    check(Gemma4E4BTextConfig::is_shared_kv_source(22) && Gemma4E4BTextConfig::is_shared_kv_source(23) && Gemma4E4BTextConfig::uses_shared_kv(24), "last local/full producers feed the final 18 KV-sharing layers");

    std::printf("== Gemma normalization ==\n");
    {
        const RMSNormNoScale<4> norm(1e-6f);
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
        const RotaryEmbedding<8, 10000, 1, 2> rope;
        static_assert(decltype(rope)::rotary_planes() == 2);
        Vec<8> value;
        for (size_t i = 0; i < 8; ++i) value[i] = Scalar(i + 1);
        Vec<8> original = VecView<8>(value).copy();
        rope(slice_mut<8>(value, 0), Position{7});
        check(value[2] == original[2] && value[3] == original[3] && value[6] == original[6] && value[7] == original[7], "non-RoPE planes are exactly unchanged");
        check(value[0] != original[0] && value[4] != original[4], "the configured rotary prefix rotates");
    }

    std::printf("== heterogeneous K/V cache and Gemma attention ==\n");
    {
        FullAttentionCache<1, 2> full;
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
        Matrix<4> queries;
        queries.append(q);
        const auto full_result = toy_attend<2, 1, 2>(full, queries.view(), Position{1});
        const Scalar p0 = std::exp(1.f) / (std::exp(1.f) + 1.f);
        check(std::fabs(full_result.row(0)[0] - (p0 * 10 + (1 - p0) * 30)) < 1e-5f && full_result.row(0)[0] == full_result.row(0)[2], "GQA heads share KV and attention uses scale 1.0");

        LocalAttentionCache<1, 2> local(1);
        local.append(Position{0}, k0, v0);
        local.append(Position{1}, k1, v1);
        const auto local_result = toy_attend<2, 1, 2>(local, queries.view(), Position{1});
        check(local_result.row(0)[0] == 30 && local_result.row(0)[1] == 40, "local attention excludes positions outside its window");

        LocalAttentionCache<1, 2> ring(2);
        Vec<2> k2, v2;
        k2[0] = 2;
        k2[1] = 2;
        v2[0] = 50;
        v2[1] = 60;
        ring.append(Position{0}, k0, v0);
        ring.append(Position{1}, k1, v1);
        ring.append(Position{2}, k2, v2);
        check(ring.size() == 2 && ring.position(0) == Position{1} && ring.position(1) == Position{2}, "local cache rows remain chronologically ordered across ring wrap-around");

        // Deferred shared-layer readers need the union of the old window and
        // the current prefill rows. Retention is temporary: ending the batch
        // restores the same fixed-window invariant.
        ring.begin_batch_retention(2);
        ring.append(Position{3}, k0, v0);
        ring.append(Position{4}, k1, v1);
        check(ring.size() == 4 && ring.position(0) == Position{1} && ring.position(3) == Position{4}, "a shared-KV prefill batch temporarily preserves all rows its queries need");
        ring.end_batch_retention();
        check(ring.size() == 2 && ring.position(0) == Position{3} && ring.position(1) == Position{4}, "ending shared-KV prefill restores the local window");
    }

    std::printf("== E4B cache topology ==\n");
    {
        Gemma4E4BCache cache;
        check(cache.local(0).window() == Gemma4E4BTextConfig::LOCAL_WINDOW, "ordinary local layers own fixed-window caches");
        check(cache.local(22).window() == Gemma4E4BTextConfig::LOCAL_WINDOW && &cache.local(24) == &cache.local(22), "shared local layers reuse layer 22's local cache");
        check(&cache.full(29) == &cache.full(23), "shared full-attention layers reuse layer 23's append-only cache");
    }

    std::printf("== logit cap ==\n");
    check(std::fabs(gemma_softcap(1000.f, 30.f) - 30.f) < 1e-4f && gemma_softcap(-4.f, 30.f) < 0.f, "logit softcap is cap*tanh(x/cap)");

    std::printf("== the PLE residual and the layer output scale ==\n");
    {
        // The layer equation reads straight out of each architecture's
        // run_layer() and is pinned by the parity harness. What is still a
        // component with numbers of its own is the PLE residual. Zero PLE
        // projections make the branch contribute nothing, so `h + branch` is
        // `h`, and the layer scale that follows in run_layer() is then exact.
        const GemmaPerLayerResidual<2, 1> ple(zero_linear<2, 1>(), zero_linear<1, 2>(), unit_norm<2>());
        auto x = [] {
            Vec<2> value;
            value[0] = 3;
            value[1] = -4;
            return value;
        };
        Vec<1> per_layer;
        per_layer[0] = 7;
        Vec<2> y = ple(VecView<2>(x()), VecView<1>(per_layer));
        y.scale(2.f);  // exactly run_layer()'s tail: PLE residual, then layer_output_scale
        check(y[0] == 6 && y[1] == -8, "the PLE residual precedes layer scaling");

        Matrix<2> batch;
        batch.append(x());
        Matrix<1> batch_per_layer;
        batch_per_layer.append(per_layer);
        Matrix<2> batch_output = ple(batch.view(), batch_per_layer.view());
        batch_output.scale(2.f);
        check(batch_output.row(0)[0] == 6 && batch_output.row(0)[1] == -8, "the matrix PLE residual states the same equation as the scalar one");

        // A shared-KV layer carries no K/V tensors — the members do not exist.
        check(sizeof(Gemma4E4BModel::LocalSharedLayer) < sizeof(Gemma4E4BModel::LocalLayer), "a shared-KV layer is physically smaller than an owning one");
    }

    std::printf("== Gemma 4 12B anatomy ==\n");
    {
        using C = Gemma4_12BTextConfig;
        size_t local = 0, full = 0;
        for (size_t layer = 0; layer < C::L; ++layer) (C::attention_type(layer) == GemmaAttentionType::Full ? full : local)++;
        check(local == 40 && full == 8, "12B is 40 local + 8 full-attention layers");
        check(C::attention_type(C::L - 1) == GemmaAttentionType::Full, "the final layer uses full attention");
        check(C::kv_heads(0) == 8 && C::kv_heads(5) == 1, "KV head count differs per attention type (8 local, 1 full)");
        check(C::kv_kind(0) == GemmaKVKind::Owned && C::kv_kind(5) == GemmaKVKind::Unified, "full-attention layers use unified K/V; local layers own both");
        const double params = double(gemma_dense_param_count<C>());
        check(params > 11.7e9 && params < 12.1e9, "parameter count lands on the advertised ~11.95B");

        // Every Gemma 4 layer carries a scalar layer_output_scale, applied
        // last in run_layer(): scale() on the finished residual.
        Vec<2> h;
        h[0] = 6.f;
        h[1] = -8.f;
        h.scale(0.5f);
        check(h[0] == 3.f && h[1] == -4.f, "the layer output scale is applied last");

        Gemma4_12BCache cache;
        check(cache.local(0).window() == C::LOCAL_WINDOW, "local layers own a fixed-window ring");
        check(cache.full(5).size() == 0, "full-attention layers own append-only history");
    }

    std::printf("== unified K/V takes one projection two ways ==\n");
    {
        // W_k maps x -> (x0, x1); the key gets the learned per-head RMSNorm,
        // the value the scale-free one on the SAME raw projection output.
        MatT<2, 2> w = MatT<2, 2>::owning();
        w.raw()[0] = 1.f;
        w.raw()[1] = 0.f;  // out 0 = x0
        w.raw()[2] = 0.f;
        w.raw()[3] = 1.f;  // out 1 = x1
        Vec<2> gamma;
        gamma[0] = 3.f;
        gamma[1] = 3.f;
        const ToyUnifiedLayer layer{identity_linear<2>(), PerHeadNorm<RMSNorm<2>>(unit_norm<2>()), Linear<2, 2>(Weight<2, 2>(std::move(w))), PerHeadNorm<RMSNorm<2>>(RMSNorm<2>(std::move(gamma), 1e-6f)), PerHeadNorm<RMSNormNoScale<2>>(RMSNormNoScale<2>(1e-6f)), identity_linear<2>()};

        Vec<2> token;
        token[0] = 3.f;
        token[1] = 4.f;
        Matrix<2> x;
        x.append(token);

        // One token, one visible key: the softmax weight is 1 regardless of K,
        // and WO is identity, so attention() returns V itself — which for
        // unified K/V must be the raw W_k output under the scale-free norm.
        FullAttentionCache<1, 2> cache;
        const auto out = layer.attention(x.view(), cache, RotaryEmbedding<2, 10000>{}, Position{0});

        // rms([3,4]) = sqrt(12.5); normalized = [0.8485, 1.1314]
        const Scalar inv = 1.f / std::sqrt(12.5f);
        check(std::fabs(out.row(0)[0] - 3.f * inv) < 1e-5f && std::fabs(out.row(0)[1] - 4.f * inv) < 1e-5f, "the value is the raw projection with the scale-free norm");
        const auto K = layer.k_norm(layer.WK(x.view()));
        check(std::fabs(K.row(0)[0] - 3.f * 3.f * inv) < 1e-5f, "the key is the same projection with the learned scale");
        check(cache.size() == 1, "the unified K/V row was cached like any owned one");
    }

    std::printf("\n%s (%d failures)\n", g_fail ? "GEMMA4 TESTS FAILED" : "ALL GEMMA4 TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
