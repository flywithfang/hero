// gemma4.hpp — Gemma 4 shared text primitives and checkpoint configurations.
//
// This file supplies Gemma-specific policies to the uniform transformer core:
// two attention shapes, partial RoPE, KV sharing, PLE, and logit soft-capping.
// The token-mixing math itself (cache, mask, GQA reduction, rotary tables,
// per-head norm) is model-neutral and lives in attention.hpp. Vision and audio
// encoders connect at EmbeddedSequence<D> and do not enter these components.
#pragma once
#include "attention.hpp"
#include "components.hpp"
#include "transformer.hpp"

enum class GemmaAttentionKind : uint8_t { Sliding, Full };

struct Gemma4E4BTextConfig {
    static constexpr size_t V = 262144;
    static constexpr size_t D = 2560;
    static constexpr size_t L = 42;
    static constexpr size_t Hq = 8;
    static constexpr size_t Hkv = 2;
    static constexpr size_t LOCAL_HEAD_DIM = 256;
    static constexpr size_t GLOBAL_HEAD_DIM = 512;
    static constexpr size_t FF = 10240;
    static constexpr size_t PLE = 256;
    static constexpr size_t CTX = 131072;
    static constexpr size_t SLIDING_WINDOW = 512;
    static constexpr size_t KV_SHARED_LAYERS = 18;
    static constexpr size_t FIRST_KV_SHARED_LAYER = L - KV_SHARED_LAYERS;
    static constexpr size_t LOCAL_ROPE_BASE = 10000;
    static constexpr size_t GLOBAL_ROPE_BASE = 1000000;
    static constexpr Scalar RMS_EPS = 1e-6f;
    static constexpr Scalar LOGIT_SOFTCAP = 30.f;

    static constexpr GemmaAttentionKind attention_kind(size_t layer) {
        return layer % 6 == 5 ? GemmaAttentionKind::Full : GemmaAttentionKind::Sliding;
    }
    static constexpr bool shares_kv(size_t layer) {
        return layer >= FIRST_KV_SHARED_LAYER;
    }
    // The final non-sharing layer of each attention kind supplies all later
    // layers of that kind: local layer 22, full layer 23 for E4B's 42 layers.
    static constexpr bool stores_shared_kv(size_t layer) {
        if (layer >= FIRST_KV_SHARED_LAYER) return false;
        const GemmaAttentionKind kind = attention_kind(layer);
        for (size_t next = layer + 1; next < FIRST_KV_SHARED_LAYER; ++next)
            if (attention_kind(next) == kind) return false;
        return true;
    }
};

static_assert(Gemma4E4BTextConfig::attention_kind(4) == GemmaAttentionKind::Sliding);
static_assert(Gemma4E4BTextConfig::attention_kind(5) == GemmaAttentionKind::Full);
static_assert(Gemma4E4BTextConfig::FIRST_KV_SHARED_LAYER == 24);
static_assert(Gemma4E4BTextConfig::stores_shared_kv(22));
static_assert(Gemma4E4BTextConfig::stores_shared_kv(23));
static_assert(Gemma4E4BTextConfig::shares_kv(24));

// Gemma keeps Q/O and K/V as separate construction units because E4B's final
// 18 layers have Q/O weights but deliberately have no K/V tensors.
template <size_t D, size_t Hq, size_t HeadDim>
class GemmaQueryOutput {
public:
    static constexpr size_t QW = Hq * HeadDim;

    GemmaQueryOutput(Linear<D, QW> query, PerHeadNorm<Hq, HeadDim, RMSNorm<HeadDim>> query_norm,
                     Linear<QW, D> output)
        : query_(std::move(query)), query_norm_(std::move(query_norm)),
          output_(std::move(output)) {}

    Vec<QW> query(VecView<D> hidden) const { return query_norm_(query_(hidden)); }
    Vec<D> output(VecView<QW> attended) const { return output_(attended); }
    Matrix<QW> query(MatrixView<D> hidden) const {
        return query_norm_(query_(hidden));
    }
    Matrix<D> output(MatrixView<QW> attended) const {
        return output_(attended);
    }

private:
    Linear<D, QW> query_;
    PerHeadNorm<Hq, HeadDim, RMSNorm<HeadDim>> query_norm_;
    Linear<QW, D> output_;
};

// K and V come from the same input, so the pair is produced together: a layer
// with UNIFIED K/V has one projection and must not run it twice.
template <size_t KW>
struct GemmaKeyValuePair {
    Matrix<KW> key;
    Matrix<KW> value;
};

template <size_t D, size_t Hkv, size_t HeadDim>
class GemmaKeyValue {
public:
    static constexpr size_t KW = Hkv * HeadDim;

    GemmaKeyValue(Linear<D, KW> key, PerHeadNorm<Hkv, HeadDim, RMSNorm<HeadDim>> key_norm,
                  Linear<D, KW> value,
                  PerHeadNorm<Hkv, HeadDim, RMSNormNoScale<HeadDim>> value_norm)
        : key_(std::move(key)), key_norm_(std::move(key_norm)),
          value_(std::move(value)), value_norm_(std::move(value_norm)) {}

    Vec<KW> key(VecView<D> hidden) const { return key_norm_(key_(hidden)); }
    Vec<KW> value(VecView<D> hidden) const { return value_norm_(value_(hidden)); }
    Matrix<KW> key(MatrixView<D> hidden) const {
        return key_norm_(key_(hidden));
    }
    Matrix<KW> value(MatrixView<D> hidden) const {
        return value_norm_(value_(hidden));
    }
    GemmaKeyValuePair<KW> key_and_value(MatrixView<D> hidden) const {
        return {key(hidden), value(hidden)};
    }

private:
    Linear<D, KW> key_;
    PerHeadNorm<Hkv, HeadDim, RMSNorm<HeadDim>> key_norm_;
    Linear<D, KW> value_;
    PerHeadNorm<Hkv, HeadDim, RMSNormNoScale<HeadDim>> value_norm_;
};

// Unified K/V (Gemma 4 12B's global layers): the checkpoint carries no W_v, and
// the VALUE is the raw W_k output — taken before the key's learned norm and
// before RoPE — then normalized with the same scale-free RMS the ordinary value
// path uses. One projection, two different post-processings.
template <size_t D, size_t Hkv, size_t HeadDim>
class GemmaUnifiedKeyValue {
public:
    static constexpr size_t KW = Hkv * HeadDim;

    GemmaUnifiedKeyValue(Linear<D, KW> key,
                         PerHeadNorm<Hkv, HeadDim, RMSNorm<HeadDim>> key_norm,
                         PerHeadNorm<Hkv, HeadDim, RMSNormNoScale<HeadDim>> value_norm)
        : key_(std::move(key)), key_norm_(std::move(key_norm)),
          value_norm_(std::move(value_norm)) {}

    Vec<KW> key(VecView<D> hidden) const { return key_norm_(key_(hidden)); }
    Vec<KW> value(VecView<D> hidden) const { return value_norm_(key_(hidden)); }
    Matrix<KW> key(MatrixView<D> hidden) const { return key_norm_(key_(hidden)); }
    Matrix<KW> value(MatrixView<D> hidden) const { return value_norm_(key_(hidden)); }
    GemmaKeyValuePair<KW> key_and_value(MatrixView<D> hidden) const {
        Matrix<KW> projected = key_(hidden);
        return {key_norm_(projected.view()), value_norm_(projected.view())};
    }

private:
    Linear<D, KW> key_;
    PerHeadNorm<Hkv, HeadDim, RMSNorm<HeadDim>> key_norm_;
    PerHeadNorm<Hkv, HeadDim, RMSNormNoScale<HeadDim>> value_norm_;
};

// What a layer physically holds for K/V is a property of the checkpoint, not a
// runtime flag, so it selects a type. Three cases exist across Gemma 4:
//   Owned    W_k and W_v both present (E4B's first 24 layers, 12B's sliding)
//   Unified  W_k only, V = normalized raw W_k output (12B's global layers)
//   Shared   neither; the layer attends against what an earlier layer wrote
//            (E4B's last 18 layers) — so `key_value()` does not exist on it.
enum class GemmaKVKind : uint8_t { Owned, Unified, Shared };

template <size_t D, size_t Hq, size_t Hkv, size_t HeadDim, GemmaKVKind Kind>
class GemmaAttentionWeights;

template <size_t D, size_t Hq, size_t Hkv, size_t HeadDim>
class GemmaAttentionWeights<D, Hq, Hkv, HeadDim, GemmaKVKind::Owned> {
public:
    static constexpr size_t HEAD_DIM = HeadDim;
    static constexpr size_t KV_HEADS = Hkv;
    static constexpr GemmaKVKind KV_KIND = GemmaKVKind::Owned;
    static constexpr bool OWNS_KV = true;
    using QueryOutput = GemmaQueryOutput<D, Hq, HeadDim>;
    using KeyValue = GemmaKeyValue<D, Hkv, HeadDim>;

    GemmaAttentionWeights(QueryOutput query_output, KeyValue key_value)
        : query_output_(std::move(query_output)), key_value_(std::move(key_value)) {}

    const QueryOutput& query_output() const { return query_output_; }
    const KeyValue& key_value() const { return key_value_; }

private:
    QueryOutput query_output_;
    KeyValue key_value_;
};

template <size_t D, size_t Hq, size_t Hkv, size_t HeadDim>
class GemmaAttentionWeights<D, Hq, Hkv, HeadDim, GemmaKVKind::Unified> {
public:
    static constexpr size_t HEAD_DIM = HeadDim;
    static constexpr size_t KV_HEADS = Hkv;
    static constexpr GemmaKVKind KV_KIND = GemmaKVKind::Unified;
    static constexpr bool OWNS_KV = true;
    using QueryOutput = GemmaQueryOutput<D, Hq, HeadDim>;
    using KeyValue = GemmaUnifiedKeyValue<D, Hkv, HeadDim>;

    GemmaAttentionWeights(QueryOutput query_output, KeyValue key_value)
        : query_output_(std::move(query_output)), key_value_(std::move(key_value)) {}

    const QueryOutput& query_output() const { return query_output_; }
    const KeyValue& key_value() const { return key_value_; }

private:
    QueryOutput query_output_;
    KeyValue key_value_;
};

template <size_t D, size_t Hq, size_t Hkv, size_t HeadDim>
class GemmaAttentionWeights<D, Hq, Hkv, HeadDim, GemmaKVKind::Shared> {
public:
    static constexpr size_t HEAD_DIM = HeadDim;
    static constexpr size_t KV_HEADS = Hkv;
    static constexpr GemmaKVKind KV_KIND = GemmaKVKind::Shared;
    static constexpr bool OWNS_KV = false;
    using QueryOutput = GemmaQueryOutput<D, Hq, HeadDim>;

    explicit GemmaAttentionWeights(QueryOutput query_output)
        : query_output_(std::move(query_output)) {}

    const QueryOutput& query_output() const { return query_output_; }

private:
    QueryOutput query_output_;
};

// One layer's token mixer, shared by every Gemma 4 size. The variation is all
// in the template arguments: head width, KV head count, window, RoPE table, and
// whether this layer owns K/V at all.
//
//     Q = rope( rmsnorm_per_head( X W_q ) )      [T, Hq  * Dh]
//     K = rope( rmsnorm_per_head( X W_k ) )      [T, Hkv * Dh]
//     V =       rmsnorm_per_head( X W_v )        [T, Hkv * Dh]   (never rotated)
//     A = attend(Q, cache <- K,V) W_o            [T, D]
//
// Gemma's score scale stays at attention.hpp's default 1.0: the usual
// 1/sqrt(HeadDim) is folded into the learned per-head Q norm.
template <size_t D, size_t Hq, size_t Window, class Rope, class Attention>
Matrix<D> gemma_attend_layer(const Attention& attention, MatrixView<D> X,
                             KVCache<Attention::KV_HEADS, Attention::HEAD_DIM>& cache,
                             size_t first_position) {
    constexpr size_t Dh = Attention::HEAD_DIM;
    constexpr size_t Hkv = Attention::KV_HEADS;

    Matrix<Hq * Dh> Q = attention.query_output().query(X);
    rotate_heads<Hq, Dh>(Q, Rope{}, first_position);

    Matrix<Hq * Dh> A = [&] {
        if constexpr (Attention::OWNS_KV) {
            GemmaKeyValuePair<Hkv * Dh> kv =
                attention.key_value().key_and_value(X);
            rotate_heads<Hkv, Dh>(kv.key, Rope{}, first_position);
            return attend_and_cache<Hq, Hkv, Dh>(
                Q.view(), kv.key.view(), kv.value.view(), cache,
                first_position, Window);
        } else {
            return attend<Hq, Hkv, Dh>(Q.view(), cache, first_position, Window);
        }
    }();
    return attention.query_output().output(A.view());
}

template <size_t D>
constexpr Scalar gemma_embedding_scale() {
    // The reference stores the scale at the embedding weight dtype. For BF16
    // checkpoints this deliberately includes BF16 rounding.
    return bf16_to_fp32(fp32_to_bf16(Scalar(std::sqrt(double(D)))));
}

inline Scalar gemma_softcap(Scalar value, Scalar cap) {
    return cap * std::tanh(value / cap);
}

template <size_t N>
void gemma_softcap(Vec<N>& values, Scalar cap) {
    for (size_t i = 0; i < N; ++i) values[i] = gemma_softcap(values[i], cap);
}

// The per-layer part of E4B's PLE residual. Model-level token/context PLE
// construction is separate because multimodal soft tokens have no token id.
template <size_t D, size_t P>
class GemmaPerLayerResidual {
public:
    GemmaPerLayerResidual(Linear<D, P> gate, Linear<P, D> projection,
                          RMSNorm<D> post_norm)
        : gate_(std::move(gate)), projection_(std::move(projection)),
          post_norm_(std::move(post_norm)) {}

    Vec<D> operator()(VecView<D> hidden, VecView<P> per_layer_input) const {
        Vec<P> gated = gate_(hidden);
        gelu(gated);
        Vec<P> branch_input =
            hadamard(VecView<P>(gated), per_layer_input);
        Vec<D> branch = projection_(branch_input);
        branch = post_norm_(branch);
        branch += hidden;
        return branch;
    }
    Matrix<D> operator()(MatrixView<D> hidden,
                         MatrixView<P> per_layer_input) const {
        Matrix<P> gated = gate_(hidden);
        gelu_in_place(gated.mutable_view());
        Matrix<D> branch = projection_(
            hadamard(gated.view(), per_layer_input));
        branch = post_norm_(branch);
        return add(hidden, branch.view());
    }

private:
    Linear<D, P> gate_;
    Linear<P, D> projection_;
    RMSNorm<D> post_norm_;
};

// Builds E4B's PLE value for one input embedding. The packed token table and
// context projection both produce L consecutive P-vectors, normalized with a
// shared P-wide scale vector before their 1/sqrt(2) combination.
template <size_t D, size_t P, size_t L, size_t V>
class GemmaPerLayerInputs {
public:
    static constexpr size_t Packed = P * L;

    GemmaPerLayerInputs(Weight<Packed, V> token_embeddings,
                        Linear<D, Packed> model_projection,
                        PerHeadNorm<L, P, RMSNorm<P>> projection_norm,
                        Scalar token_scale, Scalar projection_scale,
                        Scalar combination_scale)
        : token_embeddings_(std::move(token_embeddings)),
          model_projection_(std::move(model_projection)),
          projection_norm_(std::move(projection_norm)),
          token_scale_(token_scale), projection_scale_(projection_scale),
          combination_scale_(combination_scale) {}

    Vec<Packed> operator()(VecView<D> input_embedding, TokenId identity) const {
        Vec<Packed> token = token_embeddings_.dequant_row(size_t(identity));
        scale_in_place(token, token_scale_);

        Vec<Packed> context = model_projection_(input_embedding);
        scale_in_place(context, projection_scale_);
        context = projection_norm_(context);

        return scaled_sum(
            VecView<Packed>(context), VecView<Packed>(token),
            combination_scale_);
    }
    Matrix<Packed> operator()(
        MatrixView<D> input_embeddings,
        std::span<const TokenId> identities) const {
        if (input_embeddings.rows() != identities.size())
            throw std::invalid_argument(
                "GemmaPerLayerInputs: embedding/identity row mismatch");

        Matrix<Packed> token =
            token_embeddings_.gather_rows(identities);
        scale_in_place(token.mutable_view(), token_scale_);

        Matrix<Packed> context = model_projection_(input_embeddings);
        scale_in_place(context.mutable_view(), projection_scale_);
        context = projection_norm_(context);

        return scaled_sum(
            context.view(), token.view(), combination_scale_);
    }

private:
    Weight<Packed, V> token_embeddings_;
    Linear<D, Packed> model_projection_;
    PerHeadNorm<L, P, RMSNorm<P>> projection_norm_;
    Scalar token_scale_;
    Scalar projection_scale_;
    Scalar combination_scale_;
};

// E4B adds a PLE residual and scalar after the canonical attention + channel
// branches.  Keeping this in the typed tail makes the general block reusable
// without pretending that PLE exists in every architecture.
template <size_t D, size_t P>
class GemmaPerLayerTail {
public:
    GemmaPerLayerTail(GemmaPerLayerResidual<D, P> per_layer_residual,
                      Scalar layer_scale)
        : per_layer_residual_(std::move(per_layer_residual)),
          layer_scale_(layer_scale) {}

    Vec<D> operator()(Vec<D> hidden, const VecView<P>& per_layer_input) const {
        Vec<D> output =
            per_layer_residual_(VecView<D>(hidden), per_layer_input);
        scale_in_place(output, layer_scale_);
        return output;
    }
    Matrix<D> operator()(Matrix<D> hidden,
                         MatrixView<P> per_layer_input) const {
        Matrix<D> output =
            per_layer_residual_(hidden.view(), per_layer_input);
        scale_in_place(output.mutable_view(), layer_scale_);
        return output;
    }

private:
    GemmaPerLayerResidual<D, P> per_layer_residual_;
    Scalar layer_scale_;
};

/*
h  = x + attn( pre_norm(x) )                 attention sub-layer (+ post-norm, Gemma style)
h  = h + mlp( pre_norm(h) )                  NORMAL SwiGLU FFN: gate/up [D→FF],
                                             GELU-family activation, down [FF→D]
p  = PLE_table[token][layer]                 ← the PLE addition
g  = act( h · W_gate )
h  = h + norm( (g ⊙ p) · W_proj )            a THIRD residual contribution
*/
// Not every Gemma 4 size has PLE: 12B sets hidden_size_per_layer_input to 0 and
// carries no per-layer embedding tensors at all. That is a different layer type,
// not a "has PLE" bool — so the tail is a template parameter and its absence is
// this type, which consumes no space and no time.
template <size_t D>
struct GemmaNoTail {
    Vec<D> operator()(Vec<D> hidden, NoLayerInput = {}) const { return hidden; }
    Matrix<D> operator()(Matrix<D> hidden, NoLayerInput = {}) const {
        return hidden;
    }
};

// Every Gemma 4 layer carries a scalar `layer_output_scale`, applied last.
// E4B folds it into the PLE tail; a size with no PLE still has the scalar, so
// it gets a tail that is only that.
template <size_t D>
class GemmaLayerScaleTail {
public:
    explicit GemmaLayerScaleTail(Scalar scale) : scale_(scale) {}

    Vec<D> operator()(Vec<D> hidden, NoLayerInput = {}) const {
        scale_in_place(hidden, scale_);
        return hidden;
    }
    Matrix<D> operator()(Matrix<D> hidden, NoLayerInput = {}) const {
        scale_in_place(hidden.mutable_view(), scale_);
        return hidden;
    }
    Scalar scale() const { return scale_; }

private:
    Scalar scale_;
};

// One concrete Gemma decoder layer. The equation intentionally lives here:
// attention communication, channel mixing, and the optional PLE tail are model
// anatomy rather than a compile-time assembly of generic branch aliases.
template <size_t D, size_t FF, class Attention, class Tail>
class GemmaDenseDecoderLayer {
public:
    GemmaDenseDecoderLayer(
        RMSNorm<D> attention_norm,
        Attention attention_weights,
        RMSNorm<D> post_attention_norm,
        RMSNorm<D> ffn_norm,
        GeluGatedMLP<D, FF> ffn_weights,
        RMSNorm<D> post_ffn_norm,
        Tail tail)
        : attention_norm_(std::move(attention_norm)),
          attention_weights_(std::move(attention_weights)),
          post_attention_norm_(std::move(post_attention_norm)),
          ffn_norm_(std::move(ffn_norm)),
          ffn_weights_(std::move(ffn_weights)),
          post_ffn_norm_(std::move(post_ffn_norm)),
          tail_(std::move(tail)) {}

    template <class TailInput, class EvaluateAttention>
    Vec<D> forward(
        VecView<D> X,
        const TailInput& tail_input,
        EvaluateAttention&& evaluate_attention) const {
        Vec<D> U = attention_norm_(X);
        Vec<D> A = std::forward<EvaluateAttention>(
            evaluate_attention)(
                attention_weights_, VecView<D>(U));
        A = post_attention_norm_(A);
        A += X;
        Vec<D> H = std::move(A);

        Vec<D> Z = ffn_norm_(H);
        Vec<D> F = ffn_weights_(Z);
        F = post_ffn_norm_(F);
        F += H;

        return tail_(std::move(F), tail_input);
    }

    template <class TailInput, class EvaluateAttention>
    Matrix<D> forward(
        MatrixView<D> X,
        const TailInput& tail_input,
        EvaluateAttention&& evaluate_attention) const {
        Matrix<D> U = attention_norm_(X);
        Matrix<D> A = std::forward<EvaluateAttention>(
            evaluate_attention)(
                attention_weights_, U.view());
        A = post_attention_norm_(A);
        Matrix<D> H = add(X, A.view());

        Matrix<D> Z = ffn_norm_(H.view());
        Matrix<D> F = ffn_weights_(Z.view());
        F = post_ffn_norm_(F);
        H = add(H.view(), F.view());

        return tail_(std::move(H), tail_input);
    }

private:
    RMSNorm<D> attention_norm_;
    Attention attention_weights_;
    RMSNorm<D> post_attention_norm_;
    RMSNorm<D> ffn_norm_;
    GeluGatedMLP<D, FF> ffn_weights_;
    RMSNorm<D> post_ffn_norm_;
    [[no_unique_address]] Tail tail_;
};

template <size_t D, size_t V>
class GemmaTokenIO {
public:
    GemmaTokenIO(Weight<D, V> tokens, Scalar input_scale)
        : tokens_(std::move(tokens)), input_scale_(input_scale) {}

    Vec<D> token(TokenId id) const {
        Vec<D> value = tokens_.dequant_row(size_t(id));
        scale_in_place(value, input_scale_);
        return value;
    }
    Matrix<D> tokens(std::span<const TokenId> ids) const {
        Matrix<D> values = tokens_.gather_rows(ids);
        scale_in_place(values.mutable_view(), input_scale_);
        return values;
    }
    Vec<V> logits(VecView<D> hidden) const { return tokens_.matvec(hidden); }

private:
    Weight<D, V> tokens_; // tied input embedding and LM head
    Scalar input_scale_;
};

using GemmaE4BLocalOwnAttention = GemmaAttentionWeights<
    Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::Hq, Gemma4E4BTextConfig::Hkv,
    Gemma4E4BTextConfig::LOCAL_HEAD_DIM, GemmaKVKind::Owned>;
using GemmaE4BLocalSharedAttention = GemmaAttentionWeights<
    Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::Hq, Gemma4E4BTextConfig::Hkv,
    Gemma4E4BTextConfig::LOCAL_HEAD_DIM, GemmaKVKind::Shared>;
using GemmaE4BGlobalOwnAttention = GemmaAttentionWeights<
    Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::Hq, Gemma4E4BTextConfig::Hkv,
    Gemma4E4BTextConfig::GLOBAL_HEAD_DIM, GemmaKVKind::Owned>;
using GemmaE4BGlobalSharedAttention = GemmaAttentionWeights<
    Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::Hq, Gemma4E4BTextConfig::Hkv,
    Gemma4E4BTextConfig::GLOBAL_HEAD_DIM, GemmaKVKind::Shared>;

template <class Attention>
using GemmaE4BDenseLayer = GemmaDenseDecoderLayer<
    Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::FF, Attention,
    GemmaPerLayerTail<Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::PLE>>;

using GemmaE4BLayer = std::variant<
    GemmaE4BDenseLayer<GemmaE4BLocalOwnAttention>,
    GemmaE4BDenseLayer<GemmaE4BGlobalOwnAttention>,
    GemmaE4BDenseLayer<GemmaE4BLocalSharedAttention>,
    GemmaE4BDenseLayer<GemmaE4BGlobalSharedAttention>>;

struct GemmaE4BLayerSchedule {
    static void validate(const GemmaE4BLayer& layer, size_t index) {
        using C = Gemma4E4BTextConfig;
        const bool full = C::attention_kind(index) == GemmaAttentionKind::Full;
        const bool shared = C::shares_kv(index);
        const size_t expected = !full && !shared ? 0 : full && !shared ? 1 : !full ? 2 : 3;
        if (layer.index() != expected)
            throw std::invalid_argument(
                "Gemma4E4B: layer variant disagrees with attention/KV schedule");
    }
};

using GemmaE4BModelData = GemmaPerLayerInputs<
    Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::PLE,
    Gemma4E4BTextConfig::L, Gemma4E4BTextConfig::V>;

using Gemma4E4BTextWeights = TransformerWeights<
    Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::V, Gemma4E4BTextConfig::L,
    GemmaTokenIO<Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::V>,
    GemmaE4BLayer, RMSNorm<Gemma4E4BTextConfig::D>, GemmaE4BModelData,
    GemmaE4BLayerSchedule>;

class Gemma4E4BCache {
    using C = Gemma4E4BTextConfig;
    using Local = KVCache<C::Hkv, C::LOCAL_HEAD_DIM>;
    using Global = KVCache<C::Hkv, C::GLOBAL_HEAD_DIM>;
    using Entry = std::variant<std::monostate, Local, Global>;

public:
    Gemma4E4BCache() {
        for (size_t i = 0; i < C::L; ++i) {
            if (C::shares_kv(i)) continue;
            if (C::attention_kind(i) == GemmaAttentionKind::Full)
                entries_[i].template emplace<Global>(0);
            else
                entries_[i].template emplace<Local>(C::stores_shared_kv(i) ? 0 : C::SLIDING_WINDOW);
        }
    }

    size_t tokens() const { return tokens_; }
    void advance(size_t count) {
        if (count > C::CTX - tokens_)
            throw std::length_error("Gemma4E4BCache: context exhausted");
        tokens_ += count;
    }

    Local& local(size_t layer) {
        return std::get<Local>(entries_[C::shares_kv(layer) ? 22 : layer]);
    }
    const Local& local(size_t layer) const {
        return std::get<Local>(entries_[C::shares_kv(layer) ? 22 : layer]);
    }
    Global& global(size_t layer) {
        return std::get<Global>(entries_[C::shares_kv(layer) ? 23 : layer]);
    }
    const Global& global(size_t layer) const {
        return std::get<Global>(entries_[C::shares_kv(layer) ? 23 : layer]);
    }

private:
    std::array<Entry, C::L> entries_;
    size_t tokens_ = 0;
};

struct Gemma4E4BArchitecture {
    using C = Gemma4E4BTextConfig;
    static constexpr size_t D = C::D;
    static constexpr size_t V = C::V;
    static constexpr size_t L = C::L;
    static constexpr size_t CTX = C::CTX;
    using Weights = Gemma4E4BTextWeights;
    using PrefixState = Gemma4E4BCache;
    using PreparedInput = Matrix<C::PLE * C::L>;
    // Local heads rotate every plane; global heads rotate only the first
    // quarter of theirs (partial RoPE), which is what lets a 512-wide global
    // head stay stable at 128K context.
    using LocalRope = RotaryEmbedding<C::LOCAL_HEAD_DIM, C::LOCAL_ROPE_BASE>;
    using GlobalRope = RotaryEmbedding<C::GLOBAL_HEAD_DIM, C::GLOBAL_ROPE_BASE, 1, 4>;

    static PrefixState make_prefix_state() { return {}; }
    static size_t prefix_tokens(
        const PrefixState& prefix_state) {
        return prefix_state.tokens();
    }
    static void advance_prefix(
        PrefixState& prefix_state, size_t count) {
        prefix_state.advance(count);
    }

    static EmbeddedSequence<D> embed(const Weights& weights,
                                     std::span<const TokenId> tokens,
                                     size_t first_position) {
        return embed_text_tokens<D>(tokens, first_position,
            [&](std::span<const TokenId> batch) {
                return weights.token_io().tokens(batch);
            });
    }

    static PreparedInput prepare(const Weights& weights,
                                 const EmbeddedSequence<D>& input) {
        return weights.architecture_data()(
            input.matrix(), input.ple_token_identities());
    }

    static void forward_layer(const Weights& weights,
                              PrefixState& prefix_state,
                              const EmbeddedSequence<D>& input,
                              ResidualStream<D>& residual,
                              PreparedInput& per_layer, size_t layer_index) {
        const auto& layer_variant =
            weights.layer(layer_index);
        Matrix<C::PLE> ple = slice_columns<C::PLE>(
            per_layer.view(), layer_index * C::PLE);
        const size_t first_position = input.position(0).i;
        Matrix<D> next = std::visit([&](const auto& layer) {
            return layer.forward(
                residual.matrix(), ple.view(),
                [&](const auto& attention,
                    MatrixView<D> normalized) {
                    using Attention =
                        std::decay_t<decltype(attention)>;
                    return mix_tokens<shape_of<Attention>>(
                        attention, normalized, prefix_state,
                        layer_index, first_position);
                });
        }, layer_variant);
        residual.set_matrix(std::move(next));
    }

    static Vec<V> output(
        const Weights& weights,
        const ResidualStream<D>& residual) {
        Vec<D> last = weights.final_norm()(
            residual.token(residual.tokens() - 1));
        Vec<V> logits = weights.token_io().logits(last);
        gemma_softcap(logits, C::LOGIT_SOFTCAP);
        return logits;
    }

private:
    // E4B alternates between exactly two attention shapes on a period-6
    // schedule. A shape bundles everything the checkpoint varies: how wide a
    // head is, which RoPE table rotates it, how far back it may look, and
    // which of the two caches it reads. Nothing else about attention differs
    // between a sliding layer and a full one.
    struct SlidingShape {
        static constexpr size_t HEAD_DIM = C::LOCAL_HEAD_DIM;
        static constexpr size_t WINDOW = C::SLIDING_WINDOW;
        using Rope = LocalRope;
        static auto& cache(PrefixState& prefix_state, size_t layer) {
            return prefix_state.local(layer);
        }
    };
    struct FullShape {
        static constexpr size_t HEAD_DIM = C::GLOBAL_HEAD_DIM;
        static constexpr size_t WINDOW = 0; // 0 == look back to position 0
        using Rope = GlobalRope;
        static auto& cache(PrefixState& prefix_state, size_t layer) {
            return prefix_state.global(layer);
        }
    };

    // Head width is what physically distinguishes the two shapes, so it is
    // what selects one. The static_assert makes that a compile error rather
    // than a silent mis-selection if a future config makes them equal.
    static_assert(C::LOCAL_HEAD_DIM != C::GLOBAL_HEAD_DIM,
                  "attention shape is selected by head width");
    template <class Attention>
    using shape_of = std::conditional_t<
        Attention::HEAD_DIM == C::LOCAL_HEAD_DIM, SlidingShape, FullShape>;

    // Two things vary per layer, and each is resolved at compile time from a
    // type: the Shape (head width, RoPE table, window, which cache) and the
    // KV kind. Layers 24..41 own no K/V: they contribute no rows and attend
    // against what layer 22 (sliding) or 23 (full) wrote earlier in this same
    // pass, which is why cache() redirects rather than the layer holding a
    // cache of its own. The reduction is gemma_attend_layer's.
    template <class Shape, class Attention>
    static Matrix<D> mix_tokens(const Attention& attention, MatrixView<D> X,
                                PrefixState& prefix_state, size_t layer,
                                size_t first_position) {
        return gemma_attend_layer<D, C::Hq, Shape::WINDOW,
                                  typename Shape::Rope>(
            attention, X, Shape::cache(prefix_state, layer), first_position);
    }
};

using Gemma4E4BTransformer =
    Transformer<Gemma4E4BArchitecture>;

// ============================ Gemma 4 12B Unified ============================
//
// Pinned from google/gemma-4-12B config.json (`gemma4_unified_text`), not from
// a model card. It is a SIMPLER decoder than E4B — no PLE, no shared-KV layers
// — but it varies two things E4B holds constant, and both become types rather
// than flags:
//   * KV head count differs per attention kind: 8 sliding, 1 global.
//   * Global layers have UNIFIED K/V: one W_k and no W_v, so the value is the
//     raw projection output with the scale-free norm applied.
struct Gemma4_12BTextConfig {
    static constexpr size_t V = 262144;
    static constexpr size_t D = 3840;
    static constexpr size_t L = 48;
    static constexpr size_t Hq = 16;
    static constexpr size_t LOCAL_HEAD_DIM = 256;
    static constexpr size_t GLOBAL_HEAD_DIM = 512;
    static constexpr size_t LOCAL_HKV = 8;
    static constexpr size_t GLOBAL_HKV = 1;    // num_global_key_value_heads
    static constexpr size_t FF = 15360;
    static constexpr size_t CTX = 262144;
    static constexpr size_t SLIDING_WINDOW = 1024;
    static constexpr size_t LOCAL_ROPE_BASE = 10000;
    static constexpr size_t GLOBAL_ROPE_BASE = 1000000;
    static constexpr Scalar RMS_EPS = 1e-6f;
    static constexpr Scalar LOGIT_SOFTCAP = 30.f;

    static constexpr GemmaAttentionKind attention_kind(size_t layer) {
        return layer % 6 == 5 ? GemmaAttentionKind::Full
                              : GemmaAttentionKind::Sliding;
    }
    static constexpr bool shares_kv(size_t) { return false; }  // 0 shared layers
    static constexpr GemmaKVKind kv_kind(size_t layer) {
        return attention_kind(layer) == GemmaAttentionKind::Full
            ? GemmaKVKind::Unified : GemmaKVKind::Owned;
    }
    static constexpr size_t kv_heads(size_t layer) {
        return attention_kind(layer) == GemmaAttentionKind::Full
            ? GLOBAL_HKV : LOCAL_HKV;
    }
};

// The model card requires the FINAL layer to be global. The period-6 rule
// delivers that only because 48 is a multiple of 6, so assert it rather than
// trusting the arithmetic to stay true for some future size.
static_assert(Gemma4_12BTextConfig::attention_kind(
                  Gemma4_12BTextConfig::L - 1) == GemmaAttentionKind::Full,
              "Gemma 4 12B's last layer must be a global-attention layer");
static_assert(Gemma4_12BTextConfig::attention_kind(0) ==
              GemmaAttentionKind::Sliding);
static_assert(Gemma4_12BTextConfig::kv_kind(5) == GemmaKVKind::Unified);
static_assert(Gemma4_12BTextConfig::kv_kind(4) == GemmaKVKind::Owned);

// params = tied embeddings + per-layer (W_q + W_k [+ W_v] + W_o + 3 FFN mats).
// Norms are excluded; they are ~0.05% and would only blur the check.
template <class C>
constexpr size_t gemma_dense_param_count() {
    size_t total = size_t(C::V) * C::D;
    for (size_t layer = 0; layer < C::L; ++layer) {
        const bool full = C::attention_kind(layer) == GemmaAttentionKind::Full;
        const size_t dh = full ? C::GLOBAL_HEAD_DIM : C::LOCAL_HEAD_DIM;
        const size_t kv_mats = C::kv_kind(layer) == GemmaKVKind::Unified ? 1 : 2;
        total += C::D * C::Hq * dh;                         // W_q
        total += kv_mats * C::D * C::kv_heads(layer) * dh;  // W_k (+ W_v)
        total += C::Hq * dh * C::D;                         // W_o
        total += 3 * C::D * C::FF;                          // gate / up / down
    }
    return total;
}
static_assert(gemma_dense_param_count<Gemma4_12BTextConfig>() > 11'700'000'000 &&
              gemma_dense_param_count<Gemma4_12BTextConfig>() < 12'100'000'000,
              "Gemma 4 12B lands on its advertised ~11.95B");

using Gemma12BSlidingAttention = GemmaAttentionWeights<
    Gemma4_12BTextConfig::D, Gemma4_12BTextConfig::Hq,
    Gemma4_12BTextConfig::LOCAL_HKV, Gemma4_12BTextConfig::LOCAL_HEAD_DIM,
    GemmaKVKind::Owned>;
using Gemma12BGlobalAttention = GemmaAttentionWeights<
    Gemma4_12BTextConfig::D, Gemma4_12BTextConfig::Hq,
    Gemma4_12BTextConfig::GLOBAL_HKV, Gemma4_12BTextConfig::GLOBAL_HEAD_DIM,
    GemmaKVKind::Unified>;

template <class Attention>
using Gemma12BDenseLayer = GemmaDenseDecoderLayer<
    Gemma4_12BTextConfig::D, Gemma4_12BTextConfig::FF, Attention,
    GemmaLayerScaleTail<Gemma4_12BTextConfig::D>>;

using Gemma12BLayer = std::variant<Gemma12BDenseLayer<Gemma12BSlidingAttention>,
                                   Gemma12BDenseLayer<Gemma12BGlobalAttention>>;

struct Gemma12BLayerSchedule {
    static void validate(const Gemma12BLayer& layer, size_t index) {
        using C = Gemma4_12BTextConfig;
        const size_t expected =
            C::attention_kind(index) == GemmaAttentionKind::Full ? 1u : 0u;
        if (layer.index() != expected)
            throw std::invalid_argument(
                "Gemma4_12B: layer variant disagrees with the attention schedule");
    }
};

using Gemma4_12BTextWeights = TransformerWeights<
    Gemma4_12BTextConfig::D, Gemma4_12BTextConfig::V, Gemma4_12BTextConfig::L,
    GemmaTokenIO<Gemma4_12BTextConfig::D, Gemma4_12BTextConfig::V>,
    Gemma12BLayer, RMSNorm<Gemma4_12BTextConfig::D>, NoArchitectureData,
    Gemma12BLayerSchedule>;

// Every layer owns its cache; the two kinds differ in head count, head width,
// and whether they are windowed. Sliding layers keep a ring of SLIDING_WINDOW
// rows, so their cost stops growing with T.
class Gemma4_12BCache {
    using C = Gemma4_12BTextConfig;
    using Local = KVCache<C::LOCAL_HKV, C::LOCAL_HEAD_DIM>;
    using Global = KVCache<C::GLOBAL_HKV, C::GLOBAL_HEAD_DIM>;
    using Entry = std::variant<Local, Global>;

public:
    Gemma4_12BCache() {
        for (size_t i = 0; i < C::L; ++i) {
            if (C::attention_kind(i) == GemmaAttentionKind::Full)
                entries_[i].template emplace<Global>(0);
            else
                entries_[i].template emplace<Local>(C::SLIDING_WINDOW);
        }
    }

    size_t tokens() const { return tokens_; }
    void advance(size_t count) {
        if (count > C::CTX - tokens_)
            throw std::length_error("Gemma4_12BCache: context exhausted");
        tokens_ += count;
    }

    Local& local(size_t layer) { return std::get<Local>(entries_[layer]); }
    Global& global(size_t layer) { return std::get<Global>(entries_[layer]); }

private:
    std::array<Entry, C::L> entries_;
    size_t tokens_ = 0;
};

struct Gemma4_12BArchitecture {
    using C = Gemma4_12BTextConfig;
    static constexpr size_t D = C::D;
    static constexpr size_t V = C::V;
    static constexpr size_t L = C::L;
    static constexpr size_t CTX = C::CTX;
    using Weights = Gemma4_12BTextWeights;
    using PrefixState = Gemma4_12BCache;
    using PreparedInput = NoPreparedInput;   // no PLE to prepare
    using LocalRope = RotaryEmbedding<C::LOCAL_HEAD_DIM, C::LOCAL_ROPE_BASE>;
    using GlobalRope =
        RotaryEmbedding<C::GLOBAL_HEAD_DIM, C::GLOBAL_ROPE_BASE, 1, 4>;

    static PrefixState make_prefix_state() { return {}; }
    static size_t prefix_tokens(const PrefixState& state) { return state.tokens(); }
    static void advance_prefix(PrefixState& state, size_t count) {
        state.advance(count);
    }

    static EmbeddedSequence<D> embed(const Weights& weights,
                                     std::span<const TokenId> tokens,
                                     size_t first_position) {
        return embed_text_tokens<D>(tokens, first_position,
            [&](std::span<const TokenId> batch) {
                return weights.token_io().tokens(batch);
            });
    }

    static PreparedInput prepare(const Weights&, const EmbeddedSequence<D>&) {
        return {};
    }

    static void forward_layer(const Weights& weights, PrefixState& state,
                              const EmbeddedSequence<D>& input,
                              ResidualStream<D>& residual, PreparedInput&,
                              size_t layer_index) {
        const size_t first_position = input.position(0).i;
        Matrix<D> next = std::visit([&](const auto& layer) {
            return layer.forward(
                residual.matrix(), NoLayerInput{},
                [&](const auto& attention, MatrixView<D> normalized) {
                    using Attention = std::decay_t<decltype(attention)>;
                    if constexpr (Attention::HEAD_DIM == C::LOCAL_HEAD_DIM)
                        return gemma_attend_layer<D, C::Hq, C::SLIDING_WINDOW,
                                                  LocalRope>(
                            attention, normalized, state.local(layer_index),
                            first_position);
                    else
                        return gemma_attend_layer<D, C::Hq, /*window=*/0,
                                                  GlobalRope>(
                            attention, normalized, state.global(layer_index),
                            first_position);
                });
        }, weights.layer(layer_index));
        residual.set_matrix(std::move(next));
    }

    static Vec<V> output(const Weights& weights,
                         const ResidualStream<D>& residual) {
        Vec<D> last = weights.final_norm()(
            residual.token(residual.tokens() - 1));
        Vec<V> logits = weights.token_io().logits(last);
        gemma_softcap(logits, C::LOGIT_SOFTCAP);
        return logits;
    }
};

using Gemma4_12BTransformer = Transformer<Gemma4_12BArchitecture>;
