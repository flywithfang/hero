// qwen35.hpp — Qwen 3.5 specifications for the uniform transformer core.
//
// Qwen 3.5 is the first HYBRID stack this engine runs, and it is the reason
// `transformer.hpp` talks about a "token mixer" rather than "attention": three
// of every four layers mix tokens with a gated delta network carrying a
// fixed-size state, and every fourth mixes them with gated full attention over
// a K/V cache. Everything else about the two layer kinds is identical —
// pre-norm residual, SwiGLU channel mixer, same residual stream — so both bind
// the SAME canonical TransformerBlock and differ only in the mixer type.
//
// The math lives in attention.hpp and recurrent.hpp. This file is dimensions,
// weight bundles, the layer schedule, and the architecture policy.
#pragma once
#include "attention.hpp"
#include "components.hpp"
#include "recurrent.hpp"
#include "transformer.hpp"
#include <array>
#include <variant>

// ---- configurations ---------------------------------------------------------
//
// Pinned from the reference config.json, not from a model card. Note
// Hq*HEAD_DIM != D on the 4B (16*256 = 4096 vs D = 2560): the four attention
// LAWS never assumed otherwise.
struct Qwen35_4BConfig {
    static constexpr size_t V = 248320;
    static constexpr size_t D = 2560;
    static constexpr size_t L = 32;
    static constexpr size_t CTX = 262144;

    // full-attention layers
    static constexpr size_t Hq = 16;
    static constexpr size_t Hkv = 4;
    static constexpr size_t HEAD_DIM = 256;
    static constexpr size_t ROPE_BASE = 10000000;
    static constexpr size_t ROTARY_NUM = 1;  // partial_rotary_factor 0.25
    static constexpr size_t ROTARY_DEN = 4;

    // linear-attention (gated delta net) layers
    static constexpr size_t KEY_HEAD_DIM = 128;
    static constexpr size_t VALUE_HEAD_DIM = 128;
    static constexpr size_t KEY_HEADS = 16;
    static constexpr size_t VALUE_HEADS = 32;
    static constexpr size_t CONV_WIDTH = 4;

    static constexpr size_t FF = 9216;  // SwiGLU
    static constexpr Scalar RMS_EPS = 1e-6f;
    static constexpr size_t FULL_ATTENTION_INTERVAL = 4;

    using OutputWeight = std::monostate;  // tie_word_embeddings: true

    // Three linear layers, then one full-attention layer, repeating. Writing
    // it as the reference does keeps the last layer full attention for any L
    // that is a multiple of the interval.
    static constexpr bool is_recurrent(size_t layer) { return (layer + 1) % FULL_ATTENTION_INTERVAL != 0; }
};

// 9B shares the 4B's depth and every attention/linear head shape; it is wider,
// has a wider FFN, and does not tie its embeddings. Spelled out rather than
// inherited so a config is always readable as one block of numbers.
struct Qwen35_9BConfig {
    static constexpr size_t V = 248320;
    static constexpr size_t D = 4096;
    static constexpr size_t L = 32;
    static constexpr size_t CTX = 262144;

    static constexpr size_t Hq = 16;
    static constexpr size_t Hkv = 4;
    static constexpr size_t HEAD_DIM = 256;
    static constexpr size_t ROPE_BASE = 10000000;
    static constexpr size_t ROTARY_NUM = 1;
    static constexpr size_t ROTARY_DEN = 4;

    static constexpr size_t KEY_HEAD_DIM = 128;
    static constexpr size_t VALUE_HEAD_DIM = 128;
    static constexpr size_t KEY_HEADS = 16;
    static constexpr size_t VALUE_HEADS = 32;
    static constexpr size_t CONV_WIDTH = 4;

    static constexpr size_t FF = 12288;
    static constexpr Scalar RMS_EPS = 1e-6f;
    static constexpr size_t FULL_ATTENTION_INTERVAL = 4;

    using OutputWeight = Weight<D, V>;  // tie_word_embeddings: false

    static constexpr bool is_recurrent(size_t layer) { return (layer + 1) % FULL_ATTENTION_INTERVAL != 0; }
};

// Parameter counts, as the plan's formula predicts them. Attention layers pay
// for a DOUBLE-width query projection because the gate rides along with it.
template <class C>
constexpr size_t qwen35_param_count() {
    constexpr size_t attention_layers = C::L / C::FULL_ATTENTION_INTERVAL;
    constexpr size_t recurrent_layers = C::L - attention_layers;

    constexpr size_t query_gate = C::D * 2 * C::Hq * C::HEAD_DIM;
    constexpr size_t kv = 2 * C::D * C::Hkv * C::HEAD_DIM;
    constexpr size_t out = C::Hq * C::HEAD_DIM * C::D;
    constexpr size_t attention = query_gate + kv + out;

    constexpr size_t key_width = C::KEY_HEAD_DIM * C::KEY_HEADS;
    constexpr size_t value_width = C::VALUE_HEAD_DIM * C::VALUE_HEADS;
    constexpr size_t recurrent = C::D * (2 * key_width + value_width) + C::D * value_width + value_width * C::D + 2 * C::D * C::VALUE_HEADS + C::CONV_WIDTH * (2 * key_width + value_width);

    constexpr size_t ff = 3 * C::D * C::FF;
    constexpr bool tied = std::is_same_v<typename C::OutputWeight, std::monostate>;
    constexpr size_t embed = C::V * C::D * (tied ? 1 : 2);
    return embed + attention_layers * (attention + ff) + recurrent_layers * (recurrent + ff);
}

static_assert(qwen35_param_count<Qwen35_4BConfig>() > 3'700'000'000 && qwen35_param_count<Qwen35_4BConfig>() < 4'400'000'000, "Qwen 3.5 4B lands on its advertised size");
static_assert(qwen35_param_count<Qwen35_9BConfig>() > 8'400'000'000 && qwen35_param_count<Qwen35_9BConfig>() < 9'800'000'000, "Qwen 3.5 9B lands on its advertised size");

static_assert(Qwen35_4BConfig::is_recurrent(0));
static_assert(Qwen35_4BConfig::is_recurrent(2));
static_assert(!Qwen35_4BConfig::is_recurrent(3));
static_assert(!Qwen35_4BConfig::is_recurrent(Qwen35_4BConfig::L - 1), "the final layer must be full attention");

// ---- token mixer 1: gated attention -----------------------------------------
//
// One projection emits query and gate together, head-major. After attending,
// the result is multiplied by sigmoid(gate) before the output projection —
// a learned per-channel decision about how much of the mixed value to keep.
template <size_t D, size_t Hq, size_t Hkv, size_t HeadDim>
class QwenGatedAttention {
public:
    static constexpr size_t QW = Hq * HeadDim;
    static constexpr size_t KW = Hkv * HeadDim;
    static constexpr size_t GATED_QW = 2 * QW;

    QwenGatedAttention(Linear<D, GATED_QW> query_gate, PerHeadNorm<Hq, HeadDim, RMSNorm<HeadDim>> query_norm, Linear<D, KW> key, PerHeadNorm<Hkv, HeadDim, RMSNorm<HeadDim>> key_norm, Linear<D, KW> value, Linear<QW, D> output) : query_gate_(std::move(query_gate)), query_norm_(std::move(query_norm)), key_(std::move(key)), key_norm_(std::move(key_norm)), value_(std::move(value)), output_(std::move(output)) {}

    // .first is the normalized query, .second the raw gate.
    HeadPair<Hq, HeadDim> query_and_gate(MatrixView<D> hidden) const {
        HeadPair<Hq, HeadDim> split = split_head_pairs<Hq, HeadDim>(query_gate_(hidden).view());
        split.first = query_norm_(split.first.view());
        return split;
    }
    Matrix<KW> key(MatrixView<D> hidden) const { return key_norm_(key_(hidden).view()); }
    Matrix<KW> value(MatrixView<D> hidden) const { return value_(hidden); }
    Matrix<D> output(MatrixView<QW> attended) const { return output_(attended); }

private:
    Linear<D, GATED_QW> query_gate_;
    PerHeadNorm<Hq, HeadDim, RMSNorm<HeadDim>> query_norm_;
    Linear<D, KW> key_;
    PerHeadNorm<Hkv, HeadDim, RMSNorm<HeadDim>> key_norm_;
    Linear<D, KW> value_;  // never normalized, never rotated
    Linear<QW, D> output_;
};

// ---- token mixer 2: gated delta network -------------------------------------
//
// The packed projection emits [q | k | v] over ConvChannels, which a depthwise
// causal conv1d and a SiLU pass through before the delta rule sees them. beta
// and the decay gate are separate per-value-head projections of the same input.
template <size_t D, size_t Dk, size_t Dv, size_t Hk, size_t Hv, size_t ConvWidth>
class QwenGatedDeltaNet {
public:
    static_assert(Hv % Hk == 0, "value heads must be a whole multiple of key heads");
    static constexpr size_t KEY_WIDTH = Dk * Hk;
    static constexpr size_t VALUE_WIDTH = Dv * Hv;
    static constexpr size_t CONV_CHANNELS = 2 * KEY_WIDTH + VALUE_WIDTH;

    QwenGatedDeltaNet(Linear<D, CONV_CHANNELS> qkv, Linear<D, VALUE_WIDTH> gate, Matrix<ConvWidth> conv, Linear<D, Hv> beta, Linear<D, Hv> alpha, Vec<Hv> dt_bias, Vec<Hv> decay_scale, PerHeadNorm<Hv, Dv, RMSNorm<Dv>> output_norm, Linear<VALUE_WIDTH, D> output) : qkv_(std::move(qkv)), gate_(std::move(gate)), conv_(std::move(conv)), beta_(std::move(beta)), alpha_(std::move(alpha)), dt_bias_(std::move(dt_bias)), decay_scale_(std::move(decay_scale)), output_norm_(std::move(output_norm)), output_(std::move(output)) {
        if (conv_.rows() != CONV_CHANNELS) throw std::invalid_argument("QwenGatedDeltaNet: conv channel mismatch");
        // The decay scale is -exp(A_log), so it is negative by construction and
        // exp(gate) is a contraction. A positive value would make the state
        // grow with every token and the logits diverge a few layers later,
        // which is a very hard failure to read backwards from. Reject it here.
        for (size_t h = 0; h < Hv; ++h)
            if (!(decay_scale_[h] <= 0.f)) throw std::invalid_argument("QwenGatedDeltaNet: decay scale must be <= 0 (-exp(A_log))");
    }

    Matrix<CONV_CHANNELS> qkv(MatrixView<D> hidden) const { return qkv_(hidden); }
    Matrix<VALUE_WIDTH> gate(MatrixView<D> hidden) const { return gate_(hidden); }
    const Matrix<ConvWidth>& conv() const { return conv_; }
    Matrix<D> output(MatrixView<VALUE_WIDTH> mixed) const { return output_(mixed); }
    Matrix<VALUE_WIDTH> normalize_heads(MatrixView<VALUE_WIDTH> mixed) const { return output_norm_(mixed); }

    // Per value head: beta in (0,1) is the write strength; the decay gate is
    // decay_scale * softplus(alpha + dt_bias) and is <= 0 because the converter
    // stores decay_scale = -exp(A_log). exp() of it is therefore a contraction.
    Matrix<Hv> write_strength(MatrixView<D> hidden) const {
        Matrix<Hv> beta = beta_(hidden);
        for (size_t i = 0; i < beta.rows() * Hv; ++i) beta.data()[i] = sigmoid(beta.data()[i]);
        return beta;
    }
    Matrix<Hv> decay_gate(MatrixView<D> hidden) const {
        Matrix<Hv> alpha = alpha_(hidden);
        for (size_t row = 0; row < alpha.rows(); ++row)
            for (size_t h = 0; h < Hv; ++h) {
                const Scalar raw = alpha.row(row)[h];
                alpha.row_mut(row)[h] = decay_scale_[h] * softplus(raw + dt_bias_[h]);
            }
        return alpha;
    }

    // The one head-mapping decision, in one place. ggml TILES the key heads
    // across the value heads (head h reads key head h % Hk) rather than
    // grouping them the way GQA does (h / (Hv/Hk)).
    //
    // VERIFIED against the real Qwen 3.5 4B Q4_K_M checkpoint by flipping it:
    // tiling gives "The capital of France is" -> " Paris" (logprob -0.66) and
    // coherent text; grouping gives " a" (-2.05) and gibberish. Both mappings
    // are shape-legal, so only weights could decide it.
    static constexpr size_t key_head_of(size_t value_head) { return value_head % Hk; }

private:
    Linear<D, CONV_CHANNELS> qkv_;
    Linear<D, VALUE_WIDTH> gate_;  // "z", the output gate
    Matrix<ConvWidth> conv_;       // one row of taps per channel
    Linear<D, Hv> beta_;
    Linear<D, Hv> alpha_;
    Vec<Hv> dt_bias_;
    Vec<Hv> decay_scale_;
    PerHeadNorm<Hv, Dv, RMSNorm<Dv>> output_norm_;
    Linear<VALUE_WIDTH, D> output_;
};

// ---- token I/O --------------------------------------------------------------

template <class C>
class QwenTokenIO {
public:
    using TokenWeight = Weight<C::D, C::V>;
    using OutputWeight = typename C::OutputWeight;

    QwenTokenIO(TokenWeight tokens, OutputWeight unembed) : tokens_(std::move(tokens)), unembed_(std::move(unembed)) {}

    Vec<C::D> token(TokenId id) const { return tokens_.dequant_row(size_t(id)); }
    Matrix<C::D> tokens(std::span<const TokenId> ids) const { return tokens_.gather_rows(ids); }
    Vec<C::V> logits(VecView<C::D> hidden) const {
        if constexpr (std::is_same_v<OutputWeight, std::monostate>)
            return tokens_.matvec(hidden);
        else
            return unembed_.matvec(hidden);
    }

private:
    TokenWeight tokens_;
    [[no_unique_address]] OutputWeight unembed_;
};

// ---- layers -----------------------------------------------------------------
//
// One layer's weights. Both kinds are plain pre-norm: two norms, a token mixer,
// a channel mixer, no post-norm and no tail. That is the entire topology, so it
// is four members rather than a composition of branch types. The tensor named
// `attn_post_norm` in the checkpoint is, despite its name, the channel mixer's
// INPUT norm.
//
// There is no forward() here: a layer cannot run itself, because evaluating the
// mixer needs this layer's cache or recurrent state, which belong to the
// PrefixState and not to any weight. The equation is written out in
// Qwen35Architecture::forward_layer, where the mixer call can just be a call.
template <class C, class Mixer>
struct Qwen35Block {
    RMSNorm<C::D> mixer_norm;
    Mixer mixer;
    RMSNorm<C::D> channel_norm;
    GatedMLP<C::D, C::FF> channel;
};

template <class C>
using QwenAttentionMixer = QwenGatedAttention<C::D, C::Hq, C::Hkv, C::HEAD_DIM>;

template <class C>
using QwenRecurrentMixer = QwenGatedDeltaNet<C::D, C::KEY_HEAD_DIM, C::VALUE_HEAD_DIM, C::KEY_HEADS, C::VALUE_HEADS, C::CONV_WIDTH>;

template <class C>
using Qwen35Layer = std::variant<Qwen35Block<C, QwenRecurrentMixer<C>>, Qwen35Block<C, QwenAttentionMixer<C>>>;

template <class C>
struct Qwen35LayerSchedule {
    static void validate(const Qwen35Layer<C>& layer, size_t index) {
        if (layer.index() != (C::is_recurrent(index) ? 0u : 1u)) throw std::invalid_argument("Qwen35: layer variant disagrees with the hybrid schedule");
    }
};

// ---- prefix state -----------------------------------------------------------
//
// This is where the hybrid stack shows its economics. A full-attention layer
// carries a K/V cache that grows with T; a linear layer carries a conv history
// and a state matrix whose size does not depend on T at all.
template <class C>
class Qwen35State {
public:
    using AttentionEntry = KVCache<C::Hkv, C::HEAD_DIM>;
    struct RecurrentEntry {
        CausalConv1dState<QwenRecurrentMixer<C>::CONV_CHANNELS, C::CONV_WIDTH> conv;
        DeltaNetState<C::VALUE_HEADS, C::KEY_HEAD_DIM, C::VALUE_HEAD_DIM> delta;
    };

    Qwen35State() {
        for (size_t layer = 0; layer < C::L; ++layer) {
            if (C::is_recurrent(layer))
                entries_[layer].template emplace<RecurrentEntry>();
            else
                entries_[layer].template emplace<AttentionEntry>();
        }
    }

    size_t tokens() const { return tokens_; }
    void advance(size_t count) {
        if (count > C::CTX - tokens_) throw std::length_error("Qwen35State: context exhausted");
        tokens_ += count;
    }

    AttentionEntry& attention(size_t layer) { return std::get<AttentionEntry>(entries_[layer]); }
    RecurrentEntry& recurrent(size_t layer) { return std::get<RecurrentEntry>(entries_[layer]); }

    // Bytes of carried state per token, for the two layer kinds. The whole
    // point of the hybrid is that only the second term is [GROWS-T].
    static constexpr size_t recurrent_floats_per_layer() { return DeltaNetState<C::VALUE_HEADS, C::KEY_HEAD_DIM, C::VALUE_HEAD_DIM>::floats; }
    static constexpr size_t attention_floats_per_token_per_layer() { return 2 * C::Hkv * C::HEAD_DIM; }

private:
    std::array<std::variant<AttentionEntry, RecurrentEntry>, C::L> entries_;
    size_t tokens_ = 0;
};

// ---- architecture -----------------------------------------------------------

template <class C>
struct Qwen35Architecture {
    static constexpr size_t D = C::D;
    static constexpr size_t V = C::V;
    static constexpr size_t L = C::L;
    static constexpr size_t CTX = C::CTX;

    using Weights = TransformerWeights<D, V, L, QwenTokenIO<C>, Qwen35Layer<C>, RMSNorm<D>, NoArchitectureData, Qwen35LayerSchedule<C>>;
    using PrefixState = Qwen35State<C>;
    using PreparedInput = NoPreparedInput;

    // Text positions make every MRoPE section carry the same value, so
    // interleaved MRoPE collapses exactly onto NEOX-ordered partial RoPE.
    // A vision/video checkpoint with per-axis positions would need the full
    // sectioned form; that is a different Rope type, not a flag here.
    using Rope = RotaryEmbedding<C::HEAD_DIM, C::ROPE_BASE, C::ROTARY_NUM, C::ROTARY_DEN>;

    static PrefixState make_prefix_state() { return {}; }
    static size_t prefix_tokens(const PrefixState& state) { return state.tokens(); }
    static void advance_prefix(PrefixState& state, size_t count) { state.advance(count); }

    static EmbeddedSequence<D> embed(const Weights& weights, std::span<const TokenId> tokens, size_t first_position) {
        return embed_text_tokens<D>(tokens, first_position, [&](std::span<const TokenId> batch) { return weights.token_io().tokens(batch); });
    }

    static PreparedInput prepare(const Weights&, const EmbeddedSequence<D>&) { return {}; }

    // One layer, whole. This is the payoff for calling attention a token mixer:
    // the two kinds of Qwen layer have the SAME equation, and std::visit picks
    // which mixer runs. mix_tokens is overloaded on the mixer type — gated
    // attention for the 1-in-4 full layers, the gated delta net for the rest —
    // and that overload is the only difference between a recurrent layer and an
    // attention layer in the entire stack.
    static void forward_layer(const Weights& weights, PrefixState& state, const EmbeddedSequence<D>& input, ResidualStream<D>& residual, const PreparedInput&, size_t layer_index) {
        const size_t first_position = input.position(0).i;
        const MatrixView<D> X = residual.matrix();
        Matrix<D> next = std::visit(
            [&](const auto& layer) {
                // h = x + mix( norm(x) )
                Matrix<D> U = layer.mixer_norm(X);
                Matrix<D> M = mix_tokens(layer.mixer, U.view(), state, layer_index, first_position);
                Matrix<D> H = add(X, M.view());

                // out = h + mlp( norm(h) )
                Matrix<D> Z = layer.channel_norm(H.view());
                Matrix<D> F = layer.channel(Z.view());
                return add(H.view(), F.view());
            },
            weights.layer(layer_index));
        residual.set_matrix(std::move(next));
    }

    static Vec<V> output(const Weights& weights, const ResidualStream<D>& residual) {
        Vec<D> last = weights.final_norm()(residual.token(residual.tokens() - 1));
        return weights.token_io().logits(last);
    }

private:
    // Gated attention, over T tokens starting at first_position:
    //
    //     [Q | G] = X W_qg      (head-major pairs)
    //     Q = rope( rmsnorm_per_head(Q) )
    //     K = rope( rmsnorm_per_head( X W_k ) )
    //     V =                     X W_v
    //     A = attend(Q, cache <- K,V) * sigmoid(G)
    //     out = A W_o
    static Matrix<D> mix_tokens(const QwenAttentionMixer<C>& attention, MatrixView<D> X, PrefixState& state, size_t layer, size_t first_position) {
        constexpr size_t Dh = C::HEAD_DIM;
        constexpr size_t QW = C::Hq * Dh;

        HeadPair<C::Hq, Dh> query_gate = attention.query_and_gate(X);
        Matrix<QW>& Q = query_gate.first;
        rotate_heads<C::Hq, Dh>(Q, Rope{}, first_position);

        Matrix<C::Hkv * Dh> K = attention.key(X);
        rotate_heads<C::Hkv, Dh>(K, Rope{}, first_position);
        Matrix<C::Hkv * Dh> V = attention.value(X);

        Matrix<QW> A = attend_and_cache<C::Hq, C::Hkv, Dh>(Q.view(), K.view(), V.view(), state.attention(layer), first_position, /*sliding_window=*/0,
                                                           /*score_scale=*/1.f / std::sqrt(Scalar(Dh)));

        // The gate is applied to the attended value, before W_o.
        for (size_t i = 0; i < A.rows() * QW; ++i) A.data()[i] *= sigmoid(query_gate.second.data()[i]);
        return attention.output(A.view());
    }

    // Gated delta network. The projections are batched matmuls; only the
    // recurrence itself is sequential in T, because the state at token t is
    // by definition a function of the state at t-1.
    static Matrix<D> mix_tokens(const QwenRecurrentMixer<C>& mixer, MatrixView<D> X, PrefixState& state, size_t layer, size_t /*first_position*/) {
        using Mixer = QwenRecurrentMixer<C>;
        constexpr size_t Dk = C::KEY_HEAD_DIM;
        constexpr size_t Dv = C::VALUE_HEAD_DIM;
        constexpr size_t Hk = C::KEY_HEADS;
        constexpr size_t Hv = C::VALUE_HEADS;
        static_assert(Dk == Dv, "the delta rule requires a square state block");
        const Scalar query_scale = 1.f / std::sqrt(Scalar(Dk));

        auto& entry = state.recurrent(layer);
        Matrix<Mixer::CONV_CHANNELS> qkv = mixer.qkv(X);
        Matrix<Hv> beta = mixer.write_strength(X);
        Matrix<Hv> gate = mixer.decay_gate(X);
        Matrix<Mixer::VALUE_WIDTH> mixed(X.rows());

        for (size_t t = 0; t < X.rows(); ++t) {
            Vec<Mixer::CONV_CHANNELS> stream = entry.conv.step(qkv.row(t), mixer.conv());
            silu(stream);

            // q and k are shared by every value head in a key head's group, so
            // normalize them once per token rather than once per value head.
            std::array<Vec<Dk>, Hk> query, key;
            for (size_t hk = 0; hk < Hk; ++hk) {
                query[hk] = l2_normalize<Dk>(slice<Dk>(VecView<Mixer::CONV_CHANNELS>(stream), hk * Dk));
                key[hk] = l2_normalize<Dk>(slice<Dk>(VecView<Mixer::CONV_CHANNELS>(stream), Mixer::KEY_WIDTH + hk * Dk));
                for (size_t i = 0; i < Dk; ++i) query[hk][i] *= query_scale;
            }

            Scalar* row = mixed.data() + t * Mixer::VALUE_WIDTH;
            for (size_t h = 0; h < Hv; ++h) {
                const size_t hk = Mixer::key_head_of(h);
                Vec<Dv> attended = gated_delta_step<Dk, Dv>(entry.delta.head(h), VecView<Dk>(query[hk]), VecView<Dk>(key[hk]), slice<Dv>(VecView<Mixer::CONV_CHANNELS>(stream), 2 * Mixer::KEY_WIDTH + h * Dv), gate.row(t)[h], beta.row(t)[h]);
                std::copy(attended.begin(), attended.end(), row + h * Dv);
            }
        }

        // Gated output norm: rmsnorm per head, then the SiLU'd z gate.
        Matrix<Mixer::VALUE_WIDTH> normalized = mixer.normalize_heads(mixed.view());
        Matrix<Mixer::VALUE_WIDTH> z = mixer.gate(X);
        silu_in_place(z.mutable_view());
        return mixer.output(hadamard(normalized.view(), z.view()).view());
    }
};

template <class C>
using Qwen35Weights = typename Qwen35Architecture<C>::Weights;

template <class C>
using Qwen35Transformer = Transformer<Qwen35Architecture<C>>;
