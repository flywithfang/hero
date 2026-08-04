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
// The tensors, in the order the equation applies them. The layer cannot run
// itself — attention needs this layer's cache, which belongs to the PrefixState
// — so the equation lives in Qwen35Architecture::mix_tokens.
template <size_t D, size_t Hq, size_t Hkv, size_t HeadDim>
struct QwenGatedAttention {
    static constexpr size_t QW = Hq * HeadDim;
    static constexpr size_t KW = Hkv * HeadDim;
    static constexpr size_t GATED_QW = 2 * QW;

    Linear<D, GATED_QW> WQG;  // query and gate together, head-major pairs
    PerHeadNorm<RMSNorm<HeadDim>> q_norm;
    Linear<D, KW> WK;
    PerHeadNorm<RMSNorm<HeadDim>> k_norm;
    Linear<D, KW> WV;  // never normalized, never rotated
    Linear<QW, D> WO;
};

// ---- token mixer 2: gated delta network -------------------------------------
//
// The packed projection emits [q | k | v] over ConvChannels, which a depthwise
// causal conv1d and a SiLU pass through before the delta rule sees them. beta
// and the decay gate are separate per-value-head projections of the same input.
template <size_t D, size_t Dk, size_t Dv, size_t Hk, size_t Hv, size_t ConvWidth>
struct QwenGatedDeltaNet {
    static_assert(Hv % Hk == 0, "value heads must be a whole multiple of key heads");
    static constexpr size_t KEY_WIDTH = Dk * Hk;
    static constexpr size_t VALUE_WIDTH = Dv * Hv;
    static constexpr size_t CONV_CHANNELS = 2 * KEY_WIDTH + VALUE_WIDTH;

    Linear<D, CONV_CHANNELS> WQKV;
    Linear<D, VALUE_WIDTH> WZ;  // "z", the output gate
    Matrix<ConvWidth> conv;     // one row of taps per channel
    Linear<D, Hv> Wbeta;
    Linear<D, Hv> Walpha;
    Vec<Hv> dt_bias;
    Vec<Hv> decay_scale;  // -exp(A_log), so <= 0; the loader checks it
    PerHeadNorm<RMSNorm<Dv>> head_norm;
    Linear<VALUE_WIDTH, D> WO;

    // The one head-mapping decision, in one place. ggml TILES the key heads
    // across the value heads (head h reads key head h % Hk) rather than
    // grouping them the way GQA does (h / (Hv/Hk)).
    //
    // VERIFIED against the real Qwen 3.5 4B Q4_K_M checkpoint by flipping it:
    // tiling gives "The capital of France is" -> " Paris" (logprob -0.66) and
    // coherent text; grouping gives " a" (-2.05) and gibberish. Both mappings
    // are shape-legal, so only weights could decide it.
    static constexpr size_t key_head_of(size_t value_head) { return value_head % Hk; }
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

// ---- prefix state -----------------------------------------------------------
//
// This is where the hybrid stack shows its economics. A full-attention layer
// carries a K/V cache that grows with T; a linear layer carries a conv history
// and a state matrix whose size does not depend on T at all.
template <class C>
class Qwen35State {
public:
    using AttentionEntry = FullAttentionCache<C::Hkv, C::HEAD_DIM>;
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

// ---- the model --------------------------------------------------------------
//
// Qwen 3.5: its tensors and its math, one entity — two layer vectors (3:1
// recurrent to attention), immutable and non-copyable. Call it with
// cache.evaluate(model, input).
template <class C>
class Qwen35Model {
public:
    static constexpr size_t D = C::D;
    static constexpr size_t V = C::V;
    static constexpr size_t L = C::L;
    static constexpr size_t CTX = C::CTX;

    using PrefixState = Qwen35State<C>;
    using RecurrentLayer = Qwen35Block<C, QwenRecurrentMixer<C>>;
    using AttentionLayer = Qwen35Block<C, QwenAttentionMixer<C>>;

    // Which slot of its kind's layer vector layer `i` occupies.
    static constexpr size_t position_in_kind(size_t layer) {
        size_t n = 0;
        for (size_t i = 0; i < layer; ++i)
            if (C::is_recurrent(i) == C::is_recurrent(layer)) ++n;
        return n;
    }

    // Text positions make every MRoPE section carry the same value, so
    // interleaved MRoPE collapses exactly onto NEOX-ordered partial RoPE.
    // A vision/video checkpoint with per-axis positions would need the full
    // sectioned form; that is a different Rope type, not a flag here.
    using Rope = RotaryEmbedding<C::HEAD_DIM, C::ROPE_BASE, C::ROTARY_NUM, C::ROTARY_DEN>;

    Qwen35Model(Weight<C::D, C::V> embedding, typename C::OutputWeight unembedding, std::vector<RecurrentLayer> recurrent, std::vector<AttentionLayer> attention, RMSNorm<D> final_norm) : embedding_(std::move(embedding)), unembedding_(std::move(unembedding)), recurrent_(std::move(recurrent)), attention_(std::move(attention)), final_norm_(std::move(final_norm)) {
        if (recurrent_.size() != 3 * attention_.size() || recurrent_.size() + attention_.size() != L) throw std::invalid_argument("Qwen35Model: layer kind counts disagree with the 3:1 hybrid schedule");
    }

    Qwen35Model(const Qwen35Model&) = delete;
    Qwen35Model& operator=(const Qwen35Model&) = delete;
    Qwen35Model(Qwen35Model&&) = default;
    Qwen35Model& operator=(Qwen35Model&&) = delete;


    // Qwen reads embedding rows as they are: no input scale.
    Matrix<D> embed(std::span<const TokenId> ids) const { return embedding_.gather_rows(ids); }

    // Nothing to precompute: a Qwen layer reads only the residual stream.
    Logits<V> forward(PrefixState& state, EmbeddedRows<D> input) const {
        if (input.tokens() > CTX - state.tokens()) throw std::length_error("Qwen35Model: context exhausted");

        ResidualStream<D> residual(input);
        const Position conversation_position = input.conversation_position();
        for (size_t layer = 0; layer < L; ++layer) residual.replace(forward_layer(state, conversation_position, residual.matrix(), layer));
        state.advance(input.tokens());

        const auto last = final_norm_(residual.hidden(residual.tokens() - 1));
        // 4B ties its embedding to the head; 9B carries a separate one. The
        // config says which, so the choice costs nothing at run time.
        if constexpr (std::is_same_v<typename C::OutputWeight, std::monostate>)
            return Logits<V>(embedding_.matvec(last));
        else
            return Logits<V>(unembedding_.matvec(last));
    }

private:
    Matrix<D> forward_layer(PrefixState& state, Position conversation_position, MatrixView<D> X, size_t layer_index) const {
        const size_t n = position_in_kind(layer_index);
        return C::is_recurrent(layer_index) ? run_layer(recurrent_.at(n), X, state, layer_index, conversation_position) : run_layer(attention_.at(n), X, state, layer_index, conversation_position);
    }

    // One layer, whole. This is the payoff for calling attention a token mixer:
    // the two kinds of Qwen layer have the SAME equation. mix_tokens is
    // overloaded on the mixer type — gated attention for the 1-in-4 full
    // layers, the gated delta net for the rest — and that overload is the only
    // difference between a recurrent layer and an attention layer in the stack.
    template <class Layer>
    static Matrix<D> run_layer(const Layer& layer, MatrixView<D> X, PrefixState& state, size_t layer_index, Position conversation_position) {
        // h = x + mix( norm(x) )
        const auto U = layer.mixer_norm(X);
        const auto M = mix_tokens(layer.mixer, U.view(), state, layer_index, conversation_position);
        const auto H = X + M;

        // out = h + mlp( norm(h) )
        const auto Z = layer.channel_norm(H.view());
        const auto F = layer.channel(Z.view());
        return H + F;
    }

    Weight<C::D, C::V> embedding_;
    [[no_unique_address]] typename C::OutputWeight unembedding_;  // empty when tied
    std::vector<RecurrentLayer> recurrent_;  // 3 of every 4 layers
    std::vector<AttentionLayer> attention_;  // every 4th
    RMSNorm<D> final_norm_;

    // Gated attention, over T tokens starting at conversation_position:
    //
    //     [Q | G] = X W_qg      (head-major pairs)
    //     Q = rope( rmsnorm_per_head(Q) )
    //     K = rope( rmsnorm_per_head( X W_k ) )
    //     V =                     X W_v
    //     A = attend(Q, cache <- K,V) * sigmoid(G)
    //     out = A W_o
    static Matrix<D> mix_tokens(const QwenAttentionMixer<C>& attention, MatrixView<D> X, PrefixState& state, size_t layer, Position conversation_position) {
        constexpr size_t Dh = C::HEAD_DIM;
        constexpr size_t QW = C::Hq * Dh;

        // One projection emits Q and G interleaved per head, so the split is
        // pure layout; .first is the query, .second the raw gate.
        HeadPair<C::Hq, Dh> query_gate = split_head_pairs<C::Hq, Dh>(attention.WQG(X).view());
        const auto Q = Rope{}(attention.q_norm(std::move(query_gate.first)), conversation_position);
        const auto K = Rope{}(attention.k_norm(attention.WK(X)), conversation_position);
        const auto V = attention.WV(X);

        // Qwen's reduction. Same shape as Gemma's, one difference that matters:
        // Qwen keeps an explicit 1/sqrt(Dh) score scale, where Gemma folds it
        // into the learned Q norm and passes nothing. That single constant is
        // why these are two equations rather than one with a parameter.
        FullAttentionCache<C::Hkv, Dh>& cache = state.attention(layer);
        const Scalar score_scale = 1.f / std::sqrt(Scalar(Dh));
        const auto reduce = [&](MatrixView<QW> queries, Position at) {
            std::vector<VisibleRows> visible;
            visible.reserve(queries.rows());
            for (size_t token_pos = 0; token_pos < queries.rows(); ++token_pos) visible.push_back(cache.visible_rows(at + token_pos));

            Matrix<QW> attended = Matrix<QW>::zero_rows(queries.rows());
            par_for(queries.rows() * C::Hq, [&](size_t task) {
                const size_t token_pos = task / C::Hq, head = task % C::Hq;
                const VecView<Dh> query = slice<Dh>(queries.row(token_pos), head);
                const KVHead group{head / (C::Hq / C::Hkv)};
                const VisibleRows visible_tokens = visible[token_pos];

                std::vector<Scalar> alpha;
                alpha.reserve(visible_tokens.count());
                for (size_t j = visible_tokens.first; j < visible_tokens.end; ++j) alpha.push_back(score_scale * dot(query, cache.key(j, group)));
                alpha = Softmax{}(std::move(alpha));

                Vec<Dh> head_output;
                for (size_t n = 0; n < alpha.size(); ++n) head_output.scaled_add(cache.value(visible_tokens.first + n, group), alpha[n]);
                attended.template replace_partition<Dh>(token_pos, head, head_output);
            });
            return attended;
        };

        // Full attention with no window: the cache is unbounded, so the batch
        // always fits and the eviction path cannot fire.
        for (size_t token_pos = 0; token_pos < Q.rows(); ++token_pos) cache.append(conversation_position + token_pos, K.row(token_pos), V.row(token_pos));
        Matrix<QW> A = reduce(Q.view(), conversation_position);

        // The gate is applied to the attended value, before W_o.
        A.transform_rows([&](size_t token_pos, MutVecView<QW> attended) {
            const VecView<QW> gate = query_gate.second.row(token_pos);
            for (size_t i = 0; i < QW; ++i) attended[i] *= sigmoid(gate[i]);
        });
        return attention.WO(A.view());
    }

    // Per value head: beta in (0,1) is the write strength; the decay gate is
    // decay_scale * softplus(alpha + dt_bias) and is <= 0 because the converter
    // stores decay_scale = -exp(A_log). exp() of it is therefore a contraction.
    static Matrix<C::VALUE_HEADS> write_strength(const QwenRecurrentMixer<C>& mixer, MatrixView<D> X) {
        constexpr size_t Hv = C::VALUE_HEADS;
        Matrix<Hv> beta = mixer.Wbeta(X);
        beta.transform_rows([](MutVecView<Hv> row) {
            for (size_t h = 0; h < Hv; ++h) row[h] = sigmoid(row[h]);
        });
        return beta;
    }
    static Matrix<C::VALUE_HEADS> decay_gate(const QwenRecurrentMixer<C>& mixer, MatrixView<D> X) {
        constexpr size_t Hv = C::VALUE_HEADS;
        Matrix<Hv> alpha = mixer.Walpha(X);
        alpha.transform_rows([&](MutVecView<Hv> row) {
            for (size_t h = 0; h < Hv; ++h) row[h] = mixer.decay_scale[h] * softplus(row[h] + mixer.dt_bias[h]);
        });
        return alpha;
    }

    // Gated delta network. The projections are batched matmuls; only the
    // recurrence itself is sequential in T, because the state at token t is
    // by definition a function of the state at t-1.
    static Matrix<D> mix_tokens(const QwenRecurrentMixer<C>& mixer, MatrixView<D> X, PrefixState& state, size_t layer, Position /*conversation_position*/) {
        using Mixer = QwenRecurrentMixer<C>;
        constexpr size_t Dk = C::KEY_HEAD_DIM;
        constexpr size_t Dv = C::VALUE_HEAD_DIM;
        constexpr size_t Hk = C::KEY_HEADS;
        constexpr size_t Hv = C::VALUE_HEADS;
        static_assert(Dk == Dv, "the delta rule requires a square state block");
        const Scalar query_scale = 1.f / std::sqrt(Scalar(Dk));

        auto& entry = state.recurrent(layer);
        const auto qkv = mixer.WQKV(X);
        const auto beta = write_strength(mixer, X);
        const auto gate = decay_gate(mixer, X);
        Matrix<Mixer::VALUE_WIDTH> mixed;
        mixed.reserve(X.rows());

        for (size_t t = 0; t < X.rows(); ++t) {
            const auto stream = Silu{}(entry.conv.step(qkv.row(t), mixer.conv));

            // q and k are shared by every value head in a key head's group, so
            // normalize them once per token rather than once per value head.
            std::array<Vec<Dk>, Hk> query, key;
            for (size_t hk = 0; hk < Hk; ++hk) {
                query[hk] = l2_normalize<Dk>(slice<Dk>(VecView<Mixer::CONV_CHANNELS>(stream), hk));
                key[hk] = l2_normalize<Dk>(slice<Dk, Mixer::KEY_WIDTH>(VecView<Mixer::CONV_CHANNELS>(stream), hk));
                for (size_t i = 0; i < Dk; ++i) query[hk][i] *= query_scale;
            }

            Vec<Mixer::VALUE_WIDTH> row;
            for (size_t h = 0; h < Hv; ++h) {
                const size_t hk = Mixer::key_head_of(h);
                const auto attended = gated_delta_step<Dk, Dv>(entry.delta.head(h), VecView<Dk>(query[hk]), VecView<Dk>(key[hk]), slice<Dv, 2 * Mixer::KEY_WIDTH>(VecView<Mixer::CONV_CHANNELS>(stream), h), gate.row(t)[h], beta.row(t)[h]);
                const MutVecView<Dv> partition = slice_mut<Dv>(row, h);
                for (size_t i = 0; i < Dv; ++i) partition[i] = attended[i];
            }
            mixed.append(row);
        }

        // Gated output norm: rmsnorm per head, then the SiLU'd z gate.
        mixed = mixer.head_norm(std::move(mixed));
        const auto z = Silu{}(mixer.WZ(X));
        return mixer.WO(hadamard(mixed.view(), z.view()).view());
    }
};
