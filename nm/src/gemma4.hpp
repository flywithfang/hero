// gemma4.hpp — Gemma 4 shared text primitives and checkpoint configurations.
//
// This file supplies Gemma-specific policies to the uniform transformer core:
// two attention shapes, partial RoPE, KV sharing, PLE, and logit soft-capping.
// The model-neutral pieces (K/V storage, the mask, rotary tables, per-head norm)
// live in attention.hpp. The softmax reduction over cached keys does NOT: each
// layer kind writes its own, because that equation is where families diverge.
// Vision and audio encoders connect at EmbeddedSequence<D> and do not enter
// these components.
#pragma once
#include "attention.hpp"
#include "components.hpp"
#include "transformer.hpp"
#include <array>
#include <variant>

enum class GemmaAttentionType : uint8_t { Local, Full };

struct Gemma4E4BTextConfig {
    static constexpr size_t V = 262144;
    static constexpr size_t D = 2560;
    static constexpr size_t L = 42;
    static constexpr size_t Hq = 8;
    static constexpr size_t Hkv = 2;
    static constexpr size_t LOCAL_HEAD_DIM = 256;
    static constexpr size_t FULL_HEAD_DIM = 512;
    static constexpr size_t FF = 10240;
    static constexpr size_t PLE = 256;
    static constexpr size_t CTX = 131072;
    static constexpr size_t LOCAL_WINDOW = 512;
    static constexpr size_t SHARED_KV_LAYERS = 18;
    static constexpr size_t FIRST_SHARED_KV_LAYER = L - SHARED_KV_LAYERS;
    static constexpr size_t LOCAL_ROPE_BASE = 10000;
    static constexpr size_t FULL_ROPE_BASE = 1000000;
    static constexpr Scalar RMS_EPS = 1e-6f;
    static constexpr Scalar LOGIT_SOFTCAP = 30.f;

    static constexpr GemmaAttentionType attention_type(size_t layer) { return layer % 6 == 5 ? GemmaAttentionType::Full : GemmaAttentionType::Local; }
    static constexpr bool uses_shared_kv(size_t layer) { return layer >= FIRST_SHARED_KV_LAYER; }
    // The final non-sharing layer of each attention type supplies all later
    // layers of that type: local layer 22, full layer 23 for E4B's 42 layers.
    static constexpr bool is_shared_kv_source(size_t layer) {
        if (layer >= FIRST_SHARED_KV_LAYER) return false;
        const GemmaAttentionType type = attention_type(layer);
        for (size_t next = layer + 1; next < FIRST_SHARED_KV_LAYER; ++next)
            if (attention_type(next) == type) return false;
        return true;
    }
    // Which slot of its shape's weights vector layer `i` occupies: the count
    // of earlier layers with the same (attention type, KV ownership).
    static constexpr size_t position_in_group(size_t layer) {
        size_t n = 0;
        for (size_t i = 0; i < layer; ++i)
            if (attention_type(i) == attention_type(layer) && uses_shared_kv(i) == uses_shared_kv(layer)) ++n;
        return n;
    }
};

static_assert(Gemma4E4BTextConfig::attention_type(4) == GemmaAttentionType::Local);
static_assert(Gemma4E4BTextConfig::attention_type(5) == GemmaAttentionType::Full);
static_assert(Gemma4E4BTextConfig::FIRST_SHARED_KV_LAYER == 24);
static_assert(Gemma4E4BTextConfig::is_shared_kv_source(22));
static_assert(Gemma4E4BTextConfig::is_shared_kv_source(23));
static_assert(Gemma4E4BTextConfig::uses_shared_kv(24));
static_assert(Gemma4E4BTextConfig::position_in_group(0) == 0 && Gemma4E4BTextConfig::position_in_group(6) == 5, "local owning layers count their own type only");
static_assert(Gemma4E4BTextConfig::position_in_group(5) == 0 && Gemma4E4BTextConfig::position_in_group(23) == 3, "full-own layers are 5, 11, 17, 23");
static_assert(Gemma4E4BTextConfig::position_in_group(24) == 0 && Gemma4E4BTextConfig::position_in_group(29) == 0 && Gemma4E4BTextConfig::position_in_group(35) == 1, "shared layers restart their count at layer 24");

// What a layer physically holds for K/V is a property of the checkpoint.
// Three cases exist across Gemma 4, and each is expressed by which members a
// layer struct HAS — there is no kind flag at runtime and no kind type:
//   Owned    W_k and W_v both present (E4B's first 24 layers, 12B's local)
//   Unified  W_k only, V = normalized raw W_k output (12B's full layers)
//   Shared   neither; the layer attends against what an earlier layer wrote
// This enum survives only as config/loader vocabulary (kv_kind(), the
// parameter count); the layer structs themselves just have or lack tensors.
enum class GemmaKVKind : uint8_t { Owned, Unified, Shared };

inline Scalar gemma_softcap(Scalar value, Scalar cap) { return cap * std::tanh(value / cap); }

// Value in, value out. `values` is a by-value sink: the caller's prvalue
// materializes straight into it, the cap is applied in place, and the result
// moves out — one allocation, no copy, and no caller left holding something
// half-capped. A free function that wrote through a mutable view would cost the
// same and say nothing at the call site about what it had changed.
template <size_t N>
Vec<N> gemma_softcap(Vec<N> values, Scalar cap) {
    for (size_t i = 0; i < N; ++i) values[i] = gemma_softcap(values[i], cap);
    return values;
}

// The per-layer part of E4B's PLE residual. Model-level token/context PLE
// construction is separate because multimodal soft tokens have no token id.
template <size_t D, size_t P>
class GemmaPerLayerResidual {
public:
    GemmaPerLayerResidual(Linear<D, P> gate, Linear<P, D> projection, RMSNorm<D> post_norm) : gate_(std::move(gate)), projection_(std::move(projection)), post_norm_(std::move(post_norm)) {}

    Vec<D> operator()(VecView<D> hidden, VecView<P> per_layer_input) const {
        const auto gated = Gelu{}(gate_(hidden));
        const auto branch_input = hadamard(VecView<P>(gated), per_layer_input);
        Vec<D> branch = post_norm_(projection_(branch_input));
        branch += hidden;
        return branch;
    }
    Matrix<D> operator()(MatrixView<D> hidden, MatrixView<P> per_layer_input) const {
        const auto gated = Gelu{}(gate_(hidden));
        const auto branch = post_norm_(projection_(hadamard(gated.view(), per_layer_input)));
        return hidden + branch;
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

    GemmaPerLayerInputs(Weight<Packed, V> token_embeddings, Linear<D, Packed> model_projection, PerHeadNorm<RMSNorm<P>> projection_norm, Scalar token_scale, Scalar projection_scale, Scalar combination_scale) : token_embeddings_(std::move(token_embeddings)), model_projection_(std::move(model_projection)), projection_norm_(std::move(projection_norm)), token_scale_(token_scale), projection_scale_(projection_scale), combination_scale_(combination_scale) {}

    Vec<Packed> operator()(VecView<D> input_embedding, TokenId identity) const {
        Vec<Packed> token = token_embeddings_.dequant_row(size_t(identity));
        token.scale(token_scale_);

        Vec<Packed> context = model_projection_(input_embedding);
        context.scale(projection_scale_);
        context = projection_norm_(std::move(context));  // partitions are layers here

        return scaled_sum(VecView<Packed>(context), VecView<Packed>(token), combination_scale_);
    }
    Matrix<Packed> operator()(MatrixView<D> input_embeddings, std::span<const TokenId> identities) const {
        if (input_embeddings.rows() != identities.size()) throw std::invalid_argument("GemmaPerLayerInputs: embedding/identity row mismatch");

        Matrix<Packed> token = token_embeddings_.gather_rows(identities);
        token.scale(token_scale_);

        Matrix<Packed> context = model_projection_(input_embeddings);
        context.scale(projection_scale_);
        context = projection_norm_(std::move(context));

        return scaled_sum(context.view(), token.view(), combination_scale_);
    }

private:
    Weight<Packed, V> token_embeddings_;
    Linear<D, Packed> model_projection_;
    PerHeadNorm<RMSNorm<P>> projection_norm_;
    Scalar token_scale_;
    Scalar projection_scale_;
    Scalar combination_scale_;
};


// ---- E4B layers: flat tensor lists, one struct per physical layer shape -----
//
// The layer equation, applied in member order:
//
//     h   = x + post_norm( attn( pre_norm(x) ) )
//     h   = h + post_norm( mlp( pre_norm(h) ) )         GELU-gated FFN
//     h   = h + norm( (gelu(h W_gate) (*) ple) W_proj )  the PLE residual
//     out = layer_output_scale * h
//
// Two head widths exist (local 256, full 512), and the last 18 layers carry
// no K/V tensors at all — so there are two structs, each instantiated at two
// widths. Each type implements the operation appropriate to its physical
// tensors; no member-detection convention decides its behavior elsewhere.
template <size_t Dh, class Cache>
struct GemmaE4BLayer {
    static constexpr size_t D = Gemma4E4BTextConfig::D;
    static constexpr size_t Hq = Gemma4E4BTextConfig::Hq;
    static constexpr size_t HEAD_DIM = Dh;
    static constexpr size_t Hkv = Gemma4E4BTextConfig::Hkv;

    RMSNorm<D> attn_norm;
    Linear<D, Hq * Dh> WQ;
    PerHeadNorm<RMSNorm<Dh>> q_norm;
    Linear<D, Hkv * Dh> WK;
    PerHeadNorm<RMSNorm<Dh>> k_norm;
    Linear<D, Hkv * Dh> WV;
    PerHeadNorm<RMSNormNoScale<Dh>> v_norm;  // V's norm has no learned scale
    Linear<Hq * Dh, D> WO;
    RMSNorm<D> post_attn_norm;
    RMSNorm<D> ffn_norm;
    GeluGatedMLP<D, Gemma4E4BTextConfig::FF> ffn;
    RMSNorm<D> post_ffn_norm;
    GemmaPerLayerResidual<D, Gemma4E4BTextConfig::PLE> ple;
    Scalar layer_output_scale;

    // This layer owns K and V: project them, append them, and reduce over the
    // rows the window leaves visible. Gemma folds 1/sqrt(Dh) into the learned Q
    // norm, so there is no score scale in the dot product here:
    //   score_j = dot(q_head, k_j,group)
    //   alpha   = softmax(score)
    //   out     = sum_j alpha_j * v_j,group
    // GQA is explicit: adjacent groups of Hq/Hkv query heads share one cached
    // K/V head. Rows x query heads are one flat parallel task space.
    template <HeadRotation<Dh> Rope>
    Matrix<D> attention(MatrixView<D> X, Cache& cache, const Rope& rope, Position conversation_position) const {
        const auto Q = rope(q_norm(WQ(X)), conversation_position);

        const auto K = rope(k_norm(WK(X)), conversation_position);
        const auto V = v_norm(WV(X));

        const auto reduce = [&](MatrixView<Hq * Dh> queries, Position at) {
            std::vector<VisibleRows> visible;
            visible.reserve(queries.rows());
            for (size_t row = 0; row < queries.rows(); ++row){
                 visible.push_back(cache.visible_rows(at + row));
            }

            Matrix<Hq * Dh> attended = Matrix<Hq * Dh>::zero_rows(queries.rows());
            par_for(queries.rows() * Hq, [&](size_t task) {
                const size_t row = task / Hq, head = task % Hq;
                const VecView<Dh> query = slice<Dh>(queries.row(row), head);
                const KVHead group{head / (Hq / Hkv)};
                const VisibleRows rows = visible[row];

                std::vector<Scalar> alpha;
                alpha.reserve(rows.count());
                for (size_t j = rows.first; j < rows.end; ++j) alpha.push_back(dot(query, cache.key(j, group)));
                alpha = Softmax{}(std::move(alpha));

                Vec<Dh> head_output;
                for (size_t n = 0; n < alpha.size(); ++n) head_output.scaled_add(cache.value(rows.first + n, group), alpha[n]);
                attended.template replace_partition<Dh>(row, head, head_output);
            });
            return attended;
        };

        // Prefill appends the batch once and attends in parallel. A bounded ring
        // that would evict mid-batch advances row by row instead, so the early
        // queries keep the keys they are still allowed to see.
        Matrix<Hq * Dh> A;
        if (cache.can_append_without_eviction(Q.rows())) {
            for (size_t row = 0; row < Q.rows(); ++row) {
                cache.append(conversation_position + row, K.row(row), V.row(row));
            }
            A = reduce(Q.view(), conversation_position);
        } else {
            A.reserve(Q.rows());
            for (size_t row = 0; row < Q.rows(); ++row) {
                const Position at = conversation_position + row;
                cache.append(at, K.row(row), V.row(row));
                A.append(reduce(Q.view().single_row(row), at).row(0));
            }
        }
        return WO(A.view());
    }

    template <HeadRotation<Dh> Rope>
    Matrix<D> forward(MatrixView<D> X, Cache& cache, const Rope& rope, Position conversation_position, MatrixView<Gemma4E4BTextConfig::PLE> per_layer_input) const {
        const auto U = attn_norm(X);
        const auto A = post_attn_norm(attention(U.view(), cache, rope, conversation_position));
        Matrix<D> H = X + A;

        const auto Z = ffn_norm(H.view());
        const auto F = post_ffn_norm(ffn(Z.view()));
        H = H + F;

        H = ple(H.view(), per_layer_input);
        H.scale(layer_output_scale);
        return H;
    }
};

// The last 18 layers: the same layer minus the K/V tensors, which the
// checkpoint does not carry. Not an optional, not a flag — the members
// do not exist, so misusing them is a compile error.
template <size_t Dh, class Cache>
struct GemmaE4BSharedLayer {
    static constexpr size_t D = Gemma4E4BTextConfig::D;
    static constexpr size_t Hq = Gemma4E4BTextConfig::Hq;
    static constexpr size_t Hkv = Gemma4E4BTextConfig::Hkv;
    static constexpr size_t HEAD_DIM = Dh;

    RMSNorm<D> attn_norm;
    Linear<D, Hq * Dh> WQ;
    PerHeadNorm<RMSNorm<Dh>> q_norm;
    Linear<Hq * Dh, D> WO;
    RMSNorm<D> post_attn_norm;
    RMSNorm<D> ffn_norm;
    GeluGatedMLP<D, Gemma4E4BTextConfig::FF> ffn;
    RMSNorm<D> post_ffn_norm;
    GemmaPerLayerResidual<D, Gemma4E4BTextConfig::PLE> ple;
    Scalar layer_output_scale;

    // This layer owns no K/V. It forms Q and reduces against an earlier owning
    // layer's cache, which it takes by CONST reference: never appending is a
    // property of this layer's anatomy, so the signature enforces it. The
    // reduction is the same equation the owning layer writes, minus the append.
    template <HeadRotation<Dh> Rope>
    Matrix<D> attention(MatrixView<D> X, const Cache& cache, const Rope& rope, Position conversation_position) const {
        const auto Q = rope(q_norm(WQ(X)), conversation_position);

        std::vector<VisibleRows> visible;
        visible.reserve(Q.rows());
        for (size_t row = 0; row < Q.rows(); ++row) visible.push_back(cache.visible_rows(conversation_position + row));

        Matrix<Hq * Dh> A = Matrix<Hq * Dh>::zero_rows(Q.rows());
        par_for(Q.rows() * Hq, [&](size_t task) {
            const size_t token_pos = task / Hq, head = task % Hq;
            const VecView<Dh> query = slice<Dh>(Q.row(token_pos), head);
            const KVHead group{head / (Hq / Hkv)};
            const VisibleRows visible_tokens = visible[token_pos];

            std::vector<Scalar> alpha;
            alpha.reserve(visible_tokens.count());
            for (size_t j = visible_tokens.first; j < visible_tokens.end; ++j) alpha.push_back(dot(query, cache.key(j, group)));
            alpha = Softmax{}(std::move(alpha));

            Vec<Dh> head_output;
            for (size_t n = 0; n < alpha.size(); ++n) head_output.scaled_add(cache.value(visible_tokens.first + n, group), alpha[n]);
            A.template replace_partition<Dh>(token_pos, head, head_output);
        });
        return WO(A.view());
    }

    template <HeadRotation<Dh> Rope>
    Matrix<D> forward(MatrixView<D> X, const Cache& cache, const Rope& rope, Position conversation_position, MatrixView<Gemma4E4BTextConfig::PLE> per_layer_input) const {
        const auto U = attn_norm(X);
        const auto A = post_attn_norm(attention(U.view(), cache, rope, conversation_position));
        Matrix<D> H = X + A;

        const auto Z = ffn_norm(H.view());
        const auto F = post_ffn_norm(ffn(Z.view()));
        H = H + F;

        H = ple(H.view(), per_layer_input);
        H.scale(layer_output_scale);
        return H;
    }
};

using GemmaE4BModelData = GemmaPerLayerInputs<Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::PLE, Gemma4E4BTextConfig::L, Gemma4E4BTextConfig::V>;

class Gemma4E4BCache {
    using C = Gemma4E4BTextConfig;

public:
    using Local = LocalAttentionCache<C::Hkv, C::LOCAL_HEAD_DIM>;
    using Full = FullAttentionCache<C::Hkv, C::FULL_HEAD_DIM>;

private:
    using Entry = std::variant<std::monostate, Local, Full>;

public:
    Gemma4E4BCache() {
        for (size_t layer = 0; layer < C::L; ++layer) {
            if (C::uses_shared_kv(layer)) continue;
            if (C::attention_type(layer) == GemmaAttentionType::Full)
                entries_[layer].template emplace<Full>();
            else
                entries_[layer].template emplace<Local>(C::LOCAL_WINDOW);
        }
    }

    size_t tokens() const { return tokens_; }
    void advance(size_t count) {
        if (count > C::CTX - tokens_) throw std::length_error("Gemma4E4BCache: context exhausted");
        tokens_ += count;
    }

    Local& local(size_t layer) { return std::get<Local>(entries_[C::uses_shared_kv(layer) ? 22 : layer]); }
    const Local& local(size_t layer) const { return std::get<Local>(entries_[C::uses_shared_kv(layer) ? 22 : layer]); }
    Full& full(size_t layer) { return std::get<Full>(entries_[C::uses_shared_kv(layer) ? 23 : layer]); }
    const Full& full(size_t layer) const { return std::get<Full>(entries_[C::uses_shared_kv(layer) ? 23 : layer]); }

private:
    std::array<Entry, C::L> entries_;
    size_t tokens_ = 0;
};

// ---- the model -------------------------------------------------------------
//
// Gemma 4 E4B: its tensors and its math, one entity. The layer vectors are
// private and the methods are const, so the weights are immutable once loaded;
// copying is deleted because a checkpoint is gigabytes and there is never a
// reason to duplicate one. Call it with cache.evaluate(model, input).
class Gemma4E4BModel {
public:
    using C = Gemma4E4BTextConfig;
    static constexpr size_t D = C::D;
    static constexpr size_t V = C::V;
    static constexpr size_t L = C::L;
    static constexpr size_t CTX = C::CTX;
    using PrefixState = Gemma4E4BCache;
    // Local heads rotate every plane; full-attention heads rotate only their
    // first quarter (partial RoPE), which keeps a 512-wide head stable at 128K.
    using LocalRope = RotaryEmbedding<C::LOCAL_HEAD_DIM, C::LOCAL_ROPE_BASE>;
    using FullRope = RotaryEmbedding<C::FULL_HEAD_DIM, C::FULL_ROPE_BASE, 1, 4>;
    using LocalLayer = GemmaE4BLayer<C::LOCAL_HEAD_DIM, PrefixState::Local>;
    using FullLayer = GemmaE4BLayer<C::FULL_HEAD_DIM, PrefixState::Full>;
    using LocalSharedLayer = GemmaE4BSharedLayer<C::LOCAL_HEAD_DIM, PrefixState::Local>;
    using FullSharedLayer = GemmaE4BSharedLayer<C::FULL_HEAD_DIM, PrefixState::Full>;

    // One vector per layer shape, filled in schedule order, so that
    // C::position_in_group(i) is layer i's slot.
    Gemma4E4BModel(Weight<D, V> embedding, GemmaE4BModelData per_layer_inputs, std::vector<LocalLayer> local, std::vector<FullLayer> full, std::vector<LocalSharedLayer> local_shared, std::vector<FullSharedLayer> full_shared, RMSNorm<D> final_norm) : embedding_(std::move(embedding)), per_layer_inputs_(std::move(per_layer_inputs)), local_(std::move(local)), full_(std::move(full)), local_shared_(std::move(local_shared)), full_shared_(std::move(full_shared)), final_norm_(std::move(final_norm)) {
        if (local_.size() != 20 || full_.size() != 4 || local_shared_.size() != 15 || full_shared_.size() != 3) throw std::invalid_argument("Gemma4E4BModel: layer shape counts disagree with the schedule");
    }

    Gemma4E4BModel(const Gemma4E4BModel&) = delete;
    Gemma4E4BModel& operator=(const Gemma4E4BModel&) = delete;
    Gemma4E4BModel(Gemma4E4BModel&&) = default;
    Gemma4E4BModel& operator=(Gemma4E4BModel&&) = delete;  // immutable once built

    // Input embeddings, for callers that assemble multimodal sequences.
    Matrix<D> embed(std::span<const TokenId> ids) const {
        Matrix<D> rows = embedding_.gather_rows(ids);
        rows.scale(std::sqrt(Scalar(D)));  // Gemma scales embeddings on the way in
        return rows;
    }

    // Run `input` (the rows not yet in `state`) all the way to logits. How this
    // model runs is entirely inside here: the PLE table is built once, the 42
    // layers each take their slice of it, and the state advances at the end.
    Logits<V> forward(PrefixState& state, EmbeddedRows<D> input) const {
        if (input.tokens() > CTX - state.tokens()) throw std::length_error("Gemma4E4BModel: context exhausted");

        // Once per pass, not once per layer: every layer reads its own P-wide
        // slice of this L*P-wide table.
        const Matrix<C::PLE * C::L> per_layer = per_layer_inputs_(input.matrix(), input.token_ids());

        ResidualStream<D> residual(input);
        const Position conversation_position = input.conversation_position();

        // Layer 22 supplies K/V to later local layers. Keep the old local
        // window plus this prefill batch until those deferred readers finish;
        // then restore the ordinary fixed-size ring.
        state.local(22).begin_batch_retention(input.tokens());
        for (size_t layer = 0; layer < L; ++layer) residual.replace(forward_layer(state, conversation_position, residual.matrix(), per_layer, layer));
        state.local(22).end_batch_retention();
        state.advance(input.tokens());

        const auto last = final_norm_(residual.hidden(residual.tokens() - 1));
        // Capped before it is wrapped: no Logits ever exists un-softcapped.
        return Logits<V>(gemma_softcap(embedding_.matvec(last), C::LOGIT_SOFTCAP));  // tied head
    }

private:
    // Run layer `layer_index`: rows in, the next stream out. The K/V caches in
    // `state` are the only thing it mutates — the residual is a value being
    // transformed, the caches are a structure being appended to, and the caller
    // puts the result back through replace(), which is what holds a layer to
    // its row count. `conversation_position` is where these rows sit in the
    // CONVERSATION — RoPE angles and the sliding window are absolute
    // coordinates, not offsets into this batch. A layer needs nothing else of
    // the original input: the rows arrive as `X`, and the ids were spent once,
    // above, on `per_layer`.
    //
    // The equation is the selected layer type's forward() method; the schedule is
    // two plain booleans, and each branch names exactly which vector, which
    // cache and which rope table that layer type uses. Layers 24..41 own no
    // K/V: they attend against what layer 22 (local) or 23 (full) wrote earlier
    // in this same pass, which is why local()/full() redirect rather than those
    // layers holding caches.
    Matrix<D> forward_layer(PrefixState& state, Position conversation_position, MatrixView<D> X, const Matrix<C::PLE * C::L>& per_layer, size_t layer_index) const {
        const Matrix<C::PLE> ple = slice_columns<C::PLE>(per_layer.view(), layer_index);
        const size_t n = C::position_in_group(layer_index);
        const bool full = C::attention_type(layer_index) == GemmaAttentionType::Full;
        const bool shared = C::uses_shared_kv(layer_index);

        if (!shared && !full)
            return local_.at(n).forward(X, state.local(layer_index), LocalRope{}, conversation_position, ple.view());
        if (!shared)
            return full_.at(n).forward(X, state.full(layer_index), FullRope{}, conversation_position, ple.view());
        if (!full)
            return local_shared_.at(n).forward(X, state.local(layer_index), LocalRope{}, conversation_position, ple.view());
        return full_shared_.at(n).forward(X, state.full(layer_index), FullRope{}, conversation_position, ple.view());
    }

    // Tied: the same table is the input embedding and the LM head. Gemma
    // scales rows by sqrt(D) on the way in and nothing on the way out.
    Weight<D, V> embedding_;
    GemmaE4BModelData per_layer_inputs_;
    std::vector<LocalLayer> local_;              // 20 layers
    std::vector<FullLayer> full_;                //  4 (5, 11, 17, 23)
    std::vector<LocalSharedLayer> local_shared_;  // 15
    std::vector<FullSharedLayer> full_shared_;    //  3 (29, 35, 41)
    RMSNorm<D> final_norm_;
};

// ============================ Gemma 4 12B Unified ============================
//
// Pinned from google/gemma-4-12B config.json (`gemma4_unified_text`), not from
// a model card. It is a SIMPLER decoder than E4B — no PLE, no shared-KV layers
// — but it varies two things E4B holds constant, and both become types rather
// than flags:
//   * KV head count differs per attention type: 8 local, 1 full.
//   * Full-attention layers have UNIFIED K/V: one W_k and no W_v, so the value is the
//     raw projection output with the scale-free norm applied.
struct Gemma4_12BTextConfig {
    static constexpr size_t V = 262144;
    static constexpr size_t D = 3840;
    static constexpr size_t L = 48;
    static constexpr size_t Hq = 16;
    static constexpr size_t LOCAL_HEAD_DIM = 256;
    static constexpr size_t FULL_HEAD_DIM = 512;
    static constexpr size_t LOCAL_HKV = 8;
    static constexpr size_t FULL_HKV = 1;  // num_full_key_value_heads
    static constexpr size_t FF = 15360;
    static constexpr size_t CTX = 262144;
    static constexpr size_t LOCAL_WINDOW = 1024;
    static constexpr size_t LOCAL_ROPE_BASE = 10000;
    static constexpr size_t FULL_ROPE_BASE = 1000000;
    static constexpr Scalar RMS_EPS = 1e-6f;
    static constexpr Scalar LOGIT_SOFTCAP = 30.f;

    static constexpr GemmaAttentionType attention_type(size_t layer) { return layer % 6 == 5 ? GemmaAttentionType::Full : GemmaAttentionType::Local; }
    static constexpr bool uses_shared_kv(size_t) { return false; }  // 0 shared layers
    static constexpr GemmaKVKind kv_kind(size_t layer) { return attention_type(layer) == GemmaAttentionType::Full ? GemmaKVKind::Unified : GemmaKVKind::Owned; }
    static constexpr size_t kv_heads(size_t layer) { return attention_type(layer) == GemmaAttentionType::Full ? FULL_HKV : LOCAL_HKV; }
    // Which slot of its shape's weights vector layer `i` occupies.
    static constexpr size_t position_in_group(size_t layer) {
        size_t n = 0;
        for (size_t i = 0; i < layer; ++i)
            if (attention_type(i) == attention_type(layer)) ++n;
        return n;
    }
};

// The model card requires the FINAL layer to use full attention. The period-6
// rule delivers that only because 48 is a multiple of 6, so assert it rather
// than trusting the arithmetic to stay true for some future size.
static_assert(Gemma4_12BTextConfig::attention_type(Gemma4_12BTextConfig::L - 1) == GemmaAttentionType::Full, "Gemma 4 12B's last layer must use full attention");
static_assert(Gemma4_12BTextConfig::attention_type(0) == GemmaAttentionType::Local);
static_assert(Gemma4_12BTextConfig::kv_kind(5) == GemmaKVKind::Unified);
static_assert(Gemma4_12BTextConfig::kv_kind(4) == GemmaKVKind::Owned);

// params = tied embeddings + per-layer (W_q + W_k [+ W_v] + W_o + 3 FFN mats).
// Norms are excluded; they are ~0.05% and would only blur the check.
template <class C>
constexpr size_t gemma_dense_param_count() {
    size_t total = size_t(C::V) * C::D;
    for (size_t layer = 0; layer < C::L; ++layer) {
        const bool full = C::attention_type(layer) == GemmaAttentionType::Full;
        const size_t dh = full ? C::FULL_HEAD_DIM : C::LOCAL_HEAD_DIM;
        const size_t kv_mats = C::kv_kind(layer) == GemmaKVKind::Unified ? 1 : 2;
        total += C::D * C::Hq * dh;                         // W_q
        total += kv_mats * C::D * C::kv_heads(layer) * dh;  // W_k (+ W_v)
        total += C::Hq * dh * C::D;                         // W_o
        total += 3 * C::D * C::FF;                          // gate / up / down
    }
    return total;
}
static_assert(gemma_dense_param_count<Gemma4_12BTextConfig>() > 11'700'000'000 && gemma_dense_param_count<Gemma4_12BTextConfig>() < 12'100'000'000, "Gemma 4 12B lands on its advertised ~11.95B");

// The two layer shapes the 12B checkpoint actually has. They differ in head
// width (256 vs 512), KV head count (8 vs 1), and KV behavior. The methods make
// that behavior explicit: local owns separate K/V projections; full attention
// uses one projection as both K and V and therefore carries no W_v tensor.
struct Gemma12BLocalLayer {
    static constexpr size_t D = Gemma4_12BTextConfig::D;
    static constexpr size_t Hq = Gemma4_12BTextConfig::Hq;
    static constexpr size_t HEAD_DIM = Gemma4_12BTextConfig::LOCAL_HEAD_DIM;
    static constexpr size_t Hkv = Gemma4_12BTextConfig::LOCAL_HKV;
    static constexpr size_t Dh = HEAD_DIM;
    using Cache = LocalAttentionCache<Hkv, Dh>;

    RMSNorm<D> attn_norm;
    Linear<D, Hq * Dh> WQ;
    PerHeadNorm<RMSNorm<Dh>> q_norm;
    Linear<D, Hkv * Dh> WK;
    PerHeadNorm<RMSNorm<Dh>> k_norm;
    Linear<D, Hkv * Dh> WV;
    PerHeadNorm<RMSNormNoScale<Dh>> v_norm;
    Linear<Hq * Dh, D> WO;
    RMSNorm<D> post_attn_norm;
    RMSNorm<D> ffn_norm;
    GeluGatedMLP<D, Gemma4_12BTextConfig::FF> ffn;
    RMSNorm<D> post_ffn_norm;
    Scalar layer_output_scale;

    // Local attention owns separate K and V projections, appends them to its
    // fixed-window cache, and reduces over that window:
    //   score_j = dot(q_head, k_j,group)      the Q norm carries the scale
    //   alpha   = softmax(score)
    //   out     = sum_j alpha_j * v_j,group
    template <HeadRotation<Dh> Rope>
    Matrix<D> attention(MatrixView<D> X, Cache& cache, const Rope& rope, Position conversation_position) const {
        const auto Q = rope(q_norm(WQ(X)), conversation_position);

        const auto K = rope(k_norm(WK(X)), conversation_position);
        const auto V = v_norm(WV(X));

        const auto reduce = [&](MatrixView<Hq * Dh> queries, Position at) {
            std::vector<VisibleRows> visible;
            visible.reserve(queries.rows());
            for (size_t row = 0; row < queries.rows(); ++row) visible.push_back(cache.visible_rows(at + row));

            Matrix<Hq * Dh> attended = Matrix<Hq * Dh>::zero_rows(queries.rows());
            par_for(queries.rows() * Hq, [&](size_t task) {
                const size_t row = task / Hq, head = task % Hq;
                const VecView<Dh> query = slice<Dh>(queries.row(row), head);
                const KVHead group{head / (Hq / Hkv)};
                const VisibleRows rows = visible[row];

                std::vector<Scalar> alpha;
                alpha.reserve(rows.count());
                for (size_t j = rows.first; j < rows.end; ++j) alpha.push_back(dot(query, cache.key(j, group)));
                alpha = Softmax{}(std::move(alpha));

                Vec<Dh> head_output;
                for (size_t n = 0; n < alpha.size(); ++n) head_output.scaled_add(cache.value(rows.first + n, group), alpha[n]);
                attended.template replace_partition<Dh>(row, head, head_output);
            });
            return attended;
        };

        // A 1024-wide ring evicts on any prompt longer than the window, so the
        // row-by-row path is the normal long-prefill case, not an edge.
        Matrix<Hq * Dh> A;
        if (cache.can_append_without_eviction(Q.rows())) {
            for (size_t row = 0; row < Q.rows(); ++row) cache.append(conversation_position + row, K.row(row), V.row(row));
            A = reduce(Q.view(), conversation_position);
        } else {
            A.reserve(Q.rows());
            for (size_t row = 0; row < Q.rows(); ++row) {
                const Position at = conversation_position + row;
                cache.append(at, K.row(row), V.row(row));
                A.append(reduce(Q.view().single_row(row), at).row(0));
            }
        }
        return WO(A.view());
    }

    template <HeadRotation<Dh> Rope>
    Matrix<D> forward(MatrixView<D> X, Cache& cache, const Rope& rope, Position conversation_position) const {
        const auto U = attn_norm(X);
        const auto A = post_attn_norm(attention(U.view(), cache, rope, conversation_position));
        Matrix<D> H = X + A;

        const auto Z = ffn_norm(H.view());
        const auto F = post_ffn_norm(ffn(Z.view()));
        H = H + F;

        H.scale(layer_output_scale);
        return H;
    }
};

struct Gemma12BFullLayer {
    static constexpr size_t D = Gemma4_12BTextConfig::D;
    static constexpr size_t Hq = Gemma4_12BTextConfig::Hq;
    static constexpr size_t HEAD_DIM = Gemma4_12BTextConfig::FULL_HEAD_DIM;
    static constexpr size_t Hkv = Gemma4_12BTextConfig::FULL_HKV;
    static constexpr size_t Dh = HEAD_DIM;
    using Cache = FullAttentionCache<Hkv, Dh>;

    RMSNorm<D> attn_norm;
    Linear<D, Hq * Dh> WQ;
    PerHeadNorm<RMSNorm<Dh>> q_norm;
    Linear<D, Hkv * Dh> WK;                 // one projection...
    PerHeadNorm<RMSNorm<Dh>> k_norm;        // ...the key's learned norm
    PerHeadNorm<RMSNormNoScale<Dh>> v_norm;  // ...and the value's scale-free one
    Linear<Hq * Dh, D> WO;
    RMSNorm<D> post_attn_norm;
    RMSNorm<D> ffn_norm;
    GeluGatedMLP<D, Gemma4_12BTextConfig::FF> ffn;
    RMSNorm<D> post_ffn_norm;
    Scalar layer_output_scale;

    // Unified K/V: project once. K gets its learned norm and RoPE; V gets the
    // scale-free norm directly from the same unrotated projection — which is
    // why this is the one attention that must copy: two different norms read
    // the same projected rows, so neither can consume them.
    template <HeadRotation<Dh> Rope>
    Matrix<D> attention(MatrixView<D> X, Cache& cache, const Rope& rope, Position conversation_position) const {
        const auto Q = rope(q_norm(WQ(X)), conversation_position);

        Matrix<Hkv * Dh> V = WK(X);
        const auto K = rope(k_norm(V), conversation_position);  // same projection, distinct K path
        V = v_norm(std::move(V));

        // Once the unified projection has become K and V, the reduction is the
        // ordinary one:
        //   score_j = dot(q_head, k_j,group)
        //   alpha   = softmax(score)
        //   out     = sum_j alpha_j * v_j,group
        const auto reduce = [&](MatrixView<Hq * Dh> queries, Position at) {
            std::vector<VisibleRows> visible;
            visible.reserve(queries.rows());
            for (size_t row = 0; row < queries.rows(); ++row) visible.push_back(cache.visible_rows(at + row));

            Matrix<Hq * Dh> attended = Matrix<Hq * Dh>::zero_rows(queries.rows());
            par_for(queries.rows() * Hq, [&](size_t task) {
                const size_t row = task / Hq, head = task % Hq;
                const VecView<Dh> query = slice<Dh>(queries.row(row), head);
                const KVHead group{head / (Hq / Hkv)};
                const VisibleRows rows = visible[row];

                std::vector<Scalar> alpha;
                alpha.reserve(rows.count());
                for (size_t j = rows.first; j < rows.end; ++j) alpha.push_back(dot(query, cache.key(j, group)));
                alpha = Softmax{}(std::move(alpha));

                Vec<Dh> head_output;
                for (size_t n = 0; n < alpha.size(); ++n) head_output.scaled_add(cache.value(rows.first + n, group), alpha[n]);
                attended.template replace_partition<Dh>(row, head, head_output);
            });
            return attended;
        };

        for (size_t row = 0; row < Q.rows(); ++row) cache.append(conversation_position + row, K.row(row), V.row(row));
        const auto A = reduce(Q.view(), conversation_position);
        return WO(A.view());
    }

    template <HeadRotation<Dh> Rope>
    Matrix<D> forward(MatrixView<D> X, Cache& cache, const Rope& rope, Position conversation_position) const {
        const auto U = attn_norm(X);
        const auto A = post_attn_norm(attention(U.view(), cache, rope, conversation_position));
        Matrix<D> H = X + A;

        const auto Z = ffn_norm(H.view());
        const auto F = post_ffn_norm(ffn(Z.view()));
        H = H + F;

        H.scale(layer_output_scale);
        return H;
    }
};


// Every layer owns a cache whose type states its invariant: local layers own a
// fixed-window ring; full layers own append-only history.
class Gemma4_12BCache {
    using C = Gemma4_12BTextConfig;
    using Local = LocalAttentionCache<C::LOCAL_HKV, C::LOCAL_HEAD_DIM>;
    using Full = FullAttentionCache<C::FULL_HKV, C::FULL_HEAD_DIM>;
    using Entry = std::variant<std::monostate, Local, Full>;

public:
    Gemma4_12BCache() {
        for (size_t layer = 0; layer < C::L; ++layer) {
            if (C::attention_type(layer) == GemmaAttentionType::Full)
                entries_[layer].template emplace<Full>();
            else
                entries_[layer].template emplace<Local>(C::LOCAL_WINDOW);
        }
    }

    size_t tokens() const { return tokens_; }
    void advance(size_t count) {
        if (count > C::CTX - tokens_) throw std::length_error("Gemma4_12BCache: context exhausted");
        tokens_ += count;
    }

    Local& local(size_t layer) { return std::get<Local>(entries_[layer]); }
    Full& full(size_t layer) { return std::get<Full>(entries_[layer]); }

private:
    std::array<Entry, C::L> entries_;
    size_t tokens_ = 0;
};

// Gemma 4 12B: same shape of entity as E4B, minus the PLE tensors.
class Gemma4_12BModel {
public:
    using C = Gemma4_12BTextConfig;
    static constexpr size_t D = C::D;
    static constexpr size_t V = C::V;
    static constexpr size_t L = C::L;
    static constexpr size_t CTX = C::CTX;
    using PrefixState = Gemma4_12BCache;
    using LocalRope = RotaryEmbedding<C::LOCAL_HEAD_DIM, C::LOCAL_ROPE_BASE>;
    using FullRope = RotaryEmbedding<C::FULL_HEAD_DIM, C::FULL_ROPE_BASE, 1, 4>;

    Gemma4_12BModel(Weight<D, V> embedding, std::vector<Gemma12BLocalLayer> local, std::vector<Gemma12BFullLayer> full, RMSNorm<D> final_norm) : embedding_(std::move(embedding)), local_(std::move(local)), full_(std::move(full)), final_norm_(std::move(final_norm)) {
        if (local_.size() != 40 || full_.size() != 8) throw std::invalid_argument("Gemma4_12BModel: layer shape counts disagree with the schedule");
    }

    Gemma4_12BModel(const Gemma4_12BModel&) = delete;
    Gemma4_12BModel& operator=(const Gemma4_12BModel&) = delete;
    Gemma4_12BModel(Gemma4_12BModel&&) = default;
    Gemma4_12BModel& operator=(Gemma4_12BModel&&) = delete;

    Matrix<D> embed(std::span<const TokenId> ids) const {
        Matrix<D> rows = embedding_.gather_rows(ids);
        rows.scale(std::sqrt(Scalar(D)));  // Gemma scales embeddings on the way in
        return rows;
    }

    // The same shape of pass as E4B, with nothing to precompute: 12B has no PLE.
    Logits<V> forward(PrefixState& state, EmbeddedRows<D> input) const {
        if (input.tokens() > CTX - state.tokens()) throw std::length_error("Gemma4_12BModel: context exhausted");

        ResidualStream<D> residual(input);
        const Position conversation_position = input.conversation_position();
        for (size_t layer = 0; layer < L; ++layer) residual.replace(forward_layer(state, conversation_position, residual.matrix(), layer));
        state.advance(input.tokens());

        const auto last = final_norm_(residual.hidden(residual.tokens() - 1));
        // Capped before it is wrapped: no Logits ever exists un-softcapped.
        return Logits<V>(gemma_softcap(embedding_.matvec(last), C::LOGIT_SOFTCAP));  // tied head
    }

private:
    // Each concrete 12B layer owns its equation. Unlike E4B's methods these
    // methods have no per-layer-input argument and perform no PLE residual.
    Matrix<D> forward_layer(PrefixState& state, Position conversation_position, MatrixView<D> X, size_t layer_index) const {
        const size_t n = C::position_in_group(layer_index);
        return C::attention_type(layer_index) == GemmaAttentionType::Local ? local_.at(n).forward(X, state.local(layer_index), LocalRope{}, conversation_position) : full_.at(n).forward(X, state.full(layer_index), FullRope{}, conversation_position);
    }

    // Tied, and scaled by sqrt(D) on the way in, exactly as in E4B.
    Weight<D, V> embedding_;
    std::vector<Gemma12BLocalLayer> local_;  // 40 layers
    std::vector<Gemma12BFullLayer> full_;    //  8 (every 6th)
    RMSNorm<D> final_norm_;
};
