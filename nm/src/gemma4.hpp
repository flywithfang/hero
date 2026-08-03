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

    static constexpr GemmaAttentionKind attention_kind(size_t layer) { return layer % 6 == 5 ? GemmaAttentionKind::Full : GemmaAttentionKind::Sliding; }
    static constexpr bool shares_kv(size_t layer) { return layer >= FIRST_KV_SHARED_LAYER; }
    // The final non-sharing layer of each attention kind supplies all later
    // layers of that kind: local layer 22, full layer 23 for E4B's 42 layers.
    static constexpr bool stores_shared_kv(size_t layer) {
        if (layer >= FIRST_KV_SHARED_LAYER) return false;
        const GemmaAttentionKind kind = attention_kind(layer);
        for (size_t next = layer + 1; next < FIRST_KV_SHARED_LAYER; ++next)
            if (attention_kind(next) == kind) return false;
        return true;
    }
    // Which slot of its shape's weights vector layer `i` occupies: the count
    // of earlier layers with the same (attention kind, KV ownership).
    static constexpr size_t position_in_kind(size_t layer) {
        size_t n = 0;
        for (size_t i = 0; i < layer; ++i)
            if (attention_kind(i) == attention_kind(layer) && shares_kv(i) == shares_kv(layer)) ++n;
        return n;
    }
};

static_assert(Gemma4E4BTextConfig::attention_kind(4) == GemmaAttentionKind::Sliding);
static_assert(Gemma4E4BTextConfig::attention_kind(5) == GemmaAttentionKind::Full);
static_assert(Gemma4E4BTextConfig::FIRST_KV_SHARED_LAYER == 24);
static_assert(Gemma4E4BTextConfig::stores_shared_kv(22));
static_assert(Gemma4E4BTextConfig::stores_shared_kv(23));
static_assert(Gemma4E4BTextConfig::shares_kv(24));
static_assert(Gemma4E4BTextConfig::position_in_kind(0) == 0 && Gemma4E4BTextConfig::position_in_kind(6) == 5, "sliding-own layers count their own kind only");
static_assert(Gemma4E4BTextConfig::position_in_kind(5) == 0 && Gemma4E4BTextConfig::position_in_kind(23) == 3, "full-own layers are 5, 11, 17, 23");
static_assert(Gemma4E4BTextConfig::position_in_kind(24) == 0 && Gemma4E4BTextConfig::position_in_kind(29) == 0 && Gemma4E4BTextConfig::position_in_kind(35) == 1, "shared layers restart their count at layer 24");

// What a layer physically holds for K/V is a property of the checkpoint.
// Three cases exist across Gemma 4, and each is expressed by which members a
// layer struct HAS — there is no kind flag at runtime and no kind type:
//   Owned    W_k and W_v both present (E4B's first 24 layers, 12B's sliding)
//   Unified  W_k only, V = normalized raw W_k output (12B's global layers)
//   Shared   neither; the layer attends against what an earlier layer wrote
// This enum survives only as config/loader vocabulary (kv_kind(), the
// parameter count); the layer structs themselves just have or lack tensors.
enum class GemmaKVKind : uint8_t { Owned, Unified, Shared };

inline Scalar gemma_softcap(Scalar value, Scalar cap) { return cap * std::tanh(value / cap); }

template <size_t N>
void gemma_softcap(MutVecView<N> values, Scalar cap) {
    for (size_t i = 0; i < N; ++i) values[i] = gemma_softcap(values[i], cap);
}

// The per-layer part of E4B's PLE residual. Model-level token/context PLE
// construction is separate because multimodal soft tokens have no token id.
template <size_t D, size_t P>
class GemmaPerLayerResidual {
public:
    GemmaPerLayerResidual(Linear<D, P> gate, Linear<P, D> projection, RMSNorm<D> post_norm) : gate_(std::move(gate)), projection_(std::move(projection)), post_norm_(std::move(post_norm)) {}

    Vec<D> operator()(VecView<D> hidden, VecView<P> per_layer_input) const {
        Vec<P> gated = gate_(hidden);
        Gelu::apply(gated);
        Vec<P> branch_input = hadamard(VecView<P>(gated), per_layer_input);
        Vec<D> branch = projection_(branch_input);
        branch = post_norm_(branch);
        branch += hidden;
        return branch;
    }
    Matrix<D> operator()(MatrixView<D> hidden, MatrixView<P> per_layer_input) const {
        Matrix<P> gated = gate_(hidden);
        Gelu::apply(gated);
        Matrix<D> branch = projection_(hadamard(gated.view(), per_layer_input));
        branch = post_norm_(branch);
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
        projection_norm_.apply(context);  // partitions are layers here

        return scaled_sum(VecView<Packed>(context), VecView<Packed>(token), combination_scale_);
    }
    Matrix<Packed> operator()(MatrixView<D> input_embeddings, std::span<const TokenId> identities) const {
        if (input_embeddings.rows() != identities.size()) throw std::invalid_argument("GemmaPerLayerInputs: embedding/identity row mismatch");

        Matrix<Packed> token = token_embeddings_.gather_rows(identities);
        token.scale(token_scale_);

        Matrix<Packed> context = model_projection_(input_embeddings);
        context.scale(projection_scale_);
        projection_norm_.apply(context);

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
// Two head widths exist (sliding 256, full 512), and the last 18 layers carry
// no K/V tensors at all — so there are two structs, each instantiated at two
// widths. Each type implements the operation appropriate to its physical
// tensors; no member-detection convention decides its behavior elsewhere.
template <size_t Dh>
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
    Matrix<D> attention(MatrixView<D> X, KVCache<Hkv, Dh>& cache, const Rope& rope, Position conversation_position, size_t window) const {
        Matrix<Hq * Dh> Q = WQ(X);
        q_norm.apply(Q);
        rotate_heads<Hq, Dh>(Q, rope, conversation_position);

        Matrix<Hkv * Dh> K = WK(X);
        k_norm.apply(K);
        rotate_heads<Hkv, Dh>(K, rope, conversation_position);

        Matrix<Hkv * Dh> V = WV(X);
        v_norm.apply(V);

        const auto reduce = [&](MatrixView<Hq * Dh> queries, Position at) {
            std::vector<VisibleRows> visible;
            visible.reserve(queries.rows());
            for (size_t row = 0; row < queries.rows(); ++row) visible.push_back(cache.visible_rows(at + row, CausalWindow{window}));

            Matrix<Hq * Dh> attended = Matrix<Hq * Dh>::zero_rows(queries.rows());
            par_for(queries.rows() * Hq, [&](size_t task) {
                const size_t row = task / Hq, head = task % Hq;
                const VecView<Dh> query = slice<Dh>(queries.row(row), head);
                const KVHead group{head / (Hq / Hkv)};
                const VisibleRows rows = visible[row];

                std::vector<Scalar> alpha;
                alpha.reserve(rows.count());
                for (size_t j = rows.first; j < rows.end; ++j) alpha.push_back(dot(query, cache.key(j, group)));
                Softmax::apply(alpha);

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
    Matrix<D> forward(MatrixView<D> X, KVCache<Hkv, Dh>& cache, const Rope& rope, Position conversation_position, size_t window, MatrixView<Gemma4E4BTextConfig::PLE> per_layer_input) const {
        const Matrix<D> U = attn_norm(X);
        const Matrix<D> A = post_attn_norm(attention(U.view(), cache, rope, conversation_position, window));
        Matrix<D> H = X + A;

        Matrix<D> Z = ffn_norm(H.view());
        Matrix<D> F = post_ffn_norm(ffn(Z.view()));
        H = H + F;

        H = ple(H.view(), per_layer_input);
        H.scale(layer_output_scale);
        return H;
    }
};

// The last 18 layers: the same layer minus the K/V tensors, which the
// checkpoint does not carry. Not an optional, not a flag — the members
// do not exist, so misusing them is a compile error.
template <size_t Dh>
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
    Matrix<D> attention(MatrixView<D> X, const KVCache<Hkv, Dh>& cache, const Rope& rope, Position conversation_position, size_t window) const {
        Matrix<Hq * Dh> Q = WQ(X);
        q_norm.apply(Q);
        rotate_heads<Hq, Dh>(Q, rope, conversation_position);

        std::vector<VisibleRows> visible;
        visible.reserve(Q.rows());
        for (size_t row = 0; row < Q.rows(); ++row) visible.push_back(cache.visible_rows(conversation_position + row, CausalWindow{window}));

        Matrix<Hq * Dh> A = Matrix<Hq * Dh>::zero_rows(Q.rows());
        par_for(Q.rows() * Hq, [&](size_t task) {
            const size_t row = task / Hq, head = task % Hq;
            const VecView<Dh> query = slice<Dh>(Q.row(row), head);
            const KVHead group{head / (Hq / Hkv)};
            const VisibleRows rows = visible[row];

            std::vector<Scalar> alpha;
            alpha.reserve(rows.count());
            for (size_t j = rows.first; j < rows.end; ++j) alpha.push_back(dot(query, cache.key(j, group)));
            Softmax::apply(alpha);

            Vec<Dh> head_output;
            for (size_t n = 0; n < alpha.size(); ++n) head_output.scaled_add(cache.value(rows.first + n, group), alpha[n]);
            A.template replace_partition<Dh>(row, head, head_output);
        });
        return WO(A.view());
    }

    template <HeadRotation<Dh> Rope>
    Matrix<D> forward(MatrixView<D> X, const KVCache<Hkv, Dh>& cache, const Rope& rope, Position conversation_position, size_t window, MatrixView<Gemma4E4BTextConfig::PLE> per_layer_input) const {
        Matrix<D> U = attn_norm(X);
        Matrix<D> A = attention(U.view(), cache, rope, conversation_position, window);
        A = post_attn_norm(A);
        Matrix<D> H = X + A;

        Matrix<D> Z = ffn_norm(H.view());
        Matrix<D> F = post_ffn_norm(ffn(Z.view()));
        H = H + F;

        H = ple(H.view(), per_layer_input);
        H.scale(layer_output_scale);
        return H;
    }
};

using GemmaE4BModelData = GemmaPerLayerInputs<Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::PLE, Gemma4E4BTextConfig::L, Gemma4E4BTextConfig::V>;

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
        if (count > C::CTX - tokens_) throw std::length_error("Gemma4E4BCache: context exhausted");
        tokens_ += count;
    }

    Local& local(size_t layer) { return std::get<Local>(entries_[C::shares_kv(layer) ? 22 : layer]); }
    const Local& local(size_t layer) const { return std::get<Local>(entries_[C::shares_kv(layer) ? 22 : layer]); }
    Global& global(size_t layer) { return std::get<Global>(entries_[C::shares_kv(layer) ? 23 : layer]); }
    const Global& global(size_t layer) const { return std::get<Global>(entries_[C::shares_kv(layer) ? 23 : layer]); }

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
    // Local heads rotate every plane; global heads rotate only the first
    // quarter of theirs (partial RoPE), which is what lets a 512-wide global
    // head stay stable at 128K context.
    using LocalRope = RotaryEmbedding<C::LOCAL_HEAD_DIM, C::LOCAL_ROPE_BASE>;
    using GlobalRope = RotaryEmbedding<C::GLOBAL_HEAD_DIM, C::GLOBAL_ROPE_BASE, 1, 4>;
    using SlidingLayer = GemmaE4BLayer<C::LOCAL_HEAD_DIM>;
    using GlobalLayer = GemmaE4BLayer<C::GLOBAL_HEAD_DIM>;
    using SlidingSharedLayer = GemmaE4BSharedLayer<C::LOCAL_HEAD_DIM>;
    using GlobalSharedLayer = GemmaE4BSharedLayer<C::GLOBAL_HEAD_DIM>;

    // One vector per layer shape, filled in schedule order, so that
    // C::position_in_kind(i) is layer i's slot.
    Gemma4E4BModel(Weight<D, V> embedding, GemmaE4BModelData per_layer_inputs, std::vector<SlidingLayer> sliding, std::vector<GlobalLayer> global, std::vector<SlidingSharedLayer> sliding_shared, std::vector<GlobalSharedLayer> global_shared, RMSNorm<D> final_norm) : embedding_(std::move(embedding)), per_layer_inputs_(std::move(per_layer_inputs)), sliding_(std::move(sliding)), global_(std::move(global)), sliding_shared_(std::move(sliding_shared)), global_shared_(std::move(global_shared)), final_norm_(std::move(final_norm)) {
        if (sliding_.size() != 20 || global_.size() != 4 || sliding_shared_.size() != 15 || global_shared_.size() != 3) throw std::invalid_argument("Gemma4E4BModel: layer shape counts disagree with the schedule");
    }

    Gemma4E4BModel(const Gemma4E4BModel&) = delete;
    Gemma4E4BModel& operator=(const Gemma4E4BModel&) = delete;
    Gemma4E4BModel(Gemma4E4BModel&&) = default;
    Gemma4E4BModel& operator=(Gemma4E4BModel&&) = delete;  // immutable once built

    // Input embeddings, for callers that assemble multimodal sequences.
    Matrix<D> tokens(std::span<const TokenId> ids) const {
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
        for (size_t layer = 0; layer < L; ++layer) forward_layer(state, conversation_position, residual, per_layer, layer);
        state.advance(input.tokens());

        Vec<D> last = final_norm_(residual.token(residual.tokens() - 1));
        Logits<V> logits(embedding_.matvec(last));  // tied head
        gemma_softcap<V>(logits.mutable_view(), C::LOGIT_SOFTCAP);
        return logits;
    }

private:
    // Run layer `layer_index` in place: `residual` is the only in/out, and
    // `conversation_position` is where its rows sit in the CONVERSATION — RoPE angles
    // and the sliding window are absolute coordinates, not offsets into this
    // batch. A layer needs nothing else of the original input: the rows arrive
    // through `residual`, and the ids were spent once, above, on `per_layer`.
    //
    // The equation is the selected layer type's forward() method; the schedule is
    // two plain booleans, and each branch names exactly which vector, which
    // cache, which rope table and which window that layer kind uses. Layers
    // 24..41 own no K/V: they attend against what layer 22 (sliding) or 23
    // (full) wrote earlier in this same pass, which is why local()/global()
    // redirect rather than those layers holding a cache.
    void forward_layer(PrefixState& state, Position conversation_position, ResidualStream<D>& residual, const Matrix<C::PLE * C::L>& per_layer, size_t layer_index) const {
        const Matrix<C::PLE> ple = slice_columns<C::PLE>(per_layer.view(), layer_index);
        const MatrixView<D> X = residual.matrix();
        const size_t n = C::position_in_kind(layer_index);
        const bool full = C::attention_kind(layer_index) == GemmaAttentionKind::Full;
        const bool shared = C::shares_kv(layer_index);

        Matrix<D> next = [&]() -> Matrix<D> {
            if (!shared && !full)
             return sliding_.at(n).forward(X, state.local(layer_index), LocalRope{}, conversation_position, C::SLIDING_WINDOW, ple.view());
            if (!shared)
             return global_.at(n).forward(X, state.global(layer_index), GlobalRope{}, conversation_position, /*window=*/0, ple.view());
            if (!full) 
                return sliding_shared_.at(n).forward(X, state.local(layer_index), LocalRope{}, conversation_position, C::SLIDING_WINDOW, ple.view());
            return global_shared_.at(n).forward(X, state.global(layer_index), GlobalRope{}, conversation_position, /*window=*/0, ple.view());
        }();
        residual.set_matrix(std::move(next));
    }

    // Tied: the same table is the input embedding and the LM head. Gemma
    // scales rows by sqrt(D) on the way in and nothing on the way out.
    Weight<D, V> embedding_;
    GemmaE4BModelData per_layer_inputs_;
    std::vector<SlidingLayer> sliding_;              // 20 layers
    std::vector<GlobalLayer> global_;                //  4 (5, 11, 17, 23)
    std::vector<SlidingSharedLayer> sliding_shared_;  // 15
    std::vector<GlobalSharedLayer> global_shared_;    //  3 (29, 35, 41)
    RMSNorm<D> final_norm_;
};

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
    static constexpr size_t GLOBAL_HKV = 1;  // num_global_key_value_heads
    static constexpr size_t FF = 15360;
    static constexpr size_t CTX = 262144;
    static constexpr size_t SLIDING_WINDOW = 1024;
    static constexpr size_t LOCAL_ROPE_BASE = 10000;
    static constexpr size_t GLOBAL_ROPE_BASE = 1000000;
    static constexpr Scalar RMS_EPS = 1e-6f;
    static constexpr Scalar LOGIT_SOFTCAP = 30.f;

    static constexpr GemmaAttentionKind attention_kind(size_t layer) { return layer % 6 == 5 ? GemmaAttentionKind::Full : GemmaAttentionKind::Sliding; }
    static constexpr bool shares_kv(size_t) { return false; }  // 0 shared layers
    static constexpr GemmaKVKind kv_kind(size_t layer) { return attention_kind(layer) == GemmaAttentionKind::Full ? GemmaKVKind::Unified : GemmaKVKind::Owned; }
    static constexpr size_t kv_heads(size_t layer) { return attention_kind(layer) == GemmaAttentionKind::Full ? GLOBAL_HKV : LOCAL_HKV; }
    // Which slot of its shape's weights vector layer `i` occupies.
    static constexpr size_t position_in_kind(size_t layer) {
        size_t n = 0;
        for (size_t i = 0; i < layer; ++i)
            if (attention_kind(i) == attention_kind(layer)) ++n;
        return n;
    }
};

// The model card requires the FINAL layer to be global. The period-6 rule
// delivers that only because 48 is a multiple of 6, so assert it rather than
// trusting the arithmetic to stay true for some future size.
static_assert(Gemma4_12BTextConfig::attention_kind(Gemma4_12BTextConfig::L - 1) == GemmaAttentionKind::Full, "Gemma 4 12B's last layer must be a global-attention layer");
static_assert(Gemma4_12BTextConfig::attention_kind(0) == GemmaAttentionKind::Sliding);
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
static_assert(gemma_dense_param_count<Gemma4_12BTextConfig>() > 11'700'000'000 && gemma_dense_param_count<Gemma4_12BTextConfig>() < 12'100'000'000, "Gemma 4 12B lands on its advertised ~11.95B");

// The two layer shapes the 12B checkpoint actually has. They differ in head
// width (256 vs 512), KV head count (8 vs 1), and KV behavior. The methods make
// that behavior explicit: sliding owns separate K/V projections; global uses
// one projection as both K and V and therefore carries no W_v tensor.
struct Gemma12BSlidingLayer {
    static constexpr size_t D = Gemma4_12BTextConfig::D;
    static constexpr size_t Hq = Gemma4_12BTextConfig::Hq;
    static constexpr size_t HEAD_DIM = Gemma4_12BTextConfig::LOCAL_HEAD_DIM;
    static constexpr size_t Hkv = Gemma4_12BTextConfig::LOCAL_HKV;
    static constexpr size_t Dh = HEAD_DIM;

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

    // Sliding attention owns separate K and V projections, appends them to its
    // bounded cache, and reduces over the window:
    //   score_j = dot(q_head, k_j,group)      the Q norm carries the scale
    //   alpha   = softmax(score)
    //   out     = sum_j alpha_j * v_j,group
    template <HeadRotation<Dh> Rope>
    Matrix<D> attention(MatrixView<D> X, KVCache<Hkv, Dh>& cache, const Rope& rope, Position conversation_position, size_t window) const {
        Matrix<Hq * Dh> Q = WQ(X);
        q_norm.apply(Q);
        rotate_heads<Hq, Dh>(Q, rope, conversation_position);

        Matrix<Hkv * Dh> K = WK(X);
        k_norm.apply(K);
        rotate_heads<Hkv, Dh>(K, rope, conversation_position);

        Matrix<Hkv * Dh> V = WV(X);
        v_norm.apply(V);

        const auto reduce = [&](MatrixView<Hq * Dh> queries, Position at) {
            std::vector<VisibleRows> visible;
            visible.reserve(queries.rows());
            for (size_t row = 0; row < queries.rows(); ++row) visible.push_back(cache.visible_rows(at + row, CausalWindow{window}));

            Matrix<Hq * Dh> attended = Matrix<Hq * Dh>::zero_rows(queries.rows());
            par_for(queries.rows() * Hq, [&](size_t task) {
                const size_t row = task / Hq, head = task % Hq;
                const VecView<Dh> query = slice<Dh>(queries.row(row), head);
                const KVHead group{head / (Hq / Hkv)};
                const VisibleRows rows = visible[row];

                std::vector<Scalar> alpha;
                alpha.reserve(rows.count());
                for (size_t j = rows.first; j < rows.end; ++j) alpha.push_back(dot(query, cache.key(j, group)));
                Softmax::apply(alpha);

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
    Matrix<D> forward(MatrixView<D> X, KVCache<Hkv, Dh>& cache, const Rope& rope, Position conversation_position, size_t window) const {
        Matrix<D> U = attn_norm(X);
        Matrix<D> A = attention(U.view(), cache, rope, conversation_position, window);
        A = post_attn_norm(A);
        Matrix<D> H = X + A;

        Matrix<D> Z = ffn_norm(H.view());
        Matrix<D> F = post_ffn_norm(ffn(Z.view()));
        H = H + F;

        H.scale(layer_output_scale);
        return H;
    }
};

struct Gemma12BGlobalLayer {
    static constexpr size_t D = Gemma4_12BTextConfig::D;
    static constexpr size_t Hq = Gemma4_12BTextConfig::Hq;
    static constexpr size_t HEAD_DIM = Gemma4_12BTextConfig::GLOBAL_HEAD_DIM;
    static constexpr size_t Hkv = Gemma4_12BTextConfig::GLOBAL_HKV;
    static constexpr size_t Dh = HEAD_DIM;

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
    Matrix<D> attention(MatrixView<D> X, KVCache<Hkv, Dh>& cache, const Rope& rope, Position conversation_position, size_t window) const {
        Matrix<Hq * Dh> Q = WQ(X);
        q_norm.apply(Q);
        rotate_heads<Hq, Dh>(Q, rope, conversation_position);

        Matrix<Hkv * Dh> V = WK(X);
        Matrix<Hkv * Dh> K = V;  // same projection, rotated as K and left unrotated as V
        k_norm.apply(K);
        rotate_heads<Hkv, Dh>(K, rope, conversation_position);
        v_norm.apply(V);

        // Once the unified projection has become K and V, the reduction is the
        // ordinary one:
        //   score_j = dot(q_head, k_j,group)
        //   alpha   = softmax(score)
        //   out     = sum_j alpha_j * v_j,group
        const auto reduce = [&](MatrixView<Hq * Dh> queries, Position at) {
            std::vector<VisibleRows> visible;
            visible.reserve(queries.rows());
            for (size_t row = 0; row < queries.rows(); ++row) visible.push_back(cache.visible_rows(at + row, CausalWindow{window}));

            Matrix<Hq * Dh> attended = Matrix<Hq * Dh>::zero_rows(queries.rows());
            par_for(queries.rows() * Hq, [&](size_t task) {
                const size_t row = task / Hq, head = task % Hq;
                const VecView<Dh> query = slice<Dh>(queries.row(row), head);
                const KVHead group{head / (Hq / Hkv)};
                const VisibleRows rows = visible[row];

                std::vector<Scalar> alpha;
                alpha.reserve(rows.count());
                for (size_t j = rows.first; j < rows.end; ++j) alpha.push_back(dot(query, cache.key(j, group)));
                Softmax::apply(alpha);

                Vec<Dh> head_output;
                for (size_t n = 0; n < alpha.size(); ++n) head_output.scaled_add(cache.value(rows.first + n, group), alpha[n]);
                attended.template replace_partition<Dh>(row, head, head_output);
            });
            return attended;
        };

        // Global layers retain everything, so the eviction path never fires
        // here; the branch stays because the cache, not the layer, decides.
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
    Matrix<D> forward(MatrixView<D> X, KVCache<Hkv, Dh>& cache, const Rope& rope, Position conversation_position, size_t window) const {
        Matrix<D> U = attn_norm(X);
        Matrix<D> A = attention(U.view(), cache, rope, conversation_position, window);
        A = post_attn_norm(A);
        Matrix<D> H = X + A;

        Matrix<D> Z = ffn_norm(H.view());
        Matrix<D> F = post_ffn_norm(ffn(Z.view()));
        H = H + F;

        H.scale(layer_output_scale);
        return H;
    }
};


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
        if (count > C::CTX - tokens_) throw std::length_error("Gemma4_12BCache: context exhausted");
        tokens_ += count;
    }

    Local& local(size_t layer) { return std::get<Local>(entries_[layer]); }
    Global& global(size_t layer) { return std::get<Global>(entries_[layer]); }

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
    using GlobalRope = RotaryEmbedding<C::GLOBAL_HEAD_DIM, C::GLOBAL_ROPE_BASE, 1, 4>;

    Gemma4_12BModel(Weight<D, V> embedding, std::vector<Gemma12BSlidingLayer> sliding, std::vector<Gemma12BGlobalLayer> global, RMSNorm<D> final_norm) : embedding_(std::move(embedding)), sliding_(std::move(sliding)), global_(std::move(global)), final_norm_(std::move(final_norm)) {
        if (sliding_.size() != 40 || global_.size() != 8) throw std::invalid_argument("Gemma4_12BModel: layer shape counts disagree with the schedule");
    }

    Gemma4_12BModel(const Gemma4_12BModel&) = delete;
    Gemma4_12BModel& operator=(const Gemma4_12BModel&) = delete;
    Gemma4_12BModel(Gemma4_12BModel&&) = default;
    Gemma4_12BModel& operator=(Gemma4_12BModel&&) = delete;

    Matrix<D> tokens(std::span<const TokenId> ids) const {
        Matrix<D> rows = embedding_.gather_rows(ids);
        rows.scale(std::sqrt(Scalar(D)));  // Gemma scales embeddings on the way in
        return rows;
    }

    // The same shape of pass as E4B, with nothing to precompute: 12B has no PLE.
    Logits<V> forward(PrefixState& state, EmbeddedRows<D> input) const {
        if (input.tokens() > CTX - state.tokens()) throw std::length_error("Gemma4_12BModel: context exhausted");

        ResidualStream<D> residual(input);
        const Position conversation_position = input.conversation_position();
        for (size_t layer = 0; layer < L; ++layer) forward_layer(state, conversation_position, residual, layer);
        state.advance(input.tokens());

        Vec<D> last = final_norm_(residual.token(residual.tokens() - 1));
        Logits<V> logits(embedding_.matvec(last));  // tied head
        gemma_softcap<V>(logits.mutable_view(), C::LOGIT_SOFTCAP);
        return logits;
    }

private:
    // Each concrete 12B layer owns its equation. Unlike E4B's methods these
    // methods have no per-layer-input argument and perform no PLE residual.
    void forward_layer(PrefixState& state, Position conversation_position, ResidualStream<D>& residual, size_t layer_index) const {
        const MatrixView<D> X = residual.matrix();
        const size_t n = C::position_in_kind(layer_index);

        Matrix<D> next = C::attention_kind(layer_index) == GemmaAttentionKind::Sliding ? sliding_.at(n).forward(X, state.local(layer_index), LocalRope{}, conversation_position, C::SLIDING_WINDOW) : global_.at(n).forward(X, state.global(layer_index), GlobalRope{}, conversation_position, /*window=*/0);
        residual.set_matrix(std::move(next));
    }

    // Tied, and scaled by sqrt(D) on the way in, exactly as in E4B.
    Weight<D, V> embedding_;
    std::vector<Gemma12BSlidingLayer> sliding_;  // 40 layers
    std::vector<Gemma12BGlobalLayer> global_;    //  8 (every 6th)
    RMSNorm<D> final_norm_;
};
