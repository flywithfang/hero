// transformer.hpp - model-neutral transformer assembly and execution spine.
//
// A concrete architecture supplies the model-specific operations (embedding,
// one layer, cache policy, and output head).  This file owns the invariant
// pipeline from the architecture map:
//
//   embedded token sequence [T,D] -> residual stream -> L layers -> logits[V]
//
// Immutable weights live in TransformerModel.  Mutable KV/cache state lives in
// TransformerSession.  Neither class knows whether a layer uses dense FFN or
// MoE, local or global attention, owned or shared KV, or an auxiliary residual.
#pragma once
#include "multimodal.hpp"
#include <concepts>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

struct NoArchitectureData {};
struct NoPreparedInput {};
struct NoLayerInput {};
struct AnyLayerSchedule {
    template <class Layer>
    static void validate(const Layer&, size_t) {}
};

// Normalization/residual topology belongs to a branch, not to "Transformer"
// globally.  This represents both classic Pre-Norm (NoPostTransform) and the
// Gemma-style pre+post normalized branch without anatomy booleans.
struct NoPostTransform {
    template <size_t D>
    void apply(Vec<D>&) const {}
};

template <class Norm>
class PostNormalize {
public:
    explicit PostNormalize(Norm norm) : norm_(std::move(norm)) {}

    template <size_t D>
    void apply(Vec<D>& value) const {
        value = norm_(value);
    }

private:
    Norm norm_;
};

template <size_t D, class InputNorm_, class Operation_,
          class PostTransform_ = NoPostTransform>
class ResidualBranch {
public:
    using InputNorm = InputNorm_;
    using Operation = Operation_;
    using PostTransform = PostTransform_;

    ResidualBranch(InputNorm input_norm, Operation operation,
                   PostTransform post_transform = {})
        : input_norm_(std::move(input_norm)), operation_(std::move(operation)),
          post_transform_(std::move(post_transform)) {}

    Vec<D> normalize(VecView<D> input) const { return input_norm_(input); }
    const Operation& operation() const { return operation_; }

    template <class Runner>
    Vec<D> finish(VecView<D> residual, VecView<D> normalized,
                  Runner&& run) const {
        Vec<D> branch = std::forward<Runner>(run)(operation_, normalized);
        post_transform_.apply(branch);
        branch += residual;
        return branch;
    }

    template <class Runner>
    Vec<D> forward(VecView<D> input, Runner&& run) const {
        Vec<D> normalized = normalize(input);
        return finish(input, normalized, std::forward<Runner>(run));
    }

private:
    InputNorm input_norm_;
    Operation operation_;
    [[no_unique_address]] PostTransform post_transform_;
};

template <size_t D>
struct IdentityLayerTail {
    Vec<D> operator()(VecView<D> hidden, NoLayerInput = {}) const {
        return copy(hidden);
    }
};

// A canonical decoder block is communication across tokens followed by a
// per-token channel transform.  Tail represents optional architecture-specific
// work after those branches (for example Gemma E4B's PLE residual and scale).
template <size_t D, class TokenMixerBranch_, class ChannelMixerBranch_,
          class Tail_ = IdentityLayerTail<D>>
class TransformerBlock {
public:
    using TokenMixerBranch = TokenMixerBranch_;
    using ChannelMixerBranch = ChannelMixerBranch_;
    using Tail = Tail_;

    TransformerBlock(TokenMixerBranch token_mixer,
                     ChannelMixerBranch channel_mixer,
                     Tail tail = {})
        : token_mixer_(std::move(token_mixer)),
          channel_mixer_(std::move(channel_mixer)),
          tail_(std::move(tail)) {}

    const TokenMixerBranch& token_mixer_branch() const { return token_mixer_; }
    const ChannelMixerBranch& channel_mixer_branch() const { return channel_mixer_; }
    const Tail& tail() const { return tail_; }

    template <class LayerInput, class TokenMixerRunner, class ChannelMixerRunner>
    Vec<D> forward(VecView<D> input, const LayerInput& layer_input,
                   TokenMixerRunner&& run_token_mixer,
                   ChannelMixerRunner&& run_channel_mixer) const {
        Vec<D> communicated = token_mixer_.forward(
            input, std::forward<TokenMixerRunner>(run_token_mixer));
        Vec<D> transformed = channel_mixer_.forward(
            communicated, std::forward<ChannelMixerRunner>(run_channel_mixer));
        return tail_(transformed, layer_input);
    }

private:
    TokenMixerBranch token_mixer_;
    ChannelMixerBranch channel_mixer_;
    [[no_unique_address]] Tail tail_;
};

// The mutable [T,D] stream carried through the transformer stack.  Modalities
// disappear at this boundary: text, image, and audio encoders must all produce
// decoder-width embeddings before a ResidualStream is constructed.
template <size_t D>
class ResidualStream {
public:
    explicit ResidualStream(const EmbeddedSequence<D>& input)
        : values_(input.tokens()) {
        for (size_t token = 0; token < input.tokens(); ++token)
            values_.set_row(token, input.embedding(token));
    }

    size_t tokens() const { return values_.rows(); }
    VecView<D> token(size_t i) const { return values_.row(i); }
    MutVecView<D> token_mut(size_t i) { return values_.row_mut(i); }
    void set_token(size_t i, VecView<D> value) { values_.set_row(i, value); }
    void set_token(size_t i, const Vec<D>& value) { values_.set_row(i, value); }

private:
    TokenMatrix<D> values_;
};

template <class TokenIO, size_t D, size_t V>
concept TokenInputOutput = requires(const TokenIO& token_io, TokenId id,
                                    VecView<D> hidden) {
    { token_io.token(id) } -> std::same_as<Vec<D>>;
    { token_io.logits(hidden) } -> std::same_as<Vec<V>>;
};

// Shared text frontend: model families differ in the embedding operation, not
// in how token IDs, identities, positions, and modality spans are assembled.
template <size_t D, class EmbedToken>
EmbeddedSequence<D> embed_text_tokens(std::span<const TokenId> ids,
                                      size_t first_position,
                                      EmbedToken&& embed_token) {
    if (ids.empty())
        throw std::invalid_argument("transformer: empty token sequence");
    TokenMatrix<D> embeddings(ids.size());
    std::vector<TokenId> identities(ids.begin(), ids.end());
    for (size_t token = 0; token < ids.size(); ++token)
        embeddings.set_row(token, std::invoke(embed_token, ids[token]));
    std::vector<EmbeddingSegment<D>> segments;
    segments.emplace_back(std::move(embeddings), std::move(identities));
    return compose_embeddings(std::move(segments), first_position);
}

// Immutable physical assembly shared by all transformer families.  TokenIO
// owns the input embedding/output-head relationship (including tied weights),
// Layer may itself be a variant for heterogeneous stacks, and ArchitectureData
// contains immutable model-wide policy weights such as RoPE or Gemma PLE.
template <size_t D_, size_t V_, size_t L_, class TokenIO_, class Layer_,
          class FinalNorm_, class ArchitectureData_ = NoArchitectureData,
          class LayerSchedule_ = AnyLayerSchedule>
class TransformerModel {
public:
    static constexpr size_t D = D_;
    static constexpr size_t V = V_;
    static constexpr size_t L = L_;
    using TokenIO = TokenIO_;
    using Layer = Layer_;
    using FinalNorm = FinalNorm_;
    using ArchitectureData = ArchitectureData_;
    using LayerSchedule = LayerSchedule_;

    static_assert(TokenInputOutput<TokenIO, D, V>,
                  "TokenIO must embed token IDs and project hidden states to logits");

    TransformerModel(TokenIO token_io, std::vector<Layer> layers,
                     FinalNorm final_norm, ArchitectureData architecture_data = {})
        : token_io_(std::move(token_io)), layers_(std::move(layers)),
          final_norm_(std::move(final_norm)),
          architecture_data_(std::move(architecture_data)) {
        if (layers_.size() != L)
            throw std::invalid_argument("TransformerModel: wrong layer count");
        for (size_t layer = 0; layer < layers_.size(); ++layer)
            LayerSchedule::validate(layers_[layer], layer);
    }

    const TokenIO& token_io() const { return token_io_; }
    const Layer& layer(size_t i) const { return layers_.at(i); }
    const FinalNorm& final_norm() const { return final_norm_; }
    const ArchitectureData& architecture_data() const { return architecture_data_; }

private:
    TokenIO token_io_;
    std::vector<Layer> layers_;
    FinalNorm final_norm_;
    [[no_unique_address]] ArchitectureData architecture_data_;
};

// Architecture is a stateless policy that binds a TransformerModel to its
// cache, input preparation, layer operation, and output operation.  Keeping
// these as explicit operations prevents a flat collection of anatomy booleans
// from pretending that all transformer layers have the same physical shape.
template <class A>
concept TransformerArchitecture = requires(
    const typename A::Model& model,
    typename A::State& state,
    const EmbeddedSequence<A::D>& input,
    ResidualStream<A::D>& residual,
    typename A::PreparedInput& prepared,
    std::span<const TokenId> token_ids,
    size_t layer) {
    requires A::D == A::Model::D;
    requires A::V == A::Model::V;
    requires A::L == A::Model::L;
    { A::make_state() } -> std::same_as<typename A::State>;
    { A::tokens(state) } -> std::convertible_to<size_t>;
    { A::embed(model, token_ids, size_t{}) }
        -> std::same_as<EmbeddedSequence<A::D>>;
    { A::prepare(model, input) } -> std::same_as<typename A::PreparedInput>;
    { A::forward_layer(model, state, input, residual, prepared, layer) }
        -> std::same_as<void>;
    { A::advance(state, size_t{}) } -> std::same_as<void>;
    { A::output(model, residual) } -> std::same_as<Vec<A::V>>;
};

template <class Architecture>
    requires TransformerArchitecture<Architecture>
Vec<Architecture::V> transformer_forward(
    const typename Architecture::Model& model,
    typename Architecture::State& state,
    const EmbeddedSequence<Architecture::D>& input) {
    if (input.tokens() == 0)
        throw std::invalid_argument("transformer: empty input");
    const size_t used = Architecture::tokens(state);
    if (input.position(0).i != used)
        throw std::invalid_argument("transformer: discontinuous input positions");
    if (input.tokens() > Architecture::CTX - used)
        throw std::length_error("transformer: context exhausted");

    ResidualStream<Architecture::D> residual(input);
    typename Architecture::PreparedInput prepared = Architecture::prepare(model, input);
    for (size_t layer = 0; layer < Architecture::L; ++layer)
        Architecture::forward_layer(model, state, input, residual, prepared, layer);
    Architecture::advance(state, input.tokens());
    return Architecture::output(model, residual);
}

// Common autoregressive lifecycle for decoder-only architectures.  A session
// references one immutable model and exclusively owns all mutable inference
// state.  It accepts both token IDs and already assembled multimodal sequences.
template <class Architecture>
    requires TransformerArchitecture<Architecture>
class TransformerSession {
public:
    using Model = typename Architecture::Model;
    using State = typename Architecture::State;

    explicit TransformerSession(const Model& model)
        : model_(model), state_(Architecture::make_state()) {}

    size_t context_used() const { return Architecture::tokens(state_); }
    size_t context_left() const { return Architecture::CTX - context_used(); }
    void reset() { state_ = Architecture::make_state(); }

    Vec<Architecture::V> prefill(std::span<const TokenId> tokens) {
        return forward(Architecture::embed(model_, tokens, context_used()));
    }

    Vec<Architecture::V> step(TokenId token) {
        return prefill(std::span<const TokenId>(&token, 1));
    }

    Vec<Architecture::V> forward(const EmbeddedSequence<Architecture::D>& input) {
        return transformer_forward<Architecture>(model_, state_, input);
    }

private:
    const Model& model_;
    State state_;
};
