// transformer.hpp - model-neutral transformer assembly and execution spine.
//
// A concrete architecture supplies the model-specific operations (embedding,
// one layer, prefix-state policy, and output head). This file owns the invariant
// pipeline from the architecture map:
//
//   embedded token sequence [T,D] -> residual stream -> L layers -> logits[V]
//
// Immutable weights live in the callable Transformer. PrefixCache is explicit
// memoized work for a verified input prefix. Neither abstraction knows whether
// a layer uses dense FFN or MoE, local or global attention, owned or shared KV,
// or an auxiliary residual.
#pragma once
#include "multimodal.hpp"
#include <concepts>
#include <functional>
#include <optional>
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

// The mutable [T,D] stream carried through the transformer stack.  Modalities
// disappear at this boundary: text, image, and audio encoders must all produce
// decoder-width embeddings before a ResidualStream is constructed.
template <size_t D>
class ResidualStream {
public:
    explicit ResidualStream(const EmbeddedSequence<D>& input) : values_(copy(input.matrix())) {}

    size_t tokens() const { return values_.rows(); }
    MatrixView<D> matrix() const { return values_.view(); }
    MutableMatrixView<D> matrix_mut() { return values_.mutable_view(); }
    VecView<D> token(size_t i) const { return values_.row(i); }
    MutVecView<D> token_mut(size_t i) { return values_.row_mut(i); }
    void set_token(size_t i, VecView<D> value) { values_.set_row(i, value); }
    void set_token(size_t i, const Vec<D>& value) { values_.set_row(i, value); }
    void set_matrix(Matrix<D> values) {
        if (values.rows() != values_.rows()) throw std::invalid_argument("ResidualStream: matrix row mismatch");
        values_ = std::move(values);
    }

private:
    TokenMatrix<D> values_;
};

template <class TokenIO, size_t D, size_t V>
concept TokenInputOutput = requires(const TokenIO& token_io, TokenId id, std::span<const TokenId> ids, VecView<D> hidden) {
    { token_io.token(id) } -> std::same_as<Vec<D>>;
    { token_io.tokens(ids) } -> std::same_as<Matrix<D>>;
    { token_io.logits(hidden) } -> std::same_as<Vec<V>>;
};

// Shared text frontend: model families differ in the embedding operation, not
// in how token IDs, identities, positions, and modality spans are assembled.
template <size_t D, class EmbedTokens>
EmbeddedSequence<D> embed_text_tokens(std::span<const TokenId> ids, size_t first_position, EmbedTokens&& embed_tokens) {
    if (ids.empty()) throw std::invalid_argument("transformer: empty token sequence");
    Matrix<D> embeddings = std::invoke(std::forward<EmbedTokens>(embed_tokens), ids);
    if (embeddings.rows() != ids.size()) throw std::invalid_argument("transformer: embedding row count mismatch");
    std::vector<TokenId> identities(ids.begin(), ids.end());
    std::vector<EmbeddingSegment<D>> segments;
    segments.emplace_back(std::move(embeddings), std::move(identities));
    return compose_embeddings(std::move(segments), first_position);
}

// Immutable parameters shared by every evaluation of a transformer. TokenIO
// owns the input embedding/output-head relationship (including tied weights),
// Layer may itself be a variant for heterogeneous stacks, and ArchitectureData
// contains immutable model-wide policy weights such as RoPE or Gemma PLE.
template <size_t D_, size_t V_, size_t L_, class TokenIO_, class Layer_, class FinalNorm_, class ArchitectureData_ = NoArchitectureData, class LayerSchedule_ = AnyLayerSchedule>
class TransformerWeights {
public:
    static constexpr size_t D = D_;
    static constexpr size_t V = V_;
    static constexpr size_t L = L_;
    using TokenIO = TokenIO_;
    using Layer = Layer_;
    using FinalNorm = FinalNorm_;
    using ArchitectureData = ArchitectureData_;
    using LayerSchedule = LayerSchedule_;

    static_assert(TokenInputOutput<TokenIO, D, V>, "TokenIO must embed token IDs and project hidden states to logits");

    TransformerWeights(TokenIO token_io, std::vector<Layer> layers, FinalNorm final_norm, ArchitectureData architecture_data = {}) : token_io_(std::move(token_io)), layers_(std::move(layers)), final_norm_(std::move(final_norm)), architecture_data_(std::move(architecture_data)) {
        if (layers_.size() != L) throw std::invalid_argument("TransformerWeights: wrong layer count");
        for (size_t layer = 0; layer < layers_.size(); ++layer) LayerSchedule::validate(layers_[layer], layer);
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

// Architecture is a stateless policy that binds immutable weights to derived
// prefix state, input preparation, layer operation, and output operation. Keeping
// these as explicit operations prevents a flat collection of anatomy booleans
// from pretending that all transformer layers have the same physical shape.
template <class A>
concept TransformerArchitecture = requires(const typename A::Weights& weights, typename A::PrefixState& prefix_state, const EmbeddedSequence<A::D>& input, ResidualStream<A::D>& residual, typename A::PreparedInput& prepared, std::span<const TokenId> token_ids, size_t layer) {
    requires A::D == A::Weights::D;
    requires A::V == A::Weights::V;
    requires A::L == A::Weights::L;
    { A::make_prefix_state() } -> std::same_as<typename A::PrefixState>;
    { A::prefix_tokens(prefix_state) } -> std::convertible_to<size_t>;
    { A::embed(weights, token_ids, size_t{}) } -> std::same_as<EmbeddedSequence<A::D>>;
    { A::prepare(weights, input) } -> std::same_as<typename A::PreparedInput>;
    { A::forward_layer(weights, prefix_state, input, residual, prepared, layer) } -> std::same_as<void>;
    { A::advance_prefix(prefix_state, size_t{}) } -> std::same_as<void>;
    { A::output(weights, residual) } -> std::same_as<Vec<A::V>>;
};

template <class Architecture>
    requires TransformerArchitecture<Architecture>
Vec<Architecture::V> evaluate_transformer_suffix(const typename Architecture::Weights& weights, typename Architecture::PrefixState& prefix_state, const EmbeddedSequence<Architecture::D>& input) {
    if (input.tokens() == 0) throw std::invalid_argument("transformer: empty input");
    const size_t used = Architecture::prefix_tokens(prefix_state);
    if (input.position(0).i != used) throw std::invalid_argument("transformer: cache/input prefix mismatch");
    if (input.tokens() > Architecture::CTX - used) throw std::length_error("transformer: context exhausted");

    ResidualStream<Architecture::D> residual(input);
    typename Architecture::PreparedInput prepared = Architecture::prepare(weights, input);
    for (size_t layer = 0; layer < Architecture::L; ++layer){
         Architecture::forward_layer(weights, prefix_state, input, residual, prepared, layer);
    }
    Architecture::advance_prefix(prefix_state, input.tokens());
    return Architecture::output(weights, residual);
}

template <class Architecture>
    requires TransformerArchitecture<Architecture>
class Transformer;

// Memoized intermediates for one verified model/input prefix. This object is
// derived: it can change evaluation cost, never the result for an input.
template <class Architecture>
    requires TransformerArchitecture<Architecture>
class PrefixCache {
public:
    PrefixCache() : prefix_state_(Architecture::make_prefix_state()) {}

    size_t cached_tokens() const { return cached_tokens_; }
    size_t reused_tokens() const { return reused_tokens_; }
    size_t computed_tokens() const { return computed_tokens_; }

    void clear() {
        weights_ = nullptr;
        prefix_state_ = Architecture::make_prefix_state();
        input_kind_ = InputKind::None;
        token_prefix_.clear();
        embedded_sequence_id_ = 0;
        cached_tokens_ = 0;
        last_logits_.reset();
        reused_tokens_ = 0;
        computed_tokens_ = 0;
    }

private:
    enum class InputKind { None, TokenIds, EmbeddedSequence };

    void bind(const typename Architecture::Weights& weights) {
        clear();
        weights_ = &weights;
    }

    const typename Architecture::Weights* weights_ = nullptr;
    typename Architecture::PrefixState prefix_state_;
    InputKind input_kind_ = InputKind::None;
    std::vector<TokenId> token_prefix_;
    uint64_t embedded_sequence_id_ = 0;
    size_t cached_tokens_ = 0;
    std::optional<Vec<Architecture::V>> last_logits_;
    size_t reused_tokens_ = 0;
    size_t computed_tokens_ = 0;

    friend class Transformer<Architecture>;
};

// T_weights(input, prefix_cache): an immutable, callable transformer with an
// explicit evaluator memo keyed by the complete input prefix.
template <class Architecture>
    requires TransformerArchitecture<Architecture>
class Transformer {
public:
    using Weights = typename Architecture::Weights;
    static constexpr size_t D = Architecture::D;
    static constexpr size_t V = Architecture::V;
    static constexpr size_t L = Architecture::L;
    static constexpr size_t CTX = Architecture::CTX;

    explicit Transformer(Weights weights) : weights_(std::move(weights)) {}

    const Weights& weights() const { return weights_; }

    Vec<V> operator()(std::span<const TokenId> complete_input, PrefixCache<Architecture>& memo) const {
        if (complete_input.empty()) throw std::invalid_argument("Transformer: empty input");
        if (complete_input.size() > CTX) throw std::length_error("Transformer: context exhausted");

        const bool extends_cached_prefix = memo.weights_ == &weights_ && memo.input_kind_ == PrefixCache<Architecture>::InputKind::TokenIds && memo.token_prefix_.size() <= complete_input.size() && std::equal(memo.token_prefix_.begin(), memo.token_prefix_.end(), complete_input.begin());
        if (!extends_cached_prefix) memo.bind(weights_);

        const size_t reused = memo.cached_tokens_;
        begin_evaluation(memo, reused, complete_input.size());
        if (reused == complete_input.size()) return cached_logits(memo);

        try {
            EmbeddedSequence<D> suffix = Architecture::embed(weights_, complete_input.subspan(reused), reused);
            Vec<V> logits = evaluate_transformer_suffix<Architecture>(weights_, memo.prefix_state_, suffix);
            memo.input_kind_ = PrefixCache<Architecture>::InputKind::TokenIds;
            memo.token_prefix_.assign(complete_input.begin(), complete_input.end());
            commit_evaluation(memo, complete_input.size(), logits);
            return logits;
        } catch (...) {
            memo.clear();
            throw;
        }
    }

    Vec<V> operator()(const EmbeddedSequence<D>& complete_input, PrefixCache<Architecture>& memo) const {
        if (complete_input.tokens() == 0) throw std::invalid_argument("Transformer: empty input");
        if (complete_input.first_position() != 0) throw std::invalid_argument("Transformer: expected a complete prefix at position zero");
        if (complete_input.tokens() > CTX) throw std::length_error("Transformer: context exhausted");

        const bool extends_cached_prefix = memo.weights_ == &weights_ && memo.input_kind_ == PrefixCache<Architecture>::InputKind::EmbeddedSequence && memo.embedded_sequence_id_ == complete_input.sequence_id() && memo.cached_tokens_ <= complete_input.tokens();
        if (!extends_cached_prefix) memo.bind(weights_);

        const size_t reused = memo.cached_tokens_;
        begin_evaluation(memo, reused, complete_input.tokens());
        if (reused == complete_input.tokens()) return cached_logits(memo);

        try {
            Vec<V> logits = [&] {
                if (reused == 0) return evaluate_transformer_suffix<Architecture>(weights_, memo.prefix_state_, complete_input);
                EmbeddedSequence<D> suffix = complete_input.suffix(reused);
                return evaluate_transformer_suffix<Architecture>(weights_, memo.prefix_state_, suffix);
            }();
            memo.input_kind_ = PrefixCache<Architecture>::InputKind::EmbeddedSequence;
            memo.embedded_sequence_id_ = complete_input.sequence_id();
            commit_evaluation(memo, complete_input.tokens(), logits);
            return logits;
        } catch (...) {
            memo.clear();
            throw;
        }
    }

private:
    static void begin_evaluation(PrefixCache<Architecture>& memo, size_t reused, size_t complete_tokens) {
        memo.reused_tokens_ = reused;
        memo.computed_tokens_ = complete_tokens - reused;
    }

    static Vec<V> cached_logits(const PrefixCache<Architecture>& memo) {
        if (!memo.last_logits_) throw std::logic_error("Transformer: prefix cache has no logits");
        return copy(VecView<V>(*memo.last_logits_));
    }

    static void commit_evaluation(PrefixCache<Architecture>& memo, size_t complete_tokens, const Vec<V>& logits) {
        memo.cached_tokens_ = complete_tokens;
        memo.last_logits_ = copy(VecView<V>(logits));
    }

    Weights weights_;
};
