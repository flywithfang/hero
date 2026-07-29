// transformer.hpp - the residual stream, and prefix-cached evaluation.
//
// This file does NOT define how a transformer runs. A model runs itself: it
// owns its tensors and states its own equations, and `forward(state, input)`
// is the whole of what it exposes. What lives here is the one thing that is
// genuinely the same for every model and is about CACHING rather than math —
// reuse the carried state when an input extends the prefix already computed
// for the same model, and clear it when it does not.
//
// So: no layer loop, no "prepare" step, no output head, no idea whether a
// model has layers at all.
#pragma once
#include "logits.hpp"
#include "multimodal.hpp"
#include <concepts>
#include <functional>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

struct NoPreparedInput {};

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
    { token_io.logits(hidden) } -> std::same_as<Logits<V>>;
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

// What a MODEL is. A model is one entity: its tensors and its math together,
// immutable and non-copyable. It owes the outside world exactly two things —
// turn tokens into embeddings, and run a suffix to logits:
//
//     embed(token_ids, first_position) -> EmbeddedSequence<D>
//     forward(state, input)            -> Logits<V>
//
// How a model runs — how many layers, in what order, what it precomputes once
// per pass, whether it even HAS layers — is its own business and appears
// nowhere in this file. `PrefixState` is the model's carried state (K/V caches,
// recurrent state); the only thing outside code asks of it is how many tokens
// it holds, so a cached prefix can be matched against an input.
//
// This concept generates no code — it exists so a mistake in a new model
// reports itself here instead of somewhere 200 lines deep in an instantiation.
template <class M>
concept TransformerModel = requires(const M& model, typename M::PrefixState& prefix_state, const EmbeddedSequence<M::D>& input, std::span<const TokenId> token_ids) {
    requires !std::is_copy_constructible_v<M>;  // weights are immutable and never copied
    typename M::PrefixState;
    { prefix_state.tokens() } -> std::convertible_to<size_t>;
    { model.embed(token_ids, size_t{}) } -> std::same_as<EmbeddedSequence<M::D>>;
    { model.forward(prefix_state, input) } -> std::same_as<Logits<M::V>>;
};

// A suffix must begin exactly where the carried state ends. This is the cache
// contract, not a step in anyone's pipeline — the model is simply handed the
// rows it has not seen yet.
template <class Model>
    requires TransformerModel<Model>
Logits<Model::V> run_suffix(const Model& model, typename Model::PrefixState& prefix_state, const EmbeddedSequence<Model::D>& input) {
    if (input.tokens() == 0) throw std::invalid_argument("transformer: empty input");
    if (input.position(0).i != prefix_state.tokens()) throw std::invalid_argument("transformer: cache/input prefix mismatch");
    return model.forward(prefix_state, input);
}

// Memoized intermediates for one verified model/input prefix. This object is
// derived: it can change evaluation cost, never the result for an input.
template <class Model>
    requires TransformerModel<Model>
class PrefixCache {
public:
    PrefixCache() = default;

    size_t cached_tokens() const { return cached_tokens_; }
    size_t reused_tokens() const { return reused_tokens_; }
    size_t computed_tokens() const { return computed_tokens_; }

    void clear() {
        model_ = nullptr;
        prefix_state_ = typename Model::PrefixState{};
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

    void bind(const Model& model) {
        clear();
        model_ = &model;
    }

    const Model* model_ = nullptr;
    typename Model::PrefixState prefix_state_;
    InputKind input_kind_ = InputKind::None;
    std::vector<TokenId> token_prefix_;
    uint64_t embedded_sequence_id_ = 0;
    size_t cached_tokens_ = 0;
    std::optional<Logits<Model::V>> last_logits_;
    size_t reused_tokens_ = 0;
    size_t computed_tokens_ = 0;

    template <class M>
        requires TransformerModel<M>
    friend Logits<M::V> evaluate(const M&, std::span<const TokenId>, PrefixCache<M>&);
    template <class M>
        requires TransformerModel<M>
    friend Logits<M::V> evaluate(const M&, const EmbeddedSequence<M::D>&, PrefixCache<M>&);
};

// evaluate(model, complete_input, memo) — the public entry point, and the only
// behavior here beyond the model's own math: reuse the cached state when the
// input genuinely extends the cached prefix of the SAME model, otherwise clear
// and recompute. That is identical for every family, which is why it is one
// function here instead of a copy inside each model.
template <class Model>
    requires TransformerModel<Model>
Logits<Model::V> evaluate(const Model& model, std::span<const TokenId> complete_input, PrefixCache<Model>& memo) {
    if (complete_input.empty()) throw std::invalid_argument("evaluate: empty input");
    if (complete_input.size() > Model::CTX) throw std::length_error("evaluate: context exhausted");

    const bool extends_cached_prefix = memo.model_ == &model && memo.input_kind_ == PrefixCache<Model>::InputKind::TokenIds && memo.token_prefix_.size() <= complete_input.size() && std::equal(memo.token_prefix_.begin(), memo.token_prefix_.end(), complete_input.begin());
    if (!extends_cached_prefix) memo.bind(model);

    const size_t reused = memo.cached_tokens_;
    memo.reused_tokens_ = reused;
    memo.computed_tokens_ = complete_input.size() - reused;
    if (reused == complete_input.size()) {
        if (!memo.last_logits_) throw std::logic_error("evaluate: prefix cache has no logits");
        return memo.last_logits_->copy();
    }

    try {
        EmbeddedSequence<Model::D> suffix = model.embed(complete_input.subspan(reused), reused);
        Logits<Model::V> logits = run_suffix(model, memo.prefix_state_, suffix);
        memo.input_kind_ = PrefixCache<Model>::InputKind::TokenIds;
        memo.token_prefix_.assign(complete_input.begin(), complete_input.end());
        memo.cached_tokens_ = complete_input.size();
        memo.last_logits_ = logits.copy();
        return logits;
    } catch (...) {
        memo.clear();
        throw;
    }
}

// The multimodal entry: the caller already produced decoder-width embeddings
// (text + image + audio segments), so there is nothing to tokenize.
template <class Model>
    requires TransformerModel<Model>
Logits<Model::V> evaluate(const Model& model, const EmbeddedSequence<Model::D>& complete_input, PrefixCache<Model>& memo) {
    if (complete_input.tokens() == 0) throw std::invalid_argument("evaluate: empty input");
    if (complete_input.first_position() != 0) throw std::invalid_argument("evaluate: expected a complete prefix at position zero");
    if (complete_input.tokens() > Model::CTX) throw std::length_error("evaluate: context exhausted");

    const bool extends_cached_prefix = memo.model_ == &model && memo.input_kind_ == PrefixCache<Model>::InputKind::EmbeddedSequence && memo.embedded_sequence_id_ == complete_input.sequence_id() && memo.cached_tokens_ <= complete_input.tokens();
    if (!extends_cached_prefix) memo.bind(model);

    const size_t reused = memo.cached_tokens_;
    memo.reused_tokens_ = reused;
    memo.computed_tokens_ = complete_input.tokens() - reused;
    if (reused == complete_input.tokens()) {
        if (!memo.last_logits_) throw std::logic_error("evaluate: prefix cache has no logits");
        return memo.last_logits_->copy();
    }

    try {
        Logits<Model::V> logits = [&] {
            if (reused == 0) return run_suffix(model, memo.prefix_state_, complete_input);
            EmbeddedSequence<Model::D> suffix = complete_input.suffix(reused);
            return run_suffix(model, memo.prefix_state_, suffix);
        }();
        memo.input_kind_ = PrefixCache<Model>::InputKind::EmbeddedSequence;
        memo.embedded_sequence_id_ = complete_input.sequence_id();
        memo.cached_tokens_ = complete_input.tokens();
        memo.last_logits_ = logits.copy();
        return logits;
    } catch (...) {
        memo.clear();
        throw;
    }
}
