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
    explicit ResidualStream(EmbeddedRows<D> input) : values_(copy(input.matrix())) {}

    size_t tokens() const { return values_.rows(); }
    MatrixView<D> matrix() const { return values_.view(); }
    VecView<D> token(size_t i) const { return values_.row(i); }
    void set_matrix(Matrix<D> values) {
        if (values.rows() != values_.rows()) throw std::invalid_argument("ResidualStream: matrix row mismatch");
        values_ = std::move(values);
    }

private:
    Matrix<D> values_;
};

// What a MODEL is. A model is one entity: its tensors and its math together,
// immutable and non-copyable. It owes the outside world exactly two things —
// turn token ids into decoder-width rows, and run rows to logits:
//
//     tokens(ids)           -> Matrix<D>
//     forward(state, rows)  -> Logits<V>
//
// Note what it does NOT owe: a way to build a whole EmbeddedSequence. Assembling
// one is the caller's job — append the rows it made from ids, append the rows an
// encoder made from pixels.
//
// How a model runs — how many layers, in what order, what it precomputes once
// per pass, whether it even HAS layers — is its own business and appears
// nowhere in this file. `PrefixState` is the model's carried state (K/V caches,
// recurrent state). The ONE thing outside code may ask it is how many rows it
// holds; everything else about it is the model's business, and it advances only
// by being passed to forward().
//
// This concept generates no code — it exists so a mistake in a new model
// reports itself here instead of somewhere 200 lines deep in an instantiation.
template <class M>
concept TransformerModel = requires(const M& model, typename M::PrefixState& prefix_state, EmbeddedRows<M::D> input, std::span<const TokenId> ids) {
    requires !std::is_copy_constructible_v<M>;  // weights are immutable and never copied
    typename M::PrefixState;
    { prefix_state.tokens() } -> std::convertible_to<size_t>;
    { model.tokens(ids) } -> std::same_as<Matrix<M::D>>;
    { model.forward(prefix_state, input) } -> std::same_as<Logits<M::V>>;
    { M::CTX } -> std::convertible_to<size_t>;
};

// One conversation's carried state, for one model.
//
// The MODEL owns what is in that state — K/V rows, recurrent matrices — and is
// the only thing that ever writes to it. Which is why `PrefixState` has no
// interface to speak of: the one question outside code asks it is `tokens()`,
// how many rows of the conversation it already covers, and there is deliberately
// nothing else. No add, no query, no clear — a state advances only by being
// handed to `forward`, and is forgotten by being replaced with a fresh one.
//
// What this class owns is the one question the model cannot answer: may the
// carried state be reused for the input in hand, or must it be thrown away
// first? That rule is identical for every family, which is why it lives here
// once rather than inside each model.
//
// Lifecycle, plainly: constructed against the model whose state it carries and
// never rebound, cleared to forget everything derived, destroyed with whatever
// owns the conversation.
template <TransformerModel Model>
class PrefixCache {
public:
    explicit PrefixCache(const Model& model) : model_(model) {}

    // The one public entry point: reuse the carried state when this input
    // genuinely extends the conversation already computed, otherwise start over.
    //
    // The input is always an EmbeddedSequence, never token ids. Text is not a
    // special case of anything — it is the case where every row happened to
    // come from the vocabulary — and an image row has no id even in principle.
    //
    // "Extends" means object identity plus length: a sequence keeps its id
    // across append(), so a conversation that grows hits the cache, and any
    // freshly built sequence gets a new id and starts from zero.
    Logits<Model::V> evaluate(const EmbeddedSequence<Model::D>& complete_input) {
        if (complete_input.tokens() == 0) throw std::invalid_argument("evaluate: empty input");
        if (complete_input.tokens() > Model::CTX) throw std::length_error("evaluate: context exhausted");

        // A different sequence, or one that somehow shrank: nothing carried is
        // worth keeping.
        if (embedded_sequence_id_ != complete_input.sequence_id() || cached_tokens() > complete_input.tokens()) clear();

        const size_t reused = cached_tokens();
        reused_tokens_ = reused;
        computed_tokens_ = complete_input.tokens() - reused;
        if (reused == complete_input.tokens()) {
            if (!last_logits_) throw std::logic_error("evaluate: prefix cache has no logits");
            return last_logits_->copy();
        }

        try {
            // The rows this cache has not computed: the whole prompt on the
            // first call, one token per call after that. Nothing is copied —
            // it is a view of the tail — and the model neither knows nor cares
            // which case it is in.
            const EmbeddedRows<Model::D> uncached = complete_input.from_row(reused);
            Logits<Model::V> logits = model_.forward(prefix_state_, uncached);
            if (cached_tokens() != reused + uncached.tokens()) throw std::logic_error("evaluate: the model did not advance its state by the rows it was given");
            embedded_sequence_id_ = complete_input.sequence_id();
            last_logits_ = logits.copy();
            return logits;
        } catch (...) {
            clear();
            throw;
        }
    }

    // How many rows the carried state covers. This is the MODEL's own count:
    // there is no second copy of it here to drift out of agreement.
    size_t cached_tokens() const { return prefix_state_.tokens(); }
    // How the last evaluate() split its work. The only way to observe that the
    // cache did its job at all — matching logits prove nothing, since they match
    // whether or not a single row was reused.
    size_t reused_tokens() const { return reused_tokens_; }
    size_t computed_tokens() const { return computed_tokens_; }

    void clear() {
        prefix_state_ = typename Model::PrefixState{};
        embedded_sequence_id_ = 0;
        last_logits_.reset();
        reused_tokens_ = 0;
        computed_tokens_ = 0;
    }

private:
    const Model& model_;
    typename Model::PrefixState prefix_state_;
    uint64_t embedded_sequence_id_ = 0;
    std::optional<Logits<Model::V>> last_logits_;
    size_t reused_tokens_ = 0;
    size_t computed_tokens_ = 0;
};
