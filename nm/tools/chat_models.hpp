// chat_models.hpp — the chat adapter, and the factories that bind it to a
// family. An adapter owns its chat template, prefix memoization, sampling, and
// any modality encoder; the REPL sees only what ChatModelInterface names.
#pragma once
#include "chat_session.hpp"
#include "../src/gemma4_loader.hpp"
#include "../src/gemma4_vision_loader.hpp"
#include "../src/qwen35_loader.hpp"
#include "../src/runtime.hpp"
#include "../src/tokenizer.hpp"
#include <chrono>
#include <memory>

// ONE adapter, for every family and every modality. A conversation is a
// sequence of decoder-width rows: rows the model embedded from token ids, and —
// when the family has an image encoder and the user attached a picture — rows
// that came from pixels and carry no token id at all. That second possibility
// is the only reason history is an EmbeddedSequence rather than a vector of
// ids, and a text-only family is simply the case where it never happens: the
// encoder is empty, every segment is text, and the loop below is unchanged.
// Text is not a different kind of conversation, so it does not get a different
// class.
//
// Two parameters, because a chat is two choices: which model runs, and which
// text convention wraps it. The model already carries its own D, V and CTX, so
// naming its config again here would only be a second place for those numbers
// to disagree.
template <TransformerModel Model, ChatProtocol Protocol>
class ChatModel {
    static constexpr size_t D = Model::D;
    static constexpr size_t V = Model::V;
    static constexpr size_t CTX = Model::CTX;

public:
    // Held as a callable, not as an encoder member, so the adapter names no
    // vision type: the family that has one owns it inside this closure, and the
    // families that don't leave it empty.
    using ImageEncoder = std::function<Matrix<D>(const RGBImage&)>;

    ChatModel(Model model, const GGUF& gguf, std::string description, std::string system, SamplerCfg sampling, ImageEncoder encode_image = {}) : model_(std::move(model)), tokenizer_(gguf), sampler_(sampling), system_(std::move(system)), description_(std::move(description)), encode_image_(std::move(encode_image)) {
        const auto end = tokenizer_.encode(std::string(Protocol::TURN_END), false, true);
        if (end.size() != 1 || !tokenizer_.is_eog(end.front())) throw std::runtime_error("chat template terminator '" + std::string(Protocol::TURN_END) + "' is not a single end-of-generation token");
        turn_end_ = TokenId{end.front()};
        // Tokenized once here rather than per turn: ending a thought early is
        // the model's own closing text, put into the stream on its behalf.
        thought_end_ = to_tokens(tokenizer_.encode(std::string(Protocol::THOUGHT_END), false, true));
    }

    // prefix_memo_ refers to model_, so a ChatModel stays where it was built.
    // The make_* factories return prvalues, which C++17 elides into place.
    ChatModel(const ChatModel&) = delete;
    ChatModel& operator=(const ChatModel&) = delete;
    ChatModel(ChatModel&&) = delete;
    ChatModel& operator=(ChatModel&&) = delete;

    std::string description() const { return description_; }
    bool supports_images() const { return bool(encode_image_); }
    size_t context_used() const { return complete_input_ ? complete_input_->tokens() : 0; }
    size_t context_limit() const { return CTX; }
    const ChatStats& stats() const { return stats_; }

    void reset() {
        prefix_memo_.clear();
        complete_input_.reset();
        stats_ = {};
        first_ = true;
    }

    ChatTurnResult respond(std::string_view user, const RGBImage* image, const TurnBudget& budget, const ChatTokenSink& sink) {
        if (image && !encode_image_) throw std::invalid_argument("this model has no image encoder");

        // The turn is text, then optionally image rows, then text. A template
        // with no image slot puts the whole turn in the first piece and leaves
        // the second empty.
        // Embed the whole turn before committing any of it: a turn that does
        // not fit must leave the conversation exactly as it was, and only the
        // encoders can say how many rows an image is worth.
        const ChatTurnText rendered = Protocol::render(user, first_, system_, image != nullptr, budget.thinks());
        const std::vector<TokenId> before = to_tokens(tokenizer_.encode(rendered.before_image, first_, true));
        Matrix<D> before_rows = model_.embed(before);

        ChatTurnResult result;
        Matrix<D> image_rows;
        if (image) {
            const auto vision_start = std::chrono::steady_clock::now();
            image_rows = encode_image_(*image);
            result.delta.vision_seconds = seconds_since(vision_start);
        }

        std::vector<TokenId> after;
        Matrix<D> after_rows;
        if (!rendered.after_image.empty()) {
            after = to_tokens(tokenizer_.encode(rendered.after_image, false, true));
            after_rows = model_.embed(after);
        }

        const size_t new_tokens = before_rows.rows() + image_rows.rows() + after_rows.rows();
        if (new_tokens > CTX - context_used()) throw std::length_error("prompt exceeds the remaining context");
        if (!complete_input_) complete_input_.emplace();
        complete_input_->append(std::move(before_rows), before);
        if (image_rows.rows() != 0) complete_input_->append_soft_tokens(std::move(image_rows));
        if (after_rows.rows() != 0) complete_input_->append(std::move(after_rows), after);
        first_ = false;

        const auto prefill_start = std::chrono::steady_clock::now();
        Logits<V> logits = prefix_memo_.evaluate(*complete_input_);
        result.delta.prefill_seconds = seconds_since(prefill_start);
        result.delta.prefill_tokens = new_tokens;
        typename Protocol::Formatter output(sink);
        auto complete = [&]() {
            output.flush();
            finish(result);
            return result;
        };

        // Autoregression: one forward pass predicts exactly ONE next token, so
        // the whole response is that pass repeated. `logits` always holds the
        // distribution for the token after everything in complete_input_ —
        // sample it, show it, put it in the conversation, and the extend() that
        // records it hands back the distribution for the token after THAT.
        // Each pass costs one row, not the whole history, because the prefix
        // cache already holds every row before it.
        //
        // Three ways out, and they are not the same event: the model chose to
        // stop, the context filled up, or the caller's --max ran out.
        size_t thought_tokens = 0;
        for (size_t generated = 0; generated < budget.max_new; ++generated) {
            const TokenId id = sampler_(logits);
            ++result.delta.generated_tokens;

            // The model ended its own turn. Record the terminator so the next
            // turn starts from a closed transcript; its logits predict a token
            // nobody will ask for, hence the discard.
            if (tokenizer_.is_eog(int32_t(id))) {
                if (context_used() < CTX) (void)extend(id, result.delta);
                result.truncated = false;
                return complete();
            }

            output.push(tokenizer_.decode1(int32_t(id)));
            if (output.thinking()) ++thought_tokens;

            // The text is already on screen, but there is no room left to
            // record it, so the conversation cannot continue past this token.
            if (context_used() == CTX) {
                result.truncated = true;
                return complete();
            }
            logits = extend(id, result.delta);

            // The thought has spent its budget. Closing it is not a display
            // decision — the closing text goes into the conversation, so from
            // here the model is answering, and it reads back a turn where it
            // stopped thinking of its own accord.
            if (output.thinking() && thought_tokens >= budget.thinking) end_thought(output, logits, result.delta);
        }

        // Out of budget mid-sentence. The turn is still open — and so is the
        // thought, if --max ran out inside one. Close both, so the next turn
        // appends to a transcript in which the model finished what it started
        // rather than to a channel nobody ever closed.
        result.truncated = true;
        if (result.delta.generated_tokens > 0) {
            if (output.thinking()) end_thought(output, logits, result.delta);
            close_truncated_turn(result.delta);
        }
        return complete();
    }

private:
    static double seconds_since(std::chrono::steady_clock::time_point start) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
    // The tokenizer speaks int32_t and the model speaks TokenId.
    static std::vector<TokenId> to_tokens(const std::vector<int32_t>& ids) {
        if (ids.empty()) throw std::invalid_argument("chat template produced an empty text segment");
        std::vector<TokenId> tokens;
        tokens.reserve(ids.size());
        for (int32_t id : ids) tokens.push_back(TokenId{id});
        return tokens;
    }
    // Commit one token to the conversation and return the distribution for the
    // token that follows it. Both halves matter: the append is what makes the
    // model's own output part of its input next time round (that IS what makes
    // generation autoregressive), and the evaluate is one row of work, because
    // every row before it is already in prefix_memo_.
    Logits<V> extend(TokenId id, ChatStats& delta) {
        if (!complete_input_) throw std::logic_error("chat has no complete input");
        const std::span<const TokenId> one(&id, 1);
        complete_input_->append(model_.embed(one), one);
        const auto start = std::chrono::steady_clock::now();
        Logits<V> logits = prefix_memo_.evaluate(*complete_input_);
        delta.decode_seconds += seconds_since(start);
        return logits;
    }
    // A response cut off by --max leaves the turn open. Close it, so the next
    // turn appends to a well-formed transcript instead of continuing this one.
    void close_truncated_turn(ChatStats& delta) {
        if (context_used() == CTX) return;
        (void)extend(turn_end_, delta);
    }
    // End a thought the model has not finished, by writing the text it would
    // have written itself. The formatter is fed the same pieces, so it moves to
    // the answer exactly as it would have; `logits` is left predicting the
    // token after the close, which is the first token of the answer.
    void end_thought(typename Protocol::Formatter& output, Logits<V>& logits, ChatStats& delta) {
        for (TokenId id : thought_end_) {
            if (context_used() == CTX) return;
            output.push(tokenizer_.decode1(int32_t(id)));
            logits = extend(id, delta);
        }
    }
    void finish(ChatTurnResult& result) {
        stats_.prefill_tokens += result.delta.prefill_tokens;
        stats_.generated_tokens += result.delta.generated_tokens;
        stats_.prefill_seconds += result.delta.prefill_seconds;
        stats_.decode_seconds += result.delta.decode_seconds;
        stats_.vision_seconds += result.delta.vision_seconds;
    }

    Model model_;
    PrefixCache<Model> prefix_memo_{model_};
    std::optional<EmbeddedSequence<D>> complete_input_;
    Tokenizer tokenizer_;
    Sampler sampler_;
    std::string system_;
    std::string description_;
    ImageEncoder encode_image_;
    ChatStats stats_;
    TokenId turn_end_{0};
    std::vector<TokenId> thought_end_;
    bool first_ = true;
};

// Each factory names the two choices, loads the weights, and hands the adapter
// straight to the REPL. Nothing here is a closure type any more, so the return
// type is spellable — `auto` is brevity now, not necessity.
using Gemma4E4BChat = ChatModel<Gemma4E4BModel, GemmaChatProtocol>;
using Gemma4_12BChat = ChatModel<Gemma4_12BModel, GemmaChatProtocol>;
template <class C>
using Qwen35Chat = ChatModel<Qwen35Model<C>, ChatMLProtocol>;

// The adapters the REPL will instantiate, checked against the contract here
// rather than at the switch that picks one.
static_assert(ChatModelInterface<Gemma4E4BChat>);
static_assert(ChatModelInterface<Gemma4_12BChat>);
static_assert(ChatModelInterface<Qwen35Chat<Qwen35_4BConfig>>);
static_assert(ChatModelInterface<Qwen35Chat<Qwen35_9BConfig>>);

inline Gemma4E4BChat make_gemma4_e4b_chat_model(const GGUF& gguf, const std::string& mmproj, std::string system, SamplerCfg sampling) {
    Gemma4E4BChat::ImageEncoder encode_image;
    if (!mmproj.empty()) {
        GGUF vision_gguf(mmproj);
        // Ownership is really unique — the shared_ptr is only what it takes to
        // put a non-copyable encoder inside a copyable callable.
        auto encoder = std::make_shared<const Gemma4E4BVisionEncoder>(gemma4_vision_loader::load_e4b_vision(vision_gguf));
        encode_image = [encoder](const RGBImage& image) { return encoder->encode(image); };
    }

    const std::string description = std::string("Gemma 4 E4B D=2560 L=42 V=262144") + (encode_image ? " + vision" : " (text only)");
    return Gemma4E4BChat(gemma4_loader::load_e4b_text(gguf), gguf, description, std::move(system), sampling, std::move(encode_image));
}

inline Gemma4_12BChat make_gemma4_12b_chat_model(const GGUF& gguf, std::string system, SamplerCfg sampling) {
    return Gemma4_12BChat(gemma4_loader::load_12b_text(gguf), gguf, "Gemma 4 12B D=3840 L=48 (40 local + 8 full unified-K/V, text only)", std::move(system), sampling);
}

template <class C>
Qwen35Chat<C> make_qwen35_chat_model(const GGUF& gguf, std::string system, SamplerCfg sampling, const char* name) {
    return Qwen35Chat<C>(qwen35_loader::load<C>(gguf), gguf, std::string("Qwen 3.5 ") + name + " D=" + std::to_string(C::D) + " L=" + std::to_string(C::L) + " (" + std::to_string(C::L - C::L / C::FULL_ATTENTION_INTERVAL) + " gated-deltanet + " + std::to_string(C::L / C::FULL_ATTENTION_INTERVAL) + " attention)", std::move(system), sampling);
}
