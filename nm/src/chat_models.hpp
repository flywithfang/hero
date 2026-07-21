// chat_models.hpp — model-specific adapters for the model-neutral chat loop.
#pragma once
#include "chat_session.hpp"
#include "gemma4_loader.hpp"
#include "gemma4_vision_loader.hpp"
#include "llama_loader.hpp"
#include "runtime.hpp"
#include "tokenizer.hpp"
#include <chrono>

template <class C>
class LlamaChatModel final : public ChatModel {
public:
    LlamaChatModel(const GGUF& gguf, std::string system, SamplerCfg sampling)
        : model_(llama_loader::load<C>(gguf)), tokenizer_(gguf), runtime_(model_),
          sampler_(sampling), system_(std::move(system)) {}

    std::string description() const override {
        return "Llama D=" + std::to_string(C::D) + " L=" + std::to_string(C::L) +
               " V=" + std::to_string(C::V);
    }
    bool supports_images() const override { return false; }
    size_t context_used() const override { return runtime_.context_used(); }
    size_t context_limit() const override { return C::CTX; }
    const ChatStats& stats() const override { return stats_; }

    void reset() override {
        runtime_.reset();
        history_.clear();
        stats_ = {};
        first_ = true;
    }

    ChatTurnResult respond(std::string_view user, const RGBImage* image,
                           size_t max_new, const ChatTokenSink& sink) override {
        if (image) throw std::invalid_argument("this Llama model has no image encoder");
        const auto ids = tokenizer_.chat_user_turn(std::string(user), first_, system_);
        if (ids.size() > C::CTX - runtime_.context_used())
            throw std::length_error("prompt exceeds the remaining context");
        first_ = false;

        std::vector<TokenId> prompt;
        prompt.reserve(ids.size());
        for (int32_t id : ids) {
            prompt.push_back(TokenId{id});
            history_.push_back(TokenId{id});
        }

        ChatTurnResult result;
        const auto prefill_start = std::chrono::steady_clock::now();
        Vec<C::V> logits = runtime_.prefill(prompt);
        result.delta.prefill_seconds = seconds_since(prefill_start);
        result.delta.prefill_tokens = prompt.size();

        for (size_t generated = 0; generated < max_new; ++generated) {
            TokenId id = sampler_(logits, history_);
            history_.push_back(id);
            ++result.delta.generated_tokens;
            if (tokenizer_.is_eog(int32_t(id))) {
                if (runtime_.context_used() < C::CTX) (void)accept(id, result.delta);
                result.truncated = false;
                finish(result);
                return result;
            }

            const std::string piece = tokenizer_.decode1(int32_t(id));
            sink(piece);
            if (runtime_.context_used() == C::CTX) {
                result.truncated = true;
                finish(result);
                return result;
            }
            logits = accept(id, result.delta);

            if (generated + 1 == max_new) {
                result.truncated = true;
                close_truncated_turn(TokenId{tokenizer_.eot()}, result.delta);
                finish(result);
                return result;
            }
        }
        result.truncated = true;
        finish(result);
        return result;
    }

private:
    static double seconds_since(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    Vec<C::V> accept(TokenId id, ChatStats& delta) {
        const auto start = std::chrono::steady_clock::now();
        Vec<C::V> logits = runtime_.step(id);
        delta.decode_seconds += seconds_since(start);
        return logits;
    }
    void close_truncated_turn(TokenId end, ChatStats& delta) {
        if (runtime_.context_used() == C::CTX) return;
        history_.push_back(end);
        const auto start = std::chrono::steady_clock::now();
        (void)runtime_.step(end);
        delta.decode_seconds += seconds_since(start);
    }
    void finish(ChatTurnResult& result) {
        stats_.prefill_tokens += result.delta.prefill_tokens;
        stats_.generated_tokens += result.delta.generated_tokens;
        stats_.prefill_seconds += result.delta.prefill_seconds;
        stats_.decode_seconds += result.delta.decode_seconds;
    }

    LlamaModel<C> model_;
    Tokenizer tokenizer_;
    AutoregressiveRuntime<LlamaArchitecture<C>> runtime_;
    Sampler<C::V> sampler_;
    std::string system_;
    std::vector<TokenId> history_;
    ChatStats stats_;
    bool first_ = true;
};

class Gemma4E4BChatModel final : public ChatModel {
    using C = Gemma4E4BTextConfig;

public:
    Gemma4E4BChatModel(const GGUF& text_gguf, const std::string& mmproj,
                       std::string system, SamplerCfg sampling)
        : model_(gemma4_loader::load_e4b_text(text_gguf)), tokenizer_(text_gguf),
          runtime_(model_), sampler_(sampling), system_(std::move(system)) {
        const auto end = tokenizer_.encode("<turn|>", false, true);
        if (end.size() != 1 || !tokenizer_.is_eog(end.front()))
            throw std::runtime_error("Gemma 4 tokenizer has no unique <turn|> terminator");
        turn_end_ = TokenId{end.front()};
        if (!mmproj.empty()) {
            GGUF vision_gguf(mmproj);
            vision_ = std::make_unique<Gemma4E4BVisionEncoder>(
                gemma4_vision_loader::load_e4b_vision(vision_gguf));
        }
    }

    std::string description() const override {
        return std::string("Gemma 4 E4B D=2560 L=42 V=262144") +
               (vision_ ? " + vision" : " (text only)");
    }
    bool supports_images() const override { return bool(vision_); }
    size_t context_used() const override { return runtime_.context_used(); }
    size_t context_limit() const override { return C::CTX; }
    const ChatStats& stats() const override { return stats_; }

    void reset() override {
        runtime_.reset();
        history_.clear();
        stats_ = {};
        first_ = true;
    }

    ChatTurnResult respond(std::string_view user, const RGBImage* image,
                           size_t max_new, const ChatTokenSink& sink) override {
        if (image && !vision_)
            throw std::invalid_argument("image supplied without --mmproj");

        const GemmaChatTurn rendered = render_gemma_chat_turn(
            user, first_, system_, image != nullptr);
        const auto before_ids = tokenizer_.encode(rendered.before_image, first_, true);
        const auto after_ids = tokenizer_.encode(rendered.after_image, false, true);

        std::vector<EmbeddingSegment<C::D>> segments;
        segments.push_back(text_segment(before_ids));
        ChatTurnResult result;
        if (image) {
            const auto vision_start = std::chrono::steady_clock::now();
            segments.push_back(vision_->encode(*image));
            result.delta.vision_seconds = seconds_since(vision_start);
        }
        segments.push_back(text_segment(after_ids));
        EmbeddedSequence<C::D> sequence = compose_embeddings(
            std::move(segments), runtime_.context_used());
        if (sequence.tokens() > C::CTX - runtime_.context_used())
            throw std::length_error("prompt exceeds the remaining context");
        first_ = false;
        for (int32_t id : before_ids) history_.push_back(TokenId{id});
        for (int32_t id : after_ids) history_.push_back(TokenId{id});

        const auto prefill_start = std::chrono::steady_clock::now();
        Vec<C::V> logits = runtime_.forward(sequence);
        result.delta.prefill_seconds = seconds_since(prefill_start);
        result.delta.prefill_tokens = sequence.tokens();
        GemmaChannelFormatter output(sink);
        auto complete = [&]() {
            output.flush();
            finish(result);
            return result;
        };

        for (size_t generated = 0; generated < max_new; ++generated) {
            TokenId id = sampler_(logits, history_);
            history_.push_back(id);
            ++result.delta.generated_tokens;
            if (tokenizer_.is_eog(int32_t(id))) {
                if (runtime_.context_used() < C::CTX) (void)accept(id, result.delta);
                result.truncated = false;
                return complete();
            }

            const std::string piece = tokenizer_.decode1(int32_t(id));
            output.push(piece);
            if (runtime_.context_used() == C::CTX) {
                result.truncated = true;
                return complete();
            }
            logits = accept(id, result.delta);

            if (generated + 1 == max_new) {
                result.truncated = true;
                close_truncated_turn(result.delta);
                return complete();
            }
        }
        result.truncated = true;
        return complete();
    }

private:
    static double seconds_since(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    EmbeddingSegment<C::D> text_segment(const std::vector<int32_t>& ids) const {
        if (ids.empty()) throw std::invalid_argument("Gemma chat produced an empty text segment");
        TokenMatrix<C::D> embeddings(ids.size());
        std::vector<TokenId> identities;
        identities.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            const TokenId id{ids[i]};
            identities.push_back(id);
            embeddings.set_row(i, model_.token_io().token(id));
        }
        return EmbeddingSegment<C::D>(std::move(embeddings), std::move(identities));
    }
    Vec<C::V> accept(TokenId id, ChatStats& delta) {
        const auto start = std::chrono::steady_clock::now();
        Vec<C::V> logits = runtime_.step(id);
        delta.decode_seconds += seconds_since(start);
        return logits;
    }
    void close_truncated_turn(ChatStats& delta) {
        if (runtime_.context_used() == C::CTX) return;
        history_.push_back(turn_end_);
        const auto start = std::chrono::steady_clock::now();
        (void)runtime_.step(turn_end_);
        delta.decode_seconds += seconds_since(start);
    }
    void finish(ChatTurnResult& result) {
        stats_.prefill_tokens += result.delta.prefill_tokens;
        stats_.generated_tokens += result.delta.generated_tokens;
        stats_.prefill_seconds += result.delta.prefill_seconds;
        stats_.decode_seconds += result.delta.decode_seconds;
        stats_.vision_seconds += result.delta.vision_seconds;
    }

    Gemma4E4BTextModel model_;
    Tokenizer tokenizer_;
    AutoregressiveRuntime<Gemma4E4BArchitecture> runtime_;
    Sampler<C::V> sampler_;
    std::unique_ptr<Gemma4E4BVisionEncoder> vision_;
    std::string system_;
    std::vector<TokenId> history_;
    ChatStats stats_;
    TokenId turn_end_{0};
    bool first_ = true;
};
