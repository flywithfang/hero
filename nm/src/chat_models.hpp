// chat_models.hpp — model-specific adapters for the model-neutral chat loop.
// An adapter owns its chat template, prefix memoization, sampling, and any
// modality encoder; the REPL sees only the ChatModel interface.
#pragma once
#include "chat_session.hpp"
#include "gemma4_loader.hpp"
#include "gemma4_vision_loader.hpp"
#include "qwen35_loader.hpp"
#include "runtime.hpp"
#include "tokenizer.hpp"
#include <chrono>

// A text-only adapter over any architecture whose input is token ids and whose
// turns are rendered as text. Qwen 3.5 uses it; a future text-only family
// should too rather than growing a second copy of this loop.
template <class C, class Model, class RenderTurn>
class TextChatModel : public ChatModel {
public:
    TextChatModel(Model model, const GGUF& gguf, std::string description, std::string system, SamplerCfg sampling, RenderTurn render) : model_(std::move(model)), tokenizer_(gguf), sampler_(sampling), system_(std::move(system)), description_(std::move(description)), render_(std::move(render)) {}

    std::string description() const override { return description_; }
    bool supports_images() const override { return false; }
    size_t context_used() const override { return history_.size(); }
    size_t context_limit() const override { return C::CTX; }
    const ChatStats& stats() const override { return stats_; }

    void reset() override {
        prefix_memo_.clear();
        history_.clear();
        stats_ = {};
        first_ = true;
    }

    ChatTurnResult respond(std::string_view user, const RGBImage* image, size_t max_new, const ChatTokenSink& sink) override {
        if (image) throw std::invalid_argument("this model has no image encoder");
        const std::string rendered = render_(user, first_, system_);
        const auto ids = tokenizer_.encode(rendered, first_, /*parse_special=*/true);
        if (ids.size() > C::CTX - history_.size()) throw std::length_error("prompt exceeds the remaining context");
        first_ = false;
        for (int32_t id : ids) history_.push_back(TokenId{id});

        ChatTurnResult result;
        const auto prefill_start = std::chrono::steady_clock::now();
        Vec<C::V> logits = evaluate(model_, history_, prefix_memo_);
        result.delta.prefill_seconds = seconds_since(prefill_start);
        result.delta.prefill_tokens = ids.size();

        for (size_t generated = 0; generated < max_new; ++generated) {
            TokenId id = sampler_(logits, history_);
            ++result.delta.generated_tokens;
            if (tokenizer_.is_eog(int32_t(id))) {
                if (history_.size() < C::CTX) {
                    history_.push_back(id);
                    (void)accept(result.delta);
                }
                result.truncated = false;
                finish(result);
                return result;
            }
            sink(tokenizer_.decode1(int32_t(id)));
            if (history_.size() == C::CTX) {
                result.truncated = true;
                finish(result);
                return result;
            }
            history_.push_back(id);
            logits = accept(result.delta);
        }
        result.truncated = true;
        finish(result);
        return result;
    }

private:
    static double seconds_since(std::chrono::steady_clock::time_point start) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
    Vec<C::V> accept(ChatStats& delta) {
        const auto start = std::chrono::steady_clock::now();
        Vec<C::V> logits = evaluate(model_, history_, prefix_memo_);
        delta.decode_seconds += seconds_since(start);
        return logits;
    }
    void finish(ChatTurnResult& result) {
        stats_.prefill_tokens += result.delta.prefill_tokens;
        stats_.generated_tokens += result.delta.generated_tokens;
        stats_.prefill_seconds += result.delta.prefill_seconds;
        stats_.decode_seconds += result.delta.decode_seconds;
    }

    Model model_;
    PrefixCache<Model> prefix_memo_;
    Tokenizer tokenizer_;
    Sampler<C::V> sampler_;
    std::string system_;
    std::string description_;
    RenderTurn render_;
    std::vector<TokenId> history_;
    ChatStats stats_;
    bool first_ = true;
};

static std::unique_ptr<ChatModel> make_gemma4_12b_chat_model(const GGUF& gguf, std::string system, SamplerCfg sampling) {
    auto render = [](std::string_view user, bool first, std::string_view system) {
        const GemmaChatTurn turn = render_gemma_chat_turn(user, first, system, /*has_image=*/false);
        return turn.before_image + turn.after_image;
    };
    return std::make_unique<TextChatModel<Gemma4_12BTextConfig, Gemma4_12BModel, decltype(render)>>(gemma4_loader::load_12b_text(gguf), gguf, "Gemma 4 12B D=3840 L=48 (40 sliding + 8 global unified-K/V, text only)", std::move(system), sampling, render);
}

template <class C>
static std::unique_ptr<ChatModel> make_qwen35_chat_model(const GGUF& gguf, std::string system, SamplerCfg sampling, const char* name) {
    auto render = [](std::string_view user, bool first, std::string_view system) { return render_chatml_turn(user, first, system); };
    return std::make_unique<TextChatModel<C, Qwen35Model<C>, decltype(render)>>(qwen35_loader::load<C>(gguf), gguf, std::string("Qwen 3.5 ") + name + " D=" + std::to_string(C::D) + " L=" + std::to_string(C::L) + " (" + std::to_string(C::L - C::L / C::FULL_ATTENTION_INTERVAL) + " gated-deltanet + " + std::to_string(C::L / C::FULL_ATTENTION_INTERVAL) + " attention)", std::move(system), sampling, render);
}

class Gemma4E4BChatModel final : public ChatModel {
    using C = Gemma4E4BTextConfig;

public:
    Gemma4E4BChatModel(const GGUF& text_gguf, const std::string& mmproj, std::string system, SamplerCfg sampling) : model_(gemma4_loader::load_e4b_text(text_gguf)), tokenizer_(text_gguf), sampler_(sampling), system_(std::move(system)) {
        const auto end = tokenizer_.encode("<turn|>", false, true);
        if (end.size() != 1 || !tokenizer_.is_eog(end.front())) throw std::runtime_error("Gemma 4 tokenizer has no unique <turn|> terminator");
        turn_end_ = TokenId{end.front()};
        if (!mmproj.empty()) {
            GGUF vision_gguf(mmproj);
            vision_ = std::make_unique<Gemma4E4BVisionEncoder>(gemma4_vision_loader::load_e4b_vision(vision_gguf));
        }
    }

    std::string description() const override { return std::string("Gemma 4 E4B D=2560 L=42 V=262144") + (vision_ ? " + vision" : " (text only)"); }
    bool supports_images() const override { return bool(vision_); }
    size_t context_used() const override { return complete_input_ ? complete_input_->tokens() : 0; }
    size_t context_limit() const override { return C::CTX; }
    const ChatStats& stats() const override { return stats_; }

    void reset() override {
        prefix_memo_.clear();
        complete_input_.reset();
        history_.clear();
        stats_ = {};
        first_ = true;
    }

    ChatTurnResult respond(std::string_view user, const RGBImage* image, size_t max_new, const ChatTokenSink& sink) override {
        if (image && !vision_) throw std::invalid_argument("image supplied without --mmproj");

        const GemmaChatTurn rendered = render_gemma_chat_turn(user, first_, system_, image != nullptr);
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
        size_t new_tokens = 0;
        for (const auto& segment : segments) new_tokens += segment.embeddings().rows();
        if (new_tokens > C::CTX - context_used()) throw std::length_error("prompt exceeds the remaining context");
        if (!complete_input_) {
            complete_input_.emplace(compose_embeddings(std::move(segments)));
        } else {
            for (auto& segment : segments) complete_input_->append(std::move(segment));
        }
        first_ = false;
        for (int32_t id : before_ids) history_.push_back(TokenId{id});
        for (int32_t id : after_ids) history_.push_back(TokenId{id});

        const auto prefill_start = std::chrono::steady_clock::now();
        Vec<C::V> logits = evaluate(model_, *complete_input_, prefix_memo_);
        result.delta.prefill_seconds = seconds_since(prefill_start);
        result.delta.prefill_tokens = new_tokens;
        GemmaChannelFormatter output(sink);
        auto complete = [&]() {
            output.flush();
            finish(result);
            return result;
        };

        for (size_t generated = 0; generated < max_new; ++generated) {
            TokenId id = sampler_(logits, history_);
            ++result.delta.generated_tokens;
            if (tokenizer_.is_eog(int32_t(id))) {
                if (context_used() < C::CTX) {
                    history_.push_back(id);
                    (void)accept(id, result.delta);
                }
                result.truncated = false;
                return complete();
            }

            const std::string piece = tokenizer_.decode1(int32_t(id));
            output.push(piece);
            if (context_used() == C::CTX) {
                result.truncated = true;
                return complete();
            }
            history_.push_back(id);
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
    static double seconds_since(std::chrono::steady_clock::time_point start) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
    EmbeddingSegment<C::D> text_segment(const std::vector<int32_t>& ids) const {
        if (ids.empty()) throw std::invalid_argument("Gemma chat produced an empty text segment");
        TokenMatrix<C::D> embeddings(ids.size());
        std::vector<TokenId> identities;
        identities.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            const TokenId id{ids[i]};
            identities.push_back(id);
            embeddings.set_row(i, model_.token(id));
        }
        return EmbeddingSegment<C::D>(std::move(embeddings), std::move(identities));
    }
    Vec<C::V> accept(TokenId id, ChatStats& delta) {
        if (!complete_input_) throw std::logic_error("Gemma chat has no complete input");
        std::vector<int32_t> token{int32_t(id)};
        complete_input_->append(text_segment(token));
        const auto start = std::chrono::steady_clock::now();
        Vec<C::V> logits = evaluate(model_, *complete_input_, prefix_memo_);
        delta.decode_seconds += seconds_since(start);
        return logits;
    }
    void close_truncated_turn(ChatStats& delta) {
        if (context_used() == C::CTX) return;
        history_.push_back(turn_end_);
        (void)accept(turn_end_, delta);
    }
    void finish(ChatTurnResult& result) {
        stats_.prefill_tokens += result.delta.prefill_tokens;
        stats_.generated_tokens += result.delta.generated_tokens;
        stats_.prefill_seconds += result.delta.prefill_seconds;
        stats_.decode_seconds += result.delta.decode_seconds;
        stats_.vision_seconds += result.delta.vision_seconds;
    }

    Gemma4E4BModel model_;
    PrefixCache<Gemma4E4BModel> prefix_memo_;
    std::optional<EmbeddedSequence<C::D>> complete_input_;
    Tokenizer tokenizer_;
    Sampler<C::V> sampler_;
    std::unique_ptr<Gemma4E4BVisionEncoder> vision_;
    std::string system_;
    std::vector<TokenId> history_;
    ChatStats stats_;
    TokenId turn_end_{0};
    bool first_ = true;
};
