// chat_session.hpp — model-neutral conversation contract for the CLI.
#pragma once
#include "gguf.hpp"
#include "multimodal.hpp"
#include <functional>
#include <optional>
#include <string_view>

enum class ChatModelKind : uint8_t { Llama32_1B, Llama32_3B, Llama3_8B, Gemma4E4B };

struct ChatModelIdentity {
    std::string architecture;
    size_t embedding_length = 0;
    size_t block_count = 0;
};

inline std::optional<ChatModelKind> detect_chat_model(const ChatModelIdentity& model) {
    if (model.architecture == "llama") {
        if (model.embedding_length == 2048 && model.block_count == 16)
            return ChatModelKind::Llama32_1B;
        if (model.embedding_length == 3072 && model.block_count == 28)
            return ChatModelKind::Llama32_3B;
        if (model.embedding_length == 4096 && model.block_count == 32)
            return ChatModelKind::Llama3_8B;
    }
    if (model.architecture == "gemma4" && model.embedding_length == 2560 &&
        model.block_count == 42)
        return ChatModelKind::Gemma4E4B;
    return std::nullopt;
}

inline ChatModelIdentity chat_model_identity(const GGUF& gguf) {
    const std::string architecture = gguf.get_str("general.architecture", "?");
    return ChatModelIdentity{
        architecture,
        size_t(gguf.get_int(architecture + ".embedding_length", 0)),
        size_t(gguf.get_int(architecture + ".block_count", 0)),
    };
}

struct GemmaChatTurn {
    std::string before_image;
    std::string after_image;
};

inline GemmaChatTurn render_gemma_chat_turn(std::string_view user, bool first,
                                            std::string_view system, bool has_image) {
    GemmaChatTurn turn;
    if (first) {
        turn.before_image = "<|turn>system\n<|think|>\n";
        turn.before_image += system;
        turn.before_image += "<turn|>\n";
    }
    turn.before_image += "<|turn>user\n";
    if (has_image) turn.before_image += "<|image>";
    if (has_image) turn.after_image = "<image|>";
    turn.after_image += user;
    turn.after_image += "<turn|>\n<|turn>model\n";
    return turn;
}

struct ChatStats {
    size_t prefill_tokens = 0;
    size_t generated_tokens = 0;
    double prefill_seconds = 0;
    double decode_seconds = 0;
    double vision_seconds = 0;
    double prefill_tps() const {
        return prefill_seconds > 0 ? double(prefill_tokens) / prefill_seconds : 0;
    }
    double decode_tps() const {
        return decode_seconds > 0 ? double(generated_tokens) / decode_seconds : 0;
    }
};

struct ChatTurnResult {
    ChatStats delta;
    bool truncated = false;
};

using ChatTokenSink = std::function<void(std::string_view)>;

// Gemma 4 may emit a small channel protocol before reasoning/final text:
// `<|channel>thought\n...`. Keep that protocol model-specific while presenting
// readable streaming output to the shared terminal frontend.
class GemmaChannelFormatter {
public:
    explicit GemmaChannelFormatter(ChatTokenSink sink) : sink_(std::move(sink)) {}

    void push(std::string_view piece) {
        if (!reading_channel_) {
            if (piece == "<|channel>") {
                reading_channel_ = true;
                channel_.clear();
            } else {
                sink_(piece);
            }
            return;
        }

        const size_t newline = piece.find('\n');
        channel_.append(piece.substr(0, newline));
        if (newline == std::string_view::npos) return;
        emit_label();
        reading_channel_ = false;
        if (newline + 1 < piece.size()) sink_(piece.substr(newline + 1));
    }

    void flush() {
        if (!reading_channel_) return;
        sink_("<|channel>");
        sink_(channel_);
        reading_channel_ = false;
        channel_.clear();
    }

private:
    void emit_label() {
        sink_("\n[");
        sink_(channel_);
        sink_("]\n");
        channel_.clear();
    }

    ChatTokenSink sink_;
    std::string channel_;
    bool reading_channel_ = false;
};

class ChatModel {
public:
    virtual ~ChatModel() = default;
    virtual std::string description() const = 0;
    virtual bool supports_images() const = 0;
    virtual size_t context_used() const = 0;
    virtual size_t context_limit() const = 0;
    virtual const ChatStats& stats() const = 0;
    virtual void reset() = 0;
    virtual ChatTurnResult respond(std::string_view user, const RGBImage* image,
                                   size_t max_new, const ChatTokenSink& sink) = 0;
};
