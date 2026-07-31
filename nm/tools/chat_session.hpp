// chat_session.hpp — model-neutral conversation contract for the CLI.
#pragma once
#include "../src/gguf.hpp"
#include "../src/multimodal.hpp"
#include <concepts>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

// One entry per compiled adapter. A checkpoint is matched on the architecture
// string plus the anatomy that distinguishes sizes within a family, never on
// the file name, so a re-quantized or renamed GGUF still selects correctly.
enum class ChatModelKind : uint8_t { Gemma4E4B, Gemma4_12B, Qwen35_4B, Qwen35_9B };

struct ChatModelIdentity {
    std::string architecture;
    size_t embedding_length = 0;
    size_t block_count = 0;
};

inline std::optional<ChatModelKind> detect_chat_model(const ChatModelIdentity& model) {
    if (model.architecture == "gemma4") {
        if (model.embedding_length == 2560 && model.block_count == 42) return ChatModelKind::Gemma4E4B;
        if (model.embedding_length == 3840 && model.block_count == 48) return ChatModelKind::Gemma4_12B;
    }
    // Qwen 3.5 4B and 9B share L=32, so width is what separates them.
    if (model.architecture == "qwen35" && model.block_count == 32) {
        if (model.embedding_length == 2560) return ChatModelKind::Qwen35_4B;
        if (model.embedding_length == 4096) return ChatModelKind::Qwen35_9B;
    }
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

// What every chat template renders to: the turn's text, split where image rows
// go. An image is bracketed by control tokens, so even a template that puts the
// picture ahead of the user's words has text on both sides of it. A template
// with no image slot fills `before_image` and leaves `after_image` empty.
struct ChatTurnText {
    std::string before_image;
    std::string after_image;
};

inline ChatTurnText render_gemma_chat_turn(std::string_view user, bool first, std::string_view system, bool has_image) {
    ChatTurnText turn;
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

// ChatML, the Qwen-family template. Rendered as text and then tokenized with
// parse_special so the control tokens match whole (trap T4).
inline std::string render_chatml_turn(std::string_view user, bool first, std::string_view system) {
    std::string turn;
    if (first && !system.empty()) {
        turn += "<|im_start|>system\n";
        turn += system;
        turn += "<|im_end|>\n";
    }
    turn += "<|im_start|>user\n";
    turn += user;
    turn += "<|im_end|>\n<|im_start|>assistant\n";
    return turn;
}

struct ChatStats {
    size_t prefill_tokens = 0;
    size_t generated_tokens = 0;
    double prefill_seconds = 0;
    double decode_seconds = 0;
    double vision_seconds = 0;
    double prefill_tps() const { return prefill_seconds > 0 ? double(prefill_tokens) / prefill_seconds : 0; }
    double decode_tps() const { return decode_seconds > 0 ? double(generated_tokens) / decode_seconds : 0; }
};

struct ChatTurnResult {
    ChatStats delta;
    bool truncated = false;
};

using ChatTokenSink = std::function<void(std::string_view)>;

// The family's text protocol coming OUT, the counterpart to the render lambda
// that supplies the protocol going in. An adapter is given the type; it builds
// one per turn around the terminal's sink, pushes every decoded piece through
// it, and flushes at the end of the turn.
//
// For families whose token pieces are already the final text.
class PlainChatOutput {
public:
    explicit PlainChatOutput(ChatTokenSink sink) : sink_(std::move(sink)) {}

    void push(std::string_view piece) { sink_(piece); }
    void flush() {}

private:
    ChatTokenSink sink_;
};

// Gemma 4 may emit a small channel protocol before reasoning/final text:
// `<|channel>thought\n...`. This is FAMILY anatomy — every Gemma 4 size emits
// it — so both the E4B and 12B adapters format with it.
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

// What an OUTPUT FORMATTER is: built once per turn around the terminal's sink,
// fed every decoded piece, flushed at the end. It is a type and not a callable
// because it carries state across pieces — a channel tag arrives split over
// several tokens.
template <class F>
concept ChatOutputFormatter = std::constructible_from<F, ChatTokenSink> && requires(F& output, std::string_view piece) {
    { output.push(piece) } -> std::same_as<void>;
    { output.flush() } -> std::same_as<void>;
};

// What a chat PROTOCOL is: how a turn is written going in, how the model's
// pieces are read coming out, and the token that ends a turn. These three never
// vary independently — a checkpoint trained on ChatML emits ChatML and closes
// with <|im_end|> — so they are one type, not three parameters.
//
// Stated as a concept for the same reason as TransformerModel and
// ChatModelInterface, and never as a base class: a protocol has no per-instance
// state to inherit, and half of what it owes — a compile-time terminator, a
// formatter TYPE — cannot be spelled as a virtual function at all.
template <class P>
concept ChatProtocol = requires(std::string_view user, bool first, std::string_view system, bool has_image) {
    { P::render(user, first, system, has_image) } -> std::same_as<ChatTurnText>;
    { P::TURN_END } -> std::convertible_to<std::string_view>;
    requires ChatOutputFormatter<typename P::Formatter>;
};

struct GemmaChatProtocol {
    using Formatter = GemmaChannelFormatter;
    static constexpr std::string_view TURN_END = "<turn|>";
    static ChatTurnText render(std::string_view user, bool first, std::string_view system, bool has_image) { return render_gemma_chat_turn(user, first, system, has_image); }
};

struct ChatMLProtocol {
    using Formatter = PlainChatOutput;
    static constexpr std::string_view TURN_END = "<|im_end|>";
    // ChatML has no image slot, so the whole turn is one piece of text.
    static ChatTurnText render(std::string_view user, bool first, std::string_view system, bool) { return ChatTurnText{render_chatml_turn(user, first, system), ""}; }
};

// What a CHAT ADAPTER is. An adapter owns its chat template, prefix
// memoization, sampling, and any modality encoder; the REPL calls these seven
// operations by name and knows nothing else about it.
//
// No adapter inherits anything. WHICH adapter runs is decided once, by a switch
// on the checkpoint's metadata, so the variance is in which instantiation the
// dispatch picks — not in a vtable consulted on every call. Like
// TransformerModel, this concept generates no code: it
// exists so a mistake in a new adapter reports itself here rather than deep
// inside the REPL.
template <class M>
concept ChatModelInterface = requires(M& model, const M& adapter, std::string_view user, const RGBImage* image, size_t max_new, const ChatTokenSink& sink) {
    { adapter.description() } -> std::same_as<std::string>;
    { adapter.supports_images() } -> std::same_as<bool>;
    { adapter.context_used() } -> std::same_as<size_t>;
    { adapter.context_limit() } -> std::same_as<size_t>;
    { adapter.stats() } -> std::same_as<const ChatStats&>;
    { model.reset() } -> std::same_as<void>;
    { model.respond(user, image, max_new, sink) } -> std::same_as<ChatTurnResult>;
};
