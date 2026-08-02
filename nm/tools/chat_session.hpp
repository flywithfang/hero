// chat_session.hpp — model-neutral conversation contract for the CLI.
#pragma once
#include "../src/gguf.hpp"
#include "../src/multimodal.hpp"
#include <concepts>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

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

// Gemma 4 asks for thinking ONCE, with `<|think|>` at the head of the first
// system turn — the checkpoint's `enable_thinking`. A system turn carrying
// neither that switch nor a system prompt has nothing in it, so the template
// omits it rather than opening an empty one.
inline ChatTurnText render_gemma_chat_turn(std::string_view user, bool first, std::string_view system, bool has_image, bool thinking) {
    ChatTurnText turn;
    if (first && (thinking || !system.empty())) {
        turn.before_image = "<|turn>system\n";
        if (thinking) turn.before_image += "<|think|>\n";
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
//
// Qwen puts its thinking switch in EVERY turn rather than in the system block,
// and says "do not think" by prefilling an already-closed, empty thought. The
// thinking case deliberately prefills nothing: the template opens `<think>`
// here, but an opener written by us never reaches the formatter, which reads
// the model's own tokens — and this checkpoint opens the tag itself.
inline std::string render_chatml_turn(std::string_view user, bool first, std::string_view system, bool thinking) {
    std::string turn;
    if (first && !system.empty()) {
        turn += "<|im_start|>system\n";
        turn += system;
        turn += "<|im_end|>\n";
    }
    turn += "<|im_start|>user\n";
    turn += user;
    turn += "<|im_end|>\n<|im_start|>assistant\n";
    if (!thinking) turn += "<think>\n\n</think>\n\n";
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

// What one turn may generate: the whole response, and the part of it the model
// is allowed to spend thinking. The thinking budget is a TOKEN COUNT and not a
// level, because no checkpoint here has levels — Gemma 4 and Qwen 3.5 each
// expose one boolean (`enable_thinking`) and nothing finer, so effort is
// something the application spends rather than something the model is told.
// Zero asks the template for no thought at all; anything short of SIZE_MAX
// closes the thought FOR the model once it has run that long, which is the
// only reason a running thought can be cut short at all.
struct TurnBudget {
    size_t max_new = 1024;
    size_t thinking = SIZE_MAX;
    bool thinks() const { return thinking > 0; }
};

// A turn arrives as two kinds of text, and a reader treats them differently:
// the reasoning the model does before answering, and the answer itself. Every
// family marks the boundary in its own tokens — Gemma 4 opens a channel
// (`<|channel>thought` ... `<channel|>`), Qwen wraps `<think>` ... `</think>` —
// so what is shared is the DISTINCTION, never how it is spelled. That is why it
// belongs in the sink's signature: the formatter knows the family's delimiters,
// and the terminal alone decides what a channel should look like.
enum class ChatChannel : uint8_t { Reasoning, Response };

using ChatTokenSink = std::function<void(ChatChannel, std::string_view)>;

// The part of channel formatting that is the same in every family: a channel's
// text starts at its first real character. Each delimiter is followed by the
// newline that separated it from the text (`<|channel>thought\n`, `</think>\n\n`)
// and that newline belongs to the protocol, not to what the model said.
class ChatChannelWriter {
public:
    explicit ChatChannelWriter(ChatTokenSink sink) : sink_(std::move(sink)) {}

    void write(ChatChannel channel, std::string_view piece) {
        if (channel != channel_) {
            channel_ = channel;
            at_start_ = true;
        }
        if (at_start_) {
            const size_t text = piece.find_first_not_of(" \t\r\n");
            if (text == std::string_view::npos) return;
            piece.remove_prefix(text);
            at_start_ = false;
        }
        sink_(channel, piece);
    }

private:
    ChatTokenSink sink_;
    ChatChannel channel_ = ChatChannel::Response;
    bool at_start_ = true;
};

// The family's text protocol coming OUT, the counterpart to the render lambda
// that supplies the protocol going in. An adapter is given the type; it builds
// one per turn around the terminal's sink, pushes every decoded piece through
// it, and flushes at the end of the turn.
//
// Gemma 4 puts its reasoning in a channel: `<|channel>thought\n ... <channel|>`.
// Both delimiters are single user-defined tokens, so each arrives as a whole
// piece; the name between the opener and the newline is ordinary text and can
// be split across pieces. This is FAMILY anatomy — every Gemma 4 size emits it
// — so both the E4B and 12B adapters format with it. `thought` is the only
// channel the template ever opens and the terminal labels the section itself,
// so the name is read to find where it ends and then dropped.
class GemmaChannelFormatter {
public:
    explicit GemmaChannelFormatter(ChatTokenSink sink) : out_(std::move(sink)) {}

    void push(std::string_view piece) {
        if (piece == "<|channel>") {
            reading_ = Reading::Name;
            name_.clear();
            return;
        }
        if (piece == "<channel|>") {
            reading_ = Reading::Answer;
            return;
        }
        if (reading_ != Reading::Name) {
            out_.write(reading_ == Reading::Thought ? ChatChannel::Reasoning : ChatChannel::Response, piece);
            return;
        }
        const size_t newline = piece.find('\n');
        name_.append(piece.substr(0, newline));
        if (newline == std::string_view::npos) return;
        reading_ = Reading::Thought;
        out_.write(ChatChannel::Reasoning, piece.substr(newline + 1));
    }

    // A turn that ran out mid-name never said which channel it was opening.
    // Show the text that did arrive rather than swallowing it.
    void flush() {
        if (reading_ != Reading::Name) return;
        reading_ = Reading::Thought;
        out_.write(ChatChannel::Reasoning, name_);
    }

    // An open channel counts as thinking from its opening token, name and all,
    // so a zero budget can close one before any of it reaches the screen.
    bool thinking() const { return reading_ != Reading::Answer; }

private:
    enum class Reading : uint8_t { Answer, Name, Thought };

    ChatChannelWriter out_;
    std::string name_;
    Reading reading_ = Reading::Answer;
};

// The `<think>` ... `</think>` convention, which Qwen 3.5 emits inside its
// ChatML assistant turn. Both tags are single user-defined tokens, so telling
// the two channels apart is one comparison per piece. A model that answers
// without thinking simply never opens the tag.
class ThinkTagFormatter {
public:
    explicit ThinkTagFormatter(ChatTokenSink sink) : out_(std::move(sink)) {}

    void push(std::string_view piece) {
        if (piece == "<think>") {
            thinking_ = true;
            return;
        }
        if (piece == "</think>") {
            thinking_ = false;
            return;
        }
        out_.write(thinking_ ? ChatChannel::Reasoning : ChatChannel::Response, piece);
    }
    void flush() {}
    bool thinking() const { return thinking_; }

private:
    ChatChannelWriter out_;
    bool thinking_ = false;
};

// What an OUTPUT FORMATTER is: built once per turn around the terminal's sink,
// fed every decoded piece, flushed at the end. It is a type and not a callable
// because it carries state across pieces — a channel tag arrives split over
// several tokens. That same state answers whether a thought is currently open,
// which is what a thinking budget is spent against.
template <class F>
concept ChatOutputFormatter = std::constructible_from<F, ChatTokenSink> && requires(F& output, std::string_view piece) {
    { output.push(piece) } -> std::same_as<void>;
    { output.flush() } -> std::same_as<void>;
    { std::as_const(output).thinking() } -> std::same_as<bool>;
};

// What a chat PROTOCOL is: how a turn is written going in, how the model's
// pieces are read coming out, the token that ends a turn, and the text that
// ends a thought. These never vary independently — a checkpoint trained on
// ChatML emits ChatML, closes with <|im_end|> and thinks in <think> tags — so
// they are one type, not four parameters.
//
// Stated as a concept for the same reason as TransformerModel and
// ChatModelInterface, and never as a base class: a protocol has no per-instance
// state to inherit, and half of what it owes — a compile-time terminator, a
// formatter TYPE — cannot be spelled as a virtual function at all.
template <class P>
concept ChatProtocol = requires(std::string_view user, bool first, std::string_view system, bool has_image, bool thinking) {
    { P::render(user, first, system, has_image, thinking) } -> std::same_as<ChatTurnText>;
    { P::TURN_END } -> std::convertible_to<std::string_view>;
    { P::THOUGHT_END } -> std::convertible_to<std::string_view>;
    requires ChatOutputFormatter<typename P::Formatter>;
};

struct GemmaChatProtocol {
    using Formatter = GemmaChannelFormatter;
    static constexpr std::string_view TURN_END = "<turn|>";
    // Closing the channel is how a thought is ended, whoever ends it: the model
    // emits this itself when it is done, and the adapter emits the same text on
    // its behalf when the budget runs out.
    static constexpr std::string_view THOUGHT_END = "<channel|>";
    static ChatTurnText render(std::string_view user, bool first, std::string_view system, bool has_image, bool thinking) { return render_gemma_chat_turn(user, first, system, has_image, thinking); }
};

struct ChatMLProtocol {
    using Formatter = ThinkTagFormatter;
    static constexpr std::string_view TURN_END = "<|im_end|>";
    static constexpr std::string_view THOUGHT_END = "</think>";
    // ChatML has no image slot, so the whole turn is one piece of text.
    static ChatTurnText render(std::string_view user, bool first, std::string_view system, bool, bool thinking) { return ChatTurnText{render_chatml_turn(user, first, system, thinking), ""}; }
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
concept ChatModelInterface = requires(M& model, const M& adapter, std::string_view user, const RGBImage* image, const TurnBudget& budget, const ChatTokenSink& sink) {
    { adapter.description() } -> std::same_as<std::string>;
    { adapter.supports_images() } -> std::same_as<bool>;
    { adapter.context_used() } -> std::same_as<size_t>;
    { adapter.context_limit() } -> std::same_as<size_t>;
    { adapter.stats() } -> std::same_as<const ChatStats&>;
    { model.reset() } -> std::same_as<void>;
    { model.respond(user, image, budget, sink) } -> std::same_as<ChatTurnResult>;
};
