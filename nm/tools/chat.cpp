// chat.cpp — model-neutral interactive chat over GGUF models.
//
// Model selection is metadata-driven: one switch picks the adapter, and the
// REPL is instantiated for that concrete type — no base class, no vtable. Each
// family adapter owns its template, prefix memoization, sampling, and any
// modality encoder.
#include "chat_models.hpp"
#include "../src/image_io.hpp"
#include <cstdio>
#include <iostream>
#include <optional>

struct Opts {
    std::string model;
    std::string mmproj;
    std::string initial_image;
    std::string arch;
    std::optional<std::string> system;
    SamplerCfg sampling;
    TurnBudget budget;
};

// `on`, `off`, or a token count — one knob, because thinking on and thinking
// briefly are the same setting at different values, and neither checkpoint has
// a coarser dial of its own to forward the request to.
static size_t parse_thinking(const std::string& value) {
    if (value == "off") return 0;
    if (value == "on") return SIZE_MAX;
    // stoull would wrap a negative rather than refuse it, and a wrapped budget
    // reads as "unlimited" — the opposite of what someone typing -1 wants.
    if (value.empty() || value.front() == '-') throw std::invalid_argument("a thinking budget is a token count");
    return std::stoull(value);
}

static void print_thinking(const TurnBudget& budget) {
    if (!budget.thinks())
        std::printf("[thinking: off]\n");
    else if (budget.thinking == SIZE_MAX)
        std::printf("[thinking: on]\n");
    else
        std::printf("[thinking: up to %zu tokens]\n", budget.thinking);
}

static std::optional<ChatModelKind> requested_kind(const Opts& options, const GGUF& gguf) {
    if (options.arch.empty()) return detect_chat_model(chat_model_identity(gguf));
    if (options.arch == "e4b" || options.arch == "gemma4-e4b") return ChatModelKind::Gemma4E4B;
    if (options.arch == "12b" || options.arch == "gemma4-12b") return ChatModelKind::Gemma4_12B;
    if (options.arch == "qwen35-4b") return ChatModelKind::Qwen35_4B;
    if (options.arch == "qwen35-9b") return ChatModelKind::Qwen35_9B;
    return std::nullopt;
}

static void print_help(bool images) {
    std::printf(
        "  /reset       clear the conversation and prefix cache\n"
        "  /stats       show context and cumulative throughput\n"
        "  /think X     on | off | N, how far the model may think before it answers\n"
        "  /help        show this help\n"
        "  /quit        exit\n");
    if (images)
        std::printf(
            "  /image PATH  attach an image to the next user message\n"
            "  /image clear remove the pending image\n");
}

// One turn on screen. The model says two different things in one stream of
// tokens and the reader wants them apart: the answer is plain text, which is
// what someone is usually here for, and the reasoning is dimmed AND drawn in a
// gutter, so it stays distinguishable in a terminal that ignores the escape
// codes or in a pipe, where they are noise.
//
// This is the ONLY place that knows a channel has a look. The formatters
// upstream decide what a piece IS; nothing but this class writes an escape
// code, and nothing but this class knows the box is drawn with box characters.
class TerminalTranscript {
public:
    void write(ChatChannel channel, std::string_view piece) {
        if (piece.empty()) return;
        if (!started_ || channel != channel_) begin(channel);
        if (channel == ChatChannel::Reasoning)
            write_in_gutter(piece);
        else
            std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
    }

    // Leave the cursor on a line of its own with no styling still in effect,
    // whatever the turn ended in the middle of. What follows is the stats line.
    void finish() {
        if (started_ && channel_ == ChatChannel::Reasoning) close_gutter();
        std::fflush(stdout);
    }

private:
    void begin(ChatChannel channel) {
        if (started_ && channel_ == ChatChannel::Reasoning) close_gutter();
        // An answer may share the prompt's line, since that is the short reply
        // a reader wants to see at once. Anything else is a paragraph of
        // its own.
        if (started_)
            std::fputs("\n\n", stdout);
        else if (channel == ChatChannel::Reasoning)
            std::fputs("\n", stdout);
        else
            std::fputs(" ", stdout);
        if (channel == ChatChannel::Reasoning) {
            std::fputs("\x1b[2m╭─ thinking\n│ ", stdout);
            pending_lines_ = 0;
        }
        started_ = true;
        channel_ = channel;
    }

    // The gutter is per LINE, so a piece that spans a newline is split here.
    // A newline is held until the next real text arrives, which is what keeps
    // the blank line the protocol leaves before `<channel|>` out of the box.
    void write_in_gutter(std::string_view piece) {
        while (!piece.empty()) {
            const size_t newline = piece.find('\n');
            const std::string_view text = piece.substr(0, newline);
            if (!text.empty()) {
                for (; pending_lines_ > 0; --pending_lines_) std::fputs("\n\x1b[2m│ ", stdout);
                std::fwrite(text.data(), 1, text.size(), stdout);
            }
            if (newline == std::string_view::npos) return;
            ++pending_lines_;
            piece.remove_prefix(newline + 1);
        }
    }

    void close_gutter() { std::fputs("\n\x1b[2m╰──────\x1b[0m", stdout); }

    ChatChannel channel_ = ChatChannel::Response;
    size_t pending_lines_ = 0;
    bool started_ = false;
};

// Options by value: /think edits the budget, and a setting the REPL can change
// is the REPL's own.
template <ChatModelInterface Model>
static int chat_loop(Model& model, Opts options) {
    std::printf("\n=== nm chat — %s ===\n", model.description().c_str());
    std::printf("commands: /reset  /stats  /think  /help  /quit%s   (%s sampling)\n", model.supports_images() ? "  /image PATH" : "", options.sampling.temp <= 0 ? "greedy" : "stochastic");
    print_thinking(options.budget);
    std::printf("\n");

    std::string pending_image = options.initial_image;
    std::string line;
    while (true) {
        if (!pending_image.empty()) std::printf("\x1b[2m[image: %s]\x1b[0m\n", pending_image.c_str());
        std::printf("\x1b[1muser>\x1b[0m ");
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line == "/quit" || line == "/exit") break;
        if (line == "/help") {
            print_help(model.supports_images());
            continue;
        }
        if (line == "/reset") {
            model.reset();
            pending_image.clear();
            std::printf("[context cleared]\n");
            continue;
        }
        if (line == "/stats") {
            const ChatStats& stats = model.stats();
            std::printf("[ctx %zu/%zu (%.1f%%)  prefill %.1f tok/s  decode %.1f tok/s", model.context_used(), model.context_limit(), 100.0 * model.context_used() / model.context_limit(), stats.prefill_tps(), stats.decode_tps());
            if (stats.vision_seconds > 0) std::printf("  vision %.2fs", stats.vision_seconds);
            std::printf("]\n");
            continue;
        }
        // The budget changes from the next turn, but the switch already written
        // into the transcript does not: Gemma 4 asks for thinking once, in the
        // first system turn, so a model told to think still opens a thought
        // until /reset. That thought is what the budget then closes — at 0,
        // before a word of it reaches the screen.
        if (line.rfind("/think", 0) == 0 && (line.size() == 6 || std::isspace(static_cast<unsigned char>(line[6])))) {
            size_t begin = 6;
            while (begin < line.size() && std::isspace(static_cast<unsigned char>(line[begin]))) ++begin;
            const std::string value = line.substr(begin);
            try {
                if (!value.empty()) options.budget.thinking = parse_thinking(value);
                print_thinking(options.budget);
            } catch (const std::exception&) {
                std::printf("[usage: /think on, /think off, or /think N to cap the thought at N tokens]\n");
            }
            continue;
        }
        if (line.rfind("/image", 0) == 0 && (line.size() == 6 || std::isspace(static_cast<unsigned char>(line[6])))) {
            if (!model.supports_images()) {
                std::printf("[this model has no image encoder; start Gemma with --mmproj]\n");
                continue;
            }
            size_t begin = 6;
            while (begin < line.size() && std::isspace(static_cast<unsigned char>(line[begin]))) ++begin;
            const std::string path = line.substr(begin);
            if (path.empty()) {
                std::printf("[usage: /image PATH, or /image clear]\n");
            } else if (path == "clear") {
                pending_image.clear();
                std::printf("[pending image cleared]\n");
            } else {
                try {
                    (void)load_rgb_image(path);  // fail early; decode again only when consumed
                    pending_image = path;
                    std::printf("[image attached to the next message]\n");
                } catch (const std::exception& error) {
                    std::printf("[%s]\n", error.what());
                }
            }
            continue;
        }
        if (line.empty()) continue;

        try {
            std::optional<RGBImage> image;
            if (!pending_image.empty()) image.emplace(load_rgb_image(pending_image));
            std::printf("\x1b[1massistant>\x1b[0m");
            std::fflush(stdout);
            TerminalTranscript transcript;
            const ChatTurnResult result = model.respond(line, image ? &*image : nullptr, options.budget, [&transcript](ChatChannel channel, std::string_view piece) { transcript.write(channel, piece); });
            transcript.finish();
            pending_image.clear();
            if (result.truncated) std::printf("\n[response truncated at --max %zu tokens]", options.budget.max_new);
            std::printf(
                "\n\x1b[2m[ Prompt: %zu tok, %.1f t/s | Prefix: %zu tok"
                " | Generation: %zu tok, %.1f t/s",
                result.delta.prefill_tokens, result.delta.prefill_tps(), model.context_used(), result.delta.generated_tokens, result.delta.decode_tps());
            if (result.delta.vision_seconds > 0) std::printf(" | Vision: %.2fs", result.delta.vision_seconds);
            std::printf(" ]\x1b[0m\n\n");
        } catch (const std::exception& error) {
            // The reset matters: a turn that threw mid-reasoning left the
            // gutter's dim attribute switched on.
            std::printf("\x1b[0m\n[error: %s]\n\n", error.what());
        }
    }
    std::printf("\n[bye]\n");
    return 0;
}

template <ChatModelInterface Model>
static int start_chat(Model& model, const Opts& options) {
    if (!options.initial_image.empty() && !model.supports_images()) throw std::invalid_argument("--image requires a Gemma model with --mmproj");
    return chat_loop(model, options);
}

// The one dispatch: metadata picks a case, the case names a concrete adapter,
// and each case instantiates the REPL for that type. The adapter is a named
// local, so its address is stable for as long as it runs — which matters
// because its PrefixCache identifies the model it memoized BY ADDRESS.
static int load_and_chat(const GGUF& gguf, const Opts& options) {
    const auto kind = requested_kind(options, gguf);
    if (!kind) {
        const ChatModelIdentity identity = chat_model_identity(gguf);
        throw std::runtime_error("no compiled chat adapter matches architecture='" + identity.architecture + "' D=" + std::to_string(identity.embedding_length) + " L=" + std::to_string(identity.block_count));
    }

    if (*kind != ChatModelKind::Gemma4E4B && !options.mmproj.empty()) throw std::invalid_argument("--mmproj is only supported by a multimodal model adapter");

    switch (*kind) {
        case ChatModelKind::Gemma4E4B: {
            auto model = make_gemma4_e4b_chat_model(gguf, options.mmproj, options.system.value_or(""), options.sampling);
            return start_chat(model, options);
        }
        case ChatModelKind::Gemma4_12B: {
            auto model = make_gemma4_12b_chat_model(gguf, options.system.value_or(""), options.sampling);
            return start_chat(model, options);
        }
        case ChatModelKind::Qwen35_4B: {
            auto model = make_qwen35_chat_model<Qwen35_4BConfig>(gguf, options.system.value_or(""), options.sampling, "4B");
            return start_chat(model, options);
        }
        case ChatModelKind::Qwen35_9B: {
            auto model = make_qwen35_chat_model<Qwen35_9BConfig>(gguf, options.system.value_or(""), options.sampling, "9B");
            return start_chat(model, options);
        }
    }
    throw std::logic_error("unreachable chat model kind");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s model.gguf [--mmproj FILE] [--image FILE] [--system S]\n"
                     "          [--temp T] [--top-k K] [--top-p P] [--think on|off|N]\n"
                     "          [--seed S] [--max N] [--arch e4b|12b|qwen35-4b|qwen35-9b]\n",
                     argv[0]);
        return 2;
    }

    Opts options;
    options.model = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(argument + " requires a value");
            return argv[++i];
        };
        try {
            if (argument == "--mmproj")
                options.mmproj = next();
            else if (argument == "--image")
                options.initial_image = next();
            else if (argument == "--system")
                options.system = next();
            else if (argument == "--temp")
                options.sampling.temp = std::stof(next());
            else if (argument == "--top-k")
                options.sampling.top_k = std::stoi(next());
            else if (argument == "--top-p")
                options.sampling.top_p = std::stof(next());
            else if (argument == "--seed")
                options.sampling.seed = std::stoull(next());
            else if (argument == "--think")
                options.budget.thinking = parse_thinking(next());
            else if (argument == "--max") {
                options.budget.max_new = std::stoull(next());
                if (options.budget.max_new == 0) options.budget.max_new = SIZE_MAX;
            } else if (argument == "--arch")
                options.arch = next();
            else
                throw std::invalid_argument("unknown option " + argument);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "error: %s\n", error.what());
            return 2;
        }
    }

    try {
        const GGUF gguf(options.model);
        const ChatModelIdentity identity = chat_model_identity(gguf);
        std::fprintf(stderr, "loading %s D=%zu L=%zu (%zu tensors, %zu threads)...\n", identity.architecture.c_str(), identity.embedding_length, identity.block_count, gguf.tensors().size(), nm_num_threads());
        return load_and_chat(gguf, options);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
