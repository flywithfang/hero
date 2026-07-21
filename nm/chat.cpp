// chat.cpp — model-neutral interactive chat over GGUF models.
//
// Model selection is metadata-driven. The REPL speaks only to ChatModel;
// Llama and Gemma adapters own their templates, caches, sampling and optional
// modality encoders.
#include "src/chat_models.hpp"
#include "src/image_io.hpp"
#include <cstdio>
#include <ctime>
#include <iostream>
#include <optional>

struct Opts {
    std::string model;
    std::string mmproj;
    std::string initial_image;
    std::string arch;
    std::optional<std::string> system;
    SamplerCfg sampling;
    size_t max_new = 1024;
};

static std::string llama_default_system() {
    char date[32];
    std::time_t now = std::time(nullptr);
    std::strftime(date, sizeof date, "%d %b %Y", std::localtime(&now));
    return std::string("Cutting Knowledge Date: December 2023\nToday Date: ") +
           date + "\n\n";
}

static std::optional<ChatModelKind> requested_kind(const Opts& options,
                                                    const GGUF& gguf) {
    if (options.arch.empty()) return detect_chat_model(chat_model_identity(gguf));
    if (options.arch == "1b") return ChatModelKind::Llama32_1B;
    if (options.arch == "3b") return ChatModelKind::Llama32_3B;
    if (options.arch == "8b") return ChatModelKind::Llama3_8B;
    if (options.arch == "e4b" || options.arch == "gemma4-e4b")
        return ChatModelKind::Gemma4E4B;
    return std::nullopt;
}

static std::unique_ptr<ChatModel> load_chat_model(const GGUF& gguf, const Opts& options) {
    const auto kind = requested_kind(options, gguf);
    if (!kind) {
        const ChatModelIdentity identity = chat_model_identity(gguf);
        throw std::runtime_error("no compiled chat adapter matches architecture='" +
            identity.architecture + "' D=" + std::to_string(identity.embedding_length) +
            " L=" + std::to_string(identity.block_count));
    }

    if (*kind != ChatModelKind::Gemma4E4B && !options.mmproj.empty())
        throw std::invalid_argument("--mmproj is only supported by a multimodal model adapter");

    const std::string llama_system = options.system.value_or(llama_default_system());
    const std::string gemma_system = options.system.value_or("");
    switch (*kind) {
        case ChatModelKind::Llama32_1B:
            return std::make_unique<LlamaChatModel<Llama32_1B>>(
                gguf, llama_system, options.sampling);
        case ChatModelKind::Llama32_3B:
            return std::make_unique<LlamaChatModel<Llama32_3B>>(
                gguf, llama_system, options.sampling);
        case ChatModelKind::Llama3_8B:
            return std::make_unique<LlamaChatModel<Llama3_8B>>(
                gguf, llama_system, options.sampling);
        case ChatModelKind::Gemma4E4B:
            return std::make_unique<Gemma4E4BChatModel>(
                gguf, options.mmproj, gemma_system, options.sampling);
    }
    throw std::logic_error("unreachable chat model kind");
}

static void print_help(bool images) {
    std::printf("  /reset       clear the conversation and KV cache\n"
                "  /stats       show context and cumulative throughput\n"
                "  /help        show this help\n"
                "  /quit        exit\n");
    if (images)
        std::printf("  /image PATH  attach an image to the next user message\n"
                    "  /image clear remove the pending image\n");
}

static int chat_loop(ChatModel& model, const Opts& options) {
    std::printf("\n=== nm chat — %s ===\n", model.description().c_str());
    std::printf("commands: /reset  /stats  /help  /quit%s   (%s sampling)\n\n",
                model.supports_images() ? "  /image PATH" : "",
                options.sampling.temp <= 0 ? "greedy" : "stochastic");

    std::string pending_image = options.initial_image;
    std::string line;
    while (true) {
        if (!pending_image.empty())
            std::printf("\x1b[2m[image: %s]\x1b[0m\n", pending_image.c_str());
        std::printf("\x1b[1muser>\x1b[0m ");
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line == "/quit" || line == "/exit") break;
        if (line == "/help") { print_help(model.supports_images()); continue; }
        if (line == "/reset") {
            model.reset();
            pending_image.clear();
            std::printf("[context cleared]\n");
            continue;
        }
        if (line == "/stats") {
            const ChatStats& stats = model.stats();
            std::printf("[ctx %zu/%zu (%.1f%%)  prefill %.1f tok/s  decode %.1f tok/s",
                        model.context_used(), model.context_limit(),
                        100.0 * model.context_used() / model.context_limit(),
                        stats.prefill_tps(), stats.decode_tps());
            if (stats.vision_seconds > 0)
                std::printf("  vision %.2fs", stats.vision_seconds);
            std::printf("]\n");
            continue;
        }
        if (line.rfind("/image", 0) == 0 &&
            (line.size() == 6 || std::isspace(static_cast<unsigned char>(line[6])))) {
            if (!model.supports_images()) {
                std::printf("[this session has no image encoder; start Gemma with --mmproj]\n");
                continue;
            }
            size_t begin = 6;
            while (begin < line.size() && std::isspace(static_cast<unsigned char>(line[begin])))
                ++begin;
            const std::string path = line.substr(begin);
            if (path.empty()) {
                std::printf("[usage: /image PATH, or /image clear]\n");
            } else if (path == "clear") {
                pending_image.clear();
                std::printf("[pending image cleared]\n");
            } else {
                try {
                    (void)load_rgb_image(path); // fail early; decode again only when consumed
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
            std::printf("\x1b[1massistant>\x1b[0m ");
            std::fflush(stdout);
            const ChatTurnResult result = model.respond(
                line, image ? &*image : nullptr, options.max_new,
                [](std::string_view piece) {
                    std::fwrite(piece.data(), 1, piece.size(), stdout);
                    std::fflush(stdout);
                });
            pending_image.clear();
            if (result.truncated)
                std::printf("\n[response truncated at --max %zu tokens]", options.max_new);
            std::printf("\n\x1b[2m[ Prompt: %zu tok, %.1f t/s | Cache: %zu tok"
                        " | Generation: %zu tok, %.1f t/s",
                        result.delta.prefill_tokens, result.delta.prefill_tps(),
                        model.context_used(), result.delta.generated_tokens,
                        result.delta.decode_tps());
            if (result.delta.vision_seconds > 0)
                std::printf(" | Vision: %.2fs", result.delta.vision_seconds);
            std::printf(" ]\x1b[0m\n\n");
        } catch (const std::exception& error) {
            std::printf("\n[error: %s]\n\n", error.what());
        }
    }
    std::printf("\n[bye]\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s model.gguf [--mmproj FILE] [--image FILE] [--system S]\n"
            "          [--temp T] [--top-k K] [--top-p P] [--rep-penalty R]\n"
            "          [--seed S] [--max N] [--arch 1b|3b|8b|e4b]\n", argv[0]);
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
            if      (argument == "--mmproj") options.mmproj = next();
            else if (argument == "--image") options.initial_image = next();
            else if (argument == "--system") options.system = next();
            else if (argument == "--temp") options.sampling.temp = std::stof(next());
            else if (argument == "--top-k") options.sampling.top_k = std::stoi(next());
            else if (argument == "--top-p") options.sampling.top_p = std::stof(next());
            else if (argument == "--rep-penalty") options.sampling.rep_penalty = std::stof(next());
            else if (argument == "--seed") options.sampling.seed = std::stoull(next());
            else if (argument == "--max") {
                options.max_new = std::stoull(next());
                if (options.max_new == 0) options.max_new = SIZE_MAX;
            } else if (argument == "--arch") options.arch = next();
            else throw std::invalid_argument("unknown option " + argument);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "error: %s\n", error.what());
            return 2;
        }
    }

    try {
        GGUF gguf(options.model);
        const ChatModelIdentity identity = chat_model_identity(gguf);
        std::fprintf(stderr, "loading %s D=%zu L=%zu (%zu tensors, %zu threads)...\n",
                     identity.architecture.c_str(), identity.embedding_length,
                     identity.block_count, gguf.tensors().size(), nm_num_threads());
        std::unique_ptr<ChatModel> model = load_chat_model(gguf, options);
        if (!options.initial_image.empty() && !model->supports_images())
            throw std::invalid_argument("--image requires a Gemma session with --mmproj");
        return chat_loop(*model, options);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
