#include "../tools/chat_session.hpp"
#include "../src/image_io.hpp"
#include <cstdio>

static int failures = 0;
static void check(bool condition, const char* message) {
    std::printf("  [%s] %s\n", condition ? "PASS" : "FAIL", message);
    failures += !condition;
}

int main(int argc, char** argv) {
    std::printf("== model selection ==\n");
    check(detect_chat_model({"gemma4", 2560, 42}) == ChatModelKind::Gemma4E4B, "Gemma 4 E4B is selected from GGUF anatomy");
    check(!detect_chat_model({"gemma4", 4096, 42}), "an unimplemented size in a known family is rejected, not guessed");
    check(detect_chat_model({"qwen35", 2560, 32}) == ChatModelKind::Qwen35_4B, "Qwen 3.5 4B is selected from GGUF anatomy");
    check(detect_chat_model({"qwen35", 4096, 32}) == ChatModelKind::Qwen35_9B, "Qwen 3.5 9B is separated from 4B by width, since both are L=32");
    check(!detect_chat_model({"qwen35", 2560, 42}), "a known architecture at an unimplemented depth is rejected");

    std::printf("== ChatML rendering ==\n");
    check(render_chatml_turn("Hi", true, "Be brief.", true) ==
              "<|im_start|>system\nBe brief.<|im_end|>\n"
              "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
          "first turn renders the system block then opens the assistant turn");
    check(render_chatml_turn("More", false, "ignored", true) == "<|im_start|>user\nMore<|im_end|>\n<|im_start|>assistant\n", "later turns do not repeat the system block");
    check(render_chatml_turn("More", false, "", false) == "<|im_start|>user\nMore<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", "thinking off prefills the closed, empty thought the template uses to say so");

    std::printf("== Gemma chat rendering ==\n");
    {
        const ChatTurnText turn = render_gemma_chat_turn("Hello", true, "Be concise.", false, true);
        check(turn.before_image == "<|turn>system\n<|think|>\nBe concise.<turn|>\n<|turn>user\n", "first turn renders system and thinking control exactly");
        check(turn.after_image == "Hello<turn|>\n<|turn>model\n", "text-only user and model boundaries are exact");
    }
    {
        const ChatTurnText turn = render_gemma_chat_turn("Describe it.", false, "ignored", true, true);
        check(turn.before_image == "<|turn>user\n<|image>", "later image turn opens an image segment without repeating system");
        check(turn.after_image == "<image|>Describe it.<turn|>\n<|turn>model\n", "image close and user text retain canonical order");
    }
    {
        const ChatTurnText turn = render_gemma_chat_turn("Hello", true, "Be concise.", false, false);
        check(turn.before_image == "<|turn>system\nBe concise.<turn|>\n<|turn>user\n", "thinking off drops the <|think|> switch but keeps the system prompt");
        check(render_gemma_chat_turn("Hello", true, "", false, false).before_image == "<|turn>user\n", "a system turn with neither switch nor prompt is not opened at all");
    }

    std::printf("== application image boundary ==\n");
    {
        if (argc < 2 || argc > 3) throw std::invalid_argument("expected PPM fixture path");
        const RGBImage image = load_rgb_image(argv[1]);
        check(image.width() == 2 && image.height() == 2, "dependency-free PPM loader preserves dimensions");
        check(image.pixels()[0] == 255 && image.pixels()[1] == 0 && image.pixels()[2] == 0 && image.pixels()[9] == 255, "PPM loader produces interleaved RGB for the encoder contract");
        if (argc == 3) {
            const RGBImage platform_image = load_rgb_image(argv[2]);
            check(platform_image.width() == 2 && platform_image.height() == 2, "platform decoder accepts a common image format");
            const bool same_pixels = std::equal(platform_image.pixels().begin(), platform_image.pixels().end(), image.pixels().begin(), image.pixels().end());
            if (!same_pixels) {
                std::printf("    decoded:");
                for (uint8_t value : platform_image.pixels()) std::printf(" %u", unsigned(value));
                std::printf("\n");
            }
            check(same_pixels, "platform decoder preserves RGB orientation and channel order");
        }
    }

    std::printf("== channel streaming ==\n");
    // A formatter's whole job is to say which channel a piece belongs to, so
    // the fixture records the channel with the text ("R:" reasoning, "A:"
    // answer) and asserts on both.
    std::string rendered;
    const ChatTokenSink record = [&rendered](ChatChannel channel, std::string_view piece) {
        rendered += channel == ChatChannel::Reasoning ? "R:" : "A:";
        rendered += piece;
    };
    {
        GemmaChannelFormatter formatter(record);
        formatter.push("<|channel>");
        formatter.push("thou");
        formatter.push("ght");
        formatter.push("\nHere is the reasoning.");
        formatter.push("\n");
        formatter.push("<channel|>");
        formatter.push("\n");
        formatter.push("The answer.");
        formatter.flush();
        check(rendered == "R:Here is the reasoning.R:\nA:The answer.", "Gemma's channel splits reasoning from the answer, and both delimiters and the name are consumed");
    }
    {
        rendered.clear();
        GemmaChannelFormatter formatter(record);
        formatter.push("Straight to the point.");
        formatter.flush();
        check(rendered == "A:Straight to the point.", "a Gemma turn that opens no channel is all answer");
    }
    {
        rendered.clear();
        ThinkTagFormatter formatter(record);
        formatter.push("<think>");
        formatter.push("\n");
        formatter.push("Work it out.");
        check(formatter.thinking(), "an open <think> tag is what a thinking budget is spent against");
        formatter.push("</think>");
        check(!formatter.thinking(), "the closing tag ends the thought, whoever wrote it");
        formatter.push("\n\nThe answer.");
        formatter.flush();
        check(rendered == "R:Work it out.A:The answer.", "<think> tags split the channels and the protocol's own newlines are dropped");
    }
    {
        // What a zero budget relies on: the thought is already open at the
        // token that opens it, so it can be closed before any of it is shown.
        rendered.clear();
        GemmaChannelFormatter formatter(record);
        formatter.push("<|channel>");
        check(formatter.thinking(), "a Gemma thought is open from its opening token, before the name arrives");
        formatter.push(std::string(GemmaChatProtocol::THOUGHT_END));
        check(!formatter.thinking(), "the protocol's own closing text ends it");
        formatter.push("Straight to the answer.");
        formatter.flush();
        check(rendered == "A:Straight to the answer.", "a thought closed at once shows nothing at all");
    }

    std::printf("\n%s (%d failures)\n", failures ? "CHAT SESSION TESTS FAILED" : "ALL CHAT SESSION TESTS PASSED", failures);
    return failures ? 1 : 0;
}
