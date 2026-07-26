#include "../src/chat_session.hpp"
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
    check(render_chatml_turn("Hi", true, "Be brief.") ==
              "<|im_start|>system\nBe brief.<|im_end|>\n"
              "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
          "first turn renders the system block then opens the assistant turn");
    check(render_chatml_turn("More", false, "ignored") == "<|im_start|>user\nMore<|im_end|>\n<|im_start|>assistant\n", "later turns do not repeat the system block");

    std::printf("== Gemma chat rendering ==\n");
    {
        const GemmaChatTurn turn = render_gemma_chat_turn("Hello", true, "Be concise.", false);
        check(turn.before_image == "<|turn>system\n<|think|>\nBe concise.<turn|>\n<|turn>user\n", "first turn renders system and thinking control exactly");
        check(turn.after_image == "Hello<turn|>\n<|turn>model\n", "text-only user and model boundaries are exact");
    }
    {
        const GemmaChatTurn turn = render_gemma_chat_turn("Describe it.", false, "ignored", true);
        check(turn.before_image == "<|turn>user\n<|image>", "later image turn opens an image segment without repeating system");
        check(turn.after_image == "<image|>Describe it.<turn|>\n<|turn>model\n", "image close and user text retain canonical order");
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

    std::printf("== Gemma channel streaming ==\n");
    {
        std::string rendered;
        GemmaChannelFormatter formatter([&](std::string_view piece) { rendered += piece; });
        formatter.push("<|channel>");
        formatter.push("thought");
        formatter.push("\nHere is the reasoning.");
        formatter.push("<|channel>");
        formatter.push("final\nThe answer.");
        formatter.flush();
        check(rendered == "\n[thought]\nHere is the reasoning.\n[final]\nThe answer.", "channel protocol becomes readable streaming sections");
    }

    std::printf("\n%s (%d failures)\n", failures ? "CHAT SESSION TESTS FAILED" : "ALL CHAT SESSION TESTS PASSED", failures);
    return failures ? 1 : 0;
}
