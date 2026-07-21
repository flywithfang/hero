// llama_loader.hpp - GGUF schema validation and immutable Llama assembly.
// It validates metadata against the selected specification, delegates storage
// construction to tensor_loader, and returns a fully assembled immutable model.
#pragma once
#include "llama.hpp"
#include "tensor_loader.hpp"
#include <cstdio>
#include <sstream>

namespace llama_loader {

inline void expect(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("Llama loader: config/checkpoint mismatch: " + what);
}
template <class T>
void expect_eq(const char* field, T got, T want) {
    if (got != want) {
        std::ostringstream os;
        os << "Llama loader: " << field << " = " << got << " in file, but this binary "
              "was compiled for " << want << " (wrong model for this build)";
        throw std::runtime_error(os.str());
    }
}

inline std::string blk(size_t l, const char* suffix) {
    return "blk." + std::to_string(l) + "." + suffix;
}

// Read the Llama-3.2 rope scaling block from metadata (§M1; key names per the
// llama.cpp convert script — TBD-verify against a real 3.2 dump, T9).
inline RopeScaling read_rope_scaling(const GGUF& g, const std::string& arch, size_t orig_ctx_default) {
    RopeScaling sc;
    std::string type = g.get_str(arch + ".rope.scaling.type", "");
    if (type == "llama3") {
        sc.on = true;
        sc.factor           = g.get_float(arch + ".rope.scaling.factor", 8.0);
        sc.low_freq_factor  = g.get_float(arch + ".rope.scaling.low_freq_factor", 1.0);
        sc.high_freq_factor = g.get_float(arch + ".rope.scaling.high_freq_factor", 4.0);
        sc.orig_ctx         = g.get_float(arch + ".rope.scaling.original_context_length",
                                          double(orig_ctx_default));
    }
    return sc;
}

// The full mapping. C is one of the compiled configs; arch is the GGUF
// general.architecture (all supported configs are "llama").
template <class C>
LlamaModel<C> load(const GGUF& g) {
    const std::string arch = g.get_str("general.architecture", "?");
    expect(arch == "llama", "unsupported architecture '" + arch + "' (this build is Llama)");

    // ---- validate anatomy against metadata (clear diff on mismatch) --------
    expect_eq("block_count",       (size_t)g.get_int(arch + ".block_count", -1), C::L);
    expect_eq("embedding_length",  (size_t)g.get_int(arch + ".embedding_length", -1), C::D);
    expect_eq("head_count",        (size_t)g.get_int(arch + ".attention.head_count", -1), C::Hq);
    expect_eq("head_count_kv",     (size_t)g.get_int(arch + ".attention.head_count_kv", -1), C::Hkv);
    expect_eq("feed_forward_length",(size_t)g.get_int(arch + ".feed_forward_length", -1), C::FF);
    expect_eq("rope.dimension_count",(size_t)g.get_int(arch + ".rope.dimension_count", C::Dqk), C::Dqk);
    // vocab: prefer tokens array, else token_embd second dim.
    if (g.has("tokenizer.ggml.tokens"))
        expect_eq("vocab_size", g.get("tokenizer.ggml.tokens").arr_str.size(), C::V);

    const double rms_eps = g.get_float(arch + ".attention.layer_norm_rms_epsilon", 1e-5);
    using PositionEncoding = typename C::PositionEncoding;
    const double base = g.get_float(
        arch + ".rope.freq_base", double(PositionEncoding::frequency_base));
    if (std::llround(base) != static_cast<long long>(PositionEncoding::frequency_base))
        std::fprintf(stderr,
            "Llama loader: warning: rope.freq_base=%g but compiled base=%zu\n",
            base, PositionEncoding::frequency_base);

    // ---- embeddings + tied/untied unembedding -----------------------------
    auto wte = tensor_loader::load_weight<C::D, C::V>(g, "token_embd.weight");

    using OutputWeight = typename LlamaTokenIO<C>::OutputWeight;
    OutputWeight unembed_w = [&]() -> OutputWeight {
        if constexpr (!std::is_same_v<OutputWeight, std::monostate>) {
            return tensor_loader::load_weight<C::D, C::V>(g, "output.weight");
        } else {
            expect(g.find("output.weight") == nullptr,
                   "output.weight present but config is TIED (tied unembed expected)");
            return std::monostate{};
        }
    }();

    LlamaTokenIO<C> token_io(std::move(wte), std::move(unembed_w));

    // ---- per-block tensors -------------------------------------------------
    using Definition = LlamaBlockDefinition<C>;
    using Norm = typename Definition::Norm;
    using Attn = typename Definition::Attn;
    using Channel = typename Definition::Channel;
    std::vector<LlamaBlock<C>> blocks;
    blocks.reserve(C::L);
    for (size_t l = 0; l < C::L; ++l) {
        Norm ln1(tensor_loader::load_vector<C::D>(g, blk(l, "attn_norm.weight")),
                 Scalar(rms_eps));
        Norm ln2(tensor_loader::load_vector<C::D>(g, blk(l, "ffn_norm.weight")),
                 Scalar(rms_eps));

        Attn attn{
            Linear<C::D, Attn::QW>{tensor_loader::load_weight<C::D, Attn::QW>(g, blk(l, "attn_q.weight"))},
            Linear<C::D, Attn::KW>{tensor_loader::load_weight<C::D, Attn::KW>(g, blk(l, "attn_k.weight"))},
            Linear<C::D, Attn::VW>{tensor_loader::load_weight<C::D, Attn::VW>(g, blk(l, "attn_v.weight"))},
            Linear<Attn::OW, C::D>{tensor_loader::load_weight<Attn::OW, C::D>(g, blk(l, "attn_output.weight"))}
        };

        Channel channel{
            Linear<C::D, C::FF>{tensor_loader::load_weight<C::D, C::FF>(g, blk(l, "ffn_gate.weight"))},
            Linear<C::D, C::FF>{tensor_loader::load_weight<C::D, C::FF>(g, blk(l, "ffn_up.weight"))},
            Linear<C::FF, C::D>{tensor_loader::load_weight<C::FF, C::D>(g, blk(l, "ffn_down.weight"))}
        };
        blocks.emplace_back(
            typename Definition::TokenMixerBranch(std::move(ln1), std::move(attn)),
            typename Definition::ChannelMixerBranch(std::move(ln2), std::move(channel)));
    }
    Norm ln_f(tensor_loader::load_vector<C::D>(g, "output_norm.weight"),
              Scalar(rms_eps));

    // ---- RoPE tables (with Llama-3.2 frequency scaling if present) ---------
    PositionEncoding rope(
        read_rope_scaling(g, arch, /*orig_ctx_default=*/8192));

    return LlamaModel<C>(std::move(token_io), std::move(blocks),
                         std::move(ln_f), std::move(rope));
}

} // namespace llama_loader
