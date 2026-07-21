// gemma4_loader.hpp — GGUF schema and immutable assembly for Gemma 4 E4B text.
#pragma once
#include "gemma4.hpp"
#include "gguf.hpp"
#include "tensor_loader.hpp"
#include <sstream>

namespace gemma4_loader {

using C = Gemma4E4BTextConfig;

inline void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("gemma4 loader: " + message);
}

inline void expect_integer(const GGUF& gguf, const std::string& key, size_t expected) {
    const GGUFValue& value = gguf.get(key);
    if (value.type == GGUFType::ARRAY) {
        expect(!value.arr_num.empty(), key + " is an empty array");
        for (double item : value.arr_num)
            expect(size_t(item) == expected,
                   key + " contains " + std::to_string(size_t(item)) +
                   ", expected " + std::to_string(expected));
    } else {
        expect(size_t(value.as_int()) == expected,
               key + "=" + std::to_string(value.as_int()) +
               ", expected " + std::to_string(expected));
    }
}

inline void expect_float(const GGUF& gguf, const std::string& key, Scalar expected,
                         Scalar tolerance = 1e-5f) {
    const Scalar actual = Scalar(gguf.get_float(key, std::numeric_limits<double>::quiet_NaN()));
    expect(std::isfinite(actual) && std::fabs(actual - expected) <= tolerance,
           key + "=" + std::to_string(actual) + ", expected " + std::to_string(expected));
}

inline void validate_e4b_text_metadata(const GGUF& gguf) {
    const std::string arch = gguf.get_str("general.architecture", "?");
    expect(arch == "gemma4", "general.architecture='" + arch + "', expected 'gemma4'");
    expect_integer(gguf, "gemma4.block_count", C::L);
    expect_integer(gguf, "gemma4.embedding_length", C::D);
    expect_integer(gguf, "gemma4.feed_forward_length", C::FF);
    expect_integer(gguf, "gemma4.attention.head_count", C::Hq);
    expect_integer(gguf, "gemma4.attention.head_count_kv", C::Hkv);
    expect_integer(gguf, "gemma4.attention.key_length", C::GLOBAL_HEAD_DIM);
    expect_integer(gguf, "gemma4.attention.value_length", C::GLOBAL_HEAD_DIM);
    expect_integer(gguf, "gemma4.attention.key_length_swa", C::LOCAL_HEAD_DIM);
    expect_integer(gguf, "gemma4.attention.value_length_swa", C::LOCAL_HEAD_DIM);
    expect_integer(gguf, "gemma4.attention.sliding_window", C::SLIDING_WINDOW);
    expect_integer(gguf, "gemma4.attention.shared_kv_layers", C::KV_SHARED_LAYERS);
    expect_integer(gguf, "gemma4.embedding_length_per_layer_input", C::PLE);
    expect_float(gguf, "gemma4.attention.layer_norm_rms_epsilon", C::RMS_EPS, 1e-8f);
    expect_float(gguf, "gemma4.final_logit_softcapping", C::LOGIT_SOFTCAP);
    if (gguf.has("gemma4.context_length"))
        expect_integer(gguf, "gemma4.context_length", C::CTX);
    if (gguf.has("tokenizer.ggml.tokens"))
        expect(gguf.get("tokenizer.ggml.tokens").arr_str.size() == C::V,
               "tokenizer vocabulary size differs from 262144");
}

inline std::string block(size_t layer, const char* suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

inline Scalar optional_scalar(const GGUF& gguf, const std::string& name, Scalar fallback) {
    const TensorInfo* tensor = gguf.find(name);
    if (!tensor) return fallback;
    expect(tensor->nelem() == 1, name + " is not scalar");
    Scalar value = 0;
    dequant_to_f32(tensor->type, tensor->data, &value, 1);
    return value;
}

template <size_t HeadDim>
GemmaQueryOutput<C::D, C::Hq, HeadDim> load_query_output(const GGUF& gguf, size_t layer) {
    constexpr size_t QW = C::Hq * HeadDim;
    return GemmaQueryOutput<C::D, C::Hq, HeadDim>(
        Linear<C::D, QW>(tensor_loader::load_weight<C::D, QW>(gguf, block(layer, "attn_q.weight"))),
        PerHeadNorm<C::Hq, HeadDim, RMSNorm<HeadDim>>(
            RMSNorm<HeadDim>(tensor_loader::load_vector<HeadDim>(gguf, block(layer, "attn_q_norm.weight")), C::RMS_EPS)),
        Linear<QW, C::D>(tensor_loader::load_weight<QW, C::D>(gguf, block(layer, "attn_output.weight"))));
}

template <size_t HeadDim>
GemmaKeyValue<C::D, C::Hkv, HeadDim> load_key_value(const GGUF& gguf, size_t layer) {
    constexpr size_t KW = C::Hkv * HeadDim;
    return GemmaKeyValue<C::D, C::Hkv, HeadDim>(
        Linear<C::D, KW>(tensor_loader::load_weight<C::D, KW>(gguf, block(layer, "attn_k.weight"))),
        PerHeadNorm<C::Hkv, HeadDim, RMSNorm<HeadDim>>(
            RMSNorm<HeadDim>(tensor_loader::load_vector<HeadDim>(gguf, block(layer, "attn_k_norm.weight")), C::RMS_EPS)),
        Linear<C::D, KW>(tensor_loader::load_weight<C::D, KW>(gguf, block(layer, "attn_v.weight"))),
        PerHeadNorm<C::Hkv, HeadDim, RMSNormNoScale<HeadDim>>(
            RMSNormNoScale<HeadDim>(C::RMS_EPS)));
}

template <class Attention>
GemmaE4BDenseLayer<Attention> load_dense_layer(const GGUF& gguf, size_t layer,
                                                Attention attention) {
    using Definition = GemmaDenseDecoderLayerDefinition<
        C::D, C::FF, C::PLE, Attention>;
    auto norm = [&](const char* suffix) {
        return RMSNorm<C::D>(tensor_loader::load_vector<C::D>(gguf, block(layer, suffix)), C::RMS_EPS);
    };
    GeluGatedMLP<C::D, C::FF> mlp(
        Linear<C::D, C::FF>(tensor_loader::load_weight<C::D, C::FF>(gguf, block(layer, "ffn_gate.weight"))),
        Linear<C::D, C::FF>(tensor_loader::load_weight<C::D, C::FF>(gguf, block(layer, "ffn_up.weight"))),
        Linear<C::FF, C::D>(tensor_loader::load_weight<C::FF, C::D>(gguf, block(layer, "ffn_down.weight"))));
    GemmaPerLayerResidual<C::D, C::PLE> ple(
        Linear<C::D, C::PLE>(tensor_loader::load_weight<C::D, C::PLE>(gguf, block(layer, "inp_gate.weight"))),
        Linear<C::PLE, C::D>(tensor_loader::load_weight<C::PLE, C::D>(gguf, block(layer, "proj.weight"))),
        norm("post_norm.weight"));

    return GemmaE4BDenseLayer<Attention>(
        typename Definition::TokenMixerBranch(
            norm("attn_norm.weight"), std::move(attention),
            PostNormalize<RMSNorm<C::D>>(norm("post_attention_norm.weight"))),
        typename Definition::ChannelMixerBranch(
            norm("ffn_norm.weight"), std::move(mlp),
            PostNormalize<RMSNorm<C::D>>(norm("post_ffw_norm.weight"))),
        typename Definition::Tail(
            std::move(ple),
            optional_scalar(gguf, block(layer, "layer_output_scale.weight"), 1.f)));
}

inline GemmaE4BLayer load_layer(const GGUF& gguf, size_t layer) {
    const bool full = C::attention_kind(layer) == GemmaAttentionKind::Full;
    const bool shared = C::shares_kv(layer);
    if (!full && !shared) {
        GemmaE4BLocalOwnAttention attention(
            load_query_output<C::LOCAL_HEAD_DIM>(gguf, layer),
            load_key_value<C::LOCAL_HEAD_DIM>(gguf, layer));
        return load_dense_layer(gguf, layer, std::move(attention));
    }
    if (full && !shared) {
        GemmaE4BGlobalOwnAttention attention(
            load_query_output<C::GLOBAL_HEAD_DIM>(gguf, layer),
            load_key_value<C::GLOBAL_HEAD_DIM>(gguf, layer));
        return load_dense_layer(gguf, layer, std::move(attention));
    }
    if (!full) {
        GemmaE4BLocalSharedAttention attention(load_query_output<C::LOCAL_HEAD_DIM>(gguf, layer));
        return load_dense_layer(gguf, layer, std::move(attention));
    }
    GemmaE4BGlobalSharedAttention attention(load_query_output<C::GLOBAL_HEAD_DIM>(gguf, layer));
    return load_dense_layer(gguf, layer, std::move(attention));
}

inline Gemma4E4BTextModel load_e4b_text(const GGUF& gguf) {
    validate_e4b_text_metadata(gguf);

    GemmaTokenIO<C::D, C::V> token_io(
        tensor_loader::load_weight<C::D, C::V>(gguf, "token_embd.weight"),
        std::sqrt(Scalar(C::D)));

    constexpr size_t PackedPLE = C::PLE * C::L;
    GemmaPerLayerInputs<C::D, C::PLE, C::L, C::V> ple(
        tensor_loader::load_weight<PackedPLE, C::V>(gguf, "per_layer_token_embd.weight"),
        Linear<C::D, PackedPLE>(
            tensor_loader::load_weight<C::D, PackedPLE>(gguf, "per_layer_model_proj.weight")),
        PerHeadNorm<C::L, C::PLE, RMSNorm<C::PLE>>(
            RMSNorm<C::PLE>(tensor_loader::load_vector<C::PLE>(gguf, "per_layer_proj_norm.weight"), C::RMS_EPS)),
        std::sqrt(Scalar(C::PLE)), 1.f / std::sqrt(Scalar(C::D)),
        1.f / std::sqrt(2.f));

    std::vector<GemmaE4BLayer> layers;
    layers.reserve(C::L);
    for (size_t layer = 0; layer < C::L; ++layer)
        layers.push_back(load_layer(gguf, layer));

    RMSNorm<C::D> final_norm(tensor_loader::load_vector<C::D>(gguf, "output_norm.weight"), C::RMS_EPS);
    return Gemma4E4BTextModel(std::move(token_io), std::move(layers),
                              std::move(final_norm), std::move(ple));
}

} // namespace gemma4_loader
