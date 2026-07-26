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
        for (double item : value.arr_num) expect(size_t(item) == expected, key + " contains " + std::to_string(size_t(item)) + ", expected " + std::to_string(expected));
    } else {
        expect(size_t(value.as_int()) == expected, key + "=" + std::to_string(value.as_int()) + ", expected " + std::to_string(expected));
    }
}

inline void expect_float(const GGUF& gguf, const std::string& key, Scalar expected, Scalar tolerance = 1e-5f) {
    const Scalar actual = Scalar(gguf.get_float(key, std::numeric_limits<double>::quiet_NaN()));
    expect(std::isfinite(actual) && std::fabs(actual - expected) <= tolerance, key + "=" + std::to_string(actual) + ", expected " + std::to_string(expected));
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
    if (gguf.has("gemma4.context_length")) expect_integer(gguf, "gemma4.context_length", C::CTX);
    if (gguf.has("tokenizer.ggml.tokens")) expect(gguf.get("tokenizer.ggml.tokens").arr_str.size() == C::V, "tokenizer vocabulary size differs from 262144");
}

inline std::string block(size_t layer, const char* suffix) { return "blk." + std::to_string(layer) + "." + suffix; }

inline Scalar optional_scalar(const GGUF& gguf, const std::string& name, Scalar fallback) {
    const TensorInfo* tensor = gguf.find(name);
    if (!tensor) return fallback;
    expect(tensor->nelem() == 1, name + " is not scalar");
    Scalar value = 0;
    dequant_to_f32(tensor->type, tensor->data, &value, 1);
    return value;
}

// Shared by every Gemma 4 size: Q/O and K/V construction differ only in the
// head width and KV head count the layer uses.
template <class Cfg, size_t HeadDim>
GemmaQueryOutput<Cfg::D, Cfg::Hq, HeadDim> load_query_output(const GGUF& gguf, size_t layer) {
    constexpr size_t QW = Cfg::Hq * HeadDim;
    return GemmaQueryOutput<Cfg::D, Cfg::Hq, HeadDim>(Linear<Cfg::D, QW>(tensor_loader::load_weight<Cfg::D, QW>(gguf, block(layer, "attn_q.weight"))), PerHeadNorm<Cfg::Hq, HeadDim, RMSNorm<HeadDim>>(RMSNorm<HeadDim>(tensor_loader::load_vector<HeadDim>(gguf, block(layer, "attn_q_norm.weight")), Cfg::RMS_EPS)), Linear<QW, Cfg::D>(tensor_loader::load_weight<QW, Cfg::D>(gguf, block(layer, "attn_output.weight"))));
}

template <class Cfg, size_t Hkv, size_t HeadDim>
GemmaKeyValue<Cfg::D, Hkv, HeadDim> load_key_value(const GGUF& gguf, size_t layer) {
    constexpr size_t KW = Hkv * HeadDim;
    return GemmaKeyValue<Cfg::D, Hkv, HeadDim>(Linear<Cfg::D, KW>(tensor_loader::load_weight<Cfg::D, KW>(gguf, block(layer, "attn_k.weight"))), PerHeadNorm<Hkv, HeadDim, RMSNorm<HeadDim>>(RMSNorm<HeadDim>(tensor_loader::load_vector<HeadDim>(gguf, block(layer, "attn_k_norm.weight")), Cfg::RMS_EPS)), Linear<Cfg::D, KW>(tensor_loader::load_weight<Cfg::D, KW>(gguf, block(layer, "attn_v.weight"))), PerHeadNorm<Hkv, HeadDim, RMSNormNoScale<HeadDim>>(RMSNormNoScale<HeadDim>(Cfg::RMS_EPS)));
}

// Unified K/V: the checkpoint has no attn_v at all on this layer.
template <class Cfg, size_t Hkv, size_t HeadDim>
GemmaUnifiedKeyValue<Cfg::D, Hkv, HeadDim> load_unified_key_value(const GGUF& gguf, size_t layer) {
    constexpr size_t KW = Hkv * HeadDim;
    expect(gguf.find(block(layer, "attn_v.weight")) == nullptr, block(layer, "attn_v.weight") + " is present, but this layer is configured for unified K/V");
    return GemmaUnifiedKeyValue<Cfg::D, Hkv, HeadDim>(Linear<Cfg::D, KW>(tensor_loader::load_weight<Cfg::D, KW>(gguf, block(layer, "attn_k.weight"))), PerHeadNorm<Hkv, HeadDim, RMSNorm<HeadDim>>(RMSNorm<HeadDim>(tensor_loader::load_vector<HeadDim>(gguf, block(layer, "attn_k_norm.weight")), Cfg::RMS_EPS)), PerHeadNorm<Hkv, HeadDim, RMSNormNoScale<HeadDim>>(RMSNormNoScale<HeadDim>(Cfg::RMS_EPS)));
}

template <class Attention>
GemmaE4BDenseLayer<Attention> load_dense_layer(const GGUF& gguf, size_t layer, Attention attention) {
    auto norm = [&](const char* suffix) { return RMSNorm<C::D>(tensor_loader::load_vector<C::D>(gguf, block(layer, suffix)), C::RMS_EPS); };
    GeluGatedMLP<C::D, C::FF> mlp(Linear<C::D, C::FF>(tensor_loader::load_weight<C::D, C::FF>(gguf, block(layer, "ffn_gate.weight"))), Linear<C::D, C::FF>(tensor_loader::load_weight<C::D, C::FF>(gguf, block(layer, "ffn_up.weight"))), Linear<C::FF, C::D>(tensor_loader::load_weight<C::FF, C::D>(gguf, block(layer, "ffn_down.weight"))));
    GemmaPerLayerResidual<C::D, C::PLE> ple(Linear<C::D, C::PLE>(tensor_loader::load_weight<C::D, C::PLE>(gguf, block(layer, "inp_gate.weight"))), Linear<C::PLE, C::D>(tensor_loader::load_weight<C::PLE, C::D>(gguf, block(layer, "proj.weight"))), norm("post_norm.weight"));

    return GemmaE4BDenseLayer<Attention>(norm("attn_norm.weight"), std::move(attention), norm("post_attention_norm.weight"), norm("ffn_norm.weight"), std::move(mlp), norm("post_ffw_norm.weight"), GemmaPerLayerTail<C::D, C::PLE>(std::move(ple), optional_scalar(gguf, block(layer, "layer_output_scale.weight"), 1.f)));
}

inline GemmaE4BLayer load_layer(const GGUF& gguf, size_t layer) {
    const bool full = C::attention_kind(layer) == GemmaAttentionKind::Full;
    const bool shared = C::shares_kv(layer);
    if (!full && !shared) {
        GemmaE4BLocalOwnAttention attention(load_query_output<C, C::LOCAL_HEAD_DIM>(gguf, layer), load_key_value<C, C::Hkv, C::LOCAL_HEAD_DIM>(gguf, layer));
        return load_dense_layer(gguf, layer, std::move(attention));
    }
    if (full && !shared) {
        GemmaE4BGlobalOwnAttention attention(load_query_output<C, C::GLOBAL_HEAD_DIM>(gguf, layer), load_key_value<C, C::Hkv, C::GLOBAL_HEAD_DIM>(gguf, layer));
        return load_dense_layer(gguf, layer, std::move(attention));
    }
    if (!full) {
        GemmaE4BLocalSharedAttention attention(load_query_output<C, C::LOCAL_HEAD_DIM>(gguf, layer));
        return load_dense_layer(gguf, layer, std::move(attention));
    }
    GemmaE4BGlobalSharedAttention attention(load_query_output<C, C::GLOBAL_HEAD_DIM>(gguf, layer));
    return load_dense_layer(gguf, layer, std::move(attention));
}

inline Gemma4E4BTransformer load_e4b_text(const GGUF& gguf) {
    validate_e4b_text_metadata(gguf);

    GemmaTokenIO<C::D, C::V> token_io(tensor_loader::load_weight<C::D, C::V>(gguf, "token_embd.weight"), std::sqrt(Scalar(C::D)));

    constexpr size_t PackedPLE = C::PLE * C::L;
    GemmaPerLayerInputs<C::D, C::PLE, C::L, C::V> ple(tensor_loader::load_weight<PackedPLE, C::V>(gguf, "per_layer_token_embd.weight"), Linear<C::D, PackedPLE>(tensor_loader::load_weight<C::D, PackedPLE>(gguf, "per_layer_model_proj.weight")), PerHeadNorm<C::L, C::PLE, RMSNorm<C::PLE>>(RMSNorm<C::PLE>(tensor_loader::load_vector<C::PLE>(gguf, "per_layer_proj_norm.weight"), C::RMS_EPS)), std::sqrt(Scalar(C::PLE)), 1.f / std::sqrt(Scalar(C::D)), 1.f / std::sqrt(2.f));

    std::vector<GemmaE4BLayer> layers;
    layers.reserve(C::L);
    for (size_t layer = 0; layer < C::L; ++layer) layers.push_back(load_layer(gguf, layer));

    RMSNorm<C::D> final_norm(tensor_loader::load_vector<C::D>(gguf, "output_norm.weight"), C::RMS_EPS);
    Gemma4E4BTextWeights weights(std::move(token_io), std::move(layers), std::move(final_norm), std::move(ple));
    return Gemma4E4BTransformer(std::move(weights));
}

// ============================ Gemma 4 12B Unified ============================

using C12 = Gemma4_12BTextConfig;

// A key whose value may legitimately vary per layer is stored as an array; the
// same key on a uniform model is a scalar. Accept both and check every layer.
inline void expect_per_layer(const GGUF& gguf, const std::string& key, size_t layers, size_t (*expected)(size_t)) {
    expect(gguf.has(key), "missing metadata key " + key);
    const GGUFValue& value = gguf.get(key);
    if (value.type == GGUFType::ARRAY) {
        expect(value.arr_num.size() >= layers, key + " array is shorter than block_count");
        for (size_t layer = 0; layer < layers; ++layer) expect(size_t(value.arr_num[layer]) == expected(layer), key + "[" + std::to_string(layer) + "]=" + std::to_string(size_t(value.arr_num[layer])) + ", expected " + std::to_string(expected(layer)));
        return;
    }
    for (size_t layer = 0; layer < layers; ++layer) expect(size_t(value.as_int()) == expected(layer), key + "=" + std::to_string(value.as_int()) + " but layer " + std::to_string(layer) + " expects " + std::to_string(expected(layer)));
}

inline void validate_12b_text_metadata(const GGUF& gguf) {
    const std::string arch = gguf.get_str("general.architecture", "?");
    expect(arch == "gemma4", "general.architecture='" + arch + "', expected 'gemma4'");
    expect_integer(gguf, "gemma4.block_count", C12::L);
    expect_integer(gguf, "gemma4.embedding_length", C12::D);
    expect_integer(gguf, "gemma4.feed_forward_length", C12::FF);
    expect_integer(gguf, "gemma4.attention.head_count", C12::Hq);
    expect_per_layer(gguf, "gemma4.attention.head_count_kv", C12::L, [](size_t layer) { return C12::kv_heads(layer); });
    expect_integer(gguf, "gemma4.attention.key_length", C12::GLOBAL_HEAD_DIM);
    expect_integer(gguf, "gemma4.attention.value_length", C12::GLOBAL_HEAD_DIM);
    expect_integer(gguf, "gemma4.attention.key_length_swa", C12::LOCAL_HEAD_DIM);
    expect_integer(gguf, "gemma4.attention.value_length_swa", C12::LOCAL_HEAD_DIM);
    expect_integer(gguf, "gemma4.attention.sliding_window", C12::SLIDING_WINDOW);
    expect_float(gguf, "gemma4.attention.layer_norm_rms_epsilon", C12::RMS_EPS, 1e-8f);
    expect_float(gguf, "gemma4.final_logit_softcapping", C12::LOGIT_SOFTCAP);

    // 12B has neither of E4B's two extras; if a checkpoint claims them, it is
    // not this model and must not be silently run as one.
    if (gguf.has("gemma4.attention.shared_kv_layers")) expect_integer(gguf, "gemma4.attention.shared_kv_layers", 0);
    if (gguf.has("gemma4.embedding_length_per_layer_input")) expect_integer(gguf, "gemma4.embedding_length_per_layer_input", 0);
    expect(gguf.find("per_layer_token_embd.weight") == nullptr, "checkpoint carries PLE tensors, which Gemma 4 12B does not have");

    // The sliding/global schedule is metadata; ours is compiled in.
    if (gguf.has("gemma4.attention.sliding_window_pattern")) expect_per_layer(gguf, "gemma4.attention.sliding_window_pattern", C12::L, [](size_t layer) -> size_t { return C12::attention_kind(layer) == GemmaAttentionKind::Sliding ? 1 : 0; });
    if (gguf.has("tokenizer.ggml.tokens")) expect(gguf.get("tokenizer.ggml.tokens").arr_str.size() == C12::V, "tokenizer vocabulary size differs from 262144");
}

template <class Attention>
Gemma12BDenseLayer<Attention> load_12b_dense_layer(const GGUF& gguf, size_t layer, Attention attention) {
    auto norm = [&](const char* suffix) { return RMSNorm<C12::D>(tensor_loader::load_vector<C12::D>(gguf, block(layer, suffix)), C12::RMS_EPS); };
    GeluGatedMLP<C12::D, C12::FF> mlp(Linear<C12::D, C12::FF>(tensor_loader::load_weight<C12::D, C12::FF>(gguf, block(layer, "ffn_gate.weight"))), Linear<C12::D, C12::FF>(tensor_loader::load_weight<C12::D, C12::FF>(gguf, block(layer, "ffn_up.weight"))), Linear<C12::FF, C12::D>(tensor_loader::load_weight<C12::FF, C12::D>(gguf, block(layer, "ffn_down.weight"))));
    return Gemma12BDenseLayer<Attention>(norm("attn_norm.weight"), std::move(attention), norm("post_attention_norm.weight"), norm("ffn_norm.weight"), std::move(mlp), norm("post_ffw_norm.weight"),
                                         // No PLE, but the per-layer output scalar is still present.
                                         GemmaLayerScaleTail<C12::D>(optional_scalar(gguf, block(layer, "layer_output_scale.weight"), 1.f)));
}

inline Gemma12BLayer load_12b_layer(const GGUF& gguf, size_t layer) {
    if (C12::attention_kind(layer) == GemmaAttentionKind::Sliding) return load_12b_dense_layer(gguf, layer, Gemma12BSlidingAttention(load_query_output<C12, C12::LOCAL_HEAD_DIM>(gguf, layer), load_key_value<C12, C12::LOCAL_HKV, C12::LOCAL_HEAD_DIM>(gguf, layer)));
    return load_12b_dense_layer(gguf, layer, Gemma12BGlobalAttention(load_query_output<C12, C12::GLOBAL_HEAD_DIM>(gguf, layer), load_unified_key_value<C12, C12::GLOBAL_HKV, C12::GLOBAL_HEAD_DIM>(gguf, layer)));
}

inline Gemma4_12BTransformer load_12b_text(const GGUF& gguf) {
    validate_12b_text_metadata(gguf);

    GemmaTokenIO<C12::D, C12::V> token_io(tensor_loader::load_weight<C12::D, C12::V>(gguf, "token_embd.weight"), std::sqrt(Scalar(C12::D)));

    std::vector<Gemma12BLayer> layers;
    layers.reserve(C12::L);
    for (size_t layer = 0; layer < C12::L; ++layer) layers.push_back(load_12b_layer(gguf, layer));

    RMSNorm<C12::D> final_norm(tensor_loader::load_vector<C12::D>(gguf, "output_norm.weight"), C12::RMS_EPS);
    return Gemma4_12BTransformer(Gemma4_12BTextWeights(std::move(token_io), std::move(layers), std::move(final_norm)));
}

}  // namespace gemma4_loader
