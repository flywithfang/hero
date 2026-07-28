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

// Tensor-loading shorthands shared by every Gemma 4 size. Each returns one
// finished member of a flat layer struct; the layer loaders below just list
// them, tensor name next to the member it feeds.
template <size_t D>
RMSNorm<D> load_norm(const GGUF& gguf, size_t layer, const char* suffix, Scalar eps) {
    return RMSNorm<D>(tensor_loader::load_vector<D>(gguf, block(layer, suffix)), eps);
}

template <size_t In, size_t Out>
Linear<In, Out> load_linear(const GGUF& gguf, size_t layer, const char* suffix) {
    return Linear<In, Out>(tensor_loader::load_weight<In, Out>(gguf, block(layer, suffix)));
}

template <size_t Heads, size_t HeadDim>
PerHeadNorm<Heads, HeadDim, RMSNorm<HeadDim>> load_head_norm(const GGUF& gguf, size_t layer, const char* suffix, Scalar eps) {
    return PerHeadNorm<Heads, HeadDim, RMSNorm<HeadDim>>(RMSNorm<HeadDim>(tensor_loader::load_vector<HeadDim>(gguf, block(layer, suffix)), eps));
}

template <size_t D, size_t FF>
GeluGatedMLP<D, FF> load_gelu_mlp(const GGUF& gguf, size_t layer) {
    return GeluGatedMLP<D, FF>(load_linear<D, FF>(gguf, layer, "ffn_gate.weight"), load_linear<D, FF>(gguf, layer, "ffn_up.weight"), load_linear<FF, D>(gguf, layer, "ffn_down.weight"));
}

inline GemmaPerLayerResidual<C::D, C::PLE> load_ple_residual(const GGUF& gguf, size_t layer) {
    return GemmaPerLayerResidual<C::D, C::PLE>(load_linear<C::D, C::PLE>(gguf, layer, "inp_gate.weight"), load_linear<C::PLE, C::D>(gguf, layer, "proj.weight"), load_norm<C::D>(gguf, layer, "post_norm.weight", C::RMS_EPS));
}

template <size_t Dh>
GemmaE4BLayer<Dh> load_e4b_layer(const GGUF& gguf, size_t layer) {
    using Layer = GemmaE4BLayer<Dh>;
    return Layer{.attn_norm = load_norm<C::D>(gguf, layer, "attn_norm.weight", C::RMS_EPS), .WQ = load_linear<C::D, Layer::Hq * Dh>(gguf, layer, "attn_q.weight"), .q_norm = load_head_norm<Layer::Hq, Dh>(gguf, layer, "attn_q_norm.weight", C::RMS_EPS), .WK = load_linear<C::D, Layer::Hkv * Dh>(gguf, layer, "attn_k.weight"), .k_norm = load_head_norm<Layer::Hkv, Dh>(gguf, layer, "attn_k_norm.weight", C::RMS_EPS), .WV = load_linear<C::D, Layer::Hkv * Dh>(gguf, layer, "attn_v.weight"), .v_norm = PerHeadNorm<Layer::Hkv, Dh, RMSNormNoScale<Dh>>(RMSNormNoScale<Dh>(C::RMS_EPS)), .WO = load_linear<Layer::Hq * Dh, C::D>(gguf, layer, "attn_output.weight"), .post_attn_norm = load_norm<C::D>(gguf, layer, "post_attention_norm.weight", C::RMS_EPS), .ffn_norm = load_norm<C::D>(gguf, layer, "ffn_norm.weight", C::RMS_EPS), .ffn = load_gelu_mlp<C::D, C::FF>(gguf, layer), .post_ffn_norm = load_norm<C::D>(gguf, layer, "post_ffw_norm.weight", C::RMS_EPS), .ple = load_ple_residual(gguf, layer), .layer_output_scale = optional_scalar(gguf, block(layer, "layer_output_scale.weight"), 1.f)};
}

// Shared-KV layers load only Q/O. (The GGUF may still carry attn_k/attn_v
// tensors on these layers — converters emit them — but the model never uses
// them, so they are simply not loaded.)
template <size_t Dh>
GemmaE4BSharedLayer<Dh> load_e4b_shared_layer(const GGUF& gguf, size_t layer) {
    using Layer = GemmaE4BSharedLayer<Dh>;
    return Layer{.attn_norm = load_norm<C::D>(gguf, layer, "attn_norm.weight", C::RMS_EPS), .WQ = load_linear<C::D, Layer::Hq * Dh>(gguf, layer, "attn_q.weight"), .q_norm = load_head_norm<Layer::Hq, Dh>(gguf, layer, "attn_q_norm.weight", C::RMS_EPS), .WO = load_linear<Layer::Hq * Dh, C::D>(gguf, layer, "attn_output.weight"), .post_attn_norm = load_norm<C::D>(gguf, layer, "post_attention_norm.weight", C::RMS_EPS), .ffn_norm = load_norm<C::D>(gguf, layer, "ffn_norm.weight", C::RMS_EPS), .ffn = load_gelu_mlp<C::D, C::FF>(gguf, layer), .post_ffn_norm = load_norm<C::D>(gguf, layer, "post_ffw_norm.weight", C::RMS_EPS), .ple = load_ple_residual(gguf, layer), .layer_output_scale = optional_scalar(gguf, block(layer, "layer_output_scale.weight"), 1.f)};
}

inline Gemma4E4BModel load_e4b_text(const GGUF& gguf) {
    validate_e4b_text_metadata(gguf);

    GemmaTokenIO<C::D, C::V> token_io(tensor_loader::load_weight<C::D, C::V>(gguf, "token_embd.weight"), std::sqrt(Scalar(C::D)));

    constexpr size_t PackedPLE = C::PLE * C::L;
    GemmaPerLayerInputs<C::D, C::PLE, C::L, C::V> ple(tensor_loader::load_weight<PackedPLE, C::V>(gguf, "per_layer_token_embd.weight"), Linear<C::D, PackedPLE>(tensor_loader::load_weight<C::D, PackedPLE>(gguf, "per_layer_model_proj.weight")), PerHeadNorm<C::L, C::PLE, RMSNorm<C::PLE>>(RMSNorm<C::PLE>(tensor_loader::load_vector<C::PLE>(gguf, "per_layer_proj_norm.weight"), C::RMS_EPS)), std::sqrt(Scalar(C::PLE)), 1.f / std::sqrt(Scalar(C::D)), 1.f / std::sqrt(2.f));

    // One vector per layer shape, filled in schedule order so that
    // C::position_in_kind(i) is each layer's slot. The model's constructor
    // checks the four counts against the schedule.
    std::vector<Gemma4E4BModel::SlidingLayer> sliding;
    std::vector<Gemma4E4BModel::GlobalLayer> global;
    std::vector<Gemma4E4BModel::SlidingSharedLayer> sliding_shared;
    std::vector<Gemma4E4BModel::GlobalSharedLayer> global_shared;
    for (size_t layer = 0; layer < C::L; ++layer) {
        const bool full = C::attention_kind(layer) == GemmaAttentionKind::Full;
        const bool shared = C::shares_kv(layer);
        if (!shared && !full) sliding.push_back(load_e4b_layer<C::LOCAL_HEAD_DIM>(gguf, layer));
        else if (!shared) global.push_back(load_e4b_layer<C::GLOBAL_HEAD_DIM>(gguf, layer));
        else if (!full) sliding_shared.push_back(load_e4b_shared_layer<C::LOCAL_HEAD_DIM>(gguf, layer));
        else global_shared.push_back(load_e4b_shared_layer<C::GLOBAL_HEAD_DIM>(gguf, layer));
    }
    return Gemma4E4BModel(std::move(token_io), std::move(ple), std::move(sliding), std::move(global), std::move(sliding_shared), std::move(global_shared), RMSNorm<C::D>(tensor_loader::load_vector<C::D>(gguf, "output_norm.weight"), C::RMS_EPS));
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

inline Gemma12BSlidingLayer load_12b_sliding_layer(const GGUF& gguf, size_t layer) {
    using Layer = Gemma12BSlidingLayer;
    constexpr size_t Dh = Layer::HEAD_DIM;
    return Layer{.attn_norm = load_norm<C12::D>(gguf, layer, "attn_norm.weight", C12::RMS_EPS), .WQ = load_linear<C12::D, Layer::Hq * Dh>(gguf, layer, "attn_q.weight"), .q_norm = load_head_norm<Layer::Hq, Dh>(gguf, layer, "attn_q_norm.weight", C12::RMS_EPS), .WK = load_linear<C12::D, Layer::Hkv * Dh>(gguf, layer, "attn_k.weight"), .k_norm = load_head_norm<Layer::Hkv, Dh>(gguf, layer, "attn_k_norm.weight", C12::RMS_EPS), .WV = load_linear<C12::D, Layer::Hkv * Dh>(gguf, layer, "attn_v.weight"), .v_norm = PerHeadNorm<Layer::Hkv, Dh, RMSNormNoScale<Dh>>(RMSNormNoScale<Dh>(C12::RMS_EPS)), .WO = load_linear<Layer::Hq * Dh, C12::D>(gguf, layer, "attn_output.weight"), .post_attn_norm = load_norm<C12::D>(gguf, layer, "post_attention_norm.weight", C12::RMS_EPS), .ffn_norm = load_norm<C12::D>(gguf, layer, "ffn_norm.weight", C12::RMS_EPS), .ffn = load_gelu_mlp<C12::D, C12::FF>(gguf, layer), .post_ffn_norm = load_norm<C12::D>(gguf, layer, "post_ffw_norm.weight", C12::RMS_EPS), .layer_output_scale = optional_scalar(gguf, block(layer, "layer_output_scale.weight"), 1.f)};
}

// Unified K/V: the checkpoint has no attn_v at all on this layer.
inline Gemma12BGlobalLayer load_12b_global_layer(const GGUF& gguf, size_t layer) {
    using Layer = Gemma12BGlobalLayer;
    constexpr size_t Dh = Layer::HEAD_DIM;
    expect(gguf.find(block(layer, "attn_v.weight")) == nullptr, block(layer, "attn_v.weight") + " is present, but this layer is configured for unified K/V");
    return Layer{.attn_norm = load_norm<C12::D>(gguf, layer, "attn_norm.weight", C12::RMS_EPS), .WQ = load_linear<C12::D, Layer::Hq * Dh>(gguf, layer, "attn_q.weight"), .q_norm = load_head_norm<Layer::Hq, Dh>(gguf, layer, "attn_q_norm.weight", C12::RMS_EPS), .WK = load_linear<C12::D, Layer::Hkv * Dh>(gguf, layer, "attn_k.weight"), .k_norm = load_head_norm<Layer::Hkv, Dh>(gguf, layer, "attn_k_norm.weight", C12::RMS_EPS), .v_norm = PerHeadNorm<Layer::Hkv, Dh, RMSNormNoScale<Dh>>(RMSNormNoScale<Dh>(C12::RMS_EPS)), .WO = load_linear<Layer::Hq * Dh, C12::D>(gguf, layer, "attn_output.weight"), .post_attn_norm = load_norm<C12::D>(gguf, layer, "post_attention_norm.weight", C12::RMS_EPS), .ffn_norm = load_norm<C12::D>(gguf, layer, "ffn_norm.weight", C12::RMS_EPS), .ffn = load_gelu_mlp<C12::D, C12::FF>(gguf, layer), .post_ffn_norm = load_norm<C12::D>(gguf, layer, "post_ffw_norm.weight", C12::RMS_EPS), .layer_output_scale = optional_scalar(gguf, block(layer, "layer_output_scale.weight"), 1.f)};
}

inline Gemma4_12BModel load_12b_text(const GGUF& gguf) {
    validate_12b_text_metadata(gguf);

    GemmaTokenIO<C12::D, C12::V> token_io(tensor_loader::load_weight<C12::D, C12::V>(gguf, "token_embd.weight"), std::sqrt(Scalar(C12::D)));

    std::vector<Gemma12BSlidingLayer> sliding;
    std::vector<Gemma12BGlobalLayer> global;
    for (size_t layer = 0; layer < C12::L; ++layer) {
        if (C12::attention_kind(layer) == GemmaAttentionKind::Sliding)
            sliding.push_back(load_12b_sliding_layer(gguf, layer));
        else
            global.push_back(load_12b_global_layer(gguf, layer));
    }
    return Gemma4_12BModel(std::move(token_io), std::move(sliding), std::move(global), RMSNorm<C12::D>(tensor_loader::load_vector<C12::D>(gguf, "output_norm.weight"), C12::RMS_EPS));
}

}  // namespace gemma4_loader
