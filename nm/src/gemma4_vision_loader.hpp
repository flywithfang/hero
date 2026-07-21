// gemma4_vision_loader.hpp — immutable Gemma 4 E4B vision assembly from GGUF.
#pragma once
#include "gemma4_vision.hpp"
#include "gguf.hpp"
#include "tensor_loader.hpp"

namespace gemma4_vision_loader {

using C = Gemma4E4BVisionConfig;

inline void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("gemma4 vision loader: " + message);
}

inline void expect_integer(const GGUF& gguf, const std::string& key, size_t expected) {
    const auto actual = size_t(gguf.get(key).as_int());
    expect(actual == expected, key + "=" + std::to_string(actual) +
                               ", expected " + std::to_string(expected));
}

inline void validate_e4b_vision_metadata(const GGUF& gguf) {
    expect(gguf.get_str("general.architecture", "?") == "clip",
           "general.architecture must be 'clip'");
    expect(gguf.get_int("clip.has_vision_encoder", 0) != 0,
           "checkpoint has no vision encoder");
    expect(gguf.get_str("clip.vision.projector_type", "?") == "gemma4v",
           "clip.vision.projector_type must be 'gemma4v'");
    expect_integer(gguf, "clip.vision.embedding_length", C::D);
    expect_integer(gguf, "clip.vision.projection_dim", C::OUT);
    expect_integer(gguf, "clip.vision.block_count", C::L);
    expect_integer(gguf, "clip.vision.attention.head_count", C::H);
    expect_integer(gguf, "clip.vision.feed_forward_length", C::FF);
    expect_integer(gguf, "clip.vision.patch_size", C::PATCH);
    const Scalar eps = Scalar(gguf.get_float("clip.vision.attention.layer_norm_epsilon", -1));
    expect(std::fabs(eps - C::EPS) <= 1e-8f, "unexpected vision RMS epsilon");
}

inline std::string block(size_t layer, const char* suffix) {
    return "v.blk." + std::to_string(layer) + "." + suffix;
}

inline Scalar scalar(const GGUF& gguf, const std::string& name) {
    const TensorInfo& tensor = gguf.require(name);
    expect(tensor.nelem() == 1, name + " is not scalar");
    Scalar result = 0;
    dequant_to_f32(tensor.type, tensor.data, &result, 1);
    return result;
}

template <size_t In, size_t Out>
ClippedLinear<In, Out> load_clipped(const GGUF& gguf, const std::string& prefix) {
    return ClippedLinear<In, Out>(
        Linear<In, Out>(tensor_loader::load_weight<In, Out>(gguf, prefix + ".weight")),
        scalar(gguf, prefix + ".input_min"), scalar(gguf, prefix + ".input_max"),
        scalar(gguf, prefix + ".output_min"), scalar(gguf, prefix + ".output_max"));
}

inline Gemma4E4BVisionEncoder::Layer load_layer(const GGUF& gguf, size_t layer) {
    auto norm = [&](const char* suffix) {
        return RMSNorm<C::D>(tensor_loader::load_vector<C::D>(gguf, block(layer, suffix)), C::EPS);
    };
    auto head_norm = [&](const char* suffix) {
        return RMSNorm<C::HEAD_DIM>(
            tensor_loader::load_vector<C::HEAD_DIM>(gguf, block(layer, suffix)), C::EPS);
    };
    return Gemma4E4BVisionEncoder::Layer(
        norm("ln1.weight"),
        load_clipped<C::D, C::D>(gguf, block(layer, "attn_q")),
        load_clipped<C::D, C::D>(gguf, block(layer, "attn_k")),
        load_clipped<C::D, C::D>(gguf, block(layer, "attn_v")),
        load_clipped<C::D, C::D>(gguf, block(layer, "attn_out")),
        PerHeadNorm<C::H, C::HEAD_DIM, RMSNorm<C::HEAD_DIM>>(
            head_norm("attn_q_norm.weight")),
        PerHeadNorm<C::H, C::HEAD_DIM, RMSNorm<C::HEAD_DIM>>(
            head_norm("attn_k_norm.weight")),
        RMSNormNoScale<C::HEAD_DIM>(C::EPS),
        norm("attn_post_norm.weight"), norm("ln2.weight"),
        load_clipped<C::D, C::FF>(gguf, block(layer, "ffn_gate")),
        load_clipped<C::D, C::FF>(gguf, block(layer, "ffn_up")),
        load_clipped<C::FF, C::D>(gguf, block(layer, "ffn_down")),
        norm("ffn_post_norm.weight"));
}

inline Gemma4E4BVisionEncoder load_e4b_vision(const GGUF& gguf) {
    validate_e4b_vision_metadata(gguf);

    constexpr size_t PatchValues = C::PATCH * C::PATCH * 3;
    const TensorInfo& patch = gguf.require("v.patch_embd.weight");
    expect(patch.type == GT::F32, "v.patch_embd.weight must be F32");
    expect(patch.dims == std::vector<uint64_t>({C::PATCH, C::PATCH, 3, C::D}),
           "v.patch_embd.weight has unexpected shape");
    MatT<PatchValues, C::D> patch_view;
    patch_view.p = reinterpret_cast<const Scalar*>(patch.data);
    Weight<PatchValues, C::D> patch_weight(std::move(patch_view), gguf.keepalive());

    const TensorInfo& positions = gguf.require("v.position_embd.weight");
    expect(positions.type == GT::F32, "v.position_embd.weight must be F32");
    expect(positions.dims == std::vector<uint64_t>({C::D, C::POSITION_ROWS, 2}),
           "v.position_embd.weight has unexpected shape");
    PositionTable2D<C::D, C::POSITION_ROWS> position_table(
        reinterpret_cast<const Scalar*>(positions.data), gguf.keepalive());

    std::vector<Gemma4E4BVisionEncoder::Layer> layers;
    layers.reserve(C::L);
    for (size_t layer = 0; layer < C::L; ++layer)
        layers.push_back(load_layer(gguf, layer));

    Linear<C::D, C::OUT> projection(
        tensor_loader::load_weight<C::D, C::OUT>(gguf, "mm.input_projection.weight"));
    return Gemma4E4BVisionEncoder(std::move(patch_weight), std::move(position_table),
                                  std::move(layers), std::move(projection));
}

} // namespace gemma4_vision_loader
