// qwen35_loader.hpp — GGUF schema validation and immutable assembly for
// Qwen 3.5. The loader's job is to fail loudly and specifically: a checkpoint
// that disagrees with the compiled config is a user error, not a crash and not
// a silently wrong model.
#pragma once
#include "gguf.hpp"
#include "qwen35.hpp"
#include "tensor_loader.hpp"
#include <string>

namespace qwen35_loader {

inline void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("qwen35 loader: " + message);
}

inline void expect_integer(const GGUF& gguf, const std::string& key, size_t expected) {
    expect(gguf.has(key), "missing metadata key " + key);
    const size_t actual = size_t(gguf.get_int(key, 0));
    expect(actual == expected, key + "=" + std::to_string(actual) + ", expected " + std::to_string(expected));
}

inline void expect_float(const GGUF& gguf, const std::string& key, Scalar expected, Scalar tolerance) {
    const Scalar actual = Scalar(gguf.get_float(key, std::numeric_limits<double>::quiet_NaN()));
    expect(std::isfinite(actual) && std::fabs(actual - expected) <= tolerance, key + "=" + std::to_string(actual) + ", expected " + std::to_string(expected));
}

inline std::string block(size_t layer, const char* suffix) { return "blk." + std::to_string(layer) + "." + suffix; }

template <class C>
void validate_metadata(const GGUF& gguf) {
    const std::string arch = gguf.get_str("general.architecture", "?");
    expect(arch == "qwen35", "general.architecture='" + arch + "', expected 'qwen35'");

    expect_integer(gguf, "qwen35.block_count", C::L);
    expect_integer(gguf, "qwen35.embedding_length", C::D);
    expect_integer(gguf, "qwen35.feed_forward_length", C::FF);
    expect_integer(gguf, "qwen35.attention.head_count", C::Hq);
    expect_integer(gguf, "qwen35.attention.head_count_kv", C::Hkv);
    expect_integer(gguf, "qwen35.attention.key_length", C::HEAD_DIM);
    expect_integer(gguf, "qwen35.attention.value_length", C::HEAD_DIM);
    expect_float(gguf, "qwen35.attention.layer_norm_rms_epsilon", C::RMS_EPS, 1e-8f);
    expect_float(gguf, "qwen35.rope.freq_base", Scalar(C::ROPE_BASE), Scalar(C::ROPE_BASE) * 1e-6f);

    // Partial rotation: the checkpoint states how many head channels rotate.
    expect_integer(gguf, "qwen35.rope.dimension_count", C::HEAD_DIM * C::ROTARY_NUM / C::ROTARY_DEN);

    // The gated delta network's shape. llama.cpp reuses the SSM key names for
    // these, so read them under those names even though this is not a Mamba.
    expect_integer(gguf, "qwen35.ssm.conv_kernel", C::CONV_WIDTH);
    expect_integer(gguf, "qwen35.ssm.state_size", C::KEY_HEAD_DIM);
    expect_integer(gguf, "qwen35.ssm.group_count", C::KEY_HEADS);
    expect_integer(gguf, "qwen35.ssm.time_step_rank", C::VALUE_HEADS);
    expect_integer(gguf, "qwen35.ssm.inner_size", C::VALUE_HEADS * C::VALUE_HEAD_DIM);

    // The hybrid schedule is metadata, so a checkpoint that changes it must be
    // rejected rather than quietly executed with the compiled-in pattern.
    if (gguf.has("qwen35.full_attention_interval")) expect_integer(gguf, "qwen35.full_attention_interval", C::FULL_ATTENTION_INTERVAL);
    if (gguf.has("qwen35.attention.recurrent_layers")) {
        const GGUFValue& value = gguf.get("qwen35.attention.recurrent_layers");
        expect(value.arr_num.size() >= C::L, "qwen35.attention.recurrent_layers is shorter than block_count");
        for (size_t layer = 0; layer < C::L; ++layer) expect((value.arr_num[layer] != 0) == C::is_recurrent(layer), "layer " + std::to_string(layer) + " disagrees with the compiled hybrid schedule");
    }

    if (gguf.has("qwen35.context_length")) expect(size_t(gguf.get_int("qwen35.context_length", 0)) <= C::CTX, "checkpoint context exceeds the compiled CTX");
    if (gguf.has("tokenizer.ggml.tokens")) expect(gguf.get("tokenizer.ggml.tokens").arr_str.size() == C::V, "tokenizer vocabulary size differs from the compiled V");

    // MTP/NextN is an extra head beyond the main stack. We do not run it, and
    // running the stack without it is correct, but say so rather than let a
    // user wonder why a feature they paid bytes for is inert.
    if (gguf.has("qwen35.nextn_predict_layers") && gguf.get_int("qwen35.nextn_predict_layers", 0) != 0)
        std::fprintf(stderr,
                     "qwen35 loader: checkpoint carries an MTP/NextN head; "
                     "it is not executed (single-token decoding is unaffected)\n");
}

template <class C>
RMSNorm<C::D> load_norm(const GGUF& gguf, const std::string& name) {
    return RMSNorm<C::D>(tensor_loader::load_vector<C::D>(gguf, name), C::RMS_EPS);
}

template <class C>
GatedMLP<C::D, C::FF> load_channel_mixer(const GGUF& gguf, size_t layer) {
    return GatedMLP<C::D, C::FF>(Linear<C::D, C::FF>(tensor_loader::load_weight<C::D, C::FF>(gguf, block(layer, "ffn_gate.weight"))), Linear<C::D, C::FF>(tensor_loader::load_weight<C::D, C::FF>(gguf, block(layer, "ffn_up.weight"))), Linear<C::FF, C::D>(tensor_loader::load_weight<C::FF, C::D>(gguf, block(layer, "ffn_down.weight"))));
}

template <class C>
QwenAttentionMixer<C> load_attention(const GGUF& gguf, size_t layer) {
    using Mixer = QwenAttentionMixer<C>;
    auto head_norm = [&](const char* suffix) { return PerHeadNorm<C::Hq, C::HEAD_DIM, RMSNorm<C::HEAD_DIM>>(RMSNorm<C::HEAD_DIM>(tensor_loader::load_vector<C::HEAD_DIM>(gguf, block(layer, suffix)), C::RMS_EPS)); };
    auto kv_head_norm = [&](const char* suffix) { return PerHeadNorm<C::Hkv, C::HEAD_DIM, RMSNorm<C::HEAD_DIM>>(RMSNorm<C::HEAD_DIM>(tensor_loader::load_vector<C::HEAD_DIM>(gguf, block(layer, suffix)), C::RMS_EPS)); };
    // WQG is one projection at double width: [query | gate] per head.
    return Mixer{.WQG = Linear<C::D, Mixer::GATED_QW>(tensor_loader::load_weight<C::D, Mixer::GATED_QW>(gguf, block(layer, "attn_q.weight"))), .q_norm = head_norm("attn_q_norm.weight"), .WK = Linear<C::D, Mixer::KW>(tensor_loader::load_weight<C::D, Mixer::KW>(gguf, block(layer, "attn_k.weight"))), .k_norm = kv_head_norm("attn_k_norm.weight"), .WV = Linear<C::D, Mixer::KW>(tensor_loader::load_weight<C::D, Mixer::KW>(gguf, block(layer, "attn_v.weight"))), .WO = Linear<Mixer::QW, C::D>(tensor_loader::load_weight<Mixer::QW, C::D>(gguf, block(layer, "attn_output.weight")))};
}

template <class C>
QwenRecurrentMixer<C> load_recurrent(const GGUF& gguf, size_t layer) {
    using Mixer = QwenRecurrentMixer<C>;

    // The depthwise conv is stored taps-contiguous per channel, which is
    // exactly Matrix<CONV_WIDTH> with one row per channel — no transpose.
    const TensorInfo& conv_tensor = gguf.require(block(layer, "ssm_conv1d.weight"));
    tensor_loader::expect(conv_tensor.dims.size() == 2 && conv_tensor.dims[0] == C::CONV_WIDTH && conv_tensor.dims[1] == Mixer::CONV_CHANNELS, block(layer, "ssm_conv1d.weight") + " has the wrong shape");
    Matrix<C::CONV_WIDTH> conv(Mixer::CONV_CHANNELS);
    dequant_to_f32(conv_tensor.type, conv_tensor.data, conv.data(), C::CONV_WIDTH * Mixer::CONV_CHANNELS);

    Vec<C::VALUE_HEADS> decay_scale = tensor_loader::load_vector<C::VALUE_HEADS>(gguf, block(layer, "ssm_a"));
    // Some converters store A_log and others store -exp(A_log). The sign tells
    // them apart unambiguously, and getting it wrong makes the state diverge
    // rather than fail, so normalize here instead of trusting the name.
    for (size_t h = 0; h < C::VALUE_HEADS; ++h)
        if (decay_scale[h] > 0.f) decay_scale[h] = -std::exp(decay_scale[h]);

    // After normalization the scale must be <= 0, so exp(gate) is a contraction.
    // A positive value would make the state grow with every token and the logits
    // diverge a few layers later, which is very hard to read backwards from.
    for (size_t h = 0; h < C::VALUE_HEADS; ++h)
        tensor_loader::expect(decay_scale[h] <= 0.f, block(layer, "ssm_a") + ": decay scale must be <= 0 (-exp(A_log))");

    return Mixer{.WQKV = Linear<C::D, Mixer::CONV_CHANNELS>(tensor_loader::load_weight<C::D, Mixer::CONV_CHANNELS>(gguf, block(layer, "attn_qkv.weight"))), .WZ = Linear<C::D, Mixer::VALUE_WIDTH>(tensor_loader::load_weight<C::D, Mixer::VALUE_WIDTH>(gguf, block(layer, "attn_gate.weight"))), .conv = std::move(conv), .Wbeta = Linear<C::D, C::VALUE_HEADS>(tensor_loader::load_weight<C::D, C::VALUE_HEADS>(gguf, block(layer, "ssm_beta.weight"))), .Walpha = Linear<C::D, C::VALUE_HEADS>(tensor_loader::load_weight<C::D, C::VALUE_HEADS>(gguf, block(layer, "ssm_alpha.weight"))), .dt_bias = tensor_loader::load_vector<C::VALUE_HEADS>(gguf, block(layer, "ssm_dt.bias")), .decay_scale = std::move(decay_scale), .head_norm = PerHeadNorm<C::VALUE_HEADS, C::VALUE_HEAD_DIM, RMSNorm<C::VALUE_HEAD_DIM>>(RMSNorm<C::VALUE_HEAD_DIM>(tensor_loader::load_vector<C::VALUE_HEAD_DIM>(gguf, block(layer, "ssm_norm.weight")), C::RMS_EPS)), .WO = Linear<Mixer::VALUE_WIDTH, C::D>(tensor_loader::load_weight<Mixer::VALUE_WIDTH, C::D>(gguf, block(layer, "ssm_out.weight")))};
}

// Named `post_attention_norm` in the checkpoint, but it is the channel
// mixer's INPUT norm — plain pre-norm, two norms per layer.
template <class C, class Mixer>
Qwen35Block<C, Mixer> load_block(const GGUF& gguf, size_t layer, Mixer mixer) {
    return Qwen35Block<C, Mixer>{load_norm<C>(gguf, block(layer, "attn_norm.weight")), std::move(mixer), load_norm<C>(gguf, block(layer, "post_attention_norm.weight")), load_channel_mixer<C>(gguf, layer)};
}

template <class C>
QwenTokenIO<C> load_token_io(const GGUF& gguf) {
    using OutputWeight = typename C::OutputWeight;
    auto tokens = tensor_loader::load_weight<C::D, C::V>(gguf, "token_embd.weight");
    if constexpr (std::is_same_v<OutputWeight, std::monostate>) {
        expect(gguf.find("output.weight") == nullptr, "config says tied embeddings but the checkpoint has output.weight");
        return QwenTokenIO<C>(std::move(tokens), std::monostate{});
    } else {
        return QwenTokenIO<C>(std::move(tokens), tensor_loader::load_weight<C::D, C::V>(gguf, "output.weight"));
    }
}

template <class C>
Qwen35Model<C> load(const GGUF& gguf) {
    validate_metadata<C>(gguf);
    // One vector per layer kind, in schedule order; the model's constructor
    // checks the 3:1 ratio.
    std::vector<typename Qwen35Model<C>::RecurrentLayer> recurrent;
    std::vector<typename Qwen35Model<C>::AttentionLayer> attention;
    for (size_t layer = 0; layer < C::L; ++layer) {
        if (C::is_recurrent(layer))
            recurrent.push_back(load_block<C>(gguf, layer, load_recurrent<C>(gguf, layer)));
        else
            attention.push_back(load_block<C>(gguf, layer, load_attention<C>(gguf, layer)));
    }
    return Qwen35Model<C>(load_token_io<C>(gguf), std::move(recurrent), std::move(attention), load_norm<C>(gguf, "output_norm.weight"));
}

}  // namespace qwen35_loader
