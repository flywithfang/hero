// llama.hpp - Llama-family specifications for the uniform transformer core:
// dimensions, concrete component types, cache policy, and layer execution.
#pragma once
#include "components.hpp"
#include "transformer.hpp"
#include <stdexcept>
#include <utility>
#include <variant>

// ---- configs: dimensions plus concrete component/policy types -------------
// Tiny real Llama-arch checkpoint (Karpathy stories260K) — the end-to-end
// runtime smoke test that exercises the whole loader+forward path in ms.
// Shapes read from its GGUF via gguf_dump: D=64 L=5 Hq=8 Hkv=4 Dqk=Dv=8 FF=172,
// V=512, non-tied, rope base 10000, rms. (No param_count assert: it's tiny.)
struct Stories260K {
    static constexpr size_t V=512, D=64, L=5, Hq=8, Hkv=4, Dqk=8, Dv=8,
                            FF=172, CTX=2048;
    using Norm = RMSNorm<D>;
    using PositionEncoding = Rope<Dqk, CTX, 10000>;
    using OutputWeight = Weight<D, V>;
};
struct Llama32_1B {
    static constexpr size_t V=128256, D=2048, L=16, Hq=32, Hkv=8, Dqk=64, Dv=64,
                            FF=8192, CTX=131072;
    using Norm = RMSNorm<D>;
    using PositionEncoding = Rope<Dqk, CTX, 500000>;
    using OutputWeight = std::monostate;
};
struct Llama3_8B {
    static constexpr size_t V=128256, D=4096, L=32, Hq=32, Hkv=8, Dqk=128, Dv=128,
                            FF=14336, CTX=131072;
    using Norm = RMSNorm<D>;
    using PositionEncoding = Rope<Dqk, CTX, 500000>;
    using OutputWeight = Weight<D, V>;
};
struct Llama32_3B {
    static constexpr size_t V=128256, D=3072, L=28, Hq=24, Hkv=8, Dqk=128, Dv=128,
                            FF=8192, CTX=131072;
    using Norm = RMSNorm<D>;
    using PositionEncoding = Rope<Dqk, CTX, 500000>;
    using OutputWeight = std::monostate;
};

template <class C>
constexpr size_t llama_param_count() {
    constexpr size_t QW = C::Hq * C::Dqk, KW = C::Hkv * C::Dqk;
    constexpr size_t VW = C::Hkv * C::Dv, OW = C::Hq  * C::Dv;
    constexpr size_t attn   = C::D * QW + C::D * KW + C::D * VW + OW * C::D;
    constexpr size_t ff = 3 * C::D * C::FF;
    constexpr bool tied = std::is_same_v<typename C::OutputWeight, std::monostate>;
    constexpr size_t embed  = C::V * C::D * (tied ? 1 : 2);
    return embed + C::L * (attn + ff);
}
static_assert(llama_param_count<Llama3_8B>()  > 7'800'000'000 && llama_param_count<Llama3_8B>()  < 8'300'000'000);
static_assert(llama_param_count<Llama32_3B>() > 3'000'000'000 && llama_param_count<Llama32_3B>() < 3'400'000'000);
static_assert(llama_param_count<Llama32_1B>() > 1'150'000'000 && llama_param_count<Llama32_1B>() < 1'300'000'000);

template <class C>
struct LlamaBlockDefinition {
    using Attn = Attention<C::D, C::Hq, C::Hkv, C::Dqk, C::Dv>;
    using Norm = typename C::Norm;
    using Channel = GatedMLP<C::D, C::FF>;
    using TokenMixerBranch = ResidualBranch<C::D, Norm, Attn>;
    using ChannelMixerBranch = ResidualBranch<C::D, Norm, Channel>;
    using Type = TransformerBlock<C::D, TokenMixerBranch, ChannelMixerBranch>;
};

template <class C>
using LlamaBlock = typename LlamaBlockDefinition<C>::Type;

template <class C>
class LlamaTokenIO {
public:
    using TokenWeight = Weight<C::D, C::V>;
    using OutputWeight = typename C::OutputWeight;

    LlamaTokenIO(TokenWeight wte, OutputWeight unembed_w)
        : wte_(std::move(wte)),
          unembed_w_(std::move(unembed_w)) {}

    Vec<C::D> token(TokenId t) const { return wte_.dequant_row(size_t(t)); }

    Vec<C::V> logits(VecView<C::D> h) const {
        if constexpr (std::is_same_v<OutputWeight, std::monostate>)
            return wte_.matvec(h);
        else                   return unembed_w_.matvec(h);
    }

private:
    // token_embd.weight is ggml-shape [D(ne0), V(ne1)] => MatT<D,V>: row v is a
    // token's D-vector. Tied unembed is then a single matvec_T over the same W.
    TokenWeight wte_;
    // separate output projection when NOT tied (e.g. Llama-3-8B).
    [[no_unique_address]] OutputWeight unembed_w_;
};

template <class C>
class LlamaKVCache {
    using Attn = typename LlamaBlockDefinition<C>::Attn;
    static constexpr size_t KW = Attn::KW;
    static constexpr size_t VW = Attn::VW;
    std::array<RowStore<KW>, C::L> K_;
    std::array<RowStore<VW>, C::L> V_;
    size_t T_ = 0;
public:
    static constexpr size_t floats_per_token = C::L * (KW + VW);
    size_t tokens() const { return T_; }
    void   advance(size_t n) { T_ += n; }
    void   deposit(size_t l, const Vec<KW>& k, const Vec<VW>& v) {
        K_[l].append(k); V_[l].append(v);
    }
    PastView<C::Hkv, C::Dqk, C::Dv> past(size_t l) const { return {K_[l], V_[l]}; }
};

template <class C>
struct LlamaArchitecture {
    static constexpr size_t D = C::D;
    static constexpr size_t V = C::V;
    static constexpr size_t L = C::L;
    static constexpr size_t CTX = C::CTX;

    using FinalNorm = typename LlamaBlockDefinition<C>::Norm;
    using PositionEncoding = typename C::PositionEncoding;
    using Model = TransformerModel<D, V, L, LlamaTokenIO<C>, LlamaBlock<C>,
                                   FinalNorm, PositionEncoding>;
    using State = LlamaKVCache<C>;
    using PreparedInput = NoPreparedInput;

    static State make_state() { return {}; }
    static size_t tokens(const State& state) { return state.tokens(); }
    static void advance(State& state, size_t count) { state.advance(count); }

    static EmbeddedSequence<D> embed(const Model& model,
                                     std::span<const TokenId> ids,
                                     size_t first_position) {
        return embed_text_tokens<D>(ids, first_position,
            [&](TokenId id) { return model.token_io().token(id); });
    }

    static PreparedInput prepare(const Model&, const EmbeddedSequence<D>&) {
        return {};
    }

    static void forward_layer(const Model& model, State& state,
                              const EmbeddedSequence<D>& input,
                              ResidualStream<D>& residual, PreparedInput&,
                              size_t layer_index) {
        const LlamaBlock<C>& layer = model.layer(layer_index);
        const auto& token_mixer_branch = layer.token_mixer_branch();
        const auto& attention = token_mixer_branch.operation();
        TokenMatrix<D> normalized(input.tokens());

        // Commit the complete K/V batch before evaluating causal queries, just
        // as the original Llama forward path did.
        for (size_t token = 0; token < input.tokens(); ++token) {
            Vec<D> value = token_mixer_branch.normalize(residual.token(token));
            normalized.set_row(token, value);
            auto key = attention.key(value);
            rotate_heads<C::Dqk>(key, model.architecture_data(),
                                  input.position(token).i);
            state.deposit(layer_index, key, attention.value(value));
        }

        for (size_t token = 0; token < input.tokens(); ++token) {
            Vec<D> value = token_mixer_branch.finish(
                residual.token(token), normalized.row(token),
                [&](const auto& operation, VecView<D> normalized_token) {
                    auto query = operation.query(normalized_token);
                    rotate_heads<C::Dqk>(query, model.architecture_data(),
                                          input.position(token).i);
                    return operation.attend(
                        query, state.past(layer_index), input.position(token).i + 1);
                });
            residual.set_token(token, value);
        }

        for (size_t token = 0; token < input.tokens(); ++token) {
            Vec<D> value = layer.channel_mixer_branch().forward(
                residual.token(token),
                [](const auto& channel, VecView<D> normalized) {
                    return channel(normalized);
                });
            residual.set_token(token, value);
        }
    }

    static Vec<V> output(const Model& model, const ResidualStream<D>& residual) {
        Vec<D> last = model.final_norm()(residual.token(residual.tokens() - 1));
        return model.token_io().logits(last);
    }
};

template <class C>
using LlamaModel = typename LlamaArchitecture<C>::Model;
