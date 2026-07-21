// gemma4.hpp — Gemma 4 shared text primitives and checkpoint configurations.
//
// This file supplies Gemma-specific policies to the uniform transformer core:
// two attention shapes, proportional RoPE, per-head normalization, KV sharing,
// PLE, and logit soft-capping. Vision and audio encoders connect at
// EmbeddedSequence<D> and do not enter these text components.
#pragma once
#include "components.hpp"
#include "transformer.hpp"

enum class GemmaAttentionKind : uint8_t { Sliding, Full };

struct Gemma4E4BTextConfig {
    static constexpr size_t V = 262144;
    static constexpr size_t D = 2560;
    static constexpr size_t L = 42;
    static constexpr size_t Hq = 8;
    static constexpr size_t Hkv = 2;
    static constexpr size_t LOCAL_HEAD_DIM = 256;
    static constexpr size_t GLOBAL_HEAD_DIM = 512;
    static constexpr size_t FF = 10240;
    static constexpr size_t PLE = 256;
    static constexpr size_t CTX = 131072;
    static constexpr size_t SLIDING_WINDOW = 512;
    static constexpr size_t KV_SHARED_LAYERS = 18;
    static constexpr size_t FIRST_KV_SHARED_LAYER = L - KV_SHARED_LAYERS;
    static constexpr size_t LOCAL_ROPE_BASE = 10000;
    static constexpr size_t GLOBAL_ROPE_BASE = 1000000;
    static constexpr Scalar RMS_EPS = 1e-6f;
    static constexpr Scalar LOGIT_SOFTCAP = 30.f;

    static constexpr GemmaAttentionKind attention_kind(size_t layer) {
        return layer % 6 == 5 ? GemmaAttentionKind::Full : GemmaAttentionKind::Sliding;
    }
    static constexpr bool shares_kv(size_t layer) {
        return layer >= FIRST_KV_SHARED_LAYER;
    }
    // The final non-sharing layer of each attention kind supplies all later
    // layers of that kind: local layer 22, full layer 23 for E4B's 42 layers.
    static constexpr bool stores_shared_kv(size_t layer) {
        if (layer >= FIRST_KV_SHARED_LAYER) return false;
        const GemmaAttentionKind kind = attention_kind(layer);
        for (size_t next = layer + 1; next < FIRST_KV_SHARED_LAYER; ++next)
            if (attention_kind(next) == kind) return false;
        return true;
    }
};

static_assert(Gemma4E4BTextConfig::attention_kind(4) == GemmaAttentionKind::Sliding);
static_assert(Gemma4E4BTextConfig::attention_kind(5) == GemmaAttentionKind::Full);
static_assert(Gemma4E4BTextConfig::FIRST_KV_SHARED_LAYER == 24);
static_assert(Gemma4E4BTextConfig::stores_shared_kv(22));
static_assert(Gemma4E4BTextConfig::stores_shared_kv(23));
static_assert(Gemma4E4BTextConfig::shares_kv(24));

template <size_t Heads, size_t HeadDim, class Norm>
class PerHeadNorm {
public:
    explicit PerHeadNorm(Norm norm) : norm_(std::move(norm)) {}

    Vec<Heads * HeadDim> operator()(VecView<Heads * HeadDim> x) const {
        Vec<Heads * HeadDim> y;
        for (size_t h = 0; h < Heads; ++h) {
            Vec<HeadDim> head = norm_(slice<HeadDim>(x, h * HeadDim));
            std::copy(head.begin(), head.end(), y.begin() + h * HeadDim);
        }
        return y;
    }

private:
    Norm norm_; // Gemma shares one HeadDim scale vector across all heads.
};

// Gemma keeps Q/O and K/V as separate construction units because E4B's final
// 18 layers have Q/O weights but deliberately have no K/V tensors.
template <size_t D, size_t Hq, size_t HeadDim>
class GemmaQueryOutput {
public:
    static constexpr size_t QW = Hq * HeadDim;

    GemmaQueryOutput(Linear<D, QW> query, PerHeadNorm<Hq, HeadDim, RMSNorm<HeadDim>> query_norm,
                     Linear<QW, D> output)
        : query_(std::move(query)), query_norm_(std::move(query_norm)),
          output_(std::move(output)) {}

    Vec<QW> query(VecView<D> hidden) const { return query_norm_(query_(hidden)); }
    Vec<D> output(VecView<QW> attended) const { return output_(attended); }

private:
    Linear<D, QW> query_;
    PerHeadNorm<Hq, HeadDim, RMSNorm<HeadDim>> query_norm_;
    Linear<QW, D> output_;
};

template <size_t D, size_t Hkv, size_t HeadDim>
class GemmaKeyValue {
public:
    static constexpr size_t KW = Hkv * HeadDim;

    GemmaKeyValue(Linear<D, KW> key, PerHeadNorm<Hkv, HeadDim, RMSNorm<HeadDim>> key_norm,
                  Linear<D, KW> value,
                  PerHeadNorm<Hkv, HeadDim, RMSNormNoScale<HeadDim>> value_norm)
        : key_(std::move(key)), key_norm_(std::move(key_norm)),
          value_(std::move(value)), value_norm_(std::move(value_norm)) {}

    Vec<KW> key(VecView<D> hidden) const { return key_norm_(key_(hidden)); }
    Vec<KW> value(VecView<D> hidden) const { return value_norm_(value_(hidden)); }

private:
    Linear<D, KW> key_;
    PerHeadNorm<Hkv, HeadDim, RMSNorm<HeadDim>> key_norm_;
    Linear<D, KW> value_;
    PerHeadNorm<Hkv, HeadDim, RMSNormNoScale<HeadDim>> value_norm_;
};

template <size_t D, size_t Hq, size_t Hkv, size_t HeadDim, bool HasKV>
class GemmaAttentionWeights;

template <size_t D, size_t Hq, size_t Hkv, size_t HeadDim>
class GemmaAttentionWeights<D, Hq, Hkv, HeadDim, true> {
public:
    using QueryOutput = GemmaQueryOutput<D, Hq, HeadDim>;
    using KeyValue = GemmaKeyValue<D, Hkv, HeadDim>;

    GemmaAttentionWeights(QueryOutput query_output, KeyValue key_value)
        : query_output_(std::move(query_output)), key_value_(std::move(key_value)) {}

    const QueryOutput& query_output() const { return query_output_; }
    const KeyValue& key_value() const { return key_value_; }

private:
    QueryOutput query_output_;
    KeyValue key_value_;
};

template <size_t D, size_t Hq, size_t Hkv, size_t HeadDim>
class GemmaAttentionWeights<D, Hq, Hkv, HeadDim, false> {
public:
    using QueryOutput = GemmaQueryOutput<D, Hq, HeadDim>;

    explicit GemmaAttentionWeights(QueryOutput query_output)
        : query_output_(std::move(query_output)) {}

    const QueryOutput& query_output() const { return query_output_; }

private:
    QueryOutput query_output_;
};

// Position-aware K/V rows with either full retention (capacity=0) or a ring.
// Logical row zero is always the oldest retained position.
template <size_t Hkv, size_t HeadDim>
class GemmaKVCache {
    static constexpr size_t Width = Hkv * HeadDim;

public:
    explicit GemmaKVCache(size_t capacity = 0) : capacity_(capacity) {}

    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }

    void append(Position position, VecView<Width> key, VecView<Width> value) {
        if (size_ != 0 && position.i <= this->position(size_ - 1).i)
            throw std::invalid_argument("GemmaKVCache: positions must increase");

        if (capacity_ == 0) {
            positions_.push_back(position.i);
            keys_.insert(keys_.end(), key.begin(), key.end());
            values_.insert(values_.end(), value.begin(), value.end());
            ++size_;
            return;
        }

        ensure_bounded_storage();

        size_t physical;
        if (size_ < capacity_) {
            physical = (start_ + size_) % capacity_;
            ++size_;
        } else {
            physical = start_;
            start_ = (start_ + 1) % capacity_;
        }
        positions_[physical] = position.i;
        std::copy(key.begin(), key.end(), keys_.begin() + physical * Width);
        std::copy(value.begin(), value.end(), values_.begin() + physical * Width);
    }

    Position position(size_t logical) const {
        return Position{positions_[physical(logical)]};
    }
    VecView<HeadDim> key(size_t logical, KvHead head) const {
        if (head.i >= Hkv) throw std::out_of_range("GemmaKVCache: KV head out of range");
        return VecView<HeadDim>(keys_.data() + physical(logical) * Width + head.i * HeadDim);
    }
    VecView<HeadDim> value(size_t logical, KvHead head) const {
        if (head.i >= Hkv) throw std::out_of_range("GemmaKVCache: KV head out of range");
        return VecView<HeadDim>(values_.data() + physical(logical) * Width + head.i * HeadDim);
    }

private:
    void ensure_bounded_storage() {
        if (!positions_.empty()) return;
        positions_.resize(capacity_);
        keys_.resize(capacity_ * Width);
        values_.resize(capacity_ * Width);
    }

    size_t physical(size_t logical) const {
        if (logical >= size_) throw std::out_of_range("GemmaKVCache: row out of range");
        return capacity_ == 0 ? logical : (start_ + logical) % capacity_;
    }

    size_t capacity_;
    size_t start_ = 0;
    size_t size_ = 0;
    std::vector<size_t> positions_;
    std::vector<Scalar> keys_;
    std::vector<Scalar> values_;
};

// Pure attention reduction over an already-normalized/rotated Q and cache.
// Gemma 4's attention scale is exactly 1.0. `sliding_window=0` means global.
template <size_t Hq, size_t Hkv, size_t HeadDim>
Vec<Hq * HeadDim> gemma_attend(VecView<Hq * HeadDim> query,
                               const GemmaKVCache<Hkv, HeadDim>& cache,
                               Position query_position, size_t sliding_window = 0,
                               const DecoderVisibility* visibility = nullptr,
                               size_t batch_first_position = 0) {
    static_assert(Hq % Hkv == 0);
    Vec<Hq * HeadDim> output;
    for (size_t h = 0; h < Hq; ++h) {
        const KvHead kv{h / (Hq / Hkv)};
        std::vector<size_t> rows;
        std::vector<Scalar> scores;
        rows.reserve(cache.size());
        scores.reserve(cache.size());
        for (size_t r = 0; r < cache.size(); ++r) {
            const size_t key_position = cache.position(r).i;
            const bool mask_allows = visibility
                ? visibility->allows_absolute(query_position.i, key_position, batch_first_position)
                : key_position <= query_position.i;
            if (!mask_allows) continue;
            if (sliding_window != 0 && key_position + sliding_window <= query_position.i)
                continue;
            rows.push_back(r);
            scores.push_back(dot(slice<HeadDim>(query, h * HeadDim), cache.key(r, kv)));
        }
        if (rows.empty()) throw std::runtime_error("gemma_attend: query has no visible keys");
        softmax(scores);
        Vec<HeadDim> head;
        for (size_t i = 0; i < rows.size(); ++i)
            axpy(scores[i], cache.value(rows[i], kv), head);
        std::copy(head.begin(), head.end(), output.begin() + h * HeadDim);
    }
    return output;
}

// Hugging Face proportional RoPE. `RotaryNum/RotaryDen` controls how many
// frequency planes rotate. Non-rotating planes receive inv_freq=0, hence
// cos=1/sin=0. Exponents retain HeadDim as denominator (not rotary width).
// Pairing here is the official half-split convention; GGUF conversion parity
// will decide whether a storage-level Q/K permutation adapter is required.
template <size_t HeadDim, size_t RotaryNum, size_t RotaryDen, size_t Base>
class GemmaProportionalRope {
    static_assert(HeadDim % 2 == 0);
    static_assert(RotaryDen != 0 && RotaryNum <= RotaryDen);
    static constexpr size_t Planes = HeadDim / 2;
    static constexpr size_t RotaryPlanes = RotaryNum * HeadDim / RotaryDen / 2;

public:
    GemmaProportionalRope() {
        for (size_t i = 0; i < Planes; ++i)
            inv_freq_[i] = i < RotaryPlanes
                ? Scalar(std::pow(double(Base), -double(2 * i) / double(HeadDim)))
                : 0.f;
    }

    void apply(MutVecView<HeadDim> value, size_t position) const {
        for (size_t i = 0; i < Planes; ++i) {
            const Scalar angle = Scalar(position) * inv_freq_[i];
            const Scalar c = std::cos(angle), s = std::sin(angle);
            const Scalar x = value[i], y = value[i + Planes];
            value[i] = x * c - y * s;
            value[i + Planes] = y * c + x * s;
        }
    }

    static constexpr size_t rotary_planes() { return RotaryPlanes; }

private:
    std::array<Scalar, Planes> inv_freq_{};
};

template <size_t HeadDim, size_t Base>
using GemmaDefaultRope = GemmaProportionalRope<HeadDim, 1, 1, Base>;

template <size_t Heads, size_t HeadDim, class RopeT>
void gemma_rotate_heads(Vec<Heads * HeadDim>& value, const RopeT& rope, size_t position) {
    for (size_t h = 0; h < Heads; ++h)
        rope.apply(slice_mut<HeadDim>(value, h * HeadDim), position);
}

template <size_t D>
constexpr Scalar gemma_embedding_scale() {
    // The reference stores the scale at the embedding weight dtype. For BF16
    // checkpoints this deliberately includes BF16 rounding.
    return bf16_to_fp32(fp32_to_bf16(Scalar(std::sqrt(double(D)))));
}

inline Scalar gemma_softcap(Scalar value, Scalar cap) {
    return cap * std::tanh(value / cap);
}

template <size_t N>
void gemma_softcap(Vec<N>& values, Scalar cap) {
    for (size_t i = 0; i < N; ++i) values[i] = gemma_softcap(values[i], cap);
}

// The per-layer part of E4B's PLE residual. Model-level token/context PLE
// construction is separate because multimodal soft tokens have no token id.
template <size_t D, size_t P>
class GemmaPerLayerResidual {
public:
    GemmaPerLayerResidual(Linear<D, P> gate, Linear<P, D> projection,
                          RMSNorm<D> post_norm)
        : gate_(std::move(gate)), projection_(std::move(projection)),
          post_norm_(std::move(post_norm)) {}

    Vec<D> operator()(VecView<D> hidden, VecView<P> per_layer_input) const {
        Vec<P> gated = gate_(hidden);
        gelu(gated);
        for (size_t i = 0; i < P; ++i) gated[i] *= per_layer_input[i];
        Vec<D> branch = projection_(gated);
        branch = post_norm_(branch);
        branch += hidden;
        return branch;
    }

private:
    Linear<D, P> gate_;
    Linear<P, D> projection_;
    RMSNorm<D> post_norm_;
};

// Builds E4B's PLE value for one input embedding. The packed token table and
// context projection both produce L consecutive P-vectors, normalized with a
// shared P-wide scale vector before their 1/sqrt(2) combination.
template <size_t D, size_t P, size_t L, size_t V>
class GemmaPerLayerInputs {
public:
    static constexpr size_t Packed = P * L;

    GemmaPerLayerInputs(Weight<Packed, V> token_embeddings,
                        Linear<D, Packed> model_projection,
                        PerHeadNorm<L, P, RMSNorm<P>> projection_norm,
                        Scalar token_scale, Scalar projection_scale,
                        Scalar combination_scale)
        : token_embeddings_(std::move(token_embeddings)),
          model_projection_(std::move(model_projection)),
          projection_norm_(std::move(projection_norm)),
          token_scale_(token_scale), projection_scale_(projection_scale),
          combination_scale_(combination_scale) {}

    Vec<Packed> operator()(VecView<D> input_embedding, TokenId identity) const {
        Vec<Packed> token = token_embeddings_.dequant_row(size_t(identity));
        for (size_t i = 0; i < Packed; ++i) token[i] *= token_scale_;

        Vec<Packed> context = model_projection_(input_embedding);
        for (size_t i = 0; i < Packed; ++i) context[i] *= projection_scale_;
        context = projection_norm_(context);

        for (size_t i = 0; i < Packed; ++i)
            context[i] = (context[i] + token[i]) * combination_scale_;
        return context;
    }

private:
    Weight<Packed, V> token_embeddings_;
    Linear<D, Packed> model_projection_;
    PerHeadNorm<L, P, RMSNorm<P>> projection_norm_;
    Scalar token_scale_;
    Scalar projection_scale_;
    Scalar combination_scale_;
};

// E4B adds a PLE residual and scalar after the canonical attention + channel
// branches.  Keeping this in the typed tail makes the general block reusable
// without pretending that PLE exists in Llama or every future architecture.
template <size_t D, size_t P>
class GemmaPerLayerTail {
public:
    GemmaPerLayerTail(GemmaPerLayerResidual<D, P> per_layer_residual,
                      Scalar layer_scale)
        : per_layer_residual_(std::move(per_layer_residual)),
          layer_scale_(layer_scale) {}

    Vec<D> operator()(VecView<D> hidden, const VecView<P>& per_layer_input) const {
        Vec<D> output = per_layer_residual_(hidden, per_layer_input);
        for (size_t i = 0; i < D; ++i) output[i] *= layer_scale_;
        return output;
    }

private:
    GemmaPerLayerResidual<D, P> per_layer_residual_;
    Scalar layer_scale_;
};

template <size_t D, size_t FF, size_t P, class AttentionWeights>
struct GemmaDenseDecoderLayerDefinition {
    using Norm = RMSNorm<D>;
    using TokenMixerBranch = ResidualBranch<
        D, Norm, AttentionWeights, PostNormalize<Norm>>;
    using Channel = GeluGatedMLP<D, FF>;
    using ChannelMixerBranch = ResidualBranch<
        D, Norm, Channel, PostNormalize<Norm>>;
    using Tail = GemmaPerLayerTail<D, P>;
    using Type = TransformerBlock<D, TokenMixerBranch, ChannelMixerBranch, Tail>;
};

template <size_t D, size_t FF, size_t P, class AttentionWeights>
using GemmaDenseDecoderLayer =
    typename GemmaDenseDecoderLayerDefinition<D, FF, P, AttentionWeights>::Type;

template <size_t D, size_t V>
class GemmaTokenIO {
public:
    GemmaTokenIO(Weight<D, V> tokens, Scalar input_scale)
        : tokens_(std::move(tokens)), input_scale_(input_scale) {}

    Vec<D> token(TokenId id) const {
        Vec<D> value = tokens_.dequant_row(size_t(id));
        for (size_t i = 0; i < D; ++i) value[i] *= input_scale_;
        return value;
    }
    Vec<V> logits(VecView<D> hidden) const { return tokens_.matvec(hidden); }

private:
    Weight<D, V> tokens_; // tied input embedding and LM head
    Scalar input_scale_;
};

using GemmaE4BLocalOwnAttention = GemmaAttentionWeights<
    Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::Hq, Gemma4E4BTextConfig::Hkv,
    Gemma4E4BTextConfig::LOCAL_HEAD_DIM, true>;
using GemmaE4BLocalSharedAttention = GemmaAttentionWeights<
    Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::Hq, Gemma4E4BTextConfig::Hkv,
    Gemma4E4BTextConfig::LOCAL_HEAD_DIM, false>;
using GemmaE4BGlobalOwnAttention = GemmaAttentionWeights<
    Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::Hq, Gemma4E4BTextConfig::Hkv,
    Gemma4E4BTextConfig::GLOBAL_HEAD_DIM, true>;
using GemmaE4BGlobalSharedAttention = GemmaAttentionWeights<
    Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::Hq, Gemma4E4BTextConfig::Hkv,
    Gemma4E4BTextConfig::GLOBAL_HEAD_DIM, false>;

template <class Attention>
using GemmaE4BDenseLayer = GemmaDenseDecoderLayer<
    Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::FF, Gemma4E4BTextConfig::PLE,
    Attention>;

using GemmaE4BLayer = std::variant<
    GemmaE4BDenseLayer<GemmaE4BLocalOwnAttention>,
    GemmaE4BDenseLayer<GemmaE4BGlobalOwnAttention>,
    GemmaE4BDenseLayer<GemmaE4BLocalSharedAttention>,
    GemmaE4BDenseLayer<GemmaE4BGlobalSharedAttention>>;

struct GemmaE4BLayerSchedule {
    static void validate(const GemmaE4BLayer& layer, size_t index) {
        using C = Gemma4E4BTextConfig;
        const bool full = C::attention_kind(index) == GemmaAttentionKind::Full;
        const bool shared = C::shares_kv(index);
        const size_t expected = !full && !shared ? 0 : full && !shared ? 1 : !full ? 2 : 3;
        if (layer.index() != expected)
            throw std::invalid_argument(
                "Gemma4E4B: layer variant disagrees with attention/KV schedule");
    }
};

using GemmaE4BModelData = GemmaPerLayerInputs<
    Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::PLE,
    Gemma4E4BTextConfig::L, Gemma4E4BTextConfig::V>;

using Gemma4E4BTextModel = TransformerModel<
    Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::V, Gemma4E4BTextConfig::L,
    GemmaTokenIO<Gemma4E4BTextConfig::D, Gemma4E4BTextConfig::V>,
    GemmaE4BLayer, RMSNorm<Gemma4E4BTextConfig::D>, GemmaE4BModelData,
    GemmaE4BLayerSchedule>;

class Gemma4E4BCache {
    using C = Gemma4E4BTextConfig;
    using Local = GemmaKVCache<C::Hkv, C::LOCAL_HEAD_DIM>;
    using Global = GemmaKVCache<C::Hkv, C::GLOBAL_HEAD_DIM>;
    using Entry = std::variant<std::monostate, Local, Global>;

public:
    Gemma4E4BCache() {
        for (size_t i = 0; i < C::L; ++i) {
            if (C::shares_kv(i)) continue;
            if (C::attention_kind(i) == GemmaAttentionKind::Full)
                entries_[i].template emplace<Global>(0);
            else
                entries_[i].template emplace<Local>(C::stores_shared_kv(i) ? 0 : C::SLIDING_WINDOW);
        }
    }

    size_t tokens() const { return tokens_; }
    void advance(size_t count) {
        if (count > C::CTX - tokens_)
            throw std::length_error("Gemma4E4BCache: context exhausted");
        tokens_ += count;
    }

    Local& local(size_t layer) {
        return std::get<Local>(entries_[C::shares_kv(layer) ? 22 : layer]);
    }
    const Local& local(size_t layer) const {
        return std::get<Local>(entries_[C::shares_kv(layer) ? 22 : layer]);
    }
    Global& global(size_t layer) {
        return std::get<Global>(entries_[C::shares_kv(layer) ? 23 : layer]);
    }
    const Global& global(size_t layer) const {
        return std::get<Global>(entries_[C::shares_kv(layer) ? 23 : layer]);
    }

private:
    std::array<Entry, C::L> entries_;
    size_t tokens_ = 0;
};

struct Gemma4E4BArchitecture {
    using C = Gemma4E4BTextConfig;
    static constexpr size_t D = C::D;
    static constexpr size_t V = C::V;
    static constexpr size_t L = C::L;
    static constexpr size_t CTX = C::CTX;
    using Model = Gemma4E4BTextModel;
    using State = Gemma4E4BCache;
    using PreparedInput = std::vector<Vec<C::PLE * C::L>>;
    using LocalRope = GemmaDefaultRope<C::LOCAL_HEAD_DIM, C::LOCAL_ROPE_BASE>;
    using GlobalRope = GemmaProportionalRope<C::GLOBAL_HEAD_DIM, 1, 4, C::GLOBAL_ROPE_BASE>;

    static State make_state() { return {}; }
    static size_t tokens(const State& state) { return state.tokens(); }
    static void advance(State& state, size_t count) { state.advance(count); }

    static EmbeddedSequence<D> embed(const Model& model,
                                     std::span<const TokenId> tokens,
                                     size_t first_position) {
        return embed_text_tokens<D>(tokens, first_position,
            [&](TokenId id) { return model.token_io().token(id); });
    }

    static PreparedInput prepare(const Model& model,
                                 const EmbeddedSequence<D>& input) {
        PreparedInput per_layer;
        per_layer.reserve(input.tokens());
        for (size_t token = 0; token < input.tokens(); ++token)
            per_layer.push_back(model.architecture_data()(
                input.embedding(token), input.ple_token_identity(token)));
        return per_layer;
    }

    static void forward_layer(const Model& model, State& state,
                              const EmbeddedSequence<D>& input,
                              ResidualStream<D>& residual,
                              PreparedInput& per_layer, size_t layer_index) {
        const auto& layer_variant = model.layer(layer_index);
        for (size_t token = 0; token < input.tokens(); ++token) {
            const Position position = input.position(token);
            const VecView<C::PLE> ple = slice<C::PLE>(
                VecView<C::PLE * C::L>(per_layer[token]), layer_index * C::PLE);
            Vec<D> next = std::visit([&](const auto& layer) {
                return layer.forward(
                    residual.token(token), ple,
                    [&](const auto& attention, VecView<D> normalized) {
                        return attention_step(attention, normalized, state,
                                              layer_index, position);
                    },
                    [](const auto& channel, VecView<D> normalized) {
                        return channel(normalized);
                    });
            }, layer_variant);
            residual.set_token(token, next);
        }
    }

    static Vec<V> output(const Model& model, const ResidualStream<D>& residual) {
        Vec<D> last = model.final_norm()(residual.token(residual.tokens() - 1));
        Vec<V> logits = model.token_io().logits(last);
        gemma_softcap(logits, C::LOGIT_SOFTCAP);
        return logits;
    }

private:
    static Vec<D> attention_step(const GemmaE4BLocalOwnAttention& attention,
                                 VecView<D> hidden, State& state,
                                 size_t layer, Position position) {
        auto query = attention.query_output().query(hidden);
        gemma_rotate_heads<C::Hq, C::LOCAL_HEAD_DIM>(query, LocalRope{}, position.i);
        auto key = attention.key_value().key(hidden);
        gemma_rotate_heads<C::Hkv, C::LOCAL_HEAD_DIM>(key, LocalRope{}, position.i);
        auto value = attention.key_value().value(hidden);
        auto& cache = state.local(layer);
        cache.append(position, key, value);
        auto attended = gemma_attend<C::Hq, C::Hkv, C::LOCAL_HEAD_DIM>(
            query, cache, position, C::SLIDING_WINDOW);
        return attention.query_output().output(attended);
    }

    static Vec<D> attention_step(const GemmaE4BLocalSharedAttention& attention,
                                 VecView<D> hidden, State& state,
                                 size_t layer, Position position) {
        auto query = attention.query_output().query(hidden);
        gemma_rotate_heads<C::Hq, C::LOCAL_HEAD_DIM>(query, LocalRope{}, position.i);
        auto attended = gemma_attend<C::Hq, C::Hkv, C::LOCAL_HEAD_DIM>(
            query, state.local(layer), position, C::SLIDING_WINDOW);
        return attention.query_output().output(attended);
    }

    static Vec<D> attention_step(const GemmaE4BGlobalOwnAttention& attention,
                                 VecView<D> hidden, State& state,
                                 size_t layer, Position position) {
        auto query = attention.query_output().query(hidden);
        gemma_rotate_heads<C::Hq, C::GLOBAL_HEAD_DIM>(query, GlobalRope{}, position.i);
        auto key = attention.key_value().key(hidden);
        gemma_rotate_heads<C::Hkv, C::GLOBAL_HEAD_DIM>(key, GlobalRope{}, position.i);
        auto value = attention.key_value().value(hidden);
        auto& cache = state.global(layer);
        cache.append(position, key, value);
        auto attended = gemma_attend<C::Hq, C::Hkv, C::GLOBAL_HEAD_DIM>(query, cache, position);
        return attention.query_output().output(attended);
    }

    static Vec<D> attention_step(const GemmaE4BGlobalSharedAttention& attention,
                                 VecView<D> hidden, State& state,
                                 size_t layer, Position position) {
        auto query = attention.query_output().query(hidden);
        gemma_rotate_heads<C::Hq, C::GLOBAL_HEAD_DIM>(query, GlobalRope{}, position.i);
        auto attended = gemma_attend<C::Hq, C::Hkv, C::GLOBAL_HEAD_DIM>(
            query, state.global(layer), position);
        return attention.query_output().output(attended);
    }
};
