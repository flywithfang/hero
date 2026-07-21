// components.hpp — stratum 2. Config-free components that take DIMENSIONS as
// template parameters and hold runtime weight data. Linear now carries a
// Weight<In,Out> (fp32 view / dequantised / quantised) instead of an owned Mat;
// everything above this line (Attention, GatedMLP, ...) is unchanged in shape.
#pragma once
#include "core.hpp"
#include "quant.hpp"
#include "quant_i8.hpp"
#include <array>
#include <numeric>

// ---- Linear: y = W·x (+ bias). Llama has no biases; bias stays zeroed. ------
template <size_t In, size_t Out>
class Linear {
public:
    explicit Linear(Weight<In, Out> weight)
        : weight_(std::move(weight)) {}
    Linear(Weight<In, Out> weight, Vec<Out> bias)
        : weight_(std::move(weight)), bias_(std::move(bias)), has_bias_(true) {}

    Vec<Out> operator()(VecView<In> x) const {
        Vec<Out> y = weight_.matvec(x);
        if (has_bias_) y += VecView<Out>(bias_);
        return y;
    }

private:
    Weight<In, Out> weight_;
    Vec<Out> bias_;                    // zero unless a bias tensor is loaded
    bool has_bias_ = false;
};

// Some exported multimodal checkpoints carry calibration bounds next to a
// linear.  Clipping is part of the model, not a quantization implementation
// detail: clamp the activation, apply the weight, then clamp the result.
template <size_t In, size_t Out>
class ClippedLinear {
public:
    ClippedLinear(Linear<In, Out> linear, Scalar input_min, Scalar input_max,
                  Scalar output_min, Scalar output_max)
        : linear_(std::move(linear)), input_min_(input_min), input_max_(input_max),
          output_min_(output_min), output_max_(output_max) {
        if (input_min_ > input_max_ || output_min_ > output_max_)
            throw std::invalid_argument("ClippedLinear: reversed calibration interval");
    }

    Vec<Out> operator()(VecView<In> input) const {
        Vec<In> clamped;
        for (size_t i = 0; i < In; ++i)
            clamped[i] = std::clamp(input[i], input_min_, input_max_);
        Vec<Out> output = linear_(clamped);
        for (size_t i = 0; i < Out; ++i)
            output[i] = std::clamp(output[i], output_min_, output_max_);
        return output;
    }

private:
    Linear<In, Out> linear_;
    Scalar input_min_, input_max_, output_min_, output_max_;
};

template <size_t N>
class LayerNorm {
public:
    LayerNorm(Vec<N> gamma, Vec<N> beta, Scalar eps = 1e-5f)
        : gamma_(std::move(gamma)), beta_(std::move(beta)), eps_(eps) {}

    Vec<N> operator()(VecView<N> x) const {
        Scalar mu = 0;  for (size_t i = 0; i < N; ++i) mu += x[i];      mu /= N;
        Scalar var = 0; for (size_t i = 0; i < N; ++i) var += (x[i]-mu)*(x[i]-mu);
        const Scalar inv = 1.f / std::sqrt(var / N + eps_);
        Vec<N> y;
        for (size_t i = 0; i < N; ++i) y[i] = gamma_[i] * (x[i] - mu) * inv + beta_[i];
        return y;
    }

private:
    Vec<N> gamma_, beta_;
    Scalar eps_;
};

template <size_t N>
class RMSNorm {
public:
    explicit RMSNorm(Vec<N> gamma, Scalar eps = 1e-5f)
        : gamma_(std::move(gamma)), eps_(eps) {}

    Vec<N> operator()(VecView<N> x) const {
        Scalar ms = 0;
        for (size_t i = 0; i < N; ++i) ms += x[i] * x[i];
        const Scalar inv = 1.f / std::sqrt(ms / N + eps_);
        Vec<N> y;
        for (size_t i = 0; i < N; ++i) y[i] = gamma_[i] * x[i] * inv;
        return y;
    }

private:
    Vec<N> gamma_;
    Scalar eps_;                       // llama.attention.layer_norm_rms_epsilon
};

// Gemma applies the same RMS operation without a learned scale to V heads,
// router inputs, and a few branch inputs. Keeping this a distinct type prevents
// a loader from accidentally supplying (or silently dropping) a weight tensor.
template <size_t N>
class RMSNormNoScale {
public:
    explicit RMSNormNoScale(Scalar eps = 1e-6f) : eps_(eps) {}

    Vec<N> operator()(VecView<N> x) const {
        Scalar ms = 0;
        for (size_t i = 0; i < N; ++i) ms += x[i] * x[i];
        const Scalar inv = std::pow(ms / N + eps_, -0.5f);
        Vec<N> y;
        for (size_t i = 0; i < N; ++i) y[i] = x[i] * inv;
        return y;
    }

private:
    Scalar eps_;
};

// Llama-3.2 "rope scaling" (NTK-by-parts / low-freq rescale). Applied to each
// plane's theta BEFORE table build (INFERENCE_PLAN.md §M1). Identity when off.
struct RopeScaling {
    bool   on = false;
    double factor = 8.0;               // llama.rope.scaling.factor (3.2: 32)
    double low_freq_factor = 1.0;
    double high_freq_factor = 4.0;
    double orig_ctx = 8192.0;          // original_context_length
};

// RoPE. Base and MaxPos are compile-time (from the config); scaling is runtime
// (from metadata) and applied in build(). This implementation consumes GGUF's
// interleaved/NORM-paired Llama Q/K weights.
template <size_t Dqk, size_t MaxPos, size_t Base>
class Rope {
    static constexpr size_t P = Dqk / 2;
    std::vector<Scalar> cs_, sn_;                  // [MaxPos x P]

    void build(const RopeScaling& sc) {
        cs_.assign(MaxPos * P, 0.f); sn_.assign(MaxPos * P, 0.f);
        const double two_pi = 2.0 * M_PI;
        for (size_t i = 0; i < P; ++i) {
            double theta = std::pow(double(Base), -double(2 * i) / double(Dqk));
            if (sc.on) {
                const double wavelen = two_pi / theta;
                const double low  = sc.orig_ctx / sc.low_freq_factor;
                const double high = sc.orig_ctx / sc.high_freq_factor;
                if (wavelen > low) {
                    theta = theta / sc.factor;
                } else if (wavelen >= high) {       // blend zone (high<=wl<=low)
                    const double s = (sc.orig_ctx / wavelen - sc.low_freq_factor)
                                   / (sc.high_freq_factor - sc.low_freq_factor);
                    theta = (1.0 - s) * (theta / sc.factor) + s * theta;
                } // else wavelen < high: keep theta
            }
            for (size_t p = 0; p < MaxPos; ++p) {
                cs_[p * P + i] = Scalar(std::cos(double(p) * theta));
                sn_[p * P + i] = Scalar(std::sin(double(p) * theta));
            }
        }
    }

public:
    static constexpr size_t head_dimension = Dqk;
    static constexpr size_t context_length = MaxPos;
    static constexpr size_t frequency_base = Base;

    explicit Rope(const RopeScaling& scaling = {}) { build(scaling); }

    // GGML "NORM" pairing (LLAMA_ROPE_TYPE_NORM): consecutive pairs (2i,2i+1).
    // The GGUF convert script permutes Q/K weights for this convention, so a
    // GGUF checkpoint REQUIRES interleaved pairing — using rotate_half here (the
    // HF/NEOX convention, correct only for unpermuted HF weights) is trap T2 and
    // produces plausible-looking but wrong attention. Verified by logit parity
    // vs llama.cpp on stories260K.
    void apply(MutVecView<Dqk> v, size_t pos) const {
        for (size_t i = 0; i < P; ++i) {
            const Scalar c = cs_[pos * P + i], s = sn_[pos * P + i];
            const Scalar x = v[2 * i], y = v[2 * i + 1];
            v[2 * i]     = x * c - y * s;
            v[2 * i + 1] = x * s + y * c;
        }
    }
};
struct NoRope {};

template <size_t Dqk, size_t W, class R>
void rotate_heads(Vec<W>& v, const R& rope, size_t pos) {
    static_assert(W % Dqk == 0);
    for (size_t h = 0; h < W / Dqk; ++h)
        rope.apply(slice_mut<Dqk>(v, h * Dqk), pos);
}

template <size_t D, size_t FF>
class DenseMLP {
public:
    DenseMLP(Linear<D, FF> up, Linear<FF, D> down)
        : up_(std::move(up)), down_(std::move(down)) {}
    Vec<D> operator()(VecView<D> x) const { Vec<FF> h = up_(x); gelu(h); return down_(h); }

private:
    Linear<D, FF> up_;
    Linear<FF, D> down_;
};

// SwiGLU: silu(gate) ⊙ up, then down. Llama/Qwen/Gemma anatomy.
template <size_t D, size_t FF>
class GatedMLP {
public:
    GatedMLP(Linear<D, FF> gate, Linear<D, FF> up, Linear<FF, D> down)
        : gate_(std::move(gate)), up_(std::move(up)), down_(std::move(down)) {}

    Vec<D> operator()(VecView<D> x) const {
        Vec<FF> g = gate_(x);  silu(g);
        Vec<FF> u = up_(x);
        g *= u;
        return down_(g);
    }

private:
    Linear<D, FF> gate_, up_;
    Linear<FF, D> down_;
};

// Gemma 4 uses gelu_pytorch_tanh(gate) * up rather than SwiGLU.
template <size_t D, size_t FF>
class GeluGatedMLP {
public:
    GeluGatedMLP(Linear<D, FF> gate, Linear<D, FF> up, Linear<FF, D> down)
        : gate_(std::move(gate)), up_(std::move(up)), down_(std::move(down)) {}

    Vec<D> operator()(VecView<D> x) const {
        Vec<FF> g = gate_(x); gelu(g);
        Vec<FF> u = up_(x);
        g *= u;
        return down_(g);
    }

private:
    Linear<D, FF> gate_, up_;
    Linear<FF, D> down_;
};

template <size_t D, size_t FF, size_t NE, size_t TOPK, size_t SHARED = 0>
class MoE {
    static_assert(TOPK <= NE);
public:
    MoE(Linear<D, NE> router,
        std::array<GatedMLP<D, FF>, NE> experts,
        std::array<GatedMLP<D, FF>, SHARED> shared)
        : router_(std::move(router)), experts_(std::move(experts)),
          shared_(std::move(shared)) {}

    Vec<D> operator()(VecView<D> x) const {
        Vec<NE> s = router_(x);
        softmax(std::span<Scalar>(s.begin(), NE));
        std::array<size_t, NE> idx;
        std::iota(idx.begin(), idx.end(), 0u);
        std::partial_sort(idx.begin(), idx.begin() + TOPK, idx.end(),
                          [&](size_t a, size_t b) { return s[a] > s[b]; });
        Scalar norm = 0;
        for (size_t k = 0; k < TOPK; ++k) norm += s[idx[k]];
        Vec<D> y;
        for (size_t k = 0; k < TOPK; ++k) {           // routed experts
            Vec<D> ye = experts_[idx[k]](x);
            for (size_t i = 0; i < D; ++i) y[i] += s[idx[k]] / norm * ye[i];
        }
        for (const auto& sh : shared_) y += sh(x);     // always-on path
        return y;
    }

private:
    Linear<D, NE> router_;
    std::array<GatedMLP<D, FF>, NE> experts_;
    std::array<GatedMLP<D, FF>, SHARED> shared_;
};

// Attention. The four LAWS: q-dim==k-dim (Dqk); v-dim free (Dv); Wo in=Hq*Dv;
// Wo out=D. RoPE never appears below query()/attend — position is the caller's.
template <size_t D, size_t Hq, size_t Hkv, size_t Dqk, size_t Dv>
class Attention {
public:
    static_assert(Hq % Hkv == 0, "GQA groups must divide evenly");
    static constexpr size_t QW = Hq  * Dqk;
    static constexpr size_t KW = Hkv * Dqk;
    static constexpr size_t VW = Hkv * Dv;
    static constexpr size_t OW = Hq  * Dv;
    Attention(Linear<D, QW> q_proj, Linear<D, KW> k_proj,
              Linear<D, VW> v_proj, Linear<OW, D> o_proj)
        : q_proj_(std::move(q_proj)), k_proj_(std::move(k_proj)),
          v_proj_(std::move(v_proj)), o_proj_(std::move(o_proj)) {}

    Vec<QW> query(VecView<D> xn) const { return q_proj_(xn); }
    Vec<KW> key(VecView<D> xn) const { return k_proj_(xn); }
    Vec<VW> value(VecView<D> xn) const { return v_proj_(xn); }
    static constexpr KvHead group(QHead h) { return {h.i / (Hq / Hkv)}; }

    Vec<D> attend(VecView<QW> q, PastView<Hkv, Dqk, Dv> past, size_t Tvis) const {
        Vec<OW> o = par_map<Hq, Dv>([q, past, Tvis](size_t h) {
            return attend_one(slice<Dqk>(q, h * Dqk), past, group(QHead{h}), Tvis);
        });
        return o_proj_(o);
    }
    static Vec<Dv> attend_one(VecView<Dqk> qh, PastView<Hkv, Dqk, Dv> past,
                              KvHead g, size_t Tvis) {
        std::vector<Scalar> s(Tvis);
        for (size_t j = 0; j < Tvis; ++j)              // [GROWS-T][BANDWIDTH]
            s[j] = dot(qh, past.key(j, g)) / std::sqrt(Scalar(Dqk));
        softmax(s);
        Vec<Dv> out;
        for (size_t j = 0; j < Tvis; ++j)
            axpy(s[j], past.value(j, g), out);
        return out;
    }

private:
    Linear<D, QW> q_proj_;
    Linear<D, KW> k_proj_;
    Linear<D, VW> v_proj_;
    Linear<OW, D> o_proj_;
};
