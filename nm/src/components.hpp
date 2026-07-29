// components.hpp — stratum 2. Config-free components that take DIMENSIONS as
// template parameters and hold runtime weight data: projections, norms, and
// channel mixers. Linear carries a Weight<In,Out> (fp32 view / dequantised /
// quantised) rather than an owned Mat.
//
// Token mixing (cache, mask, GQA reduction, RoPE) lives in attention.hpp,
// which builds on this file.
#pragma once
#include "core.hpp"
#include "quant.hpp"
#include "quant_i8.hpp"
#include <array>
#include <numeric>

// ---- Linear: y = W·x (+ bias). Most decoder projections have no bias, and
// the bias stays zeroed for them; Qwen-family Q/K/V projections do carry one.

template <size_t In, size_t Out>
class Linear {
public:
    explicit Linear(Weight<In, Out> weight) : weight_(std::move(weight)) {}
    Linear(Weight<In, Out> weight, Vec<Out> bias) : weight_(std::move(weight)), bias_(std::move(bias)), has_bias_(true) {}

    Vec<Out> operator()(VecView<In> x) const {
        Vec<Out> y = weight_.matvec(x);
        if (has_bias_) y += VecView<Out>(bias_);
        return y;
    }
    Matrix<Out> operator()(MatrixView<In> x) const {
        Matrix<Out> y = weight_.matmul(x);
        if (has_bias_) add_bias_in_place(y.mutable_view(), VecView<Out>(bias_));
        return y;
    }

private:
    Weight<In, Out> weight_;
    Vec<Out> bias_;  // zero unless a bias tensor is loaded
    bool has_bias_ = false;
};

// Some exported multimodal checkpoints carry calibration bounds next to a
// linear.  Clipping is part of the model, not a quantization implementation
// detail: clamp the activation, apply the weight, then clamp the result.
template <size_t In, size_t Out>
class ClippedLinear {
public:
    ClippedLinear(Linear<In, Out> linear, Scalar input_min, Scalar input_max, Scalar output_min, Scalar output_max) : linear_(std::move(linear)), input_min_(input_min), input_max_(input_max), output_min_(output_min), output_max_(output_max) {
        if (input_min_ > input_max_ || output_min_ > output_max_) throw std::invalid_argument("ClippedLinear: reversed calibration interval");
    }

    Vec<Out> operator()(VecView<In> input) const {
        Vec<In> clamped = clamp(input, input_min_, input_max_);
        return clamp(VecView<Out>(linear_(clamped)), output_min_, output_max_);
    }
    Matrix<Out> operator()(MatrixView<In> input) const {
        Matrix<In> clamped = clamp(input, input_min_, input_max_);
        return clamp(linear_(clamped.view()).view(), output_min_, output_max_);
    }

private:
    Linear<In, Out> linear_;
    Scalar input_min_, input_max_, output_min_, output_max_;
};

template <size_t N>
class RMSNorm {
public:
    explicit RMSNorm(Vec<N> gamma, Scalar eps = 1e-5f) : gamma_(std::move(gamma)), eps_(eps) {}

    Vec<N> operator()(VecView<N> x) const { return rms_norm(x, VecView<N>(gamma_), eps_); }
    Matrix<N> operator()(MatrixView<N> x) const { return rms_norm(x, VecView<N>(gamma_), eps_); }

private:
    Vec<N> gamma_;
    Scalar eps_;  // <arch>.attention.layer_norm_rms_epsilon
};

// Gemma applies the same RMS operation without a learned scale to V heads,
// router inputs, and a few branch inputs. Keeping this a distinct type prevents
// a loader from accidentally supplying (or silently dropping) a weight tensor.
template <size_t N>
class RMSNormNoScale {
public:
    explicit RMSNormNoScale(Scalar eps = 1e-6f) : eps_(eps) {}

    Vec<N> operator()(VecView<N> x) const { return rms_norm(x, eps_); }
    Matrix<N> operator()(MatrixView<N> x) const { return rms_norm(x, eps_); }

private:
    Scalar eps_;
};

// ---- channel mixers ---------------------------------------------------------
//
// SwiGLU: silu(gate) ⊙ up, then down. The Qwen-family channel mixer.
template <size_t D, size_t FF>
class GatedMLP {
public:
    GatedMLP(Linear<D, FF> gate, Linear<D, FF> up, Linear<FF, D> down) : gate_(std::move(gate)), up_(std::move(up)), down_(std::move(down)) {}

    Vec<D> operator()(VecView<D> x) const {
        Vec<FF> g = gate_(x);
        silu(g);
        Vec<FF> u = up_(x);
        g *= u;
        return down_(g);
    }
    Matrix<D> operator()(MatrixView<D> x) const {
        Matrix<FF> gate = gate_(x);
        silu_in_place(gate.mutable_view());
        Matrix<FF> up = up_(x);
        return down_(hadamard(gate.view(), up.view()));
    }

private:
    Linear<D, FF> gate_, up_;
    Linear<FF, D> down_;
};

// Gemma 4 uses gelu_pytorch_tanh(gate) * up rather than SwiGLU.
template <size_t D, size_t FF>
class GeluGatedMLP {
public:
    GeluGatedMLP(Linear<D, FF> gate, Linear<D, FF> up, Linear<FF, D> down) : gate_(std::move(gate)), up_(std::move(up)), down_(std::move(down)) {}

    Vec<D> operator()(VecView<D> x) const {
        Vec<FF> g = gate_(x);
        gelu(g);
        Vec<FF> u = up_(x);
        g *= u;
        return down_(g);
    }
    Matrix<D> operator()(MatrixView<D> x) const {
        Matrix<FF> gate = gate_(x);
        gelu_in_place(gate.mutable_view());
        Matrix<FF> up = up_(x);
        return down_(hadamard(gate.view(), up.view()));
    }

private:
    Linear<D, FF> gate_, up_;
    Linear<FF, D> down_;
};

// Sparse channel mixer: route each token to TOPK of NE experts, plus SHARED
// always-on experts. The expert body is a parameter because a family's sparse
// FFN is its dense FFN repeated — Qwen routes SwiGLU experts, Gemma routes
// GELU-gated ones — and only the routing is new. [BANDWIDTH]
template <size_t D, size_t FF, size_t NE, size_t TOPK, size_t SHARED = 0, class Expert = GatedMLP<D, FF>>
class MoE {
    static_assert(TOPK <= NE);

public:
    MoE(Linear<D, NE> router, std::array<Expert, NE> experts, std::array<Expert, SHARED> shared) : router_(std::move(router)), experts_(std::move(experts)), shared_(std::move(shared)) {}

    Vec<D> operator()(VecView<D> x) const {
        Vec<NE> s = router_(x);
        softmax(std::span<Scalar>(s.begin(), NE));
        std::array<size_t, NE> idx;
        std::iota(idx.begin(), idx.end(), 0u);
        std::partial_sort(idx.begin(), idx.begin() + TOPK, idx.end(), [&](size_t a, size_t b) { return s[a] > s[b]; });
        Scalar norm = 0;
        for (size_t k = 0; k < TOPK; ++k) norm += s[idx[k]];
        Vec<D> y;
        for (size_t k = 0; k < TOPK; ++k) {  // routed experts
            Vec<D> ye = experts_[idx[k]](x);
            y.scaled_add(VecView<D>(ye), s[idx[k]] / norm);
        }
        for (const auto& sh : shared_) y += sh(x);  // always-on path
        return y;
    }

private:
    Linear<D, NE> router_;
    std::array<Expert, NE> experts_;
    std::array<Expert, SHARED> shared_;
};
