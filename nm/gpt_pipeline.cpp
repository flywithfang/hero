// gpt_pipeline_v6.cpp   (C++20)
//
// v6: real checkpoints only. The toy config is gone; GPT-2 small (a real
// 124M model, and this project's first target) is the runtime smoke test.
// Larger configs are verified at COMPILE TIME: their Model<C> instantiates,
// and constexpr param_count<C>() must land on the advertised size or the
// file does not build.
//
// Forced generalization (courtesy of Gemma 4's config.json):
//   Hq*Dh == D is a GPT-2/Llama CONVENTION, not a law. Gemma-4-26B-A4B has
//   D=2816 but 16 heads x 256 = 4096. The law is only:
//     q_proj: D -> Hq*Dh,   o_proj: Hq*Dh -> D,   q-dim == k-dim per head.
//   Attention now carries its own width Dq = Hq*Dh, independent of D.
//
// Honesty ledger (config shapes are real; anatomy is still GPT-2):
//   - RoPE, RMSNorm, SwiGLU, sliding windows, per-layer-type attention
//     (Gemma), k==v sharing (Gemma global layers) are NOT implemented.
//     Configs marked below. param_count<> models the REAL anatomy via
//     MLP_MATS/TIED so the totals check out even where components lag.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <variant>
#include <vector>

using Scalar = float;

// ================= stratum 0: typed storage ================================

template <size_t N>
struct Vec {
    static constexpr size_t extent = N;
    std::unique_ptr<Scalar[]> p = std::make_unique<Scalar[]>(N); // zeroed
    Scalar&       operator[](size_t i)       { return p[i]; }
    const Scalar& operator[](size_t i) const { return p[i]; }
    Scalar* begin() { return p.get(); }  Scalar* end() { return p.get() + N; }
};

template <size_t N>
struct VecView {
    const Scalar* p;
    VecView(const Vec<N>& v) : p(v.p.get()) {}
    explicit VecView(const Scalar* raw) : p(raw) {}
    const Scalar& operator[](size_t i) const { return p[i]; }
    const Scalar* begin() const { return p; } const Scalar* end() const { return p + N; }
};

template <size_t N>
struct MutVecView {
    Scalar* p;
    Scalar& operator[](size_t i) const { return p[i]; }
};

template <size_t R, size_t C>
struct Mat {                               // the ONLY type that knows layout
    static constexpr size_t rows = R, cols = C;
    std::unique_ptr<Scalar[]> p = std::make_unique<Scalar[]>(R * C);
    Scalar&       operator()(size_t r, size_t c)       { return p[r * C + c]; }
    const Scalar& operator()(size_t r, size_t c) const { return p[r * C + c]; }
    VecView<C> row(size_t i) const { return VecView<C>(&p[i * C]); }
    Scalar* raw() { return p.get(); }
};

template <size_t C>
struct RowStore {                          // runtime rows x compile-time cols
    std::vector<Scalar> data;
    size_t rows = 0;
    VecView<C> row(size_t i) const { return VecView<C>(&data[i * C]); }
    void append(VecView<C> v) { data.insert(data.end(), v.begin(), v.end()); ++rows; }
};

// Two head-index types that must never be confused: with GQA, a query-head
// index and a kv-head index are DIFFERENT coordinates — and on MHA configs
// (Hq == Hkv) mixing them is silently correct, detonating only when a GQA
// config arrives. The types make the latent bug unrepresentable; the group()
// mapping in Attention is the only constructor of a KvHead from a QHead.
struct QHead  { size_t i; };
struct KvHead { size_t i; };

// One layer's visible history, with its head structure OWNED here: a cache
// row is logically [kv_head][component], and only PastView knows the layout.
// Call sites speak the domain: past.key(j, g), past.value(j, g).
template <size_t Hkv, size_t Dqk, size_t Dv>
struct PastView {
    const RowStore<Hkv * Dqk>& K;
    const RowStore<Hkv * Dv>&  V;
    VecView<Dqk> key  (size_t j, KvHead g) const { return slice<Dqk>(K.row(j), g.i * Dqk); }
    VecView<Dv>  value(size_t j, KvHead g) const { return slice<Dv> (V.row(j), g.i * Dv);  }
};

// ================= stratum 1: algebra ======================================

template <size_t M, size_t N>
Vec<N> operator*(VecView<M> x, const Mat<M, N>& W) {
    Vec<N> y;
    for (size_t m = 0; m < M; ++m) {
        const Scalar xm = x[m];
        for (size_t n = 0; n < N; ++n) y[n] += xm * W(m, n);
    }
    return y;
}
template <size_t N> Vec<N> operator+(VecView<N> a, VecView<N> b) {
    Vec<N> y; for (size_t i = 0; i < N; ++i) y[i] = a[i] + b[i]; return y;
}
template <size_t N> void operator+=(Vec<N>& y, VecView<N> x) {
    for (size_t i = 0; i < N; ++i) y[i] += x[i];
}
template <size_t N> void operator+=(Vec<N>& y, const Vec<N>& x) {
    for (size_t i = 0; i < N; ++i) y[i] += x[i];
}
template <size_t N> Vec<N> copy(VecView<N> v) {
    Vec<N> y; for (size_t i = 0; i < N; ++i) y[i] = v[i]; return y;
}
template <size_t N> Scalar dot(VecView<N> a, VecView<N> b) {
    Scalar s = 0; for (size_t i = 0; i < N; ++i) s += a[i] * b[i]; return s;
}
template <size_t N> void axpy(Scalar a, VecView<N> x, Vec<N>& y) {
    for (size_t i = 0; i < N; ++i) y[i] += a * x[i];
}
template <size_t W_, size_t N>
VecView<W_> slice(VecView<N> v, size_t off) {
    static_assert(W_ <= N);
    return VecView<W_>(&v[off]);
}
template <size_t W_, size_t N>
MutVecView<W_> slice_mut(Vec<N>& v, size_t off) {
    static_assert(W_ <= N);
    return MutVecView<W_>{&v[off]};
}
void softmax(std::span<Scalar> s) {
    Scalar mx = *std::max_element(s.begin(), s.end()), sum = 0;
    for (auto& v : s) { v = std::exp(v - mx); sum += v; }
    for (auto& v : s) v /= sum;
}
template <size_t N> void gelu(Vec<N>& v) {
    for (size_t i = 0; i < N; ++i) {
        Scalar x = v[i];
        v[i] = 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}
template <size_t N> void silu(Vec<N>& v) {            // z·sigmoid(z): soft ReLU,
    for (size_t i = 0; i < N; ++i)                     // no dead zone
        v[i] = v[i] / (1.f + std::exp(-v[i]));
}
template <size_t N> void operator*=(Vec<N>& y, const Vec<N>& x) {
    for (size_t i = 0; i < N; ++i) y[i] *= x[i];       // Hadamard ⊙
}
template <class F> void par_for(size_t n, F&& f) {
    for (size_t i = 0; i < n; ++i) f(i);
}
// Parallel MAP: the stronger contract. f(h) is a pure function of its index
// and its (by-value, read-only) captures; it RETURNS its partition. Placement
// is owned here, not by the lambda — no shared mutable capture, so tasks
// cannot race by construction. This is the seam a thread/GPU backend replaces.
template <size_t H, size_t Dim, class F>
Vec<H * Dim> par_map(F&& f) {
    Vec<H * Dim> out;
    for (size_t h = 0; h < H; ++h) {                   // independent tasks
        Vec<Dim> r = f(h);
        std::copy(r.begin(), r.end(), &out[h * Dim]);
    }
    return out;
}

// ================= stratum 2: components ===================================

enum class TokenId : int32_t {};
struct Position { size_t i; };

template <size_t In, size_t Out>
struct Linear {
    Mat<In, Out> W;
    Vec<Out>     b;
    Vec<Out> operator()(VecView<In> x) const { Vec<Out> y = x * W; y += b; return y; }
};

template <size_t N>
struct LayerNorm {
    Vec<N> gamma, beta;
    Vec<N> operator()(VecView<N> x) const {
        Scalar mu = 0;  for (size_t i = 0; i < N; ++i) mu += x[i];      mu /= N;
        Scalar var = 0; for (size_t i = 0; i < N; ++i) var += (x[i]-mu)*(x[i]-mu);
        const Scalar inv = 1.f / std::sqrt(var / N + 1e-5f);
        Vec<N> y;
        for (size_t i = 0; i < N; ++i) y[i] = gamma[i] * (x[i] - mu) * inv + beta[i];
        return y;
    }
};

// RoPE: position as phase. Per head, the Dqk dims form Dqk/2 planes; plane i
// rotates by angle pos * theta_i, theta_i = Base^(-i / (Dqk/2)) — one constant
// generates the whole frequency ladder (fast pairs: word order; slow pairs:
// document scale; the slowest period must comfortably exceed CTX, which is
// why rope_theta grew 10k -> 500k as contexts grew).
// Applied to q and k ONLY (never v), after projection, before the dot.
// k is rotated BEFORE caching, so decode pays nothing per past token.
// The dot of two rotated vectors depends only on the position DIFFERENCE:
// (R(p)q)·(R(m)k) = qᵀR(m−p)k — verified as executable code in main().
//
// PAIRING CONVENTION: rotate_half — plane i is (v[i], v[i + Dqk/2]), matching
// HF-Llama / ggml NEOX mode. The original paper interleaves (v[2i], v[2i+1]).
// Mixing conventions against a trained checkpoint = garbage attention, no
// crash. This is the RoPE analogue of the Conv1D transpose trap.
template <size_t Dqk, size_t MaxPos, size_t Base>
struct Rope {
    static constexpr size_t P = Dqk / 2;                // planes per head
    std::vector<Scalar> cs, sn;                         // [MaxPos x P] tables
    Rope() : cs(MaxPos * P), sn(MaxPos * P) {
        for (size_t i = 0; i < P; ++i) {
            const double theta = std::pow(double(Base), -double(i) / double(P));
            for (size_t p = 0; p < MaxPos; ++p) {
                cs[p * P + i] = Scalar(std::cos(double(p) * theta));
                sn[p * P + i] = Scalar(std::sin(double(p) * theta));
            }
        }
    }
    void apply(MutVecView<Dqk> v, size_t pos) const {
        for (size_t i = 0; i < P; ++i) {
            const Scalar c = cs[pos * P + i], s = sn[pos * P + i];
            const Scalar x = v[i], y = v[i + P];
            v[i]     = x * c - y * s;
            v[i + P] = x * s + y * c;
        }
    }
};
struct NoRope {};                                       // table-position models

// Rotate every head's slice of a multi-head row (q [Hq*Dqk] or k [Hkv*Dqk]):
// same position, same ladder, applied per head slice.
template <size_t Dqk, size_t W, class R>
void rotate_heads(Vec<W>& v, const R& rope, size_t pos) {
    static_assert(W % Dqk == 0);
    for (size_t h = 0; h < W / Dqk; ++h)
        rope.apply(slice_mut<Dqk>(v, h * Dqk), pos);
}

// LayerNorm minus mean and beta: re-scaling was the half that cured the
// disease (softmax saturation + gradient scale are magnitude problems), so
// modern models keep only it. Same branch-copy signature: cannot touch trunk.
template <size_t N>
struct RMSNorm {
    Vec<N> gamma;
    Vec<N> operator()(VecView<N> x) const {
        Scalar ms = 0;
        for (size_t i = 0; i < N; ++i) ms += x[i] * x[i];
        const Scalar inv = 1.f / std::sqrt(ms / N + 1e-5f);
        Vec<N> y;
        for (size_t i = 0; i < N; ++i) y[i] = gamma[i] * x[i] * inv;
        return y;
    }
};

template <size_t D, size_t FF>
struct DenseMLP {
    Linear<D, FF> up;
    Linear<FF, D> down;
    Vec<D> operator()(VecView<D> x) const {
        Vec<FF> h = up(x); gelu(h); return down(h);
    }
};

// SwiGLU: each of the FF channels gets a separately-computed volume knob.
// `up` proposes content; silu(gate) — same x through a different learned
// lens — decides smoothly how much passes. A multiplicative interaction the
// additive two-matrix MLP cannot express. Llama/Qwen/Gemma anatomy.
template <size_t D, size_t FF>
struct GatedMLP {
    Linear<D, FF> gate, up;
    Linear<FF, D> down;
    Vec<D> operator()(VecView<D> x) const {
        Vec<FF> g = gate(x);  silu(g);
        Vec<FF> u = up(x);
        g *= u;                                        // silu(gate) ⊙ up
        return down(g);
    }
};

// MoE with optional always-on shared expert(s) — Gemma 4: 128 routed top-8
// + 1 shared; DeepSeek-V3: 256 routed top-8 + 1 shared. Routed experts get
// softmax-renormalized gates; shared experts are simply added.
template <size_t D, size_t FF, size_t NE, size_t TOPK, size_t SHARED = 0>
struct MoE {
    static_assert(TOPK <= NE);
    Linear<D, NE>                       router;
    std::array<DenseMLP<D, FF>, NE>     expert;
    std::array<DenseMLP<D, FF>, SHARED> shared;
    Vec<D> operator()(VecView<D> x) const {
        Vec<NE> s = router(x);
        softmax(std::span<Scalar>(s.begin(), NE));
        std::array<size_t, NE> idx;
        std::iota(idx.begin(), idx.end(), 0u);
        std::partial_sort(idx.begin(), idx.begin() + TOPK, idx.end(),
                          [&](size_t a, size_t b) { return s[a] > s[b]; });
        Scalar norm = 0;
        for (size_t k = 0; k < TOPK; ++k) norm += s[idx[k]];
        Vec<D> y;
        par_for(TOPK, [&](size_t k) {                  // PARALLEL: experts
            Vec<D> ye = expert[idx[k]](x);
            for (size_t i = 0; i < D; ++i) y[i] += s[idx[k]] / norm * ye[i];
        });
        for (const auto& sh : shared) y += sh(x);      // always-on path
        return y;
    }
};

// Attention in its fully general form. The four LAWS (all that is rigid):
//   1. q and k share a per-head dim            -> one Dqk for both
//   2. v's per-head dim is free                -> its own Dv
//   3. Wo input = Hq * Dv                      -> heads output BLENDS OF V
//   4. Wo output = D                           -> the residual add
// Everything else — Hq, Hkv, Dqk, Dv, their products — is designer freedom.
template <size_t D, size_t Hq, size_t Hkv, size_t Dqk, size_t Dv>
struct Attention {
    static_assert(Hq % Hkv == 0, "GQA groups must divide evenly");
    static constexpr size_t QW = Hq  * Dqk;   // query width
    static constexpr size_t KW = Hkv * Dqk;   // cached key row width
    static constexpr size_t VW = Hkv * Dv;    // cached value row width
    static constexpr size_t OW = Hq  * Dv;    // concat width (law 3)
    Linear<D,  QW> q_proj;
    Linear<D,  KW> k_proj;
    Linear<D,  VW> v_proj;
    Linear<OW, D>  o_proj;                    // law 4: back to the stream

    // Split interface: the caller (who knows positions) projects the query,
    // optionally rotates it, then hands it back for scoring. Attention itself
    // stays position-agnostic — RoPE never appears below this line.
    Vec<QW> query(VecView<D> xn) const { return q_proj(xn); }

    // The GQA mapping is the ONLY way to make a KvHead from a QHead.
    static constexpr KvHead group(QHead h) { return {h.i / (Hq / Hkv)}; }

    Vec<D> attend(VecView<QW> q, PastView<Hkv, Dqk, Dv> past, size_t Tvis) const {
        // PARALLEL MAP over heads: each task gets its partition (one query
        // head), captures read-only views BY VALUE, and returns its output.
        // No shared mutable state — un-raceable by construction.
        Vec<OW> o = par_map<Hq, Dv>([q, past, Tvis](size_t h) {
            return attend_one(slice<Dqk>(q, h * Dqk), past,
                              group(QHead{h}), Tvis);
        });
        return o_proj(o);
    }

    // One head, pure: (its question, the past, its group, horizon) -> blend.
    static Vec<Dv> attend_one(VecView<Dqk> qh, PastView<Hkv, Dqk, Dv> past,
                              KvHead g, size_t Tvis) {
        std::vector<Scalar> s(Tvis);
        for (size_t j = 0; j < Tvis; ++j)              // [GROWS-T][BANDWIDTH]
            s[j] = dot(qh, past.key(j, g)) / std::sqrt(Scalar(Dqk));
        softmax(s);
        Vec<Dv> out;
        for (size_t j = 0; j < Tvis; ++j)
            axpy(s[j], past.value(j, g), out);         // out += s_j · v_j
        return out;
    }
};

// ================= stratum 3: assemblies ===================================
// Real checkpoints. Fields beyond shapes: TIED (unembedding = wte),
// MLP_MATS (2 = up/down GELU, 3 = SwiGLU gate/up/down), MOE flag.
// param_count<C>() below must land on the advertised size at COMPILE TIME.

struct GPT2Small {          // runtime smoke test — the project's first target
    static constexpr size_t V=50257, D=768, L=12, Hq=12, Hkv=12, Dqk=64, Dv=64,
                            FF=3072, CTX=1024, NE=1, TOPK=1, SHARED=0;
    static constexpr size_t MLP_MATS=2;  static constexpr bool TIED=true,  MOE=false;
    static constexpr bool   ROPE=false;  static constexpr size_t ROPE_BASE=1;
    static constexpr bool   RMS=false;
};
struct Llama32_1B {         // the M3 target. anatomy still pending: RMSNorm, SwiGLU
    static constexpr size_t V=128256, D=2048, L=16, Hq=32, Hkv=8, Dqk=64, Dv=64,
                            FF=8192, CTX=131072, NE=1, TOPK=1, SHARED=0;
    static constexpr size_t MLP_MATS=3;  static constexpr bool TIED=true,  MOE=false;
    static constexpr bool   ROPE=true;   static constexpr size_t ROPE_BASE=500000;
    static constexpr bool   RMS=true;
};                          // (3.2 additionally rescales frequencies for 128K —
                            //  the ~10-line surgery lands with the weight loader)
struct Llama3_8B {          // anatomy pending: RMSNorm, SwiGLU
    static constexpr size_t V=128256, D=4096, L=32, Hq=32, Hkv=8, Dqk=128, Dv=128,
                            FF=14336, CTX=131072, NE=1, TOPK=1, SHARED=0;
    static constexpr size_t MLP_MATS=3;  static constexpr bool TIED=false, MOE=false;
    static constexpr bool   ROPE=true;   static constexpr size_t ROPE_BASE=500000;
    static constexpr bool   RMS=true;
};
struct Llama32_3B {         // anatomy pending: RMSNorm, SwiGLU
    static constexpr size_t V=128256, D=3072, L=28, Hq=24, Hkv=8, Dqk=128, Dv=128,
                            FF=8192, CTX=131072, NE=1, TOPK=1, SHARED=0;
    static constexpr size_t MLP_MATS=3;  static constexpr bool TIED=true,  MOE=false;
    static constexpr bool   ROPE=true;   static constexpr size_t ROPE_BASE=500000;
    static constexpr bool   RMS=true;
};
struct Gemma4_26B_A4B {     // shapes real; UNREPRESENTED: 5:1 sliding/global
                            // pattern, global layers use 2 kv-heads x 512 with
                            // k==v sharing, p-RoPE, logit softcap. Uniform
                            // sliding-layer attention shapes assumed below.
    static constexpr size_t V=262144, D=2816, L=30, Hq=16, Hkv=8, Dqk=256, Dv=256,
                            FF=704,   CTX=262144, NE=128, TOPK=8, SHARED=1;
    static constexpr size_t MLP_MATS=3;  static constexpr bool TIED=true,  MOE=true;
    static constexpr bool   ROPE=true;   static constexpr size_t ROPE_BASE=1000000;
    static constexpr bool   RMS=true;
};  // note Hq*Dqk = 4096 != D = 2816 — the convention-breaker
// (All four set Dqk == Dv, the common collapse. MLA-style models split them:
//  DeepSeek-V3 keys are effectively 192-wide vs 128-wide values.)

// ---- compile-time parameter accounting (norms/biases ~0.03%, ignored) ----
template <class C>
constexpr size_t param_count() {
    constexpr size_t QW = C::Hq * C::Dqk, KW = C::Hkv * C::Dqk;
    constexpr size_t VW = C::Hkv * C::Dv, OW = C::Hq  * C::Dv;
    constexpr size_t attn   = C::D * QW + C::D * KW + C::D * VW + OW * C::D;
    constexpr size_t ff_one = C::MLP_MATS * C::D * C::FF;
    constexpr size_t ff     = C::MOE ? (C::NE + C::SHARED) * ff_one + C::D * C::NE
                                     : ff_one;
    constexpr size_t embed  = C::V * C::D * (C::TIED ? 1 : 2);
    return embed + C::L * (attn + ff);
}
// The configs must reproduce the advertised sizes, or this file won't build:
static_assert(param_count<GPT2Small>()     > 120'000'000    &&
              param_count<GPT2Small>()     < 130'000'000);      // "124M"
static_assert(param_count<Llama3_8B>()     > 7'800'000'000  &&
              param_count<Llama3_8B>()     < 8'300'000'000);    // "8B"
static_assert(param_count<Llama32_3B>()    > 3'000'000'000  &&
              param_count<Llama32_3B>()    < 3'400'000'000);    // "3.2B"
static_assert(param_count<Gemma4_26B_A4B>()> 24'000'000'000 &&
              param_count<Gemma4_26B_A4B>()< 26'000'000'000);   // "25.2B"
static_assert(param_count<Llama32_1B>()    > 1'150'000'000  &&
              param_count<Llama32_1B>()    < 1'300'000'000);    // "1B"

template <class C>
struct Block {
    using Attn = Attention<C::D, C::Hq, C::Hkv, C::Dqk, C::Dv>;
    using Norm = std::conditional_t<C::RMS, RMSNorm<C::D>, LayerNorm<C::D>>;
    using FF   = std::variant<DenseMLP<C::D, C::FF>,
                              GatedMLP<C::D, C::FF>,
                              MoE<C::D, C::FF, C::NE, C::TOPK, C::SHARED>>;
    Norm ln1, ln2;
    Attn attn;
    FF   ff;
};

struct NoPosTable {};       // RoPE models: position is computed, not stored
template <class C>
struct Embedding {
    Mat<C::V, C::D> wte;
    [[no_unique_address]]
    std::conditional_t<C::ROPE, NoPosTable, Mat<C::CTX, C::D>> wpe;
    VecView<C::D> tok(TokenId t) const { return wte.row(size_t(t)); }
    VecView<C::D> pos(Position p) const
        requires (!C::ROPE)          { return wpe.row(p.i); }
};

template <class C>
struct Model {
    Embedding<C>          embed;
    std::vector<Block<C>> block{C::L};
    typename Block<C>::Norm ln_f;
    [[no_unique_address]]
    std::conditional_t<C::ROPE, Rope<C::Dqk, C::CTX, C::ROPE_BASE>, NoRope> rope;
};

// The cache owns its invariants: one (k, v) pair per token per layer, K and V
// always in lockstep, T advanced once per committed token. Callers deposit
// and read through the API; the parallel arrays are no longer their problem.
template <class C>
class KVCache {
    static constexpr size_t KW = Block<C>::Attn::KW;
    static constexpr size_t VW = Block<C>::Attn::VW;
    std::array<RowStore<KW>, C::L> K_;
    std::array<RowStore<VW>, C::L> V_;
    size_t T_ = 0;
public:
    // the formula "cache row = Hkv*(Dqk + Dv)", as a compile-time constant:
    static constexpr size_t floats_per_token = C::L * (KW + VW);

    size_t tokens() const { return T_; }
    void   advance(size_t n) { T_ += n; }

    void deposit(size_t l, const Vec<KW>& k, const Vec<VW>& v) {
        K_[l].append(k);
        V_[l].append(v);                     // lockstep enforced HERE, once
    }
    PastView<C::Hkv, C::Dqk, C::Dv> past(size_t l) const { return {K_[l], V_[l]}; }
};

template <class C>
Vec<C::V> forward(const Model<C>& m, KVCache<C>& cache,
                  std::span<const TokenId> ids) {
    const size_t n = ids.size(), T0 = cache.tokens();

    std::vector<Vec<C::D>> x(n);
    par_for(n, [&](size_t t) {                         // PARALLEL over rows
        if constexpr (C::ROPE)
            x[t] = copy(m.embed.tok(ids[t]));          // position enters LATER,
        else                                           // inside attention
            x[t] = m.embed.tok(ids[t]) + m.embed.pos(Position{T0 + t});
    });

    for (size_t l = 0; l < C::L; ++l) {                // SEQUENTIAL: depth
        const Block<C>& B = m.block[l];
        std::vector<Vec<C::D>> xn(n);
        for (size_t t = 0; t < n; ++t) {               // ordered cache commit
            xn[t] = B.ln1(x[t]);
            auto k = B.attn.k_proj(xn[t]);
            if constexpr (C::ROPE)
                rotate_heads<C::Dqk>(k, m.rope, T0 + t);   // cached PRE-rotated
            cache.deposit(l, k, B.attn.v_proj(xn[t]));     // v: never rotated
        }
        par_for(n, [&](size_t t) {                     // PARALLEL: rows
            auto q = B.attn.query(xn[t]);
            if constexpr (C::ROPE)
                rotate_heads<C::Dqk>(q, m.rope, T0 + t);
            x[t] += B.attn.attend(q, cache.past(l), T0 + t + 1);
        });
        par_for(n, [&](size_t t) {                     // PARALLEL: rows
            Vec<C::D> branch = B.ln2(x[t]);
            x[t] += std::visit([&](auto& f) { return f(VecView<C::D>(branch)); }, B.ff);
        });
    }
    cache.advance(n);

    Vec<C::D> last = m.ln_f(x[n - 1]);
    Vec<C::V> logits;
    par_for(C::V, [&](size_t v) {
        logits[v] = dot(VecView<C::D>(last), m.embed.wte.row(v));
    });
    return logits;
}

// ================= stratum 4: runtime ======================================

template <size_t V>
TokenId sample_greedy(const Vec<V>& lg) {
    size_t best = 0;
    for (size_t v = 1; v < V; ++v) if (lg[v] > lg[best]) best = v;
    return TokenId{int32_t(best)};
}

template <class C>
class Runtime {
    const Model<C>& m;
    KVCache<C>      cache;
public:
    explicit Runtime(const Model<C>& model) : m(model) {}
    size_t context_used() const { return cache.tokens(); }
    size_t context_left() const { return C::CTX - cache.tokens(); }
    void   reset()               { cache = KVCache<C>{}; }

    Vec<C::V> prefill(std::span<const TokenId> prompt) {
        return forward(m, cache, prompt);
    }
    Vec<C::V> step(TokenId id) {
        return forward(m, cache, std::span(&id, 1));
    }
    std::vector<TokenId> generate(std::span<const TokenId> prompt,
                                  size_t max_new, TokenId eos) {
        std::vector<TokenId> out;
        Vec<C::V> logits = prefill(prompt);
        for (size_t i = 0; i < max_new; ++i) {
            TokenId id = sample_greedy(logits);
            if (id == eos) break;
            out.push_back(id);
            if (context_left() == 0) break;
            logits = step(id);
        }
        return out;
    }
};

// Larger configs: full-pipeline compile-time instantiation (no allocation).
template class Runtime<Llama32_1B>;
template class Runtime<Llama3_8B>;
template class Runtime<Llama32_3B>;
template class Runtime<Gemma4_26B_A4B>;

// ================= smoke test: real GPT-2 small ============================

std::mt19937 g_rng(42);
void fill_rnd(Scalar* p, size_t n, Scalar sd = 0.05f) {
    std::normal_distribution<Scalar> N(0.f, sd);
    for (size_t i = 0; i < n; ++i) p[i] = N(g_rng);
}
template <size_t I, size_t O> void init(Linear<I, O>& L) { fill_rnd(L.W.raw(), I * O); }
template <size_t N> void init(LayerNorm<N>& ln) {
    for (size_t i = 0; i < N; ++i) ln.gamma[i] = 1.f;
}
template <size_t N> void init(RMSNorm<N>& rn) {
    for (size_t i = 0; i < N; ++i) rn.gamma[i] = 1.f;
}

template <class C>
void report() {
    std::printf("  %-6s %-5s D=%-5zu L=%-3zu Hq=%-3zu Hkv=%-2zu Dqk=%-3zu Dv=%-3zu"
                "  ~%.2fB params  cache %.0f KB/token (fp16)\n",
                C::MOE ? "MoE" : "dense", C::ROPE ? "rope" : "table",
                C::D, C::L, C::Hq, C::Hkv, C::Dqk, C::Dv,
                double(param_count<C>()) / 1e9,
                double(KVCache<C>::floats_per_token) * 2.0 / 1024.0);
}

int main() {
    std::printf("config check (constexpr param counts, verified by static_assert):\n");
    std::printf("GPT-2 small:      "); report<GPT2Small>();
    std::printf("Llama-3.2-1B:     "); report<Llama32_1B>();
    std::printf("Llama-3-8B:       "); report<Llama3_8B>();
    std::printf("Llama-3.2-3B:     "); report<Llama32_3B>();
    std::printf("Gemma-4-26B-A4B:  "); report<Gemma4_26B_A4B>();

    // Executable proof of the RoPE identity: same offset => same score,
    // regardless of absolute position. (R(p)q)·(R(m)k) == q·R(m-p)k.
    {
        Rope<8, 64, 500000> r;                      // 4 planes, 64 positions
        Vec<8> q0, k0;
        fill_rnd(q0.begin(), 8, 1.f); fill_rnd(k0.begin(), 8, 1.f);
        auto score = [&](size_t p, size_t m2) {
            Vec<8> q = copy(VecView<8>(q0)), k = copy(VecView<8>(k0));
            r.apply(slice_mut<8>(q, 0), p);
            r.apply(slice_mut<8>(k, 0), m2);
            return dot(VecView<8>(q), VecView<8>(k));
        };
        Scalar a = score(3, 1), b = score(10, 8), c = score(60, 58);
        std::printf("\nRoPE identity (offset 2 at three absolute positions): "
                    "%.6f  %.6f  %.6f  %s\n", a, b, c,
                    (std::fabs(a-b) < 1e-4 && std::fabs(b-c) < 1e-4) ? "MATCH" : "FAIL");
    }

    // Executable properties of the new components:
    // RMSNorm is scale-invariant (the whole point: magnitude control);
    // silu(2) = 2*sigmoid(2) = 1.7616.
    {
        RMSNorm<4> rn; init(rn);
        Vec<4> a, b;
        for (size_t i = 0; i < 4; ++i) { a[i] = Scalar(2*(i+1)); b[i] = 10*a[i]; }
        Vec<4> na = rn(a), nb = rn(b);
        Scalar dmax = 0;
        for (size_t i = 0; i < 4; ++i) dmax = std::max(dmax, std::fabs(na[i]-nb[i]));
        Vec<1> z; z[0] = 2.f; silu(z);
        std::printf("RMSNorm scale-invariance: max|n(x)-n(10x)| = %.2e  %s   "
                    "silu(2) = %.4f\n", dmax, dmax < 1e-5 ? "MATCH" : "FAIL", z[0]);
    }

    // Runtime test: real GPT-2 small shapes, random weights (~0.5 GB fp32).
    using C = GPT2Small;
    auto model = std::make_unique<Model<C>>();
    Model<C>& m = *model;
    fill_rnd(m.embed.wte.raw(), C::V * C::D);
    fill_rnd(m.embed.wpe.raw(), C::CTX * C::D);
    init(m.ln_f);
    for (auto& B : m.block) {
        init(B.ln1); init(B.ln2);
        init(B.attn.q_proj); init(B.attn.k_proj);
        init(B.attn.v_proj); init(B.attn.o_proj);
        DenseMLP<C::D, C::FF> d; init(d.up); init(d.down);
        B.ff = std::move(d);
    }
    Runtime<C> session(m);
    std::vector<TokenId> prompt{TokenId{464}, TokenId{3290}, TokenId{318}};
    auto out = session.generate(prompt, 4, TokenId{50256});
    std::printf("\nGPT-2-small forward (random weights): 464 3290 318 ->");
    for (TokenId id : out) std::printf(" %d", int(id));
    std::printf("   (ctx used: %zu / %zu)\n", session.context_used(), C::CTX);
}
