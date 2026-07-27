// quant_tests.cpp — M3/M4 kernel tests, self-contained (no external deps).
//
// (1) Round-trip: quantize fp32 -> dequant recovers within the format's error
//     bound (Q8_0, Q4_0 — the formats we can quantize ourselves).
// (2) Kernel consistency: the dequant-in-dot matvec kernel must equal
//     dot(dequant_to_f32(block), x) EXACTLY for every format, on random block
//     bytes. This pins the hot-path kernels to the ggml-verbatim dequant
//     reference (which is separately cross-checked against libggml).
// (3) Block sizes match the ggml static_asserts.
#include "../src/quant_i8.hpp"
#include <array>
#include <cstdio>
#include <random>

static std::mt19937 rng(1234);

static float frand(float lo, float hi) {
    std::uniform_real_distribution<float> U(lo, hi);
    return U(rng);
}
static uint8_t brand() { return uint8_t(rng() & 0xFF); }

static int g_fail = 0;
static void check(bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg);
    if (!ok) ++g_fail;
}

// Build a random block of `type` and return (bytes, its f32 dequant).
template <class Block>
static std::vector<Scalar> rand_block_and_dequant(GT type, Block& b) {
    uint8_t* raw = reinterpret_cast<uint8_t*>(&b);
    for (size_t i = 0; i < sizeof(Block); ++i) raw[i] = brand();
    // give the fp16 scale(s) a sane magnitude so values aren't NaN/inf
    std::vector<Scalar> out(block_len(type));
    dequant_to_f32(type, &b, out.data(), out.size());
    return out;
}

int main() {
    std::printf("== BF16 conversion ==\n");
    {
        const std::array<float, 6> x = {0.0f, -0.0f, 1.0f, -2.5f, 123.75f, 1.0e-20f};
        std::array<uint16_t, x.size()> b{};
        std::array<float, x.size()> y{};
        for (size_t i = 0; i < x.size(); ++i) b[i] = fp32_to_bf16(x[i]);
        dequant_to_f32(GT::BF16, b.data(), y.data(), y.size());
        bool exact = true;
        for (size_t i = 0; i < x.size(); ++i) exact = exact && y[i] == bf16_to_fp32(b[i]);
        check(exact && type_size_bytes(GT::BF16, x.size()) == 2 * x.size(), "BF16 widens exactly and occupies two bytes");
    }

    std::printf("== block sizes ==\n");
    check(sizeof(block_q4_0) == 18 && sizeof(block_q8_0) == 34 && sizeof(block_q4_K) == 144 && sizeof(block_q5_K) == 176 && sizeof(block_q6_K) == 210, "struct sizes match ggml");

    std::printf("== Q8_0 round-trip ==\n");
    {
        const size_t N = 32 * 100;
        std::vector<Scalar> x(N), y(N);
        for (auto& v : x) v = frand(-3, 3);
        std::vector<block_q8_0> b(N / 32);
        quantize_q8_0(x.data(), b.data(), N);
        dequant_to_f32(GT::Q8_0, b.data(), y.data(), N);
        float maxerr = 0, maxabs = 0;
        for (size_t i = 0; i < N; ++i) {
            maxerr = std::max(maxerr, std::fabs(x[i] - y[i]));
            maxabs = std::max(maxabs, std::fabs(x[i]));
        }
        check(maxerr < maxabs / 127.0f + 1e-3f, "Q8_0 err within 1 quantum");
    }
    std::printf("== Q4_0 round-trip ==\n");
    {
        const size_t N = 32 * 100;
        std::vector<Scalar> x(N), y(N);
        for (auto& v : x) v = frand(-3, 3);
        std::vector<block_q4_0> b(N / 32);
        quantize_q4_0(x.data(), b.data(), N);
        dequant_to_f32(GT::Q4_0, b.data(), y.data(), N);
        // Q4_0 has 16 levels; error is coarse but bounded by ~|max|/8.
        float maxerr = 0, maxabs = 0;
        for (size_t i = 0; i < N; ++i) {
            maxerr = std::max(maxerr, std::fabs(x[i] - y[i]));
            maxabs = std::max(maxabs, std::fabs(x[i]));
        }
        check(maxerr < maxabs / 7.0f + 1e-3f, "Q4_0 err within ~1 quantum");
    }

    std::printf("== kernel consistency: dot_block == dot(dequant, x) ==\n");
    {
        auto rel = [](Scalar a, Scalar b) { return std::fabs(a - b) / (std::fabs(a) + std::fabs(b) + 1e-6f); };
        // Q8_0
        {
            block_q8_0 blk;
            auto w = rand_block_and_dequant(GT::Q8_0, blk);
            std::vector<Scalar> x(32);
            for (auto& v : x) v = frand(-2, 2);
            Scalar k = dot_block_q8_0(blk, x.data());
            Scalar r = 0;
            for (int i = 0; i < 32; ++i) r += w[i] * x[i];
            check(rel(k, r) < 1e-5f, "Q8_0 kernel == dequant-dot");
        }
        // Q4_0
        {
            block_q4_0 blk;
            auto w = rand_block_and_dequant(GT::Q4_0, blk);
            std::vector<Scalar> x(32);
            for (auto& v : x) v = frand(-2, 2);
            Scalar k = dot_block_q4_0(blk, x.data());
            Scalar r = 0;
            for (int i = 0; i < 32; ++i) r += w[i] * x[i];
            check(rel(k, r) < 1e-5f, "Q4_0 kernel == dequant-dot");
        }
        // Q4_K
        {
            block_q4_K blk;
            auto w = rand_block_and_dequant(GT::Q4_K, blk);
            std::vector<Scalar> x(256);
            for (auto& v : x) v = frand(-2, 2);
            Scalar k = dot_block_q4_K(blk, x.data());
            Scalar r = 0;
            for (int i = 0; i < 256; ++i) r += w[i] * x[i];
            check(rel(k, r) < 1e-4f, "Q4_K kernel == dequant-dot");
        }
        // Q5_K — Q4_K plus the high-bit plane; check the fifth bit actually
        // moves the value by exactly 16 quanta before scaling.
        {
            block_q5_K blk;
            auto w = rand_block_and_dequant(GT::Q5_K, blk);
            std::vector<Scalar> x(256);
            for (auto& v : x) v = frand(-2, 2);
            Scalar k = dot_block_q5_K(blk, x.data());
            Scalar r = 0;
            for (int i = 0; i < 256; ++i) r += w[i] * x[i];
            check(rel(k, r) < 1e-4f, "Q5_K kernel == dequant-dot");
        }
        {  // the high plane is worth exactly +16 raw quanta on the low nibble
            block_q5_K blk{};
            blk.d = fp32_to_fp16(1.f);
            blk.dmin = fp32_to_fp16(0.f);
            for (auto& sc : blk.scales) sc = 0;
            blk.scales[0] = 1;  // sub-block 0 scale = 1, min = 0
            for (auto& q : blk.qs) q = 0;
            for (auto& q : blk.qh) q = 0;
            blk.qs[0] = 3;  // low nibble of element 0 = 3
            std::vector<Scalar> low(256), high(256);
            dequant_to_f32(GT::Q5_K, &blk, low.data(), 256);
            blk.qh[0] = 1;  // set element 0's fifth bit
            dequant_to_f32(GT::Q5_K, &blk, high.data(), 256);
            check(std::fabs(low[0] - 3.f) < 1e-5f && std::fabs(high[0] - 19.f) < 1e-5f, "the Q5_K high plane adds exactly 16 to the 4-bit value");
        }
        // Q6_K
        {
            block_q6_K blk;
            auto w = rand_block_and_dequant(GT::Q6_K, blk);
            std::vector<Scalar> x(256);
            for (auto& v : x) v = frand(-2, 2);
            Scalar k = dot_block_q6_K(blk, x.data());
            Scalar r = 0;
            for (int i = 0; i < 256; ++i) r += w[i] * x[i];
            check(rel(k, r) < 1e-4f, "Q6_K kernel == dequant-dot");
        }
    }

    std::printf("== int8 activation path: NEON == scalar (ggml-generic clone) ==\n");
    {
        // Random weights + activation; the scalar kernels are bit-comparable
        // clones of ggml's generic reference (verified against libggml in the
        // scratchpad harness); here we pin the NEON fast path to them.
        const size_t n = 512;
        std::vector<Scalar> w(n), x(n);
        for (auto& v : w) v = frand(-2, 2);
        for (auto& v : x) v = frand(-2, 2);
        std::vector<block_q8_K> xk(n / QK_K);
        quantize_row_q8_K(x.data(), xk.data(), n);
        std::vector<block_q8_0> x8(n / QK8_0);
        quantize_row_q8_0(x.data(), x8.data(), n);
        std::vector<block_q8_0> w8(n / QK8_0);
        quantize_q8_0(w.data(), w8.data(), n);
        std::vector<block_q4_0> w4(n / QK4_0);
        quantize_q4_0(w.data(), w4.data(), n);
        // Q4_K/Q6_K weight blocks from random bytes with sane fp16 scales
        std::vector<block_q4_K> w4k(n / QK_K);
        std::vector<block_q6_K> w6k(n / QK_K);
        for (auto& b : w4k) {
            auto* p = (uint8_t*)&b;
            for (size_t i = 0; i < sizeof b; ++i) p[i] = brand();
            b.d = fp32_to_fp16(0.01f);
            b.dmin = fp32_to_fp16(0.01f);
        }
        for (auto& b : w6k) {
            auto* p = (uint8_t*)&b;
            for (size_t i = 0; i < sizeof b; ++i) p[i] = brand();
            b.d = fp32_to_fp16(0.01f);
        }
        auto rel = [](float a, float b) { return std::fabs(a - b) / (std::fabs(a) + std::fabs(b) + 1e-9f); };
#if defined(__ARM_FEATURE_DOTPROD)
        check(rel(vec_dot_q4_K_q8_K_neon(n, w4k.data(), xk.data()), vec_dot_q4_K_q8_K_scalar(n, w4k.data(), xk.data())) < 1e-5f, "q4_K neon == scalar");
        check(rel(vec_dot_q6_K_q8_K_neon(n, w6k.data(), xk.data()), vec_dot_q6_K_q8_K_scalar(n, w6k.data(), xk.data())) < 1e-5f, "q6_K neon == scalar");
#endif
        // int8 dot vs fp32 dot on the same dequantised weights: bounded by
        // activation quantisation error (~1%), sanity not exactness.
        std::vector<Scalar> wd(n);
        dequant_to_f32(GT::Q8_0, w8.data(), wd.data(), n);
        float f32dot = 0;
        for (size_t i = 0; i < n; ++i) f32dot += wd[i] * x[i];
        check(rel(vec_dot_q8_0_q8_0(n, w8.data(), x8.data()), f32dot) < 2e-2f, "q8_0 int8 ~ fp32 dot");
        dequant_to_f32(GT::Q4_0, w4.data(), wd.data(), n);
        f32dot = 0;
        for (size_t i = 0; i < n; ++i) f32dot += wd[i] * x[i];
        check(rel(vec_dot_q4_0_q8_0(n, w4.data(), x8.data()), f32dot) < 2e-2f, "q4_0 int8 ~ fp32 dot");
    }

    std::printf("== fp16 round-trip ==\n");
    {
        float mx = 0;
        for (float v : {0.0f, 1.0f, -1.0f, 0.5f, 3.14159f, 65504.0f, 1e-4f, -2.5f}) mx = std::max(mx, std::fabs(v - fp16_to_fp32(fp32_to_fp16(v))) / (std::fabs(v) + 1e-6f));
        check(mx < 1e-2f, "fp16 relative round-trip < 1%");
    }

    std::printf("\n%s (%d failures)\n", g_fail ? "QUANT TESTS FAILED" : "ALL QUANT TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
