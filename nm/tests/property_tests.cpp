// property_tests — in-binary invariants that stay green every build (plan §5):
// RoPE offset identity, RMSNorm scale invariance, softmax sums to 1, SiLU value,
// and the compile-time param_count sanity for the shipped configs.
#include "../src/llama.hpp"
#include <cstdio>
#include <random>
#include <type_traits>

static_assert(!std::is_default_constructible_v<LlamaTokenIO<Stories260K>>);
static_assert(!std::is_default_constructible_v<LlamaModel<Stories260K>>);
static_assert(TransformerArchitecture<LlamaArchitecture<Stories260K>>);
static_assert(std::is_same_v<
              decltype(std::declval<const LlamaModel<Stories260K>&>().token_io()),
              const LlamaTokenIO<Stories260K>&>);
static_assert(std::is_same_v<
              decltype(std::declval<const LlamaModel<Stories260K>&>().layer(0)),
              const LlamaBlock<Stories260K>&>);

static int g_fail = 0;
static void check(bool ok, const char* msg) { std::printf("  [%s] %s\n", ok?"PASS":"FAIL", msg); if(!ok) ++g_fail; }
static std::mt19937 rng(42);
static void fill(Scalar* p, size_t n, float sd=1.0f){ std::normal_distribution<Scalar> N(0,sd); for(size_t i=0;i<n;++i)p[i]=N(rng); }

int main() {
    std::printf("== RoPE offset identity (interleaved NORM pairing) ==\n");
    {
        // (R(p)q)·(R(m)k) depends only on m-p. Dqk=8, base 10000.
        Rope<8, 128, 10000> r;
        Vec<8> q0, k0; fill(q0.begin(),8); fill(k0.begin(),8);
        auto score = [&](size_t p, size_t m){
            Vec<8> q=copy(VecView<8>(q0)), k=copy(VecView<8>(k0));
            r.apply(slice_mut<8>(q,0), p); r.apply(slice_mut<8>(k,0), m);
            return dot(VecView<8>(q), VecView<8>(k));
        };
        Scalar a=score(3,1), b=score(10,8), c=score(60,58);
        check(std::fabs(a-b)<1e-4 && std::fabs(b-c)<1e-4, "same offset => same score at 3 abs positions");
    }
    std::printf("== RMSNorm scale invariance ==\n");
    {
        Vec<4> gamma; for(size_t i=0;i<4;++i) gamma[i]=1.f;
        RMSNorm<4> rn(std::move(gamma));
        Vec<4> a,b; for(size_t i=0;i<4;++i){ a[i]=Scalar(2*(i+1)); b[i]=10*a[i]; }
        Vec<4> na=rn(a), nb=rn(b);
        Scalar d=0; for(size_t i=0;i<4;++i) d=std::max(d,std::fabs(na[i]-nb[i]));
        check(d<1e-5f, "n(x) == n(10x)");
    }
    std::printf("== softmax sums to 1 ==\n");
    {
        std::vector<Scalar> s(50); fill(s.data(),50,3.f);
        softmax(std::span<Scalar>(s.data(),50));
        Scalar sum=0; for(Scalar v:s) sum+=v;
        check(std::fabs(sum-1.f)<1e-5f, "sum == 1");
    }
    std::printf("== SiLU value ==\n");
    { Vec<1> z; z[0]=2.f; silu(z); check(std::fabs(z[0]-1.7616f)<1e-3f, "silu(2)=1.7616"); }

    std::printf("== param_count lands on advertised sizes ==\n");
    check(llama_param_count<Llama32_1B>()>1.15e9 && llama_param_count<Llama32_1B>()<1.3e9, "1B ~1.24B");
    check(llama_param_count<Llama32_3B>()>3.0e9 && llama_param_count<Llama32_3B>()<3.4e9, "3B ~3.21B");
    check(llama_param_count<Llama3_8B>()>7.8e9 && llama_param_count<Llama3_8B>()<8.3e9, "8B ~8B");

    std::printf("\n%s (%d failures)\n", g_fail?"PROPERTY TESTS FAILED":"ALL PROPERTY TESTS PASSED", g_fail);
    return g_fail?1:0;
}
