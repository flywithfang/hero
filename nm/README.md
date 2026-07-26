# nm — a modern-transformer inference engine

Pure C++20, no third-party runtime dependencies, pure CPU. It runs **Gemma 4**
text and image-text inference from GGUF today, and its core is organized around
one question: *what is actually shared by modern transformer decoders, and what
is genuinely per-family anatomy?*

Four checkpoints run today, all verified **token-for-token against llama.cpp**
on the same Q4_K_M file:

| model | anatomy | greedy parity |
|---|---|---|
| Gemma 4 E4B | 42 layers, PLE, shared KV on the last 18 | exact |
| Gemma 4 12B Unified | 48 layers, 40 sliding + 8 global **unified K/V**, no PLE | exact |
| Qwen 3.5 4B | **hybrid**: 24 gated-DeltaNet + 8 gated attention | exact |
| Qwen 3.5 9B | same shape, wider, untied embeddings | exact |

Qwen 3.5 is the sharpest test of that question so far: three of every four
layers mix tokens with a gated delta network carrying a fixed-size state
instead of a K/V cache. Both layer kinds bind the *same* canonical transformer
block; only the mixer type differs. Gemma 4's 26B-A4B MoE is next.

Older families have been removed on purpose. The value here is the core math
and the abstractions, not model breadth.

Built with a **fidelity-first** discipline: every milestone is gated against a
reference oracle (llama.cpp / ggml) on the same weights before moving on. See
`INFERENCE_PLAN.md`.

## Build & run

CMake + Ninja. With presets (`release` / `debug` / `asan`):

```sh
cmake --workflow --preset dev    # configure + build + test in one shot
# or step by step:
cmake --preset release
cmake --build --preset release
ctest --preset release           # property, quant, gemma, multimodal, tokenizer

./build/chat gemma-4-E4B-it-Q4_0.gguf --max 256
./build/gguf_dump model.gguf

# Add vision and optionally attach an image to the first message.
./build/chat gemma-4-E4B-it-Q4_0.gguf \
  --mmproj mmproj-gemma-4-E4B-it-Q8_0.gguf \
  --image photo.png --max 256

# One-shot diagnostics. gemma4_text is the logit-parity harness.
./build/gemma4_text gemma-4-E4B-it-Q4_0.gguf "Hello" 16
./build/gemma4_image gemma-4-E4B-it-Q4_0.gguf \
  mmproj-gemma-4-E4B-it-Q8_0.gguf image.ppm "Describe this image." 64
```

Without presets:
`cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`.

`NM_THREADS=N` sets the worker count (defaults to hardware concurrency).
`NM_FP32_ACT=1` selects the fp32-activation reference path instead of the
int8-activation kernels. Fixture-dependent tests auto-skip when the vocab/model
files aren't present; override with `-DNM_VOCAB_GEMMA4=...`,
`-DNM_VOCAB_QWEN35=...`, `-DNM_GEMMA4=...` at configure time.

Inside multimodal chat, `/image PATH` attaches an image to the next user
message and `/image clear` removes it. On macOS, ImageIO supplies PNG, JPEG,
HEIC and other system formats; the portable dependency-free path accepts RGB
PPM P6/P3. `/reset`, `/stats`, `/help`, and `/quit` work for every adapter.

## Architecture — five strata (layout knowledge lives only in stratum 0)

| stratum | file | contents |
|--------|------|----------|
| 0 storage | `src/core.hpp` | `Vec/VecView/Matrix/MatrixView/MatT`, fp16/bf16 |
| 0 quant | `src/quant.hpp`, `src/quant_i8.hpp` | GGUF block formats, dequant, `Weight<In,Out>` matvec/matmul, int8-activation kernels |
| 1 algebra | `src/core.hpp` | matrix arithmetic, norms, activations, head transforms, dense attention, `par_for`/`par_map` |
| 2 components | `src/components.hpp` | `Linear/ClippedLinear/RMSNorm/RMSNormNoScale/GatedMLP/GeluGatedMLP/MoE` (config-free) |
| 2 token mixing | `src/attention.hpp` | `KVCache`, the visibility mask, GQA `attend`, `RotaryEmbedding`, `PerHeadNorm`, gated-attention head split — **model-neutral** |
| 2 token mixing | `src/recurrent.hpp` | `CausalConv1dState`, `DeltaNetState`, the gated delta rule — **model-neutral** |
| 3 transformer | `src/transformer.hpp` | shared residual stream, immutable callable `Transformer`, architecture contract, explicit `PrefixCache` |
| 3 specifications | `src/gemma4.hpp`, `src/qwen35.hpp` | family dimensions, concrete layer types, cache/state policies, input preparation and output rules |
| 3 modalities | `src/multimodal.hpp`, `src/gemma4_vision.hpp` | modality segments, image preprocessing, ViT, soft-token projection |
| 4 runtime | `src/runtime.hpp` | model-neutral sampling |
| chat application | `tools/chat.cpp`, `src/chat_session.hpp`, `src/chat_models.hpp` | conversation history, model detection, tokenization, sampling, family adapters |
| tensor loader | `src/tensor_loader.hpp` | architecture-neutral, shape-safe GGUF tensor construction |
| model loaders | `src/gemma4_loader.hpp`, `src/gemma4_vision_loader.hpp`, `src/qwen35_loader.hpp` | schema validation → immutable assemblies |
| tokenizer | `src/tokenizer.{hpp,cpp}` | byte-level BPE and sentence-style BPE |
| gguf | `src/gguf.{hpp,cpp}` | GGUF v3 mmap reader |

`MatT<In,Out>` is the ggml `[in,out]` weight layout — row = one output, loaded
**zero-copy** over the mmap. `Weight<In,Out>` dispatches `matvec` and batched
`matmul` across F32 / F16 / BF16 / Q8_0 / Q4_0 / Q4_K / Q6_K.

### The token-mixing core

`src/attention.hpp` is where the "core principles" live, and it knows about no
model at all:

```text
visible(q, k)  <=>  k <= q  and  (W == 0  or  q - k < W)     the mask, once
alpha          =  softmax_{k in visible} ( scale · q_h · k )
out_h          =  sum_k alpha_k · v_k                        gather-and-reduce
```

Three things that look like separate architectures are parameters here:

- **causal vs sliding-window** — window width `W`, with `W == 0` meaning global.
- **score scale** — `1/sqrt(HeadDim)` conventionally, or `1.0` for a family
  that folded the constant into a learned per-head Q norm (Gemma 4).
- **full vs partial rotation** — `RotaryEmbedding<HeadDim, Base, Num, Den>`
  rotates the first `Num/Den` of the frequency planes and passes the rest
  through; Gemma's global heads rotate a quarter of a 512-wide head.

`KVCache` is the same object whether a layer keeps everything or keeps a ring:
logical row zero is always the oldest retained position, so a sliding layer and
a global layer are read identically and only the storage cost differs.

`src/recurrent.hpp` is the *other* answer to "communication across tokens".
A gated delta network carries a state matrix `S` forward instead of a cache:

```text
S <- S * exp(g)            gated decay, g <= 0
d  = (v - Sᵀk) * beta      how wrong the memory currently is about v
S  <- S + k ⊗ d            rank-1 correction toward v
o  =  Sᵀq                  read
```

It is an associative memory: write `v` at key `k`, read it back at `k`, and an
orthogonal query returns nothing — all three are executable tests. The cost
difference is the whole point of a hybrid stack:

| mixer | carried state | grows with T? |
|---|---|---|
| attention | `L · Hkv · (Dqk+Dv) · T` floats | yes |
| delta network | `Hv · Dk · Dv` floats | **no** |

For Qwen 3.5 4B that is 24 of 32 layers paying 2 MB flat instead of a cache
that grows with the conversation.

### Tensor algebra and the accelerator boundary

The residual stream and every prefill intermediate are `Matrix<D>` values with
shape `[tokens, channels]`. Family code keeps that token dimension intact: it
projects Q/K/V in batches, applies typed normalization and positional
transforms, and invokes semantic attention:

```text
Q = X Wq,  K = X Wk,  V = X Wv
Y = softmax_rows(scale · Q Kᵀ + visibility_mask) V
```

Decode is the same algebra with `tokens = 1`. The storage/algebra layer chooses
the implementation: the current CPU reference uses GEMV for one row,
weight-stationary GEMM for a batch, and online per-head attention without
materializing the score matrix. A future Accelerate/AMX, SME, Metal, CUDA, or
tensor-core backend replaces the kernels beneath `Weight::matmul`,
normalization, activation, and the attention reduction; transformer, Gemma,
vision, and future Qwen code do not change.

Loops that express architecture topology stay visible — layer traversal,
heterogeneous attention schedules, expert routing, modality segments. Numeric
loops over tokens/channels/heads belong inside algebra, cache, image, or
backend kernels so they do not obscure the model equations.

### The transformer contract

The implementation follows the general transformer map in
`General_Multimodal_Transformer_Architecture_Wide.pdf`. Each encoder produces an
`EmbeddingSegment<D>`; `compose_embeddings` creates one residual-width token
sequence with modality spans. `Transformer` owns `TransformerWeights`:
immutable token I/O, typed layers, final norm, and architecture data. The
public model is an immutable callable:

```text
logits = T(complete_input, prefix_cache)
```

`PrefixCache` is explicit but contains only derived, prefix-indexed memoized
intermediates. Its `Architecture::PrefixState` member is the model-specific
derived storage — Gemma binds it to its collection of owned and shared K/V
caches; a Qwen linear-attention layer will bind it to a fixed-size recurrent
state. Callers always provide the complete input. A cache is reused only when it
belongs to the same immutable weights and its input is a verified prefix;
otherwise it is cleared and recomputed. Caching changes cost, never the result.

Gemma has a nominal `GemmaDenseDecoderLayer` whose forward function states its
attention-residual, FFN-residual, and PLE equations directly, while reusing the
shared components and satisfying the shared contract. The token-mixer name is
intentional: a Qwen layer may bind attention or a recurrent mixer without
changing the transformer lifecycle. Dense MLP versus MoE is likewise a concrete
channel-mixer type, never a runtime flag.

The REPL is an application tool, not part of the library. It depends only on
the `ChatModel` interface; conversation history, chat templates, tokenization,
prefix memoization, sampling, output protocols, and modality encoders live in
adapters selected from `general.architecture` plus the model anatomy. Adding a
model does not add a REPL or a switch inside the conversation loop.

## Tokenizer

Two dialects over the same BPE engine, selected from `tokenizer.ggml.*`:

- **byte-level** — GPT-2 byte↔printable-codepoint remap plus a hand-rolled
  pre-tokenizer scanner over unicode categories (never `std::regex`). The two
  places vocabularies actually differ are data, not code paths: digits per
  piece (`\p{N}{1,3}` vs Qwen 3.5's `\p{N}`) and whether combining marks
  continue a word (`[\p{L}\p{M}]+`).
- **sentence** — raw UTF-8 symbols, U+2581 for space, no word splitting, with
  `<0xXX>` byte fallbacks. This is Gemma 4.

Category tables are generated from llama.cpp's own unicode data
(`tools/gen_unicode_data.py`) so category classification is byte-exact against
the oracle by construction.

## What was verified (and how)

- **GGUF reader** — `gguf_dump` matches llama.cpp's listing on real files.
- **Tokenizer parity** — token ids match `llama-tokenize` on **253/253 lines**
  of an adversarial corpus (digits, combining marks, CJK, Cyrillic, Devanagari,
  emoji, contractions, code, control tokens) for **both** the Gemma 4 and
  Qwen 3.5 vocabs; decode round-trips on both.
- **Quant** — dequant is **bit-identical to ggml** (max delta 0.0) for
  Q8_0/Q4_0/Q4_K/Q6_K; matvec kernels equal dequant-then-dot.
- **int8 activations** (`src/quant_i8.hpp`) — the activation is quantized once
  per matvec (Q8_0 / Q8_K) and rows use integer dot products, as llama.cpp
  does. Verified vs libggml on identical bytes: quantizers **byte-identical**,
  scalar kernels **bit-exact** vs `ggml_vec_dot_*_generic`, NEON within 0–1 ulp
  of ggml's own NEON.
- **Gemma 4 E4B text** — the official Q4_0 GGUF schema (42 heterogeneous
  local/global layers, final-18 shared KV, PLE, partial RoPE, logit softcap)
  loads immutably; tokenizer ids and tested greedy generations match llama.cpp.
- **Gemma 4 E4B vision** — the official Q8_0 mmproj schema loads immutably;
  dynamic aspect-preserving resize, patch projection, learned 2-D positions,
  2-D RoPE, clipped ViT linears, 3×3 pooling, and decoder projection are
  implemented. The four-color image checkpoint matched llama.cpp for all first
  80 greedy multimodal tokens (49 image tokens, 74 total prompt tokens).
- **Uniform core** — extracting the token-mixing core out of the Gemma file
  left every Gemma unit test, the E4B cache topology, and the multimodal path
  unchanged. Adding Qwen 3.5's hybrid stack then required **no change** to the
  residual stream, the `PrefixCache` contract, or the channel mixer: both layer
  kinds bind the same canonical `TransformerBlock`. Adding Gemma 4 12B likewise
  touched only types (a third KV kind, per-kind KV head counts, a tail
  parameter) and left E4B's numerics untouched.
- **Gemma 4 12B** — the real checkpoint confirms every derived claim: per-layer
  `head_count_kv` is `{8,8,8,8,8,1,...}`, global layers carry **no `attn_v`**
  (unified K/V), and `rope_freqs` is `1.0` for planes 0–63 and `1e30` for
  64–255 — exactly `RotaryEmbedding<512, 1e6, 1, 4>`. The dump also caught a
  real omission: a per-layer `layer_output_scale` that a "no tail" layer would
  have silently dropped.
- **Qwen 3.5** — greedy parity holds, and the one genuinely ambiguous decision
  was settled **by experiment on real weights**: key heads are TILED across
  value heads (`h % Hk`), not grouped GQA-style. Flipping that one line turns
  " Paris" (logprob −0.66) into " a" (−2.05) and coherent text into gibberish.
  Both mappings are shape-legal, so only weights could decide it.

Reference harnesses link the prebuilt `libllama`; see `INFERENCE_PLAN.md` §3.

## Known traps handled (INFERENCE_PLAN.md §5)

T1 ggml `[in,out]` layout (→ `MatT`, zero-copy) · T2 RoPE pairing and partial
rotation · T3 hand-rolled pre-tokenizer over unicode categories · T4 special
tokens split before BPE · T5 tied unembedding · T6 fp32 accumulation everywhere
· T7 exact chat whitespace · T8 GGUF aligned data offset · T9 metadata names
read from the dump · T10 continuous absolute positions across turns.
