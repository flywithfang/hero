# nm — a uniform transformer inference engine for Llama, Gemma and Qwen

Pure C++20, with no third-party runtime dependencies. Today it loads Llama-3.2
and Gemma 4 E4B GGUFs and runs pure-CPU text or image-text inference. The core
is organized for Gemma MoE and Qwen 3.5/3.6 as the next model families. Built by
evolving the concept code in `gpt_pipeline.cpp` per `INFERENCE_PLAN.md`, with a
**fidelity-first** discipline: every milestone was gated against a reference
oracle (llama.cpp / ggml) before moving on.

## Build & run

CMake + Ninja. With presets (`release` / `debug` / `asan`):

```sh
cmake --workflow --preset dev    # configure + build + test in one shot
# or step by step:
cmake --preset release
cmake --build --preset release   # build/chat, build/gguf_dump, build/logit_diff
ctest --preset release           # self-contained suite (property, quant, e2e, tokenizer)

./build/chat model.gguf --system "You are helpful." --temp 0.7
./build/gguf_dump model.gguf

# The same chat executable auto-selects Gemma 4 E4B from GGUF metadata.
./build/chat gemma-4-E4B-it-Q4_0.gguf --max 256

# Add vision and optionally attach an image to the first message.
./build/chat gemma-4-E4B-it-Q4_0.gguf \
  --mmproj mmproj-gemma-4-E4B-it-Q8_0.gguf \
  --image photo.png --max 256

# One-shot diagnostic tools remain available.
./build/gemma4_text gemma-4-E4B-it-Q4_0.gguf "Hello" 16
./build/gemma4_image gemma-4-E4B-it-Q4_0.gguf \
  mmproj-gemma-4-E4B-it-Q8_0.gguf image.ppm \
  "Describe this image." 64
```

Without presets: `cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`.

`NM_THREADS=N` sets the worker count (defaults to hardware concurrency). The
fixture-dependent tests (e2e_stories, tokenizer_roundtrip) auto-skip when the
model/vocab files aren't present; override their paths with
`-DNM_STORIES=... -DNM_VOCAB=...` at configure time.

Inside multimodal chat, `/image PATH` attaches an image to the next user
message and `/image clear` removes it. On macOS, ImageIO supplies PNG, JPEG,
HEIC and other system-supported formats; the portable dependency-free path
accepts RGB PPM P6/P3. `/reset`, `/stats`, `/help`, and `/quit` work for every
model adapter.

## Architecture — five strata (layout knowledge lives only in stratum 0)

| stratum | file | contents |
|--------|------|----------|
| 0 storage | `src/core.hpp` | `Vec/VecView/TokenMatrix/Mat/MatT/RowStore/PastView`, fp16/bf16 |
| 0 quant | `src/quant.hpp` | GGUF block formats, dequant, `Weight<In,Out>` matvec |
| 1 algebra | `src/core.hpp` | `dot/axpy/softmax/silu`, `matvec_T`, `par_for/par_map` |
| 2 components | `src/components.hpp` | `Linear/ClippedLinear/RMSNorm/RoPE/GatedMLP/Attention` (config-free) |
| 3 transformer | `src/transformer.hpp` | shared residual stream, token-mixer/channel-mixer branches, immutable model, architecture contract, session traversal |
| 3 specifications | `src/llama.hpp`, `src/gemma4.hpp` | model-family dimensions, concrete layer types, cache policies, input preparation and output rules |
| 3 modalities | `src/multimodal.hpp`, `src/gemma4_vision.hpp` | modality segments, image preprocessing, ViT, soft-token projection |
| 4 runtime | `src/runtime.hpp` | architecture-generic `AutoregressiveRuntime` and samplers |
| chat contract | `src/chat_session.hpp`, `src/chat_models.hpp` | model detection, model-neutral session interface, Llama/Gemma adapters |
| tensor loader | `src/tensor_loader.hpp` | architecture-neutral, shape-safe GGUF tensor construction |
| model loaders | `src/llama_loader.hpp`, `src/gemma4_*_loader.hpp` | model schema validation → immutable assemblies |
| tokenizer | `src/tokenizer.{hpp,cpp}` | Llama byte BPE and Gemma 4 SPM-style BPE |
| gguf | `src/gguf.{hpp,cpp}` | GGUF v3 mmap reader |

`MatT<In,Out>` is the ggml `[in,out]` weight layout — row = one output, loaded
**zero-copy** over the mmap. `Weight<In,Out>` dispatches `matvec` across
F32 / F16 / BF16 / Q8_0 / Q4_0 / Q4_K / Q6_K; activations stay fp32 throughout.

The implementation follows the general transformer map in
`General_Multimodal_Transformer_Architecture_Wide.pdf`. Each encoder produces
an `EmbeddingSegment<D>`; `compose_embeddings` creates one residual-width token
sequence with modality spans and model metadata. `TransformerModel` owns the
immutable token I/O, typed layer stack, final norm and architecture data, while
`TransformerSession` owns mutable cache state and the common prefill/step
traversal. Llama and Gemma instantiate the same `TransformerBlock` from typed
token-mixer and channel-mixer residual branches. Their token embedding/output
heads satisfy the same `TokenInputOutput` contract. Gemma E4B adds PLE as a
typed layer tail, so it does not leak into Llama or become an anatomy boolean.
The token-mixer name is intentional: a Qwen layer may bind attention or a
recurrent mixer without changing the shared model/session lifecycle. Dense
MLP versus MoE is likewise a concrete channel-mixer type, never a runtime flag.

The REPL itself depends only on the `ChatModel` interface. Model-specific chat
templates, tokenization, KV-cache behavior, output protocols, and modality
encoders live in adapters selected from `general.architecture` plus the model
anatomy. Adding another model does not add another REPL or introduce a switch
inside the conversation loop.

## What was verified (and how)

- **M0 GGUF reader** — `gguf_dump` matches llama.cpp's listing on real files.
- **M1 forward fidelity** — greedy **argmax parity with llama.cpp on 152/152
  positions** (stories260K), last-position logit delta ~1e-3. Found the key
  trap: GGUF Llama needs **interleaved (NORM) RoPE**, not rotate_half.
- **M2 tokenizer** — **324/324 lines match** `llama_tokenize` on the llama-bpe
  vocab across an adversarial corpus (unicode, code, emoji, contractions,
  whitespace, special tokens); decode round-trips; chat-template tokens match.
- **M3/M4 quant** — dequant is **bit-identical to ggml** (max delta 0.0) for
  Q8_0/Q4_0/Q4_K/Q6_K; matvec kernels equal dequant-then-dot.
- **M5 int8 activations** (`src/quant_i8.hpp`) — like llama.cpp, the activation
  is quantized once per matvec (Q8_0 / Q8_K) and rows use integer dot products.
  Verified vs libggml on identical bytes: quantizers **byte-identical**, scalar
  kernels **bit-exact** vs `ggml_vec_dot_*_generic`, NEON within 0–1 ulp of
  ggml's own NEON. ~84 tok/s decode on 1B Q4_K_M (llama-cli: ~103).
  `NM_FP32_ACT=1` selects the fp32-activation reference path.
- **Gemma 4 E4B text** — official Q4_0 GGUF schema (42 heterogeneous
  local/global layers, final-18 shared KV, PLE, proportional RoPE, logit
  softcap) loads immutably; tokenizer IDs and tested greedy generations match
  llama.cpp.
- **Gemma 4 E4B vision** — official Q8_0 mmproj schema loads immutably; dynamic
  aspect-preserving resize, patch projection, learned 2-D positions, 2-D RoPE,
  clipped ViT linears, 3x3 pooling, and decoder projection are implemented.
  The four-color image checkpoint matched llama.cpp for all first 80 greedy
  multimodal tokens (49 image tokens, 74 total prompt tokens).
- **Multi-model chat** — the same `build/chat` executable was exercised with a
  real Llama-3.2-1B Q4_K_M and the real Gemma 4 E4B Q4_0. Both retained
  coherent state over two turns. Gemma image chat uses the same adapter and
  matched the established 74-token multimodal prompt; image decoding tests
  check both PPM and macOS PNG paths, including orientation/channel order.
- **Uniform transformer core** — Llama 3.2 1B and Gemma 4 E4B instantiate the
  same immutable model, residual-branch/block and mutable session types. After
  migration, the real-checkpoint Llama eight-token greedy sequence and Gemma
  top-5 logits/eight-token greedy sequence remained exactly unchanged; a real
  E4B image turn also completed through the shared residual-stream boundary.

Reference harnesses live in the scratchpad and link the prebuilt `libllama`.

## Known traps handled (INFERENCE_PLAN.md §4)

T1 ggml `[in,out]` layout (→ `MatT`, zero-copy) · T2 interleaved RoPE ·
T3 hand-rolled llama3 pre-tokenizer over unicode categories (no `std::regex`) ·
T4 special tokens split before BPE · T5 tied unembedding (logits = h·wte) ·
T6 fp32 accumulation everywhere · T7 exact chat whitespace · T8 GGUF aligned
data offset · T9 metadata names read from the dump · T10 continuous absolute
positions across turns.
